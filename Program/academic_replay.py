"""
Thin Orchestration Script for Academic Replay.
Consolidates logic from sweep_nn.py, acs_solver.py, and rvnd.py.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

# Import core modules
import sweep_nn
import acs_solver
import rvnd

# Config
DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
INSTANCE_PATH = DATA_DIR / "parsed_instance.json"
DISTANCE_PATH = DATA_DIR / "parsed_distance.json"
OUTPUT_PATH = DATA_DIR / "academic_replay_results.json"

def run_academic_replay(
    user_vehicles: List[Dict] = None,
    user_customers: List[Dict] = None,
    user_depot: Dict = None,
    user_acs_params: Dict = None,
    distance_multiplier: float = 1.0
) -> Dict:
    """
    Entry point for Streamlit UI to run the academic replay.
    Orchestrates Sweep -> NN -> RVND (Inter/Intra).
    """
    print("[Academic Replay] Starting Optimization via run_academic_replay...")
    
    # 1. Load or Build Instance
    if user_customers and user_vehicles and user_depot:
        instance = {
            "depot": user_depot,
            "customers": user_customers,
            "fleet": user_vehicles
        }
        # Build a temporary distance matrix based on coordinates
        from math import sqrt
        nodes = [{"id": 0, "x": user_depot["x"], "y": user_depot["y"]}] + \
                [{"id": c["id"], "x": c["x"], "y": c["y"]} for c in user_customers]
        
        n_nodes = len(nodes)
        dist_matrix = [[0.0]*n_nodes for _ in range(n_nodes)]
        travel_matrix = [[0.0]*n_nodes for _ in range(n_nodes)]
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                d = sqrt((nodes[i]["x"] - nodes[j]["x"])**2 + (nodes[i]["y"] - nodes[j]["y"])**2) * distance_multiplier
                dist_matrix[i][j] = d
                travel_matrix[i][j] = d
        
        distance_data = {
            "nodes": nodes,
            "distance_matrix": dist_matrix,
            "travel_time_matrix": travel_matrix
        }
    else:
        with open(INSTANCE_PATH, 'r', encoding='utf-8') as f:
            instance = json.load(f)
        with open(DISTANCE_PATH, 'r', encoding='utf-8') as f:
            distance_data = json.load(f)

    seed_val = user_acs_params.get("seed", 84) if user_acs_params else 84
    rng = random.Random(seed_val)
    
    # --- PHASE 1: SWEEP ---
    clusters, _, sweep_logs = sweep_nn.build_clusters(instance, academic_mode=True)
    
    # --- PHASE 2: NEAREST NEIGHBOR ---
    initial_routes = []
    nn_logs = []
    for cluster in clusters:
        route, logs = sweep_nn.nearest_neighbor_route(
            cluster, instance, distance_data, academic_mode=True
        )
        initial_routes.append({
            "cluster_id": cluster["cluster_id"],
            "vehicle_type": cluster["vehicle_type"],
            "sequence": route["sequence"],
            "total_distance": route["total_distance"],
            "total_demand": cluster["total_demand"],
            "total_tw_violation": route.get("total_tw_violation", 0),
            "total_service_time": route.get("total_service_time", 0),
            "total_travel_time": route.get("total_travel_time", 0),
            "total_wait_time": route.get("total_wait_time", 0)
        })
        nn_logs.extend(logs)

    # --- PHASE 3: ANT COLONY SYSTEM ---
    print("[Academic Replay] Running Ant Colony System...")
    acs_routes = []
    acs_logs = []
    
    # Default params if not provided
    acs_params = user_acs_params if user_acs_params else {
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.1,
        "q0": 0.9,
        "num_ants": 10,
        "max_iterations": 20
    }
    
    for idx, initial_route in enumerate(initial_routes):
        cluster = clusters[idx]
        metrics = acs_solver.acs_cluster(
            cluster, instance, distance_data, initial_route, acs_params, rng, academic_mode=True
        )
        acs_routes.append(metrics)
        if "iteration_logs" in metrics:
            acs_logs.extend(metrics["iteration_logs"])

    # --- PHASE 4: RVND (GLOBAL INTER) ---
    print("[Academic Replay] Running RVND Inter-route...")
    optimized_routes, rvnd_logs = rvnd.rvnd_inter(
        acs_routes, # Use ACS results as input for RVND
        instance,
        distance_data,
        instance["fleet"],
        rng,
        max_iterations=rvnd.MAX_INTER_ITERATIONS,
        academic_mode=True
    )
    
    # --- PHASE 5: RVND (LOCAL INTRA) ---
    intra_results = []
    all_logs = sweep_logs + nn_logs + acs_logs + rvnd_logs
    
    fleet_data = {f["id"]: f for f in instance["fleet"]}
    for route in optimized_routes:
        improved = rvnd.rvnd_intra(
            route["sequence"],
            instance,
            distance_data,
            fleet_data[route["vehicle_type"]],
            rng,
            max_iterations=rvnd.MAX_INTRA_ITERATIONS,
            academic_mode=True
        )
        intra_results.append({
            "cluster_id": route["cluster_id"],
            "old_vehicle_type": route["vehicle_type"],
            "improved": improved
        })
        if "iteration_logs" in improved:
            for log in improved["iteration_logs"]:
                log["cluster_id"] = route["cluster_id"]
                all_logs.append(log)

    # --- PHASE 5: VEHICLE REASSIGNMENT (Consistency with UI) ---
    # Sort by demand descending for greedy assignment
    intra_results.sort(key=lambda x: x["improved"]["total_demand"], reverse=True)
    
    used_vehicles = {}
    final_routes = []
    
    for res in intra_results:
        improved = res["improved"]
        total_demand = improved["total_demand"]
        new_vehicle_type = rvnd.assign_vehicle_by_demand(total_demand, instance["fleet"], used_vehicles)
        
        if new_vehicle_type is None:
            new_vehicle_type = res["old_vehicle_type"]
            improved["vehicle_assignment_failed"] = True
            reason = "Stok Habis / Kapasitas Kurang"
            status = "❌ Gagal"
        else:
            used_vehicles[new_vehicle_type] = used_vehicles.get(new_vehicle_type, 0) + 1
            improved["vehicle_assignment_failed"] = False
            reason = "Kapasitas Memadai"
            status = "✅ Assigned"
            
        # Re-evaluate if vehicle type changed
        if new_vehicle_type != res["old_vehicle_type"] and new_vehicle_type in fleet_data:
            improved = rvnd.evaluate_route(improved["sequence"], instance, distance_data, fleet_data[new_vehicle_type])
            improved["vehicle_assignment_failed"] = False

        # Add Vehicle Reassign Log for UI
        all_logs.append({
            "phase": "VEHICLE_REASSIGN",
            "cluster_id": res["cluster_id"],
            "old_vehicle": res["old_vehicle_type"],
            "new_vehicle": new_vehicle_type,
            "status": status,
            "demand": total_demand,
            "reason": reason
        })

        # Final object for result["routes"]
        # IMPORTANT: UI expects metrics (total_distance, total_service_time, etc.) at the TOP LEVEL.
        route_entry = {
            "cluster_id": res["cluster_id"],
            "vehicle_type": new_vehicle_type,
            "total_demand": total_demand, 
        }
        # Promote everything from improved to top-level for compatibility
        route_entry.update(improved)
        # Ensure improved key still exists if anything expects it
        route_entry["improved"] = improved 
        
        final_routes.append(route_entry)

    # --- PHASE 6: CALCULATE COSTS AND METADATA ---
    total_cost = 0.0
    cost_entries = []
    
    for r in final_routes:
        v_type = r["vehicle_type"]
        if v_type in fleet_data:
            f = fleet_data[v_type]
            fix = f.get("fixed_cost", 0.0)
            var = f.get("variable_cost_per_km", 0.0) * r.get("total_distance", 0.0)
            route_cost = fix + var
            cost_entries.append({
                "cluster_id": r["cluster_id"],
                "vehicle_type": v_type,
                "fixed_cost": fix,
                "variable_cost": var,
                "total_cost": route_cost
            })
            total_cost += route_cost

    summary = {
        "distance_before": sum(r["total_distance"] for r in initial_routes),
        "distance_after": sum(r["total_distance"] for r in final_routes),
        "tw_before": sum(r["total_tw_violation"] for r in initial_routes),
        "tw_after": sum(r["total_tw_violation"] for r in final_routes),
        "total_cost": total_cost
    }

    # --- PHASE 7: ADD VEHICLE AVAILABILITY FOR UI ---
    vehicle_availability = []
    available_vehicles = []
    for f in instance["fleet"]:
        units = f.get("units", 1)
        used = used_vehicles.get(f["id"], 0)
        vehicle_availability.append({
            "id": f["id"],
            "name": f.get("name", f["id"]),
            "capacity": f.get("capacity", 0),
            "units": units,
            "used": used,
            "available_units": max(0, units - used),
            "available": (units - used) > 0
        })
        if units > 0:
            available_vehicles.append(f["id"])

    return {
        "routes": final_routes,
        "summary": summary,
        "iteration_logs": all_logs,
        "costs": {
            "total_cost": total_cost,
            "breakdown": cost_entries
        },
        "dataset": {
            "fleet": instance["fleet"],
            "depot": instance["depot"],
            "customers": instance["customers"]
        },
        "vehicle_availability": vehicle_availability,
        "available_vehicles": available_vehicles,
        "mode": "ACADEMIC_REPLAY",
        "status": "success"
    }
