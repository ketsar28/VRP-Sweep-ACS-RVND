import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
INSTANCE_PATH = DATA_DIR / "parsed_instance.json"
DISTANCE_PATH = DATA_DIR / "parsed_distance.json"
ACS_PATH = DATA_DIR / "acs_routes.json"
RVND_PATH = DATA_DIR / "rvnd_routes.json"

# RVND Configuration
MAX_INTER_ITERATIONS = 50
MAX_INTRA_ITERATIONS = 100

# Neighborhood definitions
INTER_ROUTE_NEIGHBORHOODS = ["shift_1_0", "shift_2_0", "swap_1_1", "swap_2_1", "swap_2_2", "cross"]
INTRA_ROUTE_NEIGHBORHOODS = ["two_opt", "or_opt", "reinsertion", "exchange"]


def assign_vehicle_by_demand(total_demand: float, fleet_data: List[Dict], used_vehicles: Dict[str, int]) -> Optional[str]:
    """
    Assign smallest feasible vehicle based on demand intervals.
    """
    # Sort fleets by capacity (smallest first)
    sorted_fleets = sorted(fleet_data, key=lambda f: f["capacity"])
    
    for fleet in sorted_fleets:
        # Check capacity feasibility
        if fleet["capacity"] >= total_demand:
            # Check unit availability
            units_used = used_vehicles.get(fleet["id"], 0)
            if units_used < fleet["units"]:
                return fleet["id"]
    
    # No feasible vehicle available
    return None


def can_assign_fleet(demands: List[float], fleet_data: List[Dict]) -> Tuple[bool, int]:
    """
    Check if a set of demands can be assigned to available fleet stock.
    Returns: (is_feasible, number_of_unassigned_clusters)
    Using Greedy-Decreasing assignment (Best Fit for Bin Packing style logic).
    """
    sorted_demands = sorted(demands, reverse=True)
    stock = {f["id"]: f["units"] for f in fleet_data}
    fleet_lookup = sorted(fleet_data, key=lambda f: f["capacity"]) # Smallest to largest
    
    unassigned = 0
    used = {f["id"]: 0 for f in fleet_data}
    
    for d in sorted_demands:
        assigned = False
        # Find smallest vehicle that fits and has stock
        for f in fleet_lookup:
            if f["capacity"] >= d and used[f["id"]] < stock[f["id"]]:
                used[f["id"]] += 1
                assigned = True
                break
        if not assigned:
            unassigned += 1
            
    return (unassigned == 0, unassigned)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_time_to_minutes(value: str) -> float:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def minutes_to_clock(value: float) -> str:
    hours = int(value // 60)
    minutes = int(value % 60)
    seconds = round((value - math.floor(value)) * 60)
    if seconds == 0:
        return f"{hours:02d}:{minutes:02d}"
    return f"{hours:02d}:{minutes:02d}+{seconds:02d}s"


def evaluate_route(sequence: List[int], instance: dict, distance_data: dict, fleet_info: dict) -> Dict[str, float]:
    """Evaluate a single route with capacity and time window constraints."""
    node_index = {node["id"]: idx for idx, node in enumerate(distance_data["nodes"])}
    distance_matrix = distance_data["distance_matrix"]
    travel_matrix = distance_data["travel_time_matrix"]

    depot = instance["depot"]
    depot_tw = {
        "start": parse_time_to_minutes(depot["time_window"]["start"]),
        "end": parse_time_to_minutes(depot["time_window"]["end"])
    }
    depot_service = depot.get("service_time", 0)

    customers = {customer["id"]: customer for customer in instance["customers"]}

    stops = []
    total_distance = 0.0
    travel_time_sum = 0.0
    service_time_sum = 0.0
    violation_sum = 0.0
    total_demand = 0.0

    prev_node = sequence[0]
    current_time = depot_tw["start"] + depot_service

    stops.append({
        "node_id": 0,
        "arrival": depot_tw["start"],
        "arrival_str": minutes_to_clock(depot_tw["start"]),
        "departure": current_time,
        "departure_str": minutes_to_clock(current_time),
        "wait": 0.0,
        "violation": 0.0
    })

    for next_node in sequence[1:]:
        travel = travel_matrix[node_index[prev_node]][node_index[next_node]]
        distance = distance_matrix[node_index[prev_node]][node_index[next_node]]
        total_distance += distance
        travel_time_sum += travel

        arrival_no_wait = current_time + travel

        if next_node == 0:
            tw_start = depot_tw["start"]
            tw_end = depot_tw["end"]
            service_time = depot_service
        else:
            customer = customers[next_node]
            tw_start = parse_time_to_minutes(customer["time_window"]["start"])
            tw_end = parse_time_to_minutes(customer["time_window"]["end"])
            service_time = customer["service_time"]
            total_demand += customer["demand"]

        arrival = max(tw_start, arrival_no_wait)
        wait = max(0.0, tw_start - arrival_no_wait)
        violation = max(0.0, arrival - tw_end)
        departure = arrival + service_time

        if next_node != 0:
            service_time_sum += service_time
            violation_sum += violation

        stops.append({
            "node_id": next_node,
            "arrival": arrival,
            "arrival_str": minutes_to_clock(arrival),
            "departure": departure,
            "departure_str": minutes_to_clock(departure),
            "wait": wait,
            "violation": violation
        })

        prev_node = next_node
        current_time = departure

    time_component = travel_time_sum + service_time_sum
    objective = total_distance + time_component + violation_sum

    # Check capacity constraint
    capacity_violation = max(0.0, total_demand - fleet_info["capacity"])

    return {
        "sequence": sequence,
        "stops": stops,
        "total_distance": total_distance,
        "total_travel_time": travel_time_sum,
        "total_service_time": service_time_sum,
        "total_time_component": time_component,
        "total_tw_violation": violation_sum,
        "total_demand": total_demand,
        "capacity_violation": capacity_violation,
        "objective": objective,
        "feasible": violation_sum == 0 and capacity_violation == 0
    }


def is_solution_better(new_metrics: Dict, current_metrics: Dict) -> bool:
    """
    Check if new solution is better.
    
    REVISION NOTE: Per specification, acceptance is based ONLY on distance improvement.
    Time window violations are reported but do NOT invalidate solutions.
    Capacity violations DO invalidate solutions (vehicle assignment constraint).
    """
    # Reject if capacity violated (vehicle cannot physically handle demand)
    if new_metrics["capacity_violation"] > 0:
        return False
    
    # Accept if current is capacity-infeasible but new is feasible
    if current_metrics["capacity_violation"] > 0 and new_metrics["capacity_violation"] == 0:
        return True
    
    # Both capacity-feasible: compare total distance only
    # Time window violations are soft constraints - reported but not used for rejection
    return new_metrics["total_distance"] < current_metrics["total_distance"]


# ========== INTRA-ROUTE NEIGHBORHOODS ==========

def intra_two_opt(sequence: List[int], i: int, j: int) -> List[int]:
    """2-opt: reverse segment between i and j."""
    return sequence[:i] + list(reversed(sequence[i:j + 1])) + sequence[j + 1:]


def intra_or_opt(sequence: List[int], i: int, length: int, j: int) -> List[int]:
    """Or-opt: move a segment of 'length' nodes starting at i to position j."""
    if i + length > len(sequence) - 1:
        return sequence
    segment = sequence[i:i + length]
    remaining = sequence[:i] + sequence[i + length:]
    insert_pos = j if j <= i else j - length
    return remaining[:insert_pos] + segment + remaining[insert_pos:]


def intra_reinsertion(sequence: List[int], i: int, j: int) -> List[int]:
    """Reinsertion: move single customer from i to j."""
    if i == j or i == 0 or i == len(sequence) - 1:
        return sequence
    new_seq = sequence[:]
    node = new_seq.pop(i)
    if j > i:
        new_seq.insert(j - 1, node)
    else:
        new_seq.insert(j, node)
    return new_seq


def intra_exchange(sequence: List[int], i: int, j: int) -> List[int]:
    """Exchange: swap two customers."""
    if i == 0 or j == 0 or i == len(sequence) - 1 or j == len(sequence) - 1:
        return sequence
    new_seq = sequence[:]
    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq


def apply_intra_neighborhood(neighborhood: str, sequence: List[int], rng: random.Random) -> Optional[List[int]]:
    """Apply a single intra-route neighborhood move."""
    n = len(sequence)
    if n <= 2:
        return None
    
    if neighborhood == "two_opt":
        candidates = [(i, j) for i in range(1, n - 2) for j in range(i + 1, n - 1)]
        if not candidates:
            return None
        i, j = rng.choice(candidates)
        return intra_two_opt(sequence, i, j)
    
    elif neighborhood == "or_opt":
        candidates = []
        for length in [1, 2, 3]:
            for i in range(1, n - 1):
                if i + length > n - 1:
                    continue
                for j in range(1, n - length + 1):
                    if j < i or j > i + length:
                        candidates.append((i, length, j))
        if not candidates:
            return None
        i, length, j = rng.choice(candidates)
        return intra_or_opt(sequence, i, length, j)
    
    elif neighborhood == "reinsertion":
        candidates = [(i, j) for i in range(1, n - 1) for j in range(1, n) if j != i and j != i + 1]
        if not candidates:
            return None
        i, j = rng.choice(candidates)
        return intra_reinsertion(sequence, i, j)
    
    elif neighborhood == "exchange":
        candidates = [(i, j) for i in range(1, n - 1) for j in range(i + 1, n - 1)]
        if not candidates:
            return None
        i, j = rng.choice(candidates)
        return intra_exchange(sequence, i, j)
    
    return None


# ========== INTER-ROUTE NEIGHBORHOODS ==========
# Note: For single-route optimization, inter-route operators are not applicable
# These are placeholders for multi-route scenarios

def apply_inter_neighborhood(
    neighborhood: str, 
    routes: List[Dict], 
    instance: Dict, 
    distance_matrix: List[List[float]], 
    fleet_list: List[Dict],
    rng: random.Random
) -> Dict:
    """
    Apply inter-route neighborhood operator (Global RVND).
    STOCK-AWARE: Moves must lead to a valid fleet assignment.
    """
    current_distance = sum(r["total_distance"] for r in routes)
    customers = {c["id"]: c for c in instance["customers"]}
    best_move = None
    best_routes = None
    best_distance = current_distance
    
    current_demands = [r["total_demand"] for r in routes]
    _, current_unassigned = can_assign_fleet(current_demands, fleet_list)
    best_unassigned = current_unassigned

    def calc_route_distance(seq):
        return sum(distance_matrix[seq[k]][seq[k+1]] for k in range(len(seq)-1))

    def calc_route_demand(seq, customers_dict):
        return sum(customers_dict[c]["demand"] for c in seq[1:-1])

    def rebuild_route_internal(route, new_seq, new_dist, new_demand):
        updated = deepcopy(route)
        updated["sequence"] = new_seq
        updated["total_distance"] = round(new_dist, 2)
        updated["total_demand"] = new_demand
        return updated

    def evaluate_move(i, j, new_seq_a, new_seq_b):
        nonlocal best_distance, best_routes, best_move, best_unassigned
        
        dist_a = calc_route_distance(new_seq_a)
        dist_b = calc_route_distance(new_seq_b)
        demand_a = calc_route_demand(new_seq_a, customers)
        demand_b = calc_route_demand(new_seq_b, customers)
        
        total_new_dist = current_distance - routes[i]["total_distance"] - routes[j]["total_distance"] + dist_a + dist_b
        
        # Check global fleet feasibility
        new_demands = current_demands[:]
        new_demands[i] = demand_a
        new_demands[j] = demand_b
        is_feasible, unassigned = can_assign_fleet(new_demands, fleet_list)
        
        # Acceptance Logic: Priority to Feasibility, then Distance
        if unassigned < best_unassigned:
            # Major improvement: reduced number of unassigned clusters
            best_unassigned = unassigned
            best_distance = total_new_dist
            best_routes = deepcopy(routes)
            best_routes[i] = rebuild_route_internal(routes[i], new_seq_a, dist_a, demand_a)
            best_routes[j] = rebuild_route_internal(routes[j], new_seq_b, dist_b, demand_b)
            return True
        elif unassigned == best_unassigned:
            # Same level of (in)feasibility, check distance
            if total_new_dist < best_distance - 1e-4:
                best_distance = total_new_dist
                best_routes = deepcopy(routes)
                best_routes[i] = rebuild_route_internal(routes[i], new_seq_a, dist_a, demand_a)
                best_routes[j] = rebuild_route_internal(routes[j], new_seq_b, dist_b, demand_b)
                return True
        return False

    if neighborhood == "swap_1_1":
        for i, route_a in enumerate(routes):
            for j, route_b in enumerate(routes):
                if i >= j:
                    continue
                seq_a = route_a["sequence"][1:-1]
                seq_b = route_b["sequence"][1:-1]
                for ca in seq_a:
                    for cb in seq_b:
                        new_seq_a = [0] + [cb if c == ca else c for c in seq_a] + [0]
                        new_seq_b = [0] + [ca if c == cb else c for c in seq_b] + [0]
                        if evaluate_move(i, j, new_seq_a, new_seq_b):
                            best_move = {"type": "swap_1_1", "detail": f"{ca}, {cb}"}

    elif neighborhood == "shift_1_0":
        for i, route_a in enumerate(routes):
            if len(route_a["sequence"]) <= 3:
                continue
            for j, route_b in enumerate(routes):
                if i == j:
                    continue
                seq_a = route_a["sequence"][1:-1]
                seq_b = route_b["sequence"][1:-1]
                for ca in seq_a:
                    new_seq_a_inner = [c for c in seq_a if c != ca]
                    if not new_seq_a_inner:
                        continue
                    new_seq_a = [0] + new_seq_a_inner + [0]
                    for pos in range(len(seq_b) + 1):
                        new_seq_b = [0] + seq_b[:pos] + [ca] + seq_b[pos:] + [0]
                        if evaluate_move(i, j, new_seq_a, new_seq_b):
                            best_move = {"type": "shift_1_0", "detail": f"{ca} -> R{j+1}"}

    elif neighborhood == "swap_2_1":
        for i, route_a in enumerate(routes):
            if len(route_a["sequence"]) < 4:
                continue
            for j, route_b in enumerate(routes):
                if i == j:
                    continue
                seq_a = route_a["sequence"][1:-1]
                seq_b = route_b["sequence"][1:-1]
                for idx_a in range(len(seq_a) - 1):
                    ca1, ca2 = seq_a[idx_a], seq_a[idx_a+1]
                    for cb in seq_b:
                        new_seq_a = [0] + seq_a[:idx_a] + [cb] + seq_a[idx_a+2:] + [0]
                        idx_b = seq_b.index(cb)
                        new_seq_b = [0] + seq_b[:idx_b] + [ca1, ca2] + seq_b[idx_b+1:] + [0]
                        if evaluate_move(i, j, new_seq_a, new_seq_b):
                            best_move = {"type": "swap_2_1", "detail": f"({ca1},{ca2}), {cb}"}

    elif neighborhood == "swap_2_2":
        for i, route_a in enumerate(routes):
            if len(route_a["sequence"]) < 4:
                continue
            for j, route_b in enumerate(routes):
                if i >= j:
                    continue
                if len(route_b["sequence"]) < 4:
                    continue
                seq_a = route_a["sequence"][1:-1]
                seq_b = route_b["sequence"][1:-1]
                for idx_a in range(len(seq_a) - 1):
                    ca1, ca2 = seq_a[idx_a], seq_a[idx_a+1]
                    for idx_b in range(len(seq_b) - 1):
                        cb1, cb2 = seq_b[idx_b], seq_b[idx_b+1]
                        new_seq_a = [0] + seq_a[:idx_a] + [cb1, cb2] + seq_a[idx_a+2:] + [0]
                        new_seq_b = [0] + seq_b[:idx_b] + [ca1, ca2] + seq_b[idx_b+2:] + [0]
                        if evaluate_move(i, j, new_seq_a, new_seq_b):
                            best_move = {"type": "swap_2_2", "detail": f"({ca1},{ca2}), ({cb1},{cb2})"}

    elif neighborhood == "cross":
        for i, route_a in enumerate(routes):
            for j, route_b in enumerate(routes):
                if i >= j:
                    continue
                seq_a = route_a["sequence"]
                seq_b = route_b["sequence"]
                for p_a in range(1, len(seq_a) - 1):
                    for p_b in range(1, len(seq_b) - 1):
                        new_seq_a = seq_a[:p_a] + seq_b[p_b:]
                        new_seq_b = seq_b[:p_b] + seq_a[p_a:]
                        if len(new_seq_a) <= 2 or len(new_seq_b) <= 2:
                            continue
                        if evaluate_move(i, j, new_seq_a, new_seq_b):
                            best_move = {"type": "cross", "detail": f"Split at A:{p_a}, B:{p_b}"}
    
    if best_routes:
        return {
            "accepted": True, 
            "new_routes": best_routes, 
            "distance_after": best_distance, 
            "move_details": best_move,
            "unassigned_after": best_unassigned
        }
    return {"accepted": False}


# ========== INTRA-ROUTE RVND ==========

def rvnd_intra(
    sequence: List[int],
    instance: dict,
    distance_data: dict,
    fleet_info: dict,
    rng: random.Random,
    max_iterations: int
) -> Dict:
    """
    Intra-route RVND with strict neighborhood list management.
    
    Rules:
    - NL_intra starts full at beginning
    - When neighborhood improves: reset NL_intra
    - When neighborhood fails: remove from NL_intra
    - Stop when NL_intra empty OR max_iterations reached
    """
    current_solution = sequence[:]
    current_metrics = evaluate_route(current_solution, instance, distance_data, fleet_info)
    
    iter_count = 0
    iteration_logs = []
    
    while iter_count < max_iterations:
        NL_intra = INTRA_ROUTE_NEIGHBORHOODS[:]
        
        # Track if improvement was made this iteration
        improved_this_iteration = False
        neighborhood_used = None
        
        while NL_intra:
            # Select random neighborhood
            neighborhood = rng.choice(NL_intra)
            
            # Apply neighborhood
            new_solution = apply_intra_neighborhood(neighborhood, current_solution, rng)
            
            if new_solution is None:
                # No valid move
                NL_intra.remove(neighborhood)
                continue
            
            # Evaluate new solution
            new_metrics = evaluate_route(new_solution, instance, distance_data, fleet_info)
            
            # Check if better and feasible
            if is_solution_better(new_metrics, current_metrics):
                # Accept improvement
                current_solution = new_solution
                current_metrics = new_metrics
                improved_this_iteration = True
                neighborhood_used = neighborhood
                
                # Reset NL_intra (restart with full neighborhood list)
                break
            else:
                # No improvement, remove neighborhood
                NL_intra.remove(neighborhood)
        
        iter_count += 1
        
        # LOG EVERY ITERATION (not just improvements)
        iteration_logs.append({
            "iteration_id": iter_count,
            "phase": "RVND-INTRA",
            "neighborhood": neighborhood_used if improved_this_iteration else "none",
            "improved": improved_this_iteration,
            "routes_snapshot": [current_solution[:]],
            "total_distance": current_metrics["total_distance"],
            "total_service_time": current_metrics["total_service_time"],
            "total_travel_time": current_metrics["total_travel_time"],
            "objective": current_metrics["objective"]
        })
        
        # If NL_intra was exhausted without improvement, we're done
        if not NL_intra:
            break
    
    # Add iteration logs to metrics
    current_metrics["iteration_logs"] = iteration_logs
    
    return current_metrics


# ========== INTER-ROUTE RVND ==========

def rvnd_inter(
    routes: List[Dict],
    instance: dict,
    distance_data: dict,
    fleet_list: List[Dict],
    rng: random.Random,
    max_iterations: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Global Inter-route RVND.
    Returns: (optimized_routes, iteration_logs)
    """
    iteration_logs = []
    current_distance = sum(r["total_distance"] for r in routes)
    distance_matrix = distance_data["distance_matrix"]
    
    current_demands = [r["total_demand"] for r in routes]
    _, best_unassigned = can_assign_fleet(current_demands, fleet_list)
    
    NL_FULL = INTER_ROUTE_NEIGHBORHOODS[:]
    NL = NL_FULL[:]
    
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        if not NL:
            break
        
        neighborhood = rng.choice(NL)
        result = apply_inter_neighborhood(
            neighborhood, routes, instance, distance_matrix, fleet_list, rng
        )
        
        accepted = False
        if result.get("accepted"):
            new_unassigned = result["unassigned_after"]
            new_dist = result["distance_after"]
            
            # Acceptance logic: same as in evaluate_move
            if new_unassigned < best_unassigned:
                accepted = True
            elif new_unassigned == best_unassigned:
                if new_dist < current_distance - 1e-4:
                    accepted = True
            
        if accepted:
            routes = result["new_routes"]
            current_distance = result["distance_after"]
            best_unassigned = result["unassigned_after"]
            NL = NL_FULL[:] # Reset NL
            
            iteration_logs.append({
                "iteration_id": iteration,
                "phase": "RVND-INTER",
                "neighborhood": neighborhood,
                "improved": True,
                "total_distance": round(current_distance, 2),
                "unassigned": best_unassigned,
                "move": result.get("move_details")
            })
            continue

        if neighborhood in NL:
            NL.remove(neighborhood)
            
        iteration_logs.append({
            "iteration_id": iteration,
            "phase": "RVND-INTER",
            "neighborhood": neighborhood,
            "improved": False,
            "total_distance": round(current_distance, 2),
            "unassigned": best_unassigned
        })

    return routes, iteration_logs


# ========== MAIN RVND CONTROLLER ==========

def rvnd_route(route: Dict, instance: Dict, distance_data: Dict, rng: random.Random) -> Dict:
    """
    Intra-route RVND part.
    """
    sequence = deepcopy(route["sequence"])
    vehicle_type = route["vehicle_type"]
    fleet_data = {fleet["id"]: fleet for fleet in instance["fleet"]}
    fleet_info = fleet_data[vehicle_type]
    
    final_metrics = rvnd_intra(
        sequence=sequence,
        instance=instance,
        distance_data=distance_data,
        fleet_info=fleet_info,
        rng=rng,
        max_iterations=MAX_INTRA_ITERATIONS
    )
    return final_metrics


def main() -> None:
    instance = load_json(INSTANCE_PATH)
    distance_data = load_json(DISTANCE_PATH)
    acs_data = load_json(ACS_PATH)

    rng = random.Random(84)
    fleet_data = {fleet["id"]: fleet for fleet in instance["fleet"]}

    # --- 1. PREPARE INITIAL ROUTES ---
    initial_routes = []
    summary = {
        "distance_before": 0.0,
        "distance_after": 0.0,
        "objective_before": 0.0,
        "objective_after": 0.0,
        "tw_before": 0.0,
        "tw_after": 0.0,
        "capacity_violations_before": 0,
        "capacity_violations_after": 0
    }

    for route_data in acs_data["clusters"]:
        fleet_info = fleet_data[route_data["vehicle_type"]]
        metrics = evaluate_route(route_data["sequence"], instance, distance_data, fleet_info)
        
        summary["distance_before"] += metrics["total_distance"]
        summary["objective_before"] += metrics["objective"]
        summary["tw_before"] += metrics["total_tw_violation"]
        summary["capacity_violations_before"] += (1 if metrics["capacity_violation"] > 0 else 0)
        
        initial_routes.append({
            "cluster_id": route_data["cluster_id"],
            "vehicle_type": route_data["vehicle_type"],
            "sequence": route_data["sequence"],
            "total_distance": metrics["total_distance"],
            "total_demand": metrics["total_demand"]
        })

    # --- 2. GLOBAL INTER-ROUTE OPTIMIZATION ---
    optimized_routes, inter_logs = rvnd_inter(
        initial_routes, instance, distance_data, instance["fleet"], rng, MAX_INTER_ITERATIONS
    )

    # --- 3. INTRA-ROUTE OPTIMIZATION FOR EACH ---
    results = []
    all_iteration_logs = inter_logs[:]

    for route in optimized_routes:
        improved = rvnd_route(route, instance, distance_data, rng)
        
        summary["distance_after"] += improved["total_distance"]
        summary["objective_after"] += improved["objective"]
        summary["tw_after"] += improved["total_tw_violation"]
        summary["capacity_violations_after"] += (1 if improved["capacity_violation"] > 0 else 0)

        results.append({
            "cluster_id": route["cluster_id"],
            "vehicle_type": route["vehicle_type"],
            "baseline": None, # rvnd_inter already changed baseline
            "improved": improved
        })
        
        if "iteration_logs" in improved:
            for log in improved["iteration_logs"]:
                log["cluster_id"] = route["cluster_id"]
                all_iteration_logs.append(log)

    # --- 4. VEHICLE REASSIGNMENT ---
    used_vehicles = {}
    reassigned_results = []
    
    for route_result in results:
        improved = route_result["improved"]
        total_demand = improved["total_demand"]
        new_vehicle_type = assign_vehicle_by_demand(total_demand, instance["fleet"], used_vehicles)
        
        if new_vehicle_type is None:
            new_vehicle_type = route_result["vehicle_type"]
            improved["vehicle_assignment_failed"] = True
        else:
            used_vehicles[new_vehicle_type] = used_vehicles.get(new_vehicle_type, 0) + 1
            improved["vehicle_assignment_failed"] = False
        
        if new_vehicle_type != route_result["vehicle_type"]:
            fleet_info = fleet_data[new_vehicle_type]
            improved = evaluate_route(improved["sequence"], instance, distance_data, fleet_info)
            improved["vehicle_assignment_failed"] = False
        
        reassigned_results.append({
            "cluster_id": route_result["cluster_id"],
            "vehicle_type": new_vehicle_type,
            "improved": improved
        })

    summary["distance_after"] = sum(r["improved"]["total_distance"] for r in reassigned_results)
    summary["objective_after"] = sum(r["improved"]["objective"] for r in reassigned_results)
    summary["tw_after"] = sum(r["improved"]["total_tw_violation"] for r in reassigned_results)
    summary["capacity_violations_after"] = sum(1 for r in reassigned_results if r["improved"]["capacity_violation"] > 0)

    output = {
        "routes": reassigned_results,
        "summary": summary,
        "parameters": {
            "inter_neighborhoods": INTER_ROUTE_NEIGHBORHOODS,
            "intra_neighborhoods": INTRA_ROUTE_NEIGHBORHOODS,
            "max_inter_iterations": MAX_INTER_ITERATIONS,
            "max_intra_iterations": MAX_INTRA_ITERATIONS,
            "seed": 84
        },
        "vehicle_usage": used_vehicles,
        "iteration_logs": all_iteration_logs
    }

    with RVND_PATH.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(
        "RVND Results (GLOBAL):\n"
        f"  Distance: {round(summary['distance_before'], 3)} -> {round(summary['distance_after'], 3)}\n"
        f"  Objective: {round(summary['objective_before'], 3)} -> {round(summary['objective_after'], 3)}\n"
        f"  TW Violations: {round(summary['tw_before'], 3)} -> {round(summary['tw_after'], 3)}\n"
        f"  Capacity Violations: {summary['capacity_violations_before']} -> {summary['capacity_violations_after']}"
    )


if __name__ == "__main__":
    main()
