import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
INSTANCE_PATH = DATA_DIR / "parsed_instance.json"
DISTANCE_PATH = DATA_DIR / "parsed_distance.json"
CLUSTERS_PATH = DATA_DIR / "clusters.json"
INITIAL_ROUTES_PATH = DATA_DIR / "initial_routes.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_time_to_minutes(value: str) -> float:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def minutes_to_clock(value: float) -> str:
    hours = int(value // 60)
    minutes = int(value % 60)
    seconds = (value - math.floor(value)) * 60
    if seconds < 1e-6:
        return f"{hours:02d}:{minutes:02d}"
    return f"{hours:02d}:{minutes:02d}+{seconds:02.0f}s"


def compute_polar_angle(customer: dict, depot: dict, academic_mode: bool = False) -> float:
    angle = math.atan2(customer["y"] - depot["y"], customer["x"] - depot["x"])
    if angle < 0:
        angle += 2 * math.pi
    if academic_mode:
        return math.degrees(angle)
    return angle


def build_clusters(instance: dict, academic_mode: bool = False) -> Tuple[List[dict], Dict[str, int], List[dict]]:
    depot = instance["depot"]
    customers = instance["customers"]
    fleets = instance["fleet"]

    iteration_logs = []

    customer_angles = []
    for customer in customers:
        angle = compute_polar_angle(customer, depot, academic_mode)
        customer_angles.append({"customer": customer, "angle": angle})
        if academic_mode:
            iteration_logs.append({
                "phase": "SWEEP",
                "step": "polar_angle",
                "customer_id": customer["id"],
                "angle": round(angle, 2),
                "formula": f"atan2(Δy, Δx) × 180/π = {round(angle, 2)}°",
                "description": f"Menghitung sudut polar pelanggan {customer['id']} terhadap depot: {round(angle, 2)}°."
            })

    customer_angles.sort(key=lambda item: item["angle"])
    
    if academic_mode:
        iteration_logs.append({
            "phase": "SWEEP",
            "step": "sorted_order",
            "order": [c["customer"]["id"] for c in customer_angles],
            "description": "Pelanggan diurutkan berdasarkan besaran sudut polar (secara menaik) untuk menentukan urutan pengelompokan."
        })

    available_units = {fleet["id"]: fleet["units"] for fleet in fleets}
    # For greedy selection in non-academic mode
    fleets_sorted = sorted(fleets, key=lambda f: f["capacity"])

    clusters = []
    cluster_id = 1
    current_customers: List[dict] = []
    current_demand = 0
    current_vehicle = None

    unassigned = customer_angles.copy()
    
    while unassigned:
        # Academic mode uses a slightly different logic: find largest available vehicle capacity
        active_fleets = [f for f in fleets if available_units[f["id"]] > 0]
        if not active_fleets: # Fallback
            max_capacity = max(f["capacity"] for f in fleets)
        else:
            max_capacity = max(f["capacity"] for f in active_fleets)
            
        i = 0
        while i < len(unassigned):
            entry = unassigned[i]
            customer = entry["customer"]
            
            if current_demand + customer["demand"] <= max_capacity:
                current_customers.append(customer)
                current_demand += customer["demand"]
                
                if academic_mode:
                    iteration_logs.append({
                        "phase": "SWEEP",
                        "step": "customer_added",
                        "customer_id": customer["id"],
                        "demand": customer["demand"],
                        "cluster_demand": current_demand,
                        "remaining_capacity": max_capacity - current_demand,
                        "description": f"Pelanggan {customer['id']} (muatan={customer['demand']}) dimasukkan ke Cluster {cluster_id}. Sisa kapasitas: {max_capacity - current_demand}."
                    })
                
                unassigned.pop(i)
                
                # Termination check
                can_fit = any(current_demand + c["customer"]["demand"] <= max_capacity for c in unassigned)
                if not can_fit and unassigned:
                    if academic_mode:
                        iteration_logs.append({
                            "phase": "SWEEP",
                            "step": "forced_termination",
                            "cluster_id": cluster_id,
                            "reason": f"Sisa kapasitas {max_capacity - current_demand} tidak mencukupi untuk pelanggan berikutnya.",
                            "description": f"Pengelompokan Cluster {cluster_id} dihentikan karena keterbatasan kapasitas."
                        })
                    break
            else:
                i += 1
        
        if current_customers:
            # Assign vehicle
            feasible = [f for f in active_fleets if f["capacity"] >= current_demand]
            if feasible:
                chosen = min(feasible, key=lambda f: f["capacity"])
            else:
                # If no active fleet fits, maybe some exhausted one fits? Or just pick largest available
                chosen = max(fleets, key=lambda f: f["capacity"])
            
            vid = chosen["id"]
            clusters.append({
                "cluster_id": cluster_id,
                "vehicle_type": vid,
                "customer_ids": [cust["id"] for cust in current_customers],
                "total_demand": current_demand
            })
            
            if vid in available_units and available_units[vid] > 0:
                available_units[vid] -= 1
                stock_str = f" (Sisa unit: {available_units[vid]})"
            else:
                stock_str = " (Unit habis, menggunakan fallback)"

            if academic_mode:
                iteration_logs.append({
                    "phase": "SWEEP",
                    "step": "cluster_formed",
                    "cluster_id": cluster_id,
                    "customer_ids": [c["id"] for c in current_customers],
                    "total_demand": current_demand,
                    "vehicle_type": vid,
                    "vehicle_reason": f"Kapasitas {chosen['capacity']} mencukupi demand {current_demand}{stock_str}"
                })
            
            cluster_id += 1
            current_customers = []
            current_demand = 0

    used_units = {fid: f["units"] - available_units.get(fid, 0) for fid, f in {f["id"]: f for f in fleets}.items()}

    return clusters, used_units, iteration_logs


def nearest_neighbor_route(cluster: dict, instance: dict, distance_data: dict, academic_mode: bool = False) -> Tuple[dict, List[dict]]:
    """
    Build route using Nearest Neighbor with Time Window awareness.
    
    ACADEMIC MODE special handling:
    - Time Window is a HARD constraint.
    - If arrival > TW_end, customer is rejected and added to unassigned.
    - Detailed step-by-step logging.
    """
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

    iteration_logs = []
    unvisited = cluster["customer_ids"].copy()
    route_sequence = [0]
    unassigned = []
    current_node = 0
    
    total_distance = 0.0
    total_wait_time = 0.0
    total_service_time = 0.0
    total_violation = 0.0
    current_time = depot_tw["start"]
    
    stops = [{
        "node_id": 0,
        "arrival": current_time,
        "raw_arrival": current_time,
        "departure": current_time + depot_service,
        "wait": 0.0,
        "violation": 0.0,
        "tw_start": depot_tw["start"],
        "tw_end": depot_tw["end"]
    }]
    current_time += depot_service
    
    step = 1
    while unvisited:
        # Find nearest
        nearest = None
        min_dist = float('inf')
        for cid in unvisited:
            d = distance_matrix[node_index[current_node]][node_index[cid]]
            if d < min_dist:
                min_dist = d
                nearest = cid
        
        if nearest is None: break
        
        # Check TW if in academic mode
        travel = travel_matrix[node_index[current_node]][node_index[nearest]]
        arrival_time = current_time + travel
        
        cust = customers[nearest]
        tw_start = parse_time_to_minutes(cust["time_window"]["start"])
        tw_end = parse_time_to_minutes(cust["time_window"]["end"])
        
        if academic_mode and arrival_time > tw_end:
            # REJECT
            unassigned.append(nearest)
            unvisited.remove(nearest)
            iteration_logs.append({
                "phase": "NN",
                "cluster_id": cluster["cluster_id"],
                "step": step,
                "from_node": current_node,
                "to_node": nearest,
                "distance": round(min_dist, 2),
                "arrival_time": round(arrival_time, 2),
                "tw_start": tw_start,
                "tw_end": tw_end,
                "action": "DITOLAK",
                "reason": f"Waktu tiba {minutes_to_clock(arrival_time)} melewati batas operasional {minutes_to_clock(tw_end)}.",
                "description": f"Pelanggan {nearest} tidak dapat dikunjungi pada langkah ini karena melampaui jendela waktu."
            })
            step += 1
            continue
            
        # ACCEPT
        if academic_mode:
            iteration_logs.append({
                "phase": "NN",
                "cluster_id": cluster["cluster_id"],
                "step": step,
                "from_node": current_node,
                "to_node": nearest,
                "distance": round(min_dist, 2),
                "arrival_time": round(arrival_time, 2),
                "tw_start": tw_start,
                "tw_end": tw_end,
                "action": "DITERIMA",
                "description": f"Menambahkan pelanggan {nearest} ke dalam rute (jarak dari titik sebelumnya: {round(min_dist, 2)} km)."
            })
        
        route_sequence.append(nearest)
        wait = max(0.0, tw_start - arrival_time)
        service_start = max(arrival_time, tw_start)
        violation = max(0.0, arrival_time - tw_end)
        departure = service_start + cust["service_time"]
        
        total_distance += min_dist
        total_wait_time += wait
        total_service_time += cust["service_time"]
        total_violation += violation
        
        stops.append({
            "node_id": nearest,
            "raw_arrival": arrival_time,
            "arrival": service_start,
            "arrival_str": minutes_to_clock(service_start),
            "departure": departure,
            "departure_str": minutes_to_clock(departure),
            "wait": wait,
            "violation": violation,
            "tw_start": tw_start,
            "tw_end": tw_end
        })
        
        current_time = departure
        current_node = nearest
        unvisited.remove(nearest)
        step += 1
        
    # Return to depot
    if len(route_sequence) > 1:
        dist_back = distance_matrix[node_index[current_node]][0]
        travel_back = travel_matrix[node_index[current_node]][0]
        total_distance += dist_back
        route_sequence.append(0)
        final_arrival = current_time + travel_back
        
        stops.append({
            "node_id": 0,
            "arrival": final_arrival,
            "departure": final_arrival,
            "wait": 0.0,
            "violation": 0.0,
            "tw_start": depot_tw["start"],
            "tw_end": depot_tw["end"]
        })
        
        if academic_mode:
            iteration_logs.append({
                "phase": "NN",
                "cluster_id": cluster["cluster_id"],
                "step": step,
                "from_node": current_node,
                "to_node": 0,
                "distance": round(dist_back, 2),
                "description": f"Rute selesai, armada kembali ke depot (jarak tempuh akhir: {round(dist_back, 2)} km)."
            })

    # Recalculate demands for served
    served_demand = sum(customers[c]["demand"] for c in route_sequence if c != 0)

    route_data = {
        "cluster_id": cluster["cluster_id"],
        "vehicle_type": cluster["vehicle_type"],
        "sequence": route_sequence,
        "stops": stops,
        "total_distance": round(total_distance, 2),
        "total_tw_violation": round(total_violation, 2),
        "total_wait_time": round(total_wait_time, 2),
        "total_service_time": total_service_time,
        "total_travel_time": round(total_distance, 2),
        "unassigned_customers": unassigned,
        "served_demand": served_demand,
        "total_demand": cluster["total_demand"]
    }
    
    return route_data, iteration_logs


def save_clusters(clusters: List[dict], fleet_usage: Dict[str, int]) -> None:
    payload = {
        "total_clusters": len(clusters),
        "clusters": clusters,
        "fleet_usage": fleet_usage
    }
    with CLUSTERS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_initial_routes(routes: List[dict]) -> None:
    payload = {
        "routes": routes,
        "units": {"time": "minutes", "distance": "km"}
    }
    with INITIAL_ROUTES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    print("PROGRESS:sweep_nn:0:starting sweep_nn")
    instance = load_json(INSTANCE_PATH)
    distance_data = load_json(DISTANCE_PATH)
    print("PROGRESS:sweep_nn:20:loaded inputs")

    clusters, fleet_usage, _ = build_clusters(instance)
    print(f"PROGRESS:sweep_nn:50:built {len(clusters)} clusters")

    routes = []
    for cluster in clusters:
        route, _ = nearest_neighbor_route(cluster, instance, distance_data)
        routes.append(route)
    print("PROGRESS:sweep_nn:80:constructed initial routes")

    save_clusters(clusters, fleet_usage)
    save_initial_routes(routes)
    print("PROGRESS:sweep_nn:100:done")

    total_demand = sum(cluster["total_demand"] for cluster in clusters)
    total_distance = sum(route["total_distance"] for route in routes)
    violations = sum(route["total_tw_violation"] for route in routes)
    wait_time = sum(route.get("total_wait_time", 0) for route in routes)
    service_time = sum(route.get("total_service_time", 0) for route in routes)
    
    print(
        "sweep_nn: clusters=", len(clusters),
        ", fleet_usage=", fleet_usage,
        ", total_demand=", total_demand,
        ", total_distance=", round(total_distance, 2),
        ", tw_violation=", round(violations, 2),
        ", wait_time=", round(wait_time, 2),
        ", service_time=", round(service_time, 2),
        sep=""
    )


if __name__ == "__main__":
    main()
