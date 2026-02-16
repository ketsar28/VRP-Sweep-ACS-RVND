import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
INSTANCE_PATH = DATA_DIR / "parsed_instance.json"
DISTANCE_PATH = DATA_DIR / "parsed_distance.json"
ACS_PATH = DATA_DIR / "acs_routes.json"
RVND_PATH = DATA_DIR / "rvnd_routes.json"

# RVND Configuration
MAX_INTER_ITERATIONS = 300
MAX_INTRA_ITERATIONS = 400

# Neighborhood definitions
INTER_ROUTE_NEIGHBORHOODS = ["shift_1_0", "shift_2_0", "swap_1_1", "swap_2_1", "swap_2_2", "cross"]
INTRA_ROUTE_NEIGHBORHOODS = ["two_opt", "or_opt", "reinsertion", "exchange"]
MAX_LOG_CANDIDATES = 20


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


def can_assign_fleet(demands: List[float], fleet_data: List[Dict]) -> Tuple[bool, int, float]:
    """
    Checks if demands can be assigned to fleets. Efficiently.
    Returns: (feasible, unassigned_count, penalty_magnitude)
    """
    # 1. Expand fleet (Pre-sorted available units - just the capacities)
    available_units = []
    for f in fleet_data:
        cap = f["capacity"]
        for _ in range(f["units"]):
            available_units.append(cap)
    
    # Sort ASC for best-fit logic
    available_units.sort()
    
    # Sort demands DESC (hardest to fit first)
    sorted_demands = sorted(demands, reverse=True)
    
    unassigned_count = 0
    total_excess = 0.0
    
    # Track used unit indices manually to avoid object copying
    used_indices = set()
    
    for d in sorted_demands:
        assigned = False
        # Try best fit
        for idx, cap in enumerate(available_units):
            if idx not in used_indices and cap >= d:
                used_indices.add(idx)
                assigned = True
                break
        
        if not assigned:
            unassigned_count += 1
            # Penalty: find largest available unused unit
            best_unused_cap = 0
            best_idx = -1
            for idx in range(len(available_units)-1, -1, -1):
                if idx not in used_indices:
                    best_unused_cap = available_units[idx]
                    best_idx = idx
                    break
            
            if best_unused_cap > 0:
                total_excess += max(0.0, d - best_unused_cap)
                used_indices.add(best_idx)
            else:
                total_excess += d

    # Add Load Balancing Score (Sum of Squares) scaled down
    # This ensures that wider spreads (less balanced) are penalized slightly
    load_balance_score = sum(d**2 for d in demands) * 0.0001
    
    final_penalty = total_excess + load_balance_score

    return (unassigned_count == 0, unassigned_count, final_penalty)


def generate_capacity_log_candidate(routes: List[Dict], demands: List[float], instance: Dict, fleet_data: List[Dict], delta: float = 0.0) -> Dict:
    """
    Generates a 'candidate' metadata object for logging.
    This matches the format expected by academic_replay_tab.py for the matrix view.
    """
    route_sequences = []
    route_loads = []
    
    # 1. Expand fleet for greedy check
    available_units = []
    fleet_catalog = {f["id"]: f for f in fleet_data}
    unique_ids = sorted(list(fleet_catalog.keys()))
    
    # Count total available for reason string
    total_stock = {fid: fleet_catalog[fid]["units"] for fid in unique_ids}
    used_stock = {fid: 0 for fid in unique_ids}

    for fid in unique_ids:
        f = fleet_catalog[fid]
        for _ in range(f["units"]):
            available_units.append({
                "id": fid,
                "capacity": f["capacity"],
                "used": False
            })
    available_units.sort(key=lambda x: x["capacity"]) # Best Fit
    
    
    indexed_demands = sorted(enumerate(demands), key=lambda x: x[1], reverse=True)
    assignments = {} 
    
    for idx, d in indexed_demands:
        assigned_fid = "X"
        for unit in available_units:
            if not unit["used"] and unit["capacity"] >= d:
                unit["used"] = True
                assigned_fid = unit["id"]
                used_stock[assigned_fid] += 1
                break
        assignments[idx] = assigned_fid
        
    for i in range(len(routes)):
        route_sequences.append("→".join(map(str, routes[i]["sequence"])))
        fid = assignments.get(i, "X")
        route_loads.append(f"{demands[i]:.1f} kg ({fid})")
        
    is_feasible = all(fid != "X" for fid in assignments.values())
    
    stock_parts = [f"{fid}:{used_stock[fid]}/{total_stock[fid]}" for fid in unique_ids]
    stock_str = ", ".join(stock_parts)
    
    max_capacity = max((f["capacity"] for f in fleet_data), default=0)
    
    if is_feasible:
        reason = f"Terangkut ({stock_str})"
        reason_detail = f"Semua rute terangkut oleh armada tersedia. {stock_str}"
    else:
        x_count = list(assignments.values()).count("X")
        # Cari muatan yang gagal
        failed_loads = [demands[i] for i, fid in assignments.items() if fid == "X"]
        highest_failed = max(failed_loads) if failed_loads else 0
        
        reason = f"X: {x_count} Rute Berlebih"
        
        if highest_failed > max_capacity:
            reason_detail = f"Gagal: Muatan rute ({highest_failed:.1f}kg) melebihi kapasitas terbesar ({max_capacity}kg). Stok: {stock_str}"
        else:
            reason_detail = f"Gagal: Rute layak tapi stok unit tidak cukup. Stok terpakai: {stock_str}"

    return {
        "detail": "Accepted" if routes else "Stagnan",
        "route_sequences": route_sequences,
        "route_loads": route_loads,
        "feasible": is_feasible,
        "reason": reason,
        "reason_detail": reason_detail,
        "delta": round(delta, 2),
        "total_distance": round(sum(r.get("total_distance", 0) for r in routes), 2) if routes else 0.0,
        "route_distances": [r.get("total_distance", 0) for r in routes] if routes else [],
        "tw_status": "Calculating..." # To be filled by caller
    }


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_time_to_minutes(value: str) -> float:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def minutes_to_clock(value: float) -> str:
    """Format minutes to clock string HH:MM:SS."""
    h = int(value // 60)
    m = int(value % 60)
    s = int((value * 60) % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def evaluate_route(sequence: List[int], instance: dict, distance_data: dict, fleet_info: dict = None, academic_mode: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single route with capacity and time window constraints.
    In AR mode, TW is SOFT - violations are logged but don't disqualify.
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

    stops = []
    total_distance = 0.0
    total_travel_time = 0.0
    total_service_time = 0.0
    total_violation = 0.0
    total_wait_time = 0.0
    total_demand = 0.0

    current_node = sequence[0]
    current_time = depot_tw["start"]
    
    # Depot start
    stops.append({
        "node_id": 0,
        "arrival": current_time,
        "arrival_str": minutes_to_clock(current_time),
        "departure": current_time + depot_service,
        "departure_str": minutes_to_clock(current_time + depot_service),
        "wait": 0.0,
        "violation": 0.0,
        "tw_start": depot_tw["start"],
        "tw_end": depot_tw["end"]
    })
    current_time += depot_service

    for next_node in sequence[1:]:
        dist = distance_matrix[node_index[current_node]][node_index[next_node]]
        travel = travel_matrix[node_index[current_node]][node_index[next_node]]
        total_distance += dist
        total_travel_time += travel
        
        arrival_no_wait = current_time + travel
        
        if next_node == 0:
            tw_start, tw_end = depot_tw["start"], depot_tw["end"]
            service = 0
            demand = 0
        else:
            cust = customers[next_node]
            tw_start = parse_time_to_minutes(cust["time_window"]["start"])
            tw_end = parse_time_to_minutes(cust["time_window"]["end"])
            service = cust["service_time"]
            demand = cust["demand"]
            total_demand += demand

        wait = max(0.0, tw_start - arrival_no_wait)
        service_start = max(arrival_no_wait, tw_start)
        violation = max(0.0, arrival_no_wait - tw_end)
        departure = service_start + service
        
        if next_node != 0:
            total_violation += violation
            total_service_time += service
            total_wait_time += wait

        stops.append({
            "node_id": next_node,
            "raw_arrival": arrival_no_wait,
            "arrival": service_start,
            "arrival_str": minutes_to_clock(service_start),
            "departure": departure,
            "departure_str": minutes_to_clock(departure),
            "wait": wait,
            "violation": violation,
            "tw_start": tw_start,
            "tw_end": tw_end
        })
        current_node = next_node
        current_time = departure

    capacity = 100
    vtype = "Unknown"
    if fleet_info:
        vtype = fleet_info["id"]
        capacity = fleet_info["capacity"]
    elif "vehicle_type" in instance:
        vtype = instance["vehicle_type"]
        
    capacity_violation = max(0.0, total_demand - capacity)
    is_feasible = capacity_violation < 0.001
    
    if not academic_mode:
        if total_violation > 0.001:
            is_feasible = False
            
    time_component = total_travel_time + total_service_time
    objective = total_distance + time_component + total_violation

    return {
        "sequence": sequence,
        "stops": stops,
        "total_distance": round(total_distance, 3),
        "total_travel_time": round(total_travel_time, 3),
        "total_service_time": total_service_time,
        "total_time_component": round(time_component, 3),
        "total_tw_violation": round(total_violation, 3),
        "total_wait_time": round(total_wait_time, 3),
        "total_demand": total_demand,
        "capacity_violation": capacity_violation,
        "objective": round(objective, 3),
        "is_feasible": is_feasible,
        "vehicle_type": vtype,
        "capacity": capacity
    }


def is_solution_better(new_metrics: Dict, current_metrics: Dict) -> bool:
    """
    Check if new solution is better.
    STRICT VERSION:
    1. Reject if capacity exceeded or TW violated.
    2. If both are feasible, pick based on distance.
    3. If current is infeasible but new is better (less violation), accept.
    """
    # Calculate global feasibility score
    # We want to minimize (unassigned_count * 1000000 + tw_violation * 1000 + distance)
    # But current_metrics already has these.
    
    new_viol = new_metrics.get("total_tw_violation", 0.0)
    curr_viol = current_metrics.get("total_tw_violation", 0.0)
    new_cap = new_metrics.get("capacity_violation", 0.0)
    curr_cap = current_metrics.get("capacity_violation", 0.0)
    
    # Absolute rejection: If new is worse in ANY constraint, reject
    if new_cap > curr_cap + 1e-4:
        return False
    # STRICT TW: Reject if new violation increases OR if it introduces new violation where none existed
    if new_viol > curr_viol + 1e-4:
        return False
    if new_viol > 0.001 and curr_viol < 0.001:
        return False
    
    # If new is BETTER in ANY constraint, accept
    if new_cap < curr_cap - 1e-4:
        return True
    if new_viol < curr_viol - 1e-4:
        return True
    
    # Constraints are equal: compare distance
    return new_metrics["total_distance"] < current_metrics["total_distance"] - 1e-4


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


def apply_intra_neighborhood(
    neighborhood: str,
    sequence: List[int],
    instance: dict,
    distance_data: dict,
    fleet_info: dict,
    academic_mode: bool = False
) -> Tuple[bool, List[int], float, List[dict]]:
    """
    Apply intra-route neighborhood move. 
    In Academic Mode: logs ALL trial moves.
    """
    best_seq = sequence[:]
    curr_metrics = evaluate_route(sequence, instance, distance_data, fleet_info, academic_mode)
    best_dist = curr_metrics["total_distance"]
    improved = False
    candidates = []

    def log_trial(new_seq, move_name, detail):
        metrics = evaluate_route(new_seq, instance, distance_data, fleet_info, academic_mode)
        delta = round(metrics["total_distance"] - best_dist, 2)
        if len(candidates) < MAX_LOG_CANDIDATES:
            candidates.append({
                "type": move_name,
                "detail": detail,
                "feasible": metrics["is_feasible"],
                "delta": delta,
                "total_distance": metrics["total_distance"],
                "reason": "Layak" if metrics["is_feasible"] else "Melanggar Kapasitas/TW",
                "route_sequences": ["-".join(map(str, new_seq))]
            })
        return metrics

    n = len(sequence)
    if neighborhood == "two_opt":
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_seq = intra_two_opt(sequence, i, j)
                metrics = log_trial(new_seq, "two_opt", f"reverse({i},{j})")
                if metrics["is_feasible"] and round(metrics["total_distance"], 2) < round(best_dist, 2):
                    best_dist = metrics["total_distance"]
                    best_seq = new_seq
                    improved = True

    elif neighborhood == "or_opt":
        for length in [1, 2, 3]:
            for i in range(1, n - length):
                for j in range(1, n - length):
                    if i == j:
                        continue
                    new_seq = intra_or_opt(sequence, i, length, j)
                    metrics = log_trial(new_seq, "or_opt", f"move({i}:{i+length}->{j})")
                    if metrics["is_feasible"] and round(metrics["total_distance"], 2) < round(best_dist, 2):
                        best_dist = metrics["total_distance"]
                        best_seq = new_seq
                        improved = True

    elif neighborhood == "reinsertion":
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if i == j: continue
                new_seq = intra_reinsertion(sequence, i, j)
                metrics = log_trial(new_seq, "reinsertion", f"move({i}->{j})")
                if metrics["is_feasible"] and round(metrics["total_distance"], 2) < round(best_dist, 2):
                    best_dist = metrics["total_distance"]
                    best_seq = new_seq
                    improved = True

    elif neighborhood == "exchange":
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_seq = intra_exchange(sequence, i, j)
                metrics = log_trial(new_seq, "exchange", f"swap({i},{j})")
                if metrics["is_feasible"] and round(metrics["total_distance"], 2) < round(best_dist, 2):
                    best_dist = metrics["total_distance"]
                    best_seq = new_seq
                    improved = True

    return improved, best_seq, best_dist, candidates


# ========== INTER-ROUTE NEIGHBORHOODS ==========
# Note: For single-route optimization, inter-route operators are not applicable
# These are placeholders for multi-route scenarios

def apply_inter_neighborhood(
    neighborhood: str,
    routes: List[Dict],
    instance: Dict,
    distance_data: Dict,
    fleet_list: List[Dict],
    rng: random.Random,
    academic_mode: bool = False
) -> Dict:
    """
    Apply inter-route neighborhood operator (Global RVND).
    STOCK-AWARE: Moves must lead to a valid fleet assignment.
    """
    fleet_data = {f["id"]: f for f in fleet_list}
    customers = {c["id"]: c for c in instance["customers"]}
    current_distance = sum(r["total_distance"] for r in routes)
    
    best_distance = current_distance
    best_routes = routes
    best_move = None
    
    current_demands = [r["total_demand"] for r in routes]
    _, current_unassigned, current_penalty = can_assign_fleet(current_demands, fleet_list)
    best_unassigned = current_unassigned
    best_penalty = current_penalty

    distance_matrix = distance_data["distance_matrix"]

    def calc_route_distance(seq):
        node_index = {node["id"]: idx for idx, node in enumerate(distance_data["nodes"])}
        return sum(distance_matrix[node_index[seq[k]]][node_index[seq[k+1]]] for k in range(len(seq)-1))

    def calc_route_demand(seq, customers_dict):
        return sum(customers_dict[c]["demand"] for c in seq if c != 0)

    log_candidates = []

    def evaluate_move(i, j, new_seq_a, new_seq_b, move_name, detail):
        nonlocal best_distance, best_routes, best_move, best_unassigned, best_penalty, log_candidates
        
        dist_a = calc_route_distance(new_seq_a)
        dist_b = calc_route_distance(new_seq_b)
        demand_a = calc_route_demand(new_seq_a, customers)
        demand_b = calc_route_demand(new_seq_b, customers)
        
        total_new_dist = current_distance - routes[i]["total_distance"] - routes[j]["total_distance"] + dist_a + dist_b
        
        new_demands = current_demands[:]
        new_demands[i] = demand_a
        new_demands[j] = demand_b
        
        # Evaluate with evaluate_route to check TW
        # Fallback for vehicle_type if X
        vt_a = routes[i]["vehicle_type"] if routes[i]["vehicle_type"] in fleet_data else next(iter(fleet_data))
        vt_b = routes[j]["vehicle_type"] if routes[j]["vehicle_type"] in fleet_data else next(iter(fleet_data))
        
        metrics_a = evaluate_route(new_seq_a, instance, distance_data, fleet_data[vt_a], academic_mode)
        metrics_b = evaluate_route(new_seq_b, instance, distance_data, fleet_data[vt_b], academic_mode)
        
        tw_viol_new = metrics_a["total_tw_violation"] + metrics_b["total_tw_violation"]
        tw_viol_old = routes[i]["total_tw_violation"] + routes[j]["total_tw_violation"]
        
        # Check global fleet feasibility
        feasible_fleet, unassigned, penalty = can_assign_fleet(new_demands, fleet_list)
        
        # Log this candidate
        if len(log_candidates) < MAX_LOG_CANDIDATES:
            cand_routes = deepcopy(routes)
            cand_routes[i] = metrics_a
            cand_routes[j] = metrics_b
            log_cand = generate_capacity_log_candidate(cand_routes, new_demands, instance, fleet_list, delta=total_new_dist - current_distance)
            log_cand["detail"] = detail
            log_cand["tw_status"] = "Layak" if tw_viol_new < 0.001 else f"Terlambat {tw_viol_new:.1f}m"
            log_candidates.append(log_cand)

        accepted = False
        if unassigned < best_unassigned:
            accepted = True
        elif unassigned == best_unassigned:
            # Common improvement rules for consistency
            if tw_viol_new < tw_viol_old - 0.001:
                accepted = True
            elif tw_viol_new < tw_viol_old + 0.001: # Check distance if TW same or better
                if unassigned > 0:
                    if penalty < best_penalty - 1e-4:
                        accepted = True
                    elif abs(penalty - best_penalty) < 1e-4 and total_new_dist < best_distance - 0.01:
                        accepted = True
                else:
                    if total_new_dist < best_distance - 0.01:
                        accepted = True
        
        if accepted:
            best_unassigned = unassigned
            best_penalty = penalty
            best_distance = total_new_dist
            best_routes = deepcopy(routes)
            
            # Preserve cluster_id and other metadata
            metrics_a["cluster_id"] = routes[i].get("cluster_id")
            metrics_b["cluster_id"] = routes[j].get("cluster_id")
            
            best_routes[i] = metrics_a
            best_routes[j] = metrics_b
            best_move = {"type": move_name, "detail": detail}
            return True
        return False

    n_routes = len(routes)
    if neighborhood == "swap_1_1":
        for i in range(n_routes):
            for j in range(i + 1, n_routes):
                seq_i = routes[i]["sequence"][1:-1]
                seq_j = routes[j]["sequence"][1:-1]
                for ca in seq_i:
                    for cb in seq_j:
                        new_a = [0] + [cb if c == ca else c for c in seq_i] + [0]
                        new_b = [0] + [ca if c == cb else c for c in seq_j] + [0]
                        evaluate_move(i, j, new_a, new_b, "swap_1_1", f"{ca}, {cb}")

    elif neighborhood == "shift_1_0":
        for i in range(n_routes):
            if len(routes[i]["sequence"]) < 3:
                continue
            for j in range(n_routes):
                if i == j:
                    continue
                seq_i = routes[i]["sequence"][1:-1]
                seq_j = routes[j]["sequence"][1:-1]
                for ca in seq_i:
                    new_a = [0] + [c for c in seq_i if c != ca] + [0]
                    for pos in range(len(seq_j) + 1):
                        new_b = [0] + seq_j[:pos] + [ca] + seq_j[pos:] + [0]
                        evaluate_move(i, j, new_a, new_b, "shift_1_0", f"{ca} -> R{j+1}")

    elif neighborhood == "shift_2_0":
        for i in range(n_routes):
            if len(routes[i]["sequence"]) < 4:
                continue
                seq_j = routes[j]["sequence"][1:-1]
                for idx in range(len(seq_i) - 1):
                    ca1, ca2 = seq_i[idx], seq_i[idx+1]
                    new_a = [0] + seq_i[:idx] + seq_i[idx+2:] + [0]
                    for pos in range(len(seq_j) + 1):
                        new_b = [0] + seq_j[:pos] + [ca1, ca2] + seq_j[pos:] + [0]
                        evaluate_move(i, j, new_a, new_b, "shift_2_0", f"({ca1},{ca2}) -> R{j+1}")

    elif neighborhood == "swap_2_1":
        for i in range(n_routes):
            if len(routes[i]["sequence"]) < 4:
                continue
            for j in range(n_routes):
                if i == j:
                    continue
                seq_a = routes[i]["sequence"][1:-1]
                seq_b = routes[j]["sequence"][1:-1]
                for idx_a in range(len(seq_a) - 1):
                    ca1, ca2 = seq_a[idx_a], seq_a[idx_a+1]
                    for cb in seq_b:
                        new_a = [0] + seq_a[:idx_a] + [cb] + seq_a[idx_a+2:] + [0]
                        idx_b = seq_b.index(cb)
                        new_b = [0] + seq_b[:idx_b] + [ca1, ca2] + seq_b[idx_b+1:] + [0]
                        evaluate_move(i, j, new_a, new_b, "swap_2_1", f"({ca1},{ca2}), {cb}")

    elif neighborhood == "swap_2_2":
        for i in range(n_routes):
            if len(routes[i]["sequence"]) < 4:
                continue
            for j in range(i + 1, n_routes):
                if len(routes[j]["sequence"]) < 4:
                    continue
                seq_a = routes[i]["sequence"][1:-1]
                seq_b = routes[j]["sequence"][1:-1]
                for idx_a in range(len(seq_a) - 1):
                    ca1, ca2 = seq_a[idx_a], seq_a[idx_a+1]
                    for idx_b in range(len(seq_b) - 1):
                        cb1, cb2 = seq_b[idx_b], seq_b[idx_b+1]
                        new_a = [0] + seq_a[:idx_a] + [cb1, cb2] + seq_a[idx_a+2:] + [0]
                        new_b = [0] + seq_b[:idx_b] + [ca1, ca2] + seq_b[idx_b+2:] + [0]
                        evaluate_move(i, j, new_a, new_b, "swap_2_2", f"({ca1},{ca2}), ({cb1},{cb2})")

    elif neighborhood == "cross":
        for i in range(n_routes):
            for j in range(i + 1, n_routes):
                seq_a = routes[i]["sequence"]
                seq_b = routes[j]["sequence"]
                for cut_a in range(1, len(seq_a) - 1):
                    for cut_b in range(1, len(seq_b) - 1):
                        new_a = seq_a[:cut_a] + seq_b[cut_b:]
                        new_b = seq_b[:cut_b] + seq_a[cut_a:]
                        evaluate_move(i, j, new_a, new_b, "cross", f"cross at({cut_a},{cut_b})")

    return {
        "accepted": best_move is not None,
        "new_routes": best_routes if best_routes else routes,
        "distance_before": current_distance,
        "distance_after": best_distance,
        "best_move": best_move,
        "candidates": log_candidates,
        "unassigned_after": best_unassigned,
        "penalty_magnitude": best_penalty
    }


# ========== INTRA-ROUTE RVND ==========

def rvnd_intra(
    sequence: List[int],
    instance: dict,
    distance_data: dict,
    fleet_info: dict,
    rng: random.Random,
    max_iterations: int,
    academic_mode: bool = False
) -> Dict:
    """
    Intra-route RVND with strict neighborhood list management.
    """
    current_solution = sequence[:]
    current_metrics = evaluate_route(current_solution, instance, distance_data, fleet_info, academic_mode)
    
    iter_count = 0
    iteration_logs = []
    
    while iter_count < max_iterations:
        NL_intra = INTRA_ROUTE_NEIGHBORHOODS[:]
        improved_this_iteration = False
        neighborhood_used = None
        
        while NL_intra:
            neighborhood = rng.choice(NL_intra)
            improved, new_seq, new_dist, candidates = apply_intra_neighborhood(
                neighborhood, current_solution, instance, distance_data, fleet_info, academic_mode
            )
            
            if not improved:
                NL_intra.remove(neighborhood)
                continue
            
            current_solution = new_seq
            current_metrics = evaluate_route(current_solution, instance, distance_data, fleet_info, academic_mode)
            improved_this_iteration = True
            neighborhood_used = neighborhood
            break
        
        iter_count += 1
        iteration_logs.append({
            "iteration_id": iter_count,
            "phase": "RVND-INTRA",
            "neighborhood": neighborhood_used if improved_this_iteration else "none",
            "improved": improved_this_iteration,
            "routes_snapshot": [current_solution[:]],
            "total_distance": current_metrics["total_distance"],
            "total_service_time": current_metrics["total_service_time"],
            "total_travel_time": current_metrics["total_travel_time"],
            "objective": current_metrics["objective"],
            "candidates": candidates if academic_mode else []
        })
        
        if not NL_intra:
            break
    
    current_metrics["iteration_logs"] = iteration_logs
    return current_metrics


# ========== INTER-ROUTE RVND ==========

def rvnd_inter(
    routes: List[Dict],
    instance: dict,
    distance_data: dict,
    fleet_list: List[Dict],
    rng: random.Random,
    max_iterations: int,
    academic_mode: bool = False
) -> Tuple[List[Dict], List[Dict]]:
    """RVND Controller for Inter-Route neighborhoods."""
    iteration_logs = []
    current_routes = deepcopy(routes)
    current_distance = sum(r["total_distance"] for r in current_routes)
    
    current_demands = [r["total_demand"] for r in current_routes]
    _, best_unassigned, best_penalty = can_assign_fleet(current_demands, fleet_list)
    
    NL_FULL = INTER_ROUTE_NEIGHBORHOODS[:]
    NL = NL_FULL[:]
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        if not NL:
            break
        
        neighborhood = rng.choice(NL)
        print(f"[Academic Replay] Running neighborhood: {neighborhood} (Iter {iteration})...")
        result = apply_inter_neighborhood(
            neighborhood, current_routes, instance, distance_data, fleet_list, rng, academic_mode
        )
        
        if result["accepted"]:
            current_routes = result["new_routes"]
            current_distance = result["distance_after"]
            best_unassigned = result["unassigned_after"]
            best_penalty = result["penalty_magnitude"]
            NL = NL_FULL[:]
            
            iteration_logs.append({
                "iteration_id": iteration,
                "phase": "RVND-INTER",
                "neighborhood": neighborhood,
                "improved": True,
                "routes_snapshot": [deepcopy(r["sequence"]) for r in current_routes],
                "total_distance": round(current_distance, 2),
                "unassigned": best_unassigned,
                "penalty": round(best_penalty, 2),
                "move": result.get("best_move"),
                "candidates": result.get("candidates", []) if academic_mode else []
            })
            continue

        # If academic_mode, log even failed iterations
        if academic_mode:
            iteration_logs.append({
                "iteration_id": iteration,
                "phase": "RVND-INTER",
                "neighborhood": neighborhood,
                "improved": False,
                "routes_snapshot": [deepcopy(r["sequence"]) for r in current_routes],
                "total_distance": round(current_distance, 2),
                "unassigned": best_unassigned,
                "penalty": round(best_penalty, 2),
                "message": f"Neighborhood {neighborhood} gagal menemukan perbaikan.",
                "candidates": result.get("candidates", [])
            })

        NL.remove(neighborhood)
        if not NL:
            iteration_logs.append({
                "iteration_id": iteration,
                "phase": "RVND-INTER",
                "type": "stagnant",
                "neighborhood": "All",
                "message": "Iterasi Berakhir (Konvergen).",
                "total_distance": round(current_distance, 2),
                "total_demand": sum(current_demands),
                "routes_snapshot": [deepcopy(r["sequence"]) for r in current_routes],
                "candidates": result.get("candidates", []) if academic_mode else []
            })
            
            if best_unassigned > 0:
                rebal_routes, rebal_success, rebal_info = attempt_load_rebalance(
                    current_routes, instance, distance_data, fleet_list, rng
                )
                if rebal_success:
                    current_routes = rebal_routes
                    current_distance = sum(r["total_distance"] for r in current_routes)
                    current_demands = [r["total_demand"] for r in current_routes]
                    _, best_unassigned, best_penalty = can_assign_fleet(current_demands, fleet_list)
                    NL = NL_FULL[:]
                    iteration_logs.append({
                        "iteration_id": iteration, "phase": "RVND-INTER-REBALANCE",
                        "improved": True, 
                        "routes_snapshot": [deepcopy(r["sequence"]) for r in current_routes],
                        "total_distance": round(current_distance, 2),
                        "unassigned": best_unassigned, "penalty": round(best_penalty, 2),
                        "move": rebal_info.get("detail"), "candidates": []
                    })

    return current_routes, iteration_logs


def attempt_load_rebalance(
    routes: List[Dict],
    instance: dict,
    distance_data: dict,
    fleet_list: List[Dict],
    rng: random.Random
) -> Tuple[List[Dict], bool, dict]:
    """
    Forcefully attempt to move nodes from overloaded/X routes to any available capacity.
    """
    distance_matrix = distance_data["distance_matrix"]
    fleet_data = {f["id"]: f for f in fleet_list}
    current_demands = [r["total_demand"] for r in routes]
    _, unassigned, penalty = can_assign_fleet(current_demands, fleet_list)
    
    if unassigned == 0:
        return routes, False, {}

    # Identify "problem" routes (heuristic: likely the largest ones)
    sorted_indices = sorted(range(len(current_demands)), key=lambda k: current_demands[k], reverse=True)
    
    best_routes = None
    best_unassigned = unassigned
    best_penalty = penalty

    customers = {c["id"]: c for c in instance["customers"]}

    def calc_route_distance(seq):
        return sum(distance_matrix[seq[k]][seq[k+1]] for k in range(len(seq)-1))

    def calc_route_demand(seq):
        return sum(customers[c]["demand"] for c in seq[1:-1])

    # Try shifting nodes from HEAVIEST routes to OTHERS
    for i in sorted_indices[:2]: # Only try moving from top 2 heaviest
        route_source = routes[i]
        seq_source = route_source["sequence"][1:-1]
        
        if not seq_source:
            continue

        for j in range(len(routes)):
            if i == j: 
                continue
            
            # Try moving EACH node from source to target
            for node in seq_source:
                # Construct new sequences
                new_seq_a_inner = [c for c in seq_source if c != node]
                new_seq_a = [0] + new_seq_a_inner + [0]
                
                route_target = routes[j]
                seq_target = route_target["sequence"][1:-1]
                
                # Check insert at every position? Too slow. Just best fit or random? 
                # Let's try inserting at end for speed (RVND will optimize dist later)
                new_seq_b = [0] + seq_target + [node] + [0]
                
                # Evaluate
                dem_a = calc_route_demand(new_seq_a)
                dem_b = calc_route_demand(new_seq_b)
                
                temp_demands = current_demands[:]
                temp_demands[i] = dem_a
                temp_demands[j] = dem_b
                
                _, u_new, p_new = can_assign_fleet(temp_demands, fleet_list)
                
                # STRICT IMPROVEMENT in Feasibility
                if u_new < best_unassigned or (u_new == best_unassigned and p_new < best_penalty - 1.0):
                    best_unassigned = u_new
                    best_penalty = p_new
                    best_routes = deepcopy(routes)
                    
                    # Update source/target in best_routes
                    metrics_a = evaluate_route(new_seq_a, instance, distance_data, fleet_data[best_routes[i]["vehicle_type"]])
                    metrics_b = evaluate_route(new_seq_b, instance, distance_data, fleet_data[best_routes[j]["vehicle_type"]])
                    
                    best_routes[i] = metrics_a
                    best_routes[j] = metrics_b
                    
    if best_routes:
        return best_routes, True, {"type": "REBALANCE", "detail": "Berhasil memindahkan node untuk mengurangi overload/unassigned."}
    
    return routes, False, {}


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

    # --- 1. PRE-OPTIMIZATION SETUP ---
    # Use dynamic seed for production so each run explores new paths
    base_seed = int(time.time())
    rng = random.Random(base_seed)
    
    fleet_data = {fleet["id"]: fleet for fleet in instance["fleet"]}

    # --- 1. PREPARE INITIAL ROUTES (Including Failed Clusters & Unserved Customers) ---
    all_acs_clusters = acs_data.get("clusters", []) + acs_data.get("failed_clusters", [])
    
    # Track served customers
    served_customers = set()
    for route_data in all_acs_clusters:
        for cid in route_data["sequence"]:
            if cid != 0:
                served_customers.add(cid)
    
    # Identify unserved customers
    all_customers = [c["id"] for c in instance["customers"]]
    unserved_customers = [cid for cid in all_customers if cid not in served_customers]
    
    initial_routes = []
    summary = {
        "distance_before": 0.0,
        "objective_before": 0.0,
        "tw_before": 0.0,
        "capacity_violations_before": 0
    }

    # Process ACS Routes
    for route_data in all_acs_clusters:
        # For failed clusters, vehicle_type might be "N/A"
        vt = route_data.get("vehicle_type", "N/A")
        if vt not in fleet_data:
            # Pick largest available fleet as baseline for evaluation
            largest_fleet_id = max(instance["fleet"], key=lambda x: x["capacity"])["id"]
            vt = largest_fleet_id
            
        fleet_info = fleet_data[vt]
        metrics = evaluate_route(route_data["sequence"], instance, distance_data, fleet_info)
        
        summary["distance_before"] += metrics["total_distance"]
        summary["objective_before"] += metrics["objective"]
        summary["tw_before"] += metrics["total_tw_violation"]
        summary["capacity_violations_before"] += (1 if metrics["capacity_violation"] > 0 else 0)
        
        initial_routes.append({
            "cluster_id": route_data.get("cluster_id", 999),
            "vehicle_type": vt,
            "sequence": route_data["sequence"],
            "total_distance": metrics["total_distance"],
            "total_demand": metrics["total_demand"],
            "total_tw_violation": metrics["total_tw_violation"]
        })

    # PROVIGEROUS: Handle unserved customers by adding them to new routes or existing ones
    if unserved_customers:
        print(f"DEBUG: Found {len(unserved_customers)} unserved customers. Creating recovery routes.")
        # Greedily create new routes for unserved ones
        # Use largest vehicle as baseline
        largest_fleet_id = max(instance["fleet"], key=lambda x: x["capacity"])["id"]
        
        # Simply create one route per unserved customer as recovery starting point
        for cid in unserved_customers:
            recov_seq = [0, cid, 0]
            metrics = evaluate_route(recov_seq, instance, distance_data, fleet_data[largest_fleet_id])
            initial_routes.append({
                "cluster_id": 888, # Recovery marker
                "vehicle_type": largest_fleet_id,
                "sequence": recov_seq,
                "total_distance": metrics["total_distance"],
                "total_demand": metrics["total_demand"],
                "total_tw_violation": metrics["total_tw_violation"]
            })
            summary["distance_before"] += metrics["total_distance"]
            summary["objective_before"] += metrics["objective"]
            summary["tw_before"] += metrics["total_tw_violation"]
            summary["capacity_violations_before"] += (1 if metrics["capacity_violation"] > 0 else 0)

    # --- 1.5 STRICT INITIAL FLEET ALLOCATION ---
    # Goal: Start with a valid set of vehicles from the global fleet.
    # If demand > available capacity, we force assignment to the largest available vehicle,
    # creating a penalty that RVND must resolve.
    
    print("\n[RVND] Strict Initial Fleet Allocation...")
    
    # 1. Create Pool of All Available Units
    available_units = []
    for f in instance["fleet"]:
        for _ in range(f["units"]):
            available_units.append({
                "id": f["id"],
                "capacity": f["capacity"]
            })
    
    # Sort Fleet Pool: Ascending Capacity (Smallest First -> Best Fit)
    available_units.sort(key=lambda x: x["capacity"])
    
    # 2. Sort Routes by Demand: Descending (Largest assignments first)
    # maximizing chance for large routes to get large vehicles
    initial_routes_sorted = sorted(initial_routes, key=lambda x: x["total_demand"], reverse=True)
    
    final_initial_routes = []
    
    for route in initial_routes_sorted:
        demand = route["total_demand"]
        assigned_unit = None
        
        # Try to find Best Fit (Smallest capacity >= demand)
        best_fit_idx = -1
        for i, unit in enumerate(available_units):
            if unit["capacity"] >= demand:
                best_fit_idx = i
                break
        
        if best_fit_idx != -1:
            # Found a valid vehicle
            assigned_unit = available_units.pop(best_fit_idx)
            route["vehicle_assignment_note"] = "Best Fit"
        else:
            # No valid vehicle found.
            # Assign LARGEST available vehicle to minimize violation.
            if available_units:
                # available_units is sorted ASC, so largest is at end
                assigned_unit = available_units.pop()
                route["vehicle_assignment_note"] = "Forced Overflow ( Largest Available)"
            else:
                # No vehicles left at all!
                # This is a critical infeasibility (more routes than vehicles).
                # We keep the original vehicle type but it's effectively "X".
                route["vehicle_assignment_note"] = "X (No Vehicles Left)"
                # We don't change vehicle_type here, preserving original intent 
                # but valid solutions are impossible without merging.
        
        if assigned_unit:
            route["vehicle_type"] = assigned_unit["id"]
            
            # Re-evaluate metrics with new vehicle (capacity might differ)
            fleet_info_new = fleet_data[assigned_unit["id"]]
            metrics = evaluate_route(route["sequence"], instance, distance_data, fleet_info_new)
            
            route["total_distance"] = metrics["total_distance"]
            route["total_demand"] = metrics["total_demand"]
            route["capacity_violation"] = metrics["capacity_violation"] # Crucial for penalty
            route["objective"] = metrics["objective"]
            route["total_tw_violation"] = metrics["total_tw_violation"]
            
            # If violated, this objective will be HIGH, driving optimization.
        
        final_initial_routes.append(route)

    # 3. Turn Remaining Vehicles into Empty Routes
    next_cluster_id = max((r["cluster_id"] for r in initial_routes), default=0) + 1
    
    for unit in available_units:
        final_initial_routes.append({
            "cluster_id": next_cluster_id,
            "vehicle_type": unit["id"],
            "sequence": [0, 0],
            "total_distance": 0.0,
            "total_demand": 0.0,
            "capacity_violation": 0.0,
            "objective": 0.0,
            "total_tw_violation": 0.0,
            "is_empty_injection": True
        })
        next_cluster_id += 1
        
    # Restore original order (by Cluster ID) for stability, though RVND doesn't care
    # Actually, let's keep them in the processed order or just valid list
    initial_routes = final_initial_routes
    
    print(f"[RVND] Total routes after strict allocation: {len(initial_routes)}")
    for r in initial_routes:
        note = r.get("vehicle_assignment_note", "Injection")
        v_type = r["vehicle_type"]
        d = r["total_demand"]
        cap = fleet_data[v_type]["capacity"]
        print(f"  - ID {r['cluster_id']}: Demand {d:.1f} -> Fleet {v_type} (Cap {cap}) [{note}]")

    # --- 2. GLOBAL INTER-ROUTE OPTIMIZATION (MULTI-START) ---
    MAX_RETRIES = 50
    best_optimization_result = None
    best_unassigned = float('inf')
    best_penalty = float('inf')
    best_objective = float('inf')
    
    assigned_routes_backup = deepcopy(initial_routes) # Backup for retries

    for attempt in range(1, MAX_RETRIES + 1):
        current_seed = base_seed + attempt - 1
        print(f"\n[RVND] Optimization Attempt {attempt}/{MAX_RETRIES} (Seed {current_seed})")
        
        # Reset routes for this attempt
        current_routes_input = deepcopy(assigned_routes_backup)
        
        # Create a new RNG for this attempt
        current_rng = random.Random(current_seed)

        # Run Optimization
        optimized_routes, inter_logs = rvnd_inter(
            current_routes_input, 
            instance, 
            distance_data, 
            instance["fleet"], # Pass the list of fleet objects
            current_rng,
            MAX_INTER_ITERATIONS
        )
        
        # Evaluate Result Feasibility
        final_demands = [r["total_demand"] for r in optimized_routes]
        _, unassigned, penalty = can_assign_fleet(final_demands, instance["fleet"])
        
        # Calculate Objective for comparison
        total_obj = sum(r["objective"] for r in optimized_routes)
        
        print(f"  Result: Unassigned={unassigned}, Penalty={penalty:.2f}, Obj={total_obj:.2f}")

        # Check if this is the best result so far
        is_better = False
        if unassigned < best_unassigned:
            is_better = True
        elif unassigned == best_unassigned:
            if penalty < best_penalty - 1e-4: # Use a small epsilon for float comparison
                is_better = True
            elif abs(penalty - best_penalty) < 1e-4 and total_obj < best_objective - 1e-4:
                is_better = True
        
        if is_better:
            print("  -> New Best Result Found!")
            best_unassigned = unassigned
            best_penalty = penalty
            best_objective = total_obj
            best_optimization_result = (optimized_routes, inter_logs)
            
        # If perfect result, stop early
        # Note: penalty includes load_balance_score, so it's > 0. We check total_excess.
        if unassigned == 0 and abs(penalty - (sum(d**2 for d in final_demands)*0.0001)) < 1e-4:
            print("  -> Feasible solution found. Stopping retries.")
            break
            
    # Use best result found
    if best_optimization_result is None:
        # Fallback if no optimization result was ever recorded (e.g., MAX_RETRIES=0 or an error)
        # This should ideally not happen if MAX_RETRIES > 0
        print("[RVND] Warning: No best optimization result found. Using last attempt's result.")
        # In a real scenario, you might want to raise an error or handle this more robustly
        # For now, we'll just use the result from the last attempt, which might not be "best"
        # or could be undefined if the loop didn't run.
        # To be safe, we can re-run the first attempt if best_optimization_result is still None.
        current_rng = random.Random(base_seed) # Reset to dynamic base seed
        optimized_routes, inter_logs = rvnd_inter(
            deepcopy(assigned_routes_backup), 
            instance, 
            distance_data, 
            instance["fleet"], 
            current_rng, 
            MAX_INTER_ITERATIONS
        )
    else:
        optimized_routes, inter_logs = best_optimization_result

    # --- 3. INTRA-ROUTE OPTIMIZATION ---
    print("\n[RVND] Starting Intra-Route Optimization on Best Result...")
    results = []
    
    # Filter logs to keep only the best run's logs (or maybe all? let's keep best for clarity)
    all_iteration_logs = inter_logs 

    for route in optimized_routes:
        # Need to use the original rng for intra-route, or re-seed it
        # For consistency, let's use the original rng (seed 84) for intra-route
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
    # IMPORTANT: Sort results by demand DESCENDING before assigning vehicles.
    # This ensures consistency with the Global Feasibility logic (can_assign_fleet).
    results.sort(key=lambda x: x["improved"]["total_demand"], reverse=True)
    
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
        
        status = "Assigned" if new_vehicle_type is not None else "Gagal"
        all_iteration_logs.append({
            "phase": "VEHICLE_REASSIGN",
            "cluster_id": route_result["cluster_id"],
            "status": status,
            "new_vehicle": new_vehicle_type or "-"
        })

        reassigned_results.append({
            "cluster_id": route_result["cluster_id"],
            "vehicle_type": new_vehicle_type,
            "improved": improved
        })

    summary["distance_after"] = sum(r["improved"]["total_distance"] for r in reassigned_results)
    summary["objective_after"] = sum(r["improved"]["objective"] for r in reassigned_results)
    summary["tw_after"] = sum(r["improved"]["total_tw_violation"] for r in reassigned_results)
    summary["capacity_violations_after"] = sum(1 for r in reassigned_results if r["improved"]["capacity_violation"] > 0)

    return {
        "routes": reassigned_results,
        "summary": summary,
        "parameters": {
            "inter_neighborhoods": INTER_ROUTE_NEIGHBORHOODS,
            "intra_neighborhoods": INTRA_ROUTE_NEIGHBORHOODS,
            "max_inter_iterations": MAX_INTER_ITERATIONS,
            "max_intra_iterations": MAX_INTRA_ITERATIONS,
            "seed": base_seed
        },
        "vehicle_usage": used_vehicles,
        "iteration_logs": all_iteration_logs
    }
