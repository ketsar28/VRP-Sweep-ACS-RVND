import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
INSTANCE_PATH = DATA_DIR / "parsed_instance.json"
DISTANCE_PATH = DATA_DIR / "parsed_distance.json"
ACS_PATH = DATA_DIR / "acs_routes.json"
RVND_PATH = DATA_DIR / "rvnd_routes.json"

# RVND Configuration
MAX_INTER_ITERATIONS = 150
MAX_INTRA_ITERATIONS = 200

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


def can_assign_fleet(demands: List[float], fleet_data: List[Dict]) -> Tuple[bool, int, float]:
    """
    Checks if demands can be assigned to fleets.
    Returns: (feasible, unassigned_count, penalty_magnitude)
    
    Penalty now includes:
    1. Excess Demand (how much unassigned routes exceed available capacity)
    2. Load Balancing Tie-Breaker (sum of squares) to encourage reducing max demands
    """
    # 1. Expand fleet (greedy approach matches UI logic)
    available_units = []
    for f in fleet_data:
        for _ in range(f["units"]):
            available_units.append({
                "id": f["id"],
                "capacity": f["capacity"],
                "used": False
            })
    
    # Sort fleets ASC by capacity (Smallest First -> Best Fit)
    # This prevents using a large vehicle for a small demand when a smaller vehicle is available.
    available_units.sort(key=lambda x: x["capacity"], reverse=False)
    
    # Sort demands DESC (hardest to fit first)
    sorted_demands = sorted(demands, reverse=True)
    
    unassigned_count = 0
    total_excess = 0.0
    
    for d in sorted_demands:
        assigned = False
        # Try to fit in best available
        for unit in available_units:
            if not unit["used"] and unit["capacity"] >= d:
                unit["used"] = True
                assigned = True
                break
        
        if not assigned:
            unassigned_count += 1
            # Calculate Excess: Distance to the LARGEST unused vehicle
            # This provides a better gradient for moves.
            best_unused_capacity = 0.0
            best_unit_idx = -1
            
            # Find Largest Unused (available_units is sorted ASC, so search from end)
            for i in range(len(available_units)-1, -1, -1):
                unit = available_units[i]
                if not unit["used"]:
                    best_unused_capacity = unit["capacity"]
                    best_unit_idx = i
                    break
            
            if best_unused_capacity > 0:
                total_excess += max(0.0, d - best_unused_capacity)
                available_units[best_unit_idx]["used"] = True
            else:
                total_excess += d

    # Add Load Balancing Score (Sum of Squares) scaled down
    # This ensures that [148, 131] (SqSum=39065) is worse than [138, 131] (SqSum=36205)
    # even if Excess is constant.
    # Scale factor: 0.0001 (so 100kg reduction ~ 1.0 penalty point)
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
    
    # 1. Expand fleet for greedy check (matches can_assign_fleet logic)
    available_units = []
    for f in fleet_data:
        for _ in range(f["units"]):
            available_units.append({
                "id": f["id"],
                "capacity": f["capacity"],
                "used": False
            })
    available_units.sort(key=lambda x: x["capacity"]) # Best Fit
    
    # Sort demands for assignment logic
    indexed_demands = sorted(enumerate(demands), key=lambda x: x[1], reverse=True)
    
    assignments = {} # Index -> Fleet ID
    
    for idx, d in indexed_demands:
        assigned_fid = "Ghost"
        for unit in available_units:
            if not unit["used"] and unit["capacity"] >= d:
                unit["used"] = True
                assigned_fid = unit["id"]
                break
        assignments[idx] = assigned_fid
        
    for i in range(len(routes)):
        route_sequences.append("→".join(map(str, routes[i]["sequence"])))
        
        # Load string: "Demand kg (Vehicle ID)"
        fid = assignments.get(i, "Ghost")
        route_loads.append(f"{demands[i]:.1f} kg ({fid})")
        
    return {
        "detail": "Accepted Move" if routes else "Stagnan",
        "route_sequences": route_sequences,
        "route_loads": route_loads,
        "feasible": all(fid != "Ghost" for fid in assignments.values()),
        "reason": "Layak" if all(fid != "Ghost" for fid in assignments.values()) else "Kapasitas Terlampaui",
        "delta": round(delta, 2)
    }


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
    wait_sum = 0.0
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
            wait_sum += wait

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
        "total_wait_time": wait_sum,
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
    _, current_unassigned, current_penalty = can_assign_fleet(current_demands, fleet_list)
    best_unassigned = current_unassigned
    best_penalty = current_penalty

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
        nonlocal best_distance, best_routes, best_move, best_unassigned, best_penalty
        
        dist_a = calc_route_distance(new_seq_a)
        dist_b = calc_route_distance(new_seq_b)
        demand_a = calc_route_demand(new_seq_a, customers)
        demand_b = calc_route_demand(new_seq_b, customers)
        
        total_new_dist = current_distance - routes[i]["total_distance"] - routes[j]["total_distance"] + dist_a + dist_b
        
        # Check global fleet feasibility
        new_demands = current_demands[:]
        new_demands[i] = demand_a
        new_demands[j] = demand_b
        _, unassigned, penalty = can_assign_fleet(new_demands, fleet_list)
        
        accepted = False
        
        # Acceptance Logic: Priority to Feasibility Gradient, then Distance
        if unassigned < best_unassigned:
            # Case 1: Reduced number of unassigned clusters
            accepted = True
        elif unassigned == best_unassigned:
            if unassigned > 0:
                # Still failing, check if we reduced the magnitude of overload
                if penalty < best_penalty - 1e-4:
                    accepted = True
                elif abs(penalty - best_penalty) < 1e-4 and total_new_dist < best_distance - 1e-4:
                    # Same magnitude, check distance
                    accepted = True
            else:
                # Fully feasible: standard distance comparison
                if total_new_dist < best_distance - 1e-4:
                    accepted = True
        
        if accepted:
            best_unassigned = unassigned
            best_penalty = penalty
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
            # 2026-02-13 FIX: Allow moving the LAST customer out (emptying the route)
            # Route with 1 customer has length 3: [0, C, 0].
            # So len <= 2 is empty. We allow len >= 3.
            if len(route_a["sequence"]) < 3:
                continue
            for j, route_b in enumerate(routes):
                if i == j:
                    continue
                seq_a = route_a["sequence"][1:-1]
                seq_b = route_b["sequence"][1:-1]
                for ca in seq_a:
                    new_seq_a_inner = [c for c in seq_a if c != ca]
                    # If empty, new_seq_a is [0, 0]
                    new_seq_a = [0] + new_seq_a_inner + [0]
                    for pos in range(len(seq_b) + 1):
                        new_seq_b = [0] + seq_b[:pos] + [ca] + seq_b[pos:] + [0]
                        if evaluate_move(i, j, new_seq_a, new_seq_b):
                            best_move = {"type": "shift_1_0", "detail": f"{ca} -> R{j+1}"}

    elif neighborhood == "shift_2_0":
        for i, route_a in enumerate(routes):
            # Allow emptying route with 2 customers (len 4: [0, C1, C2, 0])
            if len(route_a["sequence"]) < 4:
                continue
            for j, route_b in enumerate(routes):
                if i == j:
                    continue
                seq_a = route_a["sequence"][1:-1]
                seq_b = route_b["sequence"][1:-1]
                for idx_a in range(len(seq_a) - 1):
                    ca1, ca2 = seq_a[idx_a], seq_a[idx_a+1]
                    new_seq_a_inner = seq_a[:idx_a] + seq_a[idx_a+2:]
                    new_seq_a = [0] + new_seq_a_inner + [0]
                    for pos in range(len(seq_b) + 1):
                        new_seq_b = [0] + seq_b[:pos] + [ca1, ca2] + seq_b[pos:] + [0]
                        if evaluate_move(i, j, new_seq_a, new_seq_b):
                            best_move = {"type": "shift_2_0", "detail": f"({ca1},{ca2}) -> R{j+1}"}

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
            "unassigned_after": best_unassigned,
            "penalty_magnitude": best_penalty
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
    _, best_unassigned, best_penalty = can_assign_fleet(current_demands, fleet_list)
    
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
            new_penalty = result["penalty_magnitude"]
            new_dist = result["distance_after"]
            
            # Acceptance logic: same as in evaluate_move
            if new_unassigned < best_unassigned:
                accepted = True
            elif new_unassigned == best_unassigned:
                if new_unassigned > 0:
                    if new_penalty < best_penalty - 1e-4:
                        accepted = True
                    elif abs(new_penalty - best_penalty) < 1e-4 and new_dist < current_distance - 1e-4:
                        accepted = True
                else:
                    if new_dist < current_distance - 1e-4:
                        accepted = True
            
        if accepted:
            old_dist = current_distance
            routes = result["new_routes"]
            current_distance = result["distance_after"]
            best_unassigned = result["unassigned_after"]
            best_penalty = result["penalty_magnitude"]
            NL = NL_FULL[:] # Reset NL
            
            # Generate capacity validation details for logging
            logs_demands = [r["total_demand"] for r in routes]
            delta_val = result["distance_after"] - old_dist
            current_candidate = generate_capacity_log_candidate(routes, logs_demands, instance, fleet_list, delta=delta_val)
            
            iteration_logs.append({
                "iteration_id": iteration,
                "phase": "RVND-INTER",
                "neighborhood": neighborhood,
                "improved": True,
                "total_distance": round(current_distance, 2),
                "unassigned": best_unassigned,
                "penalty": round(best_penalty, 2),
                "move": result.get("move_details"),
                "candidates": [current_candidate] # Show the accepted move as the candidate
            })
            continue

        if neighborhood in NL:
            NL.remove(neighborhood)
            
            # If ALL neighborhoods are exhausted, then we are truly stagnant (local optimum)
            if not NL:
                logs_demands = [r["total_demand"] for r in routes]
                current_candidate = generate_capacity_log_candidate(routes, logs_demands, instance, fleet_list, delta=0.0)
                current_candidate["detail"] = "Local Optimum Reached"
                
                iteration_logs.append({
                    "iteration_id": iteration,
                    "phase": "RVND-INTER",
                    "type": "stagnant",
                    "neighborhood": "All",
                    "message": f"Iterasi Berakhir (Konvergen). Pada iterasi {iteration}, algoritma telah mencapai titik optimal lokal (tidak ditemukan rute yang lebih baik lagi).",
                    "candidates": [current_candidate]
                })

            # --- EMERGENCY REBALANCE CHECK ---
            if best_unassigned > 0:
                rebal_routes, rebal_success, rebal_info = attempt_load_rebalance(
                    routes, instance, distance_matrix, fleet_list, rng
                )
                if rebal_success:
                    routes = rebal_routes
                    # Recalculate metrics
                    current_distance = sum(r["total_distance"] for r in routes)
                    current_demands = [r["total_demand"] for r in routes]
                    _, best_unassigned, best_penalty = can_assign_fleet(current_demands, fleet_list)
                    
                    # Log rebalance as a candidate
                    current_candidate = generate_capacity_log_candidate(routes, current_demands, instance, fleet_list)
                    current_candidate["detail"] = rebal_info.get("detail", "Force Shift")
                    
                    iteration_logs.append({
                        "iteration_id": iteration,
                        "phase": "RVND-INTER-REBALANCE",
                        "neighborhood": "force_shift",
                        "improved": True,
                        "total_distance": round(current_distance, 2),
                        "unassigned": best_unassigned,
                        "penalty": round(best_penalty, 2),
                        "move": rebal_info.get("detail"),
                        "candidates": [current_candidate]
                    })
                    NL = NL_FULL[:] # Reset NL

    return routes, iteration_logs


def attempt_load_rebalance(
    routes: List[Dict],
    instance: dict,
    distance_matrix: List[List[float]],
    fleet_list: List[Dict],
    rng: random.Random
) -> Tuple[List[Dict], bool, Dict]:
    """
    Emergency rebalancing: Force moves from overloaded clusters to underloaded ones.
    """
    current_demands = [r["total_demand"] for r in routes]
    _, unassigned, penalty = can_assign_fleet(current_demands, fleet_list)
    
    if unassigned == 0:
        return routes, False, {}

    # Identify "problem" routes (heuristic: likely the largest ones)
    sorted_indices = sorted(range(len(current_demands)), key=lambda k: current_demands[k], reverse=True)
    
    best_routes = None
    best_unassigned = unassigned
    best_penalty = penalty
    success = False
    move_detail = ""

    customers = {c["id"]: c for c in instance["customers"]}

    def calc_route_distance(seq):
        return sum(distance_matrix[seq[k]][seq[k+1]] for k in range(len(seq)-1))

    def calc_route_demand(seq):
        return sum(customers[c]["demand"] for c in seq[1:-1])

    # Try shifting nodes from HEAVIEST routes to OTHERS
    for i in sorted_indices[:2]: # Only try moving from top 2 heaviest
        route_source = routes[i]
        seq_source = route_source["sequence"][1:-1]
        
        if not seq_source: continue

        for j in range(len(routes)):
            if i == j: continue
            
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
                    dist_a = calc_route_distance(new_seq_a)
                    dist_b = calc_route_distance(new_seq_b)
                    
                    best_routes[i]["sequence"] = new_seq_a
                    best_routes[i]["total_distance"] = dist_a
                    best_routes[i]["total_demand"] = dem_a
                    
                    best_routes[j]["sequence"] = new_seq_b
                    best_routes[j]["total_distance"] = dist_b
                    best_routes[j]["total_demand"] = dem_b
                    
                    success = True
                    move_detail = f"Force Shift {node} from R{i+1} to R{j+1}"
                    return best_routes, True, {"detail": move_detail, "unassigned": u_new, "penalty": p_new}

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
                # We keep the original vehicle type but it's effectively "Ghost".
                # RVND will likely fail to solve this unless it merges routes.
                route["vehicle_assignment_note"] = "Ghost (No Vehicles Left)"
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

    output = {
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
