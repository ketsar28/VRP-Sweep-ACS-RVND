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
    
    REVISION NOTE: Vehicle assignment rules per specification:
    - A: demand ≤ 60
    - B: 60 < demand ≤ 100  
    - C: 100 < demand ≤ 150
    
    Choose smallest feasible vehicle that:
    1. Can handle the demand (capacity)
    2. Has available units (stock not exceeded)
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

def apply_inter_neighborhood(neighborhood: str, routes: List[Dict], rng: random.Random) -> Optional[List[Dict]]:
    """Apply inter-route neighborhood (placeholder for single-route case)."""
    # In single-route case, inter-route operations don't apply
    # Return None to indicate no move available
    return None


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
    fleet_data: dict,
    rng: random.Random,
    max_iterations: int
) -> List[Dict]:
    """
    Inter-route RVND with strict neighborhood list management.
    
    For single-route case, this is a placeholder.
    In multi-route scenarios, this would apply shift/swap/cross operators.
    """
    # Placeholder for single-route case
    # In actual multi-route implementation, this would manage inter-route moves
    return routes


# ========== MAIN RVND CONTROLLER ==========

def rvnd_route(route: Dict, instance: Dict, distance_data: Dict, rng: random.Random) -> Dict:
    """
    Two-level RVND: Inter-route → Intra-route with strict iteration control.
    
    For single-route case:
    - Inter-route phase is skipped (no other routes to interact with)
    - Intra-route phase performs all local search
    
    Rules:
    - Iteration counters never reset
    - Each level has independent NL management
    - Only feasible improvements accepted
    """
    sequence = deepcopy(route["sequence"])
    vehicle_type = route["vehicle_type"]
    
    # Get fleet information
    fleet_data = {fleet["id"]: fleet for fleet in instance["fleet"]}
    fleet_info = fleet_data[vehicle_type]
    
    # Baseline evaluation
    baseline = evaluate_route(sequence, instance, distance_data, fleet_info)
    
    # For single-route case, we only apply intra-route RVND
    # (Inter-route requires multiple routes)
    
    iter_inter = 0
    iter_intra = 0
    
    # INTER-ROUTE PHASE (skipped for single-route)
    # In multi-route scenario, this would run with MAX_INTER_ITERATIONS
    
    # INTRA-ROUTE PHASE
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

    results = []
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

    for route in acs_data["clusters"]:
        fleet_info = fleet_data[route["vehicle_type"]]
        baseline = evaluate_route(route["sequence"], instance, distance_data, fleet_info)
        improved = rvnd_route(route, instance, distance_data, rng)

        summary["distance_before"] += baseline["total_distance"]
        summary["distance_after"] += improved["total_distance"]
        summary["objective_before"] += baseline["objective"]
        summary["objective_after"] += improved["objective"]
        summary["tw_before"] += baseline["total_tw_violation"]
        summary["tw_after"] += improved["total_tw_violation"]
        summary["capacity_violations_before"] += (1 if baseline["capacity_violation"] > 0 else 0)
        summary["capacity_violations_after"] += (1 if improved["capacity_violation"] > 0 else 0)

        results.append({
            "cluster_id": route["cluster_id"],
            "vehicle_type": route["vehicle_type"],
            "baseline": baseline,
            "improved": improved
        })

    # REVISION NOTE: After RVND, reassign vehicles based on new demand distribution
    # This ensures smallest feasible vehicle is used and stock limits are respected
    used_vehicles = {}
    reassigned_results = []
    all_iteration_logs = []
    
    for route_result in results:
        improved = route_result["improved"]
        total_demand = improved["total_demand"]
        
        # Reassign vehicle based on current demand
        new_vehicle_type = assign_vehicle_by_demand(total_demand, instance["fleet"], used_vehicles)
        
        if new_vehicle_type is None:
            # Stock exceeded - keep original assignment (mark as infeasible)
            new_vehicle_type = route_result["vehicle_type"]
            improved["vehicle_assignment_failed"] = True
        else:
            used_vehicles[new_vehicle_type] = used_vehicles.get(new_vehicle_type, 0) + 1
            improved["vehicle_assignment_failed"] = False
        
        # Re-evaluate with correct vehicle type if changed
        if new_vehicle_type != route_result["vehicle_type"]:
            fleet_info = fleet_data[new_vehicle_type]
            improved = evaluate_route(improved["sequence"], instance, distance_data, fleet_info)
            improved["vehicle_assignment_failed"] = False
        
        # Collect iteration logs
        if "iteration_logs" in improved:
            for log in improved["iteration_logs"]:
                log["cluster_id"] = route_result["cluster_id"]
                log["vehicle_type"] = new_vehicle_type
                all_iteration_logs.append(log)
        
        reassigned_results.append({
            "cluster_id": route_result["cluster_id"],
            "vehicle_type": new_vehicle_type,
            "baseline": route_result["baseline"],
            "improved": improved
        })

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
        "RVND Results:\n"
        f"  Distance: {round(summary['distance_before'], 3)} -> {round(summary['distance_after'], 3)}\n"
        f"  Objective: {round(summary['objective_before'], 3)} -> {round(summary['objective_after'], 3)}\n"
        f"  TW Violations: {round(summary['tw_before'], 3)} -> {round(summary['tw_after'], 3)}\n"
        f"  Capacity Violations: {summary['capacity_violations_before']} -> {summary['capacity_violations_after']}"
    )


if __name__ == "__main__":
    main()
