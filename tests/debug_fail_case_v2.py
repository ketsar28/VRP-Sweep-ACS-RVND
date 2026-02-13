import random
from typing import List, Dict, Tuple
from copy import deepcopy

# Mock Customer Data
customers = {
    1: {"id": 1, "demand": 18.0},
    2: {"id": 2, "demand": 149.0},
    3: {"id": 3, "demand": 51.0},
    4: {"id": 4, "demand": 40.0},
    8: {"id": 8, "demand": 47.0},
    10: {"id": 10, "demand": 43.0}
}
instance = {"customers": [v for k,v in customers.items()]}

# Initial Routes (Problematic State)
# R4 is 130.0 (Overloaded for Fleet B, Fleet C taken by R2)
routes = [
    {"id": 1, "sequence": [0, 1, 0], "total_demand": 18.0, "total_distance": 10},
    {"id": 2, "sequence": [0, 2, 0], "total_demand": 149.0, "total_distance": 20},
    {"id": 3, "sequence": [0, 3, 0], "total_demand": 51.0, "total_distance": 10},
    {"id": 4, "sequence": [0, 10, 8, 4, 0], "total_demand": 130.0, "total_distance": 50}
]

fleet_list = [
    {"id": "Fleet A", "capacity": 60, "units": 2},
    {"id": "Fleet B", "capacity": 100, "units": 2},
    {"id": "Fleet C", "capacity": 150, "units": 1}
]

# Mock Distance Matrix (All 1.0)
distance_matrix = [[1.0]*15 for _ in range(15)]

# --- REPLICATE CODE FROM RVND.PY ---
def can_assign_fleet(demands: List[float], fleet_data: List[Dict]) -> Tuple[bool, int, float]:
    """
    Greedy check: can we assign current route demands to available fleet?
    """
    available_units = []
    for f in fleet_data:
        for _ in range(f["units"]):
            available_units.append({
                "type": f["id"],
                "capacity": f["capacity"],
                "used": False
            })
    
    sorted_demands = sorted(demands, reverse=True)
    available_units.sort(key=lambda x: x["capacity"], reverse=True)
    
    unassigned_count = 0
    penalty_magnitude = 0.0
    
    for d in sorted_demands:
        assigned = False
        for unit in available_units:
            if not unit["used"] and unit["capacity"] >= d:
                unit["used"] = True
                assigned = True
                break
        
        if not assigned:
            unassigned_count += 1
            penalty_magnitude += d

    return (unassigned_count == 0, unassigned_count, penalty_magnitude)

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
        val = 0
        for c in seq[1:-1]:
            if c in customers:
                val += customers[c]["demand"]
            elif c == 2: # Mock for R2
                val += 149.0
        return val

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

print("--- TESTING REBALANCE ---")
rng = random.Random(42)
new_routes, success, info = attempt_load_rebalance(routes, instance, distance_matrix, fleet_list, rng)

if success:
    print(f"SUCCESS: {info}")
    for r in new_routes:
        print(f"R{r['id']}: Dem={r['total_demand']}, Seq={r['sequence']}")
else:
    print("FAILURE: No rebalance found.")
