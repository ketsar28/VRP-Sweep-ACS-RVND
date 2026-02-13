from typing import List, Dict, Tuple
import copy

def can_assign_fleet(demands: List[float], fleet_data: List[Dict]) -> Tuple[bool, int, float]:
    """
    Greedy check: can we assign current route demands to available fleet?
    """
    # 1. Expand fleet into individual units
    available_units = []
    for f in fleet_data:
        for _ in range(f["units"]):
            available_units.append({
                "type": f["id"],
                "capacity": f["capacity"],
                "used": False
            })
    
    # 2. Sort demands DESC (hardest to fit first)
    # We need to keep track of original indices if we want to know WHICH failed, 
    # but for boolean check, just values are enough.
    sorted_demands = sorted(demands, reverse=True)
    
    # 3. Sort fleet by capacity DESC
    available_units.sort(key=lambda x: x["capacity"], reverse=True)
    
    unassigned_count = 0
    penalty_magnitude = 0.0
    
    assignments = []

    for d in sorted_demands:
        assigned = False
        for unit in available_units:
            if not unit["used"] and unit["capacity"] >= d:
                unit["used"] = True
                assigned = True
                assignments.append((d, unit["type"], unit["capacity"]))
                break
        
        if not assigned:
            unassigned_count += 1
            penalty_magnitude += d
            assignments.append((d, "UNASSIGNED", 0))

    print(f"Assignments: {assignments}")
    return (unassigned_count == 0, unassigned_count, penalty_magnitude)

# Define Problem State (Based on User Screenshot)
# Fleets
fleet_data = [
    {"id": "Fleet A", "capacity": 60, "units": 2},
    {"id": "Fleet B", "capacity": 100, "units": 2},
    {"id": "Fleet C", "capacity": 150, "units": 1}
]

# Current Demands (Approximate from Screenshot)
# Cluster 1: 18.0
# Cluster 2: 149.0
# Cluster 3: 51.0
# Cluster 4: 130.0
current_demands = [18.0, 149.0, 51.0, 130.0]

print("--- Initial State ---")
_, unassigned, penalty = can_assign_fleet(current_demands, fleet_data)
print(f"Unassigned: {unassigned}, Penalty: {penalty}")

# Theoretical improved state (Shift 40kg from C4 to C1)
# C1: 18 + 40 = 58 (Fits A)
# C4: 130 - 40 = 90 (Fits B)
# C2: 149 (Fits C)
# C3: 51 (Fits A)
improved_demands = [58.0, 149.0, 51.0, 90.0]

print("\n--- Improved State (Simulated Shift) ---")
_, unassigned_improved, penalty_improved = can_assign_fleet(improved_demands, fleet_data)
print(f"Unassigned: {unassigned_improved}, Penalty: {penalty_improved}")
