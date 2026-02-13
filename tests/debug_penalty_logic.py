from typing import List, Dict, Tuple

def can_assign_fleet_legacy(demands: List[float], fleet_data: List[Dict]) -> Tuple[bool, int, float]:
    available_units = []
    for f in fleet_data:
        for _ in range(f["units"]):
            available_units.append({"capacity": f["capacity"], "used": False})
    
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
            penalty_magnitude += d # LEGACY: Add full demand
            
    return (unassigned_count == 0, unassigned_count, penalty_magnitude)

def can_assign_fleet_improved(demands: List[float], fleet_data: List[Dict]) -> Tuple[bool, int, float]:
    """
    Improved: Calculates 'Excess Penalty' - how much demand EXCEEDS the best available vehicle.
    """
    available_units = []
    for f in fleet_data:
        for _ in range(f["units"]):
            available_units.append({"capacity": f["capacity"], "used": False})
    
    # Sort demands DESC
    sorted_demands = sorted(demands, reverse=True)
    # Sort fleets DESC
    available_units.sort(key=lambda x: x["capacity"], reverse=True)
    
    unassigned_count = 0
    penalty_magnitude = 0.0
    
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
            # If not assigned, find the LARGEST unused vehicle to calculate "Excess"
            # If no vehicles left, excess is full demand.
            best_unused = -1
            best_unit_idx = -1
            
            for idx, unit in enumerate(available_units):
                if not unit["used"]:
                    # Since units are sorted DESC, the first unused is the largest available
                    best_unused = unit["capacity"]
                    best_unit_idx = idx
                    break
            
            if best_unused != -1:
                # We "force assign" to this smaller vehicle and count the overflow
                excess = d - best_unused
                penalty_magnitude += excess
                # Mark as used so other unassigned don't count against same vehicle?
                # Actually yes, otherwise we double count capacity.
                available_units[best_unit_idx]["used"] = True 
            else:
                # No vehicles left at all
                penalty_magnitude += d

    return (unassigned_count == 0, unassigned_count, penalty_magnitude)

# Test Case
# Fleets: C(150), B(100), B(100), A(60), A(60)
fleet_data = [
    {"id": "Fleet A", "capacity": 60, "units": 2},
    {"id": "Fleet B", "capacity": 100, "units": 2},
    {"id": "Fleet C", "capacity": 150, "units": 1}
]

# Scenario:
# Cluster 3: 131 (Takes C)
# Cluster 4: 148 (Needs C, but taken -> Fails. Remaining best is B=100)
# Others: 51, 18 (Fit A/B)
demands_initial = [148.0, 131.0, 51.0, 18.0]

print("--- INITIAL STATE [148, 131, 51, 18] ---")
_, u_leg, p_leg = can_assign_fleet_legacy(demands_initial, fleet_data)
_, u_imp, p_imp = can_assign_fleet_improved(demands_initial, fleet_data)
print(f"Legacy: Unassigned={u_leg}, Penalty={p_leg}")
print(f"Improved: Unassigned={u_imp}, Penalty={p_imp}")

# Move 10kg from 148 -> 51.
# New: 138, 131, 61, 18
demands_step1 = [138.0, 131.0, 61.0, 18.0]
# 138 takes C. 131 fails against B(100). Excess = 31.
# 61 fits B. 18 fits A.
print("\n--- STEP 1: Move 10kg (148->138, 51->61) ---")
_, u_leg, p_leg = can_assign_fleet_legacy(demands_step1, fleet_data)
_, u_imp, p_imp = can_assign_fleet_improved(demands_step1, fleet_data)
print(f"Legacy: Unassigned={u_leg}, Penalty={p_leg} (CHANGE?)")
print(f"Improved: Unassigned={u_imp}, Penalty={p_imp} (CHANGE?)")

# Move another 10kg from 138 -> 61.
# New: 128, 131, 71, 18
demands_step2 = [128.0, 131.0, 71.0, 18.0]
# 131 takes C. 128 fails against B(100). Excess = 28.
# 71 fits B. 18 fits A.
print("\n--- STEP 2: Move 10kg (138->128, 61->71) ---")
_, u_leg, p_leg = can_assign_fleet_legacy(demands_step2, fleet_data)
_, u_imp, p_imp = can_assign_fleet_improved(demands_step2, fleet_data)
print(f"Legacy: Unassigned={u_leg}, Penalty={p_leg} (CHANGE?)")
print(f"Improved: Unassigned={u_imp}, Penalty={p_imp} (CHANGE?)")
