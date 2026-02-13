import sys
import os

# Add Program to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Program.rvnd import can_assign_fleet

# Test Case
fleet_data = [
    {"id": "Fleet A", "capacity": 60, "units": 2},
    {"id": "Fleet B", "capacity": 100, "units": 2},
    {"id": "Fleet C", "capacity": 150, "units": 1}
]

# Scenario: 148, 131. Only C(150), B(100) available for them.
demands_initial = [148.0, 131.0, 51.0, 18.0]
_, u1, p1 = can_assign_fleet(demands_initial, fleet_data)
print(f"State 1 [148, 131]: Unassigned={u1}, Penalty={p1:.4f}")

# Move 10kg from 148 -> 51 (New: 138, 131, 61)
demands_step1 = [138.0, 131.0, 61.0, 18.0]
_, u2, p2 = can_assign_fleet(demands_step1, fleet_data)
print(f"State 2 [138, 131]: Unassigned={u2}, Penalty={p2:.4f}")

if p2 < p1:
    print("SUCCESS: Penalty decreased (Gradient exists!)")
else:
    print("FAILURE: Penalty did not decrease.")

# Move more: 148->99. (New: 99, 131, 100, 18)
demands_feasible = [99.0, 131.0, 100.0, 18.0]
_, u3, p3 = can_assign_fleet(demands_feasible, fleet_data)
print(f"State 3 [99, 131]: Unassigned={u3}, Penalty={p3:.4f}")

if u3 == 0:
    print("SUCCESS: Feasible state reached.")
else:
    print(f"FAILURE: Should be feasible but U={u3}")
