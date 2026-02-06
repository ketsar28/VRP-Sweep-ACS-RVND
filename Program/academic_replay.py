"""
Academic Replay Mode - "Hitung Manual MFVRPTE RVND"

This module reproduces EXACTLY the computations from the Word document.
This module reproduces computations using Dynamic ACS logic to simulate manual calculation variability.
Optimization logic is now probabilistic matching standard ACS.

MODE: ACADEMIC_REPLAY (default for validation)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import random

# ============================================================
# CONFIGURATION
# ============================================================

MODE = "ACADEMIC_REPLAY"  # default for validation
# MODE = "OPTIMIZATION"  # for normal operation

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
ACADEMIC_OUTPUT_PATH = DATA_DIR / "academic_replay_results.json"

# ============================================================
# HARD-CODED DATASET FROM WORD DOCUMENT
# ============================================================

ACADEMIC_DATASET = {
    "depot": {
        "id": 0,
        "name": "Depot",
        "x": 0.0,
        "y": 0.0,
        "time_window": {"start": "08:00", "end": "17:00"},
        "service_time": 0
    },
    "customers": [
        {"id": 1, "name": "C1", "x": 2.0, "y": 3.0, "demand": 10,
            "service_time": 5, "time_window": {"start": "08:00", "end": "12:00"}},
        {"id": 2, "name": "C2", "x": 5.0, "y": 1.0, "demand": 15,
            "service_time": 10, "time_window": {"start": "08:00", "end": "14:00"}},
        {"id": 3, "name": "C3", "x": 6.0, "y": 4.0, "demand": 20,
            "service_time": 8, "time_window": {"start": "09:00", "end": "15:00"}},
        {"id": 4, "name": "C4", "x": 8.0, "y": 2.0, "demand": 25,
            "service_time": 12, "time_window": {"start": "08:30", "end": "16:00"}},
        {"id": 5, "name": "C5", "x": 3.0, "y": 6.0, "demand": 30,
            "service_time": 15, "time_window": {"start": "10:00", "end": "14:00"}},
        {"id": 6, "name": "C6", "x": 7.0, "y": 5.0, "demand": 18,
            "service_time": 7, "time_window": {"start": "08:00", "end": "13:00"}},
        {"id": 7, "name": "C7", "x": 4.0, "y": 8.0, "demand": 22,
            "service_time": 9, "time_window": {"start": "09:00", "end": "16:00"}},
        {"id": 8, "name": "C8", "x": 1.0, "y": 5.0, "demand": 12,
            "service_time": 6, "time_window": {"start": "08:00", "end": "11:00"}},
        {"id": 9, "name": "C9", "x": 9.0, "y": 7.0, "demand": 28,
            "service_time": 11, "time_window": {"start": "10:00", "end": "15:00"}},
        {"id": 10, "name": "C10", "x": 5.0, "y": 9.0, "demand": 35,
            "service_time": 14, "time_window": {"start": "09:00", "end": "17:00"}}
    ],
    "fleet": [
        {"id": "A", "capacity": 60, "units": 2, "fixed_cost": 50000,
            "variable_cost_per_km": 1000, "available_from": "08:00", "available_until": "17:00"},
        {"id": "B", "capacity": 100, "units": 2, "fixed_cost": 60000,
            "variable_cost_per_km": 1000, "available_from": "08:00", "available_until": "17:00"},
        {"id": "C", "capacity": 150, "units": 1, "fixed_cost": 70000,
            "variable_cost_per_km": 1000, "available_from": "08:00", "available_until": "17:00"}
    ],
    "acs_parameters": {
        "alpha": 0.5,
        "beta": 2,
        "rho": 0.2,
        "q0": 0.85,
        "num_ants": 2,
        "max_iterations": 2
    },
    "objective_weights": {
        "w1_distance": 1.0,
        "w2_time": 1.0,
        "w3_tw_violation": 1.0
    }
}

# ============================================================
# DEPRECATED: HARDCODED DATA FROM WORD DOCUMENT
# These are kept for reference only - actual algorithms now use
# DYNAMIC CALCULATION based on user input data.
# ============================================================

# NOTE: Polar angles are now calculated dynamically in academic_sweep()
# WORD_POLAR_ANGLES = {...}  # DEPRECATED

# NOTE: Customers are now sorted dynamically by polar angle
# WORD_SORTED_CUSTOMERS = [...]  # DEPRECATED

# NOTE: Clusters are now formed dynamically based on user's vehicle capacities
# WORD_CLUSTERS = [...]  # DEPRECATED

# NOTE: Random values for ACS are now generated dynamically
# WORD_RANDOM_VALUES = {...}  # DEPRECATED

# NOTE: Expected routes are calculated dynamically, not hardcoded
# WORD_EXPECTED_ROUTES = {...}  # DEPRECATED

# NOTE: RVND moves are now generated dynamically
# WORD_RVND_MOVES = [...]  # DEPRECATED

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def parse_time_to_minutes(value: str) -> float:
    """Convert HH:MM to minutes from midnight."""
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def minutes_to_clock(value: float) -> str:
    """Convert minutes to HH:MM format."""
    hours = int(value // 60)
    minutes = int(value % 60)
    return f"{hours:02d}:{minutes:02d}"


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute Euclidean distance."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def compute_service_time_from_sequence(sequence: List[int], dataset: Dict) -> float:
    """
    Compute total service time from a route sequence.
    Service time = sum of customer service times (depot has 0 service time).
    Service time must never be zero unless all customers have zero service time.
    """
    customers = {c["id"]: c for c in dataset["customers"]}
    total_service_time = 0.0

    for node_id in sequence:
        if node_id != 0 and node_id in customers:
            total_service_time += customers[node_id].get("service_time", 0)

    return total_service_time


def compute_polar_angle_degrees(customer: Dict, depot: Dict) -> float:
    """Compute polar angle in degrees (as in Word)."""
    angle_rad = math.atan2(
        customer["y"] - depot["y"], customer["x"] - depot["x"])
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return round(angle_deg, 2)


def build_distance_matrix(dataset: Dict, multiplier: float = 1.0) -> List[List[float]]:
    """Build distance matrix including depot (node 0).

    Matrix structure:
    - Index 0 = Depot
    - Index 1 = customers[0] (first customer in dataset["customers"])
    - Index 2 = customers[1] (second customer in dataset["customers"])
    - etc.

    IMPORTANT: Matrix indices are based on POSITION in dataset["customers"] list,
    NOT on customer IDs! Use build_id_to_idx_mapping() to get the correct mapping.

    Args:
        dataset: The dataset containing depot and customers
        multiplier: Factor to multiply Euclidean distance by (e.g., 1.5 for tortuosity)
    """
    depot = dataset["depot"]
    customers = dataset["customers"]

    nodes = [depot] + customers
    n = len(nodes)

    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                dist = euclidean_distance(
                    nodes[i]["x"], nodes[i]["y"],
                    nodes[j]["x"], nodes[j]["y"]
                )
                matrix[i][j] = dist * multiplier

    return matrix


def build_id_to_idx_mapping(dataset: Dict) -> Dict[int, int]:
    """
    Build a mapping from customer ID to distance matrix index.

    Struktur mapping:
    - Depot (ID 0) → Index 0
    - Customer n (ID apapun) → Index berdasarkan posisi di dataset["customers"]

    Contoh:
    - dataset["customers"] = [{"id": 1}, {"id": 5}, {"id": 10}]
    - Mapping: {0: 0, 1: 1, 5: 2, 10: 3}

    Fungsi ini WAJIB digunakan setiap kali mengakses distance_matrix.
    """
    id_to_idx = {0: 0}  # Depot selalu di index 0

    for idx, customer in enumerate(dataset["customers"]):
        customer_id = customer["id"]
        matrix_idx = idx + 1  # +1 karena depot di index 0
        id_to_idx[customer_id] = matrix_idx

    return id_to_idx


# ============================================================
# VEHICLE AVAILABILITY FUNCTIONS
# ============================================================

def is_vehicle_available(vehicle: Dict) -> bool:
    """
    Check if a vehicle is available (has valid availability times).
    If available_from or available_until is empty/None, vehicle is NOT available.
    """
    available_from = vehicle.get("available_from", "")
    available_until = vehicle.get("available_until", "")

    # Vehicle is available only if BOTH times are specified
    return bool(available_from and available_until)


def get_vehicle_availability_minutes(vehicle: Dict) -> Tuple[float, float]:
    """
    Get vehicle availability window in minutes from midnight.
    Returns (start_minutes, end_minutes).
    """
    if not is_vehicle_available(vehicle):
        return (0, 0)

    start = parse_time_to_minutes(vehicle["available_from"])
    end = parse_time_to_minutes(vehicle["available_until"])
    return (start, end)


def does_route_fit_vehicle_availability(
    route_start_time: float,
    route_end_time: float,
    vehicle: Dict
) -> bool:
    """
    Check if a route's service timeline fits within vehicle availability.

    Args:
        route_start_time: Route departure time in minutes from midnight
        route_end_time: Route return time to depot in minutes from midnight
        vehicle: Vehicle dict with available_from and available_until

    Returns:
        True if route fits within vehicle availability window
    """
    if not is_vehicle_available(vehicle):
        return False

    veh_start, veh_end = get_vehicle_availability_minutes(vehicle)

    # Route must start after vehicle becomes available
    # Route must end before vehicle availability ends
    return route_start_time >= veh_start and route_end_time <= veh_end


def get_available_vehicles(fleet: List[Dict]) -> List[Dict]:
    """
    Filter fleet to only include available vehicles.
    A vehicle is available if it has both available_from and available_until set.
    """
    available = []
    for vehicle in fleet:
        if is_vehicle_available(vehicle):
            available.append(vehicle)
    return available


def get_vehicle_availability_status(fleet: List[Dict]) -> List[Dict]:
    """
    Get availability status for all vehicles (for display).
    """
    status_list = []
    for vehicle in fleet:
        available = is_vehicle_available(vehicle)
        status = {
            "vehicle_id": vehicle["id"],
            "capacity": vehicle["capacity"],
            "units": vehicle.get("units", 1),
            "available": available,
            "available_from": vehicle.get("available_from", ""),
            "available_until": vehicle.get("available_until", ""),
            "status": "✅ Available" if available else "❌ Not Available"
        }
        if available:
            status["time_window"] = f"{vehicle['available_from']} – {vehicle['available_until']}"
        else:
            status["time_window"] = "(Not Set)"
        status_list.append(status)
    return status_list


# ============================================================
# SWEEP ALGORITHM (DETERMINISTIC)
# ============================================================

def get_vehicle_type_for_demand(
    demand: int,
    fleet: List[Dict],
    route_start_time: float = 480.0,  # Default 08:00
    route_end_time: float = 1020.0,   # Default 17:00
    check_availability: bool = True
) -> Tuple[str, str]:
    """
    Get appropriate vehicle type for given demand.
    Checks: capacity constraint AND availability time constraint.

    Args:
        demand: Total demand of the route
        fleet: Fleet data with capacity and availability times
        route_start_time: Route departure time in minutes from midnight
        route_end_time: Route return time to depot in minutes from midnight
        check_availability: Whether to check vehicle availability time

    Returns:
        Tuple of (vehicle_id, selection_reason)

    Selection Rules:
    1. Filter by availability (if check_availability=True)
    2. Filter by capacity (demand <= capacity)
    3. Select smallest capacity vehicle from remaining
    """
    # Get available vehicles only
    if check_availability:
        available_fleet = get_available_vehicles(fleet)
    else:
        available_fleet = fleet

    if not available_fleet:
        return (None, "No vehicles available (all have empty availability times)")

    # Sort by capacity (ascending) to prefer smallest feasible vehicle
    sorted_fleet = sorted(available_fleet, key=lambda v: v["capacity"])

    for vehicle in sorted_fleet:
        # Check capacity
        if demand > vehicle["capacity"]:
            continue

        # Check availability time window if enabled
        if check_availability:
            if not does_route_fit_vehicle_availability(route_start_time, route_end_time, vehicle):
                continue

        # Vehicle is valid
        reason = f"Demand {demand} fits in {vehicle['id']} (capacity ≤ {vehicle['capacity']})"
        if check_availability:
            reason += f", available {vehicle['available_from']}–{vehicle['available_until']}"
        return (vehicle["id"], reason)

    # No vehicle fits - return largest available as fallback
    if sorted_fleet:
        largest = sorted_fleet[-1]
        return (largest["id"], f"Demand {demand} exceeds all capacities, using largest: {largest['id']}")

    return (None, "No feasible vehicle found")


def academic_sweep(dataset: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform Sweep algorithm with DYNAMIC clustering based on user input data.

    Algoritma Sweep (Gillett & Miller, 1974):
    1. Hitung sudut polar setiap customer relative ke depot
    2. Urutkan customer berdasarkan sudut polar (ascending)
    3. Bentuk cluster dengan kapasitas kendaraan sebagai batasan

    FORCED TERMINATION LOGIC:
    - Setelah menambahkan customer, cek apakah ada customer lain yang muat
    - Jika tidak ada customer yang muat, STOP cluster (forced termination)
    - Customer yang belum terlayani masuk ke cluster berikutnya

    Returns: (clusters, iteration_logs)
    """
    depot = dataset["depot"]
    customers = dataset["customers"]
    fleet = dataset["fleet"]

    iteration_logs = []

    # Step 1: Hitung sudut polar setiap customer
    customer_angles = []
    for c in customers:
        angle = compute_polar_angle_degrees(c, depot)
        customer_angles.append({
            "id": c["id"],
            "angle": angle,
            "demand": c["demand"],
            "customer_data": c
        })
        iteration_logs.append({
            "phase": "SWEEP",
            "step": "polar_angle",
            "customer_id": c["id"],
            "angle": round(angle, 2),
            "formula": f"atan2({c['y'] - depot['y']}, {c['x'] - depot['x']}) × 180/π = {round(angle, 2)}°"
        })

    # Step 2: Urutkan berdasarkan sudut polar (ascending)
    sorted_customers = sorted(customer_angles, key=lambda x: x["angle"])
    sorted_ids = [c["id"] for c in sorted_customers]

    iteration_logs.append({
        "phase": "SWEEP",
        "step": "sorted_order",
        "order": sorted_ids,
        "description": "Customer diurutkan berdasarkan sudut polar (ascending)"
    })

    # Step 3: Bentuk cluster berdasarkan kapasitas kendaraan
    # TRACK AVAILABLE UNITS DYNAMICALLY
    vehicle_counts = {v['id']: v.get('units', 1) for v in fleet}

    clusters = []
    unassigned = sorted_customers.copy()
    cluster_id = 0

    while unassigned:
        cluster_id += 1

        # Determine active fleet (vehicles with remaining units)
        active_fleet = [v for v in fleet if vehicle_counts.get(v['id'], 0) > 0]

        # Fallback if no vehicles left (should not happen with sufficient fleet)
        if not active_fleet:
            active_fleet = fleet
            iteration_logs.append({
                "phase": "SWEEP",
                "step": "fleet_exhausted",
                "cluster_id": cluster_id,
                "description": "⚠️ Semua kendaraan habis terpakai! Menggunakan armada penuh sebagai fallback."
            })

        # Set max_capacity based on LARGEST AVAILABLE vehicle
        # Sort desc by capacity
        active_fleet_sorted = sorted(
            active_fleet, key=lambda v: v["capacity"], reverse=True)
        max_capacity = active_fleet_sorted[0]["capacity"] if active_fleet_sorted else 100

        current_cluster = {
            "cluster_id": cluster_id,
            "customer_ids": [],
            "total_demand": 0,
            "vehicle_type": None
        }

        remaining_capacity = max_capacity

        # Iterasi customer yang belum tergabung
        i = 0
        while i < len(unassigned):
            customer = unassigned[i]

            # Cek apakah customer muat di cluster ini
            if current_cluster["total_demand"] + customer["demand"] <= remaining_capacity:
                current_cluster["customer_ids"].append(customer["id"])
                current_cluster["total_demand"] += customer["demand"]

                iteration_logs.append({
                    "phase": "SWEEP",
                    "step": "customer_added",
                    "cluster_id": cluster_id,
                    "customer_id": customer["id"],
                    "demand": customer["demand"],
                    "cluster_demand": current_cluster["total_demand"],
                    "remaining_capacity": remaining_capacity - current_cluster["total_demand"],
                    "description": f"Customer {customer['id']} (demand={customer['demand']}) ditambahkan ke Cluster {cluster_id}"
                })

                unassigned.pop(i)

                # FORCED TERMINATION CHECK:
                # Cek apakah masih ada customer yang bisa muat
                can_any_fit = False
                for remaining in unassigned:
                    if current_cluster["total_demand"] + remaining["demand"] <= remaining_capacity:
                        can_any_fit = True
                        break

                if not can_any_fit and unassigned:
                    iteration_logs.append({
                        "phase": "SWEEP",
                        "step": "forced_termination",
                        "cluster_id": cluster_id,
                        "reason": f"Tidak ada customer tersisa yang muat (sisa kapasitas: {remaining_capacity - current_cluster['total_demand']})",
                        "description": f"Cluster {cluster_id} dihentikan (forced termination)"
                    })
                    break
            else:
                i += 1

        # Pilih kendaraan yang sesuai untuk cluster dari ACTIVE FLEET
        if current_cluster["customer_ids"]:
            # Use active_fleet to ensure we pick an available one
            # Note: get_vehicle_type_for_demand performs Check 1 (Availability Time) and Check 2 (Capacity)
            # We already filtered for Unit Availability in active_fleet.
            vehicle_id, reason = get_vehicle_type_for_demand(
                current_cluster["total_demand"], active_fleet
            )
            current_cluster["vehicle_type"] = vehicle_id

            # Decrement unit count
            if vehicle_id in vehicle_counts:
                vehicle_counts[vehicle_id] -= 1
                reason += f" (Sisa unit: {vehicle_counts[vehicle_id]})"

            iteration_logs.append({
                "phase": "SWEEP",
                "step": "cluster_formed",
                "cluster_id": cluster_id,
                "customer_ids": current_cluster["customer_ids"],
                "total_demand": current_cluster["total_demand"],
                "vehicle_type": vehicle_id,
                "vehicle_reason": reason
            })

            clusters.append(current_cluster)

        # Safety check: prevent infinite loop
        if not current_cluster["customer_ids"]:
            # Tidak ada customer yang bisa ditambahkan, force add satu
            if unassigned:
                forced_customer = unassigned.pop(0)
                current_cluster["customer_ids"].append(forced_customer["id"])
                current_cluster["total_demand"] = forced_customer["demand"]

                # Try to assign strict, if fail, fallback to any
                vehicle_id, reason = get_vehicle_type_for_demand(
                    current_cluster["total_demand"], active_fleet
                )
                if not vehicle_id:
                    vehicle_id, reason = get_vehicle_type_for_demand(
                        current_cluster["total_demand"], fleet
                    )

                current_cluster["vehicle_type"] = vehicle_id
                clusters.append(current_cluster)

                iteration_logs.append({
                    "phase": "SWEEP",
                    "step": "forced_cluster",
                    "cluster_id": cluster_id,
                    "customer_ids": current_cluster["customer_ids"],
                    "reason": "Customer demand melebihi kapasitas, dialokasikan secara paksa"
                })

    return clusters, iteration_logs


# ============================================================
# NEAREST NEIGHBOR (TIME-WINDOW AWARE - WORD COMPLIANT)
# ============================================================

def academic_nearest_neighbor(
    cluster: Dict,
    dataset: Dict,
    distance_matrix: List[List[float]]
) -> Tuple[Dict, List[Dict]]:
    """
    Perform Nearest Neighbor with TIME WINDOW awareness (WORD-COMPLIANT).

    PSEUDOCODE (from Word document):
    - Find nearest customer from current position
    - Calculate arrival_time = current_time + travel_time
    - If arrival_time > TW_end: REJECT customer (mark as unassigned)
    - If arrival_time < TW_start: WAIT until TW_start
    - Service starts at max(arrival_time, TW_start)
    - Departure = service_start + service_time

    TIME WINDOW IS HARD CONSTRAINT IN NN PHASE.
    Rejected customers are stored as unassigned.

    Returns: (route, iteration_logs)
    """
    customer_ids = cluster["customer_ids"].copy()
    customers = {c["id"]: c for c in dataset["customers"]}
    depot = dataset["depot"]

    iteration_logs = []

    # Use centralized ID to matrix index mapping function
    id_to_idx = build_id_to_idx_mapping(dataset)

    # Build route using NN with TW awareness
    sequence = [0]
    remaining = set(customer_ids)
    unassigned = []  # Customers rejected due to TW violation
    current = 0  # Start at depot (this is the customer ID, not matrix index)

    total_distance = 0.0
    total_service_time = 0.0
    total_wait_time = 0.0
    current_time = parse_time_to_minutes(depot["time_window"]["start"])

    stops = [{
        "node_id": 0,
        "arrival": current_time,
        "departure": current_time,
        "wait": 0,
        "violation": 0,
        "tw_start": current_time,
        "tw_end": parse_time_to_minutes(depot["time_window"]["end"])
    }]

    step = 1

    while remaining:
        # Find nearest customer from current position
        nearest = None
        nearest_dist = float('inf')

        # Get matrix index for current position - direct access, no fallback
        current_idx = id_to_idx[current]

        for cid in remaining:
            # Get matrix index for customer - direct access, no fallback
            cid_idx = id_to_idx[cid]

            # Safety check for matrix bounds
            if current_idx >= len(distance_matrix) or cid_idx >= len(distance_matrix[0]):
                continue

            dist = distance_matrix[current_idx][cid_idx]
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = cid

        if nearest is None:
            break

        # Calculate arrival time
        travel_time = nearest_dist  # Speed = 1 unit/min
        arrival_time = current_time + travel_time

        customer = customers[nearest]
        tw_start = parse_time_to_minutes(customer["time_window"]["start"])
        tw_end = parse_time_to_minutes(customer["time_window"]["end"])
        service = customer["service_time"]

        # CHECK TIME WINDOW (HARD CONSTRAINT)
        if arrival_time > tw_end:
            # REJECT: arrival_time > TW_end
            # Mark customer as unassigned and continue
            unassigned.append(nearest)
            remaining.remove(nearest)

            iteration_logs.append({
                "phase": "NN",
                "cluster_id": cluster["cluster_id"],
                "step": step,
                "from_node": current,
                "to_node": nearest,
                "distance": round(nearest_dist, 2),
                "arrival_time": round(arrival_time, 2),
                "tw_end": tw_end,
                "action": "REJECTED",
                "reason": f"Arrival {round(arrival_time, 2)} > TW_end {tw_end}",
                "description": f"Customer {nearest} REJECTED (arrival {round(arrival_time, 2)} > TW_end {tw_end})"
            })
            step += 1
            continue

        # ACCEPT: arrival_time <= TW_end
        # Log the selection
        iteration_logs.append({
            "phase": "NN",
            "cluster_id": cluster["cluster_id"],
            "step": step,
            "from_node": current,
            "to_node": nearest,
            "distance": round(nearest_dist, 2),
            "arrival_time": round(arrival_time, 2),
            "tw_start": tw_start,
            "tw_end": tw_end,
            "action": "ACCEPTED",
            "description": f"Select customer {nearest} (distance = {round(nearest_dist, 2)})"
        })

        # Update route
        sequence.append(nearest)
        total_distance += nearest_dist

        # WAIT if early (arrival < TW_start)
        wait = max(0, tw_start - arrival_time)
        total_wait_time += wait

        # SERVICE STARTS at max(arrival, TW_start)
        service_start = max(arrival_time, tw_start)

        # Departure = service_start + service_time
        departure = service_start + service
        total_service_time += service

        stops.append({
            "node_id": nearest,
            "raw_arrival": round(arrival_time, 2),
            "arrival": round(service_start, 2),
            "departure": round(departure, 2),
            "wait": round(wait, 2),
            "violation": 0,  # No violation if accepted
            "service_time": service,
            "tw_start": tw_start,
            "tw_end": tw_end
        })

        current_time = departure
        current = nearest
        remaining.remove(nearest)
        step += 1

    # Return to depot
    if len(sequence) > 1:  # Only if we visited at least one customer
        current_idx = id_to_idx[current]  # Direct access, no fallback
        return_dist = distance_matrix[current_idx][0]
        total_distance += return_dist
        sequence.append(0)

        final_arrival = current_time + return_dist
        stops.append({
            "node_id": 0,
            "arrival": round(final_arrival, 2),
            "departure": round(final_arrival, 2),
            "wait": 0,
            "violation": 0,
            "tw_start": parse_time_to_minutes(depot["time_window"]["start"]),
            "tw_end": parse_time_to_minutes(depot["time_window"]["end"])
        })

        iteration_logs.append({
            "phase": "NN",
            "cluster_id": cluster["cluster_id"],
            "step": step,
            "from_node": current,
            "to_node": 0,
            "distance": round(return_dist, 2),
            "description": f"Return to depot (distance = {round(return_dist, 2)})"
        })
    else:
        # No customers served, just depot
        sequence.append(0)

    # Log unassigned customers if any
    if unassigned:
        iteration_logs.append({
            "phase": "NN",
            "cluster_id": cluster["cluster_id"],
            "step": "unassigned",
            "unassigned_customers": unassigned,
            "count": len(unassigned),
            "description": f"{len(unassigned)} customer(s) rejected due to TW violation"
        })

    # Recalculate demand for served customers only
    served_customers = [c for c in sequence if c != 0]
    served_demand = sum(customers[c]["demand"] for c in served_customers)

    route = {
        "cluster_id": cluster["cluster_id"],
        "vehicle_type": cluster["vehicle_type"],
        "sequence": sequence,
        "stops": stops,
        "total_distance": round(total_distance, 2),
        "total_service_time": total_service_time,
        "total_travel_time": round(total_distance, 2),  # Speed = 1
        "total_tw_violation": 0.0,  # No violations in NN (hard constraint)
        "total_wait_time": round(total_wait_time, 2),
        "unassigned_customers": unassigned,
        "served_demand": served_demand,
        "total_demand": cluster["total_demand"]
    }

    return route, iteration_logs


# ============================================================
# ACS (DYNAMIC MODE - No Hardcoded Data)
# ============================================================

# NOTE: ACS sekarang menggunakan dynamic route construction
# berdasarkan pheromone dan heuristic, bukan hardcoded routes.
# Setiap semut membangun rute secara dinamis menggunakan:
# - Pheromone levels (tau)
# - Heuristic information (eta = 1/distance)
# - Probability-based selection


def academic_acs_cluster(
    cluster: Dict,
    dataset: Dict,
    distance_matrix: List[List[float]],
    initial_route: Dict
) -> Tuple[Dict, List[Dict]]:
    """
    Perform ACS in ACADEMIC REPLAY MODE (WORD-COMPLIANT).

    PSEUDOCODE (from Word document):
    PSEUDOCODE (Dynamic Simulation):
    - Routes are constructed using ACS State Transition Rule
    - Probabilistic selection based on Pheromone & Heuristic
    - Time windows are SOFT CONSTRAINTS (log violations but accept route)
    - Acceptance is based on DISTANCE ONLY
    - Pheromone updates still occur for educational purposes

    Returns: (best_route, iteration_logs)
    """
    acs_params = dataset["acs_parameters"]
    alpha = acs_params["alpha"]
    beta = acs_params["beta"]
    rho = acs_params["rho"]
    q0 = acs_params.get("q0", 0.9)
    num_ants = acs_params["num_ants"]
    max_iterations = acs_params["max_iterations"]

    customer_ids = cluster["customer_ids"]
    customers = {c["id"]: c for c in dataset["customers"]}

    iteration_logs = []

    # Build ID to matrix index mapping for distance lookups
    id_to_idx = build_id_to_idx_mapping(dataset)

    # Initialize pheromone (tau0 = 1 / (n * L_nn))
    n = len(customer_ids)
    nn_length = initial_route["total_distance"]
    tau0 = 1 / (n * nn_length) if nn_length > 0 else 0.1

    # Pheromone matrix
    all_nodes = [0] + customer_ids
    pheromone = {(i, j): tau0 for i in all_nodes for j in all_nodes if i != j}

    iteration_logs.append({
        "phase": "ACS",
        "cluster_id": cluster["cluster_id"],
        "step": "init_pheromone",
        "mode": "ACADEMIC_REPLAY",
        "tau0": round(tau0, 6),
        "nn_length": nn_length,
        "formula": f"tau0 = 1 / ({n} × {nn_length}) = {round(tau0, 6)}",
        "description": "Using PREDEFINED routes from Word document"
    })

    # Compute initial objective function Z = αD + βT + γTW
    initial_objective = compute_objective(initial_route, dataset)
    initial_route["objective"] = initial_objective

    best_route = initial_route
    # OBJECTIVE FUNCTION (NOT DISTANCE ONLY!)
    best_objective = initial_objective

    iteration_logs.append({
        "phase": "ACS",
        "cluster_id": cluster["cluster_id"],
        "step": "init_objective",
        "initial_distance": initial_route["total_distance"],
        "initial_time": initial_route["total_travel_time"] + initial_route["total_service_time"],
        "initial_tw_violation": initial_route.get("total_tw_violation", 0),
        "initial_objective": round(initial_objective, 2),
        "formula": "Z = α×D + β×T + γ×TW_violation",
        "description": f"Initial objective Z = {round(initial_objective, 2)}"
    })

    for iteration in range(1, max_iterations + 1):
        iteration_best_route = None
        iteration_best_objective = float('inf')

        for ant in range(1, num_ants + 1):
            # DYNAMIC ROUTE CONSTRUCTION: Generate route using ACS probability rule
            # Instead of hardcoded routes, we generate permutations dynamically
            remaining = customer_ids.copy()
            ant_route_seq = []
            current = 0  # Start from depot

            # Use pheromone and heuristic to construct route
            while remaining:
                # Calculate probabilities for each remaining customer
                probs = []
                for cid in remaining:
                    # Use id_to_idx mapping for matrix access
                    current_idx = id_to_idx[current]
                    cid_idx = id_to_idx[cid]

                    # Heuristic: inverse of distance
                    dist = distance_matrix[current_idx][cid_idx]
                    eta = 1.0 / max(dist, 0.001)  # Heuristic (1/distance)

                    # Pheromone level
                    tau = pheromone.get((current, cid), tau0)

                    # Probability = tau^alpha * eta^beta
                    prob = (tau ** alpha) * (eta ** beta)
                    probs.append((cid, prob))

                # Normalize probabilities
                total_prob = sum(p[1] for p in probs)
                if total_prob > 0:
                    probs = [(cid, p / total_prob) for cid, p in probs]

                # Select next customer using ACS Transition Rule
                if not probs:
                    next_cust = remaining[0]
                else:
                    # Generate random q
                    q = random.random()

                    if q <= q0:
                        # Exploitation (ArgMax)
                        next_cust = max(probs, key=lambda x: x[1])[0]
                    else:
                        # Exploration (Roulette Wheel / Weighted Random)
                        candidates = [p[0] for p in probs]
                        weights = [p[1] for p in probs]
                        next_cust = random.choices(
                            candidates, weights=weights, k=1)[0]

                ant_route_seq.append(next_cust)
                remaining.remove(next_cust)
                current = next_cust

            # Build full route with depot bookends
            route = [0] + ant_route_seq + [0]

            iteration_logs.append({
                "phase": "ACS",
                "cluster_id": cluster["cluster_id"],
                "iteration": iteration,
                "ant": ant,
                "step": "route_constructed",
                "route": route,
                "description": f"Rute dinamis untuk cluster {cluster['cluster_id']}, iterasi {iteration}, semut {ant}"
            })

            # Evaluate route (TW is SOFT CONSTRAINT - log but accept)
            route_result = evaluate_route(
                route, cluster, dataset, distance_matrix)
            route_distance = route_result["total_distance"]

            # COMPUTE OBJECTIVE FUNCTION Z = α*D + β*T + γ*TW_violation (WORD FORMULA!)
            route_objective = compute_objective(route_result, dataset)
            route_result["objective"] = route_objective

            # Log time window information (soft constraint)
            tw_violations = route_result.get("tw_violations_detail", [])
            if tw_violations:
                iteration_logs.append({
                    "phase": "ACS",
                    "cluster_id": cluster["cluster_id"],
                    "iteration": iteration,
                    "ant": ant,
                    "step": "tw_soft_constraint",
                    "total_tw_violation": route_result["total_tw_violation"],
                    "violations": tw_violations,
                    "description": f"Pelanggaran TW tercatat (soft constraint): {route_result['total_tw_violation']} menit"
                })

            # Log with FULL OBJECTIVE FUNCTION
            weights = dataset["objective_weights"]
            iteration_logs.append({
                "phase": "ACS",
                "cluster_id": cluster["cluster_id"],
                "iteration": iteration,
                "ant": ant,
                "step": "route_evaluation",
                "route": route,
                "distance": route_distance,
                "travel_time": route_result["total_travel_time"],
                "service_time": route_result["total_service_time"],
                "total_time": route_result["total_travel_time"] + route_result["total_service_time"],
                "tw_violation": route_result["total_tw_violation"],
                "wait_time": route_result.get("total_wait_time", 0),
                "objective_formula": f"Z = {weights['w1_distance']}×D + {weights['w2_time']}×T + {weights['w3_tw_violation']}×V",
                "objective_calculation": f"Z = {weights['w1_distance']}×{route_distance} + {weights['w2_time']}×{route_result['total_travel_time'] + route_result['total_service_time']} + {weights['w3_tw_violation']}×{route_result['total_tw_violation']}",
                "objective": round(route_objective, 2),
                "acceptance_criterion": "FUNGSI_TUJUAN",
                "description": f"Z = {round(route_objective, 2)} (penerimaan berdasarkan Z = αD + βT + γTW)"
            })

            # Local pheromone update for educational purposes
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                old_tau = pheromone.get((u, v), tau0)
                new_tau = (1 - rho) * old_tau + rho * tau0
                pheromone[(u, v)] = new_tau

            # ACCEPTANCE BASED ON OBJECTIVE FUNCTION Z (NOT DISTANCE ONLY!)
            if route_objective < iteration_best_objective:
                iteration_best_objective = route_objective
                iteration_best_route = route_result

        # Global pheromone update on iteration best
        if iteration_best_route:
            L_best = iteration_best_route["total_distance"]
            delta = 1 / L_best if L_best > 0 else 0
            for i in range(len(iteration_best_route["sequence"]) - 1):
                u = iteration_best_route["sequence"][i]
                v = iteration_best_route["sequence"][i + 1]
                old_tau = pheromone.get((u, v), tau0)
                new_tau = (1 - rho) * old_tau + rho * delta
                pheromone[(u, v)] = new_tau

        # Update global best (OBJECTIVE FUNCTION!)
        if iteration_best_objective < best_objective:
            best_objective = iteration_best_objective
            best_route = iteration_best_route

            iteration_logs.append({
                "phase": "ACS",
                "cluster_id": cluster["cluster_id"],
                "iteration": iteration,
                "step": "new_best_found",
                "new_best_objective": round(best_objective, 2),
                "new_best_distance": best_route["total_distance"],
                "description": f"Rute terbaik baru ditemukan: Z = {round(best_objective, 2)}"
            })

        iteration_logs.append({
            "phase": "ACS",
            "cluster_id": cluster["cluster_id"],
            "iteration": iteration,
            "step": "iteration_summary",
            "best_route": best_route["sequence"],
            "best_objective": round(best_route.get("objective", best_objective), 2),
            "best_distance": best_route["total_distance"],
            "best_service_time": best_route["total_service_time"],
            "best_tw_violation": best_route.get("total_tw_violation", 0),
            "acceptance_criterion": "FUNGSI_TUJUAN (Z = αD + βT + γTW)"
        })

    return best_route, iteration_logs


def evaluate_route(
    sequence: List[int],
    cluster: Dict,
    dataset: Dict,
    distance_matrix: List[List[float]]
) -> Dict:
    """
    Evaluate a route and compute metrics with TIME WINDOW handling.

    TIME WINDOW LOGIC (soft constraints):
    - Arrival time = previous departure + travel time
    - Service starts at max(arrival_time, TW_start)
    - If arrival_time < TW_start: vehicle waits (wait = TW_start - arrival)
    - If arrival_time > TW_end: mark violation (violation = arrival - TW_end)
    - Customer is STILL served even with violation
    - Solution is NOT rejected based on time windows
    - Acceptance criteria remain based on DISTANCE ONLY
    """
    customers = {c["id"]: c for c in dataset["customers"]}
    depot = dataset["depot"]

    # Use centralized ID to matrix index mapping function
    id_to_idx = build_id_to_idx_mapping(dataset)

    total_distance = 0.0
    total_service_time = 0.0
    total_tw_violation = 0.0
    total_wait_time = 0.0
    current_time = parse_time_to_minutes(depot["time_window"]["start"])

    stops = []
    tw_violations_detail = []  # Track detailed violations

    # Add depot departure
    stops.append({
        "node_id": 0,
        "arrival": current_time,
        "departure": current_time,
        "wait": 0,
        "violation": 0,
        "tw_start": current_time,
        "tw_end": parse_time_to_minutes(depot["time_window"]["end"])
    })

    for i in range(len(sequence) - 1):
        current = sequence[i]
        next_node = sequence[i + 1]

        # Use id_to_idx mapping for matrix access - NO FALLBACK, must be in mapping
        current_idx = id_to_idx[current]
        next_idx = id_to_idx[next_node]
        dist = distance_matrix[current_idx][next_idx]
        total_distance += dist

        travel_time = dist  # Speed = 1 (distance unit per minute)

        # ARRIVAL TIME = previous departure + travel time
        raw_arrival = current_time + travel_time

        if next_node == 0:
            # Back to depot
            stops.append({
                "node_id": 0,
                "arrival": round(raw_arrival, 2),
                "departure": round(raw_arrival, 2),
                "wait": 0,
                "violation": 0,
                "tw_start": parse_time_to_minutes(depot["time_window"]["start"]),
                "tw_end": parse_time_to_minutes(depot["time_window"]["end"])
            })
            current_time = raw_arrival
        else:
            customer = customers[next_node]
            tw_start = parse_time_to_minutes(customer["time_window"]["start"])
            tw_end = parse_time_to_minutes(customer["time_window"]["end"])
            service = customer["service_time"]

            # WAIT TIME: if arrival < TW_start, vehicle waits
            wait = max(0, tw_start - raw_arrival)
            total_wait_time += wait

            # SERVICE STARTS at max(arrival, TW_start)
            service_start = max(raw_arrival, tw_start)

            # VIOLATION: if arrival > TW_end, mark violation (soft constraint)
            # Note: We compare raw_arrival against TW_end
            violation = max(0, raw_arrival - tw_end)
            total_tw_violation += violation

            if violation > 0:
                tw_violations_detail.append({
                    "customer_id": next_node,
                    "arrival": round(raw_arrival, 2),
                    "tw_end": tw_end,
                    "violation_minutes": round(violation, 2)
                })

            # DEPARTURE = service start + service time
            departure = service_start + service
            total_service_time += service

            stops.append({
                "node_id": next_node,
                "raw_arrival": round(raw_arrival, 2),
                "arrival": round(service_start, 2),  # Service start time
                "departure": round(departure, 2),
                "wait": round(wait, 2),
                "violation": round(violation, 2),
                "service_time": service,
                "tw_start": tw_start,
                "tw_end": tw_end
            })

            current_time = departure

    return {
        "cluster_id": cluster["cluster_id"],
        "vehicle_type": cluster["vehicle_type"],
        "sequence": sequence,
        "stops": stops,
        "total_distance": round(total_distance, 2),
        "total_service_time": total_service_time,
        "total_travel_time": round(total_distance, 2),
        "total_tw_violation": round(total_tw_violation, 2),
        "total_wait_time": round(total_wait_time, 2),
        "tw_violations_detail": tw_violations_detail,
        "total_demand": cluster["total_demand"]
    }


def compute_objective(route: Dict, dataset: Dict) -> float:
    """Compute objective function as in Word: Z = w1*D + w2*T + w3*V"""
    weights = dataset["objective_weights"]
    w1 = weights.get("w1_distance", 1.0)
    w2 = weights.get("w2_time", 1.0)
    w3 = weights.get("w3_tw_violation", 1.0)

    D = route["total_distance"]
    T = route["total_service_time"] + route["total_travel_time"]
    V = route["total_tw_violation"]

    return w1 * D + w2 * T + w3 * V


# ============================================================
# RVND (DYNAMIC MODE - No Hardcoded Values)
# ============================================================

# NOTE: RVND sekarang menggunakan dynamic neighborhood generation
# berdasarkan rute aktual, bukan hardcoded swaps dari dokumen Word.
#
# Neighborhoods yang digunakan:
# - swap_1_1: Tukar 1 customer antar rute
# - shift_1_0: Pindahkan 1 customer ke rute lain
# - two_opt: Reverse segment dalam rute
# - exchange: Tukar posisi 2 customer dalam rute
# - or_opt: Pindahkan segment pendek dalam rute


def academic_rvnd(
    routes: List[Dict],
    dataset: Dict,
    distance_matrix: List[List[float]],
    max_inter_iterations: int = 50,
    max_intra_iterations: int = 100
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform RVND in ACADEMIC REPLAY MODE (WORD-COMPLIANT).

    PSEUDOCODE (from Word document):
    - Use PREDEFINED_SWAP_PAIRS for inter-route moves
    - CAPACITY is HARD CONSTRAINT (reject if violated)
    - TIME WINDOWS are SOFT CONSTRAINTS (log but accept)
    - Accept improvements based on DISTANCE ONLY
    - RESET neighborhood list on improvement
    - LOG EVERY ITERATION

    Returns: (improved_routes, iteration_logs)
    """
    iteration_logs = []
    current_routes = deepcopy(routes)

    # INTER-ROUTE RVND (using predefined swap pairs)
    current_routes, inter_logs = academic_rvnd_inter(
        current_routes, dataset, distance_matrix, max_inter_iterations
    )
    iteration_logs.extend(inter_logs)

    # INTRA-ROUTE RVND (per route, using predefined moves)
    for route in current_routes:
        intra_logs = academic_rvnd_intra(
            route, dataset, distance_matrix, max_intra_iterations
        )
        iteration_logs.extend(intra_logs)

    return current_routes, iteration_logs


def academic_rvnd_inter(
    routes: List[Dict],
    dataset: Dict,
    distance_matrix: List[List[float]],
    max_iterations: int = 50
) -> Tuple[List[Dict], List[Dict]]:
    """
    Inter-route RVND in ACADEMIC REPLAY MODE.

    Uses predefined swap pairs from Word document.
    CAPACITY is HARD CONSTRAINT - moves violating capacity are REJECTED.
    TIME WINDOWS are SOFT - violations are logged but move is accepted.
    DISTANCE ONLY acceptance criterion.
    RESET NL on improvement.
    LOG EVERY ITERATION.

    Returns: (final_routes, iteration_logs)
    """
    iteration_logs = []

    # KODE BARU (normalize ID):
    def normalize_vehicle_id(vid):
        """Normalize vehicle ID to match between components."""
        if vid.startswith("Vehicle "):
            return vid.replace("Vehicle ", "")
        return vid

    fleet = {}
    for f in dataset["fleet"]:
        # Simpan dengan KEDUA format ID
        fleet[f["id"]] = f
        fleet[normalize_vehicle_id(f["id"])] = f
        # fleet = {f["id"]: f for f in dataset["fleet"]}
        customers = {c["id"]: c for c in dataset["customers"]}

    # Neighborhood list (exact order from Word)
    NL_FULL = ["swap_1_1", "shift_1_0", "swap_2_1", "swap_2_2", "cross"]
    NL = NL_FULL[:]

    iteration = 0

    # Log every iteration up to max_iterations
    while iteration < max_iterations:
        iteration += 1
        improved_this_iteration = False
        neighborhood_used = None
        move_details = None

        # DYNAMIC NEIGHBORHOOD EXPLORATION
        # Try each neighborhood in current NL
        for neighborhood in NL[:]:
            result = apply_inter_neighborhood(
                neighborhood, routes, dataset, distance_matrix, fleet)

            if result["accepted"]:
                routes = result["new_routes"]
                improved_this_iteration = True
                neighborhood_used = neighborhood
                move_details = result.get("move_details")
                NL = NL_FULL[:]  # RESET NL on improvement
                break
            else:
                NL.remove(neighborhood)

        # LOG EVERY ITERATION (mandatory per spec)
        current_distance = sum(r["total_distance"] for r in routes)
        current_service_time = sum(
            compute_service_time_from_sequence(r["sequence"], dataset) for r in routes
        )

        log_entry = {
            "iteration_id": iteration,
            "phase": "RVND-INTER",
            "mode": "ACADEMIC_REPLAY",
            "neighborhood": neighborhood_used if improved_this_iteration else "none",
            "improved": improved_this_iteration,
            "routes_snapshot": [r["sequence"] for r in routes],
            "total_distance": round(current_distance, 2),
            "total_service_time": current_service_time,
            "vehicle_usage": {r["vehicle_type"]: 1 for r in routes},
            "acceptance_criterion": "DISTANCE_ONLY"
        }

        if move_details:
            log_entry["move"] = move_details

        iteration_logs.append(log_entry)

        # Early stopping if NL exhausted
        if not NL:
            iteration_logs.append({
                "iteration_id": iteration,
                "phase": "RVND-INTER",
                "step": "terminated",
                "reason": "Neighborhood list exhausted",
                "final_distance": round(current_distance, 2)
            })
            break

    return routes, iteration_logs


def apply_inter_neighborhood(
    neighborhood: str,
    routes: List[Dict],
    dataset: Dict,
    distance_matrix: List[List[float]],
    fleet: Dict
) -> Dict:
    """
    Apply inter-route neighborhood operator.
    CAPACITY is HARD CONSTRAINT - reject if violated.
    DISTANCE ONLY acceptance.
    """
    current_distance = sum(r["total_distance"] for r in routes)
    customers = {c["id"]: c for c in dataset["customers"]}
    best_move = None
    best_routes = None
    best_distance = current_distance

    if neighborhood == "swap_1_1":
        # Swap 1 customer between 2 routes
        for i, route_a in enumerate(routes):
            for j, route_b in enumerate(routes):
                if i >= j:
                    continue

                seq_a = route_a["sequence"][1:-1]
                seq_b = route_b["sequence"][1:-1]

                for ca in seq_a:
                    for cb in seq_b:
                        new_seq_a = [0] + [c if c !=
                                           ca else cb for c in seq_a] + [0]
                        new_seq_b = [0] + [c if c !=
                                           cb else ca for c in seq_b] + [0]

                        # CAPACITY HARD CONSTRAINT
                        demand_a = sum(customers[c]["demand"]
                                       for c in new_seq_a[1:-1])
                        demand_b = sum(customers[c]["demand"]
                                       for c in new_seq_b[1:-1])

                        cap_a = fleet[route_a["vehicle_type"]]["capacity"]
                        cap_b = fleet[route_b["vehicle_type"]]["capacity"]

                        if demand_a > cap_a or demand_b > cap_b:
                            continue  # REJECT: Capacity violation

                        dist_a = sum(
                            distance_matrix[new_seq_a[k]][new_seq_a[k+1]] for k in range(len(new_seq_a)-1))
                        dist_b = sum(
                            distance_matrix[new_seq_b[k]][new_seq_b[k+1]] for k in range(len(new_seq_b)-1))

                        other_distance = sum(
                            r["total_distance"] for r in routes if r not in [route_a, route_b])
                        total_new = dist_a + dist_b + other_distance

                        if total_new < best_distance:
                            best_distance = total_new
                            best_move = {"swap": (ca, cb), "routes": (i, j)}

    accepted = best_move is not None and best_distance < current_distance

    return {
        "best_move": best_move,
        "accepted": accepted,
        "distance_before": round(current_distance, 2),
        "distance_after": round(best_distance, 2),
        "new_routes": routes  # Would need to rebuild routes with the move
    }


def academic_rvnd_intra(
    route: Dict,
    dataset: Dict,
    distance_matrix: List[List[float]],
    max_iterations: int = 100
) -> List[Dict]:
    """
    Intra-route RVND in ACADEMIC REPLAY MODE.

    Uses predefined moves from Word document.
    CAPACITY not applicable (same route).
    TIME WINDOWS are SOFT - violations logged but accepted.
    DISTANCE ONLY acceptance criterion.
    RESET NL on improvement.
    LOG EVERY ITERATION.
    """
    iteration_logs = []

    NL_FULL = ["or_opt", "reinsertion", "exchange", "two_opt"]
    NL = NL_FULL[:]

    sequence = route["sequence"]
    current_distance = sum(
        distance_matrix[sequence[i]][sequence[i+1]] for i in range(len(sequence)-1))

    iteration = 0
    cluster_id = route["cluster_id"]

    # Log every iteration up to max_iterations
    while iteration < max_iterations:
        iteration += 1
        improved_this_iteration = False
        neighborhood_used = None

        # DYNAMIC NEIGHBORHOOD EXPLORATION
        # Try each neighborhood in current NL
        for neighborhood in NL[:]:
            result = apply_intra_neighborhood(
                neighborhood, sequence, distance_matrix)

            # DISTANCE ONLY acceptance
            if result["accepted"]:
                sequence = result["new_sequence"]
                current_distance = result["distance_after"]
                improved_this_iteration = True
                neighborhood_used = neighborhood
                NL = NL_FULL[:]  # RESET NL on improvement
                break
            else:
                NL.remove(neighborhood)

        # Calculate service time (accumulated from customer service times)
        current_service_time = compute_service_time_from_sequence(
            sequence, dataset)

        # LOG EVERY ITERATION (mandatory per spec)
        iteration_logs.append({
            "iteration_id": iteration,
            "phase": "RVND-INTRA",
            "mode": "ACADEMIC_REPLAY",
            "cluster_id": cluster_id,
            "neighborhood": neighborhood_used if improved_this_iteration else "none",
            "improved": improved_this_iteration,
            "routes_snapshot": [sequence[:]],
            "total_distance": round(current_distance, 2),
            "total_service_time": current_service_time,
            "vehicle_usage": {route["vehicle_type"]: 1},
            "acceptance_criterion": "DISTANCE_ONLY"
        })

        # Early stopping if NL exhausted
        if not NL:
            break

    # Update route with final sequence and recalculated metrics
    route["sequence"] = sequence
    route["total_distance"] = round(current_distance, 2)
    route["total_service_time"] = compute_service_time_from_sequence(
        sequence, dataset)

    return iteration_logs


def apply_intra_neighborhood(
    neighborhood: str,
    sequence: List[int],
    distance_matrix: List[List[float]]
) -> Dict:
    """
    Apply intra-route neighborhood operator.
    DISTANCE ONLY acceptance criterion.
    """
    current_distance = sum(
        distance_matrix[sequence[i]][sequence[i+1]] for i in range(len(sequence)-1))
    best_sequence = sequence
    best_distance = current_distance

    customers = sequence[1:-1]  # Exclude depots
    n = len(customers)

    if neighborhood == "two_opt":
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                new_customers = customers[:i] + \
                    list(reversed(customers[i:j])) + customers[j:]
                new_seq = [0] + new_customers + [0]
                new_dist = sum(
                    distance_matrix[new_seq[k]][new_seq[k+1]] for k in range(len(new_seq)-1))

                if new_dist < best_distance:
                    best_distance = new_dist
                    best_sequence = new_seq

    elif neighborhood == "or_opt":
        for length in [1, 2, 3]:
            for i in range(n - length + 1):
                segment = customers[i:i+length]
                remaining = customers[:i] + customers[i+length:]

                for j in range(len(remaining) + 1):
                    new_customers = remaining[:j] + segment + remaining[j:]
                    new_seq = [0] + new_customers + [0]
                    new_dist = sum(
                        distance_matrix[new_seq[k]][new_seq[k+1]] for k in range(len(new_seq)-1))

                    if new_dist < best_distance:
                        best_distance = new_dist
                        best_sequence = new_seq

    elif neighborhood == "reinsertion":
        for i in range(n):
            customer = customers[i]
            remaining = customers[:i] + customers[i+1:]

            for j in range(len(remaining) + 1):
                new_customers = remaining[:j] + [customer] + remaining[j:]
                new_seq = [0] + new_customers + [0]
                new_dist = sum(
                    distance_matrix[new_seq[k]][new_seq[k+1]] for k in range(len(new_seq)-1))

                if new_dist < best_distance:
                    best_distance = new_dist
                    best_sequence = new_seq

    elif neighborhood == "exchange":
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_customers = customers[:]
                new_customers[i], new_customers[j] = new_customers[j], new_customers[i]
                new_seq = [0] + new_customers + [0]
                new_dist = sum(
                    distance_matrix[new_seq[k]][new_seq[k+1]] for k in range(len(new_seq)-1))

                if new_dist < best_distance:
                    best_distance = new_dist
                    best_sequence = new_seq

    accepted = best_distance < current_distance

    return {
        "new_sequence": best_sequence,
        "distance_before": round(current_distance, 2),
        "distance_after": round(best_distance, 2),
        "accepted": accepted
    }


# ============================================================
# VEHICLE REASSIGNMENT
# ============================================================

def reassign_vehicles(
    routes: List[Dict],
    dataset: Dict,
    check_availability: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Reassign vehicles based on route demand AND availability time.

    Selection Criteria:
    1. Vehicle must be AVAILABLE (non-empty availability time)
    2. Vehicle capacity must fit route demand
    3. Vehicle availability time must cover route time window
    4. Select smallest feasible vehicle (efficiency)
    """
    iteration_logs = []
    fleet = dataset["fleet"]

    # Filter to available vehicles only
    if check_availability:
        available_fleet = get_available_vehicles(fleet)
    else:
        available_fleet = fleet

    fleet_sorted = sorted(available_fleet, key=lambda f: f["capacity"])
    available_units = {f["id"]: f["units"] for f in available_fleet}

    for route in routes:
        demand = route["total_demand"]
        old_vehicle = route.get("vehicle_type", "?")

        # Estimate route time window (from travel time + service time)
        route_start = 480.0  # 08:00 default departure
        total_time = route.get("total_time", 0) or route.get(
            "total_distance", 0) * 2  # Estimate
        route_end = route_start + total_time

        # Find smallest feasible and available vehicle
        new_vehicle = None
        selection_reason = ""

        for f in fleet_sorted:
            vehicle_id = f["id"]

            # Check unit availability
            if available_units.get(vehicle_id, 0) <= 0:
                continue

            # Check capacity
            if f["capacity"] < demand:
                continue

            # Check availability time window (if enabled)
            if check_availability:
                if not does_route_fit_vehicle_availability(route_start, route_end, f):
                    continue

            # Vehicle is feasible
            new_vehicle = vehicle_id
            selection_reason = f"Demand {demand} ≤ capacity {f['capacity']}"
            if check_availability and f.get("available_from") and f.get("available_until"):
                selection_reason += f", available {f['available_from']}–{f['available_until']}"
            break

        if new_vehicle:
            available_units[new_vehicle] -= 1
            route["vehicle_type"] = new_vehicle

            iteration_logs.append({
                "phase": "VEHICLE_REASSIGN",
                "cluster_id": route["cluster_id"],
                "demand": demand,
                "old_vehicle": old_vehicle,
                "new_vehicle": new_vehicle,
                "reason": selection_reason,
                "status": "✅ Assigned"
            })
        else:
            # No vehicle available - keep old or mark as error
            iteration_logs.append({
                "phase": "VEHICLE_REASSIGN",
                "cluster_id": route["cluster_id"],
                "demand": demand,
                "old_vehicle": old_vehicle,
                "new_vehicle": None,
                "reason": "No feasible vehicle: all unavailable or insufficient capacity",
                "status": "❌ No Vehicle"
            })

    return routes, iteration_logs


# ============================================================
# ROUTE STRUCTURE VALIDATION (CRITICAL)
# ============================================================

def validate_route_structure(routes: List[Dict]) -> List[Dict]:
    """
    Validate that each route has correct MFVRP structure:
    - Each route = [DEPOT, customers..., DEPOT]
    - Depot appears ONLY at index 0 and last index
    - No depot in the middle of a route
    - No depot-to-depot chaining between routes

    Returns: List of validation results
    """
    validation_results = []

    for route in routes:
        sequence = route.get("sequence", [])
        cluster_id = route.get("cluster_id", "?")

        issues = []

        # Check 1: Route starts with depot (0)
        if len(sequence) < 2:
            issues.append("Route too short (< 2 nodes)")
        elif sequence[0] != 0:
            issues.append(
                f"Route does not start with depot (starts with {sequence[0]})")

        # Check 2: Route ends with depot (0)
        if len(sequence) >= 2 and sequence[-1] != 0:
            issues.append(
                f"Route does not end with depot (ends with {sequence[-1]})")

        # Check 3: No depot in the middle
        middle_depots = [i for i, node in enumerate(
            sequence[1:-1], 1) if node == 0]
        if middle_depots:
            issues.append(
                f"Depot found in middle at positions: {middle_depots}")

        # Check 4: Route has exactly 2 depots
        depot_count = sequence.count(0)
        if depot_count != 2:
            issues.append(f"Expected 2 depots, found {depot_count}")

        validation_results.append({
            "cluster_id": cluster_id,
            "sequence": sequence,
            "valid": len(issues) == 0,
            "issues": issues
        })

    return validation_results


# ============================================================
# COST CALCULATION
# ============================================================

def get_vehicle_data(fleet, vehicle_type):
    """Get vehicle data with fallback for different ID formats."""
    # Try exact match
    if vehicle_type in fleet:
        return fleet[vehicle_type]

    # Try with "Vehicle " prefix
    full_id = f"Vehicle {vehicle_type}"
    if full_id in fleet:
        return fleet[full_id]

    # Try without "Vehicle " prefix
    short_id = vehicle_type.replace("Vehicle ", "")
    if short_id in fleet:
        return fleet[short_id]

    # Raise error if still not found
    raise KeyError(
        f"Vehicle type '{vehicle_type}' not found in fleet. Available: {list(fleet.keys())}")


def compute_costs(routes: List[Dict], dataset: Dict) -> Dict:
    """Compute costs as in Word document."""
    fleet = {f["id"]: f for f in dataset["fleet"]}

    total_fixed = 0.0
    total_variable = 0.0

    cost_breakdown = []

    for route in routes:
        try:
            vehicle = get_vehicle_data(fleet, route["vehicle_type"])
            fixed = vehicle["fixed_cost"]
            variable = vehicle["variable_cost_per_km"] * \
                route["total_distance"]

            cost_breakdown.append({
                "cluster_id": route["cluster_id"],
                "vehicle_type": route["vehicle_type"],
                "fixed_cost": fixed,
                "variable_cost": variable,
                "total_cost": fixed + variable
            })

            total_fixed += fixed
            total_variable += variable
        except KeyError as e:
            print(
                f"⚠️ Error computing cost for route {route.get('cluster_id')}: {e}")
            cost_breakdown.append({
                "cluster_id": route.get("cluster_id"),
                "error": str(e)
            })

    return {
        "total_fixed_cost": total_fixed,
        "total_variable_cost": total_variable,
        "total_cost": total_fixed + total_variable,
        "breakdown": cost_breakdown
    }


# ============================================================
# VALIDATION AGAINST WORD DOCUMENT
# ============================================================

def validate_routes(routes: List[Dict], dataset: Dict) -> List[Dict]:
    """
    Validate routes are mathematically correct (dynamic validation).

    Checks:
    - Route starts and ends at depot (ID 0)
    - All customers in cluster are visited exactly once
    - No duplicate customers
    - Capacity constraints respected (if vehicle assigned)

    Returns list of validation results for each route.
    """
    validation_results = []
    customers_dict = {c["id"]: c for c in dataset["customers"]}

    for route in routes:
        cluster_id = route.get("cluster_id", "?")
        sequence = route.get("sequence", [])
        issues = []

        # Check 1: Route starts with depot
        if not sequence or sequence[0] != 0:
            issues.append("Rute tidak dimulai dari depot")

        # Check 2: Route ends with depot
        if len(sequence) < 2 or sequence[-1] != 0:
            issues.append("Rute tidak berakhir di depot")

        # Check 3: All cluster customers visited
        expected_customers = set(
            route.get("customer_ids", route.get("cluster", {}).get("customer_ids", [])))
        actual_customers = set([n for n in sequence if n != 0])
        missing = expected_customers - actual_customers
        extra = actual_customers - expected_customers

        if missing:
            issues.append(f"Customer tidak terlayani: {missing}")
        if extra:
            issues.append(f"Customer tidak diharapkan: {extra}")

        # Check 4: No duplicates (except depot)
        customer_visits = [n for n in sequence if n != 0]
        if len(customer_visits) != len(set(customer_visits)):
            issues.append("Ada customer yang dikunjungi lebih dari sekali")

        validation_results.append({
            "cluster_id": cluster_id,
            "sequence": sequence,
            "valid": len(issues) == 0,
            "issues": issues
        })

    return validation_results


# ============================================================
# MAIN ACADEMIC REPLAY FUNCTION
# ============================================================

def run_academic_replay(
    user_vehicles: Optional[List[Dict]] = None,
    user_customers: Optional[List[Dict]] = None,
    user_depot: Optional[Dict] = None,
    user_acs_params: Optional[Dict] = None,
    distance_multiplier: float = 1.0
) -> Dict:
    """
    Run the complete academic replay pipeline with DYNAMIC user data.

    Args:
        user_vehicles: List of user-defined vehicles, format:
            [
                {"id": "Truk A", "name": "Truk A", "capacity": 60, "units": 2, 
                 "available_from": "08:00", "available_until": "17:00",
                 "fixed_cost": 50000, "variable_cost_per_km": 1000},
                {"id": "Truk B", "name": "Truk B", "capacity": 100, "units": 1, ...}
            ]
        user_customers: List of customers from Input Titik + Input Data, format:
            [
                {"id": 1, "name": "C1", "x": 2.0, "y": 3.0, "demand": 10,
                 "service_time": 5, "time_window": {"start": "08:00", "end": "12:00"}},
                ...
            ]
        user_depot: Depot data, format:
            {"id": 0, "name": "Depot", "x": 0.0, "y": 0.0,
             "time_window": {"start": "08:00", "end": "17:00"}, "service_time": 0}
        user_acs_params: ACS algorithm parameters, format:
            {"alpha": 1.0, "beta": 2.0, "rho": 0.1, "q0": 0.9, 
             "num_ants": 10, "max_iterations": 50}

    Returns:
        Dict with all iteration logs for display in UI.
    """
    print("=" * 60)
    print("MFVRPTW OPTIMIZATION - Dynamic Input Mode")
    print("=" * 60)

    # ============================================================
    # BUILD DATASET DYNAMICALLY (use defaults if not provided)
    # ============================================================

    # Start with default structure
    dataset = {
        "depot": user_depot if user_depot else ACADEMIC_DATASET["depot"],
        "customers": [],
        "fleet": [],
        "acs_parameters": user_acs_params if user_acs_params else ACADEMIC_DATASET.get("acs_parameters", {
            "alpha": 1.0, "beta": 2.0, "rho": 0.1, "q0": 0.9,
            "num_ants": 10, "max_iterations": 50
        }),
        "objective_weights": ACADEMIC_DATASET.get("objective_weights", {
            "w1_distance": 1.0, "w2_time": 1.0, "w3_tw_violation": 1.0
        })
    }

    # Apply customers (from user or fallback to default)
    if user_customers and len(user_customers) > 0:
        # User provided customers - build from user data!
        dataset["customers"] = []
        for i, c in enumerate(user_customers):
            customer = {
                "id": c.get("id", i + 1),
                "name": c.get("name", f"C{i + 1}"),
                "x": c.get("x", c.get("lng", 0)),
                "y": c.get("y", c.get("lat", 0)),
                "demand": c.get("demand", 0),
                "service_time": c.get("service_time", 10),
                "time_window": c.get("time_window", {
                    "start": c.get("tw_start", "08:00"),
                    "end": c.get("tw_end", "17:00")
                })
            }
            dataset["customers"].append(customer)
        print(
            f"[PRE] Using {len(dataset['customers'])} DYNAMIC customers from user input")
    else:
        # Fallback to default dataset for demo/testing
        dataset["customers"] = deepcopy(ACADEMIC_DATASET["customers"])
        print(
            f"[PRE] Using {len(dataset['customers'])} DEFAULT customers (Word document)")

    # Log ACS params being used
    acs_p = dataset["acs_parameters"]
    print(f"[ACS] α={acs_p.get('alpha', 1)}, β={acs_p.get('beta', 2)}, ρ={acs_p.get('rho', 0.1)}, "
          f"q0={acs_p.get('q0', 0.9)}, ants={acs_p.get('num_ants', 10)}, iter={acs_p.get('max_iterations', 50)}")

    # ============================================================
    # APPLY USER-DEFINED VEHICLES (FULLY DYNAMIC!)
    # ============================================================
    user_vehicle_selection_log = []

    if user_vehicles and len(user_vehicles) > 0:
        print("\n[PRE] Applying User-Defined Vehicles...")

        # Build fleet ONLY from ENABLED user-defined vehicles
        updated_fleet = []
        for uv in user_vehicles:
            vid = uv.get("id", uv.get("name", "Unknown"))
            # Default to enabled if not specified
            is_enabled = uv.get("enabled", True)

            vehicle = {
                "id": vid,
                "name": uv.get("name", vid),
                "capacity": uv.get("capacity", 100),
                "units": uv.get("units", 1),
                "available_from": uv.get("available_from", "08:00"),
                "available_until": uv.get("available_until", "17:00"),
                "fixed_cost": uv.get("fixed_cost", 50000),
                "variable_cost_per_km": uv.get("variable_cost_per_km", 1000)
            }

            if is_enabled:
                updated_fleet.append(vehicle)
                user_vehicle_selection_log.append({
                    "phase": "USER_VEHICLE_SELECTION",
                    "vehicle_id": vid,
                    "vehicle_name": vehicle["name"],
                    "capacity": vehicle["capacity"],
                    "enabled": True,
                    "units": vehicle["units"],
                    "available_from": vehicle["available_from"],
                    "available_until": vehicle["available_until"],
                    "status": f"✅ Aktif ({vehicle['units']} unit, {vehicle['available_from']}–{vehicle['available_until']})"
                })
                print(
                    f"   ✅ Vehicle '{vid}': {vehicle['units']} unit, capacity {vehicle['capacity']}, {vehicle['available_from']}–{vehicle['available_until']}")
            else:
                user_vehicle_selection_log.append({
                    "phase": "USER_VEHICLE_SELECTION",
                    "vehicle_id": vid,
                    "vehicle_name": vehicle["name"],
                    "capacity": vehicle["capacity"],
                    "enabled": False,
                    "units": vehicle["units"],
                    "available_from": vehicle["available_from"],
                    "available_until": vehicle["available_until"],
                    "status": "❌ Tidak aktif (dinonaktifkan oleh user)"
                })
                print(f"   ❌ Vehicle '{vid}': DISABLED by user")

        # Replace fleet with ENABLED user-defined vehicles ONLY
        dataset["fleet"] = updated_fleet
        print(f"   → {len(updated_fleet)} fleet types ACTIVE for routing")

        # Check if any vehicles are active
        if len(updated_fleet) == 0:
            print("   ❌ CRITICAL: All fleets are disabled!")
            return {
                "mode": "ACADEMIC_REPLAY",
                "error": "Semua fleet dinonaktifkan! Aktifkan minimal 1 fleet di tab 'Input Data'.",
                "user_vehicle_selection": user_vehicle_selection_log,
                "vehicle_availability": [],
                "available_vehicles": [],
                "dataset": dataset,
                "clusters": [],
                "routes": [],
                "costs": {"total_cost": 0},
                "iteration_logs": []
            }
    else:
        # NO USER VEHICLES = CANNOT RUN!
        print("\n[PRE] ❌ CRITICAL: No fleet defined by user!")
        print("   User MUST add fleet in Input Data tab before running.")
        return {
            "mode": "ACADEMIC_REPLAY",
            "error": "Tidak ada fleet! Silakan tambah fleet di tab 'Input Data' terlebih dahulu.",
            "user_vehicle_selection": [],
            "vehicle_availability": [],
            "available_vehicles": [],
            "dataset": dataset,
            "clusters": [],
            "routes": [],
            "costs": {"total_cost": 0},
            "iteration_logs": []
        }

    distance_matrix = build_distance_matrix(
        dataset, multiplier=distance_multiplier)

    all_logs = []
    # Add user selection logs FIRST
    all_logs.extend(user_vehicle_selection_log)

    # ============================================================
    # 0. FLEET AVAILABILITY CHECK
    # ============================================================
    print("\n[0/5] Checking Fleet Availability...")
    fleet = dataset["fleet"]

    if len(fleet) == 0:
        print("   ❌ CRITICAL: No fleet selected by user! Cannot proceed.")
        return {
            "mode": "ACADEMIC_REPLAY",
            "error": "No vehicles selected by user",
            "user_vehicle_selection": user_vehicle_selection_log,
            "vehicle_availability": [],
            "available_vehicles": [],
            "dataset": dataset,
            "clusters": [],
            "routes": [],
            "costs": {"total_cost": 0},
            "iteration_logs": all_logs
        }

    availability_status = get_vehicle_availability_status(fleet)
    available_fleet = get_available_vehicles(fleet)

    # Log availability status
    for status in availability_status:
        all_logs.append({
            "phase": "VEHICLE_AVAILABILITY",
            "vehicle_id": status["vehicle_id"],
            "capacity": status["capacity"],
            "units": status.get("units", 1),
            "available": status["available"],
            "time_window": status.get("time_window", "Not Set"),
            "status": status["status"]
        })

        icon = "✅" if status["available"] else "❌"
        print(f"   {icon} Fleet {status['vehicle_id']}: {status['status']}")

    print(
        f"   → {len(available_fleet)}/{len(fleet)} fleet types available for routing")

    if len(available_fleet) == 0:
        print("   ❌ CRITICAL: No fleet available! Cannot proceed with routing.")
        return {
            "mode": "ACADEMIC_REPLAY",
            "error": "No vehicles available",
            "vehicle_availability": availability_status,
            "dataset": dataset,
            "clusters": [],
            "routes": [],
            "costs": {"total_cost": 0},
            "iteration_logs": all_logs
        }

    # 1. SWEEP CLUSTERING
    print("\n[1/5] Running SWEEP algorithm...")
    clusters, sweep_logs = academic_sweep(dataset)
    all_logs.extend(sweep_logs)
    print(f"   Formed {len(clusters)} clusters")

    # 2. NEAREST NEIGHBOR (TIME-WINDOW AWARE)
    print("\n[2/5] Running Nearest Neighbor (TW-Aware)...")
    initial_routes = []
    all_unassigned = []  # Customers rejected due to TW violation

    for cluster in clusters:
        route, nn_logs = academic_nearest_neighbor(
            cluster, dataset, distance_matrix)
        initial_routes.append(route)
        all_logs.extend(nn_logs)
        # SUMMARY LOG for NN
        all_logs.append({
            "phase": "NN_SUMMARY",
            "cluster_id": cluster['cluster_id'],
            "route_sequence": "-".join(map(str, route['sequence'])),
            "total_distance": route['total_distance'],
            "vehicle_type": route.get("vehicle_type", "Belum Ditentukan")
        })

        # Track unassigned customers
        unassigned = route.get("unassigned_customers", [])
        if unassigned:
            all_unassigned.extend(unassigned)
            print(
                f"   Cluster {cluster['cluster_id']}: {route['sequence']} (dist={route['total_distance']}) ⚠️ {len(unassigned)} unassigned")
        else:
            print(
                f"   Cluster {cluster['cluster_id']}: {route['sequence']} (dist={route['total_distance']})")

    if all_unassigned:
        print(
            f"   ⚠️ Total unassigned customers (TW hard constraint): {all_unassigned}")

    # 3. ACS
    print("\n[3/5] Running ACS...")
    acs_routes = []
    for i, cluster in enumerate(clusters):
        route, acs_logs = academic_acs_cluster(
            cluster, dataset, distance_matrix, initial_routes[i])
        acs_routes.append(route)
        all_logs.extend(acs_logs)
        # SUMMARY LOG for ACS
        all_logs.append({
            "phase": "ACS_SUMMARY",
            "cluster_id": cluster['cluster_id'],
            "route_sequence": "-".join(map(str, route['sequence'])),
            "total_distance": route['total_distance'],
            "vehicle_type": route.get("vehicle_type", "Belum Ditentukan")
        })
        print(
            f"   Cluster {cluster['cluster_id']}: {route['sequence']} (dist={route['total_distance']})")

    # 4. RVND
    print("\n[4/5] Running RVND...")
    final_routes, rvnd_logs = academic_rvnd(
        acs_routes, dataset, distance_matrix)
    all_logs.extend(rvnd_logs)
    # SUMMARY LOG for RVND
    for route in final_routes:
        all_logs.append({
            "phase": "RVND_SUMMARY",
            "cluster_id": route['cluster_id'],
            "route_sequence": "-".join(map(str, route['sequence'])),
            "total_distance": route['total_distance'],
            "vehicle_type": route.get("vehicle_type", "Belum Ditentukan")
        })
    for route in final_routes:
        print(
            f"   Cluster {route['cluster_id']}: {route['sequence']} (dist={route['total_distance']})")

    # 5. FLEET REASSIGNMENT
    print("\n[5/5] Reassigning fleets...")
    final_routes, vehicle_logs = reassign_vehicles(final_routes, dataset)
    all_logs.extend(vehicle_logs)

    # COST CALCULATION
    costs = compute_costs(final_routes, dataset)
    print(f"\n   Total Cost: Rp {costs['total_cost']:,.0f}")

    # TIME WINDOW SUMMARY
    print("\n" + "=" * 60)
    print("FLEET TIME WINDOW SUMMARY (Soft Constraints)")
    print("=" * 60)
    total_violations = 0
    total_wait = 0
    for route in final_routes:
        tw_viol = route.get("total_tw_violation", 0)
        wait = route.get("total_wait_time", 0)
        total_violations += tw_viol
        total_wait += wait

        violations_detail = route.get("tw_violations_detail", [])
        if violations_detail:
            print(
                f"   Cluster {route['cluster_id']}: ⚠️ {len(violations_detail)} violation(s), {tw_viol:.1f} min total")
            for v in violations_detail:
                print(
                    f"      - Customer {v['customer_id']}: arrived {v['arrival']:.1f}, TW ends {v['tw_end']:.0f}, late by {v['violation_minutes']:.1f} min")
        else:
            print(
                f"   Cluster {route['cluster_id']}: ✅ No violations (wait time: {wait:.1f} min)")

    print(f"\n   Total Wait Time: {total_wait:.1f} min")
    print(f"   Total TW Violations: {total_violations:.1f} min")
    if total_violations > 0:
        print(
            "   ℹ️ Time windows are SOFT constraints in ACS/RVND - solutions NOT rejected")

    # UNASSIGNED CUSTOMERS SUMMARY (from NN hard constraint)
    if all_unassigned:
        print("\n" + "=" * 60)
        print("UNASSIGNED CUSTOMERS (NN Hard Constraint)")
        print("=" * 60)
        print(
            f"   {len(all_unassigned)} customer(s) rejected by NN due to TW violation")
        print(f"   Customer IDs: {all_unassigned}")
        print("   ℹ️ In NN phase, TW is HARD constraint: arrival > TW_end = REJECT")

    # ROUTE STRUCTURE VALIDATION (CRITICAL)
    print("\n" + "=" * 60)
    print("ROUTE STRUCTURE VALIDATION")
    print("=" * 60)
    structure_validation = validate_route_structure(final_routes)

    structure_valid = True
    for v in structure_validation:
        status = "✅ VALID" if v["valid"] else "❌ INVALID"
        print(f"   Cluster {v['cluster_id']}: {status}")
        if not v["valid"]:
            structure_valid = False
            for issue in v["issues"]:
                print(f"      ⚠️ {issue}")

    if structure_valid:
        print("   ✅ All routes have correct MFVRP structure")
    else:
        print("   ❌ CRITICAL: Route structure validation FAILED")

    # VALIDASI RUTE (Dynamic)
    print("\n" + "=" * 60)
    print("VALIDASI RUTE (Otomatis)")
    print("=" * 60)
    validation = validate_routes(final_routes, dataset)

    all_valid = True
    for v in validation:
        status = "✅ VALID" if v["valid"] else "❌ INVALID"
        print(f"   Cluster {v['cluster_id']}: {status}")
        if not v["valid"]:
            all_valid = False
            for issue in v.get("issues", []):
                print(f"      ⚠️ {issue}")

    # Save results
    output = {
        "mode": "ACADEMIC_REPLAY",
        "user_vehicle_selection": user_vehicle_selection_log,
        "vehicle_availability": availability_status,
        "available_vehicles": [v["id"] for v in available_fleet],
        "dataset": dataset,
        "clusters": clusters,
        "routes": final_routes,
        "unassigned_customers": all_unassigned,
        "costs": costs,
        "structure_validation": structure_validation,
        "structure_valid": structure_valid,
        "validation": validation,
        "all_valid": all_valid and structure_valid,
        "iteration_logs": all_logs
    }

    with ACADEMIC_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {ACADEMIC_OUTPUT_PATH}")

    return output


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if MODE == "ACADEMIC_REPLAY":
        result = run_academic_replay()
    else:
        print("MODE is set to OPTIMIZATION. Use normal pipeline instead.")
