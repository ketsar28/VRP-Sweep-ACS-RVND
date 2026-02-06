
import json
import math

# Data Definitions
coords = [
    (0, 0),     # 0 Depot
    (8, 12),    # 1
    (20, 15),   # 2
    (5, 25),    # 3
    (24, 4),    # 4
    (28, 18),   # 5
    (14, 6),    # 6
    (18, 24),   # 7
    (32, 8),    # 8
    (10, 22),   # 9
    (36, 22)    # 10
]

demands = [0, 51, 17, 12, 40, 68, 18, 24, 47, 28, 43]

# Format: (Start, End, Service)
time_windows = [
    ("08:30", "17:00", 0),   # 0
    ("08:20", "10:00", 25),  # 1
    ("07:00", "09:45", 9),   # 2
    ("09:20", "12:00", 6),   # 3
    ("08:30", "11:30", 20),  # 4
    ("09:00", "11:45", 34),  # 5
    ("08:30", "11:00", 9),   # 6
    ("08:20", "11:00", 12),  # 7
    ("08:30", "13:00", 24),  # 8
    ("09:00", "11:00", 15),  # 9
    ("09:00", "12:30", 23)   # 10
]

names = [
    "Depot (Gudang)", "RSUD Kota Malang", "Puskesmas Dinoyo", "Puskesmas Mojolar",
    "Puskesmas Arjowinangun", "RS Lavalette", "Puskesmas Sukun", "Klinik Kahuripan",
    "RS Mitra Sehat", "Puskesmas Lowokwaru", "RS Panti Nirmala"
]

# Calculate Distance Matrix


def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


matrix = []
for i in range(len(coords)):
    row = []
    for j in range(len(coords)):
        d = dist(coords[i], coords[j])
        row.append(round(d, 5))
    matrix.append(row)

# Structure
data = {
    "points": {
        "depots": [],
        "customers": []
    },
    "inputData": {
        "customerTimeWindows": [],
        "customerDemand": [],
        "distanceMatrix": matrix
    },
    "user_vehicles": [],
    "acs_params": {
        "alpha": 0.5,
        "beta": 2.0,
        "rho": 0.2,
        "q0": 0.85,
        "num_ants": 1,  # Based on text "Banyak Semut 1"
        "max_iterations": 50
    },
    "kapasitas_kendaraan": 100,  # Legacy
    "iterasi": 50
}

# Fill Points and InputData
# Depot
data["points"]["depots"].append({
    "id": 0,
    "name": names[0],
    "x": coords[0][0],
    "y": coords[0][1],
    "time_window": {"start": time_windows[0][0], "end": time_windows[0][1]},
    "service_time": time_windows[0][2]
})

# Customers
for i in range(1, 11):
    c = {
        "id": i,
        "name": names[i],
        "x": coords[i][0],
        "y": coords[i][1],
        "demand": float(demands[i]),
        "time_window": {"start": time_windows[i][0], "end": time_windows[i][1]},
        "service_time": time_windows[i][2]
    }
    data["points"]["customers"].append(c)

    # inputData arrays (only for customers)
    data["inputData"]["customerTimeWindows"].append({
        "demand": float(demands[i]),
        "tw_start": time_windows[i][0],
        "tw_end": time_windows[i][1],
        "service_time": time_windows[i][2]
    })
    data["inputData"]["customerDemand"].append(float(demands[i]))

# Vehicles
# 1. Fleet A
data["user_vehicles"].append({
    "id": "Vehicle A", "name": "Vehicle A", "capacity": 60, "units": 2,
    "available_from": "08:00", "available_until": "17:00", "enabled": True,
    "fixed_cost": 50000, "variable_cost_per_km": 1000
})
# 2. Fleet B
data["user_vehicles"].append({
    "id": "Vehicle B", "name": "Vehicle B", "capacity": 100, "units": 2,
    "available_from": "08:00", "available_until": "17:00", "enabled": True,
    "fixed_cost": 60000, "variable_cost_per_km": 1000
})
# 3. Fleet C
data["user_vehicles"].append({
    "id": "Vehicle C", "name": "Vehicle C", "capacity": 150, "units": 1,
    "available_from": "08:00", "available_until": "17:00", "enabled": True,
    "fixed_cost": 70000, "variable_cost_per_km": 1000
})

# Write JSON
with open('d:/PORTFOLIO/NUR/Route-Optimization/Program/gui/data/progress-acs1.json', 'w') as f:
    json.dump(data, f, indent=2)

print("JSON generated successfully.")
