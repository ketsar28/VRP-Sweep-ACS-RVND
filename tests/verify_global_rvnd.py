
import sys
import os


# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Program')))

from academic_replay import academic_rvnd, apply_inter_neighborhood

def test_global_rvnd():
    print("Testing Global RVND Logic...")

    # Mock Data
    dataset = {
        "fleet": [
            {"id": "Box Kecil", "capacity": 100},
            {"id": "Engkel", "capacity": 200}
        ],
        "customers": [
            {"id": 1, "demand": 10}, {"id": 2, "demand": 10},
            {"id": 3, "demand": 10}, {"id": 4, "demand": 10}
        ]
    }
    
    # Mock Distance Matrix (4x4)
    # 0 is Depot
    distance_matrix = [
        [0, 10, 10, 20, 20],
        [10, 0, 5, 25, 25],
        [10, 5, 0, 25, 25],
        [20, 25, 25, 0, 5],
        [20, 25, 25, 5, 0]
    ]

    # Two initial routes (Cluster 1 and Cluster 2)
    # Route 1: 0 -> 1 -> 2 -> 0 (Total dist approx 25)
    # Route 2: 0 -> 3 -> 4 -> 0 (Total dist approx 45)
    routes = [
        {
            "cluster_id": 1,
            "vehicle_type": "Box Kecil",
            "sequence": [0, 1, 2, 0],
            "total_distance": 25,
            "total_demand": 20
        },
        {
            "cluster_id": 2,
            "vehicle_type": "Box Kecil",
            "sequence": [0, 3, 4, 0],
            "total_distance": 45,
            "total_demand": 20
        }
    ]

    print("Initial Routes:")
    for r in routes:
        print(f"Cluster {r['cluster_id']}: {r['sequence']}")

    # Apply Global RVND (Mocking academic_rvnd call)
    # We specifically want to test if apply_inter_neighborhood generates candidates spanning these routes
    
    print("\n[Test 1] Checking Candidate Generation for Global Swap...")
    
    # We'll call apply_inter_neighborhood directly to see candidates
    # Using 'swap_1_1' 
    fleet_dict = {f["id"]: f for f in dataset["fleet"]}
    
    result = apply_inter_neighborhood(
        "swap_1_1", 
        routes, 
        dataset, 
        distance_matrix, 
        fleet_dict
    )
    
    candidates = result.get("candidates", [])
    print(f"Generated {len(candidates)} candidates.")
    
    has_cross_cluster_candidate = False
    for c in candidates:
        print(f"  - Type: {c['type']}, Routes: {c['routes']}, Detail: {c['detail']}, Feasible: {c['feasible']}")
        # Check if candidate involves Route 1 and Route 2
        if c['routes'] == (1, 2):
            has_cross_cluster_candidate = True
            
    if has_cross_cluster_candidate:
        print("✅ SUCCESS: Found candidates spanning across clusters (Global Optimization verified).")
    else:
        print("❌ FAILURE: No cross-cluster candidates found.")
        
    # Check logs for "routes_snapshot" structure
    print("\n[Test 2] Running full academic_rvnd simulation...")
    _, logs = academic_rvnd(routes, dataset, distance_matrix)
    
    if logs:
        first_log = logs[0]
        snapshot = first_log.get("routes_snapshot", [])
        print(f"Log Snapshot Routes Count: {len(snapshot)}")
        if len(snapshot) == 2:
            print("✅ SUCCESS: Logs capture all global routes.")
        else:
            print(f"❌ FAILURE: Expected 2 routes in snapshot, found {len(snapshot)}.")
    else:
        print("⚠️ Warning: No logs returned (maybe no improvement found).")

if __name__ == "__main__":
    test_global_rvnd()
