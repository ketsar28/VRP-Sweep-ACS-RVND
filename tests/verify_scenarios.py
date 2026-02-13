import sys
from pathlib import Path
import json

# Add Program/gui to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "Program" / "gui"))

import agents

def test_scenario(filename):
    print(f"\nTesting {filename}...")
    with open(f"Program/data/samples/{filename}", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Simulate the state passed from UI
    # Note: agents.py now expects 'user_vehicles' in the state
    state = {
        "points": data.get("points", {}),
        "inputData": data.get("inputData", {}),
        "user_vehicles": data.get("user_vehicles", [])
    }
    
    valid, errors = agents.validate_state(state)
    if valid:
        print(f"PASS: Validation SUCCESS for {filename}")
    else:
        print(f"FAIL: Validation FAILED for {filename}")
        for error in errors:
            print(f"  - {error}")

if __name__ == "__main__":
    test_scenario("scenario_3_katering_20.json")
    test_scenario("scenario_4_limbah_15.json")
