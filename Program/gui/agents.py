from __future__ import annotations

import json
import shutil
import sys
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Any, List, Tuple

from path_helper import DATA_DIR, PROGRAM_DIR

PARSED_INSTANCE = DATA_DIR / "parsed_instance.json"
PARSED_DISTANCE = DATA_DIR / "parsed_distance.json"
FINAL_SOLUTION = DATA_DIR / "final_solution.json"
ACS_ROUTES = DATA_DIR / "acs_routes.json"
RVND_ROUTES = DATA_DIR / "rvnd_routes.json"


def _is_square_matrix(mat: List[List[Any]]) -> bool:
    if not isinstance(mat, list) or not mat:
        return False
    n = len(mat)
    return all(isinstance(row, list) and len(row) == n for row in mat)


def validate_state(state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Strict validation according to rules in the coordinator spec.

    state is expected to have keys: points (with depots/customers), inputData (vehicleCapacity, iterations, customerDemand, distanceMatrix)
    Returns (valid, list_of_errors).
    """
    errors: List[str] = []

    points = state.get("points", {})
    depots = points.get("depots", [])
    customers = points.get("customers", [])

    if len(depots) < 1:
        errors.append("Harus ada minimal 1 depot")
    if len(customers) < 1:
        errors.append("Harus ada minimal 1 customer")

    inputData = state.get("inputData", {})
    demands = inputData.get("customerDemand", [])
    if len(demands) != len(customers):
        errors.append("Jumlah nilai permintaan tidak sama dengan jumlah customer")
    else:
        for i, d in enumerate(demands):
            try:
                if float(d) < 0:
                    errors.append(f"Demand untuk customer index {i} bernilai negatif")
            except Exception:
                errors.append(f"Demand untuk customer index {i} bukan numerik")

    dist = inputData.get("distanceMatrix")
    if dist is None:
        errors.append("Matriks jarak belum disediakan")
    else:
        if not _is_square_matrix(dist):
            errors.append("Matriks jarak harus berbentuk matriks bujur sangkar")
        else:
            n = len(dist)
            # Check non-diagonal values only (diagonal is guaranteed 0 by UI)
            for i in range(n):
                for j in range(n):
                    # Skip diagonal (i == j) - already guaranteed to be 0
                    if i == j:
                        continue
                    try:
                        val = float(dist[i][j])
                        if val < 0:
                            errors.append(f"Matriks jarak mengandung nilai negatif pada ({i},{j})")
                    except (ValueError, TypeError):
                        errors.append(f"Nilai pada ({i},{j}) bukan numerik")
            
            # Enforce symmetry on non-diagonal values
            for i in range(n):
                for j in range(i + 1, n):
                    try:
                        a = float(dist[i][j])
                        b = float(dist[j][i])
                        if abs(a - b) > 1e-6:
                            errors.append(f"Matriks tidak simetris pada ({i},{j}) vs ({j},{i})")
                    except (ValueError, TypeError):
                        pass  # Already reported above

    # total demand vs fleet capacity
    fleets = state.get("user_vehicles", [])
    
    # Fallback to PARSED_INSTANCE if state is not provided
    if not fleets:
        try:
            if PARSED_INSTANCE.exists():
                with PARSED_INSTANCE.open("r", encoding="utf-8") as fh:
                    template = json.load(fh)
                fleets = template.get("fleet", [])
        except Exception:
            pass

    if fleets:
        active_fleets = [f for f in fleets if f.get("enabled", True)]
        total_capacity = sum((float(f.get("capacity", 0)) * int(f.get("units", 0))) for f in active_fleets)
        total_demand = sum(float(x) for x in demands) if demands else 0
        
        if not active_fleets:
            errors.append("Tidak ada armada aktif yang dipilih")
        elif total_demand > total_capacity + 1e-9:
            errors.append(f"Total demand ({total_demand:,.0f}) melebihi total kapasitas armada ({total_capacity:,.0f})")
    else:
        errors.append("Data armada tidak ditemukan untuk validasi kapasitas")

    return (len(errors) == 0), errors


def _write_parsed_instance(state: Dict[str, Any]) -> None:
    """Build parsed_instance.json from existing template and user state."""
    # Load template to get fleet and acs params
    with PARSED_INSTANCE.open("r", encoding="utf-8") as fh:
        template = json.load(fh)

    # Build new instance preserving fleet and acs params
    new_inst = template.copy()
    points = state.get("points", {})
    depots = points.get("depots", [])
    customers = points.get("customers", [])

    # Map depots/customers into template format
    if depots:
        new_inst["depot"] = depots[0]
    else:
        new_inst["depot"] = template.get("depot", {})

    # build customers list using demands from inputData if provided
    demands = state.get("inputData", {}).get("customerDemand", [])
    new_customers = []
    for idx, cust in enumerate(customers):
        nc = {
            "id": int(cust.get("id", idx + 1)),
            "name": cust.get("name", f"Cust{idx+1}"),
            "x": float(cust.get("x", 0)),
            "y": float(cust.get("y", 0)),
            "demand": float(demands[idx]) if idx < len(demands) else float(cust.get("demand", 0)),
            "time_window": cust.get("time_window", template.get("customers", [{}])[0].get("time_window", {})),
            "service_time": cust.get("service_time", 0),
        }
        new_customers.append(nc)

    new_inst["customers"] = new_customers

    # write
    with PARSED_INSTANCE.open("w", encoding="utf-8") as fh:
        json.dump(new_inst, fh, indent=2)


def _write_parsed_distance(state: Dict[str, Any]) -> None:
    """Write parsed_distance.json using provided distance matrix and nodes."""
    points = state.get("points", {})
    depots = points.get("depots", [])
    customers = points.get("customers", [])
    labels = []
    nodes = []
    # order: depot(s) then customers — pipeline expects first node as depot
    if depots:
        for d in depots:
            labels.append(str(int(d.get("id", 0))))
            nodes.append({"id": int(d.get("id", 0)), "name": d.get("name", "Depot")})
    for c in customers:
        labels.append(str(int(c.get("id", 0))))
        nodes.append({"id": int(c.get("id", 0)), "name": c.get("name", "Customer")})

    dist = state.get("inputData", {}).get("distanceMatrix", [])
    travel = [row[:] for row in dist]

    payload = {
        "nodes": nodes,
        "distance_matrix": dist,
        "travel_time_matrix": travel,
        "units": {"distance": "km", "time": "minutes"}
    }

    with PARSED_DISTANCE.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _run_module_main(module_name: str, progress_callback=None) -> None:
    """Import and run a pipeline module's main() function directly.

    Ini adalah pengganti subprocess.Popen — modul dijalankan langsung
    dalam proses yang sama, sehingga kompatibel dengan PyInstaller.
    """
    # Ensure Program/ dir is in sys.path for imports
    program_str = str(PROGRAM_DIR)
    if program_str not in sys.path:
        sys.path.insert(0, program_str)

    # Import the module dynamically
    import importlib
    module = importlib.import_module(module_name)

    # Capture stdout to feed progress_callback
    if progress_callback:
        buf = io.StringIO()
        with redirect_stdout(buf):
            module.main()
        # Feed captured output to progress callback
        for line in buf.getvalue().splitlines():
            try:
                progress_callback(line.rstrip())
            except Exception:
                pass
    else:
        module.main()


def run_pipeline(state: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
    """Run the deterministic pipeline (sweep -> nn -> acs -> rvnd -> final_integration).

    This function overwrites `parsed_instance.json` and `parsed_distance.json` with
    user-provided inputs, runs the module functions directly (no subprocess!),
    and returns the final_solution.json payload.
    Backups of original parsed files are kept and restored on error.
    """
    # Backup
    inst_bak = PARSED_INSTANCE.with_suffix('.json.bak')
    dist_bak = PARSED_DISTANCE.with_suffix('.json.bak')
    try:
        if PARSED_INSTANCE.exists():
            shutil.copy2(PARSED_INSTANCE, inst_bak)
        if PARSED_DISTANCE.exists():
            shutil.copy2(PARSED_DISTANCE, dist_bak)

        # write instance and distance
        _write_parsed_instance(state)
        _write_parsed_distance(state)

        # Run each pipeline step as direct function call (NO subprocess!)
        pipeline_modules = [
            "sweep_nn",
            "acs_solver",
            "rvnd",
            "final_integration",
        ]
        for module_name in pipeline_modules:
            _run_module_main(module_name, progress_callback)

        # read final solution
        with FINAL_SOLUTION.open("r", encoding="utf-8") as fh:
            result = json.load(fh)
        
        # Add iteration logs from ACS and RVND to result
        try:
            if ACS_ROUTES.exists():
                with ACS_ROUTES.open("r", encoding="utf-8") as fh:
                    acs_data = json.load(fh)
                    result["acs_data"] = {
                        "iteration_logs": acs_data.get("iteration_logs", [])
                    }
        except Exception:
            result["acs_data"] = {"iteration_logs": []}
        
        try:
            if RVND_ROUTES.exists():
                with RVND_ROUTES.open("r", encoding="utf-8") as fh:
                    rvnd_data = json.load(fh)
                    result["rvnd_data"] = {
                        "iteration_logs": rvnd_data.get("iteration_logs", [])
                    }
        except Exception:
            result["rvnd_data"] = {"iteration_logs": []}

        return result
    except Exception as e:
        # restore backups
        if inst_bak.exists():
            shutil.move(str(inst_bak), str(PARSED_INSTANCE))
        if dist_bak.exists():
            shutil.move(str(dist_bak), str(PARSED_DISTANCE))
        raise RuntimeError(f"Pipeline failed: {e}") from e
