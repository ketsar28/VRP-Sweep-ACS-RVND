from __future__ import annotations
import agents

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent gui directory to path for agents import
_gui_dir = Path(__file__).resolve().parent.parent
if str(_gui_dir) not in sys.path:
    sys.path.insert(0, str(_gui_dir))

try:
    from utils import save_to_autosave
except ImportError:
    # Jika utils tidak ditemukan, dibuat fungsi dummy
    def save_to_autosave(): pass


def _update_vehicle_field(idx: int, field: str, value) -> None:
    """Helper to update a vehicle field in session state."""
    if idx < len(st.session_state["user_vehicles"]):
        st.session_state["user_vehicles"][idx][field] = value
        # Also update id if name changes
        if field == "name":
            st.session_state["user_vehicles"][idx]["id"] = value


def _get_next_vehicle_letter() -> str:
    """Mencari huruf selanjutnya untuk ID armada (A, B, C...)"""
    user_vehicles = st.session_state.get("user_vehicles", [])

    # Get all used letters
    used_letters = set()
    for v in user_vehicles:
        name = v.get("name", "")
        if name.startswith("Fleet ") and len(name) > 6:
            letter = name[6:]  # Get letter after "Fleet "
            if len(letter) == 1 and letter.isalpha():
                used_letters.add(letter.upper())

    # Find next available letter
    for i in range(26):
        letter = chr(ord('A') + i)
        if letter not in used_letters:
            return letter

    # If all letters used, use AA, AB, etc.
    return f"A{len(user_vehicles) + 1}"


def _get_default_capacity_for_letter(letter: str) -> int:
    """Get default capacity based on vehicle letter."""
    # Default capacities: A=60, B=100, C=150, D+=200
    defaults = {'A': 60, 'B': 100, 'C': 150}
    if letter in defaults:
        return defaults[letter]
    return 200  # Default for D onwards


def render_input_data() -> None:
    """Render tab Input Data. Di sini saya mengatur kapasitas, iterasi, demand, dll."""


    # Initialize session state
    if "inputData" not in st.session_state:
        st.session_state["inputData"] = {}
    if "iterasi" not in st.session_state:
        st.session_state["iterasi"] = 2
    if "last_loaded_file" not in st.session_state:
        st.session_state["last_loaded_file"] = None
    if "distanceMatrix_size" not in st.session_state:
        st.session_state["distanceMatrix_size"] = 0

    # Initialize DYNAMIC vehicle list (user-defined, no defaults!)
    if "user_vehicles" not in st.session_state:
        # Empty by default - user MUST add vehicles
        st.session_state["user_vehicles"] = []

    inputData = st.session_state["inputData"]

    # ===== SECTION 1: Kendaraan User-Defined (DYNAMIC) =====
    with st.container():
        st.subheader("🚛 Armada Kendaraan")
        user_vehicles = st.session_state["user_vehicles"]

        # Migration: Rename "Vehicle" to "Fleet" if present
        for v in user_vehicles:
            if v.get("name", "").startswith("Vehicle "):
                v["name"] = v["name"].replace("Vehicle ", "Fleet ")
                if v.get("id", "").startswith("Vehicle "):
                    v["id"] = v["id"].replace("Vehicle ", "Fleet ")

        # Display existing vehicles
        if user_vehicles:
            # Card Layout for Vehicles (Desktop & Mobile friendly)
            # No header row needed for card layout
            for idx, vehicle in enumerate(user_vehicles):
                vehicle_name = vehicle.get(
                    "name", f"Fleet {chr(ord('A') + idx)}")
                capacity = vehicle.get("capacity", 100)

                # Create card container for each vehicle
                with st.container(border=True):
                    # --- ROW 1: Title Only (Clean) ---
                    st.markdown(f"#### 🚛 {vehicle_name}")
                    
                    # --- ROW 2: Inputs (Grid) ---
                    # Capacity & Units
                    c_row2_1, c_row2_2 = st.columns(2)
                    with c_row2_1:
                        cap_val = st.number_input(
                            "Kapasitas",
                            min_value=1,
                            value=capacity,
                            key=f"veh_cap_{idx}"
                        )
                        st.session_state["user_vehicles"][idx]["capacity"] = cap_val

                    with c_row2_2:
                        units_val = st.number_input(
                            "Jumlah Unit",
                            min_value=1, max_value=10,
                            value=vehicle.get("units", 2),
                            key=f"veh_units_{idx}"
                        )
                        st.session_state["user_vehicles"][idx]["units"] = units_val

                    # Time Window
                    c_row3_1, c_row3_2 = st.columns(2)
                    with c_row3_1:
                        from_val = st.text_input(
                            "Jam Mulai",
                            value=vehicle.get("available_from", "08:00"),
                            key=f"veh_from_{idx}"
                        )
                        st.session_state["user_vehicles"][idx]["available_from"] = from_val

                    with c_row3_2:
                        until_val = st.text_input(
                            "Jam Selesai",
                            value=vehicle.get("available_until", "17:00"),
                            key=f"veh_until_{idx}"
                        )
                        st.session_state["user_vehicles"][idx]["available_until"] = until_val

                    # Costs
                    c_row4_1, c_row4_2 = st.columns(2)
                    with c_row4_1:
                        fixed_cost = st.number_input(
                            "Fixed Cost",
                            min_value=0,
                            value=vehicle.get("fixed_cost", 50000),
                            step=1000,
                            key=f"veh_fixed_{idx}"
                        )
                        st.session_state["user_vehicles"][idx]["fixed_cost"] = fixed_cost

                    with c_row4_2:
                        var_cost = st.number_input(
                            "Cost/km",
                            min_value=0,
                            value=vehicle.get("variable_cost_per_km", 1000),
                            step=100,
                            key=f"veh_var_{idx}"
                        )
                        st.session_state["user_vehicles"][idx]["variable_cost_per_km"] = var_cost

                    # --- ROW 3: Footer Actions (Toggle Active | Delete) ---
                    st.divider()
                    c_act_1, c_act_2 = st.columns([0.8, 0.2], vertical_alignment="center")
                    
                    with c_act_1:
                        enabled = st.toggle(
                            "Status: Aktif",
                            value=vehicle.get("enabled", True),
                            key=f"veh_enabled_{idx}"
                        )
                        st.session_state["user_vehicles"][idx]["enabled"] = enabled

                    with c_act_2:
                         if st.button("🗑️ Hapus", key=f"del_veh_{idx}", help=f"Hapus {vehicle_name}"):
                            st.session_state["user_vehicles"].pop(idx)
                            save_to_autosave()
                            st.rerun()
                    
                    save_to_autosave()
        else:
            st.warning(
                "⚠️ Belum ada rute optimasi. Silakan klik tombol di bawah untuk menjalankan proses optimasi.")

        if st.button("➕ Tambah Kendaraan Baru", key="btn_add_vehicle", type="primary"):
            next_letter = _get_next_vehicle_letter()
            new_vehicle = {
                "id": f"Fleet {next_letter}",
                "name": f"Fleet {next_letter}",
                "capacity": _get_default_capacity_for_letter(next_letter),
                "units": 2,
                "available_from": "08:00",
                "available_until": "17:00",
                "enabled": True,
                "fixed_cost": 50000,
                "variable_cost_per_km": 1000
            }
            st.session_state["user_vehicles"].append(new_vehicle)
            save_to_autosave()
            st.rerun()

        # Summary of active vehicles
        st.markdown("---")
        if user_vehicles:
            active_vehicles = [
                v for v in user_vehicles if v.get("enabled", True)]
            inactive_vehicles = [
                v for v in user_vehicles if not v.get("enabled", True)]

            if active_vehicles:
                active_summary = ", ".join([
                    f"**{v['name']}**: {v['units']} unit ({v['available_from']}–{v['available_until']})"
                    for v in active_vehicles
                ])
                st.success(f"✅ Armada aktif: {active_summary}")

            if inactive_vehicles:
                inactive_names = ", ".join(
                    [f"**{v['name']}**" for v in inactive_vehicles])
                st.warning(f"❌ Kendaraan tidak aktif: {inactive_names}")

            if not active_vehicles:
                st.error(
                    "⚠️ Belum ada armada yang aktif. Minimal aktifin 1 ya biar bisa running.")
        else:
            st.info(
                "ℹ️ Tidak bisa menjalankan optimasi tanpa armada. Tambahkan unit armada baru di bawah.")

    st.divider()

    # ===== SECTION 2: ACS Parameters =====
    with st.expander("⚙️ Parameter ACS (Ant Colony System)", expanded=True):
        # Initialize ACS params in session state
        if "acs_params" not in st.session_state:
            st.session_state["acs_params"] = {
                "alpha": 1.0,      # Pheromone importance
                "beta": 2.0,       # Heuristic importance
                "rho": 0.1,        # Evaporation rate
                "q0": 0.9,         # Exploitation probability
                "num_ants": 10,    # Number of ants
                "max_iterations": 50  # Max iterations
            }

        acs_params = st.session_state["acs_params"]

        # Create columns for parameter inputs
        col1, col2, col3 = st.columns(3)

        with col1:
            alpha = st.number_input(
                "Alpha (α) - Bobot Pheromone",
                min_value=0.1, max_value=5.0, step=0.1,
                value=float(acs_params.get("alpha", 1.0)),
                help="Pengaruh pheromone terhadap pemilihan rute (default: 1.0)"
            )
            acs_params["alpha"] = alpha

            beta = st.number_input(
                "Beta (β) - Bobot Heuristik",
                min_value=0.1, max_value=10.0, step=0.1,
                value=float(acs_params.get("beta", 2.0)),
                help="Pengaruh jarak (visibility) terhadap pemilihan rute (default: 2.0)"
            )
            acs_params["beta"] = beta

        with col2:
            rho = st.number_input(
                "Rho (ρ) - Tingkat Evaporasi",
                min_value=0.01, max_value=1.0, step=0.01,
                value=float(acs_params.get("rho", 0.1)),
                help="Tingkat penguapan pheromone (default: 0.1)"
            )
            acs_params["rho"] = rho

            q0 = st.number_input(
                "Q0 - Probabilitas Eksploitasi",
                min_value=0.0, max_value=1.0, step=0.05,
                value=float(acs_params.get("q0", 0.5)),
                help="Probabilitas memilih jalur terbaik vs eksplorasi (default: 0.5)"
            )
            acs_params["q0"] = q0

        with col3:
            num_ants = st.number_input(
                "Jumlah Semut",
                min_value=1, max_value=100, step=1,
                value=int(acs_params.get("num_ants", 10)),
                help="Jumlah agen semut dalam setiap iterasi (default: 10)"
            )
            acs_params["num_ants"] = num_ants

            max_iterations = st.number_input(
                "Maksimum Iterasi",
                min_value=1, max_value=500, step=10,
                value=int(acs_params.get("max_iterations", 50)),
                help="Jumlah maksimum iterasi algoritma (default: 50)"
            )
            acs_params["max_iterations"] = max_iterations

        # Reset to defaults button
        if st.button("🔄 Reset ke Default", key="btn_reset_acs"):
            st.session_state["acs_params"] = {
                "alpha": 1.0,
                "beta": 2.0,
                "rho": 0.1,
                "q0": 0.5,
                "num_ants": 10,
                "max_iterations": 50
            }
            st.rerun()
            save_to_autosave()

        st.session_state["acs_params"] = acs_params
        # Also update iterasi for backward compatibility
        st.session_state["iterasi"] = max_iterations

    st.divider()

    # ===== SECTION 3: Permintaan Customer =====
    st.subheader("📦 Permintaan Pelanggan")

    if "points" not in st.session_state or not st.session_state.points.get("customers"):
        st.info("ℹ️ Isi dulu daftar pelanggan di tab 'Input Titik' ya.")
    else:
        customers = st.session_state.points["customers"]
        n = len(customers)

        # Use separate key for DATA storage (not widget state)
        data_key = f"customer_tw_data_{n}"

        # Initialize data if not exists or size changed
        if data_key not in st.session_state:
            customer_tw_data = inputData.get("customerTimeWindows", [])
            if not customer_tw_data or len(customer_tw_data) != n:
                customer_tw_data = []
                for c in customers:
                    customer_tw_data.append({
                        "demand": c.get("demand", 0),
                        "tw_start": c.get("tw_start", c.get("time_window", {}).get("start", "08:00")),
                        "tw_end": c.get("tw_end", c.get("time_window", {}).get("end", "17:00")),
                        "service_time": c.get("service_time", 10)
                    })
            st.session_state[data_key] = customer_tw_data

        # Build DataFrame from stored data
        customer_table_data = []
        for i, cust in enumerate(customers):
            tw_data = st.session_state[data_key][i] if i < len(
                st.session_state[data_key]) else {}
            customer_table_data.append({
                "Pelanggan": cust.get("name", f"Pelanggan {i+1}"),
                "Demand": float(tw_data.get("demand", 0)),
                "TW Start": str(tw_data.get("tw_start", "08:00")),
                "TW End": str(tw_data.get("tw_end", "17:00")),
                "Service (menit)": int(tw_data.get("service_time", 10))
            })

        df_cust = pd.DataFrame(customer_table_data)
        # Shift index to start from 1 instead of 0
        df_cust.index = range(1, len(df_cust) + 1)

        st.markdown(
            "*Masukkan demand & time window untuk setiap customer.*")

        edited_cust = st.data_editor(
            df_cust,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "Pelanggan": st.column_config.TextColumn("Pelanggan", disabled=True),
                "Demand": st.column_config.NumberColumn("Demand", min_value=0, format="%d"),
                "TW Start": st.column_config.TextColumn("TW Start", help="Format: HH:MM (misal 08:00)"),
                "TW End": st.column_config.TextColumn("TW End", help="Format: HH:MM (misal 17:00)"),
                "Service (menit)": st.column_config.NumberColumn("Service (menit)", min_value=0, format="%d")
            }
        )

        # Sync edited data back to session state and inputData
        if edited_cust is not None:
            updated_tw_data = []
            updated_demands = []
            for row in edited_cust.to_dict(orient="records"):
                updated_tw_data.append({
                    "demand": float(row["Demand"]),
                    "tw_start": row["TW Start"],
                    "tw_end": row["TW End"],
                    "service_time": int(row["Service (menit)"])
                })
                updated_demands.append(float(row["Demand"]))

            # Update BOTH session state data and inputData
            st.session_state[data_key] = updated_tw_data
            inputData["customerTimeWindows"] = updated_tw_data
            inputData["customerDemand"] = updated_demands
            inputData["customerDemand"] = updated_demands
            st.session_state["inputData"] = inputData
            save_to_autosave()

    st.divider()

    # ===== SUMMARY METRICS =====
    # Calculate totals based on current session state data
    if "points" in st.session_state and st.session_state.points.get("customers"):
        current_data = st.session_state.get(data_key, [])
        total_demand = sum(item.get("demand", 0) for item in current_data)
        total_service = sum(item.get("service_time", 0) for item in current_data)

        st.markdown("##### 📊 Ringkasan Total")
        m1, m2 = st.columns(2)
        m1.metric("Total Permintaan (Demand)", f"{total_demand:,.0f}")
        m2.metric("Total Waktu Layanan (Service)", f"{total_service:,.0f} menit")

    st.divider()

    # ===== SECTION 4: Tabel Jarak (Distance Matrix) =====
    st.subheader("📏 Matriks Jarak")

    # NEW: Distance Multiplier (Layout yang lebih rapi)
    st.markdown("##### ⚙️ Konfigurasi Jarak")

    dist_multiplier = st.number_input(
        "Faktor Pengali (Multiplier)",
        min_value=0.1, max_value=10.0, step=0.1,
        value=float(st.session_state.get("distance_multiplier", 1.5)),
        help="Faktor pengali untuk jarak Euclidean. Default 1.0 (murni). Gunakan 1.5 untuk memperhitungkan tortuosity (kelok jalan)."
    )
    st.caption(
        "ℹ️ **Tips:** Gunakan **1.5** untuk memperhitungkan _tortuosity_ (kelok jalan).")

    st.session_state["distance_multiplier"] = dist_multiplier
    # Also save to inputData
    if "inputData" in st.session_state:
        st.session_state["inputData"]["distance_multiplier"] = dist_multiplier

    nodes = []
    nodes_map = {}

    if "points" in st.session_state:
        pts = st.session_state.points
        # Depot first, then customers
        for i, d in enumerate(pts.get("depots", [])):
            node_id = int(d.get("id", i))
            nodes.append(node_id)
            nodes_map[node_id] = d.get("name", f"Depot {i}")
        for i, c in enumerate(pts.get("customers", [])):
            node_id = int(c.get("id", i + 1))
            nodes.append(node_id)
            nodes_map[node_id] = c.get("name", f"Pelanggan {i+1}")

    size = len(nodes)

    if size == 0:
        st.info("ℹ️ Set dulu depot & pelanggan di tab 'Input Titik' biar matriksnya muncul.")
    else:
        # Show node mapping for reference in a cleaner way
        with st.expander("📋 Pemetaan Node", expanded=False):
            # Create a clean dataframe for mapping
            mapping_data = [{"ID": nid, "Nama Lokasi": nodes_map.get(
                nid, str(nid))} for nid in nodes]
            st.dataframe(
                pd.DataFrame(mapping_data),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "ID": st.column_config.TextColumn("Node ID", width="small"),
                    "Nama Lokasi": st.column_config.TextColumn("Nama Lokasi")
                }
            )

        # Create column config - all columns editable with 5 decimal places
        col_config = {}
        for j in range(size):
            col_key = f"col_{j}"
            col_config[col_key] = st.column_config.NumberColumn(
                f"{j}",
                min_value=0.0,
                step=0.00001,
                format="%.5f",
                width="small",  # Make columns compact
                help=f"Jarak dari/ke node {j}"
            )

        # Use separate key for DATA storage (not widget state)
        data_key = f"distance_matrix_data_{size}"

        # Initialize data if not exists or size changed
        if data_key not in st.session_state:
            if not inputData.get("distanceMatrix") or len(inputData.get("distanceMatrix", [])) != size:
                inputData["distanceMatrix"] = [
                    [0.0 for _ in range(size)] for _ in range(size)]
            st.session_state[data_key] = inputData["distanceMatrix"]

        # --- Manual Update Form (Symmetric) ---
        with st.expander("✏️ Update Jarak Manual", expanded=False):
            st.caption(
                "Update jarak antar dua titik (otomatis bolak-balik A↔B).")
            c_from, c_to, c_dist, c_btn = st.columns(
                [2, 2, 1.5, 1], vertical_alignment="bottom")

            # Create options list for selectbox (ID - Name)
            # Use Index + Name to represent distinct nodes clearly
            valid_indices = range(size)
            node_options = [
                f"{i} - {nodes_map.get(nodes[i], f'Node {i}')}" for i in valid_indices]

            with c_from:
                sel_from_str = st.selectbox(
                    "Dari", options=node_options, key="manual_dist_from")
            with c_to:
                sel_to_str = st.selectbox(
                    "Ke", options=node_options, key="manual_dist_to")
            with c_dist:
                val_dist = st.number_input(
                    "Jarak (km)", min_value=0.0, step=0.1, format="%.2f", key="manual_dist_val")
            with c_btn:
                if st.button("Update", type="primary", use_container_width=True, key="btn_update_dist"):
                    # Parse indices from the string "Index - Name"
                    # We rely on string format matching above loop
                    idx_from = int(sel_from_str.split(" - ")[0])
                    idx_to = int(sel_to_str.split(" - ")[0])

                    if idx_from == idx_to:
                        st.toast("Jarak ke diri sendiri harus 0!", icon="⚠️")
                    else:
                        # Update BOTH directions (Symmetric)
                        st.session_state[data_key][idx_from][idx_to] = val_dist
                        st.session_state[data_key][idx_to][idx_from] = val_dist

                        # Force update inputData too so it sticks
                        inputData["distanceMatrix"] = st.session_state[data_key]
                        st.session_state["inputData"] = inputData

                        st.toast(
                            f"Jarak Node {idx_from} ↔ Node {idx_to} = {val_dist} km", icon="💾")
                        save_to_autosave()
                        st.rerun()

        # Build DataFrame from stored data
        df_data = {}
        for j in range(size):
            col_key = f"col_{j}"
            df_data[col_key] = [st.session_state[data_key][i][j]
                                for i in range(size)]
        df_matrix = pd.DataFrame(df_data, index=list(range(size)))
        df_matrix.index.name = "From\\To"

        # Use st.data_editor without key to avoid widget state conflicts
        edited = st.data_editor(
            df_matrix,
            use_container_width=True,
            num_rows="fixed",
            column_config=col_config
        )

        # Convert back to matrix format and enforce symmetric + diagonal zero
        if edited is not None:
            mat = []
            for i in range(size):
                row = []
                for j in range(size):
                    col_key = f"col_{j}"
                    val = edited[col_key].iloc[i] if col_key in edited.columns else 0.0
                    # Force diagonal to 0
                    if i == j:
                        row.append(0.0)
                    else:
                        try:
                            val_float = float(val) if val is not None else 0.0
                            row.append(val_float)
                        except (ValueError, TypeError):
                            row.append(0.0)
                mat.append(row)

            # Enforce symmetry: ensure mat[i][j] == mat[j][i]
            for i in range(size):
                for j in range(i + 1, size):
                    val_ij = mat[i][j]
                    val_ji = mat[j][i]

                    # If they differ, sync them - prefer non-zero value
                    if val_ij != val_ji:
                        if val_ij != 0 or val_ji == 0:
                            mat[j][i] = val_ij
                        else:
                            mat[i][j] = val_ji

            # Update BOTH session state data and inputData
            st.session_state[data_key] = mat
            inputData["distanceMatrix"] = mat
            save_to_autosave()

    st.session_state["inputData"] = inputData

    st.divider()

    # ===== ACTION BUTTONS: Save, Load, Process =====
    st.subheader("🚀 Eksekusi")

    # Row 1: Main action buttons
    button_cols = st.columns(2)

    with button_cols[0]:
        if st.button("💾 Simpan Progres", use_container_width=True, key="btn_save_progress"):
            progress_data = {
                "points": st.session_state.get("points", {}),
                "inputData": st.session_state.get("inputData", {}),
                "user_vehicles": st.session_state.get("user_vehicles", []),
                "acs_params": st.session_state.get("acs_params", {}),
                "kapasitas_kendaraan": st.session_state.get("kapasitas_kendaraan", 100),
                "iterasi": st.session_state.get("iterasi", 2)
            }
            st.session_state["saved_progress"] = progress_data

            # Also offer download
            json_str = json.dumps(progress_data, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="progress.json",
                mime="application/json",
                key="btn_download_progress"
            )

    with button_cols[1]:
        if st.button("✅ Simpan & Validasi Data", use_container_width=True, key="btn_validate", type="primary"):
            st.session_state["data_validated"] = False
            
            # Clear previous results to ensure fresh start
            if "result" in st.session_state:
                del st.session_state["result"]
            if "academic_result" in st.session_state:
                del st.session_state["academic_result"]

            # Build state
            state = {
                "points": st.session_state.get("points", {}),
                "inputData": st.session_state.get("inputData", {}),
                "user_vehicles": st.session_state.get("user_vehicles", [])
            }

            # Validate
            valid, errors = agents.validate_state(state)
            if not valid:
                error_msg = "❌ Validasi gagal:\n" + \
                    "\n".join([f"• {e}" for e in errors])
                st.error(error_msg)
            else:
                st.session_state["data_validated"] = True
                st.success("✅ Data beres & sudah tervalidasi! Bisa lanjut ke tab **'Proses Optimasi'** sekarang.")


    # Row 2: Load progress (in expander for cleaner UI)
    with st.expander("📂 Muat Data Lama"):
        uploaded_file = st.file_uploader(
            "Pilih file JSON",
            type=["json"],
            key="upload_progress_file",
            help="Upload file JSON hasil simpan sebelumnya",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            # Check if we already processed this file in this session
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("last_loaded_file") != file_id:
                try:
                    loaded_data = json.load(uploaded_file)
                    st.session_state["points"] = loaded_data.get("points", {})
                    st.session_state["inputData"] = loaded_data.get(
                        "inputData", {})
                    st.session_state["user_vehicles"] = loaded_data.get(
                        "user_vehicles", [])
                    st.session_state["acs_params"] = loaded_data.get(
                        "acs_params", {})
                    st.session_state["kapasitas_kendaraan"] = loaded_data.get(
                        "kapasitas_kendaraan", 100)
                    st.session_state["iterasi"] = loaded_data.get("iterasi", 2)
                    st.session_state["last_loaded_file"] = file_id
                    st.session_state["distanceMatrix_size"] = len(
                        loaded_data.get("inputData", {}).get("distanceMatrix", []))

                    # Clear ALL cached data keys to force refresh with new data
                    keys_to_clear = [k for k in list(st.session_state.keys())
                                     if k.startswith("customer_tw_data_") or
                                     k.startswith("distance_matrix_data_") or
                                     k.startswith("customer_tw_editor_") or
                                     k.startswith("distance_matrix_editor_")]
                    for k in keys_to_clear:
                        del st.session_state[k]

                    st.success(
                        "✅ Progres lama berhasil dimuat! Refresh halaman dulu ya biar datanya muncul.")
                    save_to_autosave()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Gagal memuat file: {e}")
        else:
            # Reset tracker when file is removed so it can be re-loaded
            st.session_state["last_loaded_file"] = None
