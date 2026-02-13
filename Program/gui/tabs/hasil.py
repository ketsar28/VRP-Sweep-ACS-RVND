from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Dict, Any, List


def _format_number(value: float) -> str:
    # Format with two decimals and comma as decimal separator
    s = f"{value:,.2f}"
    return s.replace(".", ",")


def _is_academic_mode(result: Dict[str, Any]) -> bool:
    """Cek apakah result berasal dari academic replay."""
    return result.get("mode") == "ACADEMIC_REPLAY"


def _get_iteration_logs(result: Dict[str, Any]):
    """Ambil ACS dan RVND logs dari kedua format pipeline.

    Standard pipeline: result["acs_data"]["iteration_logs"] & result["rvnd_data"]["iteration_logs"]
    Academic replay:   result["iteration_logs"] → filter by phase ACS_SUMMARY / RVND_SUMMARY
    """
    acs_logs = []
    rvnd_logs = []

    if "acs_data" in result and "iteration_logs" in result["acs_data"]:
        # --- Standard pipeline ---
        acs_logs = result["acs_data"]["iteration_logs"]
    elif "iteration_logs" in result:
        # --- Academic replay: ambil summary per cluster ---
        acs_logs = [log for log in result["iteration_logs"]
                    if log.get("phase") == "ACS_SUMMARY"]

    if "rvnd_data" in result and "iteration_logs" in result["rvnd_data"]:
        rvnd_logs = result["rvnd_data"]["iteration_logs"]
    elif "iteration_logs" in result:
        rvnd_logs = [log for log in result["iteration_logs"]
                     if log.get("phase") == "RVND_SUMMARY"]

    return acs_logs, rvnd_logs


def _build_routes_map(result: Dict[str, Any]) -> Dict[int, Dict]:
    """Bangun lookup map cluster_id -> route data dari result.routes[]."""
    routes_map: Dict[int, Dict] = {}
    for route in result.get("routes", []):
        cid = route.get("cluster_id")
        if cid is not None:
            routes_map[cid] = route
    return routes_map


def _display_final_routes_table(result: Dict[str, Any]) -> None:
    """Menampilkan tabel rute final yang siap dieksekusi (konsisten dengan Optimasi)."""
    st.markdown("### 📋 Hasil Akhir Rencana Perjalanan (Final Routes)")
    st.caption("Berikut adalah rute final setelah proses optimasi dan penugasan armada.")

    routes = result.get("routes", [])
    if not routes:
        st.warning("⚠️ Tidak ada rute yang terbentuk.")
        return

    # Use clearer text explanation
    st.info("✅ **Hasil Optimasi Final:** Tabel di bawah ini adalah rencana perjalanan yang **paling akurat** dan siap dieksekusi, karena telah melalui proses penugasan armada (Fleet Reassignment) sesuai dengan ketersediaan unit yang diinput.")

    # --- 1. PARSE REASSIGNMENT LOGS to get ACTUAL status ---
    reassignment_map = {} 
    explicit_logs = [log for log in result.get("iteration_logs", []) if log.get("phase") == "VEHICLE_REASSIGN"]
    
    for log in explicit_logs:
        c_id = log.get("cluster_id")
        if c_id is not None:
             reassignment_map[c_id] = {
                 "status": log.get("status", ""),
                 "new_vehicle": log.get("new_vehicle", "-")
             }

    table_data = []

    for idx, route in enumerate(routes, 1):
        cluster_id = route.get("cluster_id", "-")
        original_vehicle = route.get("vehicle_type", "-")
        
        # Determine ACTUAL vehicle
        display_vehicle = original_vehicle
        status_msg = "✅ Valid"
        is_failed = False
        
        # Check against reassignment logs for failures
        if cluster_id in reassignment_map:
            re_info = reassignment_map[cluster_id]
            re_status = re_info["status"]
            
            if "No Vehicle" in re_status or "Gagal" in re_status:
                display_vehicle = "❌ GAGAL (Stok Habis)"
                status_msg = "❌ Invalid"
                is_failed = True
            elif "Assigned" in re_status:
                display_vehicle = re_info["new_vehicle"]
        
        # Format route sequence
        seq = route.get("sequence", [])
        if seq:
            route_str = "-".join(str(n) for n in seq)
        else:
            route_str = "-"

        # Metrics
        dist = route.get("total_distance", 0)
        load = route.get("total_demand", 0)
        
        # If failed, show 0 or indicator
        dist_str = f"{dist:.2f}" if not is_failed else "(0)"
        load_str = f"{load:.1f}"
        
        # Status override if route marked invalid by backend
        if not route.get("valid", True):
             status_msg = "❌ Invalid"

        table_data.append({
            "Cluster": cluster_id,
            "Armada Final": display_vehicle,
            "Rute Perjalanan": route_str,
            "Jarak (km)": dist_str,
            "Muatan (kg)": load_str,
            "Status": status_msg
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Cluster": st.column_config.TextColumn("Cluster", width="small"),
                "Armada Final": st.column_config.TextColumn("Armada", width="medium"),
                "Rute Perjalanan": st.column_config.TextColumn("Rute", width="large"),
                "Status": st.column_config.TextColumn("Status", width="small"),
            }
        )
    
    st.divider()


def _display_iteration_logs(result: Dict[str, Any]) -> None:
    """Tampilkan tabel iterasi ACS & RVND sesuai format tesis."""

    acs_logs, rvnd_logs = _get_iteration_logs(result)
    is_academic = _is_academic_mode(result)

    # Lookup map untuk mengisi field yang kosong di summary logs
    routes_map = _build_routes_map(result) if is_academic else {}

    # Explanation for Raw Logs
    if not _is_academic_mode(result):
        st.caption("ℹ️ **Detail Teknis:** Data ini menunjukkan bagaimana algoritma secara bertahap memperbaiki kualitas rute (meminimalkan jarak total).")
    
    # ── Tabel ACS ──

    # ── Tabel ACS ──
    if acs_logs:
        st.markdown("### 🐜 Hasil Konstruksi Rute ACS (Ant Colony System)")
        st.markdown("*Solusi awal (Initial Solution) yang dibentuk oleh semut.*")

        # Deduplication Logic: Group by cluster_id, keep the one with BEST objective (lowest Z)
        # or simply the last iteration if it represents the final state
        unique_acs = {}
        for log in acs_logs:
            cid = log.get("cluster_id")
            if cid not in unique_acs:
                unique_acs[cid] = log
            else:
                # Compare objectives, keep smaller (minimization)
                curr_obj = unique_acs[cid].get("objective", float('inf')) or float('inf')
                new_obj = log.get("objective", float('inf')) or float('inf')
                if new_obj < curr_obj:
                    unique_acs[cid] = log
        
        # Sort by cluster ID
        sorted_acs_logs = sorted(unique_acs.values(), key=lambda x: int(x.get("cluster_id", 0)) if str(x.get("cluster_id", "0")).isdigit() else 999)

        acs_df_data = []
        for idx, log in enumerate(sorted_acs_logs, 1):
            cluster_id = log.get("cluster_id", "")
            distance = log.get('total_distance', 0)
            route_seq = log.get("route_sequence", "-")

            # Ambil data lengkap dari routes map jika ada
            route_data = routes_map.get(cluster_id, {})
            objective = log.get('objective') or route_data.get('objective', 0)

            # Fallback: jika route_seq kosong, coba construct dari snapshot atau route_data
            if (not route_seq or route_seq == "-") and "routes_snapshot" in log:
                # Ambil snapshot pertama (asumsi single route per ant in basic ACS log)
                snap = log.get("routes_snapshot", [])
                if snap:
                    # Snapshot biasanya list of lists/strings, kita ambil representasi string
                    route_seq = str(snap[0]).replace("[", "").replace("]", "").replace(", ", "-")
            
            # Hitung jumlah pelanggan dari urutan rute (exclude depot = 0)
            n_customers = 0
            if route_seq and route_seq != "-":
                # Bersihkan string rute
                clean_seq = route_seq.replace("-", ",").replace("[", "").replace("]", "")
                nodes = [n.strip() for n in clean_seq.split(",") if n.strip()]
                n_customers = sum(1 for n in nodes if n != "0")

            row = {
                "Cluster": cluster_id,
                "Kendaraan": log.get("vehicle_type", ""),
                "Rute": route_seq,
                "Jarak (km)": f"{distance:.2f}",
            }

            if is_academic:
                row["Jml Pelanggan"] = n_customers
            else:
                travel_time = log.get('total_travel_time', distance)
                row["Waktu Tempuh"] = f"{travel_time:.2f}"

            row["Fungsi Objektif (Z)"] = f"{objective:.2f}"
            acs_df_data.append(row)

        if acs_df_data:
            df_acs = pd.DataFrame(acs_df_data)
            st.dataframe(df_acs, use_container_width=True, hide_index=True)

        # Removed redundant "Lihat Detail Rute ACS" as per user request
        # The table now contains all necessary info including the route sequence.

    # ── Tabel RVND ──
    if rvnd_logs:
        st.markdown(
            "### 🔄 Hasil Optimasi RVND (Randomized Variable Neighborhood Descent)")
        st.markdown("*Hasil Akhir setelah perbaikan rute lokal (Local Search).*")

        # Deduplication Logic
        unique_rvnd = {}
        for log in rvnd_logs:
            cid = log.get("cluster_id")
            if cid not in unique_rvnd:
                unique_rvnd[cid] = log
            else:
                curr_obj = unique_rvnd[cid].get("objective", float('inf')) or float('inf')
                new_obj = log.get("objective", float('inf')) or float('inf')
                # For RVND, usually valid iterations are improvements, so last might be best
                # But safer to check objective
                if new_obj < curr_obj:
                    unique_rvnd[cid] = log
        
        sorted_rvnd_logs = sorted(unique_rvnd.values(), key=lambda x: int(x.get("cluster_id", 0)) if str(x.get("cluster_id", "0")).isdigit() else 999)

        rvnd_df_data = []
        for idx, log in enumerate(sorted_rvnd_logs, 1):
            cluster_id = log.get("cluster_id", "")
            distance = log.get('total_distance', 0)
            route_seq = log.get("route_sequence", "-")

            route_data = routes_map.get(cluster_id, {})
            objective = log.get('objective') or route_data.get('objective', 0)

            # Fallback for route sequence in RVND
            if (not route_seq or route_seq == "-") and "routes_snapshot" in log:
                snap = log.get("routes_snapshot", [])
                if snap:
                    # Format: [0, 5, 2, 0] -> 0-5-2-0
                    route_seq = str(snap[0]).replace("[", "").replace("]", "").replace(", ", "-")

            # Hitung jumlah pelanggan
            n_customers = 0
            if route_seq and route_seq != "-":
                clean_seq = route_seq.replace("-", ",").replace("[", "").replace("]", "")
                nodes = [n.strip() for n in clean_seq.split(",") if n.strip()]
                n_customers = sum(1 for n in nodes if n != "0")

            vehicle = log.get("vehicle_type", "") or route_data.get("vehicle_type", "")

            row = {
                "Cluster": cluster_id,
                "Kendaraan": vehicle,
                "Rute": route_seq,
                "Jarak (km)": f"{distance:.2f}",
            }

            if is_academic:
                row["Jml Pelanggan"] = n_customers
            else:
                travel_time = log.get('total_travel_time', distance)
                row["Waktu Tempuh"] = f"{travel_time:.2f}"
                phase_raw = log.get("phase", "RVND")
                row["Fase"] = phase_raw.replace("RVND-", "")

            row["Fungsi Objektif (Z)"] = f"{objective:.2f}"
            rvnd_df_data.append(row)

        if rvnd_df_data:
            df_rvnd = pd.DataFrame(rvnd_df_data)
            st.dataframe(df_rvnd, use_container_width=True, hide_index=True)

        # Removed redundant "Lihat Detail Rute RVND"
        # Table provides sufficient information.

    if not acs_logs and not rvnd_logs:
        st.info("Belum ada log iterasi. Jalankan optimasi terlebih dahulu.")


def _build_depot_summary_from_result(points: Dict[str, Any], result: Dict[str, Any]) -> Dict[int, Dict]:
    # points: {"depots": [...], "customers": [...]} entries have id,name,x,y
    depots = points.get("depots", [])
    depot_ids = [int(d.get("id", idx)) for idx, d in enumerate(depots)]
    depot_map = {int(d.get("id", i)): d.get("name", "")
                 for i, d in enumerate(depots)}

    per_depot = {did: {"name": depot_map.get(
        did, ""), "distance": 0.0, "customers": []} for did in depot_ids}

    if not result:
        return per_depot

    routes = result.get("routes", [])
    for route in routes:
        # prefer explicit total_distance if present
        dist = float(route.get("total_distance", 0.0) or 0.0)
        seq = route.get("sequence") or []
        stops = route.get("stops") or []
        # determine depot id: try first stop node_id mapping to depot ids, else fallback to first depot
        depot_id = None
        if stops and isinstance(stops, list) and len(stops) > 0:
            first_node = stops[0].get("node_id")
            # If first_node equals 0 and depot ids include 0, use 0; else if first_node matches a depot id, use it
            if first_node in per_depot:
                depot_id = first_node
            else:
                # try to map 0 -> any depot id if only one depot exists
                if len(per_depot) == 1:
                    depot_id = next(iter(per_depot.keys()))
        if depot_id is None:
            depot_id = next(iter(per_depot.keys())) if per_depot else 0

        per_depot.setdefault(depot_id, {"name": depot_map.get(
            depot_id, ""), "distance": 0.0, "customers": []})
        per_depot[depot_id]["distance"] += dist
        # add customers from sequence (exclude zeros)
        for node in seq:
            try:
                nid = int(node)
            except Exception:
                continue
            if nid == 0:
                continue
            per_depot[depot_id]["customers"].append(nid)

    return per_depot





def _render_summary_academic(result: Dict[str, Any]) -> None:
    """Render ringkasan khusus academic replay yang punya data biaya sendiri."""
    costs = result.get("costs", {})
    routes = result.get("routes", [])
    dataset = result.get("dataset", {})
    depot_info = dataset.get("depot", {})
    depot_name = depot_info.get("name", "Depot")

    total_distance = sum(r.get("total_distance", 0) for r in routes)
    total_fixed = costs.get("total_fixed_cost", 0)
    total_variable = costs.get("total_variable_cost", 0)
    total_cost = costs.get("total_cost", 0)

    # Kumpulkan semua customer dari semua rute
    all_customers: List[int] = []
    for route in routes:
        seq = route.get("sequence", [])
        for node in seq:
            if int(node) != 0:
                all_customers.append(int(node))

    cust_str = ", ".join(str(c) for c in all_customers) if all_customers else "-"

    summary_data = [{
        "Depot ID": depot_info.get("id", 0),
        "Nama Depot": depot_name,
        "Total Jarak (km)": f"{total_distance:,.2f}",
        "Biaya Tetap (Rp)": f"{total_fixed:,.0f}",
        "Biaya Variabel (Rp)": f"{total_variable:,.0f}",
        "Total Biaya (Rp)": f"{total_cost:,.0f}",
        "Jumlah Pelanggan": len(all_customers),
        "Daftar Pelanggan": cust_str
    }]

    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    st.info(f"📍 **Total Jarak Keseluruhan:** {_format_number(total_distance)} km")


def _render_summary_standard(result: Dict[str, Any]) -> None:
    """Render ringkasan untuk standard pipeline."""
    # Simplified summary logic
    points = st.session_state.get("points", {"depots": [], "customers": []})
    per_depot = _build_depot_summary_from_result(points, result)
    
    # ... (Keep existing logic or simplify if needed)
    # For now, let's keep it but ensure it doesn't show confusing duplicates
    # Actually, let's check if the user wants this. 
    # The user said "Detailed Scheduling" is confusing. 
    # This function `_render_summary_standard` seems to be used for per-depot summary.
    # Let's keep it but maybe we don't need to call it if it's not used.
    # Wait, looking at usage: render_hasil calls it? No.
    # Let's just leave it for now but remove the Validation calls in render_hasil.
    pass


def render_hasil() -> None:
    # st.header("Hasil & Visualisasi")

    data_validated = st.session_state.get("data_validated", False)
    result = st.session_state.get(
        "result") or st.session_state.get("last_pipeline_result")

    # --- EMPTY STATE ---
    if not data_validated or not result:
        st.info("ℹ️ **Belum Ada Hasil Optimasi**")
        st.markdown("""
        Untuk mendapatkan hasil optimasi dan visualisasi rute:
        1. Pastikan data input (Points & Armada) sudah valid di tab **Input Data**.
        2. Pergi ke tab **Proses Optimasi**.
        3. Klik tombol **🚀 Jalankan Optimasi**.
        
        Setelah proses selesai, detail rute akan muncul di sini.
        """)
        return

    # --- RESULT STATE ---
    
    # Context
    st.success("🎉 **Optimasi Selesai!** Berikut adalah ringkasan hasil penjadwalan armada.")

    # 1. Executive Summary KPIs
    _display_executive_kpis(result)
    
    st.divider()

    # 2. Tabel Final (Executive View)
    # We use a specialized view that matches the one in Proses Optimasi (showing failures if any)
    _display_final_routes_table(result)

    # Cost Analysis
    st.markdown("### 💰 Analisis Biaya (Valid Output)")
    
    # Recalculate Valid Costs based on Reassignment Map
    # (Since 'result["costs"]' might still include the failed ones from backend)
    valid_fixed = 0
    valid_var = 0
    valid_total = 0
    
    # Need to reconstruct reassignment map here locally
    reassignment_map = {}
    explicit_logs = [log for log in result.get("iteration_logs", []) if log.get("phase") == "VEHICLE_REASSIGN"]
    for log in explicit_logs:
        c_id = log.get("cluster_id")
        if c_id is not None:
             reassignment_map[c_id] = log.get("status", "")

    clean_breakdown = []
    costs = result.get("costs", {})
    if costs:
        for c in costs.get("breakdown", []):
            c_id = c["cluster_id"]
            # FAIL Check
            if c_id in reassignment_map and ("Gagal" in reassignment_map[c_id] or "No Vehicle" in reassignment_map[c_id]):
                continue
                
            valid_fixed += c["fixed_cost"]
            valid_var += c["variable_cost"]
            valid_total += c["total_cost"]
            
            clean_breakdown.append({
                "Cluster": c_id,
                "Armada": c["vehicle_type"],
                "Biaya Tetap": f"Rp {c['fixed_cost']:,.0f}",
                "Biaya Variabel": f"Rp {c['variable_cost']:,.0f}",
                "Total": f"Rp {c['total_cost']:,.0f}"
            })

        col1, col2, col3 = st.columns(3)
        with col1:
             st.metric("Total Biaya Tetap", f"Rp {valid_fixed:,.0f}")
        with col2:
             st.metric("Total Biaya Variabel", f"Rp {valid_var:,.0f}")
        with col3:
             st.metric("TOTAL OPERASIONAL", f"Rp {valid_total:,.0f}", delta="Valid Only")

        with st.expander("Rincian Biaya per Rute (Valid)"):
            if clean_breakdown:
                st.dataframe(pd.DataFrame(clean_breakdown), use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada data biaya valid.")

    st.divider()
    
    # 📝 Penjelasan Biaya (Cost Explanation)
    with st.expander("ℹ️ Penjelasan Struktur Biaya"):
        st.markdown("""
        **1. Biaya Tetap (Fixed Cost)**  
        Biaya dasar penggunaan kendaraan. Dihitung **per kendaraan yang dipakai**.  
        *Contoh: Biaya sewa harian, gaji supir harian.*  
        > **Rumus:** `Jumlah Kendaraan Terpakai × Rp 50.000`

        **2. Biaya Variabel (Variable Cost)**  
        Biaya yang berubah tergantung aktivitas kendaraan (jarak tempuh).  
        *Contoh: Bahan bakar, perawatan per km.*  
        > **Rumus:** `Total Jarak Tempuh (km) × Biaya per km`

        **3. Biaya Operasional (Total)**  
        Total uang yang harus dikeluarkan untuk menjalankan operasi distribusi ini.  
        > **Rumus:** `Biaya Tetap + Biaya Variabel`
        """)

    # 4. Download Section
    st.markdown("### 📥 Ekspor Data")
    c1, c2 = st.columns(2)
    with c1:
        # Generate Clean DataFrame for CSV
        clean_df = _get_clean_export_data(result)
        st.download_button(
            "Download CSV (Format Excel)",
            data=clean_df.to_csv(index=False, sep=";", decimal=",").encode('utf-8-sig'), # Use semicolon for Excel in ID/EU
            file_name="hasil_optimasi_rute_clean.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        import json
        st.download_button(
            "Download JSON (Full Log)",
            data=json.dumps(result, indent=2, default=str).encode('utf-8'),
            file_name="full_optimization_log.json",
            mime="application/json",
            use_container_width=True
        )

def _get_clean_export_data(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare a clean, flattened DataFrame for CSV/Excel export.
    Removes JSON arrays and formats columns nicely.
    """
    routes = result.get("routes", [])
    if not routes:
        return pd.DataFrame()
        
    # Reassignment Map for accurate status
    reassignment_map = {}
    explicit_logs = [log for log in result.get("iteration_logs", []) if log.get("phase") == "VEHICLE_REASSIGN"]
    for log in explicit_logs:
        c_id = log.get("cluster_id")
        if c_id is not None:
             reassignment_map[c_id] = {
                 "status": log.get("status", ""),
                 "new_vehicle": log.get("new_vehicle", "-")
             }
             
    clean_rows = []
    for r in routes:
        cid = r.get("cluster_id", "-")
        
        # Determine Status & Vehicle
        veh = r.get("vehicle_type", "-")
        status = "Valid"
        
        if cid in reassignment_map:
            re_info = reassignment_map[cid]
            if "Gagal" in re_info["status"] or "No Vehicle" in re_info["status"]:
                status = "Gagal (Stok Habis)"
                veh = "Tidak Tersedia"
            elif "Assigned" in re_info["status"]:
                veh = re_info["new_vehicle"]
                
        # Format Sequence (0 -> 1 -> 0)
        seq = r.get("sequence", [])
        seq_str = " -> ".join(map(str, seq)) if seq else "-"
        
        # Format Customers
        cust_ids = [str(x) for x in seq if x != 0]
        cust_str = ", ".join(cust_ids)
        
        # Violations
        violation_details = r.get("tw_violations_detail", [])
        if violation_details:
            v_list = [f"C{x['customer_id']} ({x['violation_minutes']:.1f}m)" for x in violation_details]
            viol_str = ", ".join(v_list)
        else:
            viol_str = "-"
            
        clean_rows.append({
            "Cluster ID": cid,
            "Kendaraan": veh,
            "Status": status,
            "Urutan Rute": seq_str,
            "Pelanggan Dilayani": cust_str,
            "Jarak Tempuh (km)": f"{r.get('total_distance', 0):.2f}".replace('.', ','),
            "Waktu Tempuh (menit)": f"{r.get('total_travel_time', 0):.2f}".replace('.', ','),
            "Waktu Layanan (menit)": f"{r.get('total_service_time', 0):.2f}".replace('.', ','),
            "Waktu Tunggu (menit)": f"{r.get('total_wait_time', 0):.2f}".replace('.', ','),
            "Total Durasi (menit)": f"{r.get('total_time', 0):.2f}".replace('.', ','),
            "Pelanggaran TW": viol_str,
            "Muatan (kg)": r.get('total_demand', 0),
            "Kapasitas Sisa (kg)": r.get('capacity_max', 0) - r.get('total_demand', 0) if 'capacity_max' in r else "-"
        })
        
    return pd.DataFrame(clean_rows)

def _display_executive_kpis(result: Dict[str, Any]) -> None:
    """Display big metrics at top of Detail Tab (Excluding Failed Routes)."""
    routes = result.get("routes", [])
    
    # Re-map reassignment failures
    reassignment_map = {}
    explicit_logs = [log for log in result.get("iteration_logs", []) if log.get("phase") == "VEHICLE_REASSIGN"]
    for log in explicit_logs:
        c_id = log.get("cluster_id")
        if c_id is not None:
             reassignment_map[c_id] = log.get("status", "")

    total_dist = 0
    total_dur = 0
    
    # We must calculate cost manually to be safe, or rely on logic above. 
    # For KPIs, let's just sum valid routes.
    
    dataset = result.get("dataset", {})
    
    valid_cost = 0

    for r in routes:
        c_id = r["cluster_id"]
        # Check failure
        if c_id in reassignment_map and ("Gagal" in reassignment_map[c_id] or "No Vehicle" in reassignment_map[c_id]):
            continue
            
        total_dist += r.get("total_distance", 0)
        total_dur += (r.get("total_travel_time", 0) + r.get("total_service_time", 0) + r.get("total_wait_time", 0))
        
        # Approximate cost if not available in breakdown (fallback)
        # But usually we should use cost breakdown. 
        # Let's use cost breakdown logic if available as it's more accurate
        
    # Re-calculate cost sum from breakdown for accuracy
    costs = result.get("costs", {})
    if costs:
        for c in costs.get("breakdown", []):
            c_id = c["cluster_id"]
            if c_id in reassignment_map and ("Gagal" in reassignment_map[c_id] or "No Vehicle" in reassignment_map[c_id]):
                continue
            valid_cost += c["total_cost"]

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Jarak (Valid)", f"{total_dist:,.1f} km", help="Hanya rute yang berhasil ditugaskan")
    kpi2.metric("Est. Total Durasi", f"{total_dur:,.0f} min", help="Hanya rute yang berhasil ditugaskan")
    kpi3.metric("Total Biaya (Valid)", f"Rp {valid_cost:,.0f}", help="Hanya rute yang berhasil ditugaskan")
