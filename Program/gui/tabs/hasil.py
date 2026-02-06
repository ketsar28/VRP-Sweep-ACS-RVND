from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Dict, Any, List


def _format_number(value: float) -> str:
    # Format with two decimals and comma as decimal separator
    s = f"{value:,.2f}"
    return s.replace(".", ",")


def _display_iteration_logs(result: Dict[str, Any]) -> None:
    """Display ACS and RVND iterations matching thesis format."""

    # Check if we have iteration logs in the result
    acs_logs = []
    rvnd_logs = []

    # Try to get ACS iteration logs from result structure
    if "acs_data" in result and "iteration_logs" in result["acs_data"]:
        acs_logs = result["acs_data"]["iteration_logs"]

    # Try to get RVND iteration logs from result structure
    if "rvnd_data" in result and "iteration_logs" in result["rvnd_data"]:
        rvnd_logs = result["rvnd_data"]["iteration_logs"]

    # Display ACS iterations - thesis format
    if acs_logs:
        st.markdown("### 🐜 Hasil Pengklasteran ACS (Ant Colony System)")
        st.markdown("*Solusi dengan Clustering + Nearest Neighboorhood*")

        acs_df_data = []
        for log in acs_logs:
            distance = log.get('total_distance', 0)
            travel_time = log.get('total_travel_time', 0)
            objective = log.get('objective', 0)
            acs_df_data.append({
                "Iterasi": log.get("iteration_id", ""),
                "Cluster": log.get("cluster_id", ""),
                "Jarak (km)": f"{distance:.2f}",
                "Waktu Perjalanan": f"{travel_time:.2f}",
                "Kendaraan": log.get("vehicle_type", ""),
                "Fungsi Objektif (Z)": f"{objective:.2f}"
            })

        if acs_df_data:
            df_acs = pd.DataFrame(acs_df_data)
            st.dataframe(df_acs, use_container_width=True, hide_index=True)

        with st.expander("📋 Lihat Detail Rute ACS"):
            for log in acs_logs:
                iter_id = log.get('iteration_id', '?')
                cluster_id = log.get('cluster_id', '?')
                st.markdown(f"**Iterasi {iter_id} - Cluster {cluster_id}**")
                routes = log.get("routes_snapshot", [])
                for idx, route in enumerate(routes):
                    st.text(f"Rute {idx + 1}: {route}")
                st.divider()

    # Display RVND iterations - thesis format
    if rvnd_logs:
        st.markdown(
            "### 🔄 Hasil RVND (Randomized Variable Neighborhood Descent)")
        st.markdown("*Optimasi lokal untuk memperbaiki solusi*")

        rvnd_df_data = []
        for log in rvnd_logs:
            distance = log.get('total_distance', 0)
            travel_time = log.get('total_travel_time', 0)
            objective = log.get('objective', 0)
            rvnd_df_data.append({
                "Iterasi": log.get("iteration_id", ""),
                "Cluster": log.get("cluster_id", ""),
                "Fase": log.get("phase", "RVND").replace("RVND-", ""),
                "Jarak (km)": f"{distance:.2f}",
                "Waktu Perjalanan": f"{travel_time:.2f}",
                "Kendaraan": log.get("vehicle_type", ""),
                "Fungsi Objektif (Z)": f"{objective:.2f}"
            })

        if rvnd_df_data:
            df_rvnd = pd.DataFrame(rvnd_df_data)
            st.dataframe(df_rvnd, use_container_width=True, hide_index=True)

        with st.expander("📋 Lihat Detail Rute RVND"):
            for log in rvnd_logs:
                iter_id = log.get('iteration_id', '?')
                cluster_id = log.get('cluster_id', '?')
                phase = log.get('phase', '')
                st.markdown(
                    f"**Iterasi {iter_id} - Cluster {cluster_id} - {phase}**")
                routes = log.get("routes_snapshot", [])
                for idx, route in enumerate(routes):
                    st.text(f"Rute {idx + 1}: {route}")
                st.divider()

    if not acs_logs and not rvnd_logs:
        st.info("Belum ada log iterasi. Jalankan optimasi terlebih dahulu.")


def _build_depot_summary_from_result(points: Dict[str, Any], result: Dict[str, Any]) -> Dict[int, Dict]:
    # points: {"depots": [...], "customers": [...]} entries have id,name,x,y
    depots = points.get("depots", [])
    customers = points.get("customers", [])
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


def render_hasil() -> None:
    st.header("Hasil")

    data_validated = st.session_state.get("data_validated", False)
    result = st.session_state.get(
        "result") or st.session_state.get("last_pipeline_result")

    if not data_validated or not result:
        st.info(
            "Belum ada hasil. Tekan 'Hasil' di menu 'Input Data' untuk menjalankan komputasi terlebih dahulu.")
        return

    # Display iteration logs FIRST (academic requirement)
    _display_iteration_logs(result)

    # Display summary table (DataFrame)
    st.divider()
    st.subheader("📈 Ringkasan Solusi Akhir")

    points = st.session_state.get("points", {"depots": [], "customers": []})
    per_depot = _build_depot_summary_from_result(points, result)

    summary_data = []
    total_all_distance = 0.0

    # Present in order of depot index (0..)
    user_vehicles = st.session_state.get("user_vehicles", [])
    vehicle_map = {v["id"]: v for v in user_vehicles}

    # Present in order of depot index (0..)
    total_fixed_cost = 0.0
    total_variable_cost = 0.0
    total_all_cost = 0.0

    for idx, (depot_id, info) in enumerate(sorted(per_depot.items(), key=lambda x: int(x[0]))):
        dist = info.get("distance", 0.0) or 0.0
        total_all_distance += float(dist)

        cust_list = info.get("customers", [])
        if cust_list:
            cust_str = ", ".join(str(c) for c in cust_list)
            status = f"{len(cust_list)} Customer"
        else:
            cust_str = "-"
            status = "Tidak ada customer"

        # Calculate costs for routes originating from this depot
        depot_fixed_cost = 0.0
        depot_variable_cost = 0.0

        # We need to scan routes to see which ones belong here.
        # Since per_depot doesn't store route objects directly, we infer from routes list.
        for route in result.get("routes", []):
            # Simplified match: if route starts with this depot (usually implied for now as we have 1 depot)
            # Ideally we check route['stops'][0]['node_id']
            stops = route.get("stops", [])
            if stops and stops[0].get("node_id") == depot_id:
                v_type = route.get("vehicle_type")
                v_dist = route.get("total_distance", 0)

                vehicle = vehicle_map.get(v_type, {})
                f_cost = vehicle.get("fixed_cost", 0)
                v_cost_per_km = vehicle.get("variable_cost_per_km", 0)

                depot_fixed_cost += f_cost
                depot_variable_cost += (v_dist * v_cost_per_km)

        total_fixed_cost += depot_fixed_cost
        total_variable_cost += depot_variable_cost
        depot_total_cost = depot_fixed_cost + depot_variable_cost
        total_all_cost += depot_total_cost

        summary_data.append({
            "Depot ID": depot_id,
            "Nama Depot": info.get("name", f"Depot {depot_id}"),
            "Total Jarak (km)": f"{dist:,.2f}",
            "Biaya Tetap (Rp)": f"{depot_fixed_cost:,.0f}",
            "Biaya Variabel (Rp)": f"{depot_variable_cost:,.0f}",
            "Total Biaya (Rp)": f"{depot_total_cost:,.0f}",
            "Jumlah Customer": len(cust_list),
            "Daftar Customer": cust_str
        })

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        # Total metrics
        st.info(
            f"📍 **Total Jarak Keseluruhan:** {_format_number(total_all_distance)} km")
    else:
        st.warning("Tidak ada data ringkasan depot.")

    # Merged Visualization Section
    st.divider()
    try:
        from graph_hasil import render_graph_hasil
        render_graph_hasil()
    except ImportError:
        st.error("Gagal memuat visualisasi rute.")
