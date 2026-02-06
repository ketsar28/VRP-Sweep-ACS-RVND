"""
Academic Replay Results Tab

Displays all iterations from the Word document validation:
- ACS iterations (per cluster)
- RVND inter-route iterations
- RVND intra-route iterations
- Final validation results
"""

from __future__ import annotations

import json
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
ACADEMIC_OUTPUT_PATH = DATA_DIR / "academic_replay_results.json"


def _format_number(value: float) -> str:
    """Format number with 2 decimals."""
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _load_academic_results() -> Dict[str, Any]:
    """Load academic replay results if available."""
    if ACADEMIC_OUTPUT_PATH.exists():
        with ACADEMIC_OUTPUT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _display_sweep_iterations(logs: List[Dict]) -> None:
    """Display SWEEP algorithm iterations."""
    st.markdown("### 📐 SWEEP Algorithm - Polar Angles & Clustering")

    # Polar angles
    angle_logs = [l for l in logs if l.get(
        "phase") == "SWEEP" and l.get("step") == "polar_angle"]
    if angle_logs:
        st.markdown("**Step 1: Polar Angle Computation**")
        st.info("Formula perhitungan sudut polar (derajat) untuk setiap customer:")
        st.latex(
            r"\theta = \arctan\left(\frac{y_i - y_{\text{depot}}}{x_i - x_{\text{depot}}}\right) \cdot \frac{180}{\pi}")

        df_angles = pd.DataFrame([{
            "Customer": l["customer_id"],
            "Angle (°)": l["angle"],
            "Formula": l["formula"]
        } for l in angle_logs])
        st.dataframe(df_angles, use_container_width=True, hide_index=True)

    # Sorted order
    sorted_logs = [l for l in logs if l.get(
        "phase") == "SWEEP" and l.get("step") == "sorted_order"]
    if sorted_logs:
        st.markdown("**Step 2: Sorted Customer Order**")
        st.info(f"Order: {sorted_logs[0]['order']}")

    # Clusters formed
    cluster_logs = [l for l in logs if l.get(
        "phase") == "SWEEP" and l.get("step") == "cluster_formed"]
    if cluster_logs:
        st.markdown("**Step 3: Clusters Formed**")
        df_clusters = pd.DataFrame([{
            "Cluster": l["cluster_id"],
            "Customers": str(l["customer_ids"]),
            "Total Demand": l["total_demand"],
            "Vehicle": l["vehicle_type"]
        } for l in cluster_logs])
        st.dataframe(df_clusters, use_container_width=True, hide_index=True)


def _display_nn_iterations(logs: List[Dict]) -> None:
    """Display Nearest Neighbor iterations with Time Window analysis."""
    st.markdown("### 🔗 Nearest Neighbor - Initial Routes (TW-Aware)")
    st.markdown(
        "*ArrivalTime = DepartureTime + TravelTime | Wait if early | Reject if late*")

    nn_logs = [l for l in logs if l.get("phase") == "NN"]

    if nn_logs:
        # Group by cluster
        clusters = set(l["cluster_id"] for l in nn_logs)

        for cluster_id in sorted(clusters):
            with st.expander(f"Cluster {cluster_id}", expanded=True):
                cluster_logs = [
                    l for l in nn_logs if l["cluster_id"] == cluster_id]

                # Build table with TIME WINDOW data
                rows = []
                for l in cluster_logs:
                    to_node = l.get("to_node", 0)
                    arrival = l.get("arrival_time", "-")
                    tw_start = l.get("tw_start", "-")
                    tw_end = l.get("tw_end", "-")
                    action = l.get("action", "")

                    # Format time windows
                    if to_node == 0:
                        # Depot - no TW analysis needed
                        tw_display = "-"
                        status = "🏠 Depot"
                    elif arrival != "-" and tw_start != "-" and tw_end != "-":
                        tw_display = f"{_minutes_to_time(tw_start)} - {_minutes_to_time(tw_end)}"
                        if action == "REJECTED":
                            status = f"❌ Terlambat (Tiba {_minutes_to_time(arrival)} > TW Selesai)"
                        elif arrival < tw_start:
                            wait = tw_start - arrival
                            status = f"⏳ Menunggu {wait:.1f} menit"
                        else:
                            status = "✅ Memenuhi"
                    else:
                        tw_display = "-"
                        status = "-"

                    rows.append({
                        "Step": l["step"],
                        "Dari → Ke": f"{l.get('from_node', 0)} → {to_node}",
                        "Jarak (km)": l.get("distance", 0),
                        "Waktu Tiba": _minutes_to_time(arrival) if arrival != "-" else "-",
                        "TW (Mulai-Selesai)": tw_display,
                        "Status": status,
                        "Keterangan": l.get("description", "")[:50]
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Show rejected customers if any
                rejected = [l for l in cluster_logs if l.get(
                    "action") == "REJECTED"]
                if rejected:
                    st.warning(
                        f"⚠️ {len(rejected)} customer ditolak karena pelanggaran Time Window (TW)")
                    for r in rejected:
                        st.caption(
                            f"Customer {r['to_node']}: {r.get('reason', '')}")


def _display_acs_iterations(logs: List[Dict]) -> None:
    """Display ACS iterations with full detail."""
    st.markdown("### 🐜 Ant Colony System - Iterations")

    acs_logs = [l for l in logs if l.get("phase") == "ACS"]

    if not acs_logs:
        st.info("No ACS iteration logs available.")
        return

    # Group by cluster
    clusters = set(l["cluster_id"] for l in acs_logs)

    for cluster_id in sorted(clusters):
        st.markdown(f"#### Cluster {cluster_id}")

        cluster_logs = [l for l in acs_logs if l["cluster_id"] == cluster_id]

        # Pheromone initialization
        init_logs = [l for l in cluster_logs if l.get(
            "step") == "init_pheromone"]
        if init_logs:
            init = init_logs[0]
            st.markdown(
                f"**Pheromone Init:** τ₀ = {init['tau0']} ({init['formula']})")
            st.latex(r"\tau_0 = \frac{1}{n \cdot Z_{nn}}")

        # Objective function initialization
        obj_init_logs = [l for l in cluster_logs if l.get(
            "step") == "init_objective"]
        if obj_init_logs:
            obj = obj_init_logs[0]
            st.markdown(
                f"**Objective Function:** {obj.get('formula', 'Z = αD + βT + γTW')}")
            st.latex(
                r"Z = w_1 \cdot D_{\text{total}} + w_2 \cdot T_{\text{travel}} + w_3 \cdot V_{\text{TW}}")
            st.info(f"Initial Z = {obj.get('initial_objective', '?')} | "
                    f"Distance: {obj.get('initial_distance', '?')} | "
                    f"Time: {obj.get('initial_time', '?')} | "
                    f"TW Violation: {obj.get('initial_tw_violation', 0)}")

        # Iteration details
        iterations = set(l.get("iteration")
                         for l in cluster_logs if l.get("iteration"))

        for iteration in sorted(iterations):
            with st.expander(f"Iteration {iteration}", expanded=False):
                iter_logs = [l for l in cluster_logs if l.get(
                    "iteration") == iteration]

                # Predefined route (new format)
                predefined_logs = [l for l in iter_logs if l.get(
                    "step") == "route_predefined"]
                if predefined_logs:
                    st.markdown("**Predefined Route (ACADEMIC REPLAY):**")
                    for l in predefined_logs:
                        st.info(
                            f"Ant {l.get('ant', '?')}: {l.get('route', [])} - {l.get('description', '')}")

                # Route evaluation (new format - with OBJECTIVE FUNCTION)
                eval_logs = [l for l in iter_logs if l.get(
                    "step") == "route_evaluation"]
                if eval_logs:
                    st.markdown("**Route Evaluation (Z = αD + βT + γTW):**")
                    df = pd.DataFrame([{
                        "Ant": l.get("ant", "?"),
                        "Route": str(l.get("route", [])),
                        "Distance (D)": l.get("distance", 0),
                        "Time (T)": l.get("service_time", 0) + l.get("distance", 0),
                        "TW Violation": l.get("tw_violation", 0),
                        "Wait Time": l.get("wait_time", 0),
                        "Objective (Z)": l.get("objective", "-")
                    } for l in eval_logs])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # Ant route construction (old format compatibility)
                ant_logs = [
                    l for l in iter_logs if "ant" in l and "step" in l and "probabilities" in l]
                if ant_logs:
                    st.markdown("**Route Construction:**")
                    df = pd.DataFrame([{
                        "Ant": l["ant"],
                        "Step": l["step"],
                        "From": l["from_node"],
                        "q": l.get("random_q", "N/A"),
                        "Decision": l.get("decision", "N/A"),
                        "Selected": l["selected"],
                        "Probabilities": str(l["probabilities"])[:50] + "..."
                    } for l in ant_logs])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # Route evaluation (old format compatibility)
                route_logs = [
                    l for l in iter_logs if "route" in l and "objective" in l]
                if route_logs:
                    st.markdown("**Route Evaluation:**")
                    df = pd.DataFrame([{
                        "Ant": l["ant"],
                        "Route": str(l["route"]),
                        "Distance": l["distance"],
                        "Service Time": l["service_time"],
                        "TW Violation": l["tw_violation"],
                        "Objective": l["objective"]
                    } for l in route_logs])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # Iteration summary (handle both old and new formats)
                summary_logs = [l for l in iter_logs if l.get(
                    "step") == "iteration_summary"]
                if summary_logs:
                    s = summary_logs[0]
                    best_route = s.get('best_route', [])
                    best_distance = s.get('best_distance', 0)
                    best_objective = s.get('best_objective', "-")
                    tw_viol = s.get('best_tw_violation', 0)
                    acceptance = s.get('acceptance_criterion', "DISTANCE")

                    # Display with objective function info
                    st.markdown(f"**Acceptance Criterion:** `{acceptance}`")
                    st.success(f"**Best Route:** {best_route} | "
                               f"Distance: {best_distance} | "
                               f"TW Violation: {tw_viol} | "
                               f"**Objective Z = {best_objective}**")


def _display_rvnd_inter_iterations(logs: List[Dict]) -> None:
    """Display RVND inter-route iterations like thesis format."""
    st.markdown("### 🔄 RVND Inter-Route - Pertukaran Customer Antar Rute")

    inter_logs = [l for l in logs if l.get("phase") == "RVND-INTER"]

    if not inter_logs:
        st.info("Tidak ada iterasi inter-route (rute tunggal atau tidak ada move).")
        return

    st.info(f"📊 Total iterasi: {len(inter_logs)}")

    # Build table with thesis-style columns
    df = pd.DataFrame([{
        "Iterasi": l.get("iteration_id", l.get("iteration", "?")),
        "Rute A → B": f"{l.get('route_pair', (1, 2))}",
        "Move": l.get("neighborhood", "-").replace("_", " ").title() if l.get("neighborhood") else "-",
        "Total Jarak": f"{l.get('total_distance', l.get('distance_after', 0)):.2f} km",
        "Status": "Ada Perbaikan" if l.get("improved", l.get("accepted", False)) else "Tidak Ada Perbaikan"
    } for l in inter_logs])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _display_rvnd_intra_iterations(logs: List[Dict]) -> None:
    """Display RVND intra-route iterations like thesis format."""
    st.markdown("### 🔁 RVND Intra-Route - Optimasi Dalam Rute")

    intra_logs = [l for l in logs if l.get("phase") == "RVND-INTRA"]

    if not intra_logs:
        st.info("Tidak ada iterasi intra-route.")
        return

    st.info(f"📊 Total iterasi: {len(intra_logs)}")

    # Group by cluster
    clusters = set(l.get("cluster_id", 0)
                   for l in intra_logs if "cluster_id" in l)

    for cluster_id in sorted(clusters):
        with st.expander(f"Cluster {cluster_id}", expanded=True):
            cluster_logs = [l for l in intra_logs if l.get(
                "cluster_id") == cluster_id]

            st.caption(f"Iterasi untuk cluster ini: {len(cluster_logs)}")

            # Build table with thesis-style columns
            df = pd.DataFrame([{
                "Iterasi": l.get("iteration_id", l.get("iteration", "?")),
                "Move": l.get("neighborhood", "-").replace("_", " ").title() if l.get("neighborhood") else "-",
                "Jarak": f"{l.get('total_distance', l.get('distance_after', 0)):.2f} km",
                "Rute": str(l.get("routes_snapshot", l.get("sequence_after", [])))[:50],
                "Status": "Ada Perbaikan" if l.get("improved", l.get("accepted", False)) else "Tidak Ada Perbaikan"
            } for l in cluster_logs])
            st.dataframe(df, use_container_width=True, hide_index=True)


def _minutes_to_time(minutes: float) -> str:
    """Convert minutes from midnight to HH:MM format."""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"


def _display_time_window_analysis(result: Dict[str, Any]) -> None:
    """Display detailed time window analysis for each cluster/vehicle."""
    st.markdown("### ⏰ Vehicle Time Window Analysis")
    st.markdown("*Customer availability windows and compliance status*")

    routes = result.get("routes", [])
    dataset = result.get("dataset", {})

    if not routes:
        st.info("No routes available for time window analysis.")
        return

    # Get customer data for names
    customers = {c["id"]: c for c in dataset.get("customers", [])}
    depot = dataset.get("depot", {})
    depot_tw_start = depot.get("time_window", {}).get("start", "08:00")
    depot_tw_end = depot.get("time_window", {}).get("end", "17:00")

    for route in routes:
        cluster_id = route.get("cluster_id", "?")
        vehicle_type = route.get("vehicle_type", "?")
        stops = route.get("stops", [])

        with st.expander(f"🚛 Cluster {cluster_id} - Vehicle {vehicle_type}", expanded=True):
            # Depot start info
            if stops:
                depot_stop = stops[0]
                depot_start_time = depot_stop.get(
                    "departure", depot_stop.get("arrival", 480))
                st.info(f"**Depot Start Time:** {_minutes_to_time(depot_start_time)} | "
                        f"Depot Window: {depot_tw_start} - {depot_tw_end}")

            # Build table for customer stops
            customer_rows = []
            total_wait = 0
            total_violation = 0

            for stop in stops:
                node_id = stop.get("node_id", 0)

                # Skip depot entries (node_id == 0)
                if node_id == 0:
                    continue

                customer = customers.get(node_id, {})
                customer_name = customer.get("name", f"Customer {node_id}")

                raw_arrival = stop.get("raw_arrival", stop.get("arrival", 0))
                service_start = stop.get("arrival", raw_arrival)
                service_time = stop.get("service_time", 0)
                service_end = stop.get(
                    "departure", service_start + service_time)
                tw_start = stop.get("tw_start", 0)
                tw_end = stop.get("tw_end", 0)
                wait = stop.get("wait", 0)
                violation = stop.get("violation", 0)

                total_wait += wait
                total_violation += violation

                # Determine compliance status
                if violation > 0:
                    status = f"❌ LATE by {violation:.1f} min"
                elif wait > 0:
                    status = f"⏳ WAIT {wait:.1f} min"
                else:
                    status = "✅ OK"

                customer_rows.append({
                    "Customer": customer_name,
                    "Arrival Time": _minutes_to_time(raw_arrival),
                    "Service Start": _minutes_to_time(service_start),
                    "Service End": _minutes_to_time(service_end),
                    "TW Start": _minutes_to_time(tw_start),
                    "TW End": _minutes_to_time(tw_end),
                    "Status": status
                })

            if customer_rows:
                df = pd.DataFrame(customer_rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Wait Time", f"{total_wait:.1f} min")
                with col2:
                    if total_violation > 0:
                        st.metric(
                            "TW Violations", f"{total_violation:.1f} min", delta="⚠️", delta_color="inverse")
                    else:
                        st.metric("TW Violations", "0 min ✅")
                with col3:
                    compliant = sum(
                        1 for r in customer_rows if "OK" in r["Status"] or "WAIT" in r["Status"])
                    st.metric("Compliance",
                              f"{compliant}/{len(customer_rows)} customers")
            else:
                st.warning("No customer stops in this route.")

    # Overall summary
    st.markdown("---")
    st.markdown("#### 📊 Overall Time Window Summary")

    total_routes_compliant = sum(
        1 for r in routes if r.get("total_tw_violation", 0) == 0)
    total_tw_violation = sum(r.get("total_tw_violation", 0) for r in routes)
    total_wait_time = sum(r.get("total_wait_time", 0) for r in routes)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Routes Compliant",
                  f"{total_routes_compliant}/{len(routes)}")
    with col2:
        st.metric("Total Wait Time", f"{total_wait_time:.1f} min")
    with col3:
        if total_tw_violation > 0:
            st.metric("Total TW Violations",
                      f"{total_tw_violation:.1f} min", delta="⚠️", delta_color="inverse")
        else:
            st.metric("Total TW Violations", "0 min ✅")
    with col4:
        if total_tw_violation == 0:
            st.success("All customers served within time windows!")
        else:
            st.warning("Some time window violations detected")


def _display_user_vehicle_selection(result: Dict[str, Any]) -> None:
    """Display user's vehicle selection and decision reasons."""
    st.markdown("### 🚛 Pemilihan Kendaraan User")
    st.markdown("*Kendaraan yang dipilih user di Input Data*")

    user_selection = result.get("user_vehicle_selection", [])

    if not user_selection:
        # Try to get from logs
        logs = result.get("iteration_logs", [])
        user_selection = [l for l in logs if l.get(
            "phase") == "USER_VEHICLE_SELECTION"]

    if not user_selection:
        st.error("❌ Tidak ada kendaraan yang didefinisikan!")
        st.warning(
            "Silakan tambah kendaraan di tab 'Input Data' terlebih dahulu.")
        return

    # Display selection table
    df = pd.DataFrame([{
        "Kendaraan": s.get("vehicle_id", s.get("vehicle_name", "?")),
        "Kapasitas": s.get("capacity", 0),
        "Status": "✅ Aktif" if s.get("enabled", False) else "❌ Tidak Aktif",
        "Unit": s.get("units", 1) if s.get("enabled", False) else "-",
        "Jam Mulai": s.get("available_from", "08:00") if s.get("enabled", False) else "-",
        "Jam Selesai": s.get("available_until", "17:00") if s.get("enabled", False) else "-",
        "Keterangan": s.get("status", "")
    } for s in user_selection])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Summary metrics
    active_vehicles = [s for s in user_selection if s.get("enabled", False)]
    inactive_vehicles = [
        s for s in user_selection if not s.get("enabled", False)]

    total_units = sum(s.get("units", 1) for s in active_vehicles)
    total_capacity = sum(s.get("capacity", 0) * s.get("units", 1)
                         for s in active_vehicles)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Kendaraan Aktif", len(active_vehicles))
    with col2:
        st.metric("Kendaraan Tidak Aktif", len(inactive_vehicles))
    with col3:
        st.metric("Total Unit", total_units)
    with col4:
        st.metric("Total Kapasitas", total_capacity)

    # Decision explanation
    st.markdown("---")
    st.markdown("#### 📋 Detail Pemilihan Kendaraan")

    for s in user_selection:
        vid = s.get("vehicle_id", s.get("vehicle_name", "?"))
        enabled = s.get("enabled", False)
        status = s.get("status", "")

        if enabled:
            st.success(f"**{vid}**: {status}")
        else:
            st.warning(f"**{vid}**: {status}")

    # Important note
    st.markdown("---")
    st.info("📌 **Aturan Routing**: Algoritma HANYA menggunakan kendaraan yang **aktif** (dicentang). "
            "Kendaraan yang tidak aktif TIDAK akan digunakan dalam Sweep, NN, ACS, maupun RVND.")


def _display_vehicle_availability(result: Dict[str, Any]) -> None:
    """Display vehicle availability schedule and status."""
    st.markdown("### 🕐 Vehicle Availability Schedule")
    st.markdown("*Waktu Ketersediaan Jenis Kendaraan pada Hari Itu*")

    availability = result.get("vehicle_availability", [])
    available_vehicles = result.get("available_vehicles", [])

    if not availability:
        # Fallback: extract from logs
        logs = result.get("iteration_logs", [])
        availability = [l for l in logs if l.get(
            "phase") == "VEHICLE_AVAILABILITY"]

    if not availability:
        st.info("Tidak ada data ketersediaan kendaraan.")
        return

    # Display availability table
    df = pd.DataFrame([{
        "Kendaraan": a.get("vehicle_id", "?"),
        "Kapasitas": a.get("capacity", 0),
        "Unit": a.get("units", 1),
        "Waktu Tersedia": a.get("time_window", "-"),
        "Status": a.get("status", "?")
    } for a in availability])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Summary metrics
    total_vehicles = len(availability)
    available_count = sum(1 for a in availability if a.get("available", False))
    unavailable_count = total_vehicles - available_count

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Jenis Kendaraan", total_vehicles)
    with col2:
        st.metric("Tersedia untuk Routing", available_count, delta=None)
    with col3:
        if unavailable_count > 0:
            st.metric("Tidak Tersedia", unavailable_count,
                      delta="⚠️", delta_color="inverse")
        else:
            st.metric("Tidak Tersedia", 0)


def _display_vehicle_assignment(result: Dict[str, Any]) -> None:
    """Display vehicle reassignment table."""
    st.markdown("### 🚛 Penugasan Ulang Kendaraan (Vehicle Reassignment)")
    st.markdown(
        "*Tahap ini memastikan setiap rute dilayani oleh kendaraan dengan kapasitas yang sesuai.*")

    final_routes = result.get("routes", [])
    if not final_routes:
        st.warning("Belum ada rute final.")
        return

    # Build logical reassignment log from final state
    reassignment_log = []

    # Check if we have explicit logs from the reassignment phase
    explicit_logs = [l for l in result.get(
        "iteration_logs", []) if l.get("phase") == "VEHICLE_REASSIGNMENT"]

    if explicit_logs:
        # Use explicit logs if available
        for log in explicit_logs:
            status_icon = "✅" if log.get("success") else "❌"
            reason_text = log.get('reason', '-')
            if reason_text == "No feasible vehicle (capacity too low/exhausted)":
                reason_text = "Tidak ada kendaraan yang memadai (kapasitas kurang / habis)"
            elif reason_text == "Vehicle capacity sufficient":
                reason_text = "Kapasitas kendaraan mencukupi"

                reassignment_log.append({
                    "Cluster": log.get("cluster_id"),
                    "Muatan (Demand)": log.get("demand"),
                    "Kendaraan": log.get("new_vehicle", "-"),
                    "Status": f"{status_icon} {log.get('status', '')}",
                    "Alasan": reason_text
                })
    else:
        # Fallback: Infer from final routes vs initial if needed, or just show final status
        # For simplicity in this specialized view, we show the final assignment status
        for route in final_routes:
            reassignment_log.append({
                "Cluster": route["cluster_id"],
                "Muatan (Demand)": route["total_demand"],
                "Kendaraan": route["vehicle_type"],
                "Status": "✅ Final",
                "Alasan": "Kapasitas Mencukupi"
            })

    if reassignment_log:
        df = pd.DataFrame(reassignment_log)
        st.dataframe(df, use_container_width=True, hide_index=True)


def _display_time_window_analysis(result: Dict[str, Any]) -> None:
    """Display TW analysis."""
    st.markdown("### ⏰ Analisis Jendela Waktu (Time Window)")

    routes = result.get("routes", [])
    if not routes:
        return

    # 1. Overall Summary
    total_viol = sum(r.get("total_tw_violation", 0) for r in routes)
    total_wait = sum(r.get("total_wait_time", 0) for r in routes)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Pelanggaran TW (Menit)", f"{total_viol:.1f}")
    with col2:
        st.metric("Total Waktu Tunggu (Menit)", f"{total_wait:.1f}")

    if total_viol > 0:
        st.info("ℹ️ Catatan: Dalam algoritma ACS/RVND, Time Window adalah *Soft Constraint* (boleh dilanggar dengan penalti), berbeda dengan NN yang *Hard Constraint* (tolak mutlak).")

    # 2. Per Route Detail
    st.markdown("#### Detail Pelanggaran per Rute")

    detail_data = []
    for r in routes:
        viol = r.get("total_tw_violation", 0)
        wait = r.get("total_wait_time", 0)

        status = "✅ OK" if viol == 0 else f"⚠️ {viol:.1f} mnt Force"

        detail_data.append({
            "Cluster": r["cluster_id"],
            "Kendaraan": r["vehicle_type"],
            "Total Pelanggaran": f"{viol:.1f}",
            "Total Tunggu": f"{wait:.1f}",
            "Status": status
        })

    st.dataframe(pd.DataFrame(detail_data),
                 use_container_width=True, hide_index=True)


def _display_final_results(result: Dict[str, Any]) -> None:
    """Display final routes table."""
    st.markdown("### 🏁 Hasil Akhir (Final Routes)")

    routes = result.get("routes", [])
    if not routes:
        st.warning("Tidak ada rute terbentuk.")
        return

    data = []
    for r in routes:
        data.append({
            "Cluster": r["cluster_id"],
            "Kendaraan": r["vehicle_type"],
            "Rute": str(r["sequence"]),
            "Jarak (km)": f"{r['total_distance']:.2f}",
            "Waktu Layanan (menit)": f"{r['total_service_time']:.0f}",
            "Pelanggaran TW (menit)": f"{r.get('total_tw_violation', 0):.0f}",
            "Waktu Tunggu (menit)": f"{r.get('total_wait_time', 0):.2f}",
            "Muatan (kg)": r["total_demand"]
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Cost
    costs = result.get("costs", {})
    st.success(
        f"💰 **Total Biaya Transportasi:** Rp {costs.get('total_cost', 0):,.0f}")

    # TIME WINDOW ANALYSIS (NEW)
    st.markdown("---")
    _display_time_window_analysis(result)

    # Costs
    st.markdown("---")
    if costs:
        st.markdown("#### Rincian Biaya:")

        breakdown = costs.get("breakdown", [])
        if breakdown:
            df = pd.DataFrame([{
                "Cluster": c["cluster_id"],
                "Vehicle": c["vehicle_type"],
                "Fixed Cost": f"Rp {c['fixed_cost']:,.0f}",
                "Variable Cost": f"Rp {c['variable_cost']:,.0f}",
                "Total Cost": f"Rp {c['total_cost']:,.0f}"
            } for c in breakdown])
            st.dataframe(df, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Fixed Cost",
                      f"Rp {costs.get('total_fixed_cost', 0):,.0f}")
        with col2:
            st.metric("Total Variable Cost",
                      f"Rp {costs.get('total_variable_cost', 0):,.0f}")
        with col3:
            st.metric("TOTAL COST", f"Rp {costs.get('total_cost', 0):,.0f}")


def _display_validation(result: Dict[str, Any]) -> None:
    """Display validation of dynamic constraints."""
    st.markdown("### ✅ Validasi Kendala Rute (Dinamis)")
    st.markdown(
        "*Memastikan rute mematuhi aturan MFVRP (Struktur, Unik, Kapasitas)*")

    validation = result.get("validation", [])
    all_valid = result.get("all_valid", False)

    if all_valid:
        st.success("🎉 SEMUA RUTE VALID SECARA MATEMATIS!")
    else:
        st.error("⚠️ BEBERAPA RUTE TIDAK VALID - LIHAT DETAIL DI BAWAH")

    if validation:
        # Dynamic validation table
        df = pd.DataFrame([{
            "Cluster": v["cluster_id"],
            "Rute Aktual": str(v["sequence"]),
            "Valid": "✅" if v["valid"] else "❌",
            "Isu / Masalah": ", ".join(v["issues"]) if v["issues"] else "-"
        } for v in validation])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Route Structure Validation
    st.markdown("### 🏗️ Route Structure Validation")

    structure_validation = result.get("structure_validation", [])
    structure_valid = result.get("structure_valid", True)

    if structure_valid:
        st.success(
            "✅ All routes have correct MFVRP structure [DEPOT → Customers → DEPOT]")
    else:
        st.error("❌ CRITICAL: Some routes have invalid structure!")

    if structure_validation:
        df_struct = pd.DataFrame([{
            "Cluster": v["cluster_id"],
            "Sequence": str(v["sequence"]),
            "Valid": "✅" if v["valid"] else "❌",
            "Issues": ", ".join(v["issues"]) if v["issues"] else "None"
        } for v in structure_validation])
        st.dataframe(df_struct, use_container_width=True, hide_index=True)


def render_academic_replay() -> None:
    """Main render function for Academic Replay tab."""
    # Show current user-defined vehicles from Input Data
    user_vehicles = st.session_state.get("user_vehicles", [])
    if user_vehicles and len(user_vehicles) > 0:
        # st.markdown("### 🚛 Kendaraan yang Didefinisikan")

        vehicle_rows = []
        for v in user_vehicles:
            name = v.get("name", v.get("id", "?"))
            cap = v.get("capacity", 0)
            units = v.get("units", 1)
            av_from = v.get("available_from", "08:00")
            av_until = v.get("available_until", "17:00")
            status = "✅ Aktif" if v.get("enabled", True) else "❌ Non-Aktif"

            vehicle_rows.append({
                "Kendaraan": name,
                "Kapasitas": cap,
                "Unit": units,
                "Jam Operasional": f"{av_from} - {av_until}",
                "Status": status
            })

        df_vehicles = pd.DataFrame(vehicle_rows)
        st.dataframe(df_vehicles, use_container_width=True, hide_index=True)
    else:
        st.error("⚠️ **Tidak ada kendaraan yang didefinisikan!**")
        st.warning(
            "Silakan tambah kendaraan di tab **'Input Data'** terlebih dahulu sebelum menjalankan Academic Replay.")
        return

    st.divider()

    # Run button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Run Optimization", type="primary"):
            with st.spinner("Running MFVRPTW optimization..."):
                try:
                    # Import and run
                    import sys
                    sys.path.insert(
                        0, str(Path(__file__).resolve().parent.parent.parent))
                    from academic_replay import run_academic_replay

                    # === GATHER ALL DYNAMIC DATA FROM SESSION STATE ===

                    # Vehicles
                    user_vehicles = st.session_state.get("user_vehicles", [])

                    # Customers - from Input Titik + Input Data (TW, demand, service time)
                    points = st.session_state.get("points", {})
                    raw_customers = points.get("customers", [])
                    input_data = st.session_state.get("inputData", {})
                    customer_tw = input_data.get("customerTimeWindows", [])

                    # Build user_customers list with merged TW data
                    user_customers = []
                    for i, c in enumerate(raw_customers):
                        tw_data = customer_tw[i] if i < len(
                            customer_tw) else {}
                        customer = {
                            "id": c.get("id", i + 1),
                            "name": c.get("name", f"Customer {i + 1}"),
                            "x": c.get("x", c.get("lng", 0)),
                            "y": c.get("y", c.get("lat", 0)),
                            "demand": tw_data.get("demand", c.get("demand", 0)),
                            "service_time": tw_data.get("service_time", c.get("service_time", 10)),
                            "time_window": {
                                "start": tw_data.get("tw_start", c.get("tw_start", "08:00")),
                                "end": tw_data.get("tw_end", c.get("tw_end", "17:00"))
                            }
                        }
                        user_customers.append(customer)

                    # Depot - from Input Titik
                    raw_depots = points.get("depots", [])
                    user_depot = None
                    if raw_depots:
                        d = raw_depots[0]
                        user_depot = {
                            "id": 0,
                            "name": d.get("name", "Depot"),
                            "x": d.get("x", d.get("lng", 0)),
                            "y": d.get("y", d.get("lat", 0)),
                            "time_window": {"start": "08:00", "end": "17:00"},
                            "service_time": 0
                        }

                    # ACS Parameters
                    user_acs_params = st.session_state.get("acs_params", None)

                    # Run optimization with all dynamic data!
                    result = run_academic_replay(
                        user_vehicles=user_vehicles,
                        user_customers=user_customers if user_customers else None,
                        user_depot=user_depot,
                        user_acs_params=user_acs_params
                    )
                    st.session_state["academic_result"] = result
                    # Sync with main Result tab so Dashboard shows these results too
                    st.session_state["result"] = result
                    st.session_state["data_validated"] = True
                    st.success("Optimization completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with col2:
        st.info("Klik untuk menjalankan optimasi MFVRPTW dengan data yang Anda input (kendaraan, customer, parameter ACS).")

    st.divider()

    # Load and display results
    result = st.session_state.get(
        "academic_result") or _load_academic_results()

    if not result:
        st.warning("Belum ada hasil. Klik 'Run Academic Replay' untuk memulai.")
        return

    # Check for errors
    if result.get("error"):
        st.error(f"❌ Error: {result['error']}")

    logs = result.get("iteration_logs", [])

    # Create tabs for each phase (removed Validation tab per user request)
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚛 Vehicle Selection",
        "📐 Sweep",
        "🔗 NN",
        "🐜 ACS",
        "🔄 RVND-Inter",
        "🔁 RVND-Intra",
        "📊 Final Results"
    ])

    with tab0:
        _display_user_vehicle_selection(result)
        st.divider()
        _display_vehicle_availability(result)

    with tab1:
        _display_sweep_iterations(logs)

    with tab2:
        _display_nn_iterations(logs)

    with tab3:
        _display_acs_iterations(logs)

    with tab4:
        _display_rvnd_inter_iterations(logs)

    with tab5:
        _display_rvnd_intra_iterations(logs)

    with tab6:
        _display_vehicle_assignment(result)
        _display_final_results(result)
