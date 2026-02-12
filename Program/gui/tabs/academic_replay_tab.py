"""
Tab Hasil Replay Akademik

Di sini saya menampilkan semua iterasi proses perhitungan:
- Iterasi ACS (per cluster)
- Iterasi RVND inter-route & intra-route
- Hasil final validasinya
"""

from __future__ import annotations

import json
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, Any, List


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
ACADEMIC_OUTPUT_PATH = DATA_DIR / "academic_replay_results.json"


def _format_number(value: float) -> str:
    """Format number with 2 decimals."""
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _load_academic_results() -> Dict[str, Any]:
    """Load hasil rekapan kalau file-nya ada."""
    if ACADEMIC_OUTPUT_PATH.exists():
        with ACADEMIC_OUTPUT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _display_sweep_iterations(logs: List[Dict]) -> None:
    """Display SWEEP algorithm iterations."""
    st.markdown("### 📐 Algoritma SWEEP - Perhitungan & Pengelompokan")

    # Polar angles
    angle_logs = [l for l in logs if l.get(
        "phase") == "SWEEP" and l.get("step") == "polar_angle"]
    if angle_logs:
        st.markdown("**Langkah 1: Perhitungan Sudut Polar**")
        st.info("Sudut polar saya gunakan untuk menentukan posisi relatif pelanggan terhadap depot.")
        st.latex(
            r"\theta = \arctan\left(\frac{y_i - y_{\text{depot}}}{x_i - x_{\text{depot}}}\right) \cdot \frac{180}{\pi}")

        df_angles = pd.DataFrame([{
            "Pelanggan": l["customer_id"],
            "Sudut (°)": l["angle"],
            "Rumus": l["formula"]
        } for l in angle_logs])
        st.dataframe(df_angles, use_container_width=True, hide_index=True)

    # Sorted order
    sorted_logs = [l for l in logs if l.get(
        "phase") == "SWEEP" and l.get("step") == "sorted_order"]
    if sorted_logs:
        st.markdown("**Langkah 2: Pengurutan Pelanggan**")
        st.success(f"Urutan pelanggan berdasarkan sudut (polar): {sorted_logs[0]['order']}")

    # Clusters formed
    cluster_logs = [l for l in logs if l.get(
        "phase") == "SWEEP" and l.get("step") == "cluster_formed"]
    if cluster_logs:
        st.markdown("**Langkah 3: Pembentukan Cluster**")
        df_clusters = pd.DataFrame([{
            "Cluster": l["cluster_id"],
            "Daftar Pelanggan": str(l["customer_ids"]),
            "Total Muatan (kg)": l["total_demand"],
            "Armada": l["vehicle_type"]
        } for l in cluster_logs])
        st.dataframe(df_clusters, use_container_width=True, hide_index=True)


def _display_nn_iterations(logs: List[Dict]) -> None:
    """Display Nearest Neighbor iterations with Time Window analysis."""
    st.markdown("### 🔗 Algoritma Nearest Neighbor - Rute Awal")

    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 18px; border-radius: 12px; border-left: 6px solid #4a90e2; margin-bottom: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <strong style="color: #2c3e50; font-size: 1.1em;">⏱️ Konsep Perhitungan Waktu (Time Window)</strong><br>
            Rute saya bangun dengan mencari pelanggan terdekat yang masih masuk di jam operasional mereka (Time Window).
        <hr style="margin: 12px 0; border: 0; border-top: 1px solid #dee2e6;">
        <p style="font-family: 'Inter', sans-serif; font-weight: 600; color: #34495e; font-size: 1.05em; text-align: center;">
            Waktu Tiba = Waktu Berangkat + Waktu Perjalanan
        </p>
        <ul style="color: #2c3e50; margin-top: 10px; line-height: 1.6;">
            <li>✅ <strong>Memenuhi:</strong> Tiba di dalam rentang jam operasional (Time Window).</li>
            <li>⏳ <strong>Menunggu:</strong> Tiba sebelum jam operasional dimulai (Harus menunggu).</li>
            <li>❌ <strong>Ditolak:</strong> Tiba setelah jam operasional berakhir (Rute tidak valid).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    nn_logs = [l for l in logs if l.get("phase") == "NN"]

    if nn_logs:
        # Group by cluster
        clusters = set(l["cluster_id"] for l in nn_logs)

        for cluster_id in sorted(clusters):
            with st.expander(f"Cluster {cluster_id}", expanded=True):
                cluster_logs = [
                    l for l in nn_logs if l["cluster_id"] == cluster_id]
                
                # Calculate total distance for this cluster
                total_dist = sum(l.get("distance", 0) for l in cluster_logs)

                # DISPLAY SUMMARY
                summary = next((l for l in logs if l.get(
                    "phase") == "NN_SUMMARY" and l.get("cluster_id") == cluster_id), None)
                if summary:
                    st.success(
                        f"**Rute Terbentuk:** {summary['route_sequence']} menggunakan armada **{summary['vehicle_type']}** | **Total Jarak:** {total_dist:.2f} km", icon="🚚")

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
                            status = "❌ Terlambat"
                        elif arrival < tw_start:
                            wait = tw_start - arrival
                            status = f"⏳ Menunggu {wait:.1f} mnt"
                        else:
                            status = "✅ Memenuhi"
                    else:
                        tw_display = "-"
                        status = "-"

                    rows.append({
                        "Langkah": l["step"],
                        "Dari → Ke": f"{l.get('from_node', 0)} → {to_node}",
                        "Jarak (km)": l.get("distance", 0),
                        "Waktu Tiba": _minutes_to_time(arrival) if arrival != "-" else "-",
                        "Jam Operasional": tw_display,
                        "Status": status,
                        "Keterangan": l.get("description", "")[:80]
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Show rejected customers if any
                rejected = [l for l in cluster_logs if l.get(
                    "action") == "REJECTED"]
                if rejected:
                    st.error(
                        f"**Peringatan:** Terdeteksi {len(rejected)} pelanggan ditolak karena pelanggaran batasan waktu (Time Window).")
                    for r in rejected:
                        st.caption(
                            f"Pelanggan {r['to_node']}: {r.get('reason', '')}")


def _display_acs_progress_chart(cluster_logs: List[Dict], cluster_id: int) -> None:
    """Display ACS convergence chart."""
    iter_data = []
    seen_iters = set()

    # Extract best objective per iteration
    for l in cluster_logs:
        if l.get("step") == "iteration_summary":
            it = l.get("iteration")
            if it not in seen_iters:
                iter_data.append({
                    "Iterasi": it,
                    "Objective Z": float(l.get("best_objective", 0)),
                    "Jarak": float(l.get("best_distance", 0))
                })
                seen_iters.add(it)

    if not iter_data:
        return

    df = pd.DataFrame(iter_data).sort_values("Iterasi")

    # Create chart
    fig = px.line(df, x="Iterasi", y="Objective Z",
                  title=f"Konvergensi Fungsi Tujuan - Cluster {cluster_id}",
                  markers=True)
    fig.update_layout(yaxis_title="Objective Function (Z)",
                      xaxis_title="Iterasi")
    st.plotly_chart(fig, use_container_width=True)


def _display_acs_iterations(logs: List[Dict]) -> None:
    """Display ACS iterations with full detail."""
    st.markdown("### 🐜 Ant Colony System - Iterasi Semut")

    acs_logs = [l for l in logs if l.get("phase") == "ACS"]

    if not acs_logs:
        st.info("Tidak ada log iterasi ACS yang tersedia.")
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
            with st.expander("ℹ️ Keterangan Rumus & Inisialisasi"):
                st.markdown("**Inisialisasi Pheromone:**")
                st.latex(
                    r"\tau_0 = \frac{1}{n \cdot Z_{nn}}")
                st.markdown("""
                **Keterangan:**
                *   $n$: Jumlah customer
                *   $Z_{nn}$: Biaya rute dari algoritma Nearest Neighbor
                """)

        # Objective function initialization
        obj_init_logs = [l for l in cluster_logs if l.get(
            "step") == "init_objective"]
        if obj_init_logs:
            obj = obj_init_logs[0]
            # Clean formula display
            with st.expander("📊 Fungsi Tujuan & Nilai Awal", expanded=True):
                st.markdown("**Fungsi Tujuan (Objective Function):**")
                st.latex(
                    r"Z = w_1 \cdot D_{\text{total}} + w_2 \cdot T_{\text{travel}} + w_3 \cdot V_{\text{TW}}")
                st.markdown("""
                **Keterangan Variabel:**
                *   $D_{\\text{total}}$: Total Jarak Tempuh
                *   $T_{\\text{travel}}$: Total Waktu (Perjalanan + Layanan)
                *   $V_{\\text{TW}}$: Total Pelanggaran Time Window (Durasi telat/tunggu)
                
                **Keterangan Bobot:**
                *   $w_1$: Bobot Jarak
                *   $w_2$: Bobot Waktu
                *   $w_3$: Bobot Penalty (Time Window)
                """)

                st.markdown("---")
                st.markdown("**Nilai Awal (Sebelum Optimasi ACS):**")

                # Use columns for clean metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(
                    "Z Awal", f"{obj.get('initial_objective', 0):.2f}")
                m2.metric("Jarak (D)", f"{obj.get('initial_distance', 0):.2f}")
                m3.metric("Waktu (T)", f"{obj.get('initial_time', 0):.2f}")
                m4.metric("Pelanggaran TW",
                          f"{obj.get('initial_tw_violation', 0)}")

        # Iteration details with Pagination
        iterations = sorted(list(set(l.get("iteration")
                                     for l in cluster_logs if l.get("iteration"))))

        st.markdown("##### 📈 Progress Iterasi")
        _display_acs_progress_chart(cluster_logs, cluster_id)

        st.markdown("##### 📜 Detail Iterasi")

        # Pagination controls
        total_iters = len(iterations)
        items_per_page = 10

        if total_iters > items_per_page:
            # Calculate total pages correctly (ceil division)
            total_pages = (total_iters + items_per_page - 1) // items_per_page

            # Create toggle for pagination
            col_page1, col_page2 = st.columns([1, 3])
            with col_page1:
                page = st.number_input(f"Halaman (Total {total_pages} Hal)",
                                       min_value=1,
                                       max_value=total_pages,
                                       value=1, key=f"page_cluster_{cluster_id}")

            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            visible_iterations = iterations[start_idx:end_idx]

            if visible_iterations:
                st.caption(
                    f"Menampilkan iterasi {visible_iterations[0]} - {visible_iterations[-1]}")
            else:
                st.warning("Tidak ada iterasi pada halaman ini.")
        else:
            visible_iterations = iterations

        for iteration in visible_iterations:
            with st.expander(f"Iterasi {iteration}", expanded=False):
                iter_logs = [l for l in cluster_logs if l.get(
                    "iteration") == iteration]

                # Predefined route (new format)
                predefined_logs = [l for l in iter_logs if l.get(
                    "step") == "route_predefined"]
                if predefined_logs:
                    st.markdown("**Rute Terdefinisi (ACADEMIC REPLAY):**")
                    for l in predefined_logs:
                        st.info(
                            f"Semut {l.get('ant', '?')}: {l.get('route', [])} - {l.get('description', '')}")

                # Route evaluation (new format - with OBJECTIVE FUNCTION)
                eval_logs = [l for l in iter_logs if l.get(
                    "step") == "route_evaluation"]
                if eval_logs:
                    st.markdown("**Evaluasi Rute (Z = αD + βT + γTW):**")
                    df = pd.DataFrame([{
                        "Semut": l.get("ant", "?"),
                        "Rute": str(l.get("route", [])),
                        "Jarak (D)": l.get("distance", 0),
                        "Waktu (T)": l.get("service_time", 0) + l.get("distance", 0),
                        "Pelanggaran TW": l.get("tw_violation", 0),
                        "Waktu Tunggu": l.get("wait_time", 0),
                        "Fungsi Tujuan (Z)": l.get("objective", "-")
                    } for l in eval_logs])

                    # Clean up: Remove 0 values for clearer display
                    if (df["Pelanggaran TW"] == 0).all():
                        df = df.drop(columns=["Pelanggaran TW"])
                    if (df["Waktu Tunggu"] == 0).all():
                        df = df.drop(columns=["Waktu Tunggu"])

                    st.dataframe(df, use_container_width=True, hide_index=True)

                # Ant route construction (old format compatibility)
                ant_logs = [
                    l for l in iter_logs if "ant" in l and "step" in l and "probabilities" in l]
                if ant_logs:
                    st.markdown("**Konstruksi Rute:**")
                    df = pd.DataFrame([{
                        "Semut": l["ant"],
                        "Langkah": l["step"],
                        "Dari Node": l["from_node"],
                        "q": l.get("random_q", "N/A"),
                        "Keputusan": l.get("decision", "N/A"),
                        "Terpilih": l["selected"],
                        "Probabilitas": str(l["probabilities"])[:50] + "..."
                    } for l in ant_logs])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # Route evaluation (old format compatibility)
                route_logs = [
                    l for l in iter_logs if "route" in l and "objective" in l]
                if route_logs:
                    st.markdown("**Evaluasi Rute:**")
                    df = pd.DataFrame([{
                        "Semut": l["ant"],
                        "Rute": str(l["route"]),
                        "Jarak": l["distance"],
                        "Waktu Layanan": l["service_time"],
                        "Pelanggaran TW": l["tw_violation"],
                        "Fungsi Tujuan": l["objective"]
                    } for l in route_logs])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # Iteration summary (handle both old and new formats)
                summary_logs = [l for l in iter_logs if l.get("step") == "iteration_summary"]
                if summary_logs:
                    s = summary_logs[0]
                    best_route = s.get('best_route', [])
                    best_distance = s.get('best_distance', 0)
                    best_objective = s.get('best_objective', "-")
                    tw_viol = s.get('best_tw_violation', 0)

                    # Modern display with styling
                    with st.container():
                        st.markdown("#### 🏆 Rute Terbaik pada Iterasi Ini")
                        st.info(f"**Rute:** {best_route}")

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Fungsi Tujuan (Z)", str(best_objective))
                        m2.metric("Jarak (D)", f"{best_distance:.2f}")
                        if tw_viol > 0:
                            m3.metric("Pelanggaran TW",
                                      f"{tw_viol}", delta_color="inverse")
                        else:
                            m3.metric("Pelanggaran TW", "0",
                                      delta_color="normal")

                        # Status kelayakan solusi
                        if tw_viol > 0:
                            m4.metric("Status", "⚠️ Tidak Layak")
                        else:
                            m4.metric("Status", "✅ Layak")


def _display_rvnd_inter_iterations(logs: List[Dict]) -> None:
    """Display RVND inter-route iterations with improved formatting."""
    st.markdown("### 🔄 RVND Inter-Route - Pertukaran Pelanggan Antar Rute")

    inter_logs = [l for l in logs if l.get("phase") == "RVND-INTER"]

    if not inter_logs:
        st.info("Nggak ada iterasi inter-route (mungkin cuma 1 rute atau emang nggak ada move yang lebih baik).")
        return

    # === SUMMARY CARDS ===
    total_iters = len(inter_logs)
    improved_count = sum(1 for l in inter_logs if l.get("improved", False))

    # Get distance progression
    distances = [l.get("total_distance", 0) for l in inter_logs]
    first_distance = distances[0] if distances else 0
    last_distance = distances[-1] if distances else 0
    delta_pct = ((last_distance - first_distance) /
                 first_distance * 100) if first_distance > 0 else 0

    # Count neighborhood usage
    neighborhood_counts = {}
    for l in inter_logs:
        nh = l.get("neighborhood", "unknown")
        neighborhood_counts[nh] = neighborhood_counts.get(nh, 0) + 1

    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Iterasi", total_iters)
    with col2:
        st.metric("Ada Perbaikan",
                  f"{improved_count} ({improved_count/total_iters*100:.0f}%)" if total_iters > 0 else "0")
    with col3:
        st.metric("Jarak Awal", f"{first_distance:.2f} km")
    with col4:
        delta_color = "inverse" if delta_pct < 0 else "normal"
        st.metric("Jarak Akhir", f"{last_distance:.2f} km",
                  f"{delta_pct:+.1f}%", delta_color=delta_color)

    # Neighborhood usage bar
    if neighborhood_counts:
        nh_text = " | ".join([f"**{k.replace('_', ' ').title()}**: {v}x" for k,
                             v in sorted(neighborhood_counts.items(), key=lambda x: -x[1])])
        st.caption(f"📊 Penggunaan Jenis Gerakan (Neighborhood): {nh_text}")
        
        with st.expander("ℹ️ Apa itu Swap, Shift, Cross di Inter-RVND?"):
            st.markdown("""
            Ini adalah jenis pergerakan **Antar Rute** (memindahkan pelanggan dari Rute A ke Rute B):
            - **Swap 1 1**: Tukar 1 pelanggan Rute A dengan 1 pelanggan Rute B (1-Lawan-1).
            - **Swap 1 0 (Shift)**: Pindahkan 1 pelanggan dari Rute A ke Rute B tanpa balasan.
            - **Swap 2 1**: Tukar 2 pelanggan Rute A dengan 1 pelanggan Rute B.
            - **Swap 2 2**: Tukar 2 pelanggan Rute A dengan 2 pelanggan Rute B (2-Lawan-2).
            - **Cross**: Potong dan silang sambung dua rute (Cross exchange).
            - **Initial**: Kondisi awal sebelum dilakukan pergerakan apapun (basis pembanding).
            """)

    st.divider()

    # === ITERATION TABLE ===
    # Helper function to format route sequence
    def format_route(seq):
        if isinstance(seq, list):
            return "→".join(str(n) for n in seq)
        return str(seq)

    # Build table with improved columns
    table_data = []
    prev_distance = None

    for l in inter_logs:
        iter_id = l.get("iteration_id", l.get("iteration", "?"))
        neighborhood = l.get("neighborhood", "-")
        improved = l.get("improved", False)
        total_dist = l.get("total_distance", 0)
        routes_snapshot = l.get("routes_snapshot", [])

        # Itung delta dari iterasi sebelumnya
        if prev_distance is not None:
            delta = total_dist - prev_distance
            delta_str = f"{delta:+.2f}" if delta != 0 else "0.00"
        else:
            delta_str = "-"

        # Format routes - tampilkan semua rute dengan jelas
        routes_display = []
        for i, route in enumerate(routes_snapshot):
            route_str = format_route(route)
            # Remove truncation logic to show full route
            routes_display.append(f"R{i+1}:{route_str}")
        routes_str = " | ".join(routes_display)

        # Format neighborhood nicely
        # Format neighborhood nicely
        nh_display = neighborhood.replace("_", "(").replace(
            "1", "1)").replace("2", "2)") if neighborhood else "-"
        nh_display = nh_display.replace("()", "").title()

        # Status yang lebih informatif - BERDASARKAN DELTA SEBENARNYA (FLOAT)
        if prev_distance is not None:
            delta_val = total_dist - prev_distance
            if delta_val < -0.001:  # Ada penghematan nyata (toleransi 0.001)
                status_str = "✅ Hemat"
            elif abs(delta_val) <= 0.001:  # Delta mendekati 0
                status_str = "⏸️ Stagnan"
            else:  # Delta positif (lebih buruk)
                status_str = "⏭️ Tidak Ada Perbaikan"
        else:
            status_str = "-"  # Iterasi pertama

        table_data.append({
            "Iter": iter_id,
            "Neighborhood": nh_display,
            "Rute Hasil": routes_str if routes_str else "-",
            "Total Jarak": f"{total_dist:.2f} km",
            "Δ Jarak": delta_str,
            "Status": status_str
        })

        prev_distance = total_dist

    df = pd.DataFrame(table_data)
    st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Rute Hasil": st.column_config.TextColumn("Rute Hasil", width="medium"),
        }
    )


def _display_rvnd_intra_iterations(logs: List[Dict]) -> None:
    """Display RVND intra-route iterations with improved formatting."""
    st.markdown("### 🔁 RVND Intra-Route - Optimasi Dalam Rute")

    intra_logs = [log for log in logs if log.get("phase") == "RVND-INTRA"]

    if not intra_logs:
        st.info("Belum ada iterasi intra-route nih.")
        return

    # === OVERALL SUMMARY ===
    total_iters = len(intra_logs)
    improved_count = sum(1 for log in intra_logs if log.get("improved", False))

    # Count neighborhoods used
    neighborhood_counts = {}
    for log in intra_logs:
        nh = log.get("neighborhood", "none")
        if nh and nh not in ["none", "initial", "stagnan"]:
            neighborhood_counts[nh] = neighborhood_counts.get(nh, 0) + 1

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Iterasi Intra-Route", total_iters)
    with col2:
        st.metric("Total Perbaikan",
                  f"{improved_count} ({improved_count/total_iters*100:.0f}%)" if total_iters > 0 else "0")

    if neighborhood_counts:
        nh_text = " | ".join([f"**{k.replace('_', '-').title()}**: {v}x" for k,
                             v in sorted(neighborhood_counts.items(), key=lambda x: -x[1])])
        st.caption(f"📊 Gerakan yang Berhasil: {nh_text}")

    st.divider()

    # Helper function to format route sequence
    def format_route(seq):
        if isinstance(seq, list):
            # Handle nested list (routes_snapshot often contains [[0,1,2,0]])
            if len(seq) > 0 and isinstance(seq[0], list):
                return "→".join(str(n) for n in seq[0])
            return "→".join(str(n) for n in seq)
        return str(seq)

    # Group by cluster
    clusters = set(log.get("cluster_id", 0)
                   for log in intra_logs if "cluster_id" in log)

    for cluster_id in sorted(clusters):
        cluster_logs = [log for log in intra_logs if log.get(
            "cluster_id") == cluster_id]

        if not cluster_logs:
            continue

        # Calculate cluster-specific metrics
        cluster_improved = sum(
            1 for log in cluster_logs if log.get("improved", False))
        distances = [log.get("total_distance", 0) for log in cluster_logs]
        first_dist = distances[0] if distances else 0
        last_dist = distances[-1] if distances else 0
        delta_pct = ((last_dist - first_dist) / first_dist *
                     100) if first_dist > 0 else 0
        vehicle_type = cluster_logs[0].get("vehicle_type", "N/A") if cluster_logs else "N/A"
        
        v_label = f"({vehicle_type})" if vehicle_type not in ["?", "N/A"] else ""
        with st.expander(f"🚛 Cluster {cluster_id} {v_label} - {len(cluster_logs)} iterasi, {cluster_improved} perbaikan", expanded=False):
            # Cluster summary
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Jarak Awal", f"{first_dist:.2f} km")
            with c2:
                delta_color = "inverse" if delta_pct < 0 else "normal"
                st.metric("Jarak Akhir", f"{last_dist:.2f} km",
                          f"{delta_pct:+.1f}%", delta_color=delta_color)
            with c3:
                st.metric("Penghematan",
                          f"{abs(last_dist - first_dist):.2f} km")

            # Build iteration table with before/after comparison
            table_data = []
            prev_route = None
            prev_distance = None

            for log in cluster_logs:
                iter_id = log.get("iteration_id", log.get("iteration", "?"))
                neighborhood = log.get("neighborhood", "none")
                improved = log.get("improved", False)
                total_dist = log.get("total_distance", 0)
                routes_snapshot = log.get(
                    "routes_snapshot", log.get("sequence_after", []))

                # Current route
                current_route = format_route(routes_snapshot)

                # Before route (from previous iteration)
                before_route = prev_route if prev_route else "-"

                # Delta distance
                if prev_distance is not None:
                    delta = total_dist - prev_distance
                    delta_str = f"{delta:+.2f}"
                else:
                    delta_str = "-"

                # Format neighborhood
                if neighborhood == "initial":
                    nh_display = "🏁 Posisi Awal"
                elif neighborhood == "or_opt":
                    nh_display = "🔀 Or-Opt (Pindah)"
                elif neighborhood == "two_opt":
                    nh_display = "🔄 2-Opt (Tukar)"
                elif neighborhood == "stagnan":
                    nh_display = "⏸️ Stagnan"
                else:
                    nh_display = neighborhood.replace("_", "-").title() if neighborhood and neighborhood != "none" else "🔍 Mencari Peluang"

                table_data.append({
                    "Iter": iter_id,
                    "Move": nh_display,
                    "Rute Sebelum": before_route if before_route != current_route and neighborhood != "initial" else "-", 
                    "Rute Sesudah": current_route,
                    "Jarak": f"{total_dist:.2f} km",
                    "Δ (Selisih)": delta_str,
                    "Status": "✅ Berhasil" if improved else "➖ Tetap"
                })

                prev_route = current_route
                prev_distance = total_dist

            df = pd.DataFrame(table_data)
            st.dataframe(
                df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Rute Sebelum": st.column_config.TextColumn("Rute Sebelum", width="medium"),
                    "Rute Sesudah": st.column_config.TextColumn("Rute Sesudah", width="medium"),
                }
            )

            st.caption("""
            **Keterangan Kolom:**
            *   **Move**: Jenis perbaikan yang dicoba (Swap/Relocate). *(Cek Optimasi)* artinya sedang mencari peluang.
            *   **Delta (Δ)**: Penghematan jarak dibanding iterasi sebelumnya.
            *   **Status**: ✅ Berhasil (Jarak berkurang), ➖ Tetap (Tidak ada perubahan yang lebih baik).
            *   *Catatan:* Jika "Total Perbaikan = 0", berarti rute awal sudah optimal secara lokal.
            """)


def _minutes_to_time(minutes: float) -> str:
    """Convert minutes from midnight to HH:MM format."""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"


def _display_time_window_analysis(result: Dict[str, Any]) -> None:
    """Display detailed time window analysis for each cluster/vehicle."""
    st.markdown("### ⏰ Analisis Jendela Waktu Armada (Fleet Time Window)")
    st.markdown("*Jendela waktu layanan pelanggan dan status kepatuhan operasional*")

    routes = result.get("routes", [])
    dataset = result.get("dataset", {})

    if not routes:
        st.info("Tidak ada rute yang tersedia untuk analisis waktu.")
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

        with st.expander(f"🚛 Cluster {cluster_id} - {vehicle_type}", expanded=True):
            # Depot start info
            if stops:
                depot_stop = stops[0]
                depot_start_time = depot_stop.get(
                    "departure", depot_stop.get("arrival", 480))
                st.info(f"**Waktu Mulai Depot:** {_minutes_to_time(depot_start_time)} | "
                        f"Jendela Waktu Depot: {depot_tw_start} - {depot_tw_end}")

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
                customer_name = customer.get("name", f"Pelanggan {node_id}")

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
                    status = f"❌ TERLAMBAT {violation:.1f} mnt"
                elif wait > 0:
                    status = f"⏳ MENUNGGU {wait:.1f} mnt"
                else:
                    status = "✅ TEPAT WAKTU"

                customer_rows.append({
                    "Pelanggan": customer_name,
                    "Waktu Tiba": _minutes_to_time(raw_arrival),
                    "Mulai Layanan": _minutes_to_time(service_start),
                    "Selesai Layanan": _minutes_to_time(service_end),
                    "TW Mulai": _minutes_to_time(tw_start),
                    "TW Selesai": _minutes_to_time(tw_end),
                    "Status": status
                })

            if customer_rows:
                df = pd.DataFrame(customer_rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Waktu Tunggu", f"{total_wait:.1f} mnt")
                with col2:
                    if total_violation > 0:
                        st.metric(
                            "Pelanggaran TW", f"{total_violation:.1f} mnt", delta="⚠️", delta_color="inverse")
                    else:
                        st.metric("Pelanggaran TW", "0 mnt ✅")
                with col3:
                    compliant = sum(
                        1 for r in customer_rows if "TEPAT WAKTU" in r["Status"] or "MENUNGGU" in r["Status"])
                    st.metric("Tingkat Kesesuaian",
                              f"{compliant}/{len(customer_rows)} pelanggan")
            else:
                st.warning("Tidak ada pemberhentian pelanggan di rute ini.")

    # Overall summary
    st.markdown("---")
    st.markdown("#### 📊 Ringkasan Jadwal Keseluruhan")

    total_routes_compliant = sum(
        1 for r in routes if r.get("total_tw_violation", 0) == 0)
    total_tw_violation = sum(r.get("total_tw_violation", 0) for r in routes)
    total_wait_time = sum(r.get("total_wait_time", 0) for r in routes)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rute Tepat Waktu",
                  f"{total_routes_compliant}/{len(routes)}")
    with col2:
        st.metric("Total Waktu Tunggu", f"{total_wait_time:.1f} mnt")
    with col3:
        if total_tw_violation > 0:
            st.metric("Total Pelanggaran TW",
                      f"{total_tw_violation:.1f} mnt", delta="⚠️", delta_color="inverse")
        else:
            st.metric("Total Pelanggaran TW", "0 mnt ✅")
    with col4:
        if total_tw_violation == 0:
            st.success("Seluruh pengiriman memenuhi batasan waktu operasional.")
        else:
            st.warning("Terdeteksi pelanggaran jadwal pada beberapa rute.")


def _display_user_vehicle_selection(result: Dict[str, Any]) -> None:
    """Display user's vehicle selection and decision reasons."""
    st.markdown("### 🚛 Pemilihan Fleet User")
    st.markdown("*Fleet yang dipilih user di Input Data*")

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
        "Armada": s.get("vehicle_id", s.get("vehicle_name", "?")),
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
        st.metric("Fleet Aktif", len(active_vehicles))
    with col2:
        st.metric("Fleet Tidak Aktif", len(inactive_vehicles))
    with col3:
        st.metric("Total Unit", total_units)
    with col4:
        st.metric("Total Kapasitas", total_capacity)

    # Decision explanation
    st.markdown("---")
    st.markdown("#### 📋 Detail Pemilihan Fleet")

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
    st.info("📌 **Aturan Routing**: Algoritma HANYA menggunakan Fleet yang **aktif** (dicentang). "
            "Fleet yang tidak aktif TIDAK akan digunakan dalam Sweep, NN, ACS, maupun RVND.")


def _display_vehicle_availability(result: Dict[str, Any]) -> None:
    """Display vehicle availability schedule and status."""
    st.markdown("### 🕐 Fleet Availability Schedule")
    st.markdown("*Waktu Ketersediaan Jenis Fleet pada Hari Itu*")

    availability = result.get("vehicle_availability", [])

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
        "Fleet": a.get("vehicle_id", "?"),
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
        st.metric("Total Jenis Fleet", total_vehicles)
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
    st.markdown("### 🚛 Penugasan Ulang Armada (Fleet Reassignment)")
    st.markdown(
        "*Tahap ini memastikan setiap rute dilayani oleh armada dengan kapasitas yang sesuai.*")

    final_routes = result.get("routes", [])
    if not final_routes:
        st.warning("Belum ada rute final.")
        return

    # Build logical reassignment log from final state
    reassignment_log = []

    # Check if we have explicit logs from the reassignment phase
    explicit_logs = [l for l in result.get(
        "iteration_logs", []) if l.get("phase") == "VEHICLE_REASSIGN"]

    if explicit_logs:
        # Use explicit logs if available
        for log in explicit_logs:
            status_icon = "✅" if log.get("status") == "✅ Assigned" else "❌"
            reason_text = log.get('reason', '-')

            # Translate common backend messages to Indonesian
            if "No feasible vehicle" in reason_text:
                reason_text = "Tidak ada armada yang memadai (kapasitas kurang / habis)"
            elif "Demand" in reason_text and "capacity" in reason_text:
                # Keep technical reason but make it cleaner, e.g. "Demand 130 <= capacity 150"
                pass

            reassignment_log.append({
                "Cluster": log.get("cluster_id"),
                "Muatan (Demand)": log.get("demand"),
                "Armada Lama": log.get("old_vehicle", "-"),
                "Armada Baru": log.get("new_vehicle", "-"),
                "Status": f"{status_icon} {log.get('status', '').replace('✅ Assigned', 'Final').replace('❌ No Vehicle', 'Gagal')}",
                "Alasan": reason_text
            })
    else:
        # Fallback: Infer from final routes vs initial if needed, or just show final status
        # For simplicity in this specialized view, we show the final assignment status
        for route in final_routes:
            reassignment_log.append({
                "Cluster": route["cluster_id"],
                "Muatan (Demand)": route["total_demand"],
                "Armada": route["vehicle_type"],
                "Status": "✅ Final",
                "Alasan": "Kapasitas Memadai"
            })

    if reassignment_log:
        df = pd.DataFrame(reassignment_log)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("ℹ️ **Catatan:** Jika *Armada Baru* bernilai `None`, berarti **Stok Armada Habis** atau tidak ada kendaraan yang memiliki *sisa kapasitas* cukup untuk rute tersebut.")


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
        st.info("ℹ️ **Informasi Optimasi**: Sistem ini menggunakan pendekatan *Soft Constraint* untuk batasan waktu operasional (Time Window). Layanan tetap diupayakan pada rute yang tersedia meskipun terjadi keterlambatan (penalti biaya), guna memastikan seluruh pelanggan tetap terlayani.")

    # 2. Per Route Detail
    st.markdown("#### Detail Pelanggaran per Rute")

    detail_data = []
    for r in routes:
        viol = r.get("total_tw_violation", 0)
        wait = r.get("total_wait_time", 0)

        # Menggunakan terminologi professional untuk status rute dengan delay
        status = "✅ Sesuai Jadwal" if viol == 0 else f"⚠️ Terlambat {viol:.1f} mnt (Soft Constraint)"

        detail_data.append({
            "Cluster": r["cluster_id"],
            "Fleet": r["vehicle_type"],
            "Total Pelanggaran (Mnt)": f"{viol:.1f}",
            "Total Tunggu (Mnt)": f"{wait:.1f}",
            "Status Operasional": status
        })

    st.dataframe(pd.DataFrame(detail_data),
                 use_container_width=True, hide_index=True)


def _display_final_results(result: Dict[str, Any]) -> None:
    """Display final routes table with ACTUAL assigned vehicles and VALID costs."""
    st.markdown("### 🏁 Hasil Akhir (Final Routes)")

    routes = result.get("routes", [])
    if not routes:
        st.warning("Tidak ada rute yang terbentuk.")
        return

    # --- 1. PARSE REASSIGNMENT LOGS ---
    reassignment_map = {} 
    explicit_logs = [l for l in result.get("iteration_logs", []) if l.get("phase") == "VEHICLE_REASSIGN"]
    
    failed_clusters = []
    
    for log in explicit_logs:
        c_id = log.get("cluster_id")
        if c_id is not None:
             reassignment_map[c_id] = {
                 "status": log.get("status", ""),
                 "new_vehicle": log.get("new_vehicle", "-"),
                 "reason": log.get("reason", "")
             }
             if "No Vehicle" in log.get("status", ""):
                 failed_clusters.append({
                     "id": c_id,
                     "reason": log.get("reason", ""),
                     "needed": log.get("old_vehicle", "Unknown")
                 })

    # --- 2. RECOMMENDATIONS (If Failures Exist) ---
    if failed_clusters:
        st.error(f"⚠️ **Peringatan: Ada {len(failed_clusters)} Cluster Gagal Dilayani!**")
        
        with st.expander("💡 Rekomendasi Perbaikan (Klik untuk melihat)", expanded=True):
            for fail in failed_clusters:
                st.markdown(f"""
                - **Cluster {fail['id']}**: Gagal karena kehabisan armada (Target awal: **{fail['needed']}**).
                  - 🔧 **Saran**: Tambahkan unit **{fail['needed']}** di tab *Input Data*, atau kurangi muatan pelanggan di area tersebut.
                """)

    data = []
    dataset = result.get("dataset", {})
    fleet_dict = {f["id"]: f for f in dataset.get("fleet", [])}
    costs = result.get("costs", {})
    cost_breakdown = {c["cluster_id"]: c for c in costs.get("breakdown", [])}

    # Tracking Valid Totals
    valid_total_dist = 0
    valid_total_cost = 0
    
    for r in routes:
        cluster_id = r["cluster_id"]
        original_vehicle = r["vehicle_type"]
        
        # Determine ACTUAL vehicle
        display_vehicle = original_vehicle
        status_msg = "✅ Valid"
        is_failed = False
        
        if cluster_id in reassignment_map:
            re_info = reassignment_map[cluster_id]
            re_status = re_info["status"]
            
            if "No Vehicle" in re_status or "Gagal" in re_status:
                display_vehicle = "❌ GAGAL (Stok Habis)"
                status_msg = "❌ Invalid"
                is_failed = True
            elif "Assigned" in re_status:
                display_vehicle = re_info["new_vehicle"]

        # Calculate Duration
        travel_time = r.get('total_travel_time', 0)
        service_time = r['total_service_time']
        wait_time = r.get('total_wait_time', 0)
        total_duration = travel_time + service_time + wait_time

        # Calculate Utilization
        capacity = 0
        if not is_failed:
             if display_vehicle in fleet_dict:
                 capacity = fleet_dict[display_vehicle]["capacity"]
             elif f"Vehicle {display_vehicle}" in fleet_dict:
                 capacity = fleet_dict[f"Vehicle {display_vehicle}"]["capacity"]
             elif original_vehicle in fleet_dict:
                 capacity = fleet_dict[original_vehicle]["capacity"]

        utilization = 0
        if capacity > 0:
            utilization = (r["total_demand"] / capacity) * 100

        # Cost Logic (Show 0 if failed)
        row_cost = 0
        if cluster_id in cost_breakdown and not is_failed:
            row_cost = cost_breakdown[cluster_id]["total_cost"]
            valid_total_cost += row_cost
            valid_total_dist += r['total_distance']
        
        utilization_str = f"{utilization:.1f}%"
        if utilization > 100:
            utilization_str += " ⚠️"
        elif is_failed:
            utilization_str = "-"

        data.append({
            "Cluster": cluster_id,
            "Armada Final": display_vehicle,
            "Rute": str(r["sequence"]),
            "Jarak (km)": f"{r['total_distance']:.2f}" if not is_failed else "(0)",
            "Durasi (menit)": f"{total_duration:.0f}" if not is_failed else "(0)",
            "Utilisasi (%)": utilization_str,
            "Est. Biaya": f"Rp {row_cost:,.0f}" if not is_failed else "Rp 0 (Gagal)",
            "Status": status_msg
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Cost Summary (Valid Only)
    st.success(
        f"💰 **Total Biaya Transportasi (Valid):** Rp {valid_total_cost:,.0f}")
    if valid_total_cost < costs.get('total_cost', 0):
        st.caption(f"*(Total awal Rp {costs.get('total_cost', 0):,.0f} dikurangi rute gagal)*")

    # TIME WINDOW ANALYSIS (NEW)
    st.markdown("---")
    _display_time_window_analysis(result)

    # Costs Details
    st.markdown("---")
    if costs:
        st.markdown("#### Rincian Biaya (Valid Routes Only)")

        clean_breakdown = []
        for c in costs.get("breakdown", []):
            # Check if this cluster failed
            c_id = c["cluster_id"]
            if c_id in reassignment_map and ("Gagal" in reassignment_map[c_id]["status"] or "No Vehicle" in reassignment_map[c_id]["status"]):
                continue # Skip failed routes in cost table
            
            clean_breakdown.append({
                "Cluster": c_id,
                "Armada": c["vehicle_type"],
                "Biaya Tetap": f"Rp {c['fixed_cost']:,.0f}",
                "Biaya Variabel": f"Rp {c['variable_cost']:,.0f}",
                "Total Biaya": f"Rp {c['total_cost']:,.0f}"
            })
            
        if clean_breakdown:
            df = pd.DataFrame(clean_breakdown)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada rute valid untuk dihitung biayanya.")

    # Validation sections removed as requested by user to reduce visual noise
    # Route Structure Validation (REMOVED)
    pass


def render_academic_replay() -> None:
    """Fungsi utama buat nampilin tab Proses Optimasi."""

    # ============================================================
    # SECTION 1: Compact Vehicle Summary Card
    # ============================================================
    user_vehicles = st.session_state.get("user_vehicles", [])

    if not user_vehicles:
        st.error("⚠️ **Tidak ada armada yang didefinisikan!**")
        st.warning(
            "Silakan tambah armada di tab **'Input Data'** terlebih dahulu.")
        return

    # Calculate summary metrics
    active_vehicles = [v for v in user_vehicles if v.get("enabled", True)]
    total_units = sum(v.get("units", 1) for v in active_vehicles)
    total_capacity = sum(v.get("capacity", 0) * v.get("units", 1)
                         for v in active_vehicles)

    # Compact summary card with columns
    st.markdown("#### 🚛 Armada Tersedia")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jenis Aktif",
                  f"{len(active_vehicles)} dari {len(user_vehicles)}")
    with col2:
        st.metric("Total Unit", total_units)
    with col3:
        st.metric("Total Kapasitas", f"{total_capacity} kg")

    # Compact vehicle list using expander
    with st.expander("📋 Lihat Detail Armada", expanded=False):
        vehicle_items = []
        for v in user_vehicles:
            name = v.get("name", v.get("id", "?"))
            cap = v.get("capacity", 0)
            units = v.get("units", 1)
            av_from = v.get("available_from", "08:00")
            av_until = v.get("available_until", "17:00")
            enabled = v.get("enabled", True)

            status_icon = "🟢" if enabled else "🔴"
            vehicle_items.append({
                "": status_icon,
                "Armada": name,
                "Kapasitas": f"{cap} kg",
                "Unit": units,
                "Jam": f"{av_from} - {av_until}"
            })

        st.dataframe(pd.DataFrame(vehicle_items),
                     use_container_width=True, hide_index=True)

    # ============================================================
    # SECTION 2: Run Optimization Button (Inline)
    # ============================================================
    st.markdown("---")

    if st.button("🚀 Jalankan Optimasi", type="primary", use_container_width=True):
        with st.spinner("Menjalankan optimasi MFVRPTW..."):
            try:
                import sys
                import importlib
                sys.path.insert(
                    0, str(Path(__file__).resolve().parent.parent.parent))

                # Force reload to ensure dynamic updates (fixes TypeError on hot reload)
                import academic_replay
                importlib.reload(academic_replay)
                from academic_replay import run_academic_replay

                # Gather dynamic data
                user_vehicles = st.session_state.get("user_vehicles", [])
                points = st.session_state.get("points", {})
                raw_customers = points.get("customers", [])
                input_data = st.session_state.get("inputData", {})
                customer_tw = input_data.get("customerTimeWindows", [])

                # Build customers list
                user_customers = []
                for i, c in enumerate(raw_customers):
                    tw_data = customer_tw[i] if i < len(customer_tw) else {}
                    customer = {
                        "id": c.get("id", i + 1),
                        "name": c.get("name", f"Pelanggan {i + 1}"),
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

                # Depot
                raw_depots = points.get("depots", [])
                user_depot = None
                if raw_depots:
                    d = raw_depots[0]
                    depot_tw = d.get(
                        "time_window", {"start": "08:00", "end": "17:00"})
                    user_depot = {
                        "id": 0,
                        "name": d.get("name", "Depot"),
                        "x": d.get("x", d.get("lng", 0)),
                        "y": d.get("y", d.get("lat", 0)),
                        "time_window": {
                            "start": depot_tw.get("start", "08:00"),
                            "end": depot_tw.get("end", "17:00")
                        },
                        "service_time": d.get("service_time", 0)
                    }

                # ACS Params
                user_acs_params = st.session_state.get("acs_params", None)

                # Distance Multiplier
                distance_multiplier = float(
                    input_data.get("distance_multiplier", 1.0))

                # Run!
                result = run_academic_replay(
                    user_vehicles=user_vehicles,
                    user_customers=user_customers if user_customers else None,
                    user_depot=user_depot,
                    user_acs_params=user_acs_params,
                    distance_multiplier=distance_multiplier
                )
                st.session_state["academic_result"] = result
                st.session_state["result"] = result
                st.session_state["data_validated"] = True
                st.toast("Optimasi selesai!", icon="🎉")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.caption(
        "Klik untuk menjalankan optimasi dengan data armada, pelanggan, dan parameter ACS yang sudah diinput.")

    # ============================================================
    # SECTION 3: Results Display
    # ============================================================
    st.markdown("---")

    result = st.session_state.get("academic_result")
    if not result:
        result = _load_academic_results()
        if result:
            st.session_state["academic_result"] = result
            # Sync to global result if not present
            if "result" not in st.session_state:
                st.session_state["result"] = result
                st.session_state["data_validated"] = True

    if not result:
        st.info(
            "💡 Belum ada hasil. Klik tombol **Jalankan Optimasi** di atas ya.")
        return
    
    # Ensure global state is synced if we have a result
    if st.session_state.get("result") != result:
        st.session_state["result"] = result
        st.session_state["data_validated"] = True

    if result.get("error"):
        st.error(f"❌ Error: {result['error']}")
        return

    logs = result.get("iteration_logs", [])

    # ============================================================
    # SECTION 4: Streamlined Tabs (6 tabs, merged from 7)
    # ============================================================
    tab_sweep, tab_nn, tab_acs, tab_rvnd, tab_final = st.tabs([
        "📐 Sweep",
        "🔗 Nearest Neighbor",
        "🐜 Ant Colony System",
        "🔄 RVND",
        "📊 Hasil Akhir"
    ])

    with tab_sweep:
        _display_sweep_iterations(logs)

    with tab_nn:
        _display_nn_iterations(logs)

    with tab_acs:
        _display_acs_iterations(logs)

    with tab_rvnd:
        # Combined RVND Inter + Intra in one tab
        _display_rvnd_inter_iterations(logs)
        st.markdown("---")
        _display_rvnd_intra_iterations(logs)

    with tab_final:
        _display_vehicle_assignment(result)
        _display_final_results(result)
