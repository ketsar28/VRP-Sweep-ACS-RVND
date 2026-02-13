from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go


def render_graph_hasil(show_labels: bool = True, show_grid: bool = True) -> None:

    data_validated = st.session_state.get("data_validated", False)
    result = st.session_state.get("result") or st.session_state.get("last_pipeline_result")
    
    if not data_validated or not result:
        st.info("ℹ️ **Hasil Visualisasi Belum Tersedia**")
        st.markdown("""
        Untuk menampilkan rute, silakan ikuti langkah berikut:
        1. Validasi data pada tab **'Input Data'**.
        2. Klik tombol optimasi pada tab **'Proses Optimasi'**.
        
        Peta rute akan ditampilkan secara otomatis setelah proses perhitungan selesai.
        """)
        return

    # --- 0. PARSE & DISPLAY REASSIGNMENT FAILURES (RECOMMENDATIONS) ---
    explicit_logs = [l for l in result.get("iteration_logs", []) if l.get("phase") == "VEHICLE_REASSIGN"]
    failed_clusters = []
    failed_cluster_ids = set()

    for log in explicit_logs:
        if "No Vehicle" in log.get("status", "") or "Gagal" in log.get("status", ""):
            c_id = log.get("cluster_id")
            if c_id is not None:
                failed_cluster_ids.add(c_id)
                failed_clusters.append({
                    "id": c_id,
                    "reason": log.get("reason", ""),
                    "needed": log.get("old_vehicle", "Unknown")
                })
    
    if failed_clusters:
        st.error(f"⚠️ **Peringatan: Terdapat {len(failed_clusters)} rute tidak terlayani.** (Ditandai dengan silang abu-abu)")
        
        with st.expander("Analisis Kendala & Solusi", expanded=True):
            for fail in failed_clusters:
                st.markdown(f"""
                - **Rute {fail['id']}**: {fail.get('reason', 'Alokasi Gagal')}
                  - **Saran**: Coba tambahkan unit **{fail['needed']}** pada tab *Input Data* atau sesuaikan jumlah muatan.
                """)
        st.divider()
    # ------------------------------------------------------------


    points = st.session_state.get("points", {"depots": [], "customers": []})
    
    # Menggunakan ID unik untuk menghindari duplikasi lokasi (D: Depot, C: Pelanggan)
    depot_coords = {}
    customer_coords = {}
    
    # Lookup for Details (for Tooltips)
    node_details = {} # id (int) -> string description

    for d in points.get("depots", []):
        depot_id = int(d.get("id", 0))
        depot_coords[f"D{depot_id}"] = (float(d.get("x", 0)), float(d.get("y", 0)), d.get("name", ""))
        # Detail
        tw = d.get("time_window", {})
        node_details[0] = f"<b>{d.get('name', 'Depot')}</b><br>Type: Depot<br>TW: {tw.get('start', '-')} - {tw.get('end', '-')}"
    
    for c in points.get("customers", []):
        cust_id = int(c.get("id", 0))
        customer_coords[f"C{cust_id}"] = (float(c.get("x", 0)), float(c.get("y", 0)), c.get("name", ""))
        # Detail
        tw = c.get("time_window", {})
        node_details[cust_id] = (
            f"<b>{c.get('name', 'Pelanggan')}</b><br>"
            f"ID: {cust_id}<br>"
            f"Demand: {c.get('demand', 0)}<br>"
            f"TW: {tw.get('start', '-')} - {tw.get('end', '-')}<br>"
            f"Service: {c.get('service_time', 0)} min"
        )
    
    # Combined node_map for route lookup (node_id 0 = first depot, node_id 1+ = customers)
    node_map = {}
    
    # Depot is always node 0 in routes
    if points.get("depots"):
        first_depot = points["depots"][0]
        node_map[0] = (float(first_depot.get("x", 0)), float(first_depot.get("y", 0)), first_depot.get("name", "Depot"))
    
    # Customers are nodes 1, 2, 3, ... in routes (by their actual ID)
    for c in points.get("customers", []):
        cust_id = int(c.get("id", 1))
        node_map[cust_id] = (float(c.get("x", 0)), float(c.get("y", 0)), c.get("name", ""))

    fig = go.Figure()

    # Apply Grid Toggle
    fig.update_xaxes(showgrid=show_grid, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=show_grid, gridwidth=1, gridcolor='lightgray')

    # Color palette for distinct routes
    route_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # (Failed clusters identified above)

    # Gambar rutenya satu-satu (biar nggak nyambung-nyambung antar depot)
    routes = result.get("routes", [])
    for idx, route in enumerate(routes):
        cluster_id = route.get("cluster_id", idx + 1)
        
        # SKIP PLOTTING IF FAILED
        if cluster_id in failed_cluster_ids:
            continue
            
        seq = route.get("sequence") or []

        xs = []
        ys = []
        hover_texts = [] # Custom hover for route points
        
        # Build x,y lists for THIS route only
        for nid in seq:
            if nid in node_map:
                x, y, _ = node_map[nid]
                xs.append(x)
                ys.append(y)
                # Lookup details
                nid_int = int(nid)
                detail = node_details.get(nid_int, f"Node {nid}")
                route_info = f"<br><i>Rute Cluster {cluster_id}</i>"
                hover_texts.append(detail + route_info)
        
        # Plot this route as a separate trace
        if xs and ys and len(xs) >= 2:
            color = route_colors[idx % len(route_colors)]
            vehicle = route.get("vehicle_type", f"Route {idx+1}")
            
            fig.add_trace(go.Scatter(
                x=xs, 
                y=ys, 
                mode="lines+markers", 
                line=dict(color=color, width=2), 
                marker=dict(size=6, color=color), 
                name=f"Cluster {cluster_id} ({vehicle})",
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                showlegend=True
            ))

    # draw depots as yellow star - use depot_coords to avoid collision
    depot_x = []
    depot_y = []
    depot_names = []
    depot_hovers = []
    
    for d in points.get("depots", []):
        depot_id = int(d.get("id", 0))
        key = f"D{depot_id}"
        if key in depot_coords:
            x, y, name = depot_coords[key]
            depot_x.append(x)
            depot_y.append(y)
            depot_names.append(name)
            # 0 is usually the main depot ID for details
            # If multiple depots, we iterate. Using 0 for main.
            # But let's check exact ID matching.
            depot_hovers.append(node_details.get(0, name)) 
    
    if depot_x and depot_y:
        fig.add_trace(go.Scatter(
            x=depot_x, 
            y=depot_y, 
            mode="markers+text" if show_labels else "markers", 
            marker_symbol="star", 
            marker=dict(size=16, color="yellow", line=dict(color="black", width=1)), 
            text=depot_names, 
            hovertext=depot_hovers,
            hovertemplate="%{hovertext}<extra></extra>",
            textposition="top center", 
            name="Depot"
        ))

    # --- IDENTIFY SERVED vs UNSERVED CUSTOMERS (ROBUST WAY) ---
    all_customer_ids = set()
    customer_demand_map = {}
    for c in points.get("customers", []):
        cid = int(c.get("id", 0))
        all_customer_ids.add(cid)
        customer_demand_map[cid] = c.get("demand", 0)

    served_customer_ids = set()
    served_demand = 0
    
    # Collect served IDs from VALID routes only
    for route in routes:
        cid = route.get("cluster_id")
        if cid not in failed_cluster_ids:
            seq = route.get("sequence") or []
            for nid in seq:
                try:
                    nid_int = int(nid)
                    if nid_int != 0 and nid_int in all_customer_ids: # 0 is depot
                        served_customer_ids.add(nid_int)
                        served_demand += customer_demand_map.get(nid_int, 0)
                except (ValueError, TypeError):
                    pass
    
    unserved_customer_ids = all_customer_ids - served_customer_ids
    unserved_demand = sum(customer_demand_map.get(uid, 0) for uid in unserved_customer_ids)
    total_demand = served_demand + unserved_demand

    # Separate coordinates for plotting
    served_x, served_y, served_names, served_hovers = [], [], [], []
    unserved_x, unserved_y, unserved_names, unserved_hovers = [], [], [], []
    
    for c in points.get("customers", []):
        cust_id = int(c.get("id", 0))
        key = f"C{cust_id}"
        if key in customer_coords:
            x, y, name = customer_coords[key]
            detail = node_details.get(cust_id, name)
            
            if cust_id in unserved_customer_ids:
                unserved_x.append(x)
                unserved_y.append(y)
                unserved_names.append(f"{name} (Gagal)")
                unserved_hovers.append(f"{detail}<br><b style='color:red'>Status: Gagal/Tidak Terlayani</b>")
            else:
                served_x.append(x)
                served_y.append(y)
                served_names.append(name)
                served_hovers.append(detail)
    
    # Draw Served Customers (Red)
    if served_x:
        fig.add_trace(go.Scatter(
            x=served_x, y=served_y, 
            mode="markers+text" if show_labels else "markers", 
            marker=dict(size=8, color="red"), 
            text=served_names, 
            hovertext=served_hovers,
            textposition="bottom center",
            hovertemplate="%{hovertext}<extra></extra>",
            name="Pelanggan (Terlayani)"
        ))

    # Draw Unserved Customers (Grey X)
    if unserved_x:
        fig.add_trace(go.Scatter(
            x=unserved_x, y=unserved_y, 
            mode="markers+text" if show_labels else "markers", 
            marker=dict(size=10, color="grey", symbol="x-thin", line=dict(width=2)), 
            text=unserved_names,
            hovertext=unserved_hovers,
            hovertemplate="%{hovertext}<extra></extra>",
            name="Pelanggan (Tidak Terlayani)"
        ))

    fig.update_layout(
        height=600, 
        xaxis_title="X", 
        yaxis_title="Y",
        showlegend=True,
        # Move legend to bottom, bold title
        legend=dict(
            title=dict(text="<b>Keterangan</b>", side="top center"),
            orientation="h", 
            yanchor="top", 
            y=-0.15, 
            xanchor="center", 
            x=0.5
        ),
        title="Visualisasi Rute (Setiap rute independen)",
        margin=dict(b=100) # Add bottom margin for legend
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- SUMMARY SECTION (REQUESTED BY USER) ---
    st.markdown("### 📊 Ringkasan Pelayanan & Kapasitas")
    
    # Calculate Total Capacity of ACTIVE fleets

    # We need to access fleet data. Assuming it's passed or available. 
    # Since 'points' doesn't have fleet info, we rely on 'result.get("dataset")' or 'user_vehicle_selection' logic
    # Try to extract from result if available, else standard fallback
    dataset = result.get("dataset", {})
    available_vehicles = result.get("available_vehicles", [])
    fleet_info = dataset.get("fleet", [])
    
    active_capacity = 0
    
    # Try using explicit vehicle logs first
    vehicle_logs = result.get("vehicle_availability", [])
    if vehicle_logs:
        for vlog in vehicle_logs:
            if vlog.get("available", False):
                active_capacity += vlog.get("capacity", 0) * vlog.get("units", 1)
    else:
        # Fallback: Sum capacity of all available fleet types * their units (if known) or just basic sum
        for v in fleet_info:
            if v["id"] in available_vehicles:
                # Assuming 1 unit if not specified, or use v.get("units", 1) which might be in dataset
                active_capacity += v.get("capacity", 0) * v.get("units", 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Kapasitas Armada", f"{active_capacity} kg")
    c2.metric("Total Permintaan (Demand)", f"{total_demand} kg")
    c3.metric("Terlayani", f"{served_demand} kg", f"{len(served_customer_ids)} cust")
    c4.metric("Tidak Terlayani", f"{unserved_demand} kg", f"{len(unserved_customer_ids)} cust", delta_color="inverse")

    st.divider()

    # Detailed Lists
    col_served, col_unserved = st.columns(2)
    
    with col_served:
        with st.expander(f"✅ Pelanggan Terlayani ({len(served_customer_ids)})", expanded=True):
            if served_customer_ids:
                served_list = []
                for cid in sorted(served_customer_ids):
                    cdata = next((c for c in points["customers"] if c["id"] == cid), {})
                    served_list.append({
                        "ID": cid,
                        "Nama": cdata.get("name", ""),
                        "Muatan (kg)": cdata.get("demand", 0)
                    })
                st.dataframe(served_list, hide_index=True, use_container_width=True)
            else:
                st.info("Belum ada pelanggan terlayani.")

    with col_unserved:
        with st.expander(f"❌ Pelanggan Tidak Terlayani ({len(unserved_customer_ids)})", expanded=True):
            if unserved_customer_ids:
                unserved_list = []
                for cid in sorted(unserved_customer_ids):
                    cdata = next((c for c in points["customers"] if c["id"] == cid), {})
                    unserved_list.append({
                        "ID": cid,
                        "Nama": cdata.get("name", ""),
                        "Muatan (kg)": cdata.get("demand", 0)
                    })
                st.dataframe(unserved_list, hide_index=True, use_container_width=True)
                st.warning("⚠️ Pelanggan ini tidak masuk dalam rute, kemungkinan karena kapasitas penuh atau batasan waktu operasional.")
            else:
                st.success("Semua pelanggan terlayani!")
