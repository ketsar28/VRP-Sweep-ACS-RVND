from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any


def render_graph_hasil() -> None:
    st.header("Graph Hasil")
    data_validated = st.session_state.get("data_validated", False)
    result = st.session_state.get("result") or st.session_state.get("last_pipeline_result")
    if not data_validated or not result:
        st.info("Belum ada hasil untuk divisualisasikan. Tekan 'Hasil' di menu 'Input Data' terlebih dahulu.")
        return

    points = st.session_state.get("points", {"depots": [], "customers": []})
    
    # FIX: Use SEPARATE namespaces to avoid ID collision
    # Depot uses key "D{id}", Customer uses key "C{id}"
    # This prevents Depot 1 and Customer 1 from colliding
    depot_coords = {}  # "D{id}" -> (x, y, name)
    customer_coords = {}  # "C{id}" -> (x, y, name)
    
    for d in points.get("depots", []):
        depot_id = int(d.get("id", 0))
        depot_coords[f"D{depot_id}"] = (float(d.get("x", 0)), float(d.get("y", 0)), d.get("name", ""))
    
    for c in points.get("customers", []):
        cust_id = int(c.get("id", 0))
        customer_coords[f"C{cust_id}"] = (float(c.get("x", 0)), float(c.get("y", 0)), c.get("name", ""))
    
    # Combined node_map for route lookup (node_id 0 = first depot, node_id 1+ = customers)
    # Routes use: 0 = depot, 1-N = customers
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

    # Color palette for distinct routes
    route_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # draw routes - EACH ROUTE IS INDEPENDENT (no depot-to-depot connections)
    routes = result.get("routes", [])
    for idx, route in enumerate(routes):
        seq = route.get("sequence") or []
        xs = []
        ys = []
        # Build x,y lists for THIS route only
        for nid in seq:
            if nid in node_map:
                x, y, _ = node_map[nid]
                xs.append(x)
                ys.append(y)
        
        # Plot this route as a separate trace (NOT connected to other routes)
        if xs and ys and len(xs) >= 2:
            color = route_colors[idx % len(route_colors)]
            vehicle = route.get("vehicle_type", f"Route {idx+1}")
            cluster_id = route.get("cluster_id", idx + 1)
            fig.add_trace(go.Scatter(
                x=xs, 
                y=ys, 
                mode="lines+markers", 
                line=dict(color=color, width=2), 
                marker=dict(size=6, color=color), 
                name=f"Cluster {cluster_id} ({vehicle})",
                showlegend=True
            ))

    # draw depots as yellow star - use depot_coords to avoid collision
    depot_x = []
    depot_y = []
    depot_names = []
    for d in points.get("depots", []):
        depot_id = int(d.get("id", 0))
        key = f"D{depot_id}"
        if key in depot_coords:
            x, y, name = depot_coords[key]
            depot_x.append(x)
            depot_y.append(y)
            depot_names.append(name)
    
    if depot_x and depot_y:
        fig.add_trace(go.Scatter(x=depot_x, y=depot_y, mode="markers+text", marker_symbol="star", marker=dict(size=16, color="yellow", line=dict(color="black", width=1)), text=depot_names, textposition="top center", name="Depots"))

    # draw customers as red circles - use customer_coords to avoid collision
    cust_x = []
    cust_y = []
    cust_names = []
    for c in points.get("customers", []):
        cust_id = int(c.get("id", 0))
        key = f"C{cust_id}"
        if key in customer_coords:
            x, y, name = customer_coords[key]
            cust_x.append(x)
            cust_y.append(y)
            cust_names.append(name)
    
    if cust_x and cust_y:
        fig.add_trace(go.Scatter(x=cust_x, y=cust_y, mode="markers+text", marker=dict(size=8, color="red"), text=cust_names, textposition="bottom center", name="Customers"))

    fig.update_layout(
        height=600, 
        xaxis_title="X", 
        yaxis_title="Y", 
        showlegend=True,
        legend=dict(title="Routes", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title="Route Visualization (Each route is independent)"
    )

    st.plotly_chart(fig, use_container_width=True)
