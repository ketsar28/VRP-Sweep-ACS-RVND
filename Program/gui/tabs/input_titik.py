from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any
import sys
from pathlib import Path

# Add gui directory to path to allow importing utils
_gui_dir = Path(__file__).resolve().parent.parent
if str(_gui_dir) not in sys.path:
    sys.path.insert(0, str(_gui_dir))

try:
    from utils import save_to_autosave
except ImportError:
    def save_to_autosave(): pass


def _get_next_depot_id(depots: list) -> int:
    """Mengecek ID depot selanjutnya agar tidak duplikat."""
    if not depots:
        return 0
    return max(int(d.get("id", 0)) for d in depots) + 1


def _get_next_customer_id(customers: list) -> int:
    """Mengecek ID pelanggan selanjutnya."""
    if not customers:
        return 1
    return max(int(c.get("id", 1)) for c in customers) + 1


def render_input_titik(show_labels: bool = True, show_grid: bool = True) -> None:
    # MIGRATION: Rename "Customer X" to "Pelanggan X" in existing data
    if "points" in st.session_state:
        for c in st.session_state["points"].get("customers", []):
            if c.get("name", "").startswith("Customer "):
                c["name"] = c["name"].replace("Customer ", "Pelanggan ")

    # Inisialisasi session state jika belum ada
    if "points" not in st.session_state:
        st.session_state["points"] = {"depots": [], "customers": []}

    if "point_type" not in st.session_state:
        st.session_state["point_type"] = "Depot"

    points = st.session_state["points"]

    # ------------------ SIDEBAR ------------------
    with st.sidebar:
        st.header("📊 Statistik Data")

        # Calculate metrics
        total_depots = len(points["depots"])
        total_customers = len(points["customers"])
        total_demand = sum(c.get("demand", 0) for c in points["customers"])

        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Depot", total_depots)
        col_s2.metric("Pelanggan", total_customers)
        st.metric("Total Permintaan", f"{total_demand:,.0f}")

        st.divider()

        # st.header("⚙️ Kontrol Peta")
        # show_labels = st.toggle("Tampilkan Label", value=True)
        # show_grid = st.toggle("Tampilkan Grid", value=True)

    # Peta untuk melihat sebaran koordinat titik
    st.subheader("🗺️ Peta Titik")

    st.write("**Mode:**")
    point_type = st.radio(
        "Tipe:", ("Depot", "Pelanggan"), key="point_type_radio",
        label_visibility="collapsed", horizontal=True
    )
    st.session_state["point_type"] = point_type

    # Create interactive figure
    fig = go.Figure()

    # Add grid
    fig.update_xaxes(
        showgrid=show_grid, gridwidth=1, gridcolor='lightgray',
        zeroline=False, range=[0, 100]
    )
    fig.update_yaxes(
        showgrid=show_grid, gridwidth=1, gridcolor='lightgray',
        zeroline=False, range=[0, 100]
    )

    # Plot depots
    depot_x = [d["x"] for d in points["depots"]]
    depot_y = [d["y"] for d in points["depots"]]
    depot_names = [d["name"] for d in points["depots"]]

    if depot_x:
        fig.add_trace(go.Scatter(
            x=depot_x, y=depot_y,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(size=15, color='gold', symbol='star',
                        line=dict(width=2, color='orange')),
            text=depot_names,
            textposition='top center',
            name='Depot',
            hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<extra></extra>'
        ))

    # Plot customers
    cust_x = [c["x"] for c in points["customers"]]
    cust_y = [c["y"] for c in points["customers"]]
    cust_names = [c["name"] for c in points["customers"]]

    if cust_x:
        fig.add_trace(go.Scatter(
            x=cust_x, y=cust_y,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(size=12, color='red', symbol='circle',
                        line=dict(width=2, color='darkred')),
            text=cust_names,
            textposition='top center',
            name='Pelanggan',
            hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<extra></extra>'
        ))

    fig.update_layout(
        height=500,
        hovermode='closest',
        margin=dict(l=0, r=0, t=10, b=0),  # Maximize space
        xaxis_title="X",
        yaxis_title="Y",
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)"
    )

    # Display canvas with click detection
    st.plotly_chart(fig, use_container_width=True, key="canvas_plot")

    # Input koordinat secara manual (opsional)
    with st.expander("➕ Input Koordinat Manual"):
        # Layout: X, Y, and Add Button side-by-side
        c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment="bottom")

        with c1:
            x_coord = st.number_input(
                "Koordinat X", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key="manual_x")
        with c2:
            y_coord = st.number_input(
                "Koordinat Y", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key="manual_y")
        with c3:
            # Button is aligned to bottom to match input fields
            if st.button("Tambahkan", key="btn_add_manual", use_container_width=True, type="primary"):
                if st.session_state["point_type"] == "Depot":
                    depot_id = _get_next_depot_id(points["depots"])
                    points["depots"].append({
                        "id": depot_id,
                        "name": f"Depot {depot_id}",
                        "x": x_coord,
                        "y": y_coord,
                        "time_window": {"start": "08:30", "end": "17:00"},
                        "service_time": 0
                    })
                    st.toast(f"✅ Depot {depot_id} ditambahkan!", icon="🏭")
                else:
                    customer_id = _get_next_customer_id(points["customers"])
                    points["customers"].append({
                        "id": customer_id,
                        "name": f"Pelanggan {customer_id}",
                        "x": x_coord,
                        "y": y_coord,
                        "demand": 0.0,
                        "time_window": {"start": "08:30", "end": "17:00"},
                        "service_time": 0
                    })
                    st.toast(
                        f"✅ Pelanggan {customer_id} ditambahkan!", icon="🏢")

                st.session_state["points"] = points
                save_to_autosave()
                st.rerun()

    # Reset button moved to bottom right or main tools area
    # st.markdown("---")
    if st.button("🔄 Reset Semua Titik", key="btn_reset_canvas"):
        st.session_state["points"] = {"depots": [], "customers": []}
        save_to_autosave()
        st.rerun()

    # Display list of added points
    st.markdown("---")
    st.subheader("📋 Daftar Titik")

    tab_depots, tab_customers = st.tabs(
        ["🏭 Daftar Depot", "🏢 Daftar Pelanggan"])

    # ------------------ DEPOT EDITOR ------------------
    with tab_depots:
        if points["depots"]:
            df_depots = pd.DataFrame(points["depots"])
            # Ensure required columns exist
            for col in ["id", "name", "x", "y"]:
                if col not in df_depots.columns:
                    df_depots[col] = None

            # Show only relevant columns
            df_depots_view = df_depots[["id", "name", "x", "y"]]

            edited_df_depots = st.data_editor(
                df_depots_view,
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "name": st.column_config.TextColumn("Nama Depot", required=True),
                    "x": st.column_config.NumberColumn("X", min_value=0, max_value=100, format="%.2f"),
                    "y": st.column_config.NumberColumn("Y", min_value=0, max_value=100, format="%.2f"),
                },
                hide_index=True,
                num_rows="dynamic",
                key="editor_depots",
                use_container_width=True
            )

            # Reconstruct list from editor result
            new_depots_list = []
            original_map = {d["id"]: d for d in points["depots"]}

            for index, row in edited_df_depots.iterrows():
                row_id = row.get("id")

                # Check for NaN (newly added row)
                if pd.isna(row_id):
                    # Generate new ID
                    current_ids = [d["id"] for d in points["depots"]
                                   ] + [d.get("id", 0) for d in new_depots_list]
                    new_id = (max(current_ids) if current_ids else -1) + 1

                    new_d = {
                        "id": int(new_id),
                        "name": row.get("name") or f"Depot {new_id}",
                        "x": float(row.get("x", 50.0)),
                        "y": float(row.get("y", 50.0)),
                        "time_window": {"start": "08:30", "end": "17:00"},
                        "service_time": 0
                    }
                    new_depots_list.append(new_d)
                else:
                    # Existing row
                    curr_id = int(row_id)
                    if curr_id in original_map:
                        d = original_map[curr_id].copy()
                        d["name"] = row["name"]
                        d["x"] = float(row["x"])
                        d["y"] = float(row["y"])
                        new_depots_list.append(d)

            if new_depots_list != points["depots"]:
                points["depots"] = new_depots_list
                st.session_state["points"] = points
                save_to_autosave()
                st.rerun()
        else:
            st.info(
                "ℹ️ Belum ada depot. Silakan tambahkan melalui peta atau form input manual.")

    # ------------------ CUSTOMER EDITOR ------------------
    with tab_customers:
        if points["customers"]:
            df_customers = pd.DataFrame(points["customers"])
            # Ensure columns
            for col in ["id", "name", "x", "y", "demand"]:
                if col not in df_customers.columns:
                    if col == "demand":
                        df_customers[col] = 0.0
                    else:
                        df_customers[col] = None

            # Show relevant columns including Demand
            df_customers_view = df_customers[[
                "id", "name", "x", "y", "demand"]]

            edited_df_cust = st.data_editor(
                df_customers_view,
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "name": st.column_config.TextColumn("Nama Pelanggan", required=True),
                    "x": st.column_config.NumberColumn("X", min_value=0, max_value=100, format="%.2f"),
                    "y": st.column_config.NumberColumn("Y", min_value=0, max_value=100, format="%.2f"),
                    "demand": st.column_config.NumberColumn("Demand", min_value=0, format="%d"),
                },
                hide_index=True,
                num_rows="dynamic",
                key="editor_customers",
                use_container_width=True
            )

            new_cust_list = []
            original_cust_map = {c["id"]: c for c in points["customers"]}

            for index, row in edited_df_cust.iterrows():
                row_id = row.get("id")

                if pd.isna(row_id):
                    current_ids = [c["id"] for c in points["customers"]
                                   ] + [c.get("id", 0) for c in new_cust_list]
                    new_id = (max(current_ids) if current_ids else 0) + 1

                    new_c = {
                        "id": int(new_id),
                        "name": row.get("name") or f"Pelanggan {new_id}",
                        "x": float(row.get("x", 50.0)),
                        "y": float(row.get("y", 50.0)),
                        "demand": float(row.get("demand", 0.0)),
                        # Defaults for hidden fields
                        "time_window": {"start": "08:30", "end": "17:00"},
                        "service_time": 0
                    }
                    new_cust_list.append(new_c)
                else:
                    curr_id = int(row_id)
                    if curr_id in original_cust_map:
                        c = original_cust_map[curr_id].copy()
                        c["name"] = row["name"]
                        c["x"] = float(row["x"])
                        c["y"] = float(row["y"])
                        c["demand"] = float(row["demand"])
                        new_cust_list.append(c)

            if new_cust_list != points["customers"]:
                points["customers"] = new_cust_list
                st.session_state["points"] = points
                save_to_autosave()
                st.rerun()
        else:
            st.info(
                "ℹ️ Belum ada pelanggan. Silakan tambahkan melalui peta atau form input manual.")

        st.session_state["points"] = points


