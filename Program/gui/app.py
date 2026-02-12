"""GUI Streamlit untuk nampilin hasil rute VRP-TW.

Aplikasi ini cuma nampilin hasil optimasi yang sudah diproses (Sweep -> NN -> ACS -> RVND).
Data diambil langsung dari file JSON/Markdown hasil proses.
"""

from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Compatibility fallback: some Streamlit versions expose `data_editor` instead
# of `experimental_data_editor`. Ensure the code in tabs using
# `st.experimental_data_editor` keeps working across versions.
if not hasattr(st, "experimental_data_editor") and hasattr(st, "data_editor"):
    st.experimental_data_editor = st.data_editor


def _load_agents_module() -> object:
    agents_path = Path(__file__).resolve().parent / "agents.py"
    spec = importlib.util.spec_from_file_location("gui_agents", agents_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


# Add tabs folder to path for imports
_tabs_dir = Path(__file__).resolve().parent / "tabs"
if str(_tabs_dir) not in sys.path:
    sys.path.insert(0, str(_tabs_dir))

# Add project root to path for absolute imports (e.g. Program.gui.utils)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from input_titik import render_input_titik
except ImportError as e:
    st.error(f"Error loading Input Titik: {e}")
    def render_input_titik(): return st.error("Input Titik module not found")

try:
    from input_data import render_input_data
except ImportError as e:
    st.error(f"Error loading Input Data: {e}")
    def render_input_data(): return st.error("Input Data module not found")

try:
    from hasil import render_hasil
except ImportError as e:
    st.error(f"Error loading Hasil: {e}")
    def render_hasil(): return st.error("Hasil module not found")

try:
    from graph_hasil import render_graph_hasil
except ImportError as e:
    st.error(f"Error loading Graph Hasil: {e}")
    def render_graph_hasil(): return st.error("Graph Hasil module not found")

try:
    from utils import load_from_autosave, save_to_autosave
except ImportError as e:
    # Fallback if utils not found or error
    print(f"Error loading utils: {e}")
    def load_from_autosave(): return False
    def save_to_autosave(): pass

try:
    from academic_replay_tab import render_academic_replay
except ImportError:
    # Silently handle - academic replay is optional
    render_academic_replay = None


def _build_state_from_parsed(instance: Dict, parsed_distance: Dict) -> Dict:
    points = {"depots": [], "customers": []}
    if "depot" in instance:
        points["depots"].append(instance["depot"])
    for cust in instance.get("customers", []):
        points["customers"].append(cust)

    inputData = {
        "customerDemand": [c.get("demand", 0) for c in instance.get("customers", [])],
        "distanceMatrix": parsed_distance.get("distance_matrix", []),
    }
    return {"points": points, "inputData": inputData}


def _build_state_from_ui() -> Dict:
    """Construct the coordinator state from Streamlit session_state (UI editors).

    Looks for `points` and `inputData` (or individual `distanceMatrix` / `customerDemand`).
    """
    pts = st.session_state.get("points", {})
    input_data = st.session_state.get("inputData") or {}
    # fallback individual keys
    if not input_data:
        input_data = {}
        if "distanceMatrix" in st.session_state:
            input_data["distanceMatrix"] = st.session_state.get(
                "distanceMatrix")
        elif "distance_matrix" in st.session_state:
            input_data["distanceMatrix"] = st.session_state.get(
                "distance_matrix")
        if "customerDemand" in st.session_state:
            input_data["customerDemand"] = st.session_state.get(
                "customerDemand")
        elif "demands" in st.session_state:
            input_data["customerDemand"] = st.session_state.get("demands")
    return {"points": pts, "inputData": input_data}


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"

FINAL_SOLUTION_PATH = DATA_DIR / "final_solution.json"
FINAL_SUMMARY_PATH = DOCS_DIR / "final_summary.md"
PARSED_INSTANCE_PATH = DATA_DIR / "parsed_instance.json"
PARSED_DISTANCE_PATH = DATA_DIR / "parsed_distance.json"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_markdown(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def parse_markdown_table(markdown_text: str, header: str) -> pd.DataFrame:
    """Extract table that immediately follows the given markdown header."""
    lines = markdown_text.splitlines()
    table_lines: List[str] = []
    capture = False
    for line in lines:
        if line.strip().startswith("##") and header.lower() in line.strip().lower():
            capture = True
            continue
        if capture:
            if not line.strip():
                break
            table_lines.append(line)
    if not table_lines:
        return pd.DataFrame()

    # First line contains header row, second contains separator
    header_row = [cell.strip()
                  for cell in table_lines[0].split("|") if cell.strip()]
    data_rows = []
    for row in table_lines[2:]:
        cells = [cell.strip() for cell in row.split("|") if cell.strip()]
        if cells:
            data_rows.append(cells)
    return pd.DataFrame(data_rows, columns=header_row)


def prepare_route_table(final_solution: Dict, instance_data: Dict) -> pd.DataFrame:
    fleet_lookup = {fleet["id"]: fleet for fleet in instance_data["fleet"]}
    rows = []
    for route in final_solution["routes"]:
        vehicle = fleet_lookup[route["vehicle_type"]]
        fixed_cost = vehicle["fixed_cost"]
        variable_cost = vehicle["variable_cost_per_km"] * \
            route["total_distance"]
        rows.append(
            {
                "Cluster": route["cluster_id"],
                "Fleet": route["vehicle_type"],
                "Route": " → ".join(map(str, route["sequence"])),
                "Distance (km)": round(route["total_distance"], 3),
                "Time (min)": round(route["total_time_component"], 3),
                "TW Violation (min)": round(route["total_tw_violation"], 3),
                "Objective": round(route["objective"], 3),
                "Cost (Rp)": round(fixed_cost + variable_cost, 2),
            }
        )
    return pd.DataFrame(rows)


def build_route_plot(final_solution: Dict, instance_data: Dict) -> go.Figure:
    nodes = {instance_data["depot"]["id"]: instance_data["depot"]}
    nodes.update({cust["id"]: cust for cust in instance_data["customers"]})

    depot_id = instance_data["depot"]["id"]

    fig = go.Figure()

    colour_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for index, route in enumerate(final_solution["routes"]):
        colour = colour_palette[index % len(colour_palette)]
        seq = route["sequence"]
        x_values = [nodes[node_id]["x"] for node_id in seq]
        y_values = [nodes[node_id]["y"] for node_id in seq]

        hover_text = []
        for node_id in seq:
            node = nodes[node_id]
            if node_id == depot_id:
                demand = "-"
                tw_start = instance_data["depot"]["time_window"]["start"]
                tw_end = instance_data["depot"]["time_window"]["end"]
            else:
                demand = node["demand"]
                tw_start = node["time_window"]["start"]
                tw_end = node["time_window"]["end"]
            hover_text.append(
                f"Node {node_id}<br>Demand: {demand}<br>TW: {tw_start} – {tw_end}"
            )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=f"Cluster {route['cluster_id']} ({route['vehicle_type']})",
                line=dict(color=colour, width=3),
                marker=dict(size=10),
                hoverinfo="text",
                text=hover_text,
            )
        )

    depot = nodes[depot_id]
    fig.add_trace(
        go.Scatter(
            x=[depot["x"]],
            y=[depot["y"]],
            mode="markers",
            name="Depot",
            marker=dict(size=14, color="#000000", symbol="diamond"),
            hoverinfo="text",
            text=["Depot"],
        )
    )

    fig.update_layout(
        title="Route Layout",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        legend=dict(title="Clusters"),
        template="plotly_white",
        height=650,
    )
    return fig


def render_kpis(summary: Dict) -> None:
    st.markdown("## Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Distance (km)", f"{summary['total_distance']:.3f}")
    col2.metric("Total Cost (Rp)", f"{summary['total_cost']:,.0f}")
    col3.metric("TW Violations (min)", f"{summary['total_tw_violation']:.3f}")
    fleet_usage = ", ".join(
        f"{ftype}: {summary['fleet_usage'][ftype]}"
        for ftype in sorted(summary["fleet_usage"].keys())
    )
    col4.metric("Fleet Usage", fleet_usage)


def render_cluster_details(route_table: pd.DataFrame) -> None:
    st.markdown("## Cluster Detail")
    for _, row in route_table.sort_values("Cluster").iterrows():
        with st.expander(f"Cluster {row['Cluster']} – Fleet {row['Fleet']}"):
            st.write(
                {
                    "Route": row["Route"],
                    "Distance (km)": row["Distance (km)"],
                    "Time (min)": row["Time (min)"],
                    "TW Violation (min)": row["TW Violation (min)"],
                    "Objective": row["Objective"],
                    "Cost (Rp)": row["Cost (Rp)"]
                }
            )


def render_comparison_table(final_summary_md: str) -> None:
    comparison_df = parse_markdown_table(final_summary_md, "ACS vs.")
    if comparison_df.empty:
        st.warning("ACS vs. RVND comparison table not found in summary.")
        return
    st.markdown("## ACS vs. RVND Comparison")
    comparison_df = comparison_df.replace({"**": ""}, regex=False)
    if "Cluster" in comparison_df.columns:
        comparison_df["Cluster"] = comparison_df["Cluster"].str.replace(
            "**", "", regex=False)
    numeric_columns = [
        col for col in comparison_df.columns if col not in {"Cluster"}]
    for column in numeric_columns:
        comparison_df[column] = (
            comparison_df[column]
            .str.replace(",", "")
            .str.replace("**", "", regex=False)
            .astype(float)
        )
    st.dataframe(
        comparison_df.style.format({col: "{:.3f}" for col in numeric_columns})
    )


def render_export_section(final_solution: Dict, route_table: pd.DataFrame, final_summary_md: str) -> None:
    st.markdown("## Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download final_solution.json",
            data=json.dumps(final_solution, indent=2).encode("utf-8"),
            file_name="final_solution.json",
            mime="application/json",
        )
    with col2:
        csv_data = route_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download per-cluster summary (CSV)",
            data=csv_data,
            file_name="cluster_summary.csv",
            mime="text/csv",
        )
    st.download_button(
        "Download Markdown summary",
        data=final_summary_md.encode("utf-8"),
        file_name="final_summary.md",
        mime="text/markdown",
    )


def main() -> None:
    st.set_page_config(
        page_title="Route Optimizer | Nabilah Eva Nurhayati",
        layout="wide",
        page_icon="🚚"
    )

    # ============================================================
    # AUTOSAVE CHECK
    # ============================================================
    # Check if we need to load from autosave (only if session is empty/fresh)
    if "points" not in st.session_state or not st.session_state["points"].get("depots"):
        if load_from_autosave():
            st.toast("Data terakhir dipulihkan", icon="♻️")

    # ============================================================
    # CUSTOM CSS FOR BETTER UI
    # ============================================================
    st.markdown("""
    <style>
        /* Header styling */
        .main-header {
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 1rem;
        }
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .main-header p {
            font-size: 0.9rem;
            opacity: 0.7;
        }
        
        /* Footer styling - Clean & Static */
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            padding-bottom: 20px;
            text-align: center;
            font-size: 0.85rem;
            color: #888;
            border-top: 1px solid #eee;
        }
        .footer b {
            color: #888;
        }
        
        /* Sidebar improvements */
        section[data-testid="stSidebar"] > div {
            padding-bottom: 20px;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            border-radius: 8px;
        }
        
        /* Better spacing for main content */
        .block-container {
            padding-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ============================================================
    # HEADER - Tampilan Judul yang Menarik
    # ============================================================
    st.markdown("""
    <div class="main-header">
        <h1>🚚 Sistem Optimasi Rute Pengiriman</h1>
        <p>Tugas Akhir: Multi-Fleet Vehicle Routing Problem with Time Windows (MFVRPTW)</p>
        <p>Metode: <b>Sweep → Nearest Neighbor → Ant Colony System → RVND</b></p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # TABS UTAMA
    # ============================================================
    # ============================================================
    # ============================================================
    # SIDEBAR GLOBAL (Kontrol Peta)
    # ============================================================
    with st.sidebar:
        # Pindahkan Statistik Data Input Titik ke sini? Tidak, biarkan di modulnya.
        # Tapi Kontrol Peta kita buat global agar tidak duplikat.
        st.header("⚙️ Kontrol Peta")
        show_labels = st.toggle("Tampilkan Label", value=True, key="global_show_labels")
        show_grid = st.toggle("Tampilkan Grid", value=True, key="global_show_grid")
        st.divider()

    # ============================================================
    # TABS UTAMA
    # ============================================================
    if render_academic_replay is not None:
        tab1, tab2, tab_opt, tab_viz, tab_res = st.tabs([
            "📍 Input Titik",
            "📋 Input Data",
            "🔬 Proses Optimasi",
            "🗺️ Hasil Visualisasi",
            "📊 Detail Penjadwalan"
        ])
    else:
        # Fallback setup if academic module fails, but effectively we strive for the above
        tab1, tab2, tab_opt, tab_viz, tab_res = st.tabs([
            "📍 Input Titik", 
            "📋 Input Data",
            "🔬 Proses Optimasi",
            "🗺️ Hasil Visualisasi", 
            "📊 Detail Penjadwalan" 
        ])

    with tab1:
        render_input_titik(show_labels=show_labels, show_grid=show_grid)
    with tab2:
        render_input_data()
    with tab_opt:
        if render_academic_replay:
            render_academic_replay()
        else:
            st.error("Modul Optimasi tidak tersedia.")
    with tab_viz:
        render_graph_hasil(show_labels=show_labels, show_grid=show_grid)
    with tab_res:
        render_hasil()

    # ============================================================
  # About section
    st.sidebar.markdown("### ℹ️ Tentang Aplikasi")
    st.sidebar.info(
        "Aplikasi ini dirancang untuk melakukan input data koordinat dan menghasilkan visualisasi rute pengiriman yang optimal. "
        "Proses optimasi akan dijalankan setelah konfigurasi armada dan data pelanggan berhasil divalidasi pada tab 'Input Data'."
    )
    # ============================================================
    # FOOTER
    # ============================================================
    st.markdown("""
    <div class="footer">
        © 2026 <b>Nabilah Eva Nurhayati</b> | 
        Mahasiswi Program Studi Matematika, Universitas Negeri Malang |
        Tugas Akhir Optimasi Rute Pengiriman
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
