from __future__ import annotations
import json
import streamlit as st
from pathlib import Path
from typing import Dict, Any

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUTOSAVE_PATH = DATA_DIR / "autosave.json"


def save_to_autosave() -> None:
    """
    Saves relevant session state keys to a local JSON file.
    This acts as an 'Autosave' mechanism.
    """
    try:
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Define what keys to save
        data_to_save = {
            "points": st.session_state.get("points", {}),
            "inputData": st.session_state.get("inputData", {}),
            "user_vehicles": st.session_state.get("user_vehicles", []),
            "acs_params": st.session_state.get("acs_params", {}),
            "kapasitas_kendaraan": st.session_state.get("kapasitas_kendaraan", 100),
            "iterasi": st.session_state.get("iterasi", 2),
            "distanceMatrix_size": st.session_state.get("distanceMatrix_size", 0)
        }

        with open(AUTOSAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)

    except Exception as e:
        print(f"⚠️ Warning: Auto-save failed: {e}")


def load_from_autosave() -> bool:
    """
    Loads data from autosave.json into session_state.
    Returns True if data was loaded, False otherwise.
    """
    if not AUTOSAVE_PATH.exists():
        return False

    try:
        with open(AUTOSAVE_PATH, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        # Restore keys to session_state
        st.session_state["points"] = loaded_data.get(
            "points", {"depots": [], "customers": []})
        st.session_state["inputData"] = loaded_data.get("inputData", {})
        st.session_state["user_vehicles"] = loaded_data.get(
            "user_vehicles", [])
        st.session_state["acs_params"] = loaded_data.get("acs_params", {})
        st.session_state["kapasitas_kendaraan"] = loaded_data.get(
            "kapasitas_kendaraan", 100)
        st.session_state["iterasi"] = loaded_data.get("iterasi", 2)
        st.session_state["distanceMatrix_size"] = loaded_data.get(
            "distanceMatrix_size", 0)

        # Clean up any potential temporary keys to force re-render components if needed
        keys_to_clear = [k for k in st.session_state.keys()
                         if k.startswith("customer_tw_data_") or
                         k.startswith("distance_matrix_data_")]
        for k in keys_to_clear:
            del st.session_state[k]

        return True
    except Exception as e:
        print(f"⚠️ Warning: Auto-load failed: {e}")
        return False


def check_autosave_exists() -> bool:
    return AUTOSAVE_PATH.exists()
