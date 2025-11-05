"""
avcs_dna_matrix_spirit_app.py — AVCS DNA-MATRIX SPIRIT (Orchestrator)

Single entrypoint that:
 - initializes modules (digital_twin, data_manager, adaptive learning, plc integrator)
 - routes to ui/dashboard.py (embedded) and provides shared session-state
 - stores simple audit/log entries in data/
 - fails gracefully if optional pieces missing

Usage:
    streamlit run avcs_dna_matrix_spirit_app.py
"""
import os
from pathlib import Path
import json
from datetime import datetime

import streamlit as st

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
LOG_FILE = DATA_DIR / "system_orchestrator_log.json"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ---- Safe imports for optional modules ----
try:
    from ui.dashboard import render_dashboard
except Exception:
    render_dashboard = None

try:
    from industrial_core.data_manager import DataManager
except Exception:
    DataManager = None

try:
    from digital_twin.industrial_digital_twin import IndustrialDigitalTwin
except Exception:
    IndustrialDigitalTwin = None

try:
    from adaptive_learning.adaptive_engine import AdaptiveEngine
except Exception:
    AdaptiveEngine = None

try:
    from plc_integration.system_integrator import SoulPossessionIntegrator
except Exception:
    SoulPossessionIntegrator = None

# ---- Utilities ----
def append_log(entry: dict):
    logs = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, default=str, indent=2)

# ---- Initialize session singletons ----
def initialize_session():
    if "initialized" in st.session_state and st.session_state.initialized:
        return

    st.session_state.initialized = True
    st.session_state.start_time = datetime.utcnow().isoformat()
    # Data manager
    try:
        st.session_state.data_manager = DataManager() if DataManager else None
    except Exception as e:
        st.session_state.data_manager = None
        append_log({"ts": datetime.utcnow().isoformat(), "event": "data_manager_init_failed", "err": str(e)})

    # Digital twin
    try:
        st.session_state.digital_twin = IndustrialDigitalTwin("centrifugal_pump") if IndustrialDigitalTwin else None
    except Exception as e:
        st.session_state.digital_twin = None
        append_log({"ts": datetime.utcnow().isoformat(), "event": "digital_twin_init_failed", "err": str(e)})

    # Adaptive engine
    try:
        st.session_state.adaptive_engine = AdaptiveEngine() if AdaptiveEngine else None
    except Exception as e:
        st.session_state.adaptive_engine = None
        append_log({"ts": datetime.utcnow().isoformat(), "event": "adaptive_init_failed", "err": str(e)})

    # PLC integrator
    try:
        st.session_state.plc_integrator = SoulPossessionIntegrator() if SoulPossessionIntegrator else None
    except Exception as e:
        st.session_state.plc_integrator = None
        append_log({"ts": datetime.utcnow().isoformat(), "event": "plc_init_failed", "err": str(e)})

    # Shared runtime containers
    st.session_state.analytics_file = DATA_DIR / "analytics_log.csv"
    st.session_state.system_logs = LOG_FILE
    append_log({"ts": datetime.utcnow().isoformat(), "event": "session_initialized"})

# ---- Main ----
def main():
    st.set_page_config(page_title="AVCS DNA-MATRIX SPIRIT — Orchestrator", layout="wide")
    st.title("AVCS DNA-MATRIX SPIRIT — Orchestrator")
    st.caption("Orchestrator: bootstrapping modules and launching UI")

    initialize_session()

    # Diagnostics quick
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Digital Twin", "Loaded" if st.session_state.digital_twin else "Fallback")
    with col2:
        st.metric("Adaptive Engine", "Loaded" if st.session_state.adaptive_engine else "Fallback")
    with col3:
        st.metric("Data Manager", "Loaded" if st.session_state.data_manager else "Fallback")

    st.markdown("---")

    if render_dashboard:
        try:
            # Hand over control to UI module; pass session_state reference
            render_dashboard(session=st.session_state)
        except Exception as e:
            st.error(f"Dashboard render failed: {e}")
            append_log({"ts": datetime.utcnow().isoformat(), "event": "dashboard_failed", "err": str(e)})
    else:
        st.warning("UI module `ui.dashboard` not found. Please add `ui/dashboard.py`.")
        st.write("You can run lightweight tests here or navigate to repository files.")
        if st.button("Create placeholder analytics file"):
            # create demo analytics
            import pandas as pd, numpy as np
            times = pd.date_range(end=datetime.now(), periods=120, freq="T")
            df = pd.DataFrame({
                "timestamp": times,
                "temp": np.random.uniform(48, 78, len(times)),
                "pressure": np.random.uniform(14, 32, len(times)),
                "flow": np.random.uniform(110, 240, len(times)),
                "risk_est": (np.random.normal(30, 15, len(times))).clip(0, 100)
            })
            df.to_csv(st.session_state.analytics_file, index=False)
            st.success("Placeholder analytics generated.")
            append_log({"ts": datetime.utcnow().isoformat(), "event": "placeholder_analytics_created", "rows": len(df)})

if __name__ == "__main__":
    main()
