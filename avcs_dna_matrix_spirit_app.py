"""
avcs_dna_matrix_spirit_app.py — AVCS DNA-MATRIX SPIRIT v7.1 (Main Launcher)

Purpose:
 - Unified Streamlit entrypoint that integrates:
     - Digital Twin (digital_twin/)
     - Industrial Core / Data Manager (industrial_core/)
     - Adaptive Learning (adaptive_learning/)
     - PLC / Integration points (plc_integration/)
 - Provides 3 main modes: Operational (Twin), Analytics, Adaptive Learning (SPIRIT)
 - Robust to missing optional modules (uses polite fallbacks)
 - Auto-creates demo analytics data if missing

Author: Generated for Yeruslan / AVCS
Year: 2025
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# ----------- Optional components (fail gracefully) -----------
# Attempt imports; fall back to conservative stubs if not available.

# Digital Twin
try:
    from digital_twin.industrial_digital_twin import IndustrialDigitalTwin
except Exception:
    IndustrialDigitalTwin = None

# Data Manager
try:
    from industrial_core.data_manager import DataManager
except Exception:
    DataManager = None

# Adaptive learning modules
try:
    from adaptive_learning.adaptive_engine import analyze_learning_progress, retrain_model
    from adaptive_learning.feedback_controller import update_feedback_score
    from adaptive_learning.sample_data import get_recent_activity
except Exception:
    analyze_learning_progress = None
    retrain_model = None
    update_feedback_score = None
    get_recent_activity = None

# PLC / integrator (optional)
try:
    from plc_integration.system_integrator import SoulPossessionIntegrator
except Exception:
    SoulPossessionIntegrator = None

# ----------- Constants & Paths -----------
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
ANALYTICS_CSV = DATA_DIR / "analytics_log.csv"
APP_TITLE = "AVCS DNA-MATRIX SPIRIT v7.1"
APP_SUBTITLE = "Operational Excellence Delivered — Twin · Matrix · Spirit"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


# ----------- Utilities -----------
def safe_import_name(obj):
    return obj.__name__ if obj is not None else "not installed"


def generate_demo_analytics(path: Path, n: int = 120):
    """Generate a demo analytics CSV with realistic-ish data (minute resolution)."""
    end = datetime.now()
    times = pd.date_range(end=end, periods=n, freq="T")
    df = pd.DataFrame({
        "timestamp": times,
        "temp": np.random.uniform(48, 78, size=n) + np.linspace(0, 2, n),
        "pressure": np.random.uniform(14, 32, size=n) + np.sin(np.linspace(0, 4 * np.pi, n)),
        "flow": np.random.uniform(110, 275, size=n) + np.cos(np.linspace(0, 2 * np.pi, n)) * 8,
        "risk_est": np.clip(np.random.normal(30, 15, size=n), 0, 100),
    })
    df.to_csv(path, index=False)
    return df


def load_analytics(path: Path):
    """Load analytics CSV or create demo if missing/empty."""
    if not path.exists() or path.stat().st_size == 0:
        return generate_demo_analytics(path)
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        # In case timestamp read as string:
        if df['timestamp'].dtype == object:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except Exception:
        # fallback overwrite
        return generate_demo_analytics(path)


def append_analytics_row(path: Path, row: dict):
    """Append a single analytics row to CSV (create file if missing)."""
    df_row = pd.DataFrame([row])
    if not path.exists() or path.stat().st_size == 0:
        df_row.to_csv(path, index=False)
    else:
        df_row.to_csv(path, index=False, mode='a', header=False)


# ----------- Environment checks -----------
def environment_report():
    report = {
        "DigitalTwin": safe_import_name(IndustrialDigitalTwin),
        "DataManager": safe_import_name(DataManager),
        "AdaptiveEngine": "available" if analyze_learning_progress else "not available",
        "PLCIntegrator": safe_import_name(SoulPossessionIntegrator),
        "Streamlit": st.__version__
    }
    return report


# ----------- Main UI / App logic -----------
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🧭")

# Sidebar: brand + navigation
with st.sidebar:
    if (ASSETS_DIR / "logo.png").exists():
        st.image(str((ASSETS_DIR / "logo.png")), use_column_width=True)
    else:
        st.markdown("<h2>AVCS DNA-MATRIX SPIRIT</h2>", unsafe_allow_html=True)
    st.markdown(f"**{APP_SUBTITLE}**")
    mode = st.radio("Mode", ["Twin — Operational", "Analytics — Monitoring", "SPIRIT — Adaptive Learning", "System — Diagnostics"])

# Top header
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)
st.markdown("---")

# Diagnostics quick card
if mode == "System — Diagnostics":
    st.header("🔧 System Diagnostics & Environment")
    st.write("This page summarizes runtime availability of optional components.")
    env = environment_report()
    st.json(env)
    st.markdown("**Analytics data file:**")
    st.write(str(ANALYTICS_CSV))
    if st.button("Initialize / Recreate Demo Analytics"):
        df = generate_demo_analytics(ANALYTICS_CSV, n=240)
        st.success(f"Demo analytics generated: {len(df)} rows.")
    st.markdown("---")
    st.write("Tips:")
    st.write("- Place a `logo.png` in `/assets` to show brand.")
    st.write("- Adaptive learning modules must be in `/adaptive_learning` to enable SPIRIT features.")
    st.stop()


# ---------------- Twin — Operational ----------------
if mode == "Twin — Operational":
    st.header("🏭 Digital Twin — Operational Control")
    # Instantiate twin safely
    if IndustrialDigitalTwin:
        try:
            twin = IndustrialDigitalTwin("centrifugal_pump")
            st.success("Digital Twin module loaded.")
        except Exception as e:
            twin = None
            st.error(f"Digital Twin instantiation error: {e}")
    else:
        twin = None
        st.warning("Digital Twin module not installed. Running simulation-only mode.")

    # Twin control controls
    st.subheader("Twin Controls")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        rpm = st.slider("RPM", 500, 4000, 2950, step=50)
        load = st.selectbox("Load", ["nominal", "overload", "underload"], index=0)
    with col2:
        ambient = st.slider("Ambient Temp (°C)", -20, 60, 25)
        duty = st.slider("Duty (%)", 0, 200, 100)
    with col3:
        inject_fault = st.selectbox("Inject Failure Mode", ["none", "bearing_wear", "imbalance", "misalignment", "cavitation"], index=0)
        step_run = st.button("▶️ Apply & Simulate Step")

    # Run a step
    if step_run:
        source = "Digital Twin" if twin else "Simulation Engine"
        # If twin provides simulate_equipment_behavior, use it; else synthetic values
        try:
            if twin and hasattr(twin, "simulate_equipment_behavior"):
                twin_data = twin.simulate_equipment_behavior({
                    "rpm": rpm,
                    "load": load,
                    "ambient_temperature": ambient,
                    "failure_mode": inject_fault
                })
                # Attempt to derive key metrics gracefully
                vibration = twin_data.get("vibration_data")
                if isinstance(vibration, (list, tuple, np.ndarray)):
                    rms = float(np.sqrt(np.mean(np.asarray(vibration) ** 2))) if len(vibration) > 0 else 0.0
                elif isinstance(vibration, dict) and 'rms' in vibration:
                    rms = float(vibration['rms'])
                else:
                    rms = float(np.random.uniform(0.5, 6.0))
                temp_val = twin_data.get("thermal_data", ambient + np.random.normal(0, 2))
                noise = twin_data.get("acoustic_data", np.random.uniform(50, 95))
            else:
                # synthetic
                rms = float(np.clip(0.5 + (rpm / 4000) * 6 + (1 if inject_fault != "none" else 0) * 2 + np.random.normal(0, 0.5), 0, 12))
                temp_val = ambient + np.random.normal(0, 3)
                noise = np.random.uniform(50, 95)
            st.metric("Data source", source)
            st.metric("Estimated Vibration RMS (mm/s)", f"{rms:.2f}")
            st.metric("Estimated Surface Temp (°C)", f"{temp_val:.1f}")
            st.metric("Acoustic Level (dB)", f"{noise:.1f}")

            # Append to analytics log for continuity
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "temp": float(temp_val),
                "pressure": float(max(0.1, 15 + np.random.normal(0, 3))),
                "flow": float(max(0.0, 150 + np.random.normal(0, 20))),
                "risk_est": float(min(100, (rms / 12) * 100 + np.random.normal(0, 6)))
            }
            append_analytics_row(ANALYTICS_CSV, row)
            st.success("Step simulated and appended to analytics log.")
        except Exception as e:
            st.error(f"Simulation failed: {e}")

    st.markdown("---")
    st.subheader("Twin Visualization")
    df = load_analytics(ANALYTICS_CSV)
    if not df.empty:
        # show last 120 entries
        df_viz = df.tail(120).copy()
        df_viz = df_viz.set_index("timestamp")
        st.line_chart(df_viz[["temp", "pressure", "flow"]])
        st.area_chart(df_viz[["risk_est"]])
    else:
        st.info("No analytics data available yet. Use 'Apply & Simulate Step' to generate demo rows.")


# ---------------- Analytics — Monitoring ----------------
elif mode == "Analytics — Monitoring":
    st.header("📊 Operational Analytics & Monitoring")
    st.markdown("Live analytics derived from `data/analytics_log.csv`. Auto-generates demo dataset if missing.")

    df = load_analytics(ANALYTICS_CSV)
    if df is None or df.empty:
        st.warning("Analytics dataset empty. Generating demo data...")
        df = generate_demo_analytics(ANALYTICS_CSV, n=180)

    # KPI strip
    st.subheader("Key Performance Indicators (KPI)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Temp (°C)", f"{df['temp'].mean():.2f}")
    col2.metric("Avg Pressure (bar)", f"{df['pressure'].mean():.2f}")
    col3.metric("Avg Flow (m³/h)", f"{df['flow'].mean():.2f}")
    col4.metric("Latest Risk (%)", f"{float(df['risk_est'].iloc[-1]):.1f}")

    # Time series
    st.subheader("Time Series")
    st.plotly_chart(
        pd.concat([
            df.set_index("timestamp")[["temp", "pressure", "flow"]].tail(240)
        ], axis=1).interpolate(),
        use_container_width=True
    )

    # Risk band
    st.subheader("Risk Evolution")
    st.line_chart(df.set_index("timestamp")["risk_est"].rolling(5, min_periods=1).mean())

    # Controls
    st.markdown("---")
    st.subheader("Analytics Utilities")
    cola, colb = st.columns([2, 1])
    with cola:
        if st.button("Generate fresh demo dataset"):
            df = generate_demo_analytics(ANALYTICS_CSV, n=360)
            st.success("Demo analytics dataset reset.")
            st.experimental_rerun()
    with colb:
        if st.button("Export visible analytics (CSV)"):
            tmp = df.tail(100).to_csv(index=False)
            st.download_button("Download CSV", data=tmp, file_name="analytics_export.csv", mime="text/csv")

    st.markdown("Data preview:")
    st.dataframe(df.tail(25))


# ---------------- SPIRIT — Adaptive Learning ----------------
elif mode == "SPIRIT — Adaptive Learning":
    st.header("🧬 SPIRIT — Adaptive Learning Layer (v7.x)")
    st.markdown("Adaptive learning observes system behavior and suggests optimizations. This panel shows learning metrics and allows retraining.")

    # Check availability
    if analyze_learning_progress and update_feedback_score and get_recent_activity:
        try:
            activity = get_recent_activity()  # should return structured activity rows (list/dict/pandas)
        except Exception:
            activity = None
    else:
        activity = None

    # Feedback score & progress
    if activity is None or (isinstance(activity, (list, dict)) and len(activity) == 0):
        st.info("No activity data from adaptive_learning.sample_data found — using analytics log as fallback.")
        df_analytics = load_analytics(ANALYTICS_CSV)
        # synthesize activity from analytics
        activity = []
        for _, r in df_analytics.tail(120).iterrows():
            activity.append({
                "ts": pd.to_datetime(r["timestamp"]),
                "temp": float(r["temp"]),
                "pressure": float(r["pressure"]),
                "flow": float(r["flow"]),
                "risk": float(r.get("risk_est", 0))
            })

    # Compute feedback score
    try:
        if update_feedback_score:
            feedback_score = update_feedback_score(activity)
        else:
            # fallback heuristic
            risks = np.array([a.get("risk", 0) for a in activity[-60:]]) if activity else np.array([0])
            feedback_score = float(max(0.0, 100 - np.nanmean(risks)))
    except Exception:
        feedback_score = 0.0

    # Compute learning progress & patterns
    try:
        if analyze_learning_progress:
            learning_status = analyze_learning_progress(activity)
            # Expect learning_status to be dict with keys: progress (0-100), pattern (str), trend (list/series)
            progress_val = learning_status.get("progress", 0)
            pattern = learning_status.get("pattern", "—")
            trend = learning_status.get("trend", [0])
        else:
            # fallback: simple trend on risk
            risks = [a.get("risk", 0) for a in activity[-120:]] if activity else [0]
            progress_val = int(max(0, min(100, 100 - np.mean(risks))))
            pattern = "fallback-pattern"
            trend = risks[-120:] if len(risks) >= 1 else [0]
    except Exception:
        progress_val = 0
        pattern = "error"
        trend = [0]

    # Show metrics
    col1, col2, col3 = st.columns([2, 2, 2])
    col1.metric("Feedback Score", f"{feedback_score:.2f}")
    col2.metric("Learning Progress", f"{progress_val}%")
    col3.metric("Detected Pattern", pattern)

    # Trend chart
    st.subheader("Learning Trend")
    try:
        trend_series = pd.Series(trend[-240:]).fillna(method='ffill').fillna(0)
        st.line_chart(trend_series)
    except Exception:
        st.text("Trend visualization not available.")

    st.markdown("---")
    # Retrain button
    if st.button("🔁 Retrain SPIRIT Adaptive Model"):
        with st.spinner("Retraining adaptive model (this may take a few seconds in demo)..."):
            try:
                if retrain_model:
                    retrain_model(activity)
                    st.success("Adaptive model retrained (adaptive_learning.retrain_model executed).")
                else:
                    # fallback simulated retrain
                    time.sleep(1.2)
                    st.success("Adaptive model retrained (simulated).")
            except Exception as e:
                st.error(f"Retrain failed: {e}")

    # Export activity
    if st.button("📥 Export recent activity (JSON)"):
        try:
            payload = json.dumps(activity, default=str, indent=2)
            st.download_button("Download activity.json", data=payload, file_name="spirit_activity.json", mime="application/json")
        except Exception as e:
            st.error(f"Export failed: {e}")

    st.markdown("### Recent activity sample")
    try:
        import itertools
        sample_preview = list(itertools.islice(activity[::-1], 20))
        st.json(sample_preview)
    except Exception:
        st.text("No activity preview available.")


# Footer
st.markdown("---")
st.caption("© 2025 AVCS Systems · AVCS DNA-MATRIX SPIRIT v7.1 — Built for Industrial Adaptive Intelligence")
