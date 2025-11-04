"""
avcs_dna_matrix_spirit_app.py — AVCS DNA-MATRIX SPIRIT v7.2 (Main Launcher)

- Unified Streamlit entrypoint integrating:
    - Digital Twin (digital_twin/)
    - Industrial Core / Data Manager (industrial_core/)
    - Adaptive Learning (adaptive_learning/)
    - PLC / integrator (plc_integration/)
- Modes: Twin (Operational), Analytics (Monitoring), SPIRIT (Adaptive Learning),
  System (Diagnostics & Tools)
- Graceful fallbacks for missing optional modules
- Demo-data generation, logging, export utilities, and Dockerfile generator
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import textwrap

# -------- Project paths & constants ----------
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
ANALYTICS_CSV = DATA_DIR / "analytics_log.csv"
SYSTEM_LOG_JSON = DATA_DIR / "system_logs.json"
DOCKERFILE_PATH = ROOT / "Dockerfile"

APP_TITLE = "AVCS DNA-MATRIX SPIRIT"
APP_VERSION = "v7.2"
APP_SUBTITLE = "Operational Excellence Delivered — Twin · Matrix · Spirit"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# -------- Optional module imports (fail gracefully) ----------
try:
    from digital_twin.industrial_digital_twin import IndustrialDigitalTwin
except Exception:
    IndustrialDigitalTwin = None

try:
    from industrial_core.data_manager import DataManager
except Exception:
    DataManager = None

try:
    from adaptive_learning.adaptive_engine import analyze_learning_progress, retrain_model
    from adaptive_learning.feedback_controller import update_feedback_score
    from adaptive_learning.sample_data import get_recent_activity
except Exception:
    analyze_learning_progress = None
    retrain_model = None
    update_feedback_score = None
    get_recent_activity = None

try:
    from plc_integration.system_integrator import SoulPossessionIntegrator
except Exception:
    SoulPossessionIntegrator = None

# -------- Utilities ----------
def safe_name(obj):
    return obj.__name__ if obj is not None else "not installed"

def generate_demo_analytics(path: Path, n: int = 240):
    """Create a demo analytics CSV (minute resolution by default)."""
    now = datetime.now()
    times = pd.date_range(end=now, periods=n, freq="T")
    df = pd.DataFrame({
        "timestamp": times,
        "temp": np.random.uniform(48, 78, size=n) + np.linspace(0, 2, n),
        "pressure": np.random.uniform(14, 32, size=n) + np.sin(np.linspace(0, 4*np.pi, n)),
        "flow": np.random.uniform(110, 275, size=n) + np.cos(np.linspace(0, 2*np.pi, n))*8,
        "risk_est": np.clip(np.random.normal(30, 15, size=n), 0, 100)
    })
    df.to_csv(path, index=False)
    return df

def load_analytics(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return generate_demo_analytics(path)
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if "timestamp" in df.columns and df["timestamp"].dtype == object:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return generate_demo_analytics(path)

def append_analytics_row(path: Path, row: dict):
    df = pd.DataFrame([row])
    if not path.exists() or path.stat().st_size == 0:
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)

def append_system_log(path: Path, entry: dict):
    logs = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, default=str, indent=2)

def environment_summary():
    return {
        "DigitalTwin": safe_name(IndustrialDigitalTwin),
        "DataManager": safe_name(DataManager),
        "AdaptiveEngine": "available" if analyze_learning_progress else "not available",
        "PLCIntegrator": safe_name(SoulPossessionIntegrator),
        "Streamlit": st.__version__
    }

def write_dockerfile(path: Path):
    content = textwrap.dedent(f"""
    # Simple Dockerfile for AVCS DNA-MATRIX SPIRIT
    FROM python:3.11-slim
    WORKDIR /app
    COPY . /app
    RUN pip install --upgrade pip
    RUN pip install -r requirements.txt
    EXPOSE 8501
    CMD ["streamlit", "run", "ui/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
    """).strip()
    path.write_text(content, encoding="utf-8")
    return path

# --------- STREAMLIT UI ----------
st.set_page_config(page_title=f"{APP_TITLE} {APP_VERSION}", layout="wide", page_icon="🧭")
with st.sidebar:
    if (ASSETS_DIR / "logo.png").exists():
        st.image(str(ASSETS_DIR / "logo.png"), use_column_width=True)
    else:
        st.markdown("## AVCS DNA-MATRIX SPIRIT")
    st.markdown(f"**{APP_SUBTITLE}**")
    st.caption(f"Version: {APP_VERSION}")
    mode = st.radio("Mode", ["Twin — Operational", "Analytics — Monitoring", "SPIRIT — Adaptive Learning", "System — Diagnostics & Tools"])

st.title(f"{APP_TITLE} — {APP_VERSION}")
st.markdown(f"**{APP_SUBTITLE}**")
st.markdown("---")

# ---------------- Mode: Diagnostics & Tools ----------------
if mode == "System — Diagnostics & Tools":
    st.header("🔧 System Diagnostics & Tools")
    st.subheader("Environment Summary")
    st.json(environment_summary())

    st.markdown("### Analytics / Logs")
    st.write("Analytics file:", str(ANALYTICS_CSV))
    st.write("System log file:", str(SYSTEM_LOG_JSON))

    if st.button("Initialize / Reset Demo Analytics"):
        df = generate_demo_analytics(ANALYTICS_CSV, n=360)
        st.success(f"Demo analytics generated ({len(df)} rows).")
        append_system_log(SYSTEM_LOG_JSON, {"ts": datetime.utcnow().isoformat(), "event": "demo_analytics_reset", "rows": len(df)})

    if st.button("Clear Analytics File"):
        open(ANALYTICS_CSV, "w").close()
        st.warning("Analytics file cleared (empty).")
        append_system_log(SYSTEM_LOG_JSON, {"ts": datetime.utcnow().isoformat(), "event": "analytics_cleared"})

    if st.button("Create Dockerfile"):
        p = write_dockerfile(DOCKERFILE_PATH)
        st.success(f"Dockerfile created at {p}")
        append_system_log(SYSTEM_LOG_JSON, {"ts": datetime.utcnow().isoformat(), "event": "dockerfile_created", "path": str(p)})

    st.markdown("---")
    st.subheader("Export / Import Utilities")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export analytics (last 200 rows)"):
            df = load_analytics(ANALYTICS_CSV)
            csv = df.tail(200).to_csv(index=False)
            st.download_button("Download analytics CSV", data=csv, file_name="analytics_export.csv", mime="text/csv")
    with col2:
        uploaded = st.file_uploader("Import analytics CSV (append)", type=["csv"])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                df_up.to_csv(ANALYTICS_CSV, mode="a", header=not ANALYTICS_CSV.exists(), index=False)
                st.success("Uploaded analytics appended.")
                append_system_log(SYSTEM_LOG_JSON, {"ts": datetime.utcnow().isoformat(), "event": "analytics_imported", "rows": len(df_up)})
            except Exception as e:
                st.error(f"Import failed: {e}")

    st.markdown("---")
    st.subheader("Recent System Logs")
    if SYSTEM_LOG_JSON.exists():
        try:
            with open(SYSTEM_LOG_JSON, "r", encoding="utf-8") as f:
                logs = json.load(f)
            st.write(f"{len(logs)} log entries")
            st.json(logs[-20:])
        except Exception:
            st.warning("System log file invalid or unreadable.")
    else:
        st.info("No system logs yet. Actions will populate logs.")

    st.stop()

# ---------------- Mode: Twin — Operational ----------------
if mode == "Twin — Operational":
    st.header("🏭 Digital Twin — Operational Control")

    # instantiate twin (safe)
    if IndustrialDigitalTwin:
        try:
            twin = IndustrialDigitalTwin("centrifugal_pump")
            st.success("Digital Twin module loaded.")
        except Exception as e:
            twin = None
            st.error(f"Digital Twin init failed: {e}")
    else:
        twin = None
        st.warning("Digital Twin module not installed — running simulation fallback.")

    st.subheader("Twin Controls")
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        rpm = st.slider("RPM", 500, 4000, 2950, step=50)
        load = st.selectbox("Load", ["nominal", "overload", "underload"])
    with col_b:
        ambient = st.slider("Ambient Temp (°C)", -20, 60, 25)
        duty = st.slider("Duty (%)", 0, 200, 100)
    with col_c:
        failure = st.selectbox("Inject Failure Mode", ["none", "bearing_wear", "imbalance", "misalignment", "cavitation"])
        do_step = st.button("▶️ Apply & Simulate Step")

    if do_step:
        try:
            if twin and hasattr(twin, "simulate_equipment_behavior"):
                twin_data = twin.simulate_equipment_behavior({
                    "rpm": rpm, "load": load, "ambient_temperature": ambient, "failure_mode": failure
                })
                vib = twin_data.get("vibration_data")
                if isinstance(vib, (list, tuple, np.ndarray)):
                    rms = float(np.sqrt(np.mean(np.asarray(vib) ** 2))) if len(vib) > 0 else 0.0
                elif isinstance(vib, dict) and "rms" in vib:
                    rms = float(vib["rms"])
                else:
                    rms = float(np.random.uniform(0.5, 6.0))
                temp_val = twin_data.get("thermal_data", ambient + np.random.normal(0,2))
                noise = twin_data.get("acoustic_data", np.random.uniform(50,95))
                src = "Digital Twin"
            else:
                # synthetic fallback
                rms = float(np.clip(0.5 + (rpm/4000)*6 + (1 if failure != "none" else 0)*2 + np.random.normal(0,0.5), 0, 12))
                temp_val = ambient + np.random.normal(0,2)
                noise = float(np.random.uniform(50,95))
                src = "Simulated"

            st.metric("Data Source", src)
            st.metric("Vibration RMS (mm/s)", f"{rms:.2f}")
            st.metric("Surface Temp (°C)", f"{temp_val:.1f}")
            st.metric("Acoustic Level (dB)", f"{noise:.1f}")

            # Append to analytics log
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "temp": float(temp_val),
                "pressure": float(max(0.1, 15 + np.random.normal(0,3))),
                "flow": float(max(0.0, 150 + np.random.normal(0,20))),
                "risk_est": float(min(100, (rms/12)*100 + np.random.normal(0,5)))
            }
            append_analytics_row(ANALYTICS_CSV, row)
            append_system_log(SYSTEM_LOG_JSON, {"ts": datetime.utcnow().isoformat(), "event": "twin_step", "source": src, "failure": failure, "rms": rms})
            st.success("Step simulated and saved to analytics log.")
        except Exception as e:
            st.error(f"Simulation error: {e}")

    st.markdown("---")
    st.subheader("Twin Live Visualization")
    df = load_analytics(ANALYTICS_CSV)
    if not df.empty:
        df_viz = df.tail(240).set_index("timestamp")
        st.line_chart(df_viz[["temp","pressure","flow"]])
        st.area_chart(df_viz[["risk_est"]])
    else:
        st.info("No data yet. Press 'Apply & Simulate Step' to create demo entries.")

# ---------------- Mode: Analytics — Monitoring ----------------
elif mode == "Analytics — Monitoring":
    st.header("📊 Operational Analytics & Monitoring")
    st.markdown("Data source: `data/analytics_log.csv` (auto-generated if missing).")
    df = load_analytics(ANALYTICS_CSV)

    st.subheader("KPI")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg Temp (°C)", f"{df['temp'].mean():.2f}")
    c2.metric("Avg Pressure (bar)", f"{df['pressure'].mean():.2f}")
    c3.metric("Avg Flow (m³/h)", f"{df['flow'].mean():.2f}")
    c4.metric("Latest Risk (%)", f"{float(df['risk_est'].iloc[-1]):.1f}")

    st.subheader("Trends")
    st.plotly_chart(df.set_index("timestamp")[["temp","pressure","flow"]].tail(360).interpolate(), use_container_width=True)
    st.subheader("Risk Evolution (rolling)")
    st.line_chart(df.set_index("timestamp")["risk_est"].rolling(5,min_periods=1).mean())

    st.markdown("---")
    st.subheader("Utilities")
    a1,a2 = st.columns([2,1])
    with a1:
        if st.button("Regenerate demo analytics (360 rows)"):
            generate_demo_analytics(ANALYTICS_CSV, n=360)
            st.success("Demo analytics regenerated.")
            append_system_log(SYSTEM_LOG_JSON, {"ts": datetime.utcnow().isoformat(), "event": "analytics_regenerated"})
            st.experimental_rerun()
    with a2:
        if st.button("Download last 200 rows"):
            payload = df.tail(200).to_csv(index=False)
            st.download_button("Download CSV", data=payload, file_name="analytics_last200.csv", mime="text/csv")

    st.markdown("Data preview")
    st.dataframe(df.tail(30))

# ---------------- Mode: SPIRIT — Adaptive Learning ----------------
elif mode == "SPIRIT — Adaptive Learning":
    st.header("🧬 SPIRIT — Adaptive Learning Layer")
    st.markdown("Adaptive layer ingests activity & feedback to evolve policy. This panel shows progress & offers retrain controls.")

    # Acquire activity
    activity = None
    try:
        if get_recent_activity:
            activity = get_recent_activity()
    except Exception:
        activity = None

    if not activity:
        st.info("No `adaptive_learning.sample_data` available — synthesizing activity from analytics log.")
        df = load_analytics(ANALYTICS_CSV)
        activity = []
        for _, r in df.tail(240).iterrows():
            activity.append({
                "ts": str(r["timestamp"]),
                "temp": float(r["temp"]),
                "pressure": float(r["pressure"]),
                "flow": float(r["flow"]),
                "risk": float(r.get("risk_est", 0))
            })

    # Feedback score
    try:
        if update_feedback_score:
            feed_score = update_feedback_score(activity)
        else:
            risks = np.array([a.get("risk",0) for a in activity[-120:]]) if activity else np.array([0])
            feed_score = float(max(0.0, 100 - np.nanmean(risks)))
    except Exception:
        feed_score = 0.0

    # Learning progress
    try:
        if analyze_learning_progress:
            learning_status = analyze_learning_progress(activity)
            progress_val = learning_status.get("progress", 0)
            pattern = learning_status.get("pattern", "—")
            trend = learning_status.get("trend", [])
        else:
            risks = [a.get("risk",0) for a in activity[-240:]] if activity else [0]
            progress_val = int(max(0, min(100, 100 - np.mean(risks))))
            pattern = "fallback-pattern"
            trend = risks
    except Exception:
        progress_val = 0
        pattern = "error"
        trend = []

    colx, coly, colz = st.columns(3)
    colx.metric("Feedback Score", f"{feed_score:.2f}")
    coly.metric("Learning Progress", f"{progress_val}%")
    colz.metric("Detected Pattern", pattern)

    st.subheader("Learning Trend")
    try:
        series = pd.Series(trend[-360:]).fillna(0)
        st.line_chart(series)
    except Exception:
        st.text("No trend available.")

    st.markdown("---")
    if st.button("🔁 Retrain SPIRIT Model"):
        with st.spinner("Retraining SPIRIT (demo-mode)..."):
            try:
                if retrain_model:
                    retrain_model(activity)
                    st.success("Adaptive model retrained (adaptive_learning.retrain_model executed).")
                else:
                    time.sleep(1.0)
                    st.success("Adaptive model retrained (simulated).")
                append_system_log(SYSTEM_LOG_JSON, {"ts": datetime.utcnow().isoformat(), "event": "spirit_retrained"})
            except Exception as e:
                st.error(f"Retrain failed: {e}")

    if st.button("📥 Export activity (JSON)"):
        payload = json.dumps(activity, indent=2, default=str)
        st.download_button("Download activity.json", data=payload, file_name="spirit_activity.json", mime="application/json")

    st.markdown("Recent activity (preview)")
    try:
        sample_preview = activity[-30:][::-1]
        st.json(sample_preview)
    except Exception:
        st.text("No preview available.")

# Footer
st.markdown("---")
st.caption(f"© 2025 AVCS Systems · {APP_TITLE} {APP_VERSION} — Built for Industrial Adaptive Intelligence")
