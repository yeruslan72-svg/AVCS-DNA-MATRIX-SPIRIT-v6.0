"""
ui/dashboard.py — Dashboard UI for AVCS DNA-MATRIX SPIRIT

Provides:
 - Twin controls + step simulation
 - Analytics visualization and export
 - SPIRIT console (learning metrics, retrain)
Designed to be invoked by orchestrator with `render_dashboard(session)`
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
ANALYTICS_CSV = DATA_DIR / "analytics_log.csv"

def _ensure_analytics():
    if not ANALYTICS_CSV.exists() or ANALYTICS_CSV.stat().st_size == 0:
        times = pd.date_range(end=datetime.now(), periods=240, freq="T")
        df = pd.DataFrame({
            "timestamp": times,
            "temp": np.random.uniform(48, 78, len(times)),
            "pressure": np.random.uniform(14, 32, len(times)),
            "flow": np.random.uniform(110, 240, len(times)),
            "risk_est": (np.random.normal(30,15,len(times))).clip(0,100)
        })
        df.to_csv(ANALYTICS_CSV, index=False)
        return df
    else:
        df = pd.read_csv(ANALYTICS_CSV, parse_dates=['timestamp'])
        return df

def append_row(row: dict):
    df_row = pd.DataFrame([row])
    if not ANALYTICS_CSV.exists() or ANALYTICS_CSV.stat().st_size == 0:
        df_row.to_csv(ANALYTICS_CSV, index=False)
    else:
        df_row.to_csv(ANALYTICS_CSV, mode='a', header=False, index=False)

def render_dashboard(session):
    st.header("🎛 AVCS DNA-MATRIX SPIRIT — Dashboard")
    st.markdown("Use the left sidebar to switch modes and tweak demonstration settings.")

    # Mode selector inside dashboard (local)
    tab = st.tabs(["Twin Control", "Analytics", "SPIRIT Console", "Logs & Tools"])
    # ------- Twin Control -------
    with tab[0]:
        st.subheader("🏭 Twin Control & Quick Simulation")
        df = _ensure_analytics()

        col1, col2, col3 = st.columns(3)
        with col1:
            rpm = st.slider("RPM", 500, 4000, 2950, step=50, key="ui_rpm")
            load = st.selectbox("Load", ["nominal","overload","underload"], key="ui_load")
        with col2:
            ambient = st.slider("Ambient (°C)", -20, 60, 25, key="ui_ambient")
            duty = st.slider("Duty (%)", 0, 200, 100, key="ui_duty")
        with col3:
            fault = st.selectbox("Inject Fault", ["none","bearing_wear","imbalance","misalignment","cavitation"], key="ui_fault")
            if st.button("Simulate Step", key="ui_sim_step"):
                # simple synthetic generation
                rms = float(np.clip(0.5 + (rpm/4000)*6 + (1 if fault != "none" else 0)*2 + np.random.normal(0,0.5), 0, 12))
                temp_v = ambient + np.random.normal(0,2)
                noise = float(np.random.uniform(50,95))
                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "temp": float(temp_v),
                    "pressure": float(max(0.1, 15 + np.random.normal(0,3))),
                    "flow": float(max(0.0, 150 + np.random.normal(0,20))),
                    "risk_est": float(min(100, (rms/12)*100 + np.random.normal(0,5)))
                }
                append_row(row)
                st.success("Simulated and appended to analytics log.")
                session.get('system_logs') and session.system_logs  # no-op to ensure session visible

        st.markdown("Live preview (last 120 rows)")
        df = _ensure_analytics()
        st.line_chart(df.tail(120).set_index('timestamp')[['temp','pressure','flow']])
        st.area_chart(df.tail(120).set_index('timestamp')[['risk_est']])

    # ------- Analytics -------
    with tab[1]:
        st.subheader("📊 Analytics & Monitoring")
        df = _ensure_analytics()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg Temp (°C)", f"{df['temp'].mean():.2f}")
        c2.metric("Avg Pressure", f"{df['pressure'].mean():.2f}")
        c3.metric("Avg Flow", f"{df['flow'].mean():.2f}")
        c4.metric("Latest Risk", f"{df['risk_est'].iloc[-1]:.1f}")
        st.plotly_chart(df.set_index('timestamp')[['temp','pressure','flow']].tail(360).interpolate(), use_container_width=True)
        st.line_chart(df.set_index('timestamp')['risk_est'].rolling(5,min_periods=1).mean())
        if st.button("Download last 200 rows"):
            st.download_button("Analytics CSV", data=df.tail(200).to_csv(index=False), file_name="analytics_last200.csv")

    # ------- SPIRIT Console -------
    with tab[2]:
        st.subheader("🧬 SPIRIT Console — Adaptive Learning")
        st.markdown("Shows learning progress, feedback score, and allows retrain.")
        # attempt to use session.adaptive_engine if present
        adaptive = session.get('adaptive_engine')
        if adaptive:
            try:
                status = adaptive.status()
                st.json(status)
                st.metric("Progress", f"{status.get('progress',0)}%")
                if st.button("Retrain SPIRIT Model"):
                    with st.spinner("Retraining..."):
                        res = adaptive.retrain()
                        st.success("Retrain triggered.")
                        st.write(res)
            except Exception as e:
                st.error(f"Adaptive engine error: {e}")
        else:
            st.info("Adaptive engine not available — showing fallback metrics")
            df = _ensure_analytics()
            fallback_score = float(max(0, 100 - df['risk_est'].tail(120).mean()))
            st.metric("Feedback Score (fallback)", f"{fallback_score:.2f}")
            if st.button("Simulate retrain (fallback)"):
                st.success("Simulated retrain completed.")

    # ------- Logs & Tools -------
    with tab[3]:
        st.subheader("📝 Logs & Utilities")
        log_path = DATA_DIR / "system_orchestrator_log.json"
        if log_path.exists():
            try:
                import json
                with open(log_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
                st.write(f"{len(logs)} log entries")
                st.json(logs[-20:])
            except Exception:
                st.warning("Unable to read logs (invalid JSON).")
        else:
            st.info("No orchestrator logs yet.")
        if st.button("Open data folder (path)"):
            st.write(str(DATA_DIR.resolve()))
