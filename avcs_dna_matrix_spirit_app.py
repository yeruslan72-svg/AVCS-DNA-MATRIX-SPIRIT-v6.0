# ================================================================
# AVCS DNA-MATRIX SPIRIT v7.0 — Adaptive Intelligence Application
# ================================================================
import streamlit as st
import time
import plotly.graph_objects as go
from adaptive_learning.adaptive_core import AdaptiveCore
from adaptive_learning.sample_data import generate_sample

# -----------------------------
# Initialize core intelligence
# -----------------------------
st.set_page_config(page_title="AVCS DNA-MATRIX SPIRIT v7.0", layout="wide")
st.title("🧬 AVCS DNA-MATRIX SPIRIT v7.0")
st.caption("Adaptive Vibration Control System — AI/ML Enhanced Module")

# Adaptive Intelligence Engine
adaptive_core = AdaptiveCore()

# State for loop timing
if 'run_loop' not in st.session_state:
    st.session_state.run_loop = False

# -----------------------------
# Layout
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    mode = st.selectbox("Simulation Mode", ["normal", "degraded", "high_load", "night"], index=0)
with col2:
    interval = st.slider("Update Interval (sec)", 0.2, 5.0, 1.5, 0.1)
with col3:
    st.session_state.run_loop = st.toggle("Start Simulation", value=False)

st.divider()

# -----------------------------
# Real-time Visualization Area
# -----------------------------
placeholder_chart = st.empty()
placeholder_status = st.empty()

rpm_hist, vib_hist = [], []

# -----------------------------
# Main Loop
# -----------------------------
while st.session_state.run_loop:
    # 1. Generate telemetry
    telemetry = generate_sample(mode=mode)

    # 2. Update adaptive intelligence
    adaptive_core.update(telemetry)
    state = adaptive_core.status()

    # 3. Record data
    rpm_hist.append(telemetry["rpm"])
    vib_hist.append(telemetry["vibration"])

    # 4. Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rpm_hist[-100:], mode="lines", name="RPM"))
    fig.add_trace(go.Scatter(y=vib_hist[-100:], mode="lines", name="Vibration"))
    fig.update_layout(height=350, title="Live Sensor Stream", template="plotly_dark")
    placeholder_chart.plotly_chart(fig, use_container_width=True)

    # 5. Show status
    with placeholder_status.container():
        st.subheader("🧠 Adaptive Intelligence Status")
        st.json(state)

    # 6. Wait for next cycle
    time.sleep(interval)

st.info("Simulation stopped. Toggle 'Start Simulation' to run again.")
