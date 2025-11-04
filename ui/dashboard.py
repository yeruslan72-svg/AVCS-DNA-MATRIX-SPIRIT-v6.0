# ui/dashboard.py
"""
AVCS DNA-MATRIX SPIRIT v6.0 — Dashboard (improved v2)
- Overview tab with logo + system status
- Sensors tab with live-updating charts (Temperature, Pressure, Vibration)
- Uses digital_twin or industrial_core.DataManager if available; otherwise simulates data
"""

import streamlit as st
import time
import os
import math
import random
from datetime import datetime
from collections import deque
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import optional plumbing (safe fallbacks)
try:
    from industrial_core.data_manager import DataManager
except Exception:
    DataManager = None

try:
    from digital_twin.industrial_digital_twin import IndustrialDigitalTwin
except Exception:
    IndustrialDigitalTwin = None


# ---------------------------
# Helper: load logo
# ---------------------------
def load_logo_image():
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        try:
            return Image.open(logo_path)
        except Exception:
            return None
    return None


# ---------------------------
# Sensor buffer container
# ---------------------------
class SensorsBuffer:
    def __init__(self, maxlen=300):
        self.timestamps = deque(maxlen=maxlen)
        self.temperature = deque(maxlen=maxlen)
        self.pressure = deque(maxlen=maxlen)
        self.vibration = deque(maxlen=maxlen)

    def append(self, ts, temp, pres, vib):
        self.timestamps.append(ts)
        self.temperature.append(temp)
        self.pressure.append(pres)
        self.vibration.append(vib)

    def as_plotly(self):
        return {
            "timestamps": list(self.timestamps),
            "temperature": list(self.temperature),
            "pressure": list(self.pressure),
            "vibration": list(self.vibration),
        }


# ---------------------------
# Data source abstraction
# ---------------------------
class DataSource:
    def __init__(self):
        # Prefer digital twin if available, otherwise data manager, otherwise simulated
        self.twin = IndustrialDigitalTwin("centrifugal_pump") if IndustrialDigitalTwin else None
        self.dm = DataManager() if DataManager else None
        # keep a simple internal phase for synthetic signals
        self._phase = 0.0

    def get_sample(self):
        now = datetime.utcnow()
        if self.twin:
            try:
                twin_out = self.twin.simulate_equipment_behavior({
                    "rpm": 2950,
                    "load": "normal",
                    "ambient_temperature": 25
                })
                # Expect twin_out to contain keys as per digital twin module ("temperature_data","vibration_data","health_metrics")
                temp = None
                pres = None
                vib = None
                # Temperature: scalar or dict
                tdata = twin_out.get("temperature_data", None)
                if isinstance(tdata, dict):
                    # pick first value
                    temp = float(next(iter(tdata.values())))
                elif isinstance(tdata, (int, float)):
                    temp = float(tdata)
                # Pressure: digital twin may not provide — simulate small value
                pres = float(twin_out.get("pressure", 1.0)) if twin_out.get("pressure", None) is not None else 1.0 + random.uniform(-0.1, 0.1)
                # Vibration: try to compute RMS from returned array if present
                vdata = twin_out.get("vibration_data", None)
                if vdata is None:
                    vib = max(0.01, 0.5 + random.random() * 0.5)
                else:
                    try:
                        import numpy as _np
                        arr = _np.asarray(vdata)
                        vib = float(_np.sqrt(_np.mean(arr ** 2)))
                    except Exception:
                        vib = float(abs(sum(vdata)) / max(1, len(vdata)))
                return now, round(temp, 2) if temp is not None else round(65 + random.uniform(-2, 2), 2), round(pres, 3), round(vib, 4)
            except Exception:
                # fallback to synthetic
                return self._synthetic(now)
        elif self.dm:
            try:
                # If DataManager has a method to pull latest sensor values, attempt it
                if hasattr(self.dm, "get_latest"):
                    latest = self.dm.get_latest()
                    # expected dict with keys
                    temp = float(latest.get("temperature", 65.0))
                    pres = float(latest.get("pressure", 1.0))
                    vib = float(latest.get("vibration", 0.5))
                    return now, temp, pres, vib
            except Exception:
                return self._synthetic(now)
        else:
            return self._synthetic(now)

    def _synthetic(self, now):
        # produce stable but slightly varying signals — suitable for plots
        self._phase += 0.25
        temp = 60 + 5 * math.sin(self._phase * 0.12) + random.uniform(-0.8, 0.8)
        pres = 1.0 + 0.2 * math.sin(self._phase * 0.07) + random.uniform(-0.02, 0.02)
        vib = 0.3 + 0.4 * abs(math.sin(self._phase * 0.5)) + random.uniform(-0.02, 0.02)
        return now, round(temp, 2), round(pres, 3), round(vib, 4)


# ---------------------------
# Plotly chart builders
# ---------------------------
def build_sensor_figure(buffer: SensorsBuffer):
    data = buffer.as_plotly()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Temperature (°C)", "Pressure (bar)", "Vibration (mm/s)"),
                        vertical_spacing=0.06)

    # Temperature
    fig.add_trace(go.Scatter(x=data["timestamps"], y=data["temperature"], name="Temperature", mode="lines+markers"), row=1, col=1)
    # Pressure
    fig.add_trace(go.Scatter(x=data["timestamps"], y=data["pressure"], name="Pressure", mode="lines+markers"), row=2, col=1)
    # Vibration
    fig.add_trace(go.Scatter(x=data["timestamps"], y=data["vibration"], name="Vibration", mode="lines+markers"), row=3, col=1)

    fig.update_layout(height=800, margin=dict(t=50, b=20, l=40, r=20), showlegend=False)
    fig.update_xaxes(title_text="UTC Time", row=3, col=1)
    return fig


# ---------------------------
# UI: Overview panel
# ---------------------------
def render_overview(ds: DataSource, buf: SensorsBuffer):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### System Snapshot")
        # basic KPIs
        k1, k2, k3 = st.columns(3)
        # Health score: derive from latest vibration/temperature heuristics
        health = 100
        if len(buf.temperature) > 0 and len(buf.vibration) > 0:
            latest_temp = buf.temperature[-1]
            latest_vib = buf.vibration[-1]
            # simple scoring
            health -= max(0, (latest_temp - 60) * 0.6)
            health -= max(0, (latest_vib - 0.5) * 10)
            health = max(0, min(100, round(health, 1)))
        else:
            health = 98.5

        k1.metric("Health Index", f"{health}%", delta=f"{round(100-health,1)}%")
        k2.metric("Prevented Failures", random.randint(0, 12))
        k3.metric("Operational Efficiency", f"{round(max(60, 100 - (100-health)),1)}%")

        st.markdown("**Digital Twin Status**")
        if ds.twin:
            st.success("Digital Twin: Available")
        else:
            st.info("Digital Twin: Not available (simulating)")

        st.markdown("**PLC Integration**")
        if ds.dm:
            st.success("Data Manager: Available")
        else:
            st.info("Data Manager: Not available (simulating)")

        st.markdown("---")
        st.markdown("**Notes:** This dashboard runs simulated sensor streams by default. Connect real Digital Twin / DataManager implementations to supply live sensor data.")

    with col2:
        st.markdown("### Controls")
        st.write("Live stream controls for sensor visualization.")
        # settings appear in sidebar too
        st.write("Update interval (seconds):")
        interval = st.session_state.get("sensor_interval", 3)
        new_interval = st.number_input("Interval (s)", min_value=1, max_value=30, value=interval, step=1, key="interval_input")
        if new_interval != interval:
            st.session_state["sensor_interval"] = new_interval

        running = st.session_state.get("sensors_running", False)
        if running:
            if st.button("⏸ Pause Stream"):
                st.session_state["sensors_running"] = False
        else:
            if st.button("▶️ Start Stream"):
                st.session_state["sensors_running"] = True

        st.write("Buffer length:", buf.timestamps.maxlen)
        st.write(f"Samples in buffer: {len(buf.timestamps)}")


# ---------------------------
# UI: Sensors panel
# ---------------------------
def render_sensors_panel(buffer: SensorsBuffer):
    st.subheader("📡 Sensors — Live Monitor")
    st.caption("Temperature, Pressure, and Vibration. Data updates while stream is running.")

    # Chart placeholder
    chart_placeholder = st.empty()

    # Show latest numeric readouts in a row
    if len(buffer.timestamps) > 0:
        latest_temp = buffer.temperature[-1]
        latest_pres = buffer.pressure[-1]
        latest_vib = buffer.vibration[-1]
        a, b, c, d = st.columns([1, 1, 1, 2])
        a.metric("Temp (°C)", f"{latest_temp}")
        b.metric("Pressure (bar)", f"{latest_pres}")
        c.metric("Vibration (mm/s)", f"{latest_vib}")
        # small sparkline for vibration trend (using plotly mini-chart)
        with d:
            spark = go.Figure()
            spark.add_trace(go.Scatter(y=list(buffer.vibration)[-40:], mode="lines", showlegend=False))
            spark.update_layout(height=80, margin=dict(t=10, b=10, l=0, r=0), xaxis={'visible': False}, yaxis={'visible': False})
            st.plotly_chart(spark, use_container_width=True)
    else:
        st.info("No samples yet. Start the stream to populate live charts.")

    # Full multi-subplot chart
    fig = build_sensor_figure(buffer)
    chart_placeholder.plotly_chart(fig, use_container_width=True)


# ---------------------------
# App main
# ---------------------------
def main():
    st.set_page_config(page_title="AVCS DNA-MATRIX SPIRIT v6.0", page_icon="🧬", layout="wide")

    # Initialize or load session state objects
    if "sensors_buf" not in st.session_state:
        st.session_state["sensors_buf"] = SensorsBuffer(maxlen=300)
    if "data_source" not in st.session_state:
        st.session_state["data_source"] = DataSource()
    if "sensor_interval" not in st.session_state:
        st.session_state["sensor_interval"] = 3
    if "sensors_running" not in st.session_state:
        st.session_state["sensors_running"] = False

    # HEADER
    logo = load_logo_image()
    header_col1, header_col2 = st.columns([1, 6])
    with header_col1:
        if logo:
            st.image(logo, width=140)
        else:
            st.markdown("**AVCS DNA-MATRIX SPIRIT**")
    with header_col2:
        st.markdown("<h1 style='margin:0'>AVCS DNA-MATRIX SPIRIT v6.0</h1>", unsafe_allow_html=True)
        st.markdown("<div style='color:gray; margin-top:0.2rem'>Operational Excellence Delivered...</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar navigation
    page = st.sidebar.radio("Navigate", ["Overview", "Sensors", "Digital Twin", "PLC Integration", "Logs"])

    # Routing
    if page == "Overview":
        render_overview(st.session_state["data_source"], st.session_state["sensors_buf"])
    elif page == "Sensors":
        render_sensors_panel(st.session_state["sensors_buf"])
    elif page == "Digital Twin":
        st.subheader("Digital Twin")
        if st.session_state["data_source"].twin:
            st.success("Digital Twin module available")
            st.write("You can run simulations or tweak parameters in the digital twin module (not yet exposed in UI).")
        else:
            st.info("Digital Twin not available — running simulated sensors.")
    elif page == "PLC Integration":
        st.subheader("PLC Integration")
        st.info("PLC adapters are ready in /plc_integration — connect your controllers via the integrator module.")
    elif page == "Logs":
        st.subheader("System Logs")
        st.info("Operational logs and audit packets will appear here (when enabled).")

    # Live stream loop (non-blocking, uses rerun)
    interval = st.session_state.get("sensor_interval", 3)
    running = st.session_state.get("sensors_running", False)

    # Controls in bottom area for convenience
    with st.expander("Stream Controls"):
        st.write("Interval (seconds):", interval)
        st.write("Running:", running)

    # If running, fetch a sample and append, then schedule rerun
    if running:
        try:
            ts, temp, pres, vib = st.session_state["data_source"].get_sample()
            st.session_state["sensors_buf"].append(ts, temp, pres, vib)
        except Exception as e:
            st.error(f"Sensor sample error: {e}")

        # throttle wait and rerun
        time.sleep(max(0.1, float(interval)))
        st.experimental_rerun()
    else:
        # not running - do nothing (user can start)
        pass


if __name__ == "__main__":
    main()
