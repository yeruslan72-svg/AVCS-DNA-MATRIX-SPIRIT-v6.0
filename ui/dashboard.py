import streamlit as st
from PIL import Image
import os

def main():
    # === APP CONFIG ===
    st.set_page_config(
        page_title="AVCS DNA-MATRIX SPIRIT v6.0",
        page_icon="⚙️",
        layout="wide"
    )

    # === LOAD LOGO ===
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, use_container_width=True)
    else:
        st.warning("⚠️ Logo not found. Please check /assets/logo.png")

    # === HEADER ===
    st.markdown(
        """
        <h1 style='text-align: center; color: #0A84FF;'>AVCS DNA-MATRIX SPIRIT v6.0</h1>
        <h3 style='text-align: center; color: gray;'>Operational Excellence Delivered...</h3>
        <hr style='margin-top: 10px; margin-bottom: 30px;'>
        """,
        unsafe_allow_html=True
    )

    # === DASHBOARD BODY PLACEHOLDER ===
    st.write("🚀 Dashboard is initializing...")
  import random
import time

def render_system_overview():
    st.subheader("🧠 System Overview & Status Monitor")

    # --- COLUMNS LAYOUT ---
    col1, col2, col3, col4, col5 = st.columns(5)

    # --- STATUS SIMULATION ---
    system_status = random.choice(["ONLINE", "WARNING", "OFFLINE"])
    sensor_temp = round(random.uniform(35.0, 85.0), 1)
    sensor_pressure = round(random.uniform(1.0, 10.0), 2)
    sensor_vibration = round(random.uniform(0.1, 2.5), 2)
    stream_latency = round(random.uniform(50, 300), 1)
    plc_connected = random.choice([True, False])

    # --- COLOR MAP ---
    status_color = {
        "ONLINE": "✅",
        "WARNING": "⚠️",
        "OFFLINE": "🔴"
    }

    # --- DISPLAY STATUS ---
    col1.metric("System Status", system_status, status_color[system_status])
    col2.metric("Temp °C", sensor_temp)
    col3.metric("Pressure bar", sensor_pressure)
    col4.metric("Vibration mm/s", sensor_vibration)
    col5.metric("Latency ms", stream_latency)

    # --- PLC Connection ---
    plc_status = "🟢 Connected" if plc_connected else "🔴 Disconnected"
    st.info(f"PLC Connection: **{plc_status}**")

    st.progress(random.randint(40, 100) / 100)
    st.caption("Live status refresh every 5 seconds...")

# Inject auto-refresh
if st.button("🔄 Refresh System Overview"):
    st.rerun()

# Auto refresh every 5 seconds
time.sleep(5)
render_system_overview()

if __name__ == "__main__":
    main()
