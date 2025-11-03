import streamlit as st
import random
import time
from PIL import Image

# ==============================
# AVCS DNA-MATRIX SPIRIT DASHBOARD v6.0
# ==============================

def load_logo():
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, width=200)
    except Exception:
        st.warning("⚠️ Logo not found in assets/logo.png")

def render_system_overview():
    st.subheader("🧠 System Overview & Status Monitor")

    # --- Layout columns ---
    col1, col2, col3, col4, col5 = st.columns(5)

    # --- Simulated system data ---
    system_status = random.choice(["ONLINE", "WARNING", "OFFLINE"])
    sensor_temp = round(random.uniform(35.0, 85.0), 1)
    sensor_pressure = round(random.uniform(1.0, 10.0), 2)
    sensor_vibration = round(random.uniform(0.1, 2.5), 2)
    stream_latency = round(random.uniform(50, 300), 1)
    plc_connected = random.choice([True, False])

    # --- Color indicators ---
    status_color = {
        "ONLINE": "✅",
        "WARNING": "⚠️",
        "OFFLINE": "🔴"
    }

    # --- Display metrics ---
    col1.metric("System Status", system_status, status_color[system_status])
    col2.metric("Temp °C", sensor_temp)
    col3.metric("Pressure bar", sensor_pressure)
    col4.metric("Vibration mm/s", sensor_vibration)
    col5.metric("Latency ms", stream_latency)

    # --- PLC Connection ---
    plc_status = "🟢 Connected" if plc_connected else "🔴 Disconnected"
    st.info(f"PLC Connection: **{plc_status}**")

    # --- Sync progress bar ---
    st.progress(random.randint(40, 100) / 100)
    st.caption("Live status refresh every 5 seconds...")

# ==============================
# MAIN APP DASHBOARD
# ==============================

def main():
    st.set_page_config(
        page_title="AVCS DNA-MATRIX SPIRIT v6.0",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_logo()
    st.title("🚀 AVCS DNA-MATRIX SPIRIT v6.0 Dashboard")
    st.write("Smart Industrial Digital Twin & Control Interface")

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select view:",
        ["Overview", "Sensors", "Digital Twin", "PLC Integration", "System Settings"]
    )

    # Page routing
    if page == "Overview":
        render_system_overview()
    elif page == "Sensors":
        st.info("📡 Sensor data visualization coming soon...")
    elif page == "Digital Twin":
        st.info("🧩 Digital Twin module loading...")
    elif page == "PLC Integration":
        st.info("🔌 PLC Integration interface coming soon...")
    elif page == "System Settings":
        st.info("⚙️ System configuration settings coming soon...")

    # Auto refresh mechanism
    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main()
