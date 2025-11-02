"""
AVCS DNA-MATRIX SPIRIT v6.0
Operational Intelligence Interface
----------------------------------
Central Streamlit / CLI application that connects industrial core,
digital twin simulation, and PLC integration layers.
"""

import streamlit as st
from industrial_core.data_manager import DataManager
from industrial_core.industrial_config import IndustrialConfig
from digital_twin.industrial_digital_twin import IndustrialTwin
from plc_integration.integration_adapters import SiemensIntegrationAdapter


def main():
    st.set_page_config(page_title="AVCS DNA-MATRIX SPIRIT v6.0", layout="wide")
    st.title("⚙️ AVCS DNA-MATRIX SPIRIT v6.0")
    st.subheader("Operational Intelligence Interface")

    # Load configuration
    config = IndustrialConfig.load_default()
    data_manager = DataManager(config)
    twin = IndustrialTwin()
    plc_adapter = SiemensIntegrationAdapter()

    st.sidebar.title("System Control")
    mode = st.sidebar.radio("Select Mode", ["Dashboard", "Simulation", "Integration"])

    if mode == "Dashboard":
        st.write("### Live Asset Overview")
        health, vibration = data_manager.get_live_data()
        st.metric("Health Index", f"{health:.2f}")
        st.line_chart(vibration)

    elif mode == "Simulation":
        st.write("### Digital Twin Simulation")
        twin_output = twin.run_simulation()
        st.json(twin_output)

    elif mode == "Integration":
        st.write("### PLC Integration Guide")
        guide = plc_adapter.generate_integration_guide()
        st.json(guide)


if __name__ == "__main__":
    main()

