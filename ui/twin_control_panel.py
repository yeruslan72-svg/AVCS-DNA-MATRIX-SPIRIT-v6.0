"""
twin_control_panel.py
Control panel UI for Digital Twin parameters and quick simulation.
"""

import streamlit as st
from datetime import datetime
import random

try:
    from digital_twin.industrial_digital_twin import IndustrialDigitalTwin
except Exception:
    IndustrialDigitalTwin = None

def render_twin_control():
    st.subheader("🔧 Digital Twin Control Panel")

    col1, col2 = st.columns(2)
    with col1:
        rpm = st.slider("RPM", min_value=500, max_value=6000, value=2950, step=50)
        load = st.slider("Load (%)", min_value=0, max_value=200, value=100, step=5)
    with col2:
        ambient = st.slider("Ambient Temp (°C)", 0, 60, 25, step=1)
        mode = st.selectbox("Failure Mode", ["normal", "bearing_wear", "misalignment", "imbalance", "cavitation"])

    if st.button("▶️ Run Twin Simulation"):
        params = {
            'rpm': rpm,
            'load': load,
            'ambient_temperature': ambient,
            'failure_mode': mode,
            'timestamp': datetime.utcnow().isoformat() + "Z"
        }
        st.session_state['twin_params'] = params

        # If digital twin exists, run and show sample
        if IndustrialDigitalTwin:
            try:
                twin = IndustrialDigitalTwin("centrifugal_pump")
                out = twin.simulate_equipment_behavior({
                    'rpm': rpm,
                    'load': load/100.0,
                    'ambient_temperature': ambient,
                    'failure_mode': mode
                })
                st.success("Digital Twin simulation completed")
                st.json(out)
            except Exception as e:
                st.error(f"Twin simulation failed: {e}")
        else:
            st.info("Digital Twin module not present — showing synthetic sample")
            sample = {
                'vibration_rms': round(0.3 + random.random() * 1.5, 3),
                'temperature': round(ambient + random.random() * 10, 2),
                'note': 'synthetic sample'
            }
            st.json(sample)

    # Show last params
    if st.session_state.get('twin_params'):
        st.markdown("**Last simulation parameters:**")
        st.json(st.session_state['twin_params'])
