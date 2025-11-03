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

if __name__ == "__main__":
    main()
