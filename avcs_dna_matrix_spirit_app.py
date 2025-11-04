# ============================================================
# 🚀 AVCS DNA-MATRIX SPIRIT v7.x — Main Application Launcher
# ============================================================
# Integrates all system modules:
# - Digital Twin (MATRIX Layer)
# - Adaptive Learning (SPIRIT Layer)
# - Core Industrial Intelligence (DNA Layer)
# ============================================================

import os
import streamlit.web.cli as stcli
import sys
import subprocess

# ============================================================
# 🧭 System Metadata
# ============================================================
APP_NAME = "AVCS DNA-MATRIX SPIRIT"
APP_VERSION = "v7.x"
APP_DESCRIPTION = "Operational Excellence Delivered — The Evolution of Industrial Intelligence"

# ============================================================
# ⚙️ Directory Validation
# ============================================================

ESSENTIAL_DIRS = [
    "ui",
    "digital_twin",
    "industrial_core",
    "plc_integration",
    "adaptive_learning",
    "assets",
    "data"
]

for directory in ESSENTIAL_DIRS:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INIT] Created missing directory: {directory}")

# ============================================================
# 📦 Streamlit Entry Point
# ============================================================

def run_dashboard():
    """Launch the unified Streamlit dashboard."""
    dashboard_path = os.path.join("ui", "dashboard.py")

    if not os.path.exists(dashboard_path):
        raise FileNotFoundError(f"❌ Dashboard not found: {dashboard_path}")

    print(f"\n🔹 Launching {APP_NAME} {APP_VERSION}")
    print(f"🔸 Description: {APP_DESCRIPTION}")
    print(f"🔸 Dashboard: {dashboard_path}\n")

    # Run the Streamlit dashboard
    subprocess.run(["streamlit", "run", dashboard_path])

# ============================================================
# 🧬 Adaptive Pre-Checks
# ============================================================

def system_health_check():
    """Perform quick environment validation."""
    print("🔍 Performing system health check...")

    # Check for required Python packages
    required = ["streamlit", "pandas", "numpy"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"⚠️ Missing packages: {', '.join(missing)}")
        print("💡 Installing now...")
        subprocess.run([sys.executable, "-m", "pip", "install", *missing])

    print("✅ Environment ready.\n")

# ============================================================
# 🧠 Main Entry
# ============================================================

def main():
    """Main entry point for AVCS DNA-MATRIX SPIRIT."""
    print(f"🧬 Initializing {APP_NAME} — {APP_VERSION}")
    system_health_check()
    run_dashboard()

# ============================================================
# 🚀 Start
# ============================================================

if __name__ == "__main__":
    main()
