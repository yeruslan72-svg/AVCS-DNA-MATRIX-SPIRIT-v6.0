"""
AVCS DNA-MATRIX SPIRIT v6.0
Adaptive Core Integration Module
--------------------------------
Central orchestrator that unites:
 - ContextManager
 - AdaptiveEngine
 - PatternRecognition
 - FeedbackController
 - Sample Data Generator

Provides continuous adaptive feedback for vibration control
and predictive maintenance.
"""

import time
import random
import pandas as pd
from datetime import datetime

# === import local adaptive modules ===
from adaptive_learning.context_manager import ContextManager
from adaptive_learning.adaptive_engine import AdaptiveEngine
from adaptive_learning.pattern_recognition import PatternRecognition
from adaptive_learning.feedback_controller import FeedbackController
from adaptive_learning.sample_data import generate_sample


class AdaptiveCore:
    """Main orchestrator for adaptive learning & feedback control."""

    def __init__(self):
        self.context = ContextManager()
        self.engine = AdaptiveEngine()
        self.patterns = PatternRecognition()
        self.controller = FeedbackController()

        self.history = []  # store logs of processed cycles
        self.initialized = False

    # -------------------------------------------------------------
    def initialize(self):
        """Initialize learning components with baseline normal data."""
        print("🧠 Initializing Adaptive Core...")

        normal_samples = [generate_sample("normal") for _ in range(200)]
        df = pd.DataFrame(normal_samples).drop(columns=["timestamp"])
        self.patterns.fit_anomaly(df)
        self.initialized = True
        print("✅ Adaptive Core initialized successfully.\n")

    # -------------------------------------------------------------
    def run_cycle(self, mode: str = "normal"):
        """Run one adaptive control cycle."""
        if not self.initialized:
            raise RuntimeError("AdaptiveCore not initialized — call initialize() first.")

        # 1️⃣ Generate synthetic telemetry
        telemetry = generate_sample(mode)

        # 2️⃣ Infer operational context
        ctx = self.context.infer_context(telemetry, metadata={"operator": "Auto", "shift": "B"})

        # 3️⃣ Detect anomaly and predict label
        is_anom, score = self.patterns.detect_anomaly({k: v for k, v in telemetry.items() if k != "timestamp"})
        label_info = self.patterns.predict_health_label({k: v for k, v in telemetry.items() if k != "timestamp"})
        degraded_prob = label_info["prob"][1] if label_info and label_info["prob"] else random.random() * 0.2

        # Risk index (0–100)
        risk_index = int((abs(score) * 50) + degraded_prob * 100)
        risk_index = min(100, max(0, risk_index))

        # 4️⃣ Adaptive engine ingestion & retraining simulation
        self.engine.ingest([{"risk": risk_index}])
        if random.random() < 0.05:  # retrain occasionally
            self.engine.retrain()

        # 5️⃣ Propose actuator actions
        actions = self.controller.propose_actions(risk_index, ctx)

        # 6️⃣ Log and return summary
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": mode,
            "context": ctx["mode"],
            "vibration": telemetry["vibration"],
            "temperature": telemetry["temperature"],
            "risk_index": risk_index,
            "damper_force": actions["damper_force"],
            "note": actions["note"],
        }
        self.history.append(record)

        print(f"[{record['timestamp']}] Mode={record['context']} | "
              f"Risk={risk_index}% | Damper={actions['damper_force']} | {actions['note']}")

        return record

    # -------------------------------------------------------------
    def get_history(self) -> pd.DataFrame:
        """Return all processed cycles as a DataFrame."""
        return pd.DataFrame(self.history)

    # -------------------------------------------------------------
    def status(self):
        """Return consolidated system status."""
        s = self.engine.status()
        s["initialized"] = self.initialized
        s["cycles"] = len(self.history)
        return s


# -------------------------------------------------------------
# Standalone test (local run)
# -------------------------------------------------------------
if __name__ == "__main__":
    core = AdaptiveCore()
    core.initialize()

    for _ in range(10):
        core.run_cycle(random.choice(["normal", "degraded"]))
        time.sleep(0.3)

    print("\nSystem status:")
    print(core.status())

    df = core.get_history()
    print("\nHistory preview:")
    print(df.tail(5))
