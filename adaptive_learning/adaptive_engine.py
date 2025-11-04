# adaptive_learning/adaptive_engine.py
"""
Adaptive Engine: orchestrator for pattern recognition, context inference, and feedback suggestions.
Logs audit records into data/adaptive_audit.csv for traceability.
"""
import os
import json
from datetime import datetime
from .pattern_recognition import PatternRecognition
from .context_manager import ContextManager
from .feedback_controller import FeedbackController
import pandas as pd

AUDIT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "adaptive_audit.csv")

class AdaptiveEngine:
    def __init__(self):
        self.pr = PatternRecognition()
        self.ctx = ContextManager()
        self.fb = FeedbackController()
        os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
        if not os.path.exists(AUDIT_PATH):
            with open(AUDIT_PATH, "w") as f:
                f.write("timestamp,input,context,analysis,proposal\n")

    def analyze_and_propose(self, telemetry: dict, metadata: dict = None):
        """
        telemetry: {'vibration':..., 'temperature':..., 'pressure':..., 'rpm':..., 'load':...}
        metadata: operator, shift etc.
        Returns: dict with analysis and proposal
        """
        ctx = self.ctx.infer_context(telemetry, metadata)
        # anomaly detection
        sample = {k: float(telemetry.get(k, 0.0)) for k in ['vibration','temperature','pressure']}
        is_anom, score = self.pr.detect_anomaly(sample) if hasattr(self.pr, 'detect_anomaly') else (False, 0.0)
        # simple health/risk heuristic (can be replaced by analytics engine)
        health_score = max(0.0, 100.0 - (telemetry.get('vibration',0.0)*10 + max(0, telemetry.get('temperature',0)-70)*0.8))
        risk_index = min(100, int((100-health_score) + (30 if is_anom else 0)))

        proposal = self.fb.propose_actions(risk_index, ctx)
        analysis = {
            'health_score': round(health_score,2),
            'risk_index': risk_index,
            'is_anomaly': bool(is_anom),
            'anomaly_score': float(score)
        }

        # audit log
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'input': telemetry,
            'context': ctx,
            'analysis': analysis,
            'proposal': proposal
        }
        try:
            with open(AUDIT_PATH, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return {'analysis': analysis, 'proposal': proposal, 'context': ctx}
