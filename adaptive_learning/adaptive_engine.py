"""
adaptive_learning/adaptive_engine.py

Minimal AdaptiveEngine class with stable API:
 - status() -> dict (progress, patterns, last_trained)
 - retrain(activity=None) -> dict (result summary)
 - ingest(activity) -> None

This scaffold is intentionally simple — plug your RL/online learning inside.
"""

from datetime import datetime
import numpy as np

class AdaptiveEngine:
    def __init__(self):
        self.progress = 0
        self.pattern = "init"
        self.last_trained = None
        self.model_version = "v0.0"
        # internal counters
        self._seen = 0

    def status(self):
        return {
            "progress": int(self.progress),
            "pattern": self.pattern,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
            "model_version": self.model_version,
            "seen": int(self._seen)
        }

    def ingest(self, activity):
        """Ingest activity list/dict — update counters and quick stats"""
        if not activity:
            return
        try:
            n = len(activity)
        except Exception:
            n = 1
        self._seen += n
        # quick pattern detection stub
        risks = [a.get("risk", 0) for a in activity if isinstance(a, dict)]
        if len(risks) > 0 and np.mean(risks) > 50:
            self.pattern = "high_risk_cluster"
        else:
            self.pattern = "stable"
        # bump progress slightly
        self.progress = min(100, self.progress + min(5, n/10))

    def retrain(self, activity=None):
        """Trigger retraining — either on provided activity or internal store"""
        # Very light placeholder: simulate training and bump version
        self.ingest(activity or [])
        self.last_trained = datetime.utcnow()
        major, minor = map(int, self.model_version.lstrip("v").split("."))
        minor += 1
        self.model_version = f"v{major}.{minor}"
        # increase progress as a result
        self.progress = min(100, self.progress + 10)
        return {"status": "ok", "model_version": self.model_version, "trained_at": self.last_trained.isoformat()}
