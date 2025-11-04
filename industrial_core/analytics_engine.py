"""
analytics_engine.py
AVCS DNA-MATRIX SPIRIT v6.1
Simple analytics engine: health score, risk index, anomaly detection, RUL estimate.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional

class AnalyticsEngine:
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        # Lightweight anomaly detector for multivariate features (vib, temp, pressure)
        try:
            self.detector = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
            # not yet trained — train_on() will be used
            self._trained = False
        except Exception:
            self.detector = None
            self._trained = False

    def train_on(self, df: pd.DataFrame):
        """Train anomaly detector on dataframe of normal operation rows (columns: vib,temp,pressure)"""
        if self.detector is None or df is None or df.empty:
            self._trained = False
            return
        try:
            X = df[['vibration', 'temperature', 'pressure']].values
            self.detector.fit(X)
            self._trained = True
        except Exception:
            self._trained = False

    def compute_features(self, sample: Dict[str, float]) -> Dict[str, float]:
        """Compute derived features from a single sample dict"""
        vib = float(sample.get('vibration', 0.0))
        temp = float(sample.get('temperature', 0.0))
        pres = float(sample.get('pressure', 0.0))
        features = {
            'vibration': vib,
            'temperature': temp,
            'pressure': pres,
            'vib_temp_ratio': vib / (temp + 1e-6),
            'temp_pressure_ratio': temp / (pres + 1e-6)
        }
        return features

    def predict(self, sample: Dict[str, float]) -> Dict[str, Any]:
        """Return health_score (0-100), risk_index (0-100), anomaly(bool), rul_hours (approx) and recommended_action"""
        features = self.compute_features(sample)
        vib = features['vibration']
        temp = features['temperature']
        pres = features['pressure']

        # Simple health scoring heuristic (tunable)
        health = 100.0
        # temperature penalty
        if temp > 70:
            health -= (temp - 70) * 0.8
        # vibration penalty
        if vib > 2.0:
            health -= (vib - 2.0) * 12.0
        # pressure penalty (example)
        if pres > 5.0:
            health -= (pres - 5.0) * 3.0

        health = max(0.0, min(100.0, health))

        # anomaly detection
        anomaly = False
        conf = 0.0
        if self._trained and self.detector is not None:
            try:
                X = np.array([[vib, temp, pres]])
                score = self.detector.decision_function(X)[0]  # higher means more normal
                conf = float(score)
                pred = self.detector.predict(X)[0]  # -1 anomaly, 1 normal
                anomaly = (pred == -1)
            except Exception:
                anomaly = False
                conf = 0.0

        # risk index: inverse of health, increased with anomaly
        risk = int(max(0, min(100, (100.0 - health) * 1.1 + (40 if anomaly else 0))))

        # simple RUL model (hours) — heuristic: more risk => shorter RUL
        rul_hours = max(0, int(200 - risk * 1.5))

        # recommended action
        if risk >= 80:
            action = "EMERGENCY: immediate inspection and damper -> CRITICAL"
        elif risk >= 50:
            action = "Schedule maintenance within 24h; increase damping"
        elif risk >= 20:
            action = "Monitor closely; schedule inspection in next shift"
        else:
            action = "Normal operation"

        return {
            'health_score': round(health, 2),
            'risk_index': risk,
            'anomaly': bool(anomaly),
            'detector_confidence': round(conf, 4),
            'rul_hours': int(rul_hours),
            'recommended_action': action,
            'features': features
        }
