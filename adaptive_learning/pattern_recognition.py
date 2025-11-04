# adaptive_learning/pattern_recognition.py
"""
Pattern recognition: anomaly detection (IsolationForest) and a simple incremental classifier (SGDClassifier)
Designed for streaming data (online) and easy retraining.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class PatternRecognition:
    def __init__(self):
        # anomaly detector trained offline when data available
        self.anomaly = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self.scaler = StandardScaler()
        # incremental classifier (binary: normal / degraded) - needs partial_fit with classes
        self.classifier = SGDClassifier(loss="log", max_iter=1000, tol=1e-3)
        self._classifier_initialized = False

    def fit_anomaly(self, X: pd.DataFrame):
        """Fit IsolationForest on baseline normal data."""
        if X is None or len(X) < 50:
            return False
        cols = X.columns
        Xs = self.scaler.fit_transform(X.values)
        self.anomaly.fit(Xs)
        return True

    def detect_anomaly(self, sample: dict):
        """Return (is_anomaly, score)"""
        x = np.array([sample[k] for k in sorted(sample.keys())], dtype=float).reshape(1, -1)
        xs = self.scaler.transform(x)
        score = float(self.anomaly.decision_function(xs)[0]) if hasattr(self.anomaly, "decision_function") else 0.0
        is_anom = bool(self.anomaly.predict(xs)[0] == -1)
        return is_anom, score

    def partial_train_classifier(self, X: pd.DataFrame, y: pd.Series):
        """Train incremental classifier with partial_fit"""
        Xs = self.scaler.transform(X.values)
        if not self._classifier_initialized:
            classes = np.unique(y)
            self.classifier.partial_fit(Xs, y.values, classes=classes)
            self._classifier_initialized = True
        else:
            self.classifier.partial_fit(Xs, y.values)

    def predict_health_label(self, sample: dict):
        """Return probability of 'degraded' if classifier trained, else None"""
        if not self._classifier_initialized:
            return None
        x = np.array([sample[k] for k in sorted(sample.keys())], dtype=float).reshape(1, -1)
        xs = self.scaler.transform(x)
        prob = self.classifier.predict_proba(xs)[0].tolist() if hasattr(self.classifier, "predict_proba") else None
        label = self.classifier.predict(xs)[0]
        return {"label": int(label), "prob": prob}

    def save(self):
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
        joblib.dump(self.anomaly, os.path.join(MODEL_DIR, "anomaly_if.joblib"))
        joblib.dump(self.classifier, os.path.join(MODEL_DIR, "clf_sgd.joblib"))

    def load(self):
        try:
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
            self.anomaly = joblib.load(os.path.join(MODEL_DIR, "anomaly_if.joblib"))
            self.classifier = joblib.load(os.path.join(MODEL_DIR, "clf_sgd.joblib"))
            self._classifier_initialized = True
        except Exception:
            pass
