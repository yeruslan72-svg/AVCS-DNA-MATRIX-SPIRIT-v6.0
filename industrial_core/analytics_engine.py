"""
AVCS DNA-MATRIX SPIRIT v7.0
Enhanced Analytics Engine with Adaptive Thresholds & Digital Twin Integration
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from enum import Enum

class EquipmentState(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AnalyticsEngine:
    def __init__(self, equipment_type: str = "centrifugal_pump", random_state: int = 42):
        self.equipment_type = equipment_type
        self.logger = logging.getLogger(f"AnalyticsEngine.{equipment_type}")
        
        # ML Models
        self.detector = IsolationForest(
            n_estimators=150, 
            contamination=0.05, 
            random_state=random_state
        )
        self.scaler = StandardScaler()
        
        # Adaptive thresholds based on equipment type
        self.thresholds = self._load_equipment_thresholds(equipment_type)
        self.operation_history: List[Dict] = []
        self._trained = False
        
    def _load_equipment_thresholds(self, equipment_type: str) -> Dict[str, float]:
        """Load equipment-specific thresholds."""
        thresholds = {
            "centrifugal_pump": {
                "vibration_critical": 8.0,
                "vibration_warning": 4.0,
                "temp_critical": 85.0,
                "temp_warning": 70.0,
                "pressure_critical": 10.0,
                "pressure_warning": 6.0
            },
            "compressor": {
                "vibration_critical": 12.0,
                "vibration_warning": 6.0,
                "temp_critical": 95.0,
                "temp_warning": 80.0,
                "pressure_critical": 15.0,
                "pressure_warning": 10.0
            }
        }
        return thresholds.get(equipment_type, thresholds["centrifugal_pump"])
    
    def train_on(self, df: pd.DataFrame, features: List[str] = None):
        """Train anomaly detector on normal operation data."""
        if self.detector is None or df.empty:
            self._trained = False
            return
            
        try:
            feature_cols = features or ['vibration', 'temperature', 'pressure']
            X = df[feature_cols].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train detector
            self.detector.fit(X_scaled)
            self._trained = True
            
            self.logger.info(f"Model trained on {len(X)} samples with features {feature_cols}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self._trained = False

    def compute_advanced_features(self, sample: Dict[str, float]) -> Dict[str, float]:
        """Compute comprehensive feature set for analysis."""
        vib = float(sample.get('vibration', 0.0))
        temp = float(sample.get('temperature', 0.0))
        pres = float(sample.get('pressure', 0.0))
        rpm = float(sample.get('rpm', 2950.0))
        
        # Advanced feature engineering
        features = {
            # Raw measurements
            'vibration': vib,
            'temperature': temp, 
            'pressure': pres,
            'rpm': rpm,
            
            # Physical ratios
            'vib_temp_ratio': vib / (temp + 1e-6),
            'temp_pressure_ratio': temp / (pres + 1e-6),
            'vib_rpm_ratio': vib / (rpm / 1000 + 1e-6),
            
            # Normalized features
            'vib_norm': vib / self.thresholds['vibration_critical'],
            'temp_norm': temp / self.thresholds['temp_critical'],
            'pres_norm': pres / self.thresholds['pressure_critical'],
            
            # Composite indicators
            'severity_index': (vib/self.thresholds['vibration_critical'] + 
                             temp/self.thresholds['temp_critical'] + 
                             pres/self.thresholds['pressure_critical']) / 3.0
        }
        return features

    def calculate_health_score(self, features: Dict[str, float]) -> float:
        """Calculate comprehensive health score with equipment-specific logic."""
        vib = features['vibration']
        temp = features['temperature']
        pres = features['pressure']
        severity = features['severity_index']
        
        health = 100.0
        
        # Temperature degradation
        if temp > self.thresholds['temp_warning']:
            temp_penalty = (temp - self.thresholds['temp_warning']) * 1.2
            health -= min(temp_penalty, 40)  # Cap penalty
            
        # Vibration degradation  
        if vib > self.thresholds['vibration_warning']:
            vib_penalty = (vib - self.thresholds['vibration_warning']) * 8.0
            health -= min(vib_penalty, 50)
            
        # Pressure degradation
        if pres > self.thresholds['pressure_warning']:
            pres_penalty = (pres - self.thresholds['pressure_warning']) * 5.0
            health -= min(pres_penalty, 30)
            
        # Composite severity penalty
        health -= severity * 15.0
        
        return max(0.0, min(100.0, health))

    def detect_equipment_state(self, health: float, features: Dict[str, float]) -> EquipmentState:
        """Determine equipment operational state."""
        vib = features['vibration']
        temp = features['temperature']
        pres = features['pressure']
        
        # Emergency conditions
        if (health < 20 or 
            vib > self.thresholds['vibration_critical'] or 
            temp > self.thresholds['temp_critical'] or
            pres > self.thresholds['pressure_critical']):
            return EquipmentState.EMERGENCY
            
        # Critical state
        elif health < 40:
            return EquipmentState.CRITICAL
            
        # Degraded state
        elif health < 70:
            return EquipmentState.DEGRADED
            
        # Normal operation
        else:
            return EquipmentState.NORMAL

    def predict_advanced_rul(self, health: float, state: EquipmentState, trend: float) -> int:
        """Calculate remaining useful life with trend analysis."""
        base_rul = 0
        
        if state == EquipmentState.NORMAL:
            base_rul = 500  # hours
        elif state == EquipmentState.DEGRADED:
            base_rul = 200
        elif state == EquipmentState.CRITICAL:
            base_rul = 72
        else:  # EMERGENCY
            base_rul = 24
            
        # Adjust based on health and trend
        health_factor = health / 100.0
        trend_factor = max(0.5, 1.0 - (trend * 2.0))  # Negative trend reduces RUL
        
        rul = int(base_rul * health_factor * trend_factor)
        return max(8, rul)  # Minimum 8 hours

    def get_maintenance_recommendation(self, state: EquipmentState, health: float) -> str:
        """Generate specific maintenance recommendations."""
        recommendations = {
            EquipmentState.NORMAL: 
                "Continue normal operation. Monitor standard parameters.",
                
            EquipmentState.DEGRADED: 
                f"Increase damping force. Schedule inspection within 48h. Health: {health:.1f}%",
                
            EquipmentState.CRITICAL: 
                f"CRITICAL: Increase damping to maximum. Schedule IMMEDIATE inspection. Health: {health:.1f}%",
                
            EquipmentState.EMERGENCY: 
                f"🚨 EMERGENCY: Initiate shutdown procedure. Health: {health:.1f}%. Evacuate area if safe."
        }
        
        return recommendations[state]

    def predict(self, sample: Dict[str, float]) -> Dict[str, Any]:
        """Enhanced prediction with advanced analytics."""
        try:
            # Feature computation
            features = self.compute_advanced_features(sample)
            
            # Health assessment
            health = self.calculate_health_score(features)
            state = self.detect_equipment_state(health, features)
            
            # Anomaly detection
            anomaly, confidence = self._detect_anomaly(features)
            
            # Trend analysis (simplified)
            trend = self._analyze_trend(health)
            
            # RUL prediction
            rul_hours = self.predict_advanced_rul(health, state, trend)
            
            # Maintenance recommendation
            action = self.get_maintenance_recommendation(state, health)
            
            # Risk calculation
            risk_index = self._calculate_risk_index(health, state, anomaly)
            
            return {
                'health_score': round(health, 2),
                'risk_index': risk_index,
                'equipment_state': state.value,
                'anomaly': bool(anomaly),
                'detector_confidence': round(confidence, 4),
                'rul_hours': rul_hours,
                'recommended_action': action,
                'trend_indicator': round(trend, 4),
                'features': features,
                'timestamp': datetime.utcnow().isoformat(),
                'equipment_type': self.equipment_type
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self._get_fallback_prediction()

    def _detect_anomaly(self, features: Dict[str, float]) -> tuple:
        """Enhanced anomaly detection with scaled features."""
        if not self._trained or self.detector is None:
            return False, 0.0
            
        try:
            # Use key features for detection
            feature_vector = np.array([[
                features['vibration'],
                features['temperature'], 
                features['pressure']
            ]])
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get anomaly score
            confidence = self.detector.decision_function(feature_vector_scaled)[0]
            prediction = self.detector.predict(feature_vector_scaled)[0]
            
            return (prediction == -1), float(confidence)
            
        except Exception:
            return False, 0.0

    def _analyze_trend(self, current_health: float) -> float:
        """Analyze health degradation trend."""
        if len(self.operation_history) < 5:
            return 0.0
            
        # Simple linear trend from recent history
        recent_health = [h['health_score'] for h in self.operation_history[-10:]]
        if len(recent_health) >= 3:
            trend = np.polyfit(range(len(recent_health)), recent_health, 1)[0]
            return trend
        return 0.0

    def _calculate_risk_index(self, health: float, state: EquipmentState, anomaly: bool) -> int:
        """Calculate comprehensive risk index."""
        base_risk = 100 - health
        
        # State multipliers
        state_multipliers = {
            EquipmentState.NORMAL: 1.0,
            EquipmentState.DEGRADED: 1.3, 
            EquipmentState.CRITICAL: 1.7,
            EquipmentState.EMERGENCY: 2.0
        }
        
        risk = base_risk * state_multipliers[state]
        
        # Anomaly penalty
        if anomaly:
            risk += 20
            
        return int(max(0, min(100, risk)))

    def _get_fallback_prediction(self) -> Dict[str, Any]:
        """Provide fallback prediction when analysis fails."""
        return {
            'health_score': 50.0,
            'risk_index': 50,
            'equipment_state': 'unknown',
            'anomaly': False,
            'detector_confidence': 0.0,
            'rul_hours': 100,
            'recommended_action': 'System error - use manual monitoring',
            'trend_indicator': 0.0,
            'features': {},
            'timestamp': datetime.utcnow().isoformat(),
            'equipment_type': self.equipment_type
        }

    def update_operation_history(self, prediction: Dict[str, Any]):
        """Maintain operation history for trend analysis."""
        self.operation_history.append({
            'timestamp': datetime.utcnow(),
            'health_score': prediction['health_score'],
            'risk_index': prediction['risk_index'],
            'equipment_state': prediction['equipment_state']
        })
        
        # Keep only recent history
        if len(self.operation_history) > 100:
            self.operation_history = self.operation_history[-50:]
