"""
AVCS DNA-MATRIX SPIRIT v7.0
Advanced Digital Twin Module
----------------------------
High-fidelity equipment simulation with physics-based models,
predictive maintenance, and real-time synchronization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
from scipy import signal
import warnings


class EquipmentState(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class FailureMode(Enum):
    BEARING_WEAR = "bearing_wear"
    IMBALANCE = "imbalance"
    MISALIGNMENT = "misalignment"
    CAVITATION = "cavitation"
    LUBRICATION = "lubrication_issue"
    RESONANCE = "resonance"


@dataclass
class VibrationSignature:
    """Vibration signature analysis results."""
    rms: float
    peak: float
    crest_factor: float
    kurtosis: float
    dominant_frequencies: List[float]
    harmonic_ratios: List[float]


@dataclass
class HealthMetrics:
    """Comprehensive health assessment metrics."""
    overall_health: float  # 0.0 - 1.0
    bearing_health: float
    alignment_health: float
    balance_health: float
    lubrication_health: float
    predicted_rul: int  # hours
    confidence: float
    failure_risk: float


class IndustrialDigitalTwin:
    """
    Advanced Digital Twin for industrial equipment with physics-based modeling
    and predictive analytics.
    """

    def __init__(self, equipment_type: str, config: Dict[str, Any] = None):
        self.equipment_type = equipment_type
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(f"DigitalTwin.{equipment_type}")
        
        # State variables
        self.health_state = 1.0  # 100% healthy
        self.equipment_state = EquipmentState.NORMAL
        self.operational_hours = 0
        self.last_maintenance = datetime.utcnow()
        
        # Failure progression models
        self.degradation_rates = self._initialize_degradation_models()
        self.failure_modes = self._initialize_failure_modes()
        
        # Historical data for trend analysis
        self.operation_history: List[Dict] = []
        self.vibration_history: List[float] = []
        self.temperature_history: List[float] = []
        
        # Physical parameters
        self.physical_params = self._initialize_physical_parameters()
        
        self.logger.info(f"Digital Twin initialized for {equipment_type}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for equipment type."""
        return {
            "nominal_rpm": 2950,
            "max_temperature": 90.0,
            "max_vibration": 7.0,
            "bearing_life": 10000,  # hours
            "maintenance_interval": 2000,  # hours
            "sampling_rate": 25600,
            "degradation_rate": 0.0001  # per hour
        }

    def _initialize_physical_parameters(self) -> Dict[str, Any]:
        """Initialize physical parameters based on equipment type."""
        if "pump" in self.equipment_type.lower():
            return {
                "mass": 1500,  # kg
                "stiffness": 1e6,  # N/m
                "damping": 5000,  # Ns/m
                "bearing_stiffness": 2e6,
                "natural_frequency": 24.5  # Hz
            }
        elif "compressor" in self.equipment_type.lower():
            return {
                "mass": 3000,
                "stiffness": 2e6,
                "damping": 8000,
                "bearing_stiffness": 3e6,
                "natural_frequency": 18.2
            }
        else:
            return {
                "mass": 1000,
                "stiffness": 1.5e6,
                "damping": 6000,
                "bearing_stiffness": 2e6,
                "natural_frequency": 22.0
            }

    def _initialize_degradation_models(self) -> Dict[str, float]:
        """Initialize degradation rates for different components."""
        return {
            "bearing_wear": 0.00005,
            "imbalance_growth": 0.00002,
            "misalignment": 0.00001,
            "lubrication_degradation": 0.00003,
            "general_wear": 0.00001
        }

    def _initialize_failure_modes(self) -> Dict[FailureMode, float]:
        """Initialize failure mode progression states."""
        return {
            FailureMode.BEARING_WEAR: 0.0,
            FailureMode.IMBALANCE: 0.0,
            FailureMode.MISALIGNMENT: 0.0,
            FailureMode.CAVITATION: 0.0,
            FailureMode.LUBRICATION: 0.0,
            FailureMode.RESONANCE: 0.0
        }

    def simulate_equipment_behavior(self, operating_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced simulation of equipment behavior under specified conditions.
        """
        try:
            # Update operational state
            self._update_operational_state(operating_conditions)
            
            # Generate high-fidelity vibration data
            vibration_data = self._generate_high_fidelity_vibration(operating_conditions)
            vibration_analysis = self._analyze_vibration_signature(vibration_data)
            
            # Simulate thermal behavior
            temperature_data = self._simulate_advanced_temperature(operating_conditions)
            
            # Calculate comprehensive health metrics
            health_metrics = self._calculate_comprehensive_health()
            
            # Predict remaining useful life
            rul_prediction = self._predict_remaining_useful_life()
            
            # Detect anomalies and failures
            anomaly_detection = self._detect_anomalies(vibration_analysis, temperature_data)
            
            # Update historical data
            self._update_historical_data(vibration_analysis, temperature_data, health_metrics)
            
            simulation_result = {
                "timestamp": datetime.utcnow().isoformat(),
                "equipment_state": self.equipment_state.value,
                "vibration_data": {
                    "waveform": vibration_data.tolist(),
                    "analysis": asdict(vibration_analysis)
                },
                "temperature_data": temperature_data,
                "health_metrics": asdict(health_metrics),
                "rul_prediction": rul_prediction,
                "anomaly_detection": anomaly_detection,
                "operating_conditions": operating_conditions,
                "failure_risks": self.failure_modes.copy()
            }
            
            self.logger.debug(f"Simulation completed - Health: {health_metrics.overall_health:.3f}")
            return simulation_result
            
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            return self._get_fallback_simulation()

    def _update_operational_state(self, conditions: Dict[str, Any]):
        """Update operational state based on conditions and time."""
        # Simulate time progression (1 simulation = 1 hour of operation)
        self.operational_hours += 1
        
        # Update degradation based on operating conditions
        load_factor = conditions.get("load", 0.5)
        speed_factor = conditions.get("rpm", self.config["nominal_rpm"]) / self.config["nominal_rpm"]
        
        # Accelerate degradation under stress conditions
        stress_factor = max(1.0, load_factor * speed_factor)
        
        for mode in self.failure_modes:
            degradation_rate = self.degradation_rates.get(mode.value, 0.00001)
            self.failure_modes[mode] = min(1.0, 
                self.failure_modes[mode] + degradation_rate * stress_factor
            )
        
        # Update overall health state
        max_failure = max(self.failure_modes.values())
        self.health_state = max(0.0, 1.0 - max_failure)
        
        # Update equipment state
        if self.health_state > 0.8:
            self.equipment_state = EquipmentState.NORMAL
        elif self.health_state > 0.5:
            self.equipment_state = EquipmentState.DEGRADED
        elif self.health_state > 0.2:
            self.equipment_state = EquipmentState.CRITICAL
        else:
            self.equipment_state = EquipmentState.FAILED

    def _generate_high_fidelity_vibration(self, conditions: Dict[str, Any]) -> np.ndarray:
        """Generate high-fidelity vibration data with physics-based modeling."""
        rpm = conditions.get("rpm", self.config["nominal_rpm"])
        load = conditions.get("load", 0.5)
        failure_mode = conditions.get("failure_mode", "normal")
        
        # Time vector for 1 second of data
        fs = self.config["sampling_rate"]
        t = np.linspace(0, 1, fs, endpoint=False)
        
        # Fundamental rotational frequency
        rotational_freq = rpm / 60.0
        
        # Base vibration from rotation
        base_vibration = 0.5 * np.sin(2 * np.pi * rotational_freq * t)
        
        # Harmonic components
        harmonics = np.zeros_like(t)
        for i in range(2, 6):  # 2nd to 5th harmonics
            harmonics += 0.1 * np.sin(2 * np.pi * rotational_freq * i * t) / i
        
        # Bearing frequencies (simplified)
        bearing_freq = rotational_freq * 3.2  # BPFO approximation
        bearing_vibration = 0.2 * np.sin(2 * np.pi * bearing_freq * t)
        
        # Failure mode effects
        failure_components = self._apply_failure_modes(t, rotational_freq, failure_mode)
        
        # Load effect
        load_effect = load * 0.3 * np.sin(2 * np.pi * rotational_freq * 0.7 * t)
        
        # Combine all components
        combined_vibration = (
            base_vibration + 
            harmonics + 
            bearing_vibration + 
            failure_components + 
            load_effect
        )
        
        # Add realistic noise
        noise = np.random.normal(0, 0.05, len(t))
        
        # Apply health state degradation
        degraded_vibration = combined_vibration * (1 + (1 - self.health_state) * 2)
        
        final_vibration = degraded_vibration + noise
        
        # Scale to realistic amplitudes (mm/s)
        return final_vibration * 2.0

    def _apply_failure_modes(self, t: np.ndarray, base_freq: float, failure_mode: str) -> np.ndarray:
        """Apply specific failure mode characteristics to vibration."""
        component = np.zeros_like(t)
        
        if failure_mode == "bearing_wear":
            # Increased bearing frequency components
            bearing_harmonics = 0.3 * np.sin(2 * np.pi * base_freq * 3.2 * t)
            bearing_harmonics += 0.2 * np.sin(2 * np.pi * base_freq * 6.4 * t)
            component += bearing_harmonics * self.failure_modes[FailureMode.BEARING_WEAR]
            
        elif failure_mode == "imbalance":
            # Strong 1x rotational component
            component += 0.4 * np.sin(2 * np.pi * base_freq * t) * self.failure_modes[FailureMode.IMBALANCE]
            
        elif failure_mode == "misalignment":
            # Strong 2x rotational component
            component += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t) * self.failure_modes[FailureMode.MISALIGNMENT]
            
        elif failure_mode == "cavitation":
            # Random high-frequency bursts
            bursts = np.random.normal(0, 0.2, len(t)) * (np.random.random(len(t)) > 0.95)
            component += bursts * self.failure_modes[FailureMode.CAVITATION]
        
        return component

    def _analyze_vibration_signature(self, vibration_data: np.ndarray) -> VibrationSignature:
        """Perform comprehensive vibration analysis."""
        # Basic statistical features
        rms = np.sqrt(np.mean(vibration_data**2))
        peak = np.max(np.abs(vibration_data))
        crest_factor = peak / rms if rms > 0 else 0
        kurtosis = float(pd.Series(vibration_data).kurtosis())
        
        # Frequency domain analysis
        fs = self.config["sampling_rate"]
        freqs, power_spectrum = signal.welch(vibration_data, fs, nperseg=1024)
        
        # Find dominant frequencies
        dominant_indices = np.argsort(power_spectrum)[-5:]  # Top 5 frequencies
        dominant_frequencies = freqs[dominant_indices].tolist()
        
        # Calculate harmonic ratios
        fundamental_freq = self.config["nominal_rpm"] / 60.0
        harmonic_ratios = []
        for i in range(1, 6):
            harmonic_freq = fundamental_freq * i
            # Find power near harmonic frequency
            freq_mask = (freqs > harmonic_freq * 0.9) & (freqs < harmonic_freq * 1.1)
            if np.any(freq_mask):
                harmonic_power = np.max(power_spectrum[freq_mask])
                harmonic_ratios.append(harmonic_power / np.max(power_spectrum))
            else:
                harmonic_ratios.append(0.0)
        
        return VibrationSignature(
            rms=float(rms),
            peak=float(peak),
            crest_factor=float(crest_factor),
            kurtosis=kurtosis,
            dominant_frequencies=dominant_frequencies,
            harmonic_ratios=harmonic_ratios
        )

    def _simulate_advanced_temperature(self, conditions: Dict[str, Any]) -> Dict[str, float]:
        """Simulate thermal behavior with heat transfer modeling."""
        ambient_temp = conditions.get("ambient_temperature", 25.0)
        load = conditions.get("load", 0.5)
        rpm = conditions.get("rpm", self.config["nominal_rpm"])
        
        # Base temperature rise from operation
        base_heat = load * (rpm / self.config["nominal_rpm"]) * 30.0
        
        # Additional heat from degradation
        degradation_heat = sum(self.failure_modes.values()) * 20.0
        
        # Thermal time constant simulation
        current_temp = ambient_temp + base_heat + degradation_heat
        
        # Temperature limits based on health
        max_operating_temp = self.config["max_temperature"] * (1 + (1 - self.health_state) * 0.3)
        
        return {
            "motor_temperature": current_temp,
            "bearing_temperature": current_temp + 5.0,  # Bearings typically hotter
            "ambient_temperature": ambient_temp,
            "max_operating_temperature": max_operating_temp,
            "temperature_trend": "increasing" if current_temp > ambient_temp + 20 else "stable"
        }

    def _calculate_comprehensive_health(self) -> HealthMetrics:
        """Calculate comprehensive health metrics."""
        # Component-specific health based on failure modes
        bearing_health = 1.0 - self.failure_modes[FailureMode.BEARING_WEAR]
        alignment_health = 1.0 - self.failure_modes[FailureMode.MISALIGNMENT]
        balance_health = 1.0 - self.failure_modes[FailureMode.IMBALANCE]
        lubrication_health = 1.0 - self.failure_modes[FailureMode.LUBRICATION]
        
        # Overall health as weighted combination
        overall_health = (
            bearing_health * 0.4 +  # Bearings are most critical
            alignment_health * 0.2 +
            balance_health * 0.2 +
            lubrication_health * 0.2
        )
        
        # Confidence based on operational history
        confidence = min(1.0, self.operational_hours / 1000.0)
        
        # Failure risk assessment
        failure_risk = 1.0 - overall_health
        
        # RUL prediction (simplified)
        predicted_rul = int((1.0 - overall_health) * self.config["bearing_life"])
        
        return HealthMetrics(
            overall_health=float(overall_health),
            bearing_health=float(bearing_health),
            alignment_health=float(alignment_health),
            balance_health=float(balance_health),
            lubrication_health=float(lubrication_health),
            predicted_rul=max(24, predicted_rul),  # Minimum 24 hours
            confidence=float(confidence),
            failure_risk=float(failure_risk)
        )

    def _predict_remaining_useful_life(self) -> Dict[str, Any]:
        """Predict remaining useful life with confidence intervals."""
        base_rul = self.config["bearing_life"] * self.health_state
        
        # Adjust based on operating conditions
        stress_factor = 1.0 + (1 - self.health_state) * 0.5
        adjusted_rul = base_rul / stress_factor
        
        # Confidence bounds
        confidence_interval = adjusted_rul * 0.2  # ±20%
        
        return {
            "point_estimate": int(adjusted_rul),
            "confidence_lower": int(adjusted_rul - confidence_interval),
            "confidence_upper": int(adjusted_rul + confidence_interval),
            "units": "hours",
            "confidence_level": 0.95
        }

    def _detect_anomalies(self, vibration: VibrationSignature, temperature: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies based on vibration and temperature patterns."""
        anomalies = []
        
        # Vibration anomalies
        if vibration.rms > self.config["max_vibration"] * 0.7:
            anomalies.append({
                "type": "high_vibration",
                "severity": "warning",
                "value": vibration.rms,
                "threshold": self.config["max_vibration"] * 0.7
            })
        
        if vibration.crest_factor > 5.0:
            anomalies.append({
                "type": "high_crest_factor",
                "severity": "warning", 
                "value": vibration.crest_factor,
                "threshold": 5.0
            })
        
        # Temperature anomalies
        if temperature["motor_temperature"] > self.config["max_temperature"] * 0.8:
            anomalies.append({
                "type": "high_temperature",
                "severity": "warning",
                "value": temperature["motor_temperature"],
                "threshold": self.config["max_temperature"] * 0.8
            })
        
        return {
            "anomalies_detected": len(anomalies) > 0,
            "anomaly_count": len(anomalies),
            "anomaly_list": anomalies,
            "overall_severity": "critical" if any(a["severity"] == "critical" for a in anomalies) 
                              else "warning" if anomalies else "normal"
        }

    def _update_historical_data(self, vibration: VibrationSignature, 
                              temperature: Dict[str, float], health: HealthMetrics):
        """Update historical data for trend analysis."""
        historical_record = {
            "timestamp": datetime.utcnow(),
            "vibration_rms": vibration.rms,
            "temperature": temperature["motor_temperature"],
            "health_score": health.overall_health,
            "equipment_state": self.equipment_state.value
        }
        
        self.operation_history.append(historical_record)
        self.vibration_history.append(vibration.rms)
        self.temperature_history.append(temperature["motor_temperature"])
        
        # Keep only recent history (last 1000 records)
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-500:]
            self.vibration_history = self.vibration_history[-500:]
            self.temperature_history = self.temperature_history[-500:]

    def _get_fallback_simulation(self) -> Dict[str, Any]:
        """Provide fallback simulation data in case of errors."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "equipment_state": "unknown",
            "vibration_data": {
                "waveform": [0] * 1000,
                "analysis": {
                    "rms": 0.0,
                    "peak": 0.0,
                    "crest_factor": 0.0,
                    "kurtosis": 0.0,
                    "dominant_frequencies": [],
                    "harmonic_ratios": []
                }
            },
            "temperature_data": {
                "motor_temperature": 25.0,
                "bearing_temperature": 25.0,
                "ambient_temperature": 25.0
            },
            "health_metrics": {
                "overall_health": 0.5,
                "bearing_health": 0.5,
                "alignment_health": 0.5,
                "balance_health": 0.5,
                "lubrication_health": 0.5,
                "predicted_rul": 1000,
                "confidence": 0.0,
                "failure_risk": 0.5
            },
            "rul_prediction": {
                "point_estimate": 1000,
                "confidence_lower": 800,
                "confidence_upper": 1200,
                "units": "hours",
                "confidence_level": 0.5
            },
            "anomaly_detection": {
                "anomalies_detected": False,
                "anomaly_count": 0,
                "anomaly_list": [],
                "overall_severity": "unknown"
            },
            "operating_conditions": {},
            "failure_risks": {mode.value: 0.0 for mode in FailureMode}
        }

    def reset_twin(self):
        """Reset digital twin to initial state."""
        self.health_state = 1.0
        self.equipment_state = EquipmentState.NORMAL
        self.operational_hours = 0
        self.last_maintenance = datetime.utcnow()
        self.failure_modes = self._initialize_failure_modes()
        self.operation_history.clear()
        self.vibration_history.clear()
        self.temperature_history.clear()
        
        self.logger.info("Digital Twin reset to initial state")

    def perform_maintenance(self, maintenance_type: str = "routine"):
        """Simulate maintenance action and improve health state."""
        improvement = 0.0
        
        if maintenance_type == "routine":
            improvement = 0.1
        elif maintenance_type == "overhaul":
            improvement = 0.5
        elif maintenance_type == "bearing_replacement":
            self.failure_modes[FailureMode.BEARING_WEAR] = 0.0
            improvement = 0.3
        
        # Apply improvement to all components
        for mode in self.failure_modes:
            self.failure_modes[mode] = max(0.0, self.failure_modes[mode] - improvement)
        
        self.health_state = min(1.0, self.health_state + improvement)
        self.last_maintenance = datetime.utcnow()
        
        self.logger.info(f"Maintenance performed: {maintenance_type}, Health improved to {self.health_state:.3f}")

    def get_twin_status(self) -> Dict[str, Any]:
        """Get current digital twin status summary."""
        return {
            "equipment_type": self.equipment_type,
            "current_health": self.health_state,
            "equipment_state": self.equipment_state.value,
            "operational_hours": self.operational_hours,
            "time_since_maintenance": (datetime.utcnow() - self.last_maintenance).total_seconds() / 3600,
            "active_failure_modes": {
                mode.value: severity for mode, severity in self.failure_modes.items() 
                if severity > 0.1
            },
            "history_size": len(self.operation_history)
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create digital twin for centrifugal pump
    twin = IndustrialDigitalTwin("centrifugal_pump")
    
    # Simulate normal operation
    conditions = {
        "rpm": 2950,
        "load": 0.75,
        "ambient_temperature": 25.0,
        "failure_mode": "normal"
    }
    
    result = twin.simulate_equipment_behavior(conditions)
    print("Digital Twin Simulation Result:")
    print(f"Health Score: {result['health_metrics']['overall_health']:.3f}")
    print(f"Predicted RUL: {result['rul_prediction']['point_estimate']} hours")
    print(f"Equipment State: {result['equipment_state']}")
