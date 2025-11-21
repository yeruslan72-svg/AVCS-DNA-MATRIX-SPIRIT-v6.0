# adaptive_learning/context_manager.py
"""
AVCS DNA-MATRIX SPIRIT v7.0
Advanced Context Manager
-------------------------
Multi-dimensional context inference with ML integration,
temporal patterns, and equipment-specific operational modes.
"""

from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json


class OperationalMode(Enum):
    NORMAL = "normal"
    HIGH_LOAD = "high_load"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    NIGHT_OPERATION = "night_operation"
    DEGRADED = "degraded"
    OPTIMAL = "optimal"
    TRANSITION = "transition"


class ShiftType(Enum):
    SHIFT_A = "A"  # 06:00 - 14:00
    SHIFT_B = "B"  # 14:00 - 22:00  
    SHIFT_C = "C"  # 22:00 - 06:00
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


@dataclass
class ContextFeatures:
    """Comprehensive context features for ML inference."""
    temporal_features: Dict[str, float]
    operational_features: Dict[str, float]
    environmental_features: Dict[str, float]
    equipment_features: Dict[str, float]
    historical_features: Dict[str, float]


class ContextManager:
    """
    Advanced context inference system with multi-dimensional analysis,
    temporal patterns, and adaptive thresholding.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger("ContextManager")
        
        # Operational thresholds
        self._initialize_thresholds()
        
        # Temporal patterns
        self.seasonal_patterns = self._initialize_seasonal_patterns()
        
        # Historical context tracking
        self.context_history: List[Dict] = []
        self.mode_transitions = []
        
        # ML model placeholder (can be extended with actual ML)
        self._ml_enabled = self.config.get("ml_integration", False)
        
        self.logger.info("Advanced Context Manager initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for context management."""
        return {
            "temporal_settings": {
                "night_hours": (22, 6),  # 10 PM to 6 AM
                "peak_hours": (8, 18),   # 8 AM to 6 PM
                "weekend_days": [5, 6],  # Saturday, Sunday
            },
            "operational_thresholds": {
                "high_load": 1.3,
                "critical_load": 1.8,
                "high_temperature": 75.0,
                "critical_temperature": 85.0,
                "high_vibration": 4.0,
                "critical_vibration": 6.0,
                "startup_duration": 300,  # seconds
                "shutdown_duration": 600,  # seconds
            },
            "equipment_profiles": {
                "centrifugal_pump": {
                    "optimal_rpm_range": (2800, 3100),
                    "normal_pressure_range": (2.0, 6.0),
                    "efficiency_threshold": 0.85
                },
                "compressor": {
                    "optimal_rpm_range": (1800, 2200),
                    "normal_pressure_range": (5.0, 12.0),
                    "efficiency_threshold": 0.80
                }
            },
            "ml_integration": False,
            "history_size": 1000,
            "confidence_threshold": 0.7
        }

    def _initialize_thresholds(self):
        """Initialize operational thresholds."""
        self.thresholds = self.config["operational_thresholds"]
        self.temporal_settings = self.config["temporal_settings"]
        self.equipment_profiles = self.config["equipment_profiles"]

    def _initialize_seasonal_patterns(self) -> Dict[str, Any]:
        """Initialize seasonal and temporal patterns."""
        return {
            "seasonal_adjustments": {
                "winter": {"temperature_offset": -5.0, "load_factor": 1.1},
                "summer": {"temperature_offset": +8.0, "load_factor": 0.9},
                "spring": {"temperature_offset": +2.0, "load_factor": 1.0},
                "autumn": {"temperature_offset": -2.0, "load_factor": 1.0}
            },
            "daily_patterns": {
                "morning_peak": {"hours": (6, 9), "load_factor": 1.2},
                "evening_peak": {"hours": (17, 20), "load_factor": 1.1},
                "night_valley": {"hours": (1, 5), "load_factor": 0.7}
            }
        }

    def infer_context(self, telemetry: Dict[str, Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Advanced multi-dimensional context inference.
        
        Args:
            telemetry: Comprehensive sensor data
            metadata: Operational metadata including equipment type, operator info
            
        Returns:
            Detailed context analysis with confidence scores
        """
        inference_start = datetime.utcnow()
        
        try:
            # Extract and validate input data
            validated_telemetry = self._validate_telemetry(telemetry)
            validated_metadata = metadata or {}
            
            # Multi-dimensional feature extraction
            features = self._extract_features(validated_telemetry, validated_metadata)
            
            # Temporal context analysis
            temporal_context = self._analyze_temporal_context()
            
            # Operational context analysis
            operational_context = self._analyze_operational_context(validated_telemetry, features)
            
            # Equipment-specific context
            equipment_context = self._analyze_equipment_context(validated_telemetry, validated_metadata)
            
            # Historical context analysis
            historical_context = self._analyze_historical_context(validated_telemetry)
            
            # ML-enhanced context inference (if enabled)
            ml_context = self._ml_context_inference(features) if self._ml_enabled else {}
            
            # Composite context determination
            primary_mode, confidence = self._determine_primary_mode(
                temporal_context, operational_context, equipment_context, historical_context, ml_context
            )
            
            # Risk assessment
            risk_assessment = self._assess_operational_risk(primary_mode, validated_telemetry)
            
            # Generate comprehensive context
            context = self._compile_context(
                primary_mode, confidence, temporal_context, operational_context,
                equipment_context, historical_context, risk_assessment, validated_metadata
            )
            
            # Update context history
            self._update_context_history(context, inference_start)
            
            inference_time = (datetime.utcnow() - inference_start).total_seconds()
            context["inference_metrics"]["processing_time"] = inference_time
            
            self.logger.debug(f"Context inference completed: {primary_mode.value} (confidence: {confidence:.2f})")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Context inference failed: {e}")
            return self._get_fallback_context(telemetry, metadata)

    def _validate_telemetry(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize telemetry data."""
        required_fields = ["vibration", "temperature", "pressure", "rpm", "load"]
        validated = {}
        
        for field in required_fields:
            value = telemetry.get(field)
            if value is None:
                # Provide reasonable defaults for missing critical fields
                defaults = {
                    "vibration": 0.1,
                    "temperature": 25.0,
                    "pressure": 1.0,
                    "rpm": 0.0,
                    "load": 0.0
                }
                value = defaults[field]
                self.logger.warning(f"Missing telemetry field: {field}, using default: {value}")
            
            try:
                validated[field] = float(value)
            except (ValueError, TypeError):
                validated[field] = defaults.get(field, 0.0)
                self.logger.warning(f"Invalid telemetry value for {field}: {value}, using default: {validated[field]}")
        
        # Add optional fields if present
        for field in ["efficiency", "health_score", "power_consumption"]:
            if field in telemetry and telemetry[field] is not None:
                try:
                    validated[field] = float(telemetry[field])
                except (ValueError, TypeError):
                    pass
        
        return validated

    def _extract_features(self, telemetry: Dict[str, Any], metadata: Dict[str, Any]) -> ContextFeatures:
        """Extract comprehensive features for context analysis."""
        # Temporal features
        temporal_features = {
            "hour_of_day": datetime.utcnow().hour,
            "day_of_week": datetime.utcnow().weekday(),
            "is_weekend": datetime.utcnow().weekday() in self.temporal_settings["weekend_days"],
            "month": datetime.utcnow().month,
            "season": self._get_current_season()
        }
        
        # Operational features
        operational_features = {
            "load_ratio": telemetry["load"],
            "vibration_level": telemetry["vibration"],
            "temperature_level": telemetry["temperature"],
            "pressure_level": telemetry["pressure"],
            "rpm_ratio": telemetry["rpm"] / 3000.0,  # Normalized
            "efficiency": telemetry.get("efficiency", 1.0)
        }
        
        # Environmental features
        environmental_features = {
            "ambient_temperature": telemetry.get("ambient_temperature", 25.0),
            "humidity": telemetry.get("humidity", 50.0),
            "operational_environment": metadata.get("environment", "indoor")
        }
        
        # Equipment features
        equipment_features = {
            "equipment_type": metadata.get("equipment_type", "unknown"),
            "age_factor": metadata.get("age_factor", 1.0),
            "maintenance_status": metadata.get("maintenance_status", "normal")
        }
        
        # Historical features
        historical_features = self._compute_historical_features(telemetry)
        
        return ContextFeatures(
            temporal_features=temporal_features,
            operational_features=operational_features,
            environmental_features=environmental_features,
            equipment_features=equipment_features,
            historical_features=historical_features
        )

    def _get_current_season(self) -> str:
        """Determine current season based on month."""
        month = datetime.utcnow().month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    def _compute_historical_features(self, telemetry: Dict[str, Any]) -> Dict[str, float]:
        """Compute features based on historical context."""
        if not self.context_history:
            return {"trend_vibration": 0.0, "trend_temperature": 0.0, "mode_consistency": 1.0}
        
        # Use recent history for trend analysis
        recent_contexts = self.context_history[-10:]
        
        # Vibration trend
        vibration_trend = self._compute_trend([c["telemetry"]["vibration"] for c in recent_contexts])
        
        # Temperature trend
        temperature_trend = self._compute_trend([c["telemetry"]["temperature"] for c in recent_contexts])
        
        # Mode consistency
        recent_modes = [c["primary_mode"] for c in recent_contexts]
        mode_consistency = len(set(recent_modes)) / len(recent_modes) if recent_modes else 1.0
        
        return {
            "trend_vibration": vibration_trend,
            "trend_temperature": temperature_trend,
            "mode_consistency": mode_consistency
        }

    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend of values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)

    def _analyze_temporal_context(self) -> Dict[str, Any]:
        """Analyze temporal patterns and shifts."""
        now = datetime.utcnow()
        current_hour = now.hour
        current_weekday = now.weekday()
        
        # Time of day analysis
        is_night = (self.temporal_settings["night_hours"][0] <= current_hour <= 23 or 
                  0 <= current_hour <= self.temporal_settings["night_hours"][1])
        is_peak_hours = (self.temporal_settings["peak_hours"][0] <= current_hour <= 
                        self.temporal_settings["peak_hours"][1])
        
        # Shift determination
        shift = self._determine_shift(current_hour, current_weekday)
        
        # Seasonal adjustments
        season = self._get_current_season()
        seasonal_adjustment = self.seasonal_patterns["seasonal_adjustments"].get(season, {})
        
        # Daily pattern matching
        daily_pattern = "normal"
        for pattern_name, pattern_info in self.seasonal_patterns["daily_patterns"].items():
            start_hour, end_hour = pattern_info["hours"]
            if start_hour <= current_hour <= end_hour:
                daily_pattern = pattern_name
                break
        
        return {
            "current_hour": current_hour,
            "current_weekday": current_weekday,
            "is_night": is_night,
            "is_peak_hours": is_peak_hours,
            "shift": shift.value,
            "season": season,
            "seasonal_adjustment": seasonal_adjustment,
            "daily_pattern": daily_pattern
        }

    def _determine_shift(self, hour: int, weekday: int) -> ShiftType:
        """Determine current operational shift."""
        if weekday >= 5:  # Weekend
            return ShiftType.WEEKEND
        
        if 6 <= hour < 14:
            return ShiftType.SHIFT_A
        elif 14 <= hour < 22:
            return ShiftType.SHIFT_B
        else:
            return ShiftType.SHIFT_C

    def _analyze_operational_context(self, telemetry: Dict[str, Any], features: ContextFeatures) -> Dict[str, Any]:
        """Analyze operational state and conditions."""
        load = telemetry["load"]
        vibration = telemetry["vibration"]
        temperature = telemetry["temperature"]
        pressure = telemetry["pressure"]
        rpm = telemetry["rpm"]
        
        # Load analysis
        load_state = "normal"
        if load > self.thresholds["critical_load"]:
            load_state = "critical"
        elif load > self.thresholds["high_load"]:
            load_state = "high"
        
        # Vibration analysis
        vibration_state = "normal"
        if vibration > self.thresholds["critical_vibration"]:
            vibration_state = "critical"
        elif vibration > self.thresholds["high_vibration"]:
            vibration_state = "high"
        
        # Temperature analysis
        temperature_state = "normal"
        if temperature > self.thresholds["critical_temperature"]:
            temperature_state = "critical"
        elif temperature > self.thresholds["high_temperature"]:
            temperature_state = "high"
        
        # RPM analysis
        rpm_state = "optimal" if 2900 <= rpm <= 3100 else "suboptimal"
        
        # Efficiency analysis
        efficiency = telemetry.get("efficiency", 1.0)
        efficiency_state = "optimal" if efficiency > 0.85 else "degraded"
        
        return {
            "load_state": load_state,
            "vibration_state": vibration_state,
            "temperature_state": temperature_state,
            "rpm_state": rpm_state,
            "efficiency_state": efficiency_state,
            "overall_operational_state": self._compute_overall_operational_state(
                load_state, vibration_state, temperature_state, efficiency_state
            )
        }

    def _compute_overall_operational_state(self, load_state: str, vibration_state: str, 
                                         temperature_state: str, efficiency_state: str) -> str:
        """Compute overall operational state from component states."""
        critical_states = ["critical"]
        warning_states = ["high"]
        
        if any(state in critical_states for state in [load_state, vibration_state, temperature_state]):
            return "critical"
        elif any(state in warning_states for state in [load_state, vibration_state, temperature_state]):
            return "degraded"
        elif efficiency_state == "degraded":
            return "suboptimal"
        else:
            return "optimal"

    def _analyze_equipment_context(self, telemetry: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze equipment-specific context and performance."""
        equipment_type = metadata.get("equipment_type", "centrifugal_pump")
        equipment_profile = self.equipment_profiles.get(equipment_type, {})
        
        rpm = telemetry["rpm"]
        pressure = telemetry["pressure"]
        efficiency = telemetry.get("efficiency", 1.0)
        
        # RPM compliance
        optimal_range = equipment_profile.get("optimal_rpm_range", (2800, 3200))
        rpm_compliant = optimal_range[0] <= rpm <= optimal_range[1]
        
        # Pressure compliance
        pressure_range = equipment_profile.get("normal_pressure_range", (1.0, 8.0))
        pressure_compliant = pressure_range[0] <= pressure <= pressure_range[1]
        
        # Efficiency compliance
        efficiency_threshold = equipment_profile.get("efficiency_threshold", 0.8)
        efficiency_compliant = efficiency >= efficiency_threshold
        
        return {
            "equipment_type": equipment_type,
            "rpm_compliant": rpm_compliant,
            "pressure_compliant": pressure_compliant,
            "efficiency_compliant": efficiency_compliant,
            "overall_compliance": rpm_compliant and pressure_compliant and efficiency_compliant,
            "profile_match": self._compute_profile_match(telemetry, equipment_profile)
        }

    def _compute_profile_match(self, telemetry: Dict[str, Any], profile: Dict[str, Any]) -> float:
        """Compute how well current operation matches equipment profile."""
        match_score = 0.0
        factors = 0
        
        # RPM match
        if "optimal_rpm_range" in profile:
            rpm_range = profile["optimal_rpm_range"]
            rpm = telemetry["rpm"]
            if rpm_range[0] <= rpm <= rpm_range[1]:
                match_score += 1.0
            factors += 1
        
        # Pressure match
        if "normal_pressure_range" in profile:
            pressure_range = profile["normal_pressure_range"]
            pressure = telemetry["pressure"]
            if pressure_range[0] <= pressure <= pressure_range[1]:
                match_score += 1.0
            factors += 1
        
        # Efficiency match
        if "efficiency_threshold" in profile:
            efficiency_threshold = profile["efficiency_threshold"]
            efficiency = telemetry.get("efficiency", 1.0)
            if efficiency >= efficiency_threshold:
                match_score += 1.0
            factors += 1
        
        return match_score / factors if factors > 0 else 1.0

    def _analyze_historical_context(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze historical patterns and trends."""
        if not self.context_history:
            return {"trend": "unknown", "stability": "unknown", "anomaly_score": 0.0}
        
        recent_telemetry = [ctx["telemetry"] for ctx in self.context_history[-5:]]
        
        # Vibration trend
        vibration_trend = self._compute_trend([t["vibration"] for t in recent_telemetry])
        
        # Temperature trend
        temperature_trend = self._compute_trend([t["temperature"] for t in recent_telemetry])
        
        # Mode stability
        recent_modes = [ctx["primary_mode"] for ctx in self.context_history[-10:]]
        mode_stability = len(set(recent_modes)) == 1
        
        # Anomaly detection (simplified)
        current_vibration = telemetry["vibration"]
        historical_vibrations = [t["vibration"] for t in recent_telemetry]
        avg_vibration = np.mean(historical_vibrations)
        std_vibration = np.std(historical_vibrations)
        anomaly_score = abs(current_vibration - avg_vibration) / (std_vibration + 1e-6)
        
        return {
            "vibration_trend": vibration_trend,
            "temperature_trend": temperature_trend,
            "mode_stability": mode_stability,
            "anomaly_score": float(anomaly_score),
            "trend_direction": "increasing" if vibration_trend > 0.1 else "decreasing" if vibration_trend < -0.1 else "stable"
        }

    def _ml_context_inference(self, features: ContextFeatures) -> Dict[str, Any]:
        """ML-enhanced context inference (placeholder for actual ML integration)."""
        # This is a placeholder for actual ML model inference
        # In practice, this would use a trained model for context classification
        
        return {
            "ml_mode_suggestion": "normal",
            "ml_confidence": 0.85,
            "feature_importance": {
                "vibration_level": 0.3,
                "load_ratio": 0.25,
                "temperature_level": 0.2,
                "temporal_features": 0.15,
                "historical_trends": 0.1
            }
        }

    def _determine_primary_mode(self, temporal_context: Dict[str, Any], operational_context: Dict[str, Any],
                              equipment_context: Dict[str, Any], historical_context: Dict[str, Any],
                              ml_context: Dict[str, Any]) -> tuple:
        """Determine primary operational mode with confidence."""
        mode_scores = {}
        
        # Emergency conditions take highest priority
        if operational_context["overall_operational_state"] == "critical":
            return OperationalMode.EMERGENCY, 0.95
        
        # High load conditions
        if operational_context["load_state"] == "high" and temporal_context["is_night"]:
            mode_scores[OperationalMode.NIGHT_OPERATION] = 0.8
        elif operational_context["load_state"] == "high":
            mode_scores[OperationalMode.HIGH_LOAD] = 0.85
        
        # Degraded performance
        if operational_context["overall_operational_state"] == "degraded":
            mode_scores[OperationalMode.DEGRADED] = 0.75
        
        # Optimal conditions
        if (operational_context["overall_operational_state"] == "optimal" and 
            equipment_context["overall_compliance"]):
            mode_scores[OperationalMode.OPTIMAL] = 0.9
        
        # Normal operation (fallback)
        mode_scores[OperationalMode.NORMAL] = 0.7
        
        # Apply ML suggestions if available
        if ml_context and "ml_mode_suggestion" in ml_context:
            ml_mode = OperationalMode(ml_context["ml_mode_suggestion"])
            ml_confidence = ml_context.get("ml_confidence", 0.5)
            if ml_mode in mode_scores:
                mode_scores[ml_mode] = max(mode_scores[ml_mode], ml_confidence)
        
        # Select mode with highest score
        if mode_scores:
            primary_mode = max(mode_scores, key=mode_scores.get)
            confidence = mode_scores[primary_mode]
        else:
            primary_mode = OperationalMode.NORMAL
            confidence = 0.5
        
        return primary_mode, confidence

    def _assess_operational_risk(self, mode: OperationalMode, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Assess operational risk based on mode and telemetry."""
        base_risk = {
            OperationalMode.EMERGENCY: 90,
            OperationalMode.DEGRADED: 70,
            OperationalMode.HIGH_LOAD: 60,
            OperationalMode.NIGHT_OPERATION: 55,
            OperationalMode.TRANSITION: 45,
            OperationalMode.NORMAL: 20,
            OperationalMode.OPTIMAL: 10
        }.get(mode, 30)
        
        # Adjust based on telemetry
        vibration_penalty = max(0, (telemetry["vibration"] - 3.0) * 10)
        temperature_penalty = max(0, (telemetry["temperature"] - 70.0) * 2)
        
        total_risk = min(100, base_risk + vibration_penalty + temperature_penalty)
        
        return {
            "risk_level": total_risk,
            "risk_category": "high" if total_risk > 70 else "medium" if total_risk > 40 else "low",
            "contributing_factors": [
                f"vibration_{telemetry['vibration']:.1f}",
                f"temperature_{telemetry['temperature']:.1f}",
                f"mode_{mode.value}"
            ]
        }

    def _compile_context(self, primary_mode: OperationalMode, confidence: float,
                        temporal_context: Dict[str, Any], operational_context: Dict[str, Any],
                        equipment_context: Dict[str, Any], historical_context: Dict[str, Any],
                        risk_assessment: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive context analysis."""
        return {
            "primary_mode": primary_mode.value,
            "confidence": confidence,
            "temporal_context": temporal_context,
            "operational_context": operational_context,
            "equipment_context": equipment_context,
            "historical_context": historical_context,
            "risk_assessment": risk_assessment,
            "metadata": metadata,
            "inference_metrics": {
                "timestamp": datetime.utcnow().isoformat(),
                "features_used": len([f for f in asdict(ContextFeatures).values() if f]),
                "analysis_depth": "comprehensive"
            },
            "recommendations": self._generate_recommendations(primary_mode, risk_assessment)
        }

    def _generate_recommendations(self, mode: OperationalMode, risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate operational recommendations based on context."""
        recommendations = []
        
        if mode == OperationalMode.EMERGENCY:
            recommendations.extend([
                "Initiate emergency shutdown procedures",
                "Alert maintenance team immediately",
                "Increase monitoring frequency to maximum"
            ])
        elif mode == OperationalMode.DEGRADED:
            recommendations.extend([
                "Schedule preventive maintenance",
                "Increase damping forces",
                "Monitor component health closely"
            ])
        elif mode == OperationalMode.HIGH_LOAD:
            recommendations.extend([
                "Consider load distribution",
                "Monitor temperature trends",
                "Prepare for potential mode transition"
            ])
        elif risk_assessment["risk_category"] == "high":
            recommendations.append("Review operational parameters and safety margins")
        
        return recommendations

    def _update_context_history(self, context: Dict[str, Any], inference_start: datetime):
        """Update context history with new inference."""
        history_entry = {
            "timestamp": inference_start.isoformat(),
            "primary_mode": context["primary_mode"],
            "confidence": context["confidence"],
            "telemetry": {k: v for k, v in context.items() if k in ["vibration", "temperature", "pressure", "rpm", "load"]},
            "risk_level": context["risk_assessment"]["risk_level"]
        }
        
        self.context_history.append(history_entry)
        
        # Maintain history size
        if len(self.context_history) > self.config["history_size"]:
            self.context_history = self.context_history[-self.config["history_size"]:]

    def _get_fallback_context(self, telemetry: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback context in case of inference failure."""
        return {
            "primary_mode": "normal",
            "confidence": 0.5,
            "temporal_context": {"current_hour": datetime.utcnow().hour},
            "operational_context": {"overall_operational_state": "unknown"},
            "equipment_context": {"equipment_type": "unknown"},
            "historical_context": {"trend": "unknown"},
            "risk_assessment": {"risk_level": 50, "risk_category": "medium"},
            "metadata": metadata,
            "inference_metrics": {
                "timestamp": datetime.utcnow().isoformat(),
                "features_used": 0,
                "analysis_depth": "fallback",
                "error": "Inference failure"
            },
            "recommendations": ["Verify system operation", "Check sensor data quality"]
        }

    def get_context_history(self, last_n: int = None) -> List[Dict]:
        """Get context history with optional limit."""
        if last_n:
            return self.context_history[-last_n:]
        return self.context_history

    def get_mode_statistics(self) -> Dict[str, Any]:
        """Get statistics about operational modes."""
        if not self.context_history:
            return {"total_inferences": 0, "mode_distribution": {}}
        
        mode_counts = {}
        for context in self.context_history:
            mode = context["primary_mode"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        total = len(self.context_history)
        mode_distribution = {mode: count/total for mode, count in mode_counts.items()}
        
        return {
            "total_inferences": total,
            "mode_distribution": mode_distribution,
            "most_common_mode": max(mode_counts, key=mode_counts.get) if mode_counts else "unknown",
            "confidence_stats": {
                "avg_confidence": np.mean([c["confidence"] for c in self.context_history]),
                "min_confidence": np.min([c["confidence"] for c in self.context_history]),
                "max_confidence": np.max([c["confidence"] for c in self.context_history])
            }
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create context manager
    context_mgr = ContextManager()
    
    # Test telemetry data
    test_telemetry = {
        "vibration": 3.2,
        "temperature": 68.0,
        "pressure": 4.5,
        "rpm": 2950.0,
        "load": 1.4,
        "efficiency": 0.88
    }
    
    test_metadata = {
        "operator": "AI-System",
        "equipment_type": "centrifugal_pump",
        "environment": "indoor"
    }
    
    # Perform context inference
    context = context_mgr.infer_context(test_telemetry, test_metadata)
    
    print("Context Inference Result:")
    print(f"Primary Mode: {context['primary_mode']}")
    print(f"Confidence: {context['confidence']:.2f}")
    print(f"Risk Level: {context['risk_assessment']['risk_level']}%")
    print(f"Recommendations: {context['recommendations']}")
    
    # Get statistics
    stats = context_mgr.get_mode_statistics()
    print(f"\nMode Statistics: {stats}")
