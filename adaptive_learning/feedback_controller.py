# adaptive_learning/feedback_controller.py
"""
AVCS DNA-MATRIX SPIRIT v7.0
Advanced Feedback Controller
----------------------------
Intelligent actuator control with multi-objective optimization,
safety constraints, and predictive control strategies.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import logging
from datetime import datetime, timedelta


class ControlStrategy(Enum):
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"
    SAFETY = "safety"
    PREDICTIVE = "predictive"
    OPTIMAL = "optimal"


class ActuatorType(Enum):
    MR_DAMPER = "mr_damper"
    HYDRAULIC_ACTUATOR = "hydraulic_actuator"
    PNEUMATIC_DAMPER = "pneumatic_damper"
    ACTIVE_MOUNT = "active_mount"


@dataclass
class ControlAction:
    """Comprehensive control action specification."""
    damper_force: float
    control_strategy: str
    confidence: float
    safety_limits: Dict[str, float]
    timing: Dict[str, float]
    recommendations: List[str]
    optimization_metrics: Dict[str, float]


@dataclass
class SystemConstraints:
    """Physical and operational constraints."""
    max_force: float
    min_force: float
    max_rate_of_change: float
    temperature_limits: Dict[str, float]
    vibration_limits: Dict[str, float]
    power_limits: Dict[str, float]


class FeedbackController:
    """
    Advanced feedback controller with multi-objective optimization,
    predictive control, and safety-aware actuation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger("FeedbackController")
        
        # System constraints and limits
        self.constraints = self._initialize_constraints()
        
        # Control strategies
        self.control_strategies = self._initialize_control_strategies()
        
        # Actuator models
        self.actuator_models = self._initialize_actuator_models()
        
        # Historical control actions
        self.control_history: List[Dict] = []
        self.performance_metrics = self._initialize_metrics()
        
        # Adaptive tuning parameters
        self.adaptive_gains = self._initialize_adaptive_gains()
        self.last_action = None
        
        self.logger.info("Advanced Feedback Controller initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default controller configuration."""
        return {
            "actuator_type": "mr_damper",
            "control_strategies": {
                "conservative": {"aggression": 0.5, "safety_margin": 0.8},
                "adaptive": {"aggression": 0.7, "safety_margin": 0.6},
                "aggressive": {"aggression": 0.9, "safety_margin": 0.4},
                "safety": {"aggression": 0.3, "safety_margin": 0.9},
                "predictive": {"aggression": 0.8, "safety_margin": 0.7},
                "optimal": {"aggression": 0.75, "safety_margin": 0.65}
            },
            "response_curves": {
                "linear": lambda x: x,
                "exponential": lambda x: x ** 1.5,
                "logarithmic": lambda x: np.log1p(x),
                "sigmoid": lambda x: 1 / (1 + np.exp(-10 * (x - 0.5)))
            },
            "optimization_weights": {
                "vibration_reduction": 0.4,
                "energy_efficiency": 0.2,
                "equipment_life": 0.2,
                "safety_margin": 0.2
            },
            "timing_constraints": {
                "min_update_interval": 0.1,  # seconds
                "max_response_time": 2.0,   # seconds
                "prediction_horizon": 5.0   # seconds
            }
        }

    def _initialize_constraints(self) -> SystemConstraints:
        """Initialize system physical constraints."""
        return SystemConstraints(
            max_force=10000.0,  # Newtons
            min_force=0.0,
            max_rate_of_change=5000.0,  # N/s
            temperature_limits={
                "max_operating": 85.0,
                "warning_threshold": 70.0,
                "shutdown_threshold": 95.0
            },
            vibration_limits={
                "normal": 3.0,
                "warning": 5.0,
                "critical": 8.0
            },
            power_limits={
                "max_power": 5000.0,  # Watts
                "normal_operation": 2000.0
            }
        )

    def _initialize_control_strategies(self) -> Dict[str, Dict[str, float]]:
        """Initialize control strategy parameters."""
        return {
            "conservative": {
                "risk_threshold": 70,
                "force_multiplier": 0.6,
                "response_curve": "linear",
                "safety_factor": 0.8
            },
            "adaptive": {
                "risk_threshold": 50,
                "force_multiplier": 0.8,
                "response_curve": "exponential",
                "safety_factor": 0.6
            },
            "aggressive": {
                "risk_threshold": 30,
                "force_multiplier": 1.0,
                "response_curve": "exponential",
                "safety_factor": 0.4
            },
            "safety": {
                "risk_threshold": 85,
                "force_multiplier": 1.0,
                "response_curve": "linear",
                "safety_factor": 0.9
            },
            "predictive": {
                "risk_threshold": 40,
                "force_multiplier": 0.9,
                "response_curve": "sigmoid",
                "safety_factor": 0.7
            },
            "optimal": {
                "risk_threshold": 20,
                "force_multiplier": 0.75,
                "response_curve": "logarithmic",
                "safety_factor": 0.65
            }
        }

    def _initialize_actuator_models(self) -> Dict[str, Any]:
        """Initialize actuator performance models."""
        return {
            "mr_damper": {
                "response_time": 0.02,  # seconds
                "force_accuracy": 0.95,
                "power_consumption": lambda force: 50 + force * 0.1,
                "efficiency_curve": lambda force: 0.9 - (force / 10000) * 0.2
            },
            "hydraulic_actuator": {
                "response_time": 0.05,
                "force_accuracy": 0.92,
                "power_consumption": lambda force: 100 + force * 0.15,
                "efficiency_curve": lambda force: 0.85 - (force / 10000) * 0.25
            }
        }

    def _initialize_adaptive_gains(self) -> Dict[str, float]:
        """Initialize adaptive control gains."""
        return {
            "proportional_gain": 1.2,
            "integral_gain": 0.1,
            "derivative_gain": 0.05,
            "learning_rate": 0.01
        }

    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize performance tracking metrics."""
        return {
            "total_actions": 0,
            "vibration_reduction": 0.0,
            "energy_consumption": 0.0,
            "safety_violations": 0,
            "response_times": [],
            "control_accuracy": 1.0
        }

    def propose_actions(self, risk_index: float, context: Dict[str, Any], 
                       telemetry: Dict[str, Any] = None) -> ControlAction:
        """
        Generate intelligent control actions with multi-objective optimization.
        
        Args:
            risk_index: 0-100 risk assessment
            context: Comprehensive context from ContextManager
            telemetry: Current sensor readings (optional)
            
        Returns:
            ControlAction: Detailed control specification
        """
        optimization_start = datetime.utcnow()
        
        try:
            # Validate inputs
            validated_risk = self._validate_risk_index(risk_index)
            validated_context = self._validate_context(context)
            validated_telemetry = telemetry or {}
            
            # Determine optimal control strategy
            control_strategy = self._select_control_strategy(validated_risk, validated_context)
            
            # Calculate base force
            base_force = self._calculate_base_force(validated_risk, control_strategy)
            
            # Apply context-based adjustments
            adjusted_force = self._apply_context_adjustments(base_force, validated_context, validated_telemetry)
            
            # Apply safety constraints
            constrained_force = self._apply_safety_constraints(adjusted_force, validated_telemetry)
            
            # Apply rate limiting
            final_force = self._apply_rate_limiting(constrained_force)
            
            # Calculate control confidence
            confidence = self._calculate_control_confidence(validated_risk, control_strategy, validated_context)
            
            # Generate timing parameters
            timing = self._calculate_timing_parameters(control_strategy)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(validated_risk, control_strategy, validated_context)
            
            # Calculate optimization metrics
            optimization_metrics = self._calculate_optimization_metrics(
                final_force, validated_risk, validated_telemetry
            )
            
            # Create comprehensive control action
            control_action = ControlAction(
                damper_force=final_force,
                control_strategy=control_strategy.value,
                confidence=confidence,
                safety_limits=self._get_safety_limits(),
                timing=timing,
                recommendations=recommendations,
                optimization_metrics=optimization_metrics
            )
            
            # Update performance metrics
            self._update_performance_metrics(control_action, optimization_start)
            
            # Store action history
            self._store_control_action(control_action, validated_risk, validated_context)
            
            processing_time = (datetime.utcnow() - optimization_start).total_seconds()
            self.logger.debug(f"Control action generated in {processing_time:.3f}s: {final_force:.0f}N")
            
            return control_action
            
        except Exception as e:
            self.logger.error(f"Control action generation failed: {e}")
            return self._get_fallback_action(risk_index)

    def _validate_risk_index(self, risk_index: float) -> float:
        """Validate and normalize risk index."""
        if not isinstance(risk_index, (int, float)):
            raise ValueError("Risk index must be numeric")
        
        normalized_risk = max(0.0, min(100.0, float(risk_index)))
        
        if normalized_risk != risk_index:
            self.logger.warning(f"Risk index clamped from {risk_index} to {normalized_risk}")
        
        return normalized_risk

    def _validate_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate context structure and content."""
        required_fields = ["primary_mode", "confidence", "risk_assessment"]
        
        for field in required_fields:
            if field not in context:
                raise ValueError(f"Missing required context field: {field}")
        
        return context

    def _select_control_strategy(self, risk_index: float, context: Dict[str, Any]) -> ControlStrategy:
        """Select optimal control strategy based on risk and context."""
        primary_mode = context.get("primary_mode", "normal")
        risk_category = context["risk_assessment"].get("risk_category", "medium")
        
        # Emergency conditions
        if risk_index >= 85 or primary_mode == "emergency":
            return ControlStrategy.SAFETY
        
        # High risk conditions
        if risk_index >= 70 or risk_category == "high":
            return ControlStrategy.CONSERVATIVE
        
        # Predictive control opportunities
        if (context.get("historical_context", {}).get("trend_direction") == "increasing" and
            risk_index >= 40):
            return ControlStrategy.PREDICTIVE
        
        # Adaptive control for normal conditions
        if risk_index >= 30:
            return ControlStrategy.ADAPTIVE
        
        # Optimal control for low risk
        if risk_index < 20 and primary_mode == "optimal":
            return ControlStrategy.OPTIMAL
        
        # Default to adaptive strategy
        return ControlStrategy.ADAPTIVE

    def _calculate_base_force(self, risk_index: float, strategy: ControlStrategy) -> float:
        """Calculate base damper force using strategy-specific response curves."""
        strategy_params = self.control_strategies[strategy.value]
        
        # Normalize risk to 0-1
        normalized_risk = risk_index / 100.0
        
        # Get response curve function
        curve_name = strategy_params["response_curve"]
        curve_function = self.config["response_curves"][curve_name]
        
        # Calculate base force using response curve
        base_force_ratio = curve_function(normalized_risk)
        
        # Apply strategy multiplier
        force_multiplier = strategy_params["force_multiplier"]
        base_force = base_force_ratio * force_multiplier * self.constraints.max_force
        
        return float(base_force)

    def _apply_context_adjustments(self, base_force: float, context: Dict[str, Any], 
                                 telemetry: Dict[str, Any]) -> float:
        """Apply context-based adjustments to base force."""
        adjusted_force = base_force
        adjustment_factors = []
        
        # Temporal adjustments
        temporal_context = context.get("temporal_context", {})
        if temporal_context.get("is_night"):
            # Reduced damping at night for noise considerations
            adjusted_force *= 0.9
            adjustment_factors.append("night_operation")
        
        if temporal_context.get("is_peak_hours"):
            # Increased damping during peak hours
            adjusted_force *= 1.1
            adjustment_factors.append("peak_hours")
        
        # Operational mode adjustments
        primary_mode = context.get("primary_mode")
        if primary_mode == "high_load":
            adjusted_force *= 1.15
            adjustment_factors.append("high_load")
        elif primary_mode == "degraded":
            adjusted_force *= 1.2
            adjustment_factors.append("degraded_operation")
        
        # Equipment-specific adjustments
        equipment_context = context.get("equipment_context", {})
        if not equipment_context.get("overall_compliance", True):
            adjusted_force *= 1.1
            adjustment_factors.append("equipment_non_compliance")
        
        # Historical trend adjustments
        historical_context = context.get("historical_context", {})
        if historical_context.get("trend_direction") == "increasing":
            # Proactive increase for deteriorating conditions
            trend_magnitude = abs(historical_context.get("vibration_trend", 0))
            trend_adjustment = 1.0 + min(0.3, trend_magnitude * 10)
            adjusted_force *= trend_adjustment
            adjustment_factors.append(f"trend_adjustment_{trend_adjustment:.2f}")
        
        if adjustment_factors:
            self.logger.debug(f"Applied adjustments: {adjustment_factors}")
        
        return adjusted_force

    def _apply_safety_constraints(self, force: float, telemetry: Dict[str, Any]) -> float:
        """Apply safety constraints and limits."""
        constrained_force = force
        
        # Absolute force limits
        constrained_force = max(self.constraints.min_force, 
                               min(self.constraints.max_force, constrained_force))
        
        # Temperature-based constraints
        current_temp = telemetry.get("temperature", 25.0)
        if current_temp > self.constraints.temperature_limits["warning_threshold"]:
            # Reduce force at high temperatures to prevent overheating
            temp_factor = 1.0 - ((current_temp - 70.0) / 30.0) * 0.5
            constrained_force *= max(0.5, temp_factor)
        
        # Vibration-based constraints
        current_vibration = telemetry.get("vibration", 0.0)
        if current_vibration > self.constraints.vibration_limits["critical"]:
            # Maximum force for critical vibration
            constrained_force = self.constraints.max_force
        
        # Power constraints
        actuator_model = self.actuator_models.get(self.config["actuator_type"], {})
        power_consumption = actuator_model.get("power_consumption", lambda x: x)(constrained_force)
        if power_consumption > self.constraints.power_limits["max_power"]:
            # Scale back to stay within power limits
            power_ratio = self.constraints.power_limits["max_power"] / power_consumption
            constrained_force *= power_ratio
        
        return constrained_force

    def _apply_rate_limiting(self, force: float) -> float:
        """Apply rate limiting to prevent abrupt changes."""
        if self.last_action is None:
            # First action, no rate limiting
            self.last_action = force
            return force
        
        max_change = self.constraints.max_rate_of_change * self.config["timing_constraints"]["min_update_interval"]
        allowed_force = self.last_action
        
        if force > self.last_action + max_change:
            allowed_force = self.last_action + max_change
        elif force < self.last_action - max_change:
            allowed_force = self.last_action - max_change
        else:
            allowed_force = force
        
        self.last_action = allowed_force
        return allowed_force

    def _calculate_control_confidence(self, risk_index: float, strategy: ControlStrategy, 
                                   context: Dict[str, Any]) -> float:
        """Calculate confidence in the proposed control action."""
        base_confidence = 0.8
        
        # Adjust based on risk level
        risk_confidence = 1.0 - (risk_index / 100.0) * 0.3
        
        # Adjust based on context confidence
        context_confidence = context.get("confidence", 0.5)
        
        # Strategy-specific confidence
        strategy_confidence = {
            ControlStrategy.SAFETY: 0.95,
            ControlStrategy.CONSERVATIVE: 0.85,
            ControlStrategy.ADAPTIVE: 0.75,
            ControlStrategy.PREDICTIVE: 0.70,
            ControlStrategy.AGGRESSIVE: 0.65,
            ControlStrategy.OPTIMAL: 0.80
        }.get(strategy, 0.7)
        
        # Historical consistency
        historical_consistency = context.get("historical_context", {}).get("mode_consistency", 1.0)
        
        # Composite confidence
        confidence = (base_confidence * 0.2 + 
                     risk_confidence * 0.3 + 
                     context_confidence * 0.2 + 
                     strategy_confidence * 0.2 +
                     historical_consistency * 0.1)
        
        return max(0.1, min(1.0, confidence))

    def _calculate_timing_parameters(self, strategy: ControlStrategy) -> Dict[str, float]:
        """Calculate timing parameters for control action."""
        actuator_model = self.actuator_models.get(self.config["actuator_type"], {})
        
        return {
            "response_time": actuator_model.get("response_time", 0.02),
            "update_interval": self.config["timing_constraints"]["min_update_interval"],
            "prediction_horizon": self.config["timing_constraints"]["prediction_horizon"],
            "expected_settling_time": self._estimate_settling_time(strategy)
        }

    def _estimate_settling_time(self, strategy: ControlStrategy) -> float:
        """Estimate system settling time based on control strategy."""
        base_settling = 2.0  # seconds
        strategy_factors = {
            ControlStrategy.SAFETY: 1.5,
            ControlStrategy.CONSERVATIVE: 1.2,
            ControlStrategy.ADAPTIVE: 1.0,
            ControlStrategy.PREDICTIVE: 0.8,
            ControlStrategy.AGGRESSIVE: 1.3,
            ControlStrategy.OPTIMAL: 0.9
        }
        
        return base_settling * strategy_factors.get(strategy, 1.0)

    def _generate_recommendations(self, risk_index: float, strategy: ControlStrategy,
                                context: Dict[str, Any]) -> List[str]:
        """Generate operational recommendations."""
        recommendations = []
        
        # Risk-based recommendations
        if risk_index >= 80:
            recommendations.extend([
                "Consider immediate inspection",
                "Prepare for emergency procedures",
                "Increase monitoring frequency"
            ])
        elif risk_index >= 60:
            recommendations.extend([
                "Schedule maintenance within 24 hours",
                "Monitor temperature trends closely",
                "Review operational parameters"
            ])
        
        # Strategy-specific recommendations
        if strategy == ControlStrategy.PREDICTIVE:
            recommendations.append("Monitor predictive trends for early intervention")
        elif strategy == ControlStrategy.SAFETY:
            recommendations.append("Safety protocols active - maintain current monitoring")
        
        # Context-based recommendations
        primary_mode = context.get("primary_mode")
        if primary_mode == "degraded":
            recommendations.append("Equipment operating in degraded mode - plan maintenance")
        elif primary_mode == "high_load":
            recommendations.append("High load operation - consider load distribution")
        
        return recommendations

    def _calculate_optimization_metrics(self, force: float, risk_index: float, 
                                      telemetry: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimization performance metrics."""
        actuator_model = self.actuator_models.get(self.config["actuator_type"], {})
        
        # Estimated vibration reduction (simplified model)
        vibration_reduction = min(0.9, force / self.constraints.max_force * 0.8)
        
        # Energy efficiency
        power_func = actuator_model.get("power_consumption", lambda x: x)
        energy_consumption = power_func(force)
        efficiency = actuator_model.get("efficiency_curve", lambda x: 0.8)(force)
        
        # Equipment life impact
        life_impact = 1.0 - (force / self.constraints.max_force) * 0.3
        
        # Safety margin
        safety_margin = 1.0 - (risk_index / 100.0)
        
        return {
            "vibration_reduction": vibration_reduction,
            "energy_efficiency": efficiency,
            "equipment_life_impact": life_impact,
            "safety_margin": safety_margin,
            "power_consumption": energy_consumption,
            "overall_score": self._calculate_overall_score(
                vibration_reduction, efficiency, life_impact, safety_margin
            )
        }

    def _calculate_overall_score(self, vibration_reduction: float, efficiency: float,
                               life_impact: float, safety_margin: float) -> float:
        """Calculate overall optimization score."""
        weights = self.config["optimization_weights"]
        
        return (vibration_reduction * weights["vibration_reduction"] +
                efficiency * weights["energy_efficiency"] +
                life_impact * weights["equipment_life"] +
                safety_margin * weights["safety_margin"])

    def _get_safety_limits(self) -> Dict[str, float]:
        """Get current safety limits for reporting."""
        return {
            "max_force": self.constraints.max_force,
            "min_force": self.constraints.min_force,
            "max_temperature": self.constraints.temperature_limits["max_operating"],
            "max_vibration": self.constraints.vibration_limits["critical"],
            "max_power": self.constraints.power_limits["max_power"]
        }

    def _update_performance_metrics(self, action: ControlAction, start_time: datetime):
        """Update controller performance metrics."""
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        self.performance_metrics["total_actions"] += 1
        self.performance_metrics["response_times"].append(processing_time)
        
        # Keep only recent response times
        if len(self.performance_metrics["response_times"]) > 100:
            self.performance_metrics["response_times"] = self.performance_metrics["response_times"][-50:]
        
        # Update average response time
        avg_response = np.mean(self.performance_metrics["response_times"])
        self.performance_metrics["avg_response_time"] = avg_response

    def _store_control_action(self, action: ControlAction, risk_index: float, context: Dict[str, Any]):
        """Store control action in history."""
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "damper_force": action.damper_force,
            "control_strategy": action.control_strategy,
            "risk_index": risk_index,
            "confidence": action.confidence,
            "context_mode": context.get("primary_mode"),
            "optimization_score": action.optimization_metrics["overall_score"]
        }
        
        self.control_history.append(history_entry)
        
        # Maintain history size
        if len(self.control_history) > 1000:
            self.control_history = self.control_history[-500:]

    def _get_fallback_action(self, risk_index: float) -> ControlAction:
        """Provide fallback control action in case of errors."""
        fallback_force = min(5000.0, risk_index * 50.0)  # Conservative fallback
        
        return ControlAction(
            damper_force=fallback_force,
            control_strategy="safety",
            confidence=0.5,
            safety_limits=self._get_safety_limits(),
            timing={"response_time": 0.1, "update_interval": 1.0},
            recommendations=["System in fallback mode - verify operation"],
            optimization_metrics={"overall_score": 0.5}
        )

    def get_control_history(self, last_n: int = None) -> List[Dict]:
        """Get control action history."""
        if last_n:
            return self.control_history[-last_n:]
        return self.control_history

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        response_times = self.performance_metrics.get("response_times", [])
        
        return {
            "total_actions": self.performance_metrics["total_actions"],
            "avg_response_time": np.mean(response_times) if response_times else 0.0,
            "recent_strategy_distribution": self._get_strategy_distribution(),
            "safety_record": {
                "constraint_violations": self.performance_metrics.get("safety_violations", 0),
                "avg_confidence": np.mean([action.get("confidence", 0) for action in self.control_history[-100:]]) if self.control_history else 0.0
            },
            "optimization_performance": {
                "avg_score": np.mean([action.get("optimization_score", 0) for action in self.control_history[-100:]]) if self.control_history else 0.0
            }
        }

    def _get_strategy_distribution(self) -> Dict[str, int]:
        """Get distribution of control strategies in recent history."""
        if not self.control_history:
            return {}
        
        recent_actions = self.control_history[-100:]
        distribution = {}
        
        for action in recent_actions:
            strategy = action.get("control_strategy", "unknown")
            distribution[strategy] = distribution.get(strategy, 0) + 1
        
        return distribution

    def emergency_stop(self) -> ControlAction:
        """Generate emergency stop control action."""
        self.logger.critical("🛑 Generating emergency stop control action")
        
        return ControlAction(
            damper_force=self.constraints.max_force,
            control_strategy="safety",
            confidence=0.99,
            safety_limits=self._get_safety_limits(),
            timing={"response_time": 0.01, "update_interval": 0.1},
            recommendations=[
                "EMERGENCY STOP ACTIVATED",
                "Initiate shutdown procedures",
                "Alert maintenance team immediately"
            ],
            optimization_metrics={"overall_score": 1.0}  # Safety is paramount
        )

    def reset_controller(self):
        """Reset controller to initial state."""
        self.logger.info("Resetting Feedback Controller")
        
        self.control_history.clear()
        self.performance_metrics = self._initialize_metrics()
        self.last_action = None
        
        self.logger.info("Feedback Controller reset completed")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create controller
    controller = FeedbackController()
    
    # Test context
    test_context = {
        "primary_mode": "high_load",
        "confidence": 0.85,
        "risk_assessment": {"risk_category": "medium", "risk_level": 65},
        "temporal_context": {"is_night": False, "is_peak_hours": True},
        "equipment_context": {"overall_compliance": True},
        "historical_context": {"trend_direction": "stable", "mode_consistency": 0.9}
    }
    
    test_telemetry = {
        "vibration": 4.2,
        "temperature": 62.0,
        "pressure": 5.1
    }
    
    # Generate control action
    action = controller.propose_actions(65, test_context, test_telemetry)
    
    print("Control Action Generated:")
    print(f"Force: {action.damper_force:.0f}N")
    print(f"Strategy: {action.control_strategy}")
    print(f"Confidence: {action.confidence:.2f}")
    print(f"Recommendations: {action.recommendations}")
    print(f"Optimization Score: {action.optimization_metrics['overall_score']:.2f}")
    
    # Get performance report
    report = controller.get_performance_report()
    print(f"\nPerformance Report: {report}")
