# adaptive_learning/sample_data.py
"""
AVCS DNA-MATRIX SPIRIT v7.0
Advanced Synthetic Data Generator
-------------------------------
Comprehensive test data generation with realistic industrial patterns,
fault simulations, and temporal dynamics for system validation.
"""

import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging


class EquipmentMode(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    HIGH_LOAD = "high_load"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    BEARING_FAULT = "bearing_fault"
    IMBALANCE = "imbalance"
    MISALIGNMENT = "misalignment"
    CAVITATION = "cavitation"
    RESONANCE = "resonance"
    LUBRICATION_ISSUE = "lubrication_issue"


class TrendDirection(Enum):
    STABLE = "stable"
    INCREASING = "increasing"
    DECREASING = "decreasing"
    OSCILLATING = "oscillating"


@dataclass
class DataProfile:
    """Equipment data profile for different operational modes."""
    vibration_range: tuple
    temperature_range: tuple
    pressure_range: tuple
    rpm_range: tuple
    load_range: tuple
    noise_level: float
    trend_direction: TrendDirection
    fault_indicators: Dict[str, float]


class AdvancedDataGenerator:
    """
    Advanced synthetic data generator with realistic industrial patterns,
    fault simulations, and temporal dynamics.
    """

    def __init__(self, equipment_type: str = "centrifugal_pump", random_seed: int = None):
        self.equipment_type = equipment_type
        self.random_seed = random_seed or 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        self.logger = logging.getLogger("DataGenerator")
        self.data_profiles = self._initialize_data_profiles()
        self.temporal_state = self._initialize_temporal_state()
        
        self.logger.info(f"Advanced Data Generator initialized for {equipment_type}")

    def _initialize_data_profiles(self) -> Dict[EquipmentMode, DataProfile]:
        """Initialize data profiles for different equipment modes."""
        return {
            EquipmentMode.NORMAL: DataProfile(
                vibration_range=(0.2, 1.5),
                temperature_range=(55, 70),
                pressure_range=(3.5, 5.5),
                rpm_range=(2930, 2970),
                load_range=(0.8, 1.2),
                noise_level=0.1,
                trend_direction=TrendDirection.STABLE,
                fault_indicators={"bearing_wear": 0.0, "imbalance": 0.0, "misalignment": 0.0}
            ),
            
            EquipmentMode.DEGRADED: DataProfile(
                vibration_range=(2.0, 4.0),
                temperature_range=(70, 85),
                pressure_range=(5.0, 7.0),
                rpm_range=(2900, 3000),
                load_range=(1.1, 1.4),
                noise_level=0.3,
                trend_direction=TrendDirection.INCREASING,
                fault_indicators={"bearing_wear": 0.3, "imbalance": 0.2, "misalignment": 0.1}
            ),
            
            EquipmentMode.HIGH_LOAD: DataProfile(
                vibration_range=(1.5, 3.0),
                temperature_range=(65, 80),
                pressure_range=(6.0, 8.0),
                rpm_range=(2980, 3020),
                load_range=(1.5, 2.0),
                noise_level=0.2,
                trend_direction=TrendDirection.STABLE,
                fault_indicators={"bearing_wear": 0.1, "imbalance": 0.1, "misalignment": 0.0}
            ),
            
            EquipmentMode.STARTUP: DataProfile(
                vibration_range=(0.5, 2.0),
                temperature_range=(25, 60),
                pressure_range=(1.0, 4.0),
                rpm_range=(0, 3000),
                load_range=(0.5, 1.0),
                noise_level=0.4,
                trend_direction=TrendDirection.INCREASING,
                fault_indicators={"bearing_wear": 0.0, "imbalance": 0.0, "misalignment": 0.0}
            ),
            
            EquipmentMode.SHUTDOWN: DataProfile(
                vibration_range=(0.5, 2.0),
                temperature_range=(60, 25),
                pressure_range=(4.0, 1.0),
                rpm_range=(3000, 0),
                load_range=(1.0, 0.0),
                noise_level=0.3,
                trend_direction=TrendDirection.DECREASING,
                fault_indicators={"bearing_wear": 0.0, "imbalance": 0.0, "misalignment": 0.0}
            ),
            
            EquipmentMode.BEARING_FAULT: DataProfile(
                vibration_range=(4.0, 8.0),
                temperature_range=(75, 95),
                pressure_range=(4.0, 6.0),
                rpm_range=(2950, 3050),
                load_range=(1.0, 1.3),
                noise_level=0.6,
                trend_direction=TrendDirection.INCREASING,
                fault_indicators={"bearing_wear": 0.8, "imbalance": 0.3, "misalignment": 0.2}
            ),
            
            EquipmentMode.IMBALANCE: DataProfile(
                vibration_range=(3.0, 6.0),
                temperature_range=(65, 80),
                pressure_range=(4.5, 6.5),
                rpm_range=(2920, 2980),
                load_range=(1.0, 1.2),
                noise_level=0.5,
                trend_direction=TrendDirection.OSCILLATING,
                fault_indicators={"bearing_wear": 0.2, "imbalance": 0.7, "misalignment": 0.1}
            ),
            
            EquipmentMode.MISALIGNMENT: DataProfile(
                vibration_range=(2.5, 5.0),
                temperature_range=(70, 85),
                pressure_range=(5.0, 7.0),
                rpm_range=(2940, 2960),
                load_range=(1.1, 1.3),
                noise_level=0.4,
                trend_direction=TrendDirection.STABLE,
                fault_indicators={"bearing_wear": 0.1, "imbalance": 0.2, "misalignment": 0.6}
            ),
            
            EquipmentMode.CAVITATION: DataProfile(
                vibration_range=(1.0, 3.0),
                temperature_range=(60, 75),
                pressure_range=(1.5, 3.5),
                rpm_range=(2950, 3050),
                load_range=(0.9, 1.1),
                noise_level=0.7,
                trend_direction=TrendDirection.OSCILLATING,
                fault_indicators={"bearing_wear": 0.0, "imbalance": 0.0, "misalignment": 0.0}
            ),
            
            EquipmentMode.RESONANCE: DataProfile(
                vibration_range=(5.0, 10.0),
                temperature_range=(70, 90),
                pressure_range=(4.0, 6.0),
                rpm_range=(1480, 1520),  # Half speed for resonance
                load_range=(1.0, 1.1),
                noise_level=0.8,
                trend_direction=TrendDirection.OSCILLATING,
                fault_indicators={"bearing_wear": 0.1, "imbalance": 0.4, "misalignment": 0.2}
            ),
            
            EquipmentMode.LUBRICATION_ISSUE: DataProfile(
                vibration_range=(2.0, 4.0),
                temperature_range=(80, 100),
                pressure_range=(4.5, 6.0),
                rpm_range=(2930, 2970),
                load_range=(1.0, 1.2),
                noise_level=0.3,
                trend_direction=TrendDirection.INCREASING,
                fault_indicators={"bearing_wear": 0.4, "imbalance": 0.1, "misalignment": 0.0}
            )
        }

    def _initialize_temporal_state(self) -> Dict[str, Any]:
        """Initialize temporal state for realistic data generation."""
        return {
            "current_trend": 0.0,
            "oscillation_phase": 0.0,
            "long_term_drift": 0.0,
            "last_update": datetime.utcnow(),
            "cycle_count": 0,
            "seasonal_factor": 1.0
        }

    def generate_sample(self, mode: str = 'normal', 
                       include_advanced_metrics: bool = True,
                       add_noise: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive synthetic equipment data sample.
        
        Args:
            mode: Operational mode from EquipmentMode
            include_advanced_metrics: Include derived metrics and analysis
            add_noise: Add realistic measurement noise
            
        Returns:
            Dict: Comprehensive equipment data sample
        """
        try:
            equipment_mode = EquipmentMode(mode)
            profile = self.data_profiles[equipment_mode]
            
            # Update temporal state
            self._update_temporal_state(profile)
            
            # Generate base measurements
            base_data = self._generate_base_measurements(profile, add_noise)
            
            # Add fault-specific characteristics
            fault_data = self._add_fault_characteristics(base_data, profile, equipment_mode)
            
            # Add temporal patterns
            temporal_data = self._add_temporal_patterns(fault_data, profile)
            
            # Generate comprehensive sample
            sample = {
                'timestamp': datetime.utcnow().isoformat() + "Z",
                'equipment_mode': equipment_mode.value,
                'vibration': temporal_data['vibration'],
                'temperature': temporal_data['temperature'],
                'pressure': temporal_data['pressure'],
                'rpm': temporal_data['rpm'],
                'load': temporal_data['load'],
                'equipment_type': self.equipment_type,
                'data_quality': random.uniform(0.85, 0.99)
            }
            
            # Add advanced metrics if requested
            if include_advanced_metrics:
                sample.update(self._generate_advanced_metrics(sample, profile))
            
            # Add fault indicators
            sample.update({
                'fault_indicators': profile.fault_indicators,
                'trend_direction': profile.trend_direction.value,
                'noise_level': profile.noise_level
            })
            
            self.temporal_state['cycle_count'] += 1
            
            return sample
            
        except ValueError as e:
            self.logger.warning(f"Invalid mode '{mode}', using 'normal' as fallback: {e}")
            return self.generate_sample('normal', include_advanced_metrics, add_noise)
        
        except Exception as e:
            self.logger.error(f"Data generation failed: {e}")
            return self._generate_fallback_sample()

    def _generate_base_measurements(self, profile: DataProfile, add_noise: bool) -> Dict[str, float]:
        """Generate base equipment measurements."""
        # Base values without noise
        base_vibration = random.uniform(*profile.vibration_range)
        base_temperature = random.uniform(*profile.temperature_range)
        base_pressure = random.uniform(*profile.pressure_range)
        base_rpm = random.uniform(*profile.rpm_range)
        base_load = random.uniform(*profile.load_range)
        
        if add_noise:
            # Add realistic measurement noise
            vibration = base_vibration + random.gauss(0, profile.noise_level)
            temperature = base_temperature + random.gauss(0, profile.noise_level * 2)
            pressure = base_pressure + random.gauss(0, profile.noise_level * 0.5)
            rpm = base_rpm + random.gauss(0, profile.noise_level * 10)
            load = base_load + random.gauss(0, profile.noise_level * 0.1)
        else:
            vibration, temperature, pressure, rpm, load = (
                base_vibration, base_temperature, base_pressure, base_rpm, base_load
            )
        
        return {
            'vibration': max(0.01, round(vibration, 3)),
            'temperature': max(20, round(temperature, 1)),
            'pressure': max(0.1, round(pressure, 2)),
            'rpm': max(0, round(rpm)),
            'load': max(0.1, round(load, 2))
        }

    def _add_fault_characteristics(self, base_data: Dict[str, float], 
                                 profile: DataProfile, 
                                 mode: EquipmentMode) -> Dict[str, float]:
        """Add fault-specific characteristics to measurements."""
        data = base_data.copy()
        
        if mode == EquipmentMode.BEARING_FAULT:
            # Add high-frequency components for bearing faults
            bearing_frequency = data['rpm'] / 60.0 * 3.2  # BPFO approximation
            bearing_component = 0.5 * np.sin(2 * np.pi * bearing_frequency * 
                                           self.temporal_state['cycle_count'] * 0.1)
            data['vibration'] += abs(bearing_component)
            
        elif mode == EquipmentMode.IMBALANCE:
            # Strong 1x rotational component for imbalance
            imbalance_component = 0.3 * np.sin(2 * np.pi * data['rpm'] / 60.0 * 
                                             self.temporal_state['cycle_count'] * 0.1)
            data['vibration'] += abs(imbalance_component)
            
        elif mode == EquipmentMode.MISALIGNMENT:
            # Strong 2x rotational component for misalignment
            misalignment_component = 0.2 * np.sin(2 * np.pi * data['rpm'] / 60.0 * 2 * 
                                                self.temporal_state['cycle_count'] * 0.1)
            data['vibration'] += abs(misalignment_component)
            
        elif mode == EquipmentMode.CAVITATION:
            # Random high-frequency bursts for cavitation
            if random.random() < 0.1:  # 10% chance of cavitation burst
                cavitation_burst = random.uniform(0.5, 2.0)
                data['vibration'] += cavitation_burst
                data['pressure'] *= 0.8  # Pressure drop during cavitation
                
        elif mode == EquipmentMode.RESONANCE:
            # Resonance at specific frequency
            resonance_frequency = data['rpm'] / 60.0 * 0.5  # Half running speed
            resonance_component = 1.0 * np.sin(2 * np.pi * resonance_frequency * 
                                             self.temporal_state['cycle_count'] * 0.1)
            data['vibration'] += abs(resonance_component)
        
        return data

    def _add_temporal_patterns(self, data: Dict[str, float], profile: DataProfile) -> Dict[str, float]:
        """Add temporal patterns and trends to measurements."""
        result = data.copy()
        
        # Apply trend based on profile
        trend_factor = 1.0 + self.temporal_state['current_trend']
        
        if profile.trend_direction == TrendDirection.INCREASING:
            result['vibration'] *= trend_factor
            result['temperature'] *= trend_factor
        elif profile.trend_direction == TrendDirection.DECREASING:
            result['vibration'] /= trend_factor
            result['temperature'] /= trend_factor
        elif profile.trend_direction == TrendDirection.OSCILLATING:
            oscillation = 0.1 * np.sin(2 * np.pi * self.temporal_state['oscillation_phase'])
            result['vibration'] *= (1.0 + oscillation)
            result['temperature'] *= (1.0 + oscillation * 0.5)
        
        # Apply seasonal variations
        result['vibration'] *= self.temporal_state['seasonal_factor']
        result['temperature'] *= self.temporal_state['seasonal_factor']
        
        return result

    def _update_temporal_state(self, profile: DataProfile):
        """Update temporal state for realistic data generation."""
        current_time = datetime.utcnow()
        time_delta = (current_time - self.temporal_state['last_update']).total_seconds()
        
        # Update trend (slow drift)
        trend_change = random.gauss(0, 0.01)
        self.temporal_state['current_trend'] += trend_change
        self.temporal_state['current_trend'] = max(-0.3, min(0.3, self.temporal_state['current_trend']))
        
        # Update oscillation phase
        self.temporal_state['oscillation_phase'] += time_delta * 0.1
        if self.temporal_state['oscillation_phase'] > 1.0:
            self.temporal_state['oscillation_phase'] -= 1.0
        
        # Update long-term drift (very slow changes)
        if random.random() < 0.01:  # 1% chance per update
            drift_change = random.gauss(0, 0.001)
            self.temporal_state['long_term_drift'] += drift_change
        
        # Update seasonal factor (simulate daily/seasonal variations)
        hour_of_day = current_time.hour
        seasonal_variation = 0.1 * np.sin(2 * np.pi * hour_of_day / 24)
        self.temporal_state['seasonal_factor'] = 1.0 + seasonal_variation
        
        self.temporal_state['last_update'] = current_time

    def _generate_advanced_metrics(self, sample: Dict[str, Any], profile: DataProfile) -> Dict[str, Any]:
        """Generate advanced derived metrics and analysis."""
        vibration = sample['vibration']
        temperature = sample['temperature']
        pressure = sample['pressure']
        rpm = sample['rpm']
        load = sample['load']
        
        # Vibration analysis
        vibration_severity = min(1.0, vibration / 10.0)  # Normalize to 0-1
        crest_factor = random.uniform(2.0, 5.0) if vibration > 2.0 else random.uniform(1.5, 3.0)
        
        # Thermal analysis
        thermal_stress = max(0.0, (temperature - 60.0) / 40.0)  # Normalized thermal stress
        cooling_efficiency = max(0.1, 1.0 - thermal_stress * 0.5)
        
        # Efficiency calculations
        theoretical_power = rpm * load / 1000  # Simplified model
        actual_efficiency = random.uniform(0.75, 0.95) * (1.0 - vibration_severity * 0.2)
        
        # Health scores
        vibration_health = max(0.0, 1.0 - vibration_severity)
        thermal_health = max(0.0, 1.0 - thermal_stress)
        overall_health = (vibration_health * 0.6 + thermal_health * 0.4)
        
        return {
            'vibration_severity': round(vibration_severity, 3),
            'crest_factor': round(crest_factor, 2),
            'thermal_stress': round(thermal_stress, 3),
            'cooling_efficiency': round(cooling_efficiency, 3),
            'theoretical_power': round(theoretical_power, 2),
            'actual_efficiency': round(actual_efficiency, 3),
            'vibration_health': round(vibration_health, 3),
            'thermal_health': round(thermal_health, 3),
            'overall_health': round(overall_health, 3),
            'bearing_health': round(1.0 - profile.fault_indicators.get('bearing_wear', 0.0), 3),
            'balance_health': round(1.0 - profile.fault_indicators.get('imbalance', 0.0), 3),
            'alignment_health': round(1.0 - profile.fault_indicators.get('misalignment', 0.0), 3)
        }

    def _generate_fallback_sample(self) -> Dict[str, Any]:
        """Generate fallback sample in case of errors."""
        return {
            'timestamp': datetime.utcnow().isoformat() + "Z",
            'equipment_mode': 'normal',
            'vibration': 1.0,
            'temperature': 60.0,
            'pressure': 4.5,
            'rpm': 2950,
            'load': 1.0,
            'equipment_type': self.equipment_type,
            'data_quality': 0.5,
            'error': 'fallback_mode'
        }

    def generate_sequence(self, mode_sequence: List[str], 
                         interval: float = 1.0,
                         samples_per_mode: int = 10) -> List[Dict[str, Any]]:
        """
        Generate a sequence of samples following specified mode transitions.
        
        Args:
            mode_sequence: List of modes to transition through
            interval: Time between samples in seconds
            samples_per_mode: Samples to generate for each mode
            
        Returns:
            List: Sequence of data samples
        """
        sequence = []
        
        for mode in mode_sequence:
            for _ in range(samples_per_mode):
                sample = self.generate_sample(mode, include_advanced_metrics=True)
                sequence.append(sample)
                
                # Simulate time progression
                self.temporal_state['last_update'] += timedelta(seconds=interval)
        
        return sequence

    def generate_training_dataset(self, samples_per_mode: int = 100, 
                                include_labels: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive training dataset for ML models.
        
        Args:
            samples_per_mode: Number of samples per equipment mode
            include_labels: Include classification labels
            
        Returns:
            Dict: Training dataset with features and optional labels
        """
        features = []
        labels = []
        modes = []
        
        for equipment_mode in EquipmentMode:
            mode_name = equipment_mode.value
            
            for _ in range(samples_per_mode):
                sample = self.generate_sample(mode_name, include_advanced_metrics=True)
                
                # Extract features (basic measurements)
                feature_vector = [
                    sample['vibration'],
                    sample['temperature'],
                    sample['pressure'],
                    sample['rpm'],
                    sample['load']
                ]
                
                features.append(feature_vector)
                modes.append(mode_name)
                
                if include_labels:
                    # Simple binary classification: normal vs fault
                    is_fault = equipment_mode not in [
                        EquipmentMode.NORMAL, 
                        EquipmentMode.HIGH_LOAD,
                        EquipmentMode.STARTUP,
                        EquipmentMode.SHUTDOWN
                    ]
                    labels.append(1 if is_fault else 0)
        
        dataset = {
            'features': features,
            'modes': modes,
            'feature_names': ['vibration', 'temperature', 'pressure', 'rpm', 'load'],
            'timestamp': datetime.utcnow().isoformat(),
            'samples_per_mode': samples_per_mode,
            'total_samples': len(features)
        }
        
        if include_labels:
            dataset['labels'] = labels
            dataset['label_names'] = ['normal', 'fault']
        
        return dataset

    def get_mode_statistics(self) -> Dict[str, Any]:
        """Get statistics about available equipment modes."""
        return {
            'available_modes': [mode.value for mode in EquipmentMode],
            'normal_modes': ['normal', 'high_load', 'startup', 'shutdown'],
            'fault_modes': ['degraded', 'bearing_fault', 'imbalance', 'misalignment', 
                           'cavitation', 'resonance', 'lubrication_issue'],
            'total_modes': len(EquipmentMode),
            'data_profiles_configured': len(self.data_profiles)
        }


# Legacy function for backward compatibility
def generate_sample(mode: str = 'normal') -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    Generates synthetic data sample for local testing.
    
    Args:
        mode: Operational mode ('normal', 'degraded', etc.)
        
    Returns:
        Dict: Equipment data sample
    """
    generator = AdvancedDataGenerator()
    return generator.generate_sample(mode, include_advanced_metrics=False)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create advanced data generator
    generator = AdvancedDataGenerator("centrifugal_pump")
    
    # Test single sample generation
    print("=== Testing Single Sample Generation ===")
    sample = generator.generate_sample('bearing_fault', include_advanced_metrics=True)
    print(f"Generated sample for {sample['equipment_mode']}:")
    print(f"Vibration: {sample['vibration']}, Temperature: {sample['temperature']}")
    print(f"Overall Health: {sample.get('overall_health', 'N/A')}")
    print(f"Fault Indicators: {sample.get('fault_indicators', {})}")
    
    # Test sequence generation
    print("\n=== Testing Sequence Generation ===")
    mode_sequence = ['normal', 'degraded', 'bearing_fault', 'normal']
    sequence = generator.generate_sequence(mode_sequence, samples_per_mode=3)
    
    for i, seq_sample in enumerate(sequence):
        print(f"Sample {i+1}: {seq_sample['equipment_mode']} - "
              f"Vib: {seq_sample['vibration']:.2f}, "
              f"Temp: {seq_sample['temperature']:.1f}")
    
    # Test training dataset generation
    print("\n=== Testing Training Dataset Generation ===")
    dataset = generator.generate_training_dataset(samples_per_mode=5, include_labels=True)
    print(f"Generated dataset with {dataset['total_samples']} samples")
    print(f"Features: {len(dataset['features'])}x{len(dataset['features'][0])}")
    print(f"Labels: {len(dataset.get('labels', []))}")
    
    # Display mode statistics
    stats = generator.get_mode_statistics()
    print(f"\n=== Mode Statistics ===")
    print(f"Available modes: {stats['available_modes']}")
    print(f"Normal modes: {stats['normal_modes']}")
    print(f"Fault modes: {stats['fault_modes']}")
    
    # Test legacy function
    print("\n=== Testing Legacy Function ===")
    legacy_sample = generate_sample('degraded')
    print(f"Legacy sample: Vibration={legacy_sample['vibration']}, "
          f"Temperature={legacy_sample['temperature']}")
