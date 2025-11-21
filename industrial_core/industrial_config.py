# ============================================================
# AVCS DNA-MATRIX SPIRIT v7.0
# File: industrial_config.py
# Purpose: Advanced configuration management with validation,
#          environment awareness, and dynamic updates
# Author: AVCS Engineering Team
# ============================================================

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
import hashlib


class EquipmentType(Enum):
    CENTRIFUGAL_PUMP = "centrifugal_pump"
    COMPRESSOR = "compressor" 
    TURBINE = "turbine"
    FAN = "fan"
    CONVEYOR = "conveyor"


class EnvironmentType(Enum):
    OFFSHORE = "offshore"
    ONSHORE = "onshore"
    MARINE = "marine"
    INDUSTRIAL = "industrial"


@dataclass
class SensorConfig:
    """Configuration for individual sensor."""
    name: str
    threshold: float
    unit: str
    calibration_date: str
    critical_threshold: float
    warning_threshold: float
    sampling_rate: int


@dataclass
class DamperConfig:
    """Configuration for MR damper."""
    location: str
    max_force: float
    min_force: float
    response_time: float
    orientation: float  # degrees
    calibration_factor: float


@dataclass
class SafetyLimits:
    """Safety threshold configurations."""
    vibration_max: float
    temperature_max: float
    pressure_max: float
    noise_max: float
    current_max: float
    displacement_max: float


class IndustrialConfig:
    """
    Advanced configuration management for AVCS-DNA-MATRIX-SPIRIT v7.0.
    Features dynamic updates, environment awareness, and validation.
    """

    def __init__(self, config_path: str = None, environment: str = "offshore"):
        self.config_path = config_path or "config/avcs_config.json"
        self.environment = EnvironmentType(environment)
        self.logger = logging.getLogger("IndustrialConfig")
        
        # Ensure config directory exists
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._load_or_initialize()
        self._validate_config()

    # --------------------------------------------------------
    # Enhanced Configuration Initialization
    # --------------------------------------------------------
    def _load_or_initialize(self):
        """Load existing config or create environment-specific defaults."""
        path = Path(self.config_path)

        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                self.logger.info(f"Loaded configuration from {path}")
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Config file corrupted: {e}. Creating new one.")
                self._create_environment_config()
        else:
            self._create_environment_config()

        self._apply_config()

    def _create_environment_config(self):
        """Create environment-specific default configuration."""
        self.config = self._get_environment_defaults(self.environment)
        
        # Save the new configuration
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Created new {self.environment.value} configuration")

    def _get_environment_defaults(self, environment: EnvironmentType) -> Dict[str, Any]:
        """Get environment-specific default configurations."""
        base_config = {
            "system_name": "AVCS DNA-MATRIX SPIRIT v7.0",
            "version": "7.0.0",
            "author": "AVCS Engineering",
            "description": "Advanced Vibration Control System with AI-driven predictive intelligence",
            "environment": environment.value,
            "created": datetime.utcnow().isoformat(),
            "last_modified": datetime.utcnow().isoformat(),

            # Equipment configuration
            "EQUIPMENT": {
                "type": "centrifugal_pump",
                "model": "Sulzer MSD 450",
                "nominal_rpm": 2950,
                "power_rating": 450,  # kW
                "operating_hours": 0,
                "installation_date": "2024-01-01"
            },

            # Enhanced vibration sensors with calibration data
            "VIBRATION_SENSORS": {
                "VS1": {
                    "name": "Motor Drive End Bearing",
                    "threshold": 3.5,
                    "critical_threshold": 6.0,
                    "warning_threshold": 4.0,
                    "unit": "mm/s",
                    "calibration_date": "2024-01-15",
                    "sampling_rate": 25600,
                    "sensitivity": 100  # mV/g
                },
                "VS2": {
                    "name": "Pump Inlet Bearing", 
                    "threshold": 3.0,
                    "critical_threshold": 5.5,
                    "warning_threshold": 3.8,
                    "unit": "mm/s",
                    "calibration_date": "2024-01-15",
                    "sampling_rate": 25600,
                    "sensitivity": 100
                },
                "VS3": {
                    "name": "Gearbox Input Shaft",
                    "threshold": 3.2,
                    "critical_threshold": 5.8,
                    "warning_threshold": 4.2,
                    "unit": "mm/s", 
                    "calibration_date": "2024-01-15",
                    "sampling_rate": 25600,
                    "sensitivity": 100
                },
                "VS4": {
                    "name": "Output Coupling",
                    "threshold": 2.8,
                    "critical_threshold": 5.0,
                    "warning_threshold": 3.5,
                    "unit": "mm/s",
                    "calibration_date": "2024-01-15",
                    "sampling_rate": 25600,
                    "sensitivity": 100
                }
            },

            # Enhanced thermal monitoring
            "THERMAL_SENSORS": {
                "TS1": {
                    "name": "Motor Stator Winding",
                    "threshold": 85,
                    "critical_threshold": 110,
                    "warning_threshold": 95,
                    "unit": "°C",
                    "calibration_date": "2024-01-15"
                },
                "TS2": {
                    "name": "Gearbox Bearing Housing",
                    "threshold": 75,
                    "critical_threshold": 95,
                    "warning_threshold": 85,
                    "unit": "°C",
                    "calibration_date": "2024-01-15"
                },
                "TS3": {
                    "name": "Pump Casing",
                    "threshold": 70,
                    "critical_threshold": 90,
                    "warning_threshold": 80,
                    "unit": "°C",
                    "calibration_date": "2024-01-15"
                },
                "TS4": {
                    "name": "Lubrication Oil",
                    "threshold": 65,
                    "critical_threshold": 85,
                    "warning_threshold": 75,
                    "unit": "°C",
                    "calibration_date": "2024-01-15"
                }
            },

            # Advanced MR damper configurations
            "MR_DAMPERS": {
                "D1": {
                    "location": "Motor Base Front",
                    "max_force": 1200,  # N
                    "min_force": 50,    # N
                    "response_time": 0.02,  # seconds
                    "orientation": 45,  # degrees
                    "calibration_factor": 1.0
                },
                "D2": {
                    "location": "Motor Base Rear", 
                    "max_force": 1200,
                    "min_force": 50,
                    "response_time": 0.02,
                    "orientation": 45,
                    "calibration_factor": 1.0
                },
                "D3": {
                    "location": "Pump Support Left",
                    "max_force": 1000,
                    "min_force": 50, 
                    "response_time": 0.025,
                    "orientation": 45,
                    "calibration_factor": 1.0
                },
                "D4": {
                    "location": "Pump Support Right",
                    "max_force": 1000,
                    "min_force": 50,
                    "response_time": 0.025,
                    "orientation": 45,
                    "calibration_factor": 1.0
                }
            },

            # Adaptive force control strategies
            "DAMPER_CONTROL": {
                "standby": 0.0,
                "normal": 0.3,
                "elevated": 0.6, 
                "warning": 0.8,
                "critical": 1.0,
                "emergency": 1.0,
                "response_curve": "exponential",  # linear, exponential, logarithmic
                "adaptive_gain": 1.2
            },

            # Comprehensive safety system
            "SAFETY_SYSTEM": {
                "vibration_max": 8.0,
                "temperature_max": 120,
                "pressure_max": 25.0,
                "current_max": 550,
                "noise_max": 85,
                "displacement_max": 2.0,
                "emergency_shutdown_time": 2.0,  # seconds
                "auto_restart": False,
                "safety_margin": 0.8  # 80% of critical limits
            },

            # Digital twin and AI configuration
            "DIGITAL_TWIN": {
                "model": "centrifugal_pump_sulzer_msd",
                "update_frequency": 1.0,  # Hz
                "prediction_horizon": 24,  # hours
                "confidence_threshold": 0.75,
                "retraining_interval": 168  # hours (1 week)
            },

            # Communication and integration
            "COMMUNICATION": {
                "opc_ua_endpoint": "opc.tcp://localhost:4840",
                "mqtt_broker": "localhost:1883",
                "modbus_port": 502,
                "websocket_port": 8765,
                "api_timeout": 30,
                "data_retention_days": 90
            },

            # Maintenance and reporting
            "MAINTENANCE": {
                "inspection_interval": 720,  # hours
                "calibration_interval": 2160,  # hours (90 days)
                "reporting_interval": 24,  # hours
                "trend_analysis_window": 30,  # days
                "spare_parts": ["bearings", "seals", "dampers"]
            }
        }

        # Apply environment-specific adjustments
        environment_adjustments = {
            EnvironmentType.OFFSHORE: {
                "SAFETY_SYSTEM": {"safety_margin": 0.7, "emergency_shutdown_time": 1.5},
                "EQUIPMENT": {"corrosion_protection": "marine_grade"}
            },
            EnvironmentType.MARINE: {
                "SAFETY_SYSTEM": {"safety_margin": 0.65, "vibration_max": 7.0},
                "DAMPER_CONTROL": {"adaptive_gain": 1.4}
            },
            EnvironmentType.INDUSTRIAL: {
                "MAINTENANCE": {"inspection_interval": 480},
                "COMMUNICATION": {"data_retention_days": 60}
            }
        }

        if environment in environment_adjustments:
            self._deep_update(base_config, environment_adjustments[environment])

        return base_config

    def _deep_update(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Recursively update nested dictionaries."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    # --------------------------------------------------------
    # Configuration Validation
    # --------------------------------------------------------
    def _validate_config(self):
        """Validate configuration integrity and consistency."""
        errors = []
        warnings = []

        # Validate sensor thresholds
        for sensor_id, sensor in self.config["VIBRATION_SENSORS"].items():
            if sensor["threshold"] >= sensor["critical_threshold"]:
                errors.append(f"{sensor_id}: Warning threshold >= Critical threshold")
            if sensor["sampling_rate"] < 1000:
                warnings.append(f"{sensor_id}: Low sampling rate for vibration analysis")

        # Validate damper configurations
        for damper_id, damper in self.config["MR_DAMPERS"].items():
            if damper["max_force"] <= damper["min_force"]:
                errors.append(f"{damper_id}: Max force <= Min force")
            if damper["response_time"] > 0.1:
                warnings.append(f"{damper_id}: Slow response time for active control")

        if errors:
            self.logger.error(f"Configuration errors: {errors}")
            raise ValueError(f"Invalid configuration: {errors}")
        
        if warnings:
            self.logger.warning(f"Configuration warnings: {warnings}")

    # --------------------------------------------------------
    # Advanced Configuration Application
    # --------------------------------------------------------
    def _apply_config(self):
        """Apply configuration to object attributes with type conversion."""
        try:
            # Core system attributes
            self.system_name = self.config["system_name"]
            self.version = self.config["version"]
            self.environment = EnvironmentType(self.config["environment"])
            
            # Equipment configuration
            self.equipment_config = self.config["EQUIPMENT"]
            
            # Sensor configurations with dataclass conversion
            self.vibration_sensors = {
                k: SensorConfig(**v) for k, v in self.config["VIBRATION_SENSORS"].items()
            }
            self.thermal_sensors = {
                k: SensorConfig(**v) for k, v in self.config["THERMAL_SENSORS"].items()
            }
            
            # Damper configurations
            self.mr_dampers = {
                k: DamperConfig(**v) for k, v in self.config["MR_DAMPERS"].items()
            }
            
            # Control and safety systems
            self.damper_control = self.config["DAMPER_CONTROL"]
            self.safety_system = self.config["SAFETY_SYSTEM"]
            self.digital_twin = self.config["DIGITAL_TWIN"]
            self.communication = self.config["COMMUNICATION"]
            self.maintenance = self.config["MAINTENANCE"]
            
            # Calculate derived parameters
            self._calculate_derived_parameters()
            
        except KeyError as e:
            self.logger.error(f"Missing configuration key: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error applying configuration: {e}")
            raise

    def _calculate_derived_parameters(self):
        """Calculate derived operational parameters."""
        # Operating force ranges for dampers
        self.operating_ranges = {}
        for damper_id, damper in self.mr_dampers.items():
            self.operating_ranges[damper_id] = {
                'min_operating': damper.min_force,
                'max_operating': damper.max_force,
                'force_range': damper.max_force - damper.min_force
            }

        # Safety margins
        self.effective_limits = {
            'vibration': self.safety_system['vibration_max'] * self.safety_system['safety_margin'],
            'temperature': self.safety_system['temperature_max'] * self.safety_system['safety_margin'],
            'pressure': self.safety_system['pressure_max'] * self.safety_system['safety_margin']
        }

    # --------------------------------------------------------
    # Advanced Configuration Management
    # --------------------------------------------------------
    def update_parameter(self, section: str, key: str, value: Any, save: bool = True):
        """Safely update configuration parameter."""
        try:
            # Navigate to the section and update the key
            keys = section.split('.')
            current = self.config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]][key] = value
            
            # Update timestamp
            self.config["last_modified"] = datetime.utcnow().isoformat()
            
            # Re-apply configuration
            self._apply_config()
            
            # Save if requested
            if save:
                self.save()
                
            self.logger.info(f"Updated {section}.{key} = {value}")
            
        except KeyError as e:
            self.logger.error(f"Invalid configuration path: {section}.{key}")
            raise

    def get_parameter(self, section: str, key: str = None) -> Any:
        """Get configuration parameter with nested path support."""
        try:
            keys = section.split('.')
            current = self.config
            for k in keys:
                current = current[k]
            return current[key] if key else current
        except KeyError:
            self.logger.warning(f"Parameter not found: {section}.{key}")
            return None

    def export_section(self, section: str) -> Dict[str, Any]:
        """Export specific configuration section for external use."""
        return self.get_parameter(section)

    def create_backup(self, backup_path: str = None):
        """Create configuration backup with timestamp."""
        if backup_path is None:
            backup_dir = Path(self.config_path).parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"config_backup_{timestamp}.json"
        
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Configuration backed up to: {backup_path}")

    # --------------------------------------------------------
    # Save configuration with integrity check
    # --------------------------------------------------------
    def save(self, backup: bool = True):
        """Save configuration with optional backup and validation."""
        if backup:
            self.create_backup()
        
        # Validate before saving
        self._validate_config()
        
        # Update modification timestamp
        self.config["last_modified"] = datetime.utcnow().isoformat()
        
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Configuration saved to: {self.config_path}")

    # --------------------------------------------------------
    # Enhanced Display and Reporting
    # --------------------------------------------------------
    def show_summary(self, detailed: bool = False):
        """Display configuration summary with optional details."""
        print(f"🔧 {self.system_name} (v{self.version})")
        print(f"📍 Environment: {self.environment.value}")
        print(f"🏭 Equipment: {self.equipment_config['type']} - {self.equipment_config['model']}")
        print(f"📅 Last Modified: {self.config['last_modified']}")
        
        if detailed:
            print("\n📊 Detailed Configuration:")
            print(f"Vibration Sensors: {len(self.vibration_sensors)}")
            for sensor_id, sensor in self.vibration_sensors.items():
                print(f"  - {sensor_id}: {sensor.name} (threshold: {sensor.threshold} {sensor.unit})")
            
            print(f"MR Dampers: {len(self.mr_dampers)}")
            for damper_id, damper in self.mr_dampers.items():
                print(f"  - {damper_id}: {damper.location} (max force: {damper.max_force}N)")
            
            print(f"Safety Limits: Vibration={self.safety_system['vibration_max']}, "
                  f"Temperature={self.safety_system['temperature_max']}°C")

    def generate_config_report(self) -> Dict[str, Any]:
        """Generate comprehensive configuration report."""
        return {
            "system_info": {
                "name": self.system_name,
                "version": self.version,
                "environment": self.environment.value,
                "config_hash": self._calculate_config_hash()
            },
            "equipment_summary": {
                "type": self.equipment_config["type"],
                "model": self.equipment_config["model"],
                "sensors": {
                    "vibration": len(self.vibration_sensors),
                    "thermal": len(self.thermal_sensors)
                },
                "dampers": len(self.mr_dampers)
            },
            "safety_metrics": {
                "effective_vibration_limit": self.effective_limits['vibration'],
                "effective_temperature_limit": self.effective_limits['temperature'],
                "safety_margin": self.safety_system['safety_margin']
            },
            "maintenance_schedule": {
                "next_inspection_hours": self.maintenance['inspection_interval'],
                "next_calibration_hours": self.maintenance['calibration_interval']
            }
        }

    def _calculate_config_hash(self) -> str:
        """Calculate MD5 hash of configuration for integrity checking."""
        config_string = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_string.encode()).hexdigest()

    # --------------------------------------------------------
    # Environment-specific adjustments
    # --------------------------------------------------------
    def change_environment(self, new_environment: str, save: bool = True):
        """Change operating environment and adjust parameters."""
        old_environment = self.environment
        self.environment = EnvironmentType(new_environment)
        
        # Reload configuration for new environment
        self.config = self._get_environment_defaults(self.environment)
        self._apply_config()
        
        if save:
            self.save(backup=True)
        
        self.logger.info(f"Environment changed from {old_environment.value} to {new_environment}")


# ------------------------------------------------------------
# Example usage and testing
# ------------------------------------------------------------
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration for offshore environment
    config = IndustrialConfig(environment="offshore")
    config.show_summary(detailed=True)
    
    # Generate configuration report
    report = config.generate_config_report()
    print("\n📋 Configuration Report:")
    print(json.dumps(report, indent=2))
    
    # Example of parameter update
    config.update_parameter("DAMPER_CONTROL", "adaptive_gain", 1.5)
