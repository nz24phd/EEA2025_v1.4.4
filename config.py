# config.py
# Configuration file for BDWPT simulation platform

import os  # 确保导入os模块
import numpy as np
from datetime import datetime, timedelta

# --- START OF FINAL FIX: Define an absolute base path ---
# 获取此配置文件所在的目录的绝对路径
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# --- END OF FINAL FIX ---

class SimulationConfig:
    """Configuration parameters for the BDWPT simulation."""
    
    def __init__(self):
        # --- START OF FINAL FIX: Build all paths from the absolute base path ---
        # Data directory path
        self.data_dir = os.path.join(_BASE_DIR, "data")
        
        # Output directory paths
        self.output_dir = os.path.join(_BASE_DIR, "output")
        self.figures_dir = os.path.join(self.output_dir, "figures")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        # --- END OF FINAL FIX ---
        
        # Simulation parameters
        self.simulation_params = {
            'start_time': datetime(2024, 1, 1, 0, 0),  # Start at midnight
            'end_time': datetime(2024, 1, 1, 23, 59),  # End at 23:59
            'time_step_minutes': 15,  # 15-minute time steps
            'day_types': ['weekday', 'weekend']
        }
        
        # Traffic model parameters
        self.traffic_params = {
            'total_vehicles': 1000,
            'ev_penetration': 0.3,  # 30% EV penetration
            'trips_per_vehicle_per_day': 2.5,  # Average trips per vehicle per day
            'trip_generation_rate': 2.5,  # Average trips per vehicle per day
            'peak_hour_factor': 1.5,
            'speed_limit_kmh': 50,
            'average_trip_distance_km': 8.5  # Average trip distance in km
        }
        
        # Add time step convenience property
        self.time_step_minutes = self.simulation_params['time_step_minutes']
        
        # EV parameters
        self.ev_params = {
            'initial_soc_mean': 0.7,  # 70% average initial SOC
            'initial_soc_std': 0.15,  # 15% standard deviation
            'charging_efficiency': 0.9,  # 90% charging efficiency
            'energy_consumption_kwh_per_km': 0.15,  # 150 Wh/km (alias for compatibility)
            'min_soc_threshold': 0.2,  # 20% minimum SOC
            'max_soc_threshold': 0.9   # 90% maximum SOC for normal charging
        }
        
        # BDWPT system parameters
        self.bdwpt_params = {
            'max_power_kw': 50,  # Maximum BDWPT power per vehicle
            'charging_power_kw': 50,  # Charging power
            'discharging_power_kw': 30,  # Discharging power for V2G
            'efficiency': 0.85,  # 85% wireless power transfer efficiency
            'activation_distance_m': 5,  # Distance for BDWPT activation
            'min_vehicle_speed_kmh': 5,  # Minimum speed for BDWPT operation
            'max_vehicle_speed_kmh': 60,  # Maximum speed for BDWPT operation
            'power_control_algorithm': 'voltage_regulation'
        }
        
        # BDWPT control parameters
        self.control_params = {
            'soc_force_charge': 0.2,  # Force charging below this SoC
            'soc_force_discharge': 0.9,  # Force discharging above this SoC
            'soc_min_v2g': 0.3,  # Minimum SoC for V2G operation
            'voltage_critical_high': 1.05,  # Critical high voltage (p.u.)
            'voltage_critical_low': 0.95,  # Critical low voltage (p.u.)
            'voltage_high_threshold': 1.02,  # High voltage threshold
            'voltage_low_threshold': 0.98,  # Low voltage threshold
            'tariff_high_threshold': 20.0,  # High tariff threshold (cents/kWh)
            'tariff_low_threshold': 15.0,  # Low tariff threshold (cents/kWh)
            'hysteresis_factor': 0.1  # Hysteresis factor for mode switching
        }
        
        # Power grid parameters (IEEE 13-bus system)
        self.grid_params = {
            'base_voltage_kv': 4.16,  # 4.16 kV base voltage
            'base_power_mva': 5.0,    # 5 MVA base power
            'voltage_tolerance': 0.05,  # ±5% voltage tolerance
            'max_loading_percent': 80,  # 80% maximum loading
            'bdwpt_nodes': [632, 633, 634, 645, 646, 671, 675, 680],  # IEEE 13-bus node numbers
            'bdwpt_connection_type': 'three_phase'
        }
        
        # Scenario configuration (RESTORED)
        self.scenario_params = {
            'base_case': {
                'bdwpt_penetration': 0,
                'description': 'Baseline scenario without BDWPT'
            },
            'low_penetration': {
                'bdwpt_penetration': 10,
                'description': '10% BDWPT penetration'
            },
            'medium_penetration': {
                'bdwpt_penetration': 25,
                'description': '25% BDWPT penetration'
            },
            'high_penetration': {
                'bdwpt_penetration': 50,
                'description': '50% BDWPT penetration'
            }
        }
        
        # Penetration scenarios list for easy iteration
        self.penetration_scenarios = [0, 15, 40]
        
        # Base scenarios configuration
        self.scenarios = {
            'Weekday Peak': {
                'load_profile': 'weekday_peak',
                'traffic_multiplier': 1.5,
                'description': 'Weekday peak hours scenario'
            },
            'Weekday Off-Peak': {
                'load_profile': 'weekday_offpeak',
                'traffic_multiplier': 0.8,
                'description': 'Weekday off-peak hours scenario'
            },
            'Weekend Peak': {
                'load_profile': 'weekend_peak',
                'traffic_multiplier': 1.2,
                'description': 'Weekend peak hours scenario'
            },
            'Weekend': {
                'load_profile': 'weekend',
                'traffic_multiplier': 1.0,
                'description': 'Weekend scenario'
            }
        }
        
        # Data file paths (RESTORED and FIXED)
        self.data_paths = {
            'ev_registrations': os.path.join(self.data_dir, 'ev_registrations.csv'),
            'road_network': os.path.join(self.data_dir, 'wellington_roads.json'),
            'load_profiles': os.path.join(self.data_dir, 'load_profiles.csv'),
            'weather_data': os.path.join(self.data_dir, 'weather_data.csv')
        }
        
        # Output paths (RESTORED and FIXED)
        self.output_paths = {
            'results': self.results_dir,
            'figures': self.figures_dir,
            'logs': self.logs_dir
        }
        
        # Logging configuration (RESTORED)
        self.logging_config = {
            'level': 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'simulation.log'
        }

    def get_time_steps(self):
        """Generate list of time steps for the simulation."""
        time_steps = []
        current_time = self.simulation_params['start_time']
        end_time = self.simulation_params['end_time']
        step_delta = timedelta(minutes=self.simulation_params['time_step_minutes'])
        
        while current_time <= end_time:
            time_steps.append(current_time)
            current_time += step_delta
        
        return time_steps

    def get_time_step_minutes(self, time_step):
        """Convert datetime to minutes from start of day."""
        start_of_day = time_step.replace(hour=0, minute=0, second=0, microsecond=0)
        return int((time_step - start_of_day).total_seconds() / 60)

    def get_day_type(self, date):
        """Determine if the given date is a weekday or weekend."""
        return 'weekend' if date.weekday() >= 5 else 'weekday'

    def get_load_profile(self, node_id, time_minutes, day_type):
        """Get load profile for a specific node and time."""
        hour = time_minutes // 60
        
        if day_type == 'weekend':
            base_factor = 0.6 + 0.3 * np.sin(2 * np.pi * (hour - 8) / 24)
        else:
            morning_peak = 0.8 * np.exp(-((hour - 8) ** 2) / 8)
            evening_peak = 1.0 * np.exp(-((hour - 18) ** 2) / 12)
            base_factor = 0.4 + morning_peak + evening_peak
        
        node_factors = {
            632: 1.2, 633: 0.8, 634: 1.0, 645: 0.9, 646: 1.1,
            671: 0.7, 675: 1.3, 680: 0.6
        }
        node_factor = node_factors.get(node_id, 1.0)
        base_load_kw = 100
        
        return base_load_kw * base_factor * node_factor

    def get_time_series(self):
        """Generate time series for simulation based on configuration."""
        time_steps = self.get_time_steps()
        return {
            'time_steps': time_steps,
            'time_minutes': [self.get_time_step_minutes(ts) for ts in time_steps],
            'total_steps': len(time_steps)
        }

    def get_tariff_at_hour(self, hour):
        """Get electricity tariff rate for a specific hour."""
        if 6 <= hour < 10 or 17 <= hour < 21:
            return 25.0
        elif 10 <= hour < 17:
            return 18.0
        else:
            return 12.0

    def validate_config(self):
        """Validate configuration parameters."""
        if not all(isinstance(node, int) for node in self.grid_params['bdwpt_nodes']):
            raise ValueError("BDWPT nodes must be integers")
        
        if not all(0 <= p <= 100 for p in self.penetration_scenarios):
            raise ValueError("Penetration scenarios must be between 0 and 100")
        
        if self.simulation_params['start_time'] >= self.simulation_params['end_time']:
            raise ValueError("Start time must be before end time")
        
        if not 0 <= self.ev_params['initial_soc_mean'] <= 1:
            raise ValueError("Initial SOC mean must be between 0 and 1")
        
        return True

# Create global config instance
config = SimulationConfig()