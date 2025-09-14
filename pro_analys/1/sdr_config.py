
"""
Configuration file for SDR Application
Cấu hình các thông số mặc định
"""

import json
import os


class SDRConfig:
    """SDR Application Configuration"""

    def __init__(self, config_file='sdr_config.json'):
        self.config_file = config_file
        self.default_config = {
            # USRP Settings
            "usrp": {
                "default_device_args": "type=usrp2",
                "default_center_freq": 100e6,  # 100 MHz
                "default_sample_rate": 1e6,    # 1 MS/s
                "default_gain": 50,            # dB
                "connection_timeout": 5.0,     # seconds
                "recv_timeout": 0.1,          # seconds
                "recv_buffer_size": 1024       # samples
            },

            # GUI Settings
            "gui": {
                "update_rate": 20,              # FPS
                "spectrum_fft_size": 1024,
                "waterfall_history": 100,       # lines
                "constellation_max_points": 1000,
                "default_window_size": [1400, 900],
                "theme": "dark"
            },

            # Signal Processing
            "signal_processing": {
                "peak_detection_threshold": -50,  # dB
                "peak_prominence": 5,              # dB
                "cfar_false_alarm_rate": 0.01,
                "symbol_sync_algorithm": "gardner",
                "carrier_sync_bandwidth": 0.01,
                "timing_sync_bandwidth": 0.01
            },

            # Demodulation
            "demodulation": {
                "supported_modes": [
                    "BPSK", "QPSK", "OQPSK", 
                    "8PSK", "8QAM", "16QAM"
                ],
                "auto_detect_enabled": True,
                "classification_method": "cumulants",  # or "ml"
                "constellation_clustering": True
            },

            # Recording
            "recording": {
                "default_format": "complex64",     # numpy dtype
                "default_directory": "./recordings",
                "auto_timestamp": True,
                "max_file_size_mb": 1000,
                "compression": False
            },

            # Scanning
            "scanning": {
                "default_start_freq": 88e6,       # FM band start
                "default_end_freq": 108e6,        # FM band end  
                "default_step_size": 1e6,         # 1 MHz steps
                "default_dwell_time": 0.1,        # seconds
                "detection_threshold": -60,       # dB
                "bandwidth_threshold": 3          # dB for BW estimation
            },

            # Advanced Features
            "advanced": {
                "enable_gpu_acceleration": False,
                "enable_multithread": True,
                "max_worker_threads": 4,
                "enable_logging": True,
                "log_level": "INFO",
                "log_file": "sdr_app.log"
            }
        }

        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                config = self.default_config.copy()
                self._deep_update(config, loaded_config)
                return config
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
                return self.default_config.copy()
        else:
            # Create default config file
            self.save_config(self.default_config)
            return self.default_config.copy()

    def save_config(self, config=None):
        """Save configuration to file"""
        if config is None:
            config = self.config

        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key_path, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        self.save_config()

    def _deep_update(self, base_dict, update_dict):
        """Recursively update nested dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def get_usrp_config(self):
        """Get USRP-specific configuration"""
        return self.config['usrp']

    def get_gui_config(self):
        """Get GUI-specific configuration"""  
        return self.config['gui']

    def get_signal_processing_config(self):
        """Get signal processing configuration"""
        return self.config['signal_processing']

    def get_demodulation_config(self):
        """Get demodulation configuration"""
        return self.config['demodulation']

    def get_recording_config(self):
        """Get recording configuration"""
        return self.config['recording']

    def get_scanning_config(self):
        """Get scanning configuration"""
        return self.config['scanning']

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.default_config.copy()
        self.save_config()


# Global config instance
sdr_config = SDRConfig()


# Utility functions for common configurations
def get_usrp_device_args():
    """Get default USRP device arguments"""
    return sdr_config.get('usrp.default_device_args')

def get_default_frequency():
    """Get default center frequency"""
    return sdr_config.get('usrp.default_center_freq')

def get_default_sample_rate():
    """Get default sample rate"""
    return sdr_config.get('usrp.default_sample_rate')

def get_default_gain():
    """Get default gain"""
    return sdr_config.get('usrp.default_gain')

def get_gui_update_rate():
    """Get GUI update rate"""
    return sdr_config.get('gui.update_rate', 20)

def get_spectrum_fft_size():
    """Get spectrum FFT size"""
    return sdr_config.get('gui.spectrum_fft_size', 1024)


if __name__ == '__main__':
    # Test configuration
    print("SDR Configuration Test")
    print("=" * 30)

    config = SDRConfig()

    print(f"Device Args: {config.get('usrp.default_device_args')}")
    print(f"Center Freq: {config.get('usrp.default_center_freq')/1e6} MHz")
    print(f"Sample Rate: {config.get('usrp.default_sample_rate')/1e6} MS/s")
    print(f"GUI Update Rate: {config.get('gui.update_rate')} FPS")
    print(f"Supported Modes: {config.get('demodulation.supported_modes')}")

    # Test setting a value
    config.set('usrp.default_gain', 60)
    print(f"Updated Gain: {config.get('usrp.default_gain')} dB")
