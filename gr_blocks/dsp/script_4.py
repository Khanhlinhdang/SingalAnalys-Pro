# Create config/settings.py - Application configuration management
settings_content = '''"""
Application Configuration Management

Handles loading, saving, and managing application settings for the RF Spectrum Analyzer.
Supports YAML configuration files and command-line overrides.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class SDRConfig:
    """SDR device configuration"""
    source: str = "auto"
    center_freq: float = 433.92e6
    sample_rate: float = 2.0e6
    gain: float = 30.0
    bandwidth: Optional[float] = None
    device_args: Dict[str, Any] = field(default_factory=dict)
    antenna: str = "RX"
    auto_gain: bool = False


@dataclass
class DSPConfig:
    """Digital Signal Processing configuration"""
    fft_size: int = 2048
    overlap: float = 0.5
    window: str = "hanning"
    averaging: int = 10
    enable_filtering: bool = True
    filter_type: str = "lowpass"
    filter_cutoff: float = 1.0e6
    filter_order: int = 51
    enable_demodulation: bool = False
    demod_type: str = "FM"


@dataclass
class GUIConfig:
    """GUI configuration"""
    theme: str = "auto"
    window_width: int = 1200
    window_height: int = 800
    update_rate: int = 30  # FPS
    waterfall_height: int = 200
    spectrum_height: int = 300
    show_constellation: bool = True
    show_iq_plot: bool = True
    font_size: int = 10


@dataclass
class ProcessingConfig:
    """Signal processing configuration"""
    enable_real_time: bool = True
    buffer_size: int = 4096
    num_buffers: int = 16
    processing_threads: int = 2
    enable_recording: bool = False
    record_format: str = "sigmf"
    snapshot_trigger: str = "manual"
    detection_threshold: float = -60.0  # dBm


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file: str = "logs/rf_spectrum.log"
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True


class AppSettings:
    """Main application settings manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Default configuration file
        if config_file is None:
            config_file = Path(__file__).parent / "default_config.yaml"
        
        self.config_file = Path(config_file)
        
        # Initialize configuration sections
        self.sdr = SDRConfig()
        self.dsp = DSPConfig()
        self.gui = GUIConfig()
        self.processing = ProcessingConfig()
        self.logging = LoggingConfig()
        
        # Load configuration
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
                
                # Update configuration sections
                self._update_from_dict(config_data)
                self.logger.info(f"Configuration loaded from {self.config_file}")
            else:
                self.logger.info("No configuration file found, using defaults")
                self.save_config()  # Save default configuration
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration")
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary
            config_data = {
                'sdr': asdict(self.sdr),
                'dsp': asdict(self.dsp),
                'gui': asdict(self.gui),
                'processing': asdict(self.processing),
                'logging': asdict(self.logging)
            }
            
            # Save to YAML file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        if 'sdr' in config_data:
            self._update_dataclass(self.sdr, config_data['sdr'])
        
        if 'dsp' in config_data:
            self._update_dataclass(self.dsp, config_data['dsp'])
        
        if 'gui' in config_data:
            self._update_dataclass(self.gui, config_data['gui'])
        
        if 'processing' in config_data:
            self._update_dataclass(self.processing, config_data['processing'])
        
        if 'logging' in config_data:
            self._update_dataclass(self.logging, config_data['logging'])
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]) -> None:
        """Update dataclass fields from dictionary"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def update_from_args(self, args) -> None:
        """Update configuration from command line arguments"""
        # SDR configuration
        if hasattr(args, 'source') and args.source:
            self.sdr.source = args.source
        if hasattr(args, 'freq') and args.freq:
            self.sdr.center_freq = args.freq
        if hasattr(args, 'samplerate') and args.samplerate:
            self.sdr.sample_rate = args.samplerate
        if hasattr(args, 'gain') and args.gain is not None:
            self.sdr.gain = args.gain
        if hasattr(args, 'bandwidth') and args.bandwidth:
            self.sdr.bandwidth = args.bandwidth
        
        # DSP configuration
        if hasattr(args, 'fft_size') and args.fft_size:
            self.dsp.fft_size = args.fft_size
        if hasattr(args, 'overlap') and args.overlap is not None:
            self.dsp.overlap = args.overlap
        
        # GUI configuration
        if hasattr(args, 'theme') and args.theme:
            self.gui.theme = args.theme
        
        # Logging configuration
        if hasattr(args, 'log_level') and args.log_level:
            self.logging.level = args.log_level
        if hasattr(args, 'log_file') and args.log_file:
            self.logging.file = args.log_file
    
    def get_sdr_params(self) -> Dict[str, Any]:
        """Get SDR parameters as dictionary"""
        return asdict(self.sdr)
    
    def get_dsp_params(self) -> Dict[str, Any]:
        """Get DSP parameters as dictionary"""
        return asdict(self.dsp)
    
    def set_theme(self, theme: str) -> None:
        """Set GUI theme"""
        if theme in ["light", "dark", "auto"]:
            self.gui.theme = theme
            self.logger.info(f"Theme set to: {theme}")
        else:
            self.logger.warning(f"Invalid theme: {theme}")
    
    def validate_config(self) -> bool:
        """Validate configuration values"""
        valid = True
        
        # Validate SDR config
        if self.sdr.sample_rate <= 0:
            self.logger.error("Invalid sample rate")
            valid = False
        
        if self.sdr.center_freq <= 0:
            self.logger.error("Invalid center frequency")
            valid = False
        
        # Validate DSP config
        if self.dsp.fft_size <= 0 or (self.dsp.fft_size & (self.dsp.fft_size - 1)) != 0:
            self.logger.error("FFT size must be a power of 2")
            valid = False
        
        if not 0 <= self.dsp.overlap < 1:
            self.logger.error("Overlap must be between 0 and 1")
            valid = False
        
        # Validate GUI config
        if self.gui.update_rate <= 0:
            self.logger.error("Update rate must be positive")
            valid = False
        
        return valid
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to defaults"""
        self.sdr = SDRConfig()
        self.dsp = DSPConfig()
        self.gui = GUIConfig()
        self.processing = ProcessingConfig()
        self.logging = LoggingConfig()
        
        self.logger.info("Configuration reset to defaults")
    
    def __str__(self) -> str:
        """String representation of settings"""
        return f"""RF Spectrum Analyzer Settings:
SDR: {self.sdr}
DSP: {self.dsp}
GUI: {self.gui}
Processing: {self.processing}
Logging: {self.logging}"""


def create_default_config_file():
    """Create a default configuration file"""
    config_dir = Path(__file__).parent
    config_file = config_dir / "default_config.yaml"
    
    # Create default settings
    settings = AppSettings()
    
    # Save default configuration
    with open(config_file, 'w', encoding='utf-8') as f:
        config_data = {
            'sdr': asdict(settings.sdr),
            'dsp': asdict(settings.dsp),
            'gui': asdict(settings.gui),
            'processing': asdict(settings.processing),
            'logging': asdict(settings.logging)
        }
        yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration file created: {config_file}")


if __name__ == "__main__":
    # Create default configuration file when run directly
    create_default_config_file()
'''

with open("rf_spectrum_analyzer/config/settings.py", "w") as f:
    f.write(settings_content)

print("Created config/settings.py")