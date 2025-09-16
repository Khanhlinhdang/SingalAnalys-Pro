"""
Configuration settings for RF Spectrum Analyzer
Manages application settings, SDR device parameters, and GUI preferences.
"""

import os
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SDRSettings:
    """SDR device configuration settings."""
    device_type: str = "rtlsdr"  # rtlsdr, hackrf, pluto, soapy, usrp
    center_frequency: float = 100e6  # Hz
    sample_rate: float = 2e6  # Hz
    gain: float = 20.0  # dB
    bandwidth: float = 2e6  # Hz
    antenna: str = "RX"
    ppm_error: int = 0
    bias_tee: bool = False
    agc: bool = False
    
    # Device-specific parameters
    rtlsdr_direct_sampling: int = 0
    hackrf_amp_enable: bool = False
    hackrf_lna_gain: float = 16.0
    hackrf_vga_gain: float = 16.0
    pluto_buffer_size: int = 1024


@dataclass
class DSPSettings:
    """Digital Signal Processing settings."""
    fft_size: int = 1024
    window_type: str = "hann"  # hann, hamming, blackman, bartlett
    overlap: float = 0.5  # 0.0 to 0.9
    averaging: int = 10
    
    # Filter settings
    filter_type: str = "bandpass"  # lowpass, highpass, bandpass, bandstop
    filter_order: int = 64
    filter_cutoff_low: float = 0.1  # Normalized frequency
    filter_cutoff_high: float = 0.9  # Normalized frequency
    
    # Resampling
    resample_factor: float = 1.0
    
    # Demodulation
    demod_type: str = "none"  # none, am, fm, pm, qpsk, bpsk
    
    # Modulation Analysis
    auto_detect_modulation: bool = False
    modulation_type: str = "Unknown"  # PSK, QPSK, 8PSK, QAM16, QAM64, QAM256, FSK, GFSK, MSK, OFDM, AM, FM
    symbol_rate: float = 1000.0  # Hz
    demodulation_enabled: bool = False
    
    # Encoding Analysis
    auto_detect_coding: bool = False
    encoding_type: str = "None"  # Hamming, BCH, Reed-Solomon, Convolutional, Turbo, LDPC, Polar
    code_rate: str = "1/2"  # 1/2, 2/3, 3/4, 5/6, 7/8
    decoding_enabled: bool = False


@dataclass
class GUISettings:
    """GUI configuration settings."""
    window_width: int = 1200
    window_height: int = 800
    window_x: int = 100
    window_y: int = 100
    
    # Spectrum display
    spectrum_min_db: float = -120.0
    spectrum_max_db: float = 0.0
    spectrum_ref_level: float = 0.0
    
    # Waterfall display
    waterfall_height: int = 200
    waterfall_colormap: str = "viridis"  # viridis, plasma, inferno, magma
    
    # Update rates
    spectrum_update_rate: float = 30.0  # Hz
    waterfall_update_rate: float = 10.0  # Hz
    
    # Colors and themes
    theme: str = "dark"  # dark, light
    grid_enabled: bool = True
    grid_alpha: float = 0.3
    
    # Control panel visibility
    controls_visible: bool = True
    spectrum_visible: bool = True
    waterfall_visible: bool = True
    constellation_visible: bool = False


@dataclass
class ProcessingSettings:
    """Signal processing pipeline settings."""
    enable_iq_correction: bool = True
    enable_dc_removal: bool = True
    enable_auto_gain: bool = False
    
    # Recording settings
    record_format: str = "complex64"  # complex64, float32, int16
    record_duration: float = 10.0  # seconds
    record_trigger_level: float = -50.0  # dB
    auto_record: bool = False
    
    # Analysis settings
    enable_peak_detection: bool = True
    peak_threshold: float = -60.0  # dB
    peak_min_distance: int = 10  # bins
    
    # Measurement settings
    measurement_bandwidth: float = 1000.0  # Hz
    integration_time: float = 1.0  # seconds


@dataclass
class NetworkSettings:
    """Network and communication settings."""
    enable_web_interface: bool = False
    web_port: int = 8080
    websocket_port: int = 8081
    
    # Remote control
    enable_remote_control: bool = False
    control_port: int = 5555
    
    # Data streaming
    enable_streaming: bool = False
    streaming_format: str = "json"  # json, binary
    streaming_port: int = 5556


class Settings:
    """Main settings class that manages all configuration."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize settings with default values."""
        self.sdr = SDRSettings()
        self.dsp = DSPSettings()
        self.gui = GUISettings()
        self.processing = ProcessingSettings()
        self.network = NetworkSettings()
        
        # Configuration file paths
        self.config_dir = Path.home() / ".rf_spectrum_analyzer"
        self.config_file = self.config_dir / "config.yaml"
        
        if config_file:
            self.config_file = Path(config_file)
        
        # Load settings if config file exists
        if self.config_file.exists():
            self.load_from_file(str(self.config_file))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "sdr": asdict(self.sdr),
            "dsp": asdict(self.dsp),
            "gui": asdict(self.gui),
            "processing": asdict(self.processing),
            "network": asdict(self.network)
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load settings from dictionary."""
        if "sdr" in data:
            self.sdr = SDRSettings(**data["sdr"])
        if "dsp" in data:
            self.dsp = DSPSettings(**data["dsp"])
        if "gui" in data:
            self.gui = GUISettings(**data["gui"])
        if "processing" in data:
            self.processing = ProcessingSettings(**data["processing"])
        if "network" in data:
            self.network = NetworkSettings(**data["network"])
    
    def save_to_file(self, filename: Optional[str] = None) -> None:
        """Save settings to YAML file."""
        if filename:
            config_path = Path(filename)
        else:
            config_path = self.config_file
        
        # Create config directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            logger.info(f"Settings saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save settings to {config_path}: {e}")
    
    def load_from_file(self, filename: str) -> None:
        """Load settings from YAML file."""
        config_path = Path(filename)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_path} does not exist")
            return
        
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if data:
                self.from_dict(data)
                logger.info(f"Settings loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load settings from {config_path}: {e}")
    
    def get_device_settings(self) -> Dict[str, Any]:
        """Get device-specific settings based on selected device type."""
        base_settings = {
            "center_freq": self.sdr.center_frequency,
            "sample_rate": self.sdr.sample_rate,
            "gain": self.sdr.gain,
            "bandwidth": self.sdr.bandwidth,
        }
        
        if self.sdr.device_type == "rtlsdr":
            base_settings.update({
                "ppm_error": self.sdr.ppm_error,
                "bias_tee": self.sdr.bias_tee,
                "agc": self.sdr.agc,
                "direct_sampling": self.sdr.rtlsdr_direct_sampling,
            })
        elif self.sdr.device_type == "hackrf":
            base_settings.update({
                "amp_enable": self.sdr.hackrf_amp_enable,
                "lna_gain": self.sdr.hackrf_lna_gain,
                "vga_gain": self.sdr.hackrf_vga_gain,
            })
        elif self.sdr.device_type == "pluto":
            base_settings.update({
                "buffer_size": self.sdr.pluto_buffer_size,
                "antenna": self.sdr.antenna,
            })
        
        return base_settings
    
    def validate_settings(self) -> bool:
        """Validate current settings for consistency."""
        try:
            # Validate frequency ranges
            if self.sdr.center_frequency <= 0:
                logger.error("Center frequency must be positive")
                return False
            
            if self.sdr.sample_rate <= 0:
                logger.error("Sample rate must be positive")
                return False
            
            # Validate FFT size (must be power of 2)
            if self.dsp.fft_size & (self.dsp.fft_size - 1) != 0:
                logger.error("FFT size must be power of 2")
                return False
            
            # Validate overlap
            if not 0.0 <= self.dsp.overlap <= 0.9:
                logger.error("Overlap must be between 0.0 and 0.9")
                return False
            
            # Validate filter cutoffs
            if self.dsp.filter_cutoff_low >= self.dsp.filter_cutoff_high:
                logger.error("Lower cutoff must be less than upper cutoff")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Settings validation error: {e}")
            return False


# Global settings instance
settings = Settings()