"""
PlutoSDR Backend Implementation
Interface for Analog Devices PlutoSDR devices.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List
from rf_spectrum_analyzer.core.sdr_backend import SDRBackend
from rf_spectrum_analyzer.config.settings import Settings

logger = logging.getLogger(__name__)


class PlutoSDRBackend(SDRBackend):
    """PlutoSDR backend implementation using pyadi-iio."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.sdr = None
        self.buffer_size = settings.sdr.pluto_buffer_size
        self.uri = "ip:192.168.2.1"  # Default PlutoSDR IP
        
    def connect(self) -> bool:
        """Connect to PlutoSDR device."""
        try:
            try:
                import adi
                self.adi = adi
            except ImportError:
                logger.error("PlutoSDR library not installed. Install with: pip install pyadi-iio")
                return False
            
            # Try to connect to PlutoSDR
            self.sdr = self.adi.Pluto(uri=self.uri)
            
            if self.sdr is None:
                logger.error("Failed to connect to PlutoSDR")
                return False
            
            # Get device information
            self._log_device_info()
            
            self.connected = True
            logger.info("PlutoSDR connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to PlutoSDR: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from PlutoSDR device."""
        try:
            if self.sdr:
                self.sdr = None
            
            self.connected = False
            logger.info("PlutoSDR disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting PlutoSDR: {e}")
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure PlutoSDR device."""
        if not self.sdr:
            return False
        
        try:
            # Set sample rate
            sample_rate = config.get("sample_rate", self.sample_rate)
            self.sdr.sample_rate = int(sample_rate)
            self.sample_rate = sample_rate
            logger.info(f"Sample rate set to {sample_rate/1e6:.2f} MHz")
            
            # Set center frequency
            frequency = config.get("center_freq", self.center_frequency)
            self.sdr.rx_lo = int(frequency)
            self.center_frequency = frequency
            logger.info(f"Center frequency set to {frequency/1e6:.2f} MHz")
            
            # Set RF bandwidth
            bandwidth = config.get("bandwidth", sample_rate)
            self.sdr.rx_rf_bandwidth = int(bandwidth)
            logger.info(f"RF bandwidth set to {bandwidth/1e6:.2f} MHz")
            
            # Set gain control mode
            gain = config.get("gain", self.gain)
            if config.get("agc", False):
                self.sdr.gain_control_mode_chan0 = "slow_attack"
                logger.info("AGC enabled (slow attack)")
            else:
                self.sdr.gain_control_mode_chan0 = "manual"
                self.sdr.rx_hardwaregain_chan0 = gain
                self.gain = gain
                logger.info(f"Manual gain set to {gain} dB")
            
            # Set buffer size
            buffer_size = config.get("buffer_size", self.buffer_size)
            self.sdr.rx_buffer_size = buffer_size
            self.buffer_size = buffer_size
            logger.info(f"Buffer size set to {buffer_size}")
            
            # Configure additional PlutoSDR-specific settings
            self._configure_advanced_settings(config)
            
            logger.info("PlutoSDR configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure PlutoSDR: {e}")
            return False
    
    def _configure_advanced_settings(self, config: Dict[str, Any]) -> None:
        """Configure advanced PlutoSDR settings."""
        try:
            # Set quadrature tracking
            if "quadrature_tracking" in config:
                self.sdr.rx_quadrature_tracking_en_chan0 = config["quadrature_tracking"]
            
            # Set RF DC tracking
            if "rf_dc_tracking" in config:
                self.sdr.rx_rf_dc_offset_tracking_en_chan0 = config["rf_dc_tracking"]
            
            # Set BB DC tracking
            if "bb_dc_tracking" in config:
                self.sdr.rx_bb_dc_offset_tracking_en_chan0 = config["bb_dc_tracking"]
            
            # Set LO frequency for advanced users
            if "lo_frequency" in config:
                self.sdr.rx_lo = int(config["lo_frequency"])
            
        except Exception as e:
            logger.warning(f"Could not set advanced PlutoSDR settings: {e}")
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read IQ samples from PlutoSDR."""
        if not self.sdr or not self.connected:
            return None
        
        try:
            # Adjust buffer size if needed
            if num_samples != self.buffer_size:
                self.sdr.rx_buffer_size = num_samples
                self.buffer_size = num_samples
            
            # Read samples
            samples = self.sdr.rx()
            
            if samples is None or len(samples) == 0:
                logger.warning("No samples received from PlutoSDR")
                return None
            
            # Convert to complex64 for consistency
            return samples.astype(np.complex64)
            
        except Exception as e:
            logger.error(f"Error reading PlutoSDR samples: {e}")
            return None
    
    def set_frequency(self, frequency: float) -> bool:
        """Set center frequency."""
        if not self.sdr:
            return False
        
        try:
            # Validate frequency range
            min_freq, max_freq = self.get_frequency_range()
            if not (min_freq <= frequency <= max_freq):
                logger.error(f"Frequency {frequency/1e6:.3f} MHz out of range "
                           f"({min_freq/1e6:.1f} - {max_freq/1e6:.1f} MHz)")
                return False
            
            self.sdr.rx_lo = int(frequency)
            self.center_frequency = frequency
            logger.debug(f"Frequency set to {frequency/1e6:.3f} MHz")
            return True
            
        except Exception as e:
            logger.error(f"Error setting PlutoSDR frequency: {e}")
            return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate."""
        if not self.sdr:
            return False
        
        try:
            # Validate sample rate
            min_rate, max_rate = self.get_sample_rate_range()
            if not (min_rate <= sample_rate <= max_rate):
                logger.error(f"Sample rate {sample_rate/1e6:.3f} MHz out of range "
                           f"({min_rate/1e6:.1f} - {max_rate/1e6:.1f} MHz)")
                return False
            
            self.sdr.sample_rate = int(sample_rate)
            self.sample_rate = sample_rate
            logger.debug(f"Sample rate set to {sample_rate/1e6:.3f} MHz")
            return True
            
        except Exception as e:
            logger.error(f"Error setting PlutoSDR sample rate: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """Set RF gain."""
        if not self.sdr:
            return False
        
        try:
            # Validate gain range
            min_gain, max_gain = self.get_gain_range()
            gain = max(min_gain, min(max_gain, gain))
            
            # Set manual gain mode first
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = gain
            self.gain = gain
            
            logger.debug(f"Gain set to {gain} dB")
            return True
            
        except Exception as e:
            logger.error(f"Error setting PlutoSDR gain: {e}")
            return False
    
    def set_bandwidth(self, bandwidth: float) -> bool:
        """Set RF bandwidth."""
        if not self.sdr:
            return False
        
        try:
            self.sdr.rx_rf_bandwidth = int(bandwidth)
            logger.debug(f"Bandwidth set to {bandwidth/1e6:.3f} MHz")
            return True
            
        except Exception as e:
            logger.error(f"Error setting PlutoSDR bandwidth: {e}")
            return False
    
    def set_agc(self, mode: str = "slow_attack") -> bool:
        """Set automatic gain control mode."""
        if not self.sdr:
            return False
        
        try:
            valid_modes = ["manual", "slow_attack", "fast_attack", "hybrid"]
            if mode not in valid_modes:
                logger.error(f"Invalid AGC mode: {mode}. Valid modes: {valid_modes}")
                return False
            
            self.sdr.gain_control_mode_chan0 = mode
            logger.debug(f"AGC mode set to {mode}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting PlutoSDR AGC: {e}")
            return False
    
    def _log_device_info(self) -> None:
        """Log PlutoSDR device information."""
        if not self.sdr:
            return
        
        try:
            logger.info("PlutoSDR Device Information:")
            
            # Try to get firmware version and other info
            if hasattr(self.sdr, '_ctx'):
                ctx = self.sdr._ctx
                if hasattr(ctx, 'description'):
                    logger.info(f"  Description: {ctx.description}")
                if hasattr(ctx, 'name'):
                    logger.info(f"  Name: {ctx.name}")
            
            logger.info(f"  URI: {self.uri}")
            
        except Exception as e:
            logger.warning(f"Could not read PlutoSDR device info: {e}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive PlutoSDR device information."""
        if not self.sdr:
            return {"device_type": "PlutoSDR", "status": "disconnected"}
        
        try:
            info = {
                "device_type": "PlutoSDR",
                "center_freq": self.center_frequency,
                "sample_rate": self.sample_rate,
                "gain": self.gain,
                "buffer_size": self.buffer_size,
                "uri": self.uri,
                "status": "connected"
            }
            
            # Add device-specific information
            try:
                info["gain_control_mode"] = self.sdr.gain_control_mode_chan0
                info["rf_bandwidth"] = self.sdr.rx_rf_bandwidth
                
                if hasattr(self.sdr, '_ctx'):
                    ctx = self.sdr._ctx
                    if hasattr(ctx, 'description'):
                        info["description"] = ctx.description
            
            except Exception:
                pass
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting PlutoSDR device info: {e}")
            return {"device_type": "PlutoSDR", "status": "error", "error": str(e)}
    
    def get_frequency_range(self) -> tuple:
        """Get supported frequency range for PlutoSDR."""
        # PlutoSDR frequency range
        return (70e6, 6000e6)  # 70 MHz to 6 GHz
    
    def get_sample_rate_range(self) -> tuple:
        """Get supported sample rate range."""
        # PlutoSDR sample rate range
        return (0.065e6, 61.44e6)  # 65 kHz to 61.44 MHz
    
    def get_gain_range(self) -> tuple:
        """Get supported gain range."""
        # PlutoSDR gain range
        return (-3, 71)  # -3 to 71 dB
    
    def get_available_sample_rates(self) -> List[float]:
        """Get list of commonly used sample rates."""
        return [
            0.1e6, 0.2e6, 0.5e6, 1e6, 2e6, 4e6, 
            5e6, 10e6, 20e6, 30e6, 40e6, 50e6, 61.44e6
        ]
    
    def get_available_bandwidths(self) -> List[float]:
        """Get list of available RF bandwidths."""
        return [
            0.2e6, 0.5e6, 1e6, 2e6, 3e6, 4e6, 5e6, 
            6e6, 7e6, 8e6, 9e6, 10e6, 12e6, 14e6, 
            18e6, 20e6, 24e6, 28e6, 36e6, 40e6, 56e6
        ]