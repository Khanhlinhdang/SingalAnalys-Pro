"""
RTL-SDR Backend Implementation
Enhanced interface for RTL-SDR dongles with comprehensive configuration.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List
from rf_spectrum_analyzer.core.sdr_backend import SDRBackend
from rf_spectrum_analyzer.config.settings import Settings

logger = logging.getLogger(__name__)


class RTLSDRBackend(SDRBackend):
    """Enhanced RTL-SDR backend implementation."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.sdr = None
        self.device_index = 0
        
    def connect(self) -> bool:
        """Connect to RTL-SDR device."""
        try:
            from rtlsdr import RtlSdr
            
            # Get available devices
            device_count = RtlSdr.get_device_count()
            if device_count == 0:
                logger.error("No RTL-SDR devices found")
                return False
            
            logger.info(f"Found {device_count} RTL-SDR device(s)")
            
            # List available devices
            for i in range(device_count):
                try:
                    name = RtlSdr.get_device_name(i)
                    logger.info(f"  Device {i}: {name}")
                except:
                    logger.info(f"  Device {i}: Unknown")
            
            # Connect to first available device
            self.sdr = RtlSdr(device_index=self.device_index)
            
            # Log device information
            self._log_device_info()
            
            self.connected = True
            logger.info("RTL-SDR connected successfully")
            return True
            
        except ImportError:
            logger.error("RTL-SDR library not installed. Install with: pip install pyrtlsdr")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to RTL-SDR: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from RTL-SDR device."""
        if self.sdr:
            try:
                self.sdr.close()
                self.connected = False
                logger.info("RTL-SDR disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting RTL-SDR: {e}")
            finally:
                self.sdr = None
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure RTL-SDR device with comprehensive settings."""
        if not self.sdr:
            return False
        
        try:
            # Set center frequency
            frequency = config.get("center_freq", self.center_frequency)
            self.sdr.center_freq = int(frequency)
            self.center_frequency = frequency
            logger.info(f"Center frequency set to {frequency/1e6:.3f} MHz")
            
            # Set sample rate
            sample_rate = config.get("sample_rate", self.sample_rate)
            self.sdr.sample_rate = int(sample_rate)
            self.sample_rate = sample_rate
            logger.info(f"Sample rate set to {sample_rate/1e6:.3f} MHz")
            
            # Set gain
            gain = config.get("gain", self.gain)
            if config.get("agc", False):
                # Enable automatic gain control
                self.sdr.gain = 'auto'
                logger.info("AGC enabled")
            else:
                # Set manual gain
                valid_gains = self.sdr.get_gains()
                if valid_gains:
                    # Find closest valid gain
                    closest_gain = min(valid_gains, key=lambda x: abs(x - gain*10))
                    self.sdr.gain = closest_gain / 10.0  # RTL-SDR uses tenths of dB
                    self.gain = closest_gain / 10.0
                    logger.info(f"Gain set to {self.gain:.1f} dB")
                else:
                    self.sdr.gain = gain
                    self.gain = gain
                    logger.info(f"Gain set to {gain:.1f} dB")
            
            # Set frequency correction (PPM)
            ppm_error = config.get("ppm_error", 0)
            if ppm_error != 0:
                self.sdr.freq_correction = ppm_error
                logger.info(f"Frequency correction set to {ppm_error} PPM")
            
            # Set bias tee (if supported)
            bias_tee = config.get("bias_tee", False)
            if hasattr(self.sdr, 'set_bias_tee'):
                try:
                    self.sdr.set_bias_tee(bias_tee)
                    logger.info(f"Bias tee {'enabled' if bias_tee else 'disabled'}")
                except:
                    logger.warning("Bias tee control not supported by this device")
            
            # Set direct sampling mode
            direct_sampling = config.get("direct_sampling", 0)
            if direct_sampling in [0, 1, 2]:
                try:
                    self.sdr.set_direct_sampling(direct_sampling)
                    sampling_modes = ["Off", "I-ADC", "Q-ADC"]
                    logger.info(f"Direct sampling: {sampling_modes[direct_sampling]}")
                except:
                    logger.warning("Direct sampling control not supported")
            
            # Set bandwidth if supported
            bandwidth = config.get("bandwidth")
            if bandwidth and hasattr(self.sdr, 'bandwidth'):
                try:
                    self.sdr.bandwidth = int(bandwidth)
                    logger.info(f"Bandwidth set to {bandwidth/1e6:.3f} MHz")
                except:
                    logger.warning("Bandwidth control not supported")
            
            logger.info("RTL-SDR configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure RTL-SDR: {e}")
            return False
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read IQ samples from RTL-SDR with error handling."""
        if not self.sdr or not self.connected:
            return None
        
        try:
            # Read samples with timeout protection
            samples = self.sdr.read_samples(num_samples)
            
            if samples is None or len(samples) == 0:
                logger.warning("No samples received from RTL-SDR")
                return None
            
            # Convert to complex64 for consistency
            return samples.astype(np.complex64)
            
        except Exception as e:
            logger.error(f"Error reading RTL-SDR samples: {e}")
            return None
    
    def read_samples_async(self, num_samples: int, callback) -> bool:
        """Start asynchronous reading (if supported)."""
        if not self.sdr:
            return False
        
        try:
            if hasattr(self.sdr, 'read_samples_async'):
                self.sdr.read_samples_async(callback, num_samples)
                return True
            else:
                logger.warning("Async reading not supported by RTL-SDR library version")
                return False
        except Exception as e:
            logger.error(f"Error starting async read: {e}")
            return False
    
    def cancel_read_async(self) -> None:
        """Cancel asynchronous reading."""
        if self.sdr and hasattr(self.sdr, 'cancel_read_async'):
            try:
                self.sdr.cancel_read_async()
            except Exception as e:
                logger.error(f"Error canceling async read: {e}")
    
    def set_frequency(self, frequency: float) -> bool:
        """Set center frequency with validation."""
        if not self.sdr:
            return False
        
        try:
            # Validate frequency range
            min_freq, max_freq = self.get_frequency_range()
            if not (min_freq <= frequency <= max_freq):
                logger.error(f"Frequency {frequency/1e6:.3f} MHz out of range "
                           f"({min_freq/1e6:.1f} - {max_freq/1e6:.1f} MHz)")
                return False
            
            self.sdr.center_freq = int(frequency)
            self.center_frequency = frequency
            logger.debug(f"Frequency set to {frequency/1e6:.3f} MHz")
            return True
            
        except Exception as e:
            logger.error(f"Error setting RTL-SDR frequency: {e}")
            return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate with validation."""
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
            logger.error(f"Error setting RTL-SDR sample rate: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """Set RF gain with validation."""
        if not self.sdr:
            return False
        
        try:
            valid_gains = self.sdr.get_gains()
            if valid_gains:
                # Find closest valid gain (RTL-SDR uses tenths of dB)
                gain_tenths = gain * 10
                closest_gain = min(valid_gains, key=lambda x: abs(x - gain_tenths))
                self.sdr.gain = closest_gain / 10.0
                self.gain = closest_gain / 10.0
                logger.debug(f"Gain set to {self.gain:.1f} dB")
            else:
                self.sdr.gain = gain
                self.gain = gain
                logger.debug(f"Gain set to {gain:.1f} dB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting RTL-SDR gain: {e}")
            return False
    
    def set_agc(self, enable: bool) -> bool:
        """Enable/disable automatic gain control."""
        if not self.sdr:
            return False
        
        try:
            if enable:
                self.sdr.gain = 'auto'
            else:
                # Set to current manual gain
                self.sdr.gain = self.gain
            
            logger.debug(f"AGC {'enabled' if enable else 'disabled'}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting RTL-SDR AGC: {e}")
            return False
    
    def _log_device_info(self) -> None:
        """Log detailed device information."""
        if not self.sdr:
            return
        
        try:
            logger.info(f"RTL-SDR Device Information:")
            logger.info(f"  Tuner type: {self.sdr.get_tuner_type()}")
            
            valid_gains = self.sdr.get_gains()
            if valid_gains:
                gains_db = [g/10.0 for g in valid_gains]
                logger.info(f"  Available gains: {gains_db} dB")
            
            # Try to get device name
            try:
                from rtlsdr import RtlSdr
                device_name = RtlSdr.get_device_name(self.device_index)
                logger.info(f"  Device name: {device_name}")
            except:
                pass
            
        except Exception as e:
            logger.warning(f"Could not read device info: {e}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive RTL-SDR device information."""
        if not self.sdr:
            return {"device_type": "RTL-SDR", "status": "disconnected"}
        
        try:
            info = {
                "device_type": "RTL-SDR",
                "center_freq": self.center_frequency,
                "sample_rate": self.sample_rate,
                "gain": self.gain,
                "status": "connected"
            }
            
            # Add device-specific information
            try:
                info["tuner_type"] = self.sdr.get_tuner_type()
                
                valid_gains = self.sdr.get_gains()
                if valid_gains:
                    info["available_gains"] = [g/10.0 for g in valid_gains]
                
                from rtlsdr import RtlSdr
                info["device_name"] = RtlSdr.get_device_name(self.device_index)
                
                # Add frequency correction if set
                if hasattr(self.sdr, 'freq_correction'):
                    info["freq_correction"] = self.sdr.freq_correction
                
            except Exception:
                pass  # Don't fail if we can't get additional info
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting RTL-SDR device info: {e}")
            return {"device_type": "RTL-SDR", "status": "error", "error": str(e)}
    
    def get_frequency_range(self) -> tuple:
        """Get supported frequency range for RTL-SDR."""
        # Typical RTL-SDR frequency range (varies by tuner)
        return (24e6, 1766e6)  # 24 MHz to 1.766 GHz
    
    def get_sample_rate_range(self) -> tuple:
        """Get supported sample rate range."""
        # RTL-SDR sample rate range
        return (0.225e6, 3.2e6)  # 225 kHz to 3.2 MHz
    
    def get_available_sample_rates(self) -> List[float]:
        """Get list of commonly used sample rates."""
        return [
            0.25e6, 0.5e6, 1.024e6, 1.4e6, 1.8e6, 
            1.92e6, 2.048e6, 2.4e6, 2.8e6, 3.2e6
        ]
    
    def get_available_gains(self) -> List[float]:
        """Get list of available gain values."""
        if not self.sdr:
            return []
        
        try:
            gains = self.sdr.get_gains()
            return [g/10.0 for g in gains] if gains else []
        except:
            return []