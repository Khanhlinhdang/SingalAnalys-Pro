"""
SoapySDR Backend Implementation
Provides comprehensive interface to SoapySDR for multiple hardware types.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List
from rf_spectrum_analyzer.core.sdr_backend import SDRBackend
from rf_spectrum_analyzer.config.settings import Settings

logger = logging.getLogger(__name__)


class SoapySDRBackend(SDRBackend):
    """Enhanced SoapySDR backend with full streaming support."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.device = None
        self.rx_stream = None
        self.stream_active = False
        
    def connect(self) -> bool:
        """Connect to SoapySDR device."""
        try:
            import SoapySDR
            
            # Enumerate devices
            devices = SoapySDR.Device.enumerate()
            if not devices:
                logger.error("No SoapySDR devices found")
                return False
            
            logger.info(f"Found {len(devices)} SoapySDR device(s)")
            for i, dev in enumerate(devices):
                logger.info(f"  Device {i}: {dev}")
            
            # Create device instance (use first available)
            self.device = SoapySDR.Device(devices[0])
            
            # Query device capabilities
            self._log_device_capabilities()
            
            self.connected = True
            logger.info("SoapySDR device connected successfully")
            return True
            
        except ImportError:
            logger.error("SoapySDR library not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to SoapySDR device: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from SoapySDR device."""
        try:
            if self.rx_stream:
                self._stop_stream()
            
            if self.device:
                self.device = None
                
            self.connected = False
            logger.info("SoapySDR device disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting SoapySDR device: {e}")
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure SoapySDR device with comprehensive settings."""
        if not self.device:
            return False
        
        try:
            import SoapySDR
            
            # Set sample rate
            sample_rate = config.get("sample_rate", self.sample_rate)
            self.device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, sample_rate)
            self.sample_rate = sample_rate
            logger.info(f"Sample rate set to {sample_rate/1e6:.2f} MHz")
            
            # Set center frequency
            frequency = config.get("center_freq", self.center_frequency)
            self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, frequency)
            self.center_frequency = frequency
            logger.info(f"Center frequency set to {frequency/1e6:.2f} MHz")
            
            # Set gain
            gain = config.get("gain", self.gain)
            if self.device.hasGainMode(SoapySDR.SOAPY_SDR_RX, 0):
                # Use automatic gain control if available
                agc = config.get("agc", False)
                self.device.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, agc)
                if not agc:
                    self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
            else:
                self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
            self.gain = gain
            logger.info(f"Gain set to {gain} dB")
            
            # Set bandwidth if specified
            bandwidth = config.get("bandwidth", self.sample_rate)
            if bandwidth and hasattr(self.device, 'setBandwidth'):
                self.device.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, bandwidth)
                logger.info(f"Bandwidth set to {bandwidth/1e6:.2f} MHz")
            
            # Set antenna if specified
            antenna = config.get("antenna")
            if antenna:
                antennas = self.device.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)
                if antenna in antennas:
                    self.device.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, antenna)
                    logger.info(f"Antenna set to {antenna}")
                else:
                    logger.warning(f"Antenna {antenna} not available. Available: {antennas}")
            
            # Setup RX stream
            self._setup_stream()
            
            logger.info("SoapySDR device configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure SoapySDR device: {e}")
            return False
    
    def _setup_stream(self) -> bool:
        """Setup streaming for continuous reception."""
        try:
            import SoapySDR
            
            if self.rx_stream:
                self._stop_stream()
            
            # Create RX stream
            self.rx_stream = self.device.setupStream(
                SoapySDR.SOAPY_SDR_RX, 
                SoapySDR.SOAPY_SDR_CF32,  # Complex float32
                [0]  # Channel 0
            )
            
            # Activate stream
            self.device.activateStream(self.rx_stream)
            self.stream_active = True
            
            logger.info("SoapySDR stream setup successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup SoapySDR stream: {e}")
            return False
    
    def _stop_stream(self) -> None:
        """Stop streaming."""
        try:
            if self.device and self.rx_stream:
                if self.stream_active:
                    self.device.deactivateStream(self.rx_stream)
                    self.stream_active = False
                
                self.device.closeStream(self.rx_stream)
                self.rx_stream = None
                
                logger.info("SoapySDR stream stopped")
                
        except Exception as e:
            logger.error(f"Error stopping SoapySDR stream: {e}")
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read IQ samples from SoapySDR device."""
        if not self.device or not self.rx_stream or not self.stream_active:
            return None
        
        try:
            import SoapySDR
            
            # Allocate buffer
            buffer = np.empty(num_samples, dtype=np.complex64)
            
            # Read samples with timeout
            sr = self.device.readStream(
                self.rx_stream, 
                [buffer], 
                num_samples,
                timeoutUs=1000000  # 1 second timeout
            )
            
            if sr.ret > 0:
                return buffer[:sr.ret]
            else:
                # Handle error codes
                if sr.ret == SoapySDR.SOAPY_SDR_TIMEOUT:
                    logger.debug("SoapySDR read timeout")
                elif sr.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
                    logger.warning("SoapySDR buffer overflow")
                elif sr.ret == SoapySDR.SOAPY_SDR_UNDERFLOW:
                    logger.warning("SoapySDR buffer underflow")
                else:
                    logger.error(f"SoapySDR read error: {sr.ret}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading SoapySDR samples: {e}")
            return None
    
    def set_frequency(self, frequency: float) -> bool:
        """Set center frequency."""
        if not self.device:
            return False
        
        try:
            import SoapySDR
            self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, frequency)
            self.center_frequency = frequency
            logger.debug(f"Frequency set to {frequency/1e6:.3f} MHz")
            return True
        except Exception as e:
            logger.error(f"Error setting SoapySDR frequency: {e}")
            return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate."""
        if not self.device:
            return False
        
        try:
            import SoapySDR
            
            # Stop stream before changing sample rate
            was_active = self.stream_active
            if was_active:
                self._stop_stream()
            
            self.device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, sample_rate)
            self.sample_rate = sample_rate
            
            # Restart stream if it was active
            if was_active:
                self._setup_stream()
            
            logger.debug(f"Sample rate set to {sample_rate/1e6:.3f} MHz")
            return True
        except Exception as e:
            logger.error(f"Error setting SoapySDR sample rate: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """Set RF gain."""
        if not self.device:
            return False
        
        try:
            import SoapySDR
            self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
            self.gain = gain
            logger.debug(f"Gain set to {gain} dB")
            return True
        except Exception as e:
            logger.error(f"Error setting SoapySDR gain: {e}")
            return False
    
    def _log_device_capabilities(self) -> None:
        """Log device capabilities for debugging."""
        if not self.device:
            return
        
        try:
            import SoapySDR
            
            # Driver info
            driver = self.device.getDriverKey()
            hardware = self.device.getHardwareKey()
            logger.info(f"Device: {driver} - {hardware}")
            
            # Frequency ranges
            freq_ranges = self.device.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0)
            logger.info(f"Frequency ranges: {freq_ranges}")
            
            # Sample rate ranges
            rate_ranges = self.device.getSampleRateRange(SoapySDR.SOAPY_SDR_RX, 0)
            logger.info(f"Sample rate ranges: {rate_ranges}")
            
            # Gain ranges
            gain_range = self.device.getGainRange(SoapySDR.SOAPY_SDR_RX, 0)
            logger.info(f"Gain range: {gain_range}")
            
            # Available antennas
            antennas = self.device.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)
            logger.info(f"Available antennas: {antennas}")
            
        except Exception as e:
            logger.error(f"Error querying device capabilities: {e}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        if not self.device:
            return {"device_type": "SoapySDR", "status": "disconnected"}
        
        try:
            import SoapySDR
            
            info = {
                "device_type": "SoapySDR",
                "driver": self.device.getDriverKey(),
                "hardware": self.device.getHardwareKey(),
                "center_freq": self.center_frequency,
                "sample_rate": self.sample_rate,
                "gain": self.gain,
                "stream_active": self.stream_active,
                "status": "connected"
            }
            
            # Add frequency ranges
            freq_ranges = self.device.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0)
            if freq_ranges:
                info["freq_range_min"] = freq_ranges[0].minimum()
                info["freq_range_max"] = freq_ranges[0].maximum()
            
            # Add sample rate info
            rate_ranges = self.device.getSampleRateRange(SoapySDR.SOAPY_SDR_RX, 0)
            if rate_ranges:
                info["rate_range_min"] = rate_ranges[0].minimum()
                info["rate_range_max"] = rate_ranges[0].maximum()
            
            # Add gain info
            gain_range = self.device.getGainRange(SoapySDR.SOAPY_SDR_RX, 0)
            if gain_range:
                info["gain_range_min"] = gain_range.minimum()
                info["gain_range_max"] = gain_range.maximum()
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting SoapySDR device info: {e}")
            return {"device_type": "SoapySDR", "status": "error", "error": str(e)}
    
    def get_available_sample_rates(self) -> List[float]:
        """Get list of available sample rates."""
        if not self.device:
            return []
        
        try:
            import SoapySDR
            
            # Get discrete sample rates if available
            rates = self.device.listSampleRates(SoapySDR.SOAPY_SDR_RX, 0)
            if rates:
                return sorted(rates)
            
            # Otherwise return common rates within range
            rate_ranges = self.device.getSampleRateRange(SoapySDR.SOAPY_SDR_RX, 0)
            if rate_ranges:
                min_rate = rate_ranges[0].minimum()
                max_rate = rate_ranges[0].maximum()
                
                # Common sample rates
                common_rates = [
                    0.25e6, 0.5e6, 1e6, 2e6, 2.4e6, 3.2e6,
                    5e6, 8e6, 10e6, 20e6, 25e6, 40e6, 50e6
                ]
                
                return [rate for rate in common_rates 
                       if min_rate <= rate <= max_rate]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting sample rates: {e}")
            return []