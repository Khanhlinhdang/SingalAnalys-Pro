"""
HackRF Backend Implementation
Interface for HackRF One SDR devices.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List
from rf_spectrum_analyzer.core.sdr_backend import SDRBackend
from rf_spectrum_analyzer.config.settings import Settings

logger = logging.getLogger(__name__)


class HackRFBackend(SDRBackend):
    """HackRF One backend implementation."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.device = None
        self.is_streaming = False
        self.amp_enable = settings.sdr.hackrf_amp_enable
        self.lna_gain = settings.sdr.hackrf_lna_gain
        self.vga_gain = settings.sdr.hackrf_vga_gain
        
    def connect(self) -> bool:
        """Connect to HackRF device."""
        try:
            # Try importing hackrf library
            try:
                import hackrf
                self.hackrf = hackrf
            except ImportError:
                logger.error("HackRF library not installed. Install with: pip install hackrf")
                return False
            
            # Initialize and open device
            result = self.hackrf.init()
            if result != self.hackrf.HackrfError.HACKRF_SUCCESS:
                logger.error(f"Failed to initialize HackRF: {result}")
                return False
            
            self.device = self.hackrf.open()
            if self.device is None:
                logger.error("Failed to open HackRF device")
                return False
            
            # Get device info
            board_id = self.hackrf.board_id_read(self.device)
            version = self.hackrf.version_string_read(self.device)
            serial = self.hackrf.board_partid_serialno_read(self.device)
            
            logger.info(f"HackRF connected - Board ID: {board_id}, Version: {version}")
            logger.info(f"Serial: {serial}")
            
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to HackRF: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from HackRF device."""
        try:
            if self.is_streaming:
                self._stop_rx()
            
            if self.device:
                self.hackrf.close(self.device)
                self.device = None
            
            self.hackrf.exit()
            self.connected = False
            logger.info("HackRF disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting HackRF: {e}")
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure HackRF device."""
        if not self.device:
            return False
        
        try:
            # Set sample rate
            sample_rate = config.get("sample_rate", self.sample_rate)
            result = self.hackrf.set_sample_rate(self.device, sample_rate)
            if result != self.hackrf.HackrfError.HACKRF_SUCCESS:
                logger.error(f"Failed to set sample rate: {result}")
                return False
            self.sample_rate = sample_rate
            logger.info(f"Sample rate set to {sample_rate/1e6:.2f} MHz")
            
            # Set center frequency
            frequency = config.get("center_freq", self.center_frequency)
            result = self.hackrf.set_freq(self.device, int(frequency))
            if result != self.hackrf.HackrfError.HACKRF_SUCCESS:
                logger.error(f"Failed to set frequency: {result}")
                return False
            self.center_frequency = frequency
            logger.info(f"Center frequency set to {frequency/1e6:.2f} MHz")
            
            # Set LNA gain
            lna_gain = config.get("lna_gain", self.lna_gain)
            result = self.hackrf.set_lna_gain(self.device, int(lna_gain))
            if result != self.hackrf.HackrfError.HACKRF_SUCCESS:
                logger.error(f"Failed to set LNA gain: {result}")
                return False
            self.lna_gain = lna_gain
            logger.info(f"LNA gain set to {lna_gain} dB")
            
            # Set VGA gain
            vga_gain = config.get("vga_gain", self.vga_gain)
            result = self.hackrf.set_vga_gain(self.device, int(vga_gain))
            if result != self.hackrf.HackrfError.HACKRF_SUCCESS:
                logger.error(f"Failed to set VGA gain: {result}")
                return False
            self.vga_gain = vga_gain
            logger.info(f"VGA gain set to {vga_gain} dB")
            
            # Set amplifier
            amp_enable = config.get("amp_enable", self.amp_enable)
            result = self.hackrf.set_amp_enable(self.device, 1 if amp_enable else 0)
            if result != self.hackrf.HackrfError.HACKRF_SUCCESS:
                logger.error(f"Failed to set amplifier: {result}")
                return False
            self.amp_enable = amp_enable
            logger.info(f"Amplifier {'enabled' if amp_enable else 'disabled'}")
            
            # Set antenna power (bias tee)
            bias_tee = config.get("bias_tee", False)
            result = self.hackrf.set_antenna_enable(self.device, 1 if bias_tee else 0)
            if result != self.hackrf.HackrfError.HACKRF_SUCCESS:
                logger.warning(f"Failed to set bias tee: {result}")
            
            logger.info("HackRF configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure HackRF: {e}")
            return False
    
    def _start_rx(self) -> bool:
        """Start RX mode."""
        try:
            if self.is_streaming:
                return True
            
            result = self.hackrf.start_rx(self.device, self._rx_callback)
            if result != self.hackrf.HackrfError.HACKRF_SUCCESS:
                logger.error(f"Failed to start RX: {result}")
                return False
            
            self.is_streaming = True
            return True
            
        except Exception as e:
            logger.error(f"Error starting HackRF RX: {e}")
            return False
    
    def _stop_rx(self) -> None:
        """Stop RX mode."""
        try:
            if not self.is_streaming:
                return
            
            result = self.hackrf.stop_rx(self.device)
            if result != self.hackrf.HackrfError.HACKRF_SUCCESS:
                logger.error(f"Failed to stop RX: {result}")
            
            self.is_streaming = False
            
        except Exception as e:
            logger.error(f"Error stopping HackRF RX: {e}")
    
    def _rx_callback(self, transfer):
        """Callback for received data."""
        # This would be called by the HackRF library
        # Store received data in a buffer for read_samples
        pass
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read IQ samples from HackRF."""
        if not self.device or not self.connected:
            return None
        
        try:
            # Note: This is a simplified implementation
            # A real implementation would need proper buffering and threading
            logger.warning("HackRF read_samples not fully implemented")
            
            # Return dummy data for testing
            return np.random.normal(0, 0.1, num_samples) + \
                   1j * np.random.normal(0, 0.1, num_samples)
            
        except Exception as e:
            logger.error(f"Error reading HackRF samples: {e}")
            return None
    
    def set_frequency(self, frequency: float) -> bool:
        """Set center frequency."""
        if not self.device:
            return False
        
        try:
            result = self.hackrf.set_freq(self.device, int(frequency))
            if result == self.hackrf.HackrfError.HACKRF_SUCCESS:
                self.center_frequency = frequency
                logger.debug(f"Frequency set to {frequency/1e6:.3f} MHz")
                return True
            else:
                logger.error(f"Failed to set frequency: {result}")
                return False
        except Exception as e:
            logger.error(f"Error setting HackRF frequency: {e}")
            return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate."""
        if not self.device:
            return False
        
        try:
            # Stop streaming if active
            was_streaming = self.is_streaming
            if was_streaming:
                self._stop_rx()
            
            result = self.hackrf.set_sample_rate(self.device, sample_rate)
            if result == self.hackrf.HackrfError.HACKRF_SUCCESS:
                self.sample_rate = sample_rate
                logger.debug(f"Sample rate set to {sample_rate/1e6:.3f} MHz")
                
                # Restart streaming if it was active
                if was_streaming:
                    self._start_rx()
                
                return True
            else:
                logger.error(f"Failed to set sample rate: {result}")
                return False
        except Exception as e:
            logger.error(f"Error setting HackRF sample rate: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """Set combined gain (splits between LNA and VGA)."""
        if not self.device:
            return False
        
        try:
            # Split gain between LNA and VGA
            # HackRF LNA: 0-40 dB (8 dB steps)
            # HackRF VGA: 0-62 dB (2 dB steps)
            
            total_gain = max(0, min(gain, 102))  # Total max ~102 dB
            
            if total_gain <= 40:
                lna_gain = int(total_gain // 8) * 8
                vga_gain = 0
            else:
                lna_gain = 40
                vga_gain = min(62, int((total_gain - 40) // 2) * 2)
            
            # Set LNA gain
            result = self.hackrf.set_lna_gain(self.device, lna_gain)
            if result != self.hackrf.HackrfError.HACKRF_SUCCESS:
                logger.error(f"Failed to set LNA gain: {result}")
                return False
            
            # Set VGA gain
            result = self.hackrf.set_vga_gain(self.device, vga_gain)
            if result != self.hackrf.HackrfError.HACKRF_SUCCESS:
                logger.error(f"Failed to set VGA gain: {result}")
                return False
            
            self.lna_gain = lna_gain
            self.vga_gain = vga_gain
            self.gain = lna_gain + vga_gain
            
            logger.debug(f"Gain set to {self.gain} dB (LNA: {lna_gain}, VGA: {vga_gain})")
            return True
            
        except Exception as e:
            logger.error(f"Error setting HackRF gain: {e}")
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get HackRF device information."""
        if not self.device:
            return {"device_type": "HackRF", "status": "disconnected"}
        
        try:
            info = {
                "device_type": "HackRF",
                "center_freq": self.center_frequency,
                "sample_rate": self.sample_rate,
                "lna_gain": self.lna_gain,
                "vga_gain": self.vga_gain,
                "total_gain": self.lna_gain + self.vga_gain,
                "amp_enable": self.amp_enable,
                "is_streaming": self.is_streaming,
                "status": "connected"
            }
            
            # Add device-specific info if available
            try:
                board_id = self.hackrf.board_id_read(self.device)
                info["board_id"] = board_id
                
                version = self.hackrf.version_string_read(self.device)
                info["version"] = version
                
                serial = self.hackrf.board_partid_serialno_read(self.device)
                info["serial"] = serial
                
            except Exception:
                pass  # Don't fail if can't read device info
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting HackRF device info: {e}")
            return {"device_type": "HackRF", "status": "error", "error": str(e)}
    
    def get_frequency_range(self) -> tuple:
        """Get supported frequency range."""
        # HackRF One frequency range
        return (1e6, 6000e6)  # 1 MHz to 6 GHz
    
    def get_sample_rate_range(self) -> tuple:
        """Get supported sample rate range."""
        # HackRF One sample rate range
        return (2e6, 20e6)  # 2 MHz to 20 MHz
    
    def get_available_sample_rates(self) -> List[float]:
        """Get list of recommended sample rates."""
        # HackRF works well with these sample rates
        return [
            2e6, 4e6, 5e6, 8e6, 10e6, 
            16e6, 20e6
        ]