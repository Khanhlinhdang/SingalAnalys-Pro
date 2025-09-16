"""
SDR Backend Manager
Provides unified interface to different SDR hardware backends.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from enum import Enum

from rf_spectrum_analyzer.config.settings import Settings

logger = logging.getLogger(__name__)


class SDRConfig:
    """Configuration class for SDR parameters."""
    
    def __init__(self, sample_rate: float = 2.048e6, center_frequency: float = 100e6, 
                 gain: float = 20.0, bandwidth: float = None, device_id: str = "0"):
        self.sample_rate = sample_rate
        self.center_frequency = center_frequency
        self.gain = gain
        self.bandwidth = bandwidth if bandwidth is not None else sample_rate
        self.device_id = device_id
        
        # Validate parameters
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.center_frequency <= 0:
            raise ValueError("Center frequency must be positive")
        if self.gain < 0:
            raise ValueError("Gain cannot be negative")
        if self.bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            'sample_rate': self.sample_rate,
            'center_frequency': self.center_frequency,
            'gain': self.gain,
            'bandwidth': self.bandwidth,
            'device_id': self.device_id
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create configuration from dictionary."""
        return cls(**config_dict)


class StreamingError(Exception):
    """Exception raised during streaming operations."""
    pass


class SDRDeviceType(Enum):
    """Supported SDR device types."""
    RTLSDR = "rtlsdr"
    HACKRF = "hackrf" 
    PLUTO = "pluto"
    SOAPY = "soapy"
    USRP = "usrp"
    FILE = "file"
    AUDIO = "audio"


class SDRDevice:
    """SDR device information container."""
    
    def __init__(self, device_type: SDRDeviceType, device_id: str = "0", name: str = "Unknown"):
        self.device_type = device_type
        self.device_id = device_id
        self.name = name
        self.connected = False
        self.backend = None
    
    def __repr__(self):
        return f"SDRDevice(type={self.device_type.value}, id={self.device_id}, name={self.name})"


class SDRBackend(ABC):
    """Abstract base class for SDR backends."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.connected = False
        self.device = None
        self.sample_rate = settings.sdr.sample_rate
        self.center_frequency = settings.sdr.center_frequency
        self.gain = settings.sdr.gain
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the SDR device."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the SDR device."""
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure the SDR device."""
        pass
    
    @abstractmethod
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read IQ samples from the device."""
        pass
    
    @abstractmethod
    def set_frequency(self, frequency: float) -> bool:
        """Set center frequency."""
        pass
    
    @abstractmethod
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate."""
        pass
    
    @abstractmethod
    def set_gain(self, gain: float) -> bool:
        """Set RF gain."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        pass
    
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self.connected


class RTLSDRBackend(SDRBackend):
    """RTL-SDR backend implementation."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.sdr = None
    
    def connect(self) -> bool:
        """Connect to RTL-SDR device."""
        try:
            from rtlsdr import RtlSdr
            
            self.sdr = RtlSdr()
            self.connected = True
            logger.info("RTL-SDR connected successfully")
            return True
            
        except ImportError:
            logger.error("RTL-SDR library not installed")
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
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure RTL-SDR device."""
        if not self.sdr:
            return False
        
        try:
            self.sdr.center_freq = config.get("center_freq", self.center_frequency)
            self.sdr.sample_rate = config.get("sample_rate", self.sample_rate)
            self.sdr.gain = config.get("gain", self.gain)
            
            if "ppm_error" in config:
                self.sdr.freq_correction = config["ppm_error"]
            
            if "bias_tee" in config:
                self.sdr.bias_tee = config["bias_tee"]
            
            logger.info("RTL-SDR configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure RTL-SDR: {e}")
            return False
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read IQ samples from RTL-SDR."""
        if not self.sdr or not self.connected:
            return None
        
        try:
            samples = self.sdr.read_samples(num_samples)
            return samples
        except Exception as e:
            logger.error(f"Error reading RTL-SDR samples: {e}")
            return None
    
    def set_frequency(self, frequency: float) -> bool:
        """Set RTL-SDR center frequency."""
        if not self.sdr:
            return False
        
        try:
            self.sdr.center_freq = frequency
            self.center_frequency = frequency
            return True
        except Exception as e:
            logger.error(f"Error setting RTL-SDR frequency: {e}")
            return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set RTL-SDR sample rate."""
        if not self.sdr:
            return False
        
        try:
            self.sdr.sample_rate = sample_rate
            self.sample_rate = sample_rate
            return True
        except Exception as e:
            logger.error(f"Error setting RTL-SDR sample rate: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """Set RTL-SDR gain."""
        if not self.sdr:
            return False
        
        try:
            self.sdr.gain = gain
            self.gain = gain
            return True
        except Exception as e:
            logger.error(f"Error setting RTL-SDR gain: {e}")
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get RTL-SDR device information."""
        if not self.sdr:
            return {}
        
        return {
            "device_type": "RTL-SDR",
            "center_freq": self.sdr.center_freq,
            "sample_rate": self.sdr.sample_rate,
            "gain": self.sdr.gain,
            "freq_correction": getattr(self.sdr, 'freq_correction', 0),
        }


class HackRFBackend(SDRBackend):
    """HackRF backend implementation."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.device = None
    
    def connect(self) -> bool:
        """Connect to HackRF device."""
        try:
            # Placeholder for HackRF implementation
            # Would use libhackrf or pyhackrf
            logger.info("HackRF backend not fully implemented")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to HackRF: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from HackRF device."""
        pass
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure HackRF device."""
        return False
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read IQ samples from HackRF."""
        return None
    
    def set_frequency(self, frequency: float) -> bool:
        """Set HackRF center frequency."""
        return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set HackRF sample rate."""
        return False
    
    def set_gain(self, gain: float) -> bool:
        """Set HackRF gain."""
        return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get HackRF device information."""
        return {"device_type": "HackRF", "status": "not_implemented"}


class PlutoSDRBackend(SDRBackend):
    """PlutoSDR backend implementation."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.device = None
    
    def connect(self) -> bool:
        """Connect to PlutoSDR device."""
        try:
            # Placeholder for PlutoSDR implementation
            # Would use pyadi-iio
            logger.info("PlutoSDR backend not fully implemented")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to PlutoSDR: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from PlutoSDR device."""
        pass
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure PlutoSDR device."""
        return False
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read IQ samples from PlutoSDR."""
        return None
    
    def set_frequency(self, frequency: float) -> bool:
        """Set PlutoSDR center frequency."""
        return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set PlutoSDR sample rate."""
        return False
    
    def set_gain(self, gain: float) -> bool:
        """Set PlutoSDR gain."""
        return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get PlutoSDR device information."""
        return {"device_type": "PlutoSDR", "status": "not_implemented"}


class SoapySDRBackend(SDRBackend):
    """SoapySDR backend implementation."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.device = None
    
    def connect(self) -> bool:
        """Connect to SoapySDR device."""
        try:
            import SoapySDR
            
            # Create device
            self.device = SoapySDR.Device()
            self.connected = True
            logger.info("SoapySDR connected successfully")
            return True
            
        except ImportError:
            logger.error("SoapySDR library not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to SoapySDR: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from SoapySDR device."""
        if self.device:
            self.device = None
            self.connected = False
            logger.info("SoapySDR disconnected")
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure SoapySDR device."""
        if not self.device:
            return False
        
        try:
            import SoapySDR
            
            # Setup RX stream
            self.device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, 
                                    config.get("sample_rate", self.sample_rate))
            self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0,
                                   config.get("center_freq", self.center_frequency))
            self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0,
                              config.get("gain", self.gain))
            
            logger.info("SoapySDR configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure SoapySDR: {e}")
            return False
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read IQ samples from SoapySDR."""
        # Placeholder implementation
        return None
    
    def set_frequency(self, frequency: float) -> bool:
        """Set SoapySDR center frequency."""
        if not self.device:
            return False
        
        try:
            import SoapySDR
            self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, frequency)
            self.center_frequency = frequency
            return True
        except Exception as e:
            logger.error(f"Error setting SoapySDR frequency: {e}")
            return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set SoapySDR sample rate."""
        if not self.device:
            return False
        
        try:
            import SoapySDR
            self.device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, sample_rate)
            self.sample_rate = sample_rate
            return True
        except Exception as e:
            logger.error(f"Error setting SoapySDR sample rate: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """Set SoapySDR gain."""
        if not self.device:
            return False
        
        try:
            import SoapySDR
            self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
            self.gain = gain
            return True
        except Exception as e:
            logger.error(f"Error setting SoapySDR gain: {e}")
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get SoapySDR device information."""
        if not self.device:
            return {}
        
        return {
            "device_type": "SoapySDR",
            "center_freq": self.center_frequency,
            "sample_rate": self.sample_rate,
            "gain": self.gain,
        }


class FileBackend(SDRBackend):
    """File-based backend for reading IQ data from files."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.file_path = None
        self.file_handle = None
        self.data_format = "complex64"
        self.current_position = 0
    
    def connect(self) -> bool:
        """Connect to file source."""
        # Implementation would open file and determine format
        logger.info("File backend not fully implemented")
        return False
    
    def disconnect(self) -> None:
        """Close file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            self.connected = False
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure file backend."""
        return False
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from file."""
        return None
    
    def set_frequency(self, frequency: float) -> bool:
        """File backend doesn't support frequency changes."""
        return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """File backend doesn't support sample rate changes."""
        return False
    
    def set_gain(self, gain: float) -> bool:
        """File backend doesn't support gain changes."""
        return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get file backend information."""
        return {"device_type": "File", "status": "not_implemented"}


class SDRBackendManager:
    """Manages different SDR backends and provides unified interface."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.current_backend = None
        self.available_backends = {}
        
        # Initialize available backends
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize all available backends."""
        self.available_backends = {
            SDRDeviceType.RTLSDR: RTLSDRBackend(self.settings),
            SDRDeviceType.HACKRF: HackRFBackend(self.settings),
            SDRDeviceType.PLUTO: PlutoSDRBackend(self.settings),
            SDRDeviceType.SOAPY: SoapySDRBackend(self.settings),
            SDRDeviceType.FILE: FileBackend(self.settings),
        }
    
    def set_device_type(self, device_type: str) -> bool:
        """Set the active device type."""
        try:
            device_enum = SDRDeviceType(device_type)
            if device_enum in self.available_backends:
                self.current_backend = self.available_backends[device_enum]
                self.settings.sdr.device_type = device_type
                logger.info(f"Set device type to {device_type}")
                return True
        except ValueError:
            logger.error(f"Unknown device type: {device_type}")
        return False
    
    def connect(self) -> bool:
        """Connect to the current device."""
        if not self.current_backend:
            device_type = self.settings.sdr.device_type
            if not self.set_device_type(device_type):
                return False
        
        return self.current_backend.connect()
    
    def disconnect(self) -> None:
        """Disconnect from the current device."""
        if self.current_backend:
            self.current_backend.disconnect()
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure the current device."""
        if self.current_backend:
            return self.current_backend.configure(config)
        return False
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from the current device."""
        if self.current_backend:
            return self.current_backend.read_samples(num_samples)
        return None
    
    def set_frequency(self, frequency: float) -> bool:
        """Set center frequency."""
        if self.current_backend:
            return self.current_backend.set_frequency(frequency)
        return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate."""
        if self.current_backend:
            return self.current_backend.set_sample_rate(sample_rate)
        return False
    
    def set_gain(self, gain: float) -> bool:
        """Set RF gain."""
        if self.current_backend:
            return self.current_backend.set_gain(gain)
        return False
    
    def is_connected(self) -> bool:
        """Check if device is connected."""
        if self.current_backend:
            return self.current_backend.is_connected()
        return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get current device information."""
        if self.current_backend:
            return self.current_backend.get_device_info()
        return {}
    
    def get_available_devices(self) -> List[str]:
        """Get list of available device types."""
        return [device.value for device in SDRDeviceType]