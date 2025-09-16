# Create core/sdr_backend.py - SDR Backend Management
sdr_backend_content = '''"""
SDR Backend Management

This module manages different SDR backends (RTL-SDR, PlutoSDR, HackRF, SoapySDR)
and provides a unified interface for device control and sample acquisition.

Integrates with pyspectrum's plugin architecture for device support.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading
import queue
import time

from PySide6.QtCore import QObject, Signal

from config.settings import AppSettings
from utils.logger import get_sdr_logger


@dataclass
class DeviceInfo:
    """Information about an SDR device"""
    name: str
    driver: str
    serial: str
    index: int
    available: bool = True
    capabilities: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = {}
    
    def __str__(self):
        return f"{self.name} ({self.driver}) [{self.serial}]"


class SDRBackend(ABC):
    """Abstract base class for SDR backends"""
    
    def __init__(self, device_info: DeviceInfo, settings: AppSettings):
        self.device_info = device_info
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sdr_logger = get_sdr_logger()
        
        # Device state
        self.is_open = False
        self.is_streaming = False
        self.sample_buffer = queue.Queue(maxsize=100)
        
        # Current parameters
        self.center_freq = settings.sdr.center_freq
        self.sample_rate = settings.sdr.sample_rate
        self.gain = settings.sdr.gain
        self.bandwidth = settings.sdr.bandwidth
        
        # Performance tracking
        self.samples_received = 0
        self.overruns = 0
        self.last_status_time = time.time()
    
    @abstractmethod
    def open(self) -> bool:
        """Open connection to SDR device"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close connection to SDR device"""
        pass
    
    @abstractmethod
    def start_streaming(self) -> bool:
        """Start sample streaming"""
        pass
    
    @abstractmethod
    def stop_streaming(self) -> None:
        """Stop sample streaming"""
        pass
    
    @abstractmethod
    def set_center_frequency(self, freq: float) -> bool:
        """Set center frequency in Hz"""
        pass
    
    @abstractmethod
    def set_sample_rate(self, rate: float) -> bool:
        """Set sample rate in Hz"""
        pass
    
    @abstractmethod
    def set_gain(self, gain: float) -> bool:
        """Set RF gain in dB"""
        pass
    
    @abstractmethod
    def set_bandwidth(self, bandwidth: float) -> bool:
        """Set bandwidth in Hz"""
        pass
    
    @abstractmethod
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from device"""
        pass
    
    @abstractmethod
    def get_supported_sample_rates(self) -> List[float]:
        """Get list of supported sample rates"""
        pass
    
    @abstractmethod
    def get_frequency_range(self) -> tuple:
        """Get (min_freq, max_freq) tuple"""
        pass
    
    @abstractmethod
    def get_gain_range(self) -> tuple:
        """Get (min_gain, max_gain) tuple"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get device status information"""
        return {
            'device_info': self.device_info,
            'is_open': self.is_open,
            'is_streaming': self.is_streaming,
            'center_freq': self.center_freq,
            'sample_rate': self.sample_rate,
            'gain': self.gain,
            'bandwidth': self.bandwidth,
            'samples_received': self.samples_received,
            'overruns': self.overruns
        }
    
    def log_status(self) -> None:
        """Log current device status"""
        now = time.time()
        if now - self.last_status_time >= 10.0:  # Every 10 seconds
            status = self.get_status()
            self.sdr_logger.logger.info(f"Device Status: {status}")
            self.last_status_time = now


class DummyBackend(SDRBackend):
    """Dummy backend for testing without hardware"""
    
    def __init__(self, device_info: DeviceInfo, settings: AppSettings):
        super().__init__(device_info, settings)
        self.noise_generator = np.random.RandomState(42)
    
    def open(self) -> bool:
        self.is_open = True
        self.logger.info("Dummy backend opened")
        return True
    
    def close(self) -> None:
        self.is_open = False
        self.is_streaming = False
        self.logger.info("Dummy backend closed")
    
    def start_streaming(self) -> bool:
        if not self.is_open:
            return False
        self.is_streaming = True
        self.logger.info("Dummy backend streaming started")
        return True
    
    def stop_streaming(self) -> None:
        self.is_streaming = False
        self.logger.info("Dummy backend streaming stopped")
    
    def set_center_frequency(self, freq: float) -> bool:
        self.center_freq = freq
        return True
    
    def set_sample_rate(self, rate: float) -> bool:
        self.sample_rate = rate
        return True
    
    def set_gain(self, gain: float) -> bool:
        self.gain = gain
        return True
    
    def set_bandwidth(self, bandwidth: float) -> bool:
        self.bandwidth = bandwidth
        return True
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        if not self.is_streaming:
            return None
        
        # Generate complex noise with some test signals
        samples = self.noise_generator.normal(0, 0.1, num_samples) + \\
                 1j * self.noise_generator.normal(0, 0.1, num_samples)
        
        # Add a test tone
        t = np.arange(num_samples) / self.sample_rate
        test_tone = 0.5 * np.exp(1j * 2 * np.pi * 100e3 * t)  # 100 kHz tone
        samples += test_tone
        
        self.samples_received += num_samples
        return samples.astype(np.complex64)
    
    def get_supported_sample_rates(self) -> List[float]:
        return [250e3, 1e6, 2e6, 2.4e6, 3.2e6]
    
    def get_frequency_range(self) -> tuple:
        return (24e6, 1.8e9)
    
    def get_gain_range(self) -> tuple:
        return (0, 50)


class SDRBackendManager(QObject):
    """Manages multiple SDR backends and device discovery"""
    
    # Qt Signals
    device_connected = Signal(object)      # DeviceInfo
    device_disconnected = Signal()
    error_occurred = Signal(str)          # Error message
    
    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.sdr_logger = get_sdr_logger()
        
        # Available backends
        self.backends = {}
        self.current_backend: Optional[SDRBackend] = None
        
        # Device discovery
        self.available_devices = []
        
        # Initialize backends
        self.initialize_backends()
    
    def initialize_backends(self) -> None:
        """Initialize available SDR backends"""
        try:
            # Import and register backends
            self.register_backend("dummy", DummyBackend)
            
            try:
                from backends.rtlsdr_backend import RTLSDRBackend
                self.register_backend("rtlsdr", RTLSDRBackend)
            except ImportError as e:
                self.logger.warning(f"RTL-SDR backend not available: {e}")
            
            try:
                from backends.pluto_backend import PlutoBackend
                self.register_backend("pluto", PlutoBackend)
            except ImportError as e:
                self.logger.warning(f"Pluto backend not available: {e}")
            
            try:
                from backends.hackrf_backend import HackRFBackend
                self.register_backend("hackrf", HackRFBackend)
            except ImportError as e:
                self.logger.warning(f"HackRF backend not available: {e}")
            
            try:
                from backends.soapy_backend import SoapyBackend
                self.register_backend("soapy", SoapyBackend)
            except ImportError as e:
                self.logger.warning(f"SoapySDR backend not available: {e}")
            
            self.logger.info(f"Initialized {len(self.backends)} SDR backends")
            
        except Exception as e:
            self.logger.error(f"Backend initialization error: {e}")
    
    def register_backend(self, name: str, backend_class):
        """Register a new backend class"""
        self.backends[name] = backend_class
        self.logger.debug(f"Registered backend: {name}")
    
    def discover_devices(self) -> List[DeviceInfo]:
        """Discover available SDR devices"""
        devices = []
        
        for backend_name, backend_class in self.backends.items():
            try:
                # Get devices from each backend
                if hasattr(backend_class, 'enumerate_devices'):
                    backend_devices = backend_class.enumerate_devices()
                    devices.extend(backend_devices)
                    self.logger.debug(f"Found {len(backend_devices)} devices for {backend_name}")
                
            except Exception as e:
                self.logger.warning(f"Error discovering {backend_name} devices: {e}")
        
        # Add dummy device if no real devices found
        if not devices:
            dummy_device = DeviceInfo(
                name="Dummy SDR",
                driver="dummy",
                serial="dummy_001",
                index=0,
                available=True,
                capabilities={'test_mode': True}
            )
            devices.append(dummy_device)
        
        self.available_devices = devices
        self.logger.info(f"Discovered {len(devices)} SDR devices")
        return devices
    
    def get_available_devices(self) -> List[DeviceInfo]:
        """Get list of available devices"""
        if not self.available_devices:
            self.discover_devices()
        return self.available_devices
    
    def create_backend(self, device_info: DeviceInfo) -> Optional[SDRBackend]:
        """Create backend instance for specific device"""
        backend_class = self.backends.get(device_info.driver)
        
        if backend_class is None:
            self.logger.error(f"No backend available for driver: {device_info.driver}")
            return None
        
        try:
            backend = backend_class(device_info, self.settings)
            self.logger.info(f"Created backend for {device_info}")
            return backend
            
        except Exception as e:
            self.logger.error(f"Failed to create backend: {e}")
            return None
    
    def set_device(self, device_info: Union[DeviceInfo, int, str]) -> bool:
        """Set active SDR device"""
        try:
            # Handle different input types
            if isinstance(device_info, int):
                # Device index
                devices = self.get_available_devices()
                if 0 <= device_info < len(devices):
                    device_info = devices[device_info]
                else:
                    raise ValueError(f"Invalid device index: {device_info}")
            
            elif isinstance(device_info, str):
                # Device name or auto-select
                if device_info == "auto":
                    devices = self.get_available_devices()
                    device_info = devices[0] if devices else None
                else:
                    # Find device by name
                    devices = self.get_available_devices()
                    matching_devices = [d for d in devices if device_info.lower() in d.name.lower()]
                    device_info = matching_devices[0] if matching_devices else None
            
            if device_info is None:
                raise ValueError("No suitable device found")
            
            # Stop current backend
            if self.current_backend:
                self.current_backend.stop_streaming()
                self.current_backend.close()
                self.current_backend = None
            
            # Create new backend
            self.current_backend = self.create_backend(device_info)
            if self.current_backend is None:
                raise RuntimeError(f"Failed to create backend for {device_info}")
            
            self.logger.info(f"Set active device: {device_info}")
            self.device_connected.emit(device_info)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set device: {e}")
            self.error_occurred.emit(str(e))
            return False
    
    def initialize(self) -> bool:
        """Initialize the current backend"""
        if not self.current_backend:
            # Auto-select first available device
            if not self.set_device("auto"):
                return False
        
        try:
            # Open device connection
            if not self.current_backend.open():
                raise RuntimeError("Failed to open device")
            
            # Configure device parameters
            self.apply_settings()
            
            self.logger.info("Backend initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Backend initialization failed: {e}")
            self.error_occurred.emit(str(e))
            return False
    
    def apply_settings(self) -> None:
        """Apply current settings to backend"""
        if not self.current_backend or not self.current_backend.is_open:
            return
        
        try:
            # Apply SDR settings
            self.current_backend.set_center_frequency(self.settings.sdr.center_freq)
            self.current_backend.set_sample_rate(self.settings.sdr.sample_rate)
            self.current_backend.set_gain(self.settings.sdr.gain)
            
            if self.settings.sdr.bandwidth:
                self.current_backend.set_bandwidth(self.settings.sdr.bandwidth)
            
            self.logger.info("Settings applied to backend")
            
        except Exception as e:
            self.logger.error(f"Failed to apply settings: {e}")
            self.error_occurred.emit(str(e))
    
    def start(self) -> bool:
        """Start signal acquisition"""
        if not self.current_backend:
            self.logger.error("No backend available")
            return False
        
        try:
            if self.current_backend.start_streaming():
                self.logger.info("Signal acquisition started")
                return True
            else:
                self.logger.error("Failed to start streaming")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting acquisition: {e}")
            self.error_occurred.emit(str(e))
            return False
    
    def stop(self) -> None:
        """Stop signal acquisition"""
        if self.current_backend:
            try:
                self.current_backend.stop_streaming()
                self.logger.info("Signal acquisition stopped")
            except Exception as e:
                self.logger.error(f"Error stopping acquisition: {e}")
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from current backend"""
        if not self.current_backend:
            return None
        
        try:
            return self.current_backend.read_samples(num_samples)
        except Exception as e:
            self.logger.error(f"Error reading samples: {e}")
            self.error_occurred.emit(str(e))
            return None
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get current device status"""
        if self.current_backend:
            return self.current_backend.get_status()
        else:
            return {'error': 'No active device'}
    
    def update_settings(self, new_settings: AppSettings) -> None:
        """Update settings and apply to backend"""
        self.settings = new_settings
        self.apply_settings()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if self.current_backend:
                self.current_backend.stop_streaming()
                self.current_backend.close()
                self.current_backend = None
            
            self.logger.info("Backend manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
'''

with open("rf_spectrum_analyzer/core/sdr_backend.py", "w") as f:
    f.write(sdr_backend_content)

print("Created core/sdr_backend.py")