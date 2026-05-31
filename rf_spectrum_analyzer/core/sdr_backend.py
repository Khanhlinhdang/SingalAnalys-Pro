"""SDR backend abstractions and runtime backend manager."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from rf_spectrum_analyzer.config.settings import Settings

logger = logging.getLogger(__name__)


class SDRConfig:
    """Legacy compatibility config class used by older tests and tooling."""

    def __init__(
        self,
        sample_rate: float = 2.048e6,
        center_frequency: float = 100e6,
        gain: float = 20.0,
        bandwidth: Optional[float] = None,
        device_id: str = "0",
    ):
        self.sample_rate = sample_rate
        self.center_frequency = center_frequency
        self.gain = gain
        self.bandwidth = bandwidth if bandwidth is not None else sample_rate
        self.device_id = device_id
        self._validate()

    def _validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.center_frequency <= 0:
            raise ValueError("Center frequency must be positive")
        if self.gain < 0:
            raise ValueError("Gain cannot be negative")
        if self.bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "center_frequency": self.center_frequency,
            "gain": self.gain,
            "bandwidth": self.bandwidth,
            "device_id": self.device_id,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SDRConfig":
        return cls(**config_dict)


class StreamingError(Exception):
    """Exception raised during streaming operations."""


class SDRDeviceType(Enum):
    """Supported SDR device types."""

    RTLSDR = "rtlsdr"
    HACKRF = "hackrf"
    PLUTO = "pluto"
    SOAPY = "soapy"
    USRP = "usrp"
    SPYSERVER = "spyserver"
    FILE = "file"
    AUDIO = "audio"


class SDRDevice:
    """SDR device information container."""

    def __init__(self, device_type: SDRDeviceType, device_id: str = "0", name: str = "Unknown"):
        self.device_type = device_type
        self.device_id = device_id
        self.name = name
        self.connected = False

    def __repr__(self) -> str:
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

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the SDR device."""

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure the SDR device."""

    @abstractmethod
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read IQ samples from the device."""

    @abstractmethod
    def set_frequency(self, frequency: float) -> bool:
        """Set center frequency."""

    @abstractmethod
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate."""

    @abstractmethod
    def set_gain(self, gain: float) -> bool:
        """Set RF gain."""

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""

    def is_connected(self) -> bool:
        return self.connected


class SDRBackendManager:
    """Routes SDR operations to the selected runtime backend adapter."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.current_backend: Optional[SDRBackend] = None
        self.current_device_type = SDRDeviceType(settings.sdr.device_type)
        self._set_backend_by_type(self.current_device_type)

    @property
    def backend(self) -> Optional[SDRBackend]:
        """Compatibility alias used by app threading checks."""
        return self.current_backend

    def _set_backend_by_type(self, device_type: SDRDeviceType) -> None:
        self.current_device_type = device_type

        try:
            if device_type == SDRDeviceType.RTLSDR:
                from rf_spectrum_analyzer.backends.rtlsdr_backend import RTLSDRBackend

                self.current_backend = RTLSDRBackend(self.settings)
            elif device_type == SDRDeviceType.HACKRF:
                from rf_spectrum_analyzer.backends.hackrf_backend import HackRFBackend

                self.current_backend = HackRFBackend(self.settings)
            elif device_type == SDRDeviceType.PLUTO:
                from rf_spectrum_analyzer.backends.pluto_backend import PlutoSDRBackend

                self.current_backend = PlutoSDRBackend(self.settings)
            elif device_type == SDRDeviceType.SOAPY:
                from rf_spectrum_analyzer.backends.soapy_backend import SoapySDRBackend

                self.current_backend = SoapySDRBackend(self.settings)
            elif device_type == SDRDeviceType.USRP:
                from rf_spectrum_analyzer.backends.usrp_backend import USRPBackend

                self.current_backend = USRPBackend(use_simulator=False)
            elif device_type == SDRDeviceType.SPYSERVER:
                from rf_spectrum_analyzer.backends.spyserver_backend import SpyServerBackend

                self.current_backend = SpyServerBackend(self.settings)
            elif device_type in (SDRDeviceType.FILE, SDRDeviceType.AUDIO):
                self.current_backend = None
                logger.warning("Device type %s is not implemented for live runtime", device_type.value)
            else:
                self.current_backend = None
                logger.error("Unsupported device type: %s", device_type.value)
        except Exception as exc:
            self.current_backend = None
            logger.error("Failed to initialize backend %s: %s", device_type.value, exc)

    def set_device_type(self, device_type: str) -> bool:
        try:
            parsed = SDRDeviceType(device_type.lower())
        except ValueError:
            logger.error("Unsupported device type: %s", device_type)
            return False

        if self.current_backend and self.current_backend.is_connected():
            self.current_backend.disconnect()

        self.settings.sdr.device_type = parsed.value
        self._set_backend_by_type(parsed)
        return self.current_backend is not None

    def connect(self) -> bool:
        if not self.current_backend:
            logger.error("No backend selected")
            return False

        if self.current_backend.connect():
            config = self.settings.get_device_settings()
            if not self.current_backend.configure(config):
                logger.warning("Backend connected but configuration failed")
            return True
        return False

    def disconnect(self) -> None:
        if self.current_backend:
            self.current_backend.disconnect()

    def is_connected(self) -> bool:
        return bool(self.current_backend and self.current_backend.is_connected())

    def configure(self, config: Dict[str, Any]) -> bool:
        if not self.current_backend:
            return False
        return self.current_backend.configure(config)

    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        if not self.current_backend:
            return None
        return self.current_backend.read_samples(num_samples)

    def set_frequency(self, frequency: float) -> bool:
        if not self.current_backend:
            return False
        self.settings.sdr.center_frequency = frequency
        return self.current_backend.set_frequency(frequency)

    def set_sample_rate(self, sample_rate: float) -> bool:
        if not self.current_backend:
            return False
        self.settings.sdr.sample_rate = sample_rate
        return self.current_backend.set_sample_rate(sample_rate)

    def set_gain(self, gain: float) -> bool:
        if not self.current_backend:
            return False
        self.settings.sdr.gain = gain
        return self.current_backend.set_gain(gain)

    def set_bandwidth(self, bandwidth: float) -> bool:
        if not self.current_backend:
            return False
        self.settings.sdr.bandwidth = bandwidth
        if hasattr(self.current_backend, "set_bandwidth"):
            return self.current_backend.set_bandwidth(bandwidth)
        return True

    def get_device_info(self) -> Dict[str, Any]:
        if not self.current_backend:
            return {}
        return self.current_backend.get_device_info()

    def get_available_devices(self):
        return [
            SDRDeviceType.RTLSDR.value,
            SDRDeviceType.HACKRF.value,
            SDRDeviceType.PLUTO.value,
            SDRDeviceType.SOAPY.value,
            SDRDeviceType.USRP.value,
            SDRDeviceType.SPYSERVER.value,
            SDRDeviceType.FILE.value,
            SDRDeviceType.AUDIO.value,
        ]
