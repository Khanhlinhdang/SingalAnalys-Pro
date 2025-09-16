"""
SoapySDR Backend Implementation

Provides support for various SDR devices through the SoapySDR framework.
Supports RTL-SDR, HackRF, PlutoSDR, BladeRF, USRP, and many others.
"""

import numpy as np
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Tuple

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_CS16
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False
    SoapySDR = None
    SOAPY_SDR_RX = None
    SOAPY_SDR_CF32 = None
    SOAPY_SDR_CS16 = None

from core.sdr_backend import SDRBackend, DeviceInfo
from config.settings import AppSettings


class SoapyBackend(SDRBackend):
    """SoapySDR backend for multiple SDR devices"""

    def __init__(self, device_info: DeviceInfo, settings: AppSettings):
        super().__init__(device_info, settings)

        if not SOAPY_AVAILABLE:
            raise ImportError("SoapySDR not available")

        self.sdr = None
        self.rx_stream = None
        self.stream_mtu = 1024

        # Stream buffers
        self.buffer_size = settings.processing.buffer_size
        self.rx_buffer = np.zeros(self.buffer_size, dtype=np.complex64)

        # Threading for continuous streaming
        self.stream_thread = None
        self.streaming = False

        # Performance tracking
        self.overruns = 0
        self.underruns = 0

    @staticmethod
    def enumerate_devices() -> List[DeviceInfo]:
        """Enumerate available SoapySDR devices"""
        if not SOAPY_AVAILABLE:
            return []

        devices = []
        try:
            # Enumerate all SoapySDR devices
            soapy_devices = SoapySDR.Device.enumerate()

            for i, device_args in enumerate(soapy_devices):
                # Extract device information
                driver = device_args.get('driver', 'unknown')
                label = device_args.get('label', f'SoapySDR Device {i}')
                serial = device_args.get('serial', f'soapy_{i}')

                # Create device info
                device_info = DeviceInfo(
                    name=label,
                    driver='soapy',
                    serial=serial,
                    index=i,
                    available=True,
                    capabilities={
                        'soapy_driver': driver,
                        'soapy_args': device_args
                    }
                )

                devices.append(device_info)

        except Exception as e:
            logging.getLogger(__name__).error(f"Error enumerating SoapySDR devices: {e}")

        return devices

    def open(self) -> bool:
        """Open connection to SoapySDR device"""
        try:
            if self.is_open:
                return True

            # Get device arguments from capabilities
            device_args = self.device_info.capabilities.get('soapy_args', {})

            # Create SoapySDR device
            self.sdr = SoapySDR.Device(device_args)

            # Set initial parameters
            self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
            self.sdr.setGain(SOAPY_SDR_RX, 0, self.gain)

            # Set bandwidth if specified
            if self.bandwidth:
                self.sdr.setBandwidth(SOAPY_SDR_RX, 0, self.bandwidth)

            # Get stream MTU
            self.stream_mtu = self.sdr.getStreamMTU(SOAPY_SDR_RX, 0)
            if self.stream_mtu == 0:
                self.stream_mtu = 1024  # Default

            self.is_open = True
            self.logger.info(f"SoapySDR device opened: {self.device_info.name}")

            # Log device capabilities
            self.log_device_capabilities()

            return True

        except Exception as e:
            self.logger.error(f"Failed to open SoapySDR device: {e}")
            return False

    def close(self) -> None:
        """Close SoapySDR device"""
        try:
            # Stop streaming first
            self.stop_streaming()

            if self.sdr:
                self.sdr = None

            self.is_open = False
            self.logger.info("SoapySDR device closed")

        except Exception as e:
            self.logger.error(f"Error closing SoapySDR device: {e}")

    def start_streaming(self) -> bool:
        """Start sample streaming"""
        try:
            if not self.is_open:
                self.logger.error("Device not open")
                return False

            if self.is_streaming:
                return True

            # Setup RX stream
            self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])

            # Activate stream
            self.sdr.activateStream(self.rx_stream)

            # Start streaming thread
            self.streaming = True
            self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.stream_thread.start()

            self.is_streaming = True
            self.logger.info("SoapySDR streaming started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            return False

    def stop_streaming(self) -> None:
        """Stop sample streaming"""
        try:
            self.streaming = False

            # Wait for stream thread to finish
            if self.stream_thread:
                self.stream_thread.join(timeout=1.0)
                self.stream_thread = None

            # Deactivate and close stream
            if self.rx_stream and self.sdr:
                self.sdr.deactivateStream(self.rx_stream)
                self.sdr.closeStream(self.rx_stream)
                self.rx_stream = None

            self.is_streaming = False
            self.logger.info("SoapySDR streaming stopped")

        except Exception as e:
            self.logger.error(f"Error stopping streaming: {e}")

    def _stream_loop(self):
        """Streaming loop running in separate thread"""
        buffer = np.zeros(self.stream_mtu, dtype=np.complex64)

        while self.streaming:
            try:
                # Read samples from stream
                sr = self.sdr.readStream(self.rx_stream, [buffer], self.stream_mtu, timeoutUs=100000)

                if sr.ret > 0:
                    # Successfully read samples
                    samples = buffer[:sr.ret].copy()

                    # Add to sample buffer (thread-safe queue)
                    try:
                        self.sample_buffer.put(samples, block=False)
                        self.samples_received += len(samples)
                    except:
                        # Buffer full - increment overrun counter
                        self.overruns += 1

                elif sr.ret == SOAPY_SDR_TIMEOUT:
                    # Timeout - continue
                    continue
                elif sr.ret == SOAPY_SDR_OVERFLOW:
                    # Overflow
                    self.overruns += 1
                elif sr.ret < 0:
                    # Other error
                    self.logger.warning(f"Stream error: {sr.ret}")

            except Exception as e:
                self.logger.error(f"Stream loop error: {e}")
                time.sleep(0.01)

    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from buffer"""
        if not self.is_streaming:
            return None

        try:
            samples = []
            samples_needed = num_samples

            # Collect samples from buffer
            while samples_needed > 0 and not self.sample_buffer.empty():
                try:
                    buffer_samples = self.sample_buffer.get(block=False)

                    if len(buffer_samples) <= samples_needed:
                        samples.extend(buffer_samples)
                        samples_needed -= len(buffer_samples)
                    else:
                        # Take only what we need and put the rest back
                        samples.extend(buffer_samples[:samples_needed])
                        remaining = buffer_samples[samples_needed:]
                        self.sample_buffer.put(remaining, block=False)
                        samples_needed = 0

                except:
                    break

            if samples:
                return np.array(samples, dtype=np.complex64)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error reading samples: {e}")
            return None

    def set_center_frequency(self, freq: float) -> bool:
        """Set center frequency"""
        try:
            if self.sdr:
                self.sdr.setFrequency(SOAPY_SDR_RX, 0, freq)
                self.center_freq = freq
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting frequency: {e}")
            return False

    def set_sample_rate(self, rate: float) -> bool:
        """Set sample rate"""
        try:
            if self.sdr:
                self.sdr.setSampleRate(SOAPY_SDR_RX, 0, rate)
                self.sample_rate = rate
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting sample rate: {e}")
            return False

    def set_gain(self, gain: float) -> bool:
        """Set RF gain"""
        try:
            if self.sdr:
                self.sdr.setGain(SOAPY_SDR_RX, 0, gain)
                self.gain = gain
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting gain: {e}")
            return False

    def set_bandwidth(self, bandwidth: float) -> bool:
        """Set bandwidth"""
        try:
            if self.sdr:
                self.sdr.setBandwidth(SOAPY_SDR_RX, 0, bandwidth)
                self.bandwidth = bandwidth
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting bandwidth: {e}")
            return False

    def get_supported_sample_rates(self) -> List[float]:
        """Get supported sample rates"""
        if not self.sdr:
            return []

        try:
            ranges = self.sdr.getSampleRateRange(SOAPY_SDR_RX, 0)
            rates = []

            for rate_range in ranges:
                # Add some common rates within the range
                min_rate, max_rate = rate_range.minimum(), rate_range.maximum()
                common_rates = [250e3, 1e6, 2e6, 2.4e6, 3.2e6, 8e6, 10e6, 20e6, 30.72e6, 56e6]

                for rate in common_rates:
                    if min_rate <= rate <= max_rate:
                        rates.append(rate)

            return sorted(list(set(rates)))

        except Exception as e:
            self.logger.error(f"Error getting sample rates: {e}")
            return []

    def get_frequency_range(self) -> tuple:
        """Get frequency range"""
        if not self.sdr:
            return (0, 0)

        try:
            ranges = self.sdr.getFrequencyRange(SOAPY_SDR_RX, 0)
            if ranges:
                min_freq = min(r.minimum() for r in ranges)
                max_freq = max(r.maximum() for r in ranges)
                return (min_freq, max_freq)
            else:
                return (0, 0)
        except Exception as e:
            self.logger.error(f"Error getting frequency range: {e}")
            return (0, 0)

    def get_gain_range(self) -> tuple:
        """Get gain range"""
        if not self.sdr:
            return (0, 0)

        try:
            gain_range = self.sdr.getGainRange(SOAPY_SDR_RX, 0)
            return (gain_range.minimum(), gain_range.maximum())
        except Exception as e:
            self.logger.error(f"Error getting gain range: {e}")
            return (0, 0)

    def log_device_capabilities(self):
        """Log device capabilities and information"""
        if not self.sdr:
            return

        try:
            info = {}

            # Hardware info
            if hasattr(self.sdr, 'getHardwareInfo'):
                hw_info = self.sdr.getHardwareInfo()
                info.update(hw_info)

            # Driver info  
            info['driver_key'] = self.sdr.getDriverKey()
            info['hardware_key'] = self.sdr.getHardwareKey()

            # Frequency range
            freq_range = self.get_frequency_range()
            info['frequency_range'] = f"{freq_range[0]/1e6:.1f} - {freq_range[1]/1e6:.1f} MHz"

            # Sample rate range
            try:
                sample_rates = self.get_supported_sample_rates()
                if sample_rates:
                    info['sample_rates'] = f"{sample_rates[0]/1e6:.1f} - {sample_rates[-1]/1e6:.1f} MSps"
            except:
                pass

            # Gain range
            gain_range = self.get_gain_range()
            info['gain_range'] = f"{gain_range[0]:.1f} - {gain_range[1]:.1f} dB"

            # Antennas
            try:
                antennas = self.sdr.listAntennas(SOAPY_SDR_RX, 0)
                info['antennas'] = ', '.join(antennas)
            except:
                pass

            # Log the information
            self.sdr_logger.log_device_info(info)

        except Exception as e:
            self.logger.error(f"Error logging device capabilities: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get extended device status"""
        status = super().get_status()

        # Add SoapySDR-specific status
        if self.sdr:
            try:
                # Stream statistics
                if self.rx_stream:
                    status.update({
                        'stream_mtu': self.stream_mtu,
                        'overruns': self.overruns,
                        'underruns': self.underruns
                    })

                # Current settings
                actual_rate = self.sdr.getSampleRate(SOAPY_SDR_RX, 0)
                actual_freq = self.sdr.getFrequency(SOAPY_SDR_RX, 0)
                actual_gain = self.sdr.getGain(SOAPY_SDR_RX, 0)

                status.update({
                    'actual_sample_rate': actual_rate,
                    'actual_center_freq': actual_freq,
                    'actual_gain': actual_gain
                })

            except Exception as e:
                status['status_error'] = str(e)

        return status


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test device enumeration
    devices = SoapyBackend.enumerate_devices()
    print(f"Found {len(devices)} SoapySDR devices:")
    for device in devices:
        print(f"  {device}")

    if devices:
        # Test opening first device
        from config.settings import AppSettings
        settings = AppSettings()

        backend = SoapyBackend(devices[0], settings)

        if backend.open():
            print("Device opened successfully")

            # Print capabilities
            print(f"Frequency range: {backend.get_frequency_range()}")
            print(f"Gain range: {backend.get_gain_range()}")
            print(f"Sample rates: {backend.get_supported_sample_rates()}")

            backend.close()
        else:
            print("Failed to open device")
