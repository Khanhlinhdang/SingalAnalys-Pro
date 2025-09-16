"""
RTL-SDR Backend Implementation

Provides support for RTL-SDR USB dongles using the pyrtlsdr library.
RTL-SDR is one of the most popular and affordable SDR platforms.
"""

import numpy as np
import logging
import threading
import time
import queue
from typing import List, Dict, Any, Optional

try:
    from rtlsdr import RtlSdr
    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    RtlSdr = None

from core.sdr_backend import SDRBackend, DeviceInfo
from config.settings import AppSettings


class RTLSDRBackend(SDRBackend):
    """RTL-SDR backend implementation"""

    def __init__(self, device_info: DeviceInfo, settings: AppSettings):
        super().__init__(device_info, settings)

        if not RTLSDR_AVAILABLE:
            raise ImportError("pyrtlsdr not available")

        self.rtlsdr = None

        # RTL-SDR specific settings
        self.device_index = device_info.index
        self.test_mode = False
        self.offset_tuning = False
        self.bias_tee = False

        # Sample buffer for asynchronous reading
        self.async_buffer = queue.Queue(maxsize=100)
        self.read_samples_count = 0

        # Frequency correction
        self.freq_correction = 0  # PPM

        # Valid gain values for RTL-SDR (will be populated from device)
        self.valid_gains = []

    @staticmethod
    def enumerate_devices() -> List[DeviceInfo]:
        """Enumerate available RTL-SDR devices"""
        if not RTLSDR_AVAILABLE:
            return []

        devices = []
        try:
            # Get device count
            device_count = RtlSdr.get_device_count()

            for i in range(device_count):
                try:
                    # Get device info
                    device_name = RtlSdr.get_device_name(i)

                    # Try to get serial number
                    try:
                        temp_sdr = RtlSdr(device_index=i)
                        serial = temp_sdr.get_device_serial_addresses()[i] if temp_sdr.get_device_serial_addresses() else f"rtl_{i}"
                        temp_sdr.close()
                    except:
                        serial = f"rtl_{i}"

                    # Create device info
                    device_info = DeviceInfo(
                        name=device_name,
                        driver='rtlsdr',
                        serial=serial,
                        index=i,
                        available=True,
                        capabilities={
                            'frequency_range': (24e6, 1.8e9),  # Typical RTL-SDR range
                            'sample_rates': [250e3, 1e6, 1.024e6, 1.8e6, 1.92e6, 2.048e6, 2.4e6, 2.88e6, 3.2e6]
                        }
                    )

                    devices.append(device_info)

                except Exception as e:
                    logging.getLogger(__name__).warning(f"Error probing RTL-SDR device {i}: {e}")

        except Exception as e:
            logging.getLogger(__name__).error(f"Error enumerating RTL-SDR devices: {e}")

        return devices

    def open(self) -> bool:
        """Open RTL-SDR device"""
        try:
            if self.is_open:
                return True

            # Create RTL-SDR instance
            self.rtlsdr = RtlSdr(device_index=self.device_index)

            # Set initial parameters
            self.rtlsdr.sample_rate = self.sample_rate
            self.rtlsdr.center_freq = self.center_freq

            # Set gain
            self.valid_gains = self.rtlsdr.get_gains()
            if self.valid_gains:
                # Find closest valid gain
                closest_gain = min(self.valid_gains, key=lambda x: abs(x/10.0 - self.gain))
                self.rtlsdr.gain = closest_gain / 10.0  # Convert from tenths of dB
                self.gain = closest_gain / 10.0
            else:
                self.rtlsdr.gain = 'auto'

            # Set frequency correction if needed
            if self.freq_correction != 0:
                self.rtlsdr.freq_correction = self.freq_correction

            # Additional RTL-SDR settings
            if self.offset_tuning:
                # Enable offset tuning for better performance at low frequencies
                try:
                    self.rtlsdr.set_offset_tuning(True)
                except:
                    self.logger.warning("Offset tuning not supported")

            if self.bias_tee:
                # Enable bias tee if supported
                try:
                    self.rtlsdr.set_bias_tee(True)
                except:
                    self.logger.warning("Bias tee not supported")

            self.is_open = True
            self.logger.info(f"RTL-SDR device opened: {self.device_info.name}")

            # Log device info
            self.log_device_info()

            return True

        except Exception as e:
            self.logger.error(f"Failed to open RTL-SDR device: {e}")
            return False

    def close(self) -> None:
        """Close RTL-SDR device"""
        try:
            # Stop streaming first
            self.stop_streaming()

            if self.rtlsdr:
                self.rtlsdr.close()
                self.rtlsdr = None

            self.is_open = False
            self.logger.info("RTL-SDR device closed")

        except Exception as e:
            self.logger.error(f"Error closing RTL-SDR device: {e}")

    def start_streaming(self) -> bool:
        """Start sample streaming using async read"""
        try:
            if not self.is_open:
                self.logger.error("Device not open")
                return False

            if self.is_streaming:
                return True

            # Clear buffer
            while not self.async_buffer.empty():
                try:
                    self.async_buffer.get_nowait()
                except:
                    break

            # Start async read
            self.rtlsdr.read_samples_async(
                callback=self._async_callback,
                num_samples=self.buffer_size,
                context=None
            )

            self.is_streaming = True
            self.logger.info("RTL-SDR streaming started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start RTL-SDR streaming: {e}")
            return False

    def stop_streaming(self) -> None:
        """Stop sample streaming"""
        try:
            if self.rtlsdr and self.is_streaming:
                self.rtlsdr.cancel_read_async()

                # Give some time for the async read to stop
                time.sleep(0.1)

            self.is_streaming = False
            self.logger.info("RTL-SDR streaming stopped")

        except Exception as e:
            self.logger.error(f"Error stopping RTL-SDR streaming: {e}")

    def _async_callback(self, samples, context):
        """Callback function for async sample reading"""
        try:
            # Convert to complex64 if needed
            if samples.dtype != np.complex64:
                samples = samples.astype(np.complex64)

            # Add samples to buffer
            try:
                self.async_buffer.put(samples, block=False)
                self.samples_received += len(samples)
            except queue.Full:
                # Buffer overflow
                self.overruns += 1

        except Exception as e:
            self.logger.error(f"Async callback error: {e}")

    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from device"""
        if not self.is_streaming:
            # Synchronous read
            if not self.is_open:
                return None

            try:
                samples = self.rtlsdr.read_samples(num_samples)
                if samples is not None:
                    self.samples_received += len(samples)
                    return samples.astype(np.complex64)
                return None
            except Exception as e:
                self.logger.error(f"Error reading samples: {e}")
                return None

        else:
            # Asynchronous read from buffer
            try:
                samples = []
                samples_needed = num_samples

                # Collect samples from buffer
                while samples_needed > 0 and not self.async_buffer.empty():
                    try:
                        buffer_samples = self.async_buffer.get(timeout=0.1)

                        if len(buffer_samples) <= samples_needed:
                            samples.extend(buffer_samples)
                            samples_needed -= len(buffer_samples)
                        else:
                            # Take only what we need
                            samples.extend(buffer_samples[:samples_needed])
                            # Put the rest back (this is not ideal but works for now)
                            remaining = buffer_samples[samples_needed:]
                            try:
                                self.async_buffer.put(remaining, block=False)
                            except queue.Full:
                                pass  # Lost some samples
                            samples_needed = 0

                    except queue.Empty:
                        break

                if samples:
                    return np.array(samples, dtype=np.complex64)
                else:
                    return None

            except Exception as e:
                self.logger.error(f"Error reading async samples: {e}")
                return None

    def set_center_frequency(self, freq: float) -> bool:
        """Set center frequency"""
        try:
            if self.rtlsdr:
                self.rtlsdr.center_freq = freq
                self.center_freq = freq
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting frequency: {e}")
            return False

    def set_sample_rate(self, rate: float) -> bool:
        """Set sample rate"""
        try:
            if self.rtlsdr:
                self.rtlsdr.sample_rate = rate
                self.sample_rate = rate
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting sample rate: {e}")
            return False

    def set_gain(self, gain: float) -> bool:
        """Set RF gain"""
        try:
            if self.rtlsdr:
                if gain == 0:
                    # Auto gain
                    self.rtlsdr.gain = 'auto'
                else:
                    # Manual gain - find closest valid gain
                    if self.valid_gains:
                        closest_gain = min(self.valid_gains, key=lambda x: abs(x/10.0 - gain))
                        self.rtlsdr.gain = closest_gain / 10.0
                        self.gain = closest_gain / 10.0
                    else:
                        self.rtlsdr.gain = gain
                        self.gain = gain
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting gain: {e}")
            return False

    def set_bandwidth(self, bandwidth: float) -> bool:
        """Set bandwidth (not directly supported by RTL-SDR)"""
        self.logger.warning("RTL-SDR does not support bandwidth setting")
        self.bandwidth = bandwidth
        return True

    def get_supported_sample_rates(self) -> List[float]:
        """Get supported sample rates"""
        # Common RTL-SDR sample rates
        return [
            250000, 1000000, 1024000, 1800000, 1920000, 
            2048000, 2400000, 2880000, 3200000
        ]

    def get_frequency_range(self) -> tuple:
        """Get frequency range"""
        # Standard RTL-SDR frequency range
        return (24e6, 1.766e9)  # 24 MHz to 1766 MHz

    def get_gain_range(self) -> tuple:
        """Get gain range"""
        if self.valid_gains:
            gains_db = [g/10.0 for g in self.valid_gains]
            return (min(gains_db), max(gains_db))
        else:
            return (0, 50)  # Typical range

    def get_valid_gains(self) -> List[float]:
        """Get list of valid gain values"""
        if self.valid_gains:
            return [g/10.0 for g in self.valid_gains]
        else:
            return []

    def set_frequency_correction(self, ppm: int) -> bool:
        """Set frequency correction in PPM"""
        try:
            if self.rtlsdr:
                self.rtlsdr.freq_correction = ppm
                self.freq_correction = ppm
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting frequency correction: {e}")
            return False

    def set_offset_tuning(self, enabled: bool) -> bool:
        """Enable/disable offset tuning"""
        try:
            if self.rtlsdr:
                self.rtlsdr.set_offset_tuning(enabled)
                self.offset_tuning = enabled
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Offset tuning not supported: {e}")
            return False

    def set_bias_tee(self, enabled: bool) -> bool:
        """Enable/disable bias tee"""
        try:
            if self.rtlsdr:
                self.rtlsdr.set_bias_tee(enabled)
                self.bias_tee = enabled
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Bias tee not supported: {e}")
            return False

    def log_device_info(self):
        """Log RTL-SDR device information"""
        if not self.rtlsdr:
            return

        try:
            info = {
                'device_name': self.device_info.name,
                'device_index': self.device_index,
                'center_freq': f"{self.center_freq/1e6:.3f} MHz",
                'sample_rate': f"{self.sample_rate/1e6:.3f} MSps",
                'gain': f"{self.gain:.1f} dB",
                'freq_correction': f"{self.freq_correction} PPM",
                'offset_tuning': self.offset_tuning,
                'bias_tee': self.bias_tee
            }

            # Add valid gains
            if self.valid_gains:
                gains_str = ', '.join([f"{g/10.0:.1f}" for g in self.valid_gains])
                info['valid_gains'] = f"{gains_str} dB"

            # Get actual values from device
            try:
                info['actual_freq'] = f"{self.rtlsdr.center_freq/1e6:.3f} MHz"
                info['actual_rate'] = f"{self.rtlsdr.sample_rate/1e6:.3f} MSps"
                info['actual_gain'] = f"{self.rtlsdr.gain:.1f} dB"
            except:
                pass

            self.sdr_logger.log_device_info(info)

        except Exception as e:
            self.logger.error(f"Error logging device info: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get device status"""
        status = super().get_status()

        # Add RTL-SDR specific status
        if self.rtlsdr:
            try:
                status.update({
                    'freq_correction': self.freq_correction,
                    'offset_tuning': self.offset_tuning,
                    'bias_tee': self.bias_tee,
                    'valid_gains': self.get_valid_gains(),
                    'actual_freq': self.rtlsdr.center_freq,
                    'actual_rate': self.rtlsdr.sample_rate,
                    'actual_gain': self.rtlsdr.gain
                })
            except Exception as e:
                status['status_error'] = str(e)

        return status


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test device enumeration
    devices = RTLSDRBackend.enumerate_devices()
    print(f"Found {len(devices)} RTL-SDR devices:")
    for device in devices:
        print(f"  {device}")

    if devices:
        # Test opening first device
        from config.settings import AppSettings
        settings = AppSettings()

        backend = RTLSDRBackend(devices[0], settings)

        if backend.open():
            print("Device opened successfully")

            # Print device info
            print(f"Frequency range: {backend.get_frequency_range()}")
            print(f"Gain range: {backend.get_gain_range()}")
            print(f"Valid gains: {backend.get_valid_gains()}")
            print(f"Sample rates: {backend.get_supported_sample_rates()}")

            # Test reading some samples
            samples = backend.read_samples(1024)
            if samples is not None:
                print(f"Read {len(samples)} samples")
                print(f"Sample type: {samples.dtype}")
                print(f"Power: {10*np.log10(np.mean(np.abs(samples)**2)):.1f} dB")

            backend.close()
        else:
            print("Failed to open device")
