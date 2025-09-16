"""
PlutoSDR Backend Implementation

Provides support for Analog Devices ADALM-Pluto SDR using pyadi-iio library.
PlutoSDR is a learning module and software defined radio active learning module.
"""

import numpy as np
import logging
import threading
import time
import queue
from typing import List, Dict, Any, Optional

try:
    import adi
    PLUTO_AVAILABLE = True
except ImportError:
    PLUTO_AVAILABLE = False
    adi = None

from core.sdr_backend import SDRBackend, DeviceInfo  
from config.settings import AppSettings


class PlutoBackend(SDRBackend):
    """PlutoSDR backend implementation"""

    def __init__(self, device_info: DeviceInfo, settings: AppSettings):
        super().__init__(device_info, settings)

        if not PLUTO_AVAILABLE:
            raise ImportError("pyadi-iio not available")

        self.pluto = None
        self.uri = device_info.capabilities.get('uri', 'ip:192.168.2.1')

        # PlutoSDR specific parameters
        self.rf_bandwidth = settings.sdr.bandwidth or 20e6  # 20 MHz default
        self.buffer_size = settings.processing.buffer_size

        # Threading for continuous acquisition
        self.acquisition_thread = None
        self.acquiring = False
        self.sample_queue = queue.Queue(maxsize=100)

        # Performance counters
        self.buffer_underruns = 0
        self.buffer_overruns = 0

    @staticmethod
    def enumerate_devices() -> List[DeviceInfo]:
        """Enumerate available PlutoSDR devices"""
        if not PLUTO_AVAILABLE:
            return []

        devices = []

        # Common PlutoSDR URIs to try
        uris_to_try = [
            'ip:192.168.2.1',      # Default PlutoSDR IP
            'ip:pluto.local',      # mDNS name
            'usb:1.2.5',          # USB connection
            'usb:',               # Any USB Pluto
        ]

        for i, uri in enumerate(uris_to_try):
            try:
                # Try to connect to Pluto
                test_pluto = adi.Pluto(uri=uri)

                # Get device info
                device_name = f"ADALM-Pluto ({uri})"

                # Try to get hardware serial/info
                try:
                    hw_info = test_pluto._ctrl.find_device("ad9361-phy")
                    serial = uri  # Use URI as serial for now
                except:
                    serial = f"pluto_{i}"

                # Close test connection
                del test_pluto

                # Create device info
                device_info = DeviceInfo(
                    name=device_name,
                    driver='pluto',
                    serial=serial,
                    index=i,
                    available=True,
                    capabilities={
                        'uri': uri,
                        'frequency_range': (70e6, 6e9),  # AD9361 range
                        'sample_rates': [520833, 1e6, 2e6, 4e6, 8e6, 16e6, 20e6, 30.72e6, 61.44e6],
                        'max_bandwidth': 56e6,
                        'tx_capable': True
                    }
                )

                devices.append(device_info)
                break  # Found one, don't try other URIs

            except Exception as e:
                # Connection failed, try next URI
                continue

        return devices

    def open(self) -> bool:
        """Open PlutoSDR device"""
        try:
            if self.is_open:
                return True

            # Create PlutoSDR instance
            self.pluto = adi.Pluto(uri=self.uri)

            # Configure RX parameters
            self.pluto.rx_lo = int(self.center_freq)
            self.pluto.sample_rate = int(self.sample_rate)
            self.pluto.rx_rf_bandwidth = int(self.rf_bandwidth)
            self.pluto.rx_buffer_size = self.buffer_size

            # Set gain
            if self.settings.sdr.auto_gain:
                self.pluto.gain_control_mode_chan0 = "slow_attack"
            else:
                self.pluto.gain_control_mode_chan0 = "manual"
                self.pluto.rx_hardwaregain_chan0 = self.gain

            # Configure TX parameters (even if not used)
            self.pluto.tx_lo = int(self.center_freq)
            self.pluto.tx_rf_bandwidth = int(self.rf_bandwidth)
            self.pluto.tx_hardwaregain_chan0 = -20  # Safe TX gain

            self.is_open = True
            self.logger.info(f"PlutoSDR opened: {self.uri}")

            # Log device information
            self.log_device_info()

            return True

        except Exception as e:
            self.logger.error(f"Failed to open PlutoSDR: {e}")
            return False

    def close(self) -> None:
        """Close PlutoSDR device"""
        try:
            # Stop acquisition first
            self.stop_streaming()

            if self.pluto:
                # Clean up
                try:
                    del self.pluto
                except:
                    pass
                self.pluto = None

            self.is_open = False
            self.logger.info("PlutoSDR closed")

        except Exception as e:
            self.logger.error(f"Error closing PlutoSDR: {e}")

    def start_streaming(self) -> bool:
        """Start continuous sample acquisition"""
        try:
            if not self.is_open:
                self.logger.error("Device not open")
                return False

            if self.is_streaming:
                return True

            # Clear sample queue
            while not self.sample_queue.empty():
                try:
                    self.sample_queue.get_nowait()
                except:
                    break

            # Start acquisition thread
            self.acquiring = True
            self.acquisition_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
            self.acquisition_thread.start()

            self.is_streaming = True
            self.logger.info("PlutoSDR streaming started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            return False

    def stop_streaming(self) -> None:
        """Stop sample acquisition"""
        try:
            self.acquiring = False

            # Wait for acquisition thread to stop
            if self.acquisition_thread:
                self.acquisition_thread.join(timeout=1.0)
                self.acquisition_thread = None

            self.is_streaming = False
            self.logger.info("PlutoSDR streaming stopped")

        except Exception as e:
            self.logger.error(f"Error stopping streaming: {e}")

    def _acquisition_loop(self):
        """Continuous acquisition loop running in separate thread"""
        while self.acquiring:
            try:
                # Receive samples from PlutoSDR
                samples = self.pluto.rx()

                if samples is not None and len(samples) > 0:
                    # Convert to complex64
                    if samples.dtype != np.complex64:
                        samples = samples.astype(np.complex64)

                    # Add to queue
                    try:
                        self.sample_queue.put(samples, block=False)
                        self.samples_received += len(samples)
                    except queue.Full:
                        # Queue full - increment overrun counter
                        self.buffer_overruns += 1
                        # Remove oldest samples to make room
                        try:
                            self.sample_queue.get_nowait()
                            self.sample_queue.put(samples, block=False)
                        except:
                            pass

                # Small delay to prevent CPU overload
                time.sleep(0.001)

            except Exception as e:
                self.logger.error(f"Acquisition loop error: {e}")
                time.sleep(0.1)

    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from device"""
        if not self.is_streaming:
            # Synchronous read
            if not self.is_open:
                return None

            try:
                # Set buffer size for this read
                original_buffer_size = self.pluto.rx_buffer_size
                self.pluto.rx_buffer_size = num_samples

                # Receive samples
                samples = self.pluto.rx()

                # Restore original buffer size
                self.pluto.rx_buffer_size = original_buffer_size

                if samples is not None:
                    self.samples_received += len(samples)
                    return samples.astype(np.complex64)

                return None

            except Exception as e:
                self.logger.error(f"Error reading samples: {e}")
                return None

        else:
            # Asynchronous read from queue
            try:
                samples = []
                samples_needed = num_samples

                # Collect samples from queue
                while samples_needed > 0:
                    try:
                        buffer_samples = self.sample_queue.get(timeout=0.1)

                        if len(buffer_samples) <= samples_needed:
                            samples.extend(buffer_samples)
                            samples_needed -= len(buffer_samples)
                        else:
                            # Take only what we need
                            samples.extend(buffer_samples[:samples_needed])
                            # Put remaining samples back
                            remaining = buffer_samples[samples_needed:]
                            try:
                                self.sample_queue.put(remaining, block=False)
                            except queue.Full:
                                self.buffer_overruns += 1
                            samples_needed = 0

                        if samples_needed <= 0:
                            break

                    except queue.Empty:
                        # No more samples available
                        self.buffer_underruns += 1
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
            if self.pluto:
                self.pluto.rx_lo = int(freq)
                self.pluto.tx_lo = int(freq)  # Keep TX in sync
                self.center_freq = freq
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting frequency: {e}")
            return False

    def set_sample_rate(self, rate: float) -> bool:
        """Set sample rate"""
        try:
            if self.pluto:
                self.pluto.sample_rate = int(rate)
                self.sample_rate = rate
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting sample rate: {e}")
            return False

    def set_gain(self, gain: float) -> bool:
        """Set RX gain"""
        try:
            if self.pluto:
                if gain == 0:
                    # Auto gain
                    self.pluto.gain_control_mode_chan0 = "slow_attack"
                else:
                    # Manual gain
                    self.pluto.gain_control_mode_chan0 = "manual"
                    self.pluto.rx_hardwaregain_chan0 = gain
                self.gain = gain
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting gain: {e}")
            return False

    def set_bandwidth(self, bandwidth: float) -> bool:
        """Set RF bandwidth"""
        try:
            if self.pluto:
                self.pluto.rx_rf_bandwidth = int(bandwidth)
                self.pluto.tx_rf_bandwidth = int(bandwidth)
                self.bandwidth = bandwidth
                self.rf_bandwidth = bandwidth
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting bandwidth: {e}")
            return False

    def get_supported_sample_rates(self) -> List[float]:
        """Get supported sample rates"""
        # AD9361 supported rates (approximate)
        return [
            520833,    # Minimum
            1e6, 2e6, 4e6, 8e6, 16e6, 20e6, 
            30.72e6,   # Common LTE rate
            61.44e6    # Maximum
        ]

    def get_frequency_range(self) -> tuple:
        """Get frequency range"""
        # AD9361 frequency range
        return (70e6, 6e9)  # 70 MHz to 6 GHz

    def get_gain_range(self) -> tuple:
        """Get gain range"""
        # AD9361 gain range
        return (-3, 71)  # -3 dB to 71 dB

    def get_bandwidth_range(self) -> tuple:
        """Get bandwidth range"""
        # AD9361 bandwidth range
        return (200e3, 56e6)  # 200 kHz to 56 MHz

    def set_frequency_correction(self, ppm: float) -> bool:
        """Set frequency correction (if supported)"""
        try:
            # PlutoSDR may support XO correction through libadi-iio
            # This is implementation-dependent
            self.logger.warning("Frequency correction may not be supported")
            return True
        except Exception as e:
            self.logger.error(f"Error setting frequency correction: {e}")
            return False

    def get_temperature(self) -> Optional[float]:
        """Get device temperature if available"""
        try:
            if self.pluto:
                # Try to get temperature from AD9361
                temp_channel = self.pluto._ctrl.find_channel("temp0", True)
                if temp_channel:
                    return float(temp_channel.attrs["input"].value) / 1000.0  # Convert mC to C
        except Exception as e:
            self.logger.debug(f"Could not read temperature: {e}")

        return None

    def log_device_info(self):
        """Log PlutoSDR device information"""
        if not self.pluto:
            return

        try:
            info = {
                'uri': self.uri,
                'center_freq': f"{self.center_freq/1e6:.3f} MHz",
                'sample_rate': f"{self.sample_rate/1e6:.3f} MSps", 
                'rf_bandwidth': f"{self.rf_bandwidth/1e6:.1f} MHz",
                'rx_gain': f"{self.gain:.1f} dB",
                'buffer_size': self.buffer_size
            }

            # Get actual values from device
            try:
                info['actual_freq'] = f"{self.pluto.rx_lo/1e6:.3f} MHz"
                info['actual_rate'] = f"{self.pluto.sample_rate/1e6:.3f} MSps"
                info['actual_bandwidth'] = f"{self.pluto.rx_rf_bandwidth/1e6:.1f} MHz"
                info['gain_mode'] = self.pluto.gain_control_mode_chan0
                if hasattr(self.pluto, 'rx_hardwaregain_chan0'):
                    info['actual_gain'] = f"{self.pluto.rx_hardwaregain_chan0:.1f} dB"
            except:
                pass

            # Get temperature if available
            temp = self.get_temperature()
            if temp is not None:
                info['temperature'] = f"{temp:.1f} °C"

            # Get hardware info
            try:
                info['firmware_version'] = getattr(self.pluto, '_device_name', 'Unknown')
            except:
                pass

            self.sdr_logger.log_device_info(info)

        except Exception as e:
            self.logger.error(f"Error logging device info: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get device status"""
        status = super().get_status()

        # Add PlutoSDR specific status
        if self.pluto:
            try:
                status.update({
                    'uri': self.uri,
                    'rf_bandwidth': self.rf_bandwidth,
                    'buffer_underruns': self.buffer_underruns,
                    'buffer_overruns': self.buffer_overruns,
                    'actual_freq': self.pluto.rx_lo,
                    'actual_rate': self.pluto.sample_rate,
                    'actual_bandwidth': self.pluto.rx_rf_bandwidth,
                    'gain_mode': self.pluto.gain_control_mode_chan0
                })

                # Add temperature if available
                temp = self.get_temperature()
                if temp is not None:
                    status['temperature'] = temp

                # Add gain info
                if hasattr(self.pluto, 'rx_hardwaregain_chan0'):
                    status['actual_gain'] = self.pluto.rx_hardwaregain_chan0

            except Exception as e:
                status['status_error'] = str(e)

        return status

    def calibrate(self) -> bool:
        """Perform device calibration if supported"""
        try:
            if self.pluto:
                # Some PlutoSDR operations that might help with calibration
                # This is device/firmware specific
                self.logger.info("PlutoSDR calibration requested")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Calibration error: {e}")
            return False


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test device enumeration
    devices = PlutoBackend.enumerate_devices()
    print(f"Found {len(devices)} PlutoSDR devices:")
    for device in devices:
        print(f"  {device}")

    if devices:
        # Test opening first device
        from config.settings import AppSettings
        settings = AppSettings()

        backend = PlutoBackend(devices[0], settings)

        if backend.open():
            print("Device opened successfully")

            # Print device capabilities
            print(f"Frequency range: {backend.get_frequency_range()}")
            print(f"Gain range: {backend.get_gain_range()}")
            print(f"Bandwidth range: {backend.get_bandwidth_range()}")
            print(f"Sample rates: {backend.get_supported_sample_rates()}")

            # Test reading samples
            samples = backend.read_samples(1024)
            if samples is not None:
                print(f"Read {len(samples)} samples")
                print(f"Sample type: {samples.dtype}")
                print(f"Power: {10*np.log10(np.mean(np.abs(samples)**2)):.1f} dB")

            backend.close()
        else:
            print("Failed to open device")
