"""
HackRF Backend Implementation

Provides support for HackRF One SDR devices.
HackRF is a popular wideband SDR platform with both RX and TX capabilities.

Note: This implementation assumes pyhackrf or similar library availability.
Since there's no standardized HackRF Python library, this provides a template
that can be adapted for specific HackRF Python bindings.
"""

import numpy as np
import logging
import threading
import time
import queue
import ctypes
from typing import List, Dict, Any, Optional

try:
    # Try to import HackRF library - this may vary depending on the binding used
    # Common options: pyhackrf, hackrf-tools, or custom ctypes wrapper
    import hackrf  # This is a placeholder - actual import will depend on library
    HACKRF_AVAILABLE = True
except ImportError:
    HACKRF_AVAILABLE = False
    hackrf = None

from core.sdr_backend import SDRBackend, DeviceInfo
from config.settings import AppSettings


class HackRFBackend(SDRBackend):
    """HackRF backend implementation"""

    def __init__(self, device_info: DeviceInfo, settings: AppSettings):
        super().__init__(device_info, settings)

        if not HACKRF_AVAILABLE:
            raise ImportError("HackRF library not available")

        self.hackrf_device = None
        self.device_index = device_info.index

        # HackRF specific parameters
        self.amp_enable = False          # RF amplifier
        self.antenna_enable = False      # Antenna port power
        self.lna_gain = 16              # LNA gain (0-40 dB)
        self.vga_gain = 20              # VGA gain (0-62 dB) 
        self.baseband_filter_bw = 1.75e6  # Baseband filter bandwidth

        # Sample buffers and threading
        self.rx_buffer = queue.Queue(maxsize=100)
        self.rx_thread = None
        self.receiving = False

        # Performance counters
        self.rx_callback_count = 0
        self.rx_samples_lost = 0

    @staticmethod
    def enumerate_devices() -> List[DeviceInfo]:
        """Enumerate available HackRF devices"""
        if not HACKRF_AVAILABLE:
            return []

        devices = []
        try:
            # This will depend on the specific HackRF Python binding
            # Example implementation:

            # Get device count
            device_count = hackrf.get_device_count() if hasattr(hackrf, 'get_device_count') else 0

            for i in range(device_count):
                try:
                    # Get device serial number
                    serial = hackrf.get_device_serial(i) if hasattr(hackrf, 'get_device_serial') else f"hackrf_{i}"

                    # Create device info
                    device_info = DeviceInfo(
                        name=f"HackRF One #{i}",
                        driver='hackrf',
                        serial=serial,
                        index=i,
                        available=True,
                        capabilities={
                            'frequency_range': (1e6, 6e9),  # 1 MHz to 6 GHz
                            'sample_rates': [8e6, 10e6, 12.5e6, 16e6, 20e6],  # Common rates
                            'tx_capable': True,
                            'half_duplex': True
                        }
                    )

                    devices.append(device_info)

                except Exception as e:
                    logging.getLogger(__name__).warning(f"Error probing HackRF device {i}: {e}")

            # If no devices found through library, try fallback detection
            if not devices:
                # Create a dummy device for testing/development
                dummy_device = DeviceInfo(
                    name="HackRF One (Not Connected)",
                    driver='hackrf',
                    serial="hackrf_dummy",
                    index=0,
                    available=False,
                    capabilities={
                        'frequency_range': (1e6, 6e9),
                        'sample_rates': [8e6, 10e6, 12.5e6, 16e6, 20e6],
                        'tx_capable': True,
                        'half_duplex': True,
                        'dummy': True
                    }
                )
                devices.append(dummy_device)

        except Exception as e:
            logging.getLogger(__name__).error(f"Error enumerating HackRF devices: {e}")

        return devices

    def open(self) -> bool:
        """Open HackRF device"""
        try:
            if self.is_open:
                return True

            # Check if this is a dummy device
            if self.device_info.capabilities.get('dummy', False):
                self.logger.warning("Opening dummy HackRF device")
                self.hackrf_device = "dummy"
                self.is_open = True
                return True

            # Open actual HackRF device
            # This will depend on the specific library binding
            if hasattr(hackrf, 'open'):
                self.hackrf_device = hackrf.open(self.device_index)
            elif hasattr(hackrf, 'HackRF'):
                self.hackrf_device = hackrf.HackRF()
                self.hackrf_device.open()
            else:
                raise RuntimeError("No known method to open HackRF device")

            # Set initial parameters
            self.set_center_frequency(self.center_freq)
            self.set_sample_rate(self.sample_rate)
            self.set_gain(self.gain)

            # Set HackRF-specific parameters
            if self.bandwidth:
                self.set_baseband_filter_bandwidth(self.bandwidth)

            self.is_open = True
            self.logger.info(f"HackRF device opened: {self.device_info.name}")

            # Log device info
            self.log_device_info()

            return True

        except Exception as e:
            self.logger.error(f"Failed to open HackRF device: {e}")
            return False

    def close(self) -> None:
        """Close HackRF device"""
        try:
            # Stop streaming first
            self.stop_streaming()

            if self.hackrf_device and self.hackrf_device != "dummy":
                if hasattr(self.hackrf_device, 'close'):
                    self.hackrf_device.close()
                elif hasattr(hackrf, 'close'):
                    hackrf.close(self.hackrf_device)

                self.hackrf_device = None

            self.is_open = False
            self.logger.info("HackRF device closed")

        except Exception as e:
            self.logger.error(f"Error closing HackRF device: {e}")

    def start_streaming(self) -> bool:
        """Start RX streaming"""
        try:
            if not self.is_open:
                self.logger.error("Device not open")
                return False

            if self.is_streaming:
                return True

            # Handle dummy device
            if self.hackrf_device == "dummy":
                self.is_streaming = True
                self.receiving = True
                self.rx_thread = threading.Thread(target=self._dummy_rx_loop, daemon=True)
                self.rx_thread.start()
                self.logger.info("HackRF dummy streaming started")
                return True

            # Clear buffer
            while not self.rx_buffer.empty():
                try:
                    self.rx_buffer.get_nowait()
                except:
                    break

            # Start RX - this depends on the library binding
            self.receiving = True

            if hasattr(self.hackrf_device, 'start_rx'):
                # Method 1: Direct start_rx method
                self.hackrf_device.start_rx(callback=self._rx_callback)
            elif hasattr(hackrf, 'start_rx'):
                # Method 2: Module-level function
                hackrf.start_rx(self.hackrf_device, self._rx_callback)
            else:
                # Method 3: Manual threading with read operations
                self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
                self.rx_thread.start()

            self.is_streaming = True
            self.logger.info("HackRF streaming started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            return False

    def stop_streaming(self) -> None:
        """Stop RX streaming"""
        try:
            self.receiving = False

            if self.hackrf_device and self.hackrf_device != "dummy":
                if hasattr(self.hackrf_device, 'stop_rx'):
                    self.hackrf_device.stop_rx()
                elif hasattr(hackrf, 'stop_rx'):
                    hackrf.stop_rx(self.hackrf_device)

            # Wait for RX thread to finish
            if self.rx_thread:
                self.rx_thread.join(timeout=1.0)
                self.rx_thread = None

            self.is_streaming = False
            self.logger.info("HackRF streaming stopped")

        except Exception as e:
            self.logger.error(f"Error stopping streaming: {e}")

    def _rx_callback(self, samples):
        """RX callback for sample data"""
        try:
            self.rx_callback_count += 1

            # Convert samples to complex64
            if isinstance(samples, (bytes, bytearray)):
                # Convert I/Q bytes to complex samples
                iq_samples = np.frombuffer(samples, dtype=np.int8)
                i_samples = iq_samples[0::2].astype(np.float32) / 128.0
                q_samples = iq_samples[1::2].astype(np.float32) / 128.0
                complex_samples = i_samples + 1j * q_samples
            else:
                complex_samples = np.array(samples, dtype=np.complex64)

            # Add to buffer
            try:
                self.rx_buffer.put(complex_samples, block=False)
                self.samples_received += len(complex_samples)
            except queue.Full:
                self.rx_samples_lost += len(complex_samples)
                # Remove old samples to make room
                try:
                    self.rx_buffer.get_nowait()
                    self.rx_buffer.put(complex_samples, block=False)
                except:
                    pass

        except Exception as e:
            self.logger.error(f"RX callback error: {e}")

    def _rx_loop(self):
        """Manual RX loop for libraries without callback support"""
        while self.receiving:
            try:
                # This would need to be implemented based on the specific library
                # For example, if the library has a read_samples method:
                if hasattr(self.hackrf_device, 'read_samples'):
                    samples = self.hackrf_device.read_samples(1024)
                    if samples is not None:
                        self._rx_callback(samples)

                time.sleep(0.001)  # Small delay

            except Exception as e:
                self.logger.error(f"RX loop error: {e}")
                time.sleep(0.1)

    def _dummy_rx_loop(self):
        """Dummy RX loop for testing without hardware"""
        np.random.seed(42)

        while self.receiving:
            try:
                # Generate dummy samples
                num_samples = 1024
                noise = np.random.normal(0, 0.1, num_samples) + 1j * np.random.normal(0, 0.1, num_samples)

                # Add a test signal
                t = np.arange(num_samples) / self.sample_rate
                test_signal = 0.3 * np.exp(1j * 2 * np.pi * 1e6 * t)  # 1 MHz test tone
                samples = (noise + test_signal).astype(np.complex64)

                # Add to buffer
                try:
                    self.rx_buffer.put(samples, block=False)
                    self.samples_received += len(samples)
                except queue.Full:
                    # Buffer full, skip samples
                    self.rx_samples_lost += len(samples)

                time.sleep(0.01)  # Simulate realistic timing

            except Exception as e:
                self.logger.error(f"Dummy RX loop error: {e}")
                time.sleep(0.1)

    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from buffer"""
        if not self.is_streaming:
            return None

        try:
            samples = []
            samples_needed = num_samples
            timeout = 0.1  # 100ms timeout
            start_time = time.time()

            while samples_needed > 0 and (time.time() - start_time) < timeout:
                try:
                    buffer_samples = self.rx_buffer.get(timeout=0.01)

                    if len(buffer_samples) <= samples_needed:
                        samples.extend(buffer_samples)
                        samples_needed -= len(buffer_samples)
                    else:
                        # Take only what we need
                        samples.extend(buffer_samples[:samples_needed])
                        # Put remainder back
                        remaining = buffer_samples[samples_needed:]
                        try:
                            self.rx_buffer.put(remaining, block=False)
                        except queue.Full:
                            self.rx_samples_lost += len(remaining)
                        samples_needed = 0

                except queue.Empty:
                    continue

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
            if self.hackrf_device and self.hackrf_device != "dummy":
                if hasattr(self.hackrf_device, 'set_freq'):
                    self.hackrf_device.set_freq(freq)
                elif hasattr(hackrf, 'set_freq'):
                    hackrf.set_freq(self.hackrf_device, freq)

            self.center_freq = freq
            return True
        except Exception as e:
            self.logger.error(f"Error setting frequency: {e}")
            return False

    def set_sample_rate(self, rate: float) -> bool:
        """Set sample rate"""
        try:
            if self.hackrf_device and self.hackrf_device != "dummy":
                if hasattr(self.hackrf_device, 'set_sample_rate'):
                    self.hackrf_device.set_sample_rate(rate)
                elif hasattr(hackrf, 'set_sample_rate'):
                    hackrf.set_sample_rate(self.hackrf_device, rate)

            self.sample_rate = rate
            return True
        except Exception as e:
            self.logger.error(f"Error setting sample rate: {e}")
            return False

    def set_gain(self, gain: float) -> bool:
        """Set gains (LNA and VGA)"""
        try:
            # HackRF has separate LNA and VGA gains
            # Split the total gain between them
            total_gain = max(0, min(gain, 102))  # Limit to 0-102 dB

            # LNA: 0-40 dB in 8 dB steps
            self.lna_gain = min(40, int(total_gain * 0.4))
            self.lna_gain = (self.lna_gain // 8) * 8  # Round to 8 dB steps

            # VGA: 0-62 dB in 2 dB steps  
            remaining_gain = total_gain - self.lna_gain
            self.vga_gain = min(62, int(remaining_gain))
            self.vga_gain = (self.vga_gain // 2) * 2  # Round to 2 dB steps

            if self.hackrf_device and self.hackrf_device != "dummy":
                if hasattr(self.hackrf_device, 'set_lna_gain'):
                    self.hackrf_device.set_lna_gain(self.lna_gain)
                if hasattr(self.hackrf_device, 'set_vga_gain'):
                    self.hackrf_device.set_vga_gain(self.vga_gain)
                elif hasattr(hackrf, 'set_lna_gain') and hasattr(hackrf, 'set_vga_gain'):
                    hackrf.set_lna_gain(self.hackrf_device, self.lna_gain)
                    hackrf.set_vga_gain(self.hackrf_device, self.vga_gain)

            self.gain = self.lna_gain + self.vga_gain
            return True
        except Exception as e:
            self.logger.error(f"Error setting gain: {e}")
            return False

    def set_bandwidth(self, bandwidth: float) -> bool:
        """Set baseband filter bandwidth"""
        return self.set_baseband_filter_bandwidth(bandwidth)

    def set_baseband_filter_bandwidth(self, bandwidth: float) -> bool:
        """Set baseband filter bandwidth"""
        try:
            # HackRF supports specific bandwidth values
            valid_bandwidths = [1.75e6, 2.5e6, 3.5e6, 5e6, 5.5e6, 6e6, 7e6, 8e6, 9e6, 10e6, 12e6, 14e6, 15e6, 20e6, 24e6, 28e6]
            closest_bw = min(valid_bandwidths, key=lambda x: abs(x - bandwidth))

            if self.hackrf_device and self.hackrf_device != "dummy":
                if hasattr(self.hackrf_device, 'set_baseband_filter_bandwidth'):
                    self.hackrf_device.set_baseband_filter_bandwidth(closest_bw)
                elif hasattr(hackrf, 'set_baseband_filter_bandwidth'):
                    hackrf.set_baseband_filter_bandwidth(self.hackrf_device, closest_bw)

            self.baseband_filter_bw = closest_bw
            self.bandwidth = closest_bw
            return True
        except Exception as e:
            self.logger.error(f"Error setting bandwidth: {e}")
            return False

    def set_amp_enable(self, enable: bool) -> bool:
        """Enable/disable RF amplifier"""
        try:
            if self.hackrf_device and self.hackrf_device != "dummy":
                if hasattr(self.hackrf_device, 'set_amp_enable'):
                    self.hackrf_device.set_amp_enable(enable)
                elif hasattr(hackrf, 'set_amp_enable'):
                    hackrf.set_amp_enable(self.hackrf_device, enable)

            self.amp_enable = enable
            return True
        except Exception as e:
            self.logger.error(f"Error setting amp enable: {e}")
            return False

    def get_supported_sample_rates(self) -> List[float]:
        """Get supported sample rates"""
        # HackRF supports 8-20 MHz sample rates
        return [8e6, 10e6, 12.5e6, 16e6, 20e6]

    def get_frequency_range(self) -> tuple:
        """Get frequency range"""
        return (1e6, 6e9)  # 1 MHz to 6 GHz

    def get_gain_range(self) -> tuple:
        """Get gain range"""
        return (0, 102)  # 0 to 102 dB (LNA + VGA)

    def log_device_info(self):
        """Log HackRF device information"""
        try:
            info = {
                'device_name': self.device_info.name,
                'center_freq': f"{self.center_freq/1e6:.3f} MHz",
                'sample_rate': f"{self.sample_rate/1e6:.3f} MSps",
                'lna_gain': f"{self.lna_gain:.0f} dB",
                'vga_gain': f"{self.vga_gain:.0f} dB",
                'total_gain': f"{self.gain:.0f} dB",
                'baseband_filter_bw': f"{self.baseband_filter_bw/1e6:.1f} MHz",
                'amp_enable': self.amp_enable,
                'antenna_enable': self.antenna_enable
            }

            if self.hackrf_device == "dummy":
                info['note'] = "Dummy device (no hardware)"

            self.sdr_logger.log_device_info(info)

        except Exception as e:
            self.logger.error(f"Error logging device info: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get device status"""
        status = super().get_status()

        # Add HackRF specific status
        status.update({
            'lna_gain': self.lna_gain,
            'vga_gain': self.vga_gain,
            'baseband_filter_bw': self.baseband_filter_bw,
            'amp_enable': self.amp_enable,
            'antenna_enable': self.antenna_enable,
            'rx_callback_count': self.rx_callback_count,
            'rx_samples_lost': self.rx_samples_lost
        })

        return status


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test device enumeration
    devices = HackRFBackend.enumerate_devices()
    print(f"Found {len(devices)} HackRF devices:")
    for device in devices:
        print(f"  {device}")

    if devices:
        # Test opening first device
        from config.settings import AppSettings
        settings = AppSettings()

        backend = HackRFBackend(devices[0], settings)

        if backend.open():
            print("Device opened successfully")

            # Print device capabilities
            print(f"Frequency range: {backend.get_frequency_range()}")
            print(f"Gain range: {backend.get_gain_range()}")
            print(f"Sample rates: {backend.get_supported_sample_rates()}")

            # Test reading samples
            if backend.start_streaming():
                time.sleep(0.1)  # Let it collect some samples
                samples = backend.read_samples(1024)
                if samples is not None:
                    print(f"Read {len(samples)} samples")
                    print(f"Sample type: {samples.dtype}")
                    print(f"Power: {10*np.log10(np.mean(np.abs(samples)**2)):.1f} dB")
                backend.stop_streaming()

            backend.close()
        else:
            print("Failed to open device")
