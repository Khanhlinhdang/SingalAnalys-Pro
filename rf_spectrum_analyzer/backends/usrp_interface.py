
"""
Advanced USRP Interface Module
Optimized interface for Ettus USRP hardware integration
"""

import numpy as np
import time
import threading
from queue import Queue, Empty
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import uhd
    USRP_AVAILABLE = True
except ImportError:
    USRP_AVAILABLE = False
    print("Warning: UHD not available. USRP functionality disabled.")


class USRPInterface:
    """Advanced USRP interface with optimized streaming"""

    def __init__(self):
        self.usrp = None
        self.rx_streamer = None
        self.tx_streamer = None

        # Configuration
        self.device_args = ""
        self.sample_rate = 1e6
        self.center_freq = 100e6
        self.rx_gain = 30
        self.tx_gain = 30
        self.bandwidth = 0  # Auto

        # Streaming control
        self.streaming = False
        self.stream_thread = None
        self.data_queue = Queue(maxsize=10)

        # Statistics
        self.samples_received = 0
        self.overruns = 0
        self.underruns = 0
        self.last_stats_time = time.time()

    def get_device_list(self):
        """Get list of available USRP devices"""
        if not USRP_AVAILABLE:
            return []

        try:
            # Try different UHD Python API methods for device discovery
            device_list = []
            
            # Method 1: Try creating a USRP instance to detect devices
            try:
                usrp = uhd.usrp.MultiUSRP("")
                device_info = {
                    'index': 0,
                    'type': 'USRP',
                    'serial': 'Unknown',
                    'address': usrp.get_pp_string(),
                    'name': 'USRP Device'
                }
                device_list.append(device_info)
                print(f"Found USRP device: {device_info['name']}")
            except RuntimeError as e:
                if "No devices found" in str(e) or "No UHD devices found" in str(e):
                    print("No USRP devices connected")
                else:
                    print(f"UHD error: {e}")
            except Exception as e:
                print(f"Error detecting USRP: {e}")
            
            return device_list

        except Exception as e:
            print(f"Error getting device list: {e}")
            return []

    def connect(self, device_args="", sample_rate=1e6, center_freq=100e6):
        """Connect to USRP device"""
        if not USRP_AVAILABLE:
            raise RuntimeError("UHD not available")

        try:
            # Store configuration
            self.device_args = device_args
            self.sample_rate = sample_rate
            self.center_freq = center_freq

            # Create USRP device
            print(f"Connecting to USRP with args: {device_args}")
            self.usrp = uhd.usrp.MultiUSRP(device_args)

            # Configure sample rate
            self.usrp.set_rx_rate(sample_rate)
            self.usrp.set_tx_rate(sample_rate)

            # Configure center frequency
            self.usrp.set_rx_freq(center_freq)
            self.usrp.set_tx_freq(center_freq)

            # Configure gains
            self.usrp.set_rx_gain(self.rx_gain)
            self.usrp.set_tx_gain(self.tx_gain)

            # Configure bandwidth (if specified)
            if self.bandwidth > 0:
                self.usrp.set_rx_bandwidth(self.bandwidth)
                self.usrp.set_tx_bandwidth(self.bandwidth)

            # Create streamers
            st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            self.rx_streamer = self.usrp.get_rx_stream(st_args)
            self.tx_streamer = self.usrp.get_tx_stream(st_args)

            print("✅ USRP connected successfully")
            print(f"   Device: {self.get_device_info()}")
            print(f"   Sample Rate: {self.usrp.get_rx_rate()/1e6:.1f} MS/s")
            print(f"   Center Freq: {self.usrp.get_rx_freq()/1e6:.1f} MHz")
            print(f"   RX Gain: {self.usrp.get_rx_gain():.1f} dB")

            return True

        except Exception as e:
            print(f"❌ USRP connection failed: {e}")
            self.usrp = None
            return False

    def disconnect(self):
        """Disconnect from USRP"""
        try:
            # Stop streaming if active
            self.stop_streaming()

            # Cleanup streamers
            self.rx_streamer = None
            self.tx_streamer = None

            # Cleanup device
            self.usrp = None

            print("✅ USRP disconnected")

        except Exception as e:
            print(f"❌ USRP disconnect error: {e}")

    def is_connected(self):
        """Check if USRP is connected"""
        return self.usrp is not None

    def get_device_info(self):
        """Get device information"""
        if not self.usrp:
            return "Not connected"

        try:
            mboard_name = self.usrp.get_mboard_name()
            return f"{mboard_name}"
        except:
            return "Unknown device"

    def set_rx_parameters(self, center_freq=None, gain=None, bandwidth=None, 
                         sample_rate=None, antenna=None):
        """Set RX parameters"""
        if not self.usrp:
            return False

        try:
            if center_freq is not None:
                self.usrp.set_rx_freq(center_freq)
                self.center_freq = center_freq
                print(f"RX Freq: {center_freq/1e6:.3f} MHz")

            if gain is not None:
                self.usrp.set_rx_gain(gain)
                self.rx_gain = gain
                print(f"RX Gain: {gain:.1f} dB")

            if bandwidth is not None:
                if bandwidth > 0:
                    self.usrp.set_rx_bandwidth(bandwidth)
                    self.bandwidth = bandwidth
                    print(f"RX Bandwidth: {bandwidth/1e6:.1f} MHz")

            if sample_rate is not None:
                self.usrp.set_rx_rate(sample_rate)
                self.sample_rate = sample_rate
                print(f"Sample Rate: {sample_rate/1e6:.1f} MS/s")

                # Recreate streamers with new rate
                st_args = uhd.usrp.StreamArgs("fc32", "sc16")
                self.rx_streamer = self.usrp.get_rx_stream(st_args)

            if antenna is not None:
                self.usrp.set_rx_antenna(antenna)
                print(f"RX Antenna: {antenna}")

            return True

        except Exception as e:
            print(f"Error setting RX parameters: {e}")
            return False

    def get_rx_parameters(self):
        """Get current RX parameters"""
        if not self.usrp:
            return {}

        try:
            params = {
                'center_freq': self.usrp.get_rx_freq(),
                'sample_rate': self.usrp.get_rx_rate(),
                'gain': self.usrp.get_rx_gain(),
                'antenna': self.usrp.get_rx_antenna(),
            }

            # Get bandwidth if supported
            try:
                params['bandwidth'] = self.usrp.get_rx_bandwidth()
            except:
                params['bandwidth'] = 0

            return params

        except Exception as e:
            print(f"Error getting RX parameters: {e}")
            return {}

    def start_streaming(self, num_samples=0, duration=0):
        """Start continuous RX streaming"""
        if not self.usrp or not self.rx_streamer:
            return False

        if self.streaming:
            print("Already streaming")
            return True

        try:
            # Clear data queue
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except Empty:
                    break

            # Reset statistics
            self.samples_received = 0
            self.overruns = 0
            self.underruns = 0
            self.last_stats_time = time.time()

            # Start streaming
            self.streaming = True
            self.stream_thread = threading.Thread(
                target=self._stream_worker, 
                args=(num_samples, duration))
            self.stream_thread.daemon = True
            self.stream_thread.start()

            print("✅ USRP streaming started")
            return True

        except Exception as e:
            print(f"❌ Failed to start streaming: {e}")
            self.streaming = False
            return False

    def stop_streaming(self):
        """Stop RX streaming"""
        if not self.streaming:
            return

        try:
            self.streaming = False

            # Wait for stream thread to finish
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2.0)

            print("✅ USRP streaming stopped")

        except Exception as e:
            print(f"❌ Error stopping streaming: {e}")

    def _stream_worker(self, num_samples, duration):
        """Streaming worker thread"""
        try:
            # Setup streaming
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_continuous)
            stream_cmd.stream_now = True
            self.rx_streamer.issue_stream_cmd(stream_cmd)

            # Streaming loop
            buffer_size = int(self.sample_rate * 0.1)  # 100ms buffer
            recv_buffer = np.zeros(buffer_size, dtype=np.complex64)

            start_time = time.time()
            total_samples = 0

            while self.streaming:
                # Check duration limit
                if duration > 0 and (time.time() - start_time) > duration:
                    break

                # Check sample count limit
                if num_samples > 0 and total_samples >= num_samples:
                    break

                # Receive samples
                metadata = uhd.types.RXMetadata()
                samples_received = self.rx_streamer.recv(recv_buffer, metadata)

                if samples_received > 0:
                    # Copy received data
                    data_copy = recv_buffer[:samples_received].copy()

                    # Add to queue (non-blocking)
                    try:
                        self.data_queue.put_nowait(data_copy)
                    except:
                        # Queue full, skip this buffer
                        pass

                    # Update statistics
                    self.samples_received += samples_received
                    total_samples += samples_received

                # Check for errors
                if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                    if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                        self.overruns += 1

                # Brief pause to prevent 100% CPU usage
                time.sleep(0.001)

            # Stop streaming
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_continuous)
            self.rx_streamer.issue_stream_cmd(stream_cmd)

        except Exception as e:
            print(f"Streaming worker error: {e}")
        finally:
            self.streaming = False

    def get_samples(self, timeout=1.0):
        """Get received samples from queue"""
        try:
            return self.data_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_streaming_stats(self):
        """Get streaming statistics"""
        current_time = time.time()
        time_diff = current_time - self.last_stats_time

        if time_diff > 0:
            sample_rate = self.samples_received / time_diff
        else:
            sample_rate = 0

        stats = {
            'samples_received': self.samples_received,
            'sample_rate': sample_rate,
            'overruns': self.overruns,
            'underruns': self.underruns,
            'queue_size': self.data_queue.qsize(),
            'streaming': self.streaming
        }

        return stats

    def transmit_samples(self, samples, metadata=None):
        """Transmit samples"""
        if not self.usrp or not self.tx_streamer:
            return False

        try:
            if metadata is None:
                metadata = uhd.types.TXMetadata()
                metadata.start_of_burst = True
                metadata.end_of_burst = True
                metadata.has_time_spec = False

            samples_sent = self.tx_streamer.send(samples, metadata)
            return samples_sent == len(samples)

        except Exception as e:
            print(f"Transmit error: {e}")
            return False

    def set_time_now(self):
        """Set USRP time to now"""
        if self.usrp:
            try:
                self.usrp.set_time_now(uhd.types.TimeSpec(0.0))
                return True
            except Exception as e:
                print(f"Error setting time: {e}")
                return False
        return False


class USRPSimulator:
    """USRP simulator for testing when hardware not available"""

    def __init__(self):
        self.connected = False
        self.streaming = False
        self.sample_rate = 1e6
        self.center_freq = 100e6
        self.rx_gain = 30
        self.tx_gain = 30

        # Simulation parameters
        self.noise_power = 0.01
        self.signal_generators = []

    def get_device_list(self):
        """Simulate device list"""
        return [{
            'index': 0,
            'type': 'Simulated USRP',
            'serial': 'SIM001',
            'address': '127.0.0.1',
            'name': 'USRP Simulator'
        }]

    def connect(self, device_args="", sample_rate=1e6, center_freq=100e6):
        """Simulate connection"""
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.connected = True

        print("✅ USRP Simulator connected")
        print(f"   Sample Rate: {sample_rate/1e6:.1f} MS/s")
        print(f"   Center Freq: {center_freq/1e6:.1f} MHz")

        return True

    def disconnect(self):
        """Simulate disconnect"""
        self.connected = False
        self.streaming = False
        print("✅ USRP Simulator disconnected")

    def is_connected(self):
        """Check connection status"""
        return self.connected

    def get_device_info(self):
        """Get simulated device info"""
        return "USRP Simulator"

    def set_rx_parameters(self, center_freq=None, gain=None, bandwidth=None, 
                         sample_rate=None, antenna=None):
        """Set simulated parameters"""
        if center_freq is not None:
            self.center_freq = center_freq
        if gain is not None:
            self.rx_gain = gain
        if sample_rate is not None:
            self.sample_rate = sample_rate

        return True

    def get_rx_parameters(self):
        """Get simulated parameters"""
        return {
            'center_freq': self.center_freq,
            'sample_rate': self.sample_rate,
            'gain': self.rx_gain,
            'bandwidth': self.sample_rate,
            'antenna': 'RX2'
        }

    def start_streaming(self, num_samples=0, duration=0):
        """Start simulated streaming"""
        self.streaming = True
        print("✅ USRP Simulator streaming started")
        return True

    def stop_streaming(self):
        """Stop simulated streaming"""
        self.streaming = False
        print("✅ USRP Simulator streaming stopped")

    def get_samples(self, timeout=1.0):
        """Generate simulated samples"""
        if not self.streaming:
            return None

        # Generate 1000 samples of noise + optional signals
        num_samples = 1000
        t = np.arange(num_samples) / self.sample_rate

        # Base noise
        signal = np.sqrt(self.noise_power) * (
            np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

        # Add test signals
        # Simple sine wave at offset frequency
        test_freq = 10000  # 10 kHz offset
        test_signal = 0.1 * np.exp(1j * 2 * np.pi * test_freq * t)
        signal += test_signal

        return signal.astype(np.complex64)

    def get_streaming_stats(self):
        """Get simulated statistics"""
        return {
            'samples_received': 1000000,
            'sample_rate': self.sample_rate,
            'overruns': 0,
            'underruns': 0,
            'queue_size': 5,
            'streaming': self.streaming
        }

    def transmit_samples(self, samples, metadata=None):
        """Simulate transmission"""
        return True

    def set_time_now(self):
        """Simulate time setting"""
        return True


# Factory function to create appropriate interface
def create_usrp_interface(use_simulator=False):
    """Create USRP interface or simulator"""
    if use_simulator or not USRP_AVAILABLE:
        return USRPSimulator()
    else:
        return USRPInterface()


def test_usrp_interface():
    """Test USRP interface functionality"""
    print("🧪 Testing USRP Interface")
    print("=" * 40)

    # Test with simulator
    print("Testing with simulator...")
    usrp = create_usrp_interface(use_simulator=True)

    # Get device list
    devices = usrp.get_device_list()
    print(f"Found {len(devices)} devices:")
    for device in devices:
        print(f"  {device}")

    # Connect
    if usrp.connect(sample_rate=1e6, center_freq=100e6):
        print(f"Connected: {usrp.get_device_info()}")

        # Get parameters
        params = usrp.get_rx_parameters()
        print(f"RX Parameters: {params}")

        # Test streaming
        if usrp.start_streaming():
            time.sleep(1)

            # Get some samples
            samples = usrp.get_samples()
            if samples is not None:
                print(f"Received {len(samples)} samples")
                print(f"Power: {np.mean(np.abs(samples)**2):.6f}")

            # Get statistics
            stats = usrp.get_streaming_stats()
            print(f"Stats: {stats}")

            usrp.stop_streaming()

        usrp.disconnect()

    print("✅ USRP Interface test completed")


if __name__ == "__main__":
    test_usrp_interface()
