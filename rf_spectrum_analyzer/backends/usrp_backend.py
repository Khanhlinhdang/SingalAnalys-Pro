"""
USRP Backend for RF Spectrum Analyzer
Integrates Ettus USRP hardware using the usrp_interface module
"""

import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass

from rf_spectrum_analyzer.core.sdr_backend import SDRBackend, SDRDevice, StreamingError
from rf_spectrum_analyzer.utils.logger import get_logger

# Import USRP interface from the main project directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from rf_spectrum_analyzer.backends.usrp_interface import create_usrp_interface, USRP_AVAILABLE
    USRP_BACKEND_AVAILABLE = True
except ImportError as e:
    logger = get_logger('usrp_backend')
    logger.warning(f"USRP interface not available: {e}")
    USRP_BACKEND_AVAILABLE = False
    
    # Create dummy classes for when USRP is not available
    class DummyUSRPInterface:
        def get_device_list(self): return []
        def connect(self, *args, **kwargs): return False
        def disconnect(self): pass
        def is_connected(self): return False
        def get_device_info(self): return "USRP Not Available"
        def set_rx_parameters(self, *args, **kwargs): return False
        def get_rx_parameters(self): return {}
        def start_streaming(self, *args, **kwargs): return False
        def stop_streaming(self): pass
        def get_samples(self, *args, **kwargs): return None
        def get_streaming_stats(self): return {}
        def transmit_samples(self, *args, **kwargs): return False
        def set_time_now(self): return False
    
    def create_usrp_interface(use_simulator=False):
        return DummyUSRPInterface()
    
    USRP_AVAILABLE = False

logger = get_logger('usrp_backend')

@dataclass
class USRPDevice(SDRDevice):
    """USRP device information"""
    serial: str = ""
    address: str = ""
    product: str = ""
    fw_version: str = ""
    fpga_version: str = ""
    rx_channels: int = 1
    tx_channels: int = 1
    available_sample_rates: List[float] = None
    available_gains: List[float] = None
    available_antennas: List[str] = None
    motherboard: str = ""
    daughterboard: str = ""

class USRPBackend(SDRBackend):
    """USRP backend implementation using usrp_interface"""
    
    def __init__(self, use_simulator: bool = False):
        super().__init__()
        self.device_type = "usrp"
        self.use_simulator = use_simulator or not USRP_AVAILABLE
        
        # USRP interface
        self.usrp = None
        self.device_info = None
        
        # Streaming state
        self.streaming = False
        self.stream_thread = None
        self.samples_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Configuration
        self.center_frequency = 100e6
        self.sample_rate = 1e6
        self.gain = 30.0
        self.bandwidth = 0  # Auto
        self.antenna = "RX2"
        self.ppm_correction = 0
        
        # Statistics
        self.total_samples = 0
        self.overruns = 0
        self.underruns = 0
        self.start_time = None
        
        logger.info(f"USRP backend initialized (simulator: {self.use_simulator})")
    
    def get_device_list(self) -> List[SDRDevice]:
        """Get list of available USRP devices"""
        devices = []
        
        try:
            # Create temporary interface to get device list
            temp_usrp = create_usrp_interface(self.use_simulator)
            usrp_devices = temp_usrp.get_device_list()
            
            for i, device_info in enumerate(usrp_devices):
                if isinstance(device_info, dict):
                    device = USRPDevice(
                        id=device_info.get('index', i),
                        name=device_info.get('name', f"USRP Device {i}"),
                        serial=device_info.get('serial', ''),
                        address=device_info.get('address', ''),
                        product=device_info.get('type', 'USRP'),
                        available_sample_rates=[
                            250e3, 500e3, 1e6, 2e6, 4e6, 8e6, 10e6, 16e6, 20e6, 25e6
                        ],
                        available_gains=list(range(0, 61, 1)),  # 0-60 dB in 1 dB steps
                        available_antennas=["RX2", "TX/RX", "RX1"],
                        min_frequency=50e6,
                        max_frequency=6e9,
                        rx_channels=1,
                        tx_channels=1
                    )
                    devices.append(device)
                else:
                    # Handle string device info
                    device = USRPDevice(
                        id=i,
                        name=str(device_info),
                        product="USRP",
                        available_sample_rates=[1e6, 2e6, 4e6, 8e6, 10e6],
                        available_gains=list(range(0, 61, 1)),
                        available_antennas=["RX2", "TX/RX"],
                        min_frequency=50e6,
                        max_frequency=6e9
                    )
                    devices.append(device)
            
            logger.info(f"Found {len(devices)} USRP devices")
            return devices
            
        except Exception as e:
            logger.error(f"Error getting USRP device list: {e}")
            return []
    
    def connect(self, device_id: str = "", **kwargs) -> bool:
        """Connect to USRP device"""
        try:
            if self.is_connected():
                logger.warning("Already connected to USRP")
                return True
            
            # Create USRP interface
            self.usrp = create_usrp_interface(self.use_simulator)
            
            # Prepare device arguments
            device_args = ""
            if device_id:
                if device_id.isdigit():
                    device_args = f"serial={device_id}"
                else:
                    device_args = device_id
            
            # Set initial parameters from kwargs
            self.center_frequency = kwargs.get('center_frequency', self.center_frequency)
            self.sample_rate = kwargs.get('sample_rate', self.sample_rate)
            self.gain = kwargs.get('gain', self.gain)
            self.bandwidth = kwargs.get('bandwidth', self.bandwidth)
            self.antenna = kwargs.get('antenna', self.antenna)
            
            # Connect to device
            success = self.usrp.connect(
                device_args=device_args,
                sample_rate=self.sample_rate,
                center_freq=self.center_frequency
            )
            
            if success:
                # Configure initial parameters
                self.usrp.set_rx_parameters(
                    center_freq=self.center_frequency,
                    gain=self.gain,
                    bandwidth=self.bandwidth if self.bandwidth > 0 else None,
                    sample_rate=self.sample_rate,
                    antenna=self.antenna
                )
                
                # Get device info
                self.device_info = self.usrp.get_device_info()
                
                # Reset statistics
                self.total_samples = 0
                self.overruns = 0
                self.underruns = 0
                
                logger.info(f"Connected to USRP: {self.device_info}")
                return True
            else:
                logger.error("Failed to connect to USRP")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to USRP: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from USRP device"""
        try:
            if not self.is_connected():
                return True
            
            # Stop streaming if active
            if self.streaming:
                self.stop_streaming()
            
            # Disconnect device
            self.usrp.disconnect()
            self.usrp = None
            self.device_info = None
            
            logger.info("Disconnected from USRP")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from USRP: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if device is connected"""
        return self.usrp is not None and self.usrp.is_connected()
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        if not self.is_connected():
            return {}
        
        try:
            params = self.usrp.get_rx_parameters()
            stats = self.usrp.get_streaming_stats()
            
            return {
                'device_type': 'USRP',
                'device_info': self.device_info,
                'center_frequency': params.get('center_freq', self.center_frequency),
                'sample_rate': params.get('sample_rate', self.sample_rate),
                'gain': params.get('gain', self.gain),
                'bandwidth': params.get('bandwidth', self.bandwidth),
                'antenna': params.get('antenna', self.antenna),
                'streaming': self.streaming,
                'total_samples': self.total_samples,
                'overruns': stats.get('overruns', 0),
                'underruns': stats.get('underruns', 0),
                'queue_size': stats.get('queue_size', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return {'error': str(e)}
    
    def set_center_frequency(self, frequency: float) -> bool:
        """Set center frequency"""
        try:
            if not self.is_connected():
                return False
            
            success = self.usrp.set_rx_parameters(center_freq=frequency)
            if success:
                self.center_frequency = frequency
                logger.debug(f"Set center frequency to {frequency/1e6:.3f} MHz")
            return success
            
        except Exception as e:
            logger.error(f"Error setting center frequency: {e}")
            return False
    
    def get_center_frequency(self) -> float:
        """Get current center frequency"""
        if not self.is_connected():
            return self.center_frequency
        
        try:
            params = self.usrp.get_rx_parameters()
            return params.get('center_freq', self.center_frequency)
        except:
            return self.center_frequency
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate"""
        try:
            if not self.is_connected():
                return False
            
            success = self.usrp.set_rx_parameters(sample_rate=sample_rate)
            if success:
                self.sample_rate = sample_rate
                logger.debug(f"Set sample rate to {sample_rate/1e6:.3f} MS/s")
            return success
            
        except Exception as e:
            logger.error(f"Error setting sample rate: {e}")
            return False
    
    def get_sample_rate(self) -> float:
        """Get current sample rate"""
        if not self.is_connected():
            return self.sample_rate
        
        try:
            params = self.usrp.get_rx_parameters()
            return params.get('sample_rate', self.sample_rate)
        except:
            return self.sample_rate
    
    def set_gain(self, gain: float) -> bool:
        """Set gain"""
        try:
            if not self.is_connected():
                return False
            
            success = self.usrp.set_rx_parameters(gain=gain)
            if success:
                self.gain = gain
                logger.debug(f"Set gain to {gain} dB")
            return success
            
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            return False
    
    def get_gain(self) -> float:
        """Get current gain"""
        if not self.is_connected():
            return self.gain
        
        try:
            params = self.usrp.get_rx_parameters()
            return params.get('gain', self.gain)
        except:
            return self.gain
    
    def set_bandwidth(self, bandwidth: float) -> bool:
        """Set bandwidth"""
        try:
            if not self.is_connected():
                return False
            
            success = self.usrp.set_rx_parameters(
                bandwidth=bandwidth if bandwidth > 0 else None
            )
            if success:
                self.bandwidth = bandwidth
                logger.debug(f"Set bandwidth to {bandwidth/1e6:.3f} MHz")
            return success
            
        except Exception as e:
            logger.error(f"Error setting bandwidth: {e}")
            return False
    
    def get_bandwidth(self) -> float:
        """Get current bandwidth"""
        if not self.is_connected():
            return self.bandwidth
        
        try:
            params = self.usrp.get_rx_parameters()
            return params.get('bandwidth', self.bandwidth)
        except:
            return self.bandwidth
    
    def set_antenna(self, antenna: str) -> bool:
        """Set antenna"""
        try:
            if not self.is_connected():
                return False
            
            success = self.usrp.set_rx_parameters(antenna=antenna)
            if success:
                self.antenna = antenna
                logger.debug(f"Set antenna to {antenna}")
            return success
            
        except Exception as e:
            logger.error(f"Error setting antenna: {e}")
            return False
    
    def get_antenna(self) -> str:
        """Get current antenna"""
        if not self.is_connected():
            return self.antenna
        
        try:
            params = self.usrp.get_rx_parameters()
            return params.get('antenna', self.antenna)
        except:
            return self.antenna
    
    def start_streaming(self, callback: Optional[Callable] = None, 
                       buffer_size: int = 8192) -> bool:
        """Start streaming samples"""
        try:
            if not self.is_connected():
                raise StreamingError("Device not connected")
            
            if self.streaming:
                logger.warning("Already streaming")
                return True
            
            # Configure callback
            self.data_callback = callback
            self.buffer_size = buffer_size
            
            # Start USRP streaming
            success = self.usrp.start_streaming()
            if not success:
                raise StreamingError("Failed to start USRP streaming")
            
            # Start worker thread
            self.streaming = True
            self.start_time = time.time()
            self.stream_thread = threading.Thread(
                target=self._stream_worker,
                daemon=True
            )
            self.stream_thread.start()
            
            logger.info("Started USRP streaming")
            return True
            
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            self.streaming = False
            return False
    
    def stop_streaming(self) -> bool:
        """Stop streaming samples"""
        try:
            if not self.streaming:
                return True
            
            # Stop streaming
            self.streaming = False
            
            # Stop USRP streaming
            if self.usrp:
                self.usrp.stop_streaming()
            
            # Wait for thread to finish
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2.0)
            
            logger.info("Stopped USRP streaming")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            return False
    
    def _stream_worker(self):
        """Worker thread for sample streaming"""
        logger.debug("Stream worker thread started")
        
        try:
            while self.streaming:
                # Get samples from USRP
                samples = self.usrp.get_samples(timeout=1.0)
                
                if samples is not None and len(samples) > 0:
                    # Update statistics
                    self.total_samples += len(samples)
                    
                    # Store in buffer
                    with self.buffer_lock:
                        self.samples_buffer.extend(samples)
                        
                        # Limit buffer size
                        max_buffer_size = self.buffer_size * 10
                        if len(self.samples_buffer) > max_buffer_size:
                            self.samples_buffer = self.samples_buffer[-max_buffer_size:]
                    
                    # Call callback if provided
                    if self.data_callback:
                        try:
                            self.data_callback(samples)
                        except Exception as e:
                            logger.error(f"Error in data callback: {e}")
                
                time.sleep(0.001)  # Small delay to prevent busy waiting
                
        except Exception as e:
            logger.error(f"Error in stream worker: {e}")
        finally:
            logger.debug("Stream worker thread finished")
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from buffer"""
        try:
            with self.buffer_lock:
                if len(self.samples_buffer) >= num_samples:
                    samples = np.array(self.samples_buffer[:num_samples], dtype=np.complex64)
                    self.samples_buffer = self.samples_buffer[num_samples:]
                    return samples
                else:
                    # Return what we have
                    if self.samples_buffer:
                        samples = np.array(self.samples_buffer, dtype=np.complex64)
                        self.samples_buffer.clear()
                        return samples
                    return None
                    
        except Exception as e:
            logger.error(f"Error reading samples: {e}")
            return None
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        try:
            # Get stats from USRP
            usrp_stats = {}
            if self.usrp:
                usrp_stats = self.usrp.get_streaming_stats()
            
            # Calculate runtime
            runtime = 0
            if self.start_time:
                runtime = time.time() - self.start_time
            
            # Calculate sample rate
            effective_sample_rate = 0
            if runtime > 0:
                effective_sample_rate = self.total_samples / runtime
            
            return {
                'streaming': self.streaming,
                'total_samples': self.total_samples,
                'runtime': runtime,
                'effective_sample_rate': effective_sample_rate,
                'configured_sample_rate': self.sample_rate,
                'buffer_size': len(self.samples_buffer) if hasattr(self, 'samples_buffer') else 0,
                'overruns': usrp_stats.get('overruns', 0),
                'underruns': usrp_stats.get('underruns', 0),
                'usrp_queue_size': usrp_stats.get('queue_size', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting streaming stats: {e}")
            return {'error': str(e)}
    
    def transmit_samples(self, samples: np.ndarray, **kwargs) -> bool:
        """Transmit samples (if supported)"""
        try:
            if not self.is_connected():
                return False
            
            # Convert to complex64 if needed
            if samples.dtype != np.complex64:
                samples = samples.astype(np.complex64)
            
            # Transmit via USRP interface
            success = self.usrp.transmit_samples(samples)
            
            if success:
                logger.debug(f"Transmitted {len(samples)} samples")
            else:
                logger.warning("Failed to transmit samples")
            
            return success
            
        except Exception as e:
            logger.error(f"Error transmitting samples: {e}")
            return False
    
    def set_time_reference(self, source: str = "internal") -> bool:
        """Set time reference"""
        try:
            if not self.is_connected():
                return False
            
            # Set time to now (basic implementation)
            success = self.usrp.set_time_now()
            
            if success:
                logger.info(f"Set time reference to {source}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error setting time reference: {e}")
            return False
    
    def get_available_sample_rates(self) -> List[float]:
        """Get available sample rates"""
        return [
            250e3, 500e3, 1e6, 2e6, 4e6, 8e6, 10e6, 
            16e6, 20e6, 25e6, 30e6, 40e6, 50e6
        ]
    
    def get_available_gains(self) -> List[float]:
        """Get available gain values"""
        return list(range(0, 76, 1))  # 0-75 dB in 1 dB steps
    
    def get_available_antennas(self) -> List[str]:
        """Get available antennas"""
        return ["RX2", "TX/RX", "RX1"]
    
    def get_frequency_range(self) -> Tuple[float, float]:
        """Get frequency range"""
        return (50e6, 6e9)  # 50 MHz to 6 GHz (typical USRP range)

def test_usrp_backend():
    """Test USRP backend functionality"""
    logger.info("Testing USRP backend")
    
    # Test with simulator
    backend = USRPBackend(use_simulator=True)
    
    # Get device list
    devices = backend.get_device_list()
    logger.info(f"Found {len(devices)} devices")
    
    if devices:
        # Connect to first device
        device = devices[0]
        if backend.connect(device.id):
            logger.info(f"Connected to {device.name}")
            
            # Test parameter setting
            backend.set_center_frequency(915e6)
            backend.set_sample_rate(2e6)
            backend.set_gain(40)
            
            # Get device info
            info = backend.get_device_info()
            logger.info(f"Device info: {info}")
            
            # Test streaming
            if backend.start_streaming():
                time.sleep(1)
                
                # Read some samples
                samples = backend.read_samples(1000)
                if samples is not None:
                    logger.info(f"Read {len(samples)} samples")
                
                # Get stats
                stats = backend.get_streaming_stats()
                logger.info(f"Streaming stats: {stats}")
                
                backend.stop_streaming()
            
            backend.disconnect()
    
    logger.info("USRP backend test completed")

if __name__ == "__main__":
    test_usrp_backend()