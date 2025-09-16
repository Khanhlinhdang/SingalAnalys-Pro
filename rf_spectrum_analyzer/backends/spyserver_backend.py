"""
SpyServer Backend for RF Spectrum Analyzer
Integrates sdrconnect SpyServerClient into the RF Spectrum Analyzer backend system.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List

try:
    from sdrconnect import SpyServerClient, SDRConfig as SDRConnectConfig
    from sdrconnect.core.exceptions import SDRConnectError, ConnectionError as SDRConnectionError
    SDRCONNECT_AVAILABLE = True
except ImportError:
    SDRCONNECT_AVAILABLE = False

from rf_spectrum_analyzer.core.sdr_backend import SDRBackend, SDRDevice, SDRDeviceType, StreamingError
from rf_spectrum_analyzer.config.settings import Settings

logger = logging.getLogger(__name__)


class SpyServerBackend(SDRBackend):
    """SpyServer backend implementation using sdrconnect."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.spyserver_client = None
        self.device_type = SDRDeviceType.SPYSERVER
        self.host = getattr(settings.sdr, 'spyserver_host', '204.144.195.52')
        self.port = getattr(settings.sdr, 'spyserver_port', 5555)
        self.timeout = getattr(settings.sdr, 'spyserver_timeout', 10.0)
    
    def connect(self) -> bool:
        """Connect to SpyServer."""
        if not SDRCONNECT_AVAILABLE:
            logger.error("sdrconnect library not available. Install with: pip install sdrconnect")
            return False
        
        try:
            # Create configuration
            config = SDRConnectConfig(
                host=self.host,
                port=self.port,
                timeout=self.timeout,
                frequency=int(self.center_frequency),
                sample_rate=int(self.sample_rate),
                gain=int(self.gain) if self.gain is not None else None
            )
            
            # Create and connect client
            self.spyserver_client = SpyServerClient(config)
            self.spyserver_client.connect()
            
            # Configure initial settings
            self.spyserver_client.set_frequency(int(self.center_frequency))
            self.spyserver_client.set_sample_rate(int(self.sample_rate))
            self.spyserver_client.set_gain(int(self.gain) if self.gain is not None else None)
            
            self.connected = True
            
            # Get device info for logging
            device_info = self.spyserver_client.get_device_info()
            logger.info(f"Connected to SpyServer at {self.host}:{self.port}")
            logger.info(f"Device info: {device_info}")
            
            return True
            
        except SDRConnectError as e:
            logger.warning(f"SpyServer connection failed: {e}")
            logger.info(f"Make sure SpyServer is running at {self.host}:{self.port}")
            return False
        except ConnectionRefusedError as e:
            logger.warning(f"SpyServer connection refused at {self.host}:{self.port}")
            logger.info("SpyServer is not running or accessible. Please start SpyServer or check connection settings.")
            return False
        except OSError as e:
            if "WinError 10061" in str(e) or "connection refused" in str(e).lower():
                logger.warning(f"Cannot connect to SpyServer at {self.host}:{self.port} - server not available")
                logger.info("To use SpyServer: 1) Start SpyServer software, 2) Verify host/port settings, 3) Check firewall")
            else:
                logger.error(f"Network error connecting to SpyServer: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to SpyServer: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from SpyServer."""
        if self.spyserver_client:
            try:
                self.spyserver_client.disconnect()
                logger.info("SpyServer disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting SpyServer: {e}")
            finally:
                self.spyserver_client = None
                self.connected = False
    
    def _attempt_reconnection(self, max_retries: int = 3) -> bool:
        """Attempt to reconnect to SpyServer with retries."""
        import time
        
        for attempt in range(max_retries):
            logger.info(f"Reconnection attempt {attempt + 1}/{max_retries}")
            
            # Clean up existing connection
            if self.spyserver_client:
                try:
                    self.spyserver_client.disconnect()
                except:
                    pass
                self.spyserver_client = None
                self.connected = False
            
            # Wait before retry (exponential backoff)
            if attempt > 0:
                wait_time = min(2 ** attempt, 10)  # Max 10 seconds
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            
            # Attempt to reconnect
            if self.connect():
                logger.info("SpyServer reconnection successful")
                return True
            
        logger.error(f"Failed to reconnect to SpyServer after {max_retries} attempts")
        return False
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure SpyServer parameters."""
        if not self.connected or not self.spyserver_client:
            logger.error("SpyServer not connected")
            return False
        
        try:
            # Update frequency if provided
            if "center_freq" in config:
                frequency = int(config["center_freq"])
                self.spyserver_client.set_frequency(frequency)
                self.center_frequency = frequency
                logger.debug(f"SpyServer frequency set to {frequency} Hz")
            
            # Update sample rate if provided
            if "sample_rate" in config:
                sample_rate = int(config["sample_rate"])
                self.spyserver_client.set_sample_rate(sample_rate)
                self.sample_rate = sample_rate
                logger.debug(f"SpyServer sample rate set to {sample_rate} Hz")
            
            # Update gain if provided
            if "gain" in config:
                gain = config["gain"]
                if gain is not None:
                    gain = int(gain)
                self.spyserver_client.set_gain(gain)
                self.gain = gain
                logger.debug(f"SpyServer gain set to {gain}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure SpyServer: {e}")
            return False
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read IQ samples from SpyServer with automatic reconnection."""
        if not self.connected or not self.spyserver_client:
            logger.error("SpyServer not connected")
            return None
        
        try:
            # Start streaming if not already started
            if not self.spyserver_client.is_streaming:
                self.spyserver_client.start_streaming()
            
            # Read samples
            samples = self.spyserver_client.read_samples(num_samples)
            
            if len(samples) == 0:
                logger.warning("No samples received from SpyServer")
                return None
            
            return samples.astype(np.complex64)
            
        except (ConnectionError, OSError) as e:
            # Handle connection errors (including WinError 10053)
            error_code = getattr(e, 'winerror', getattr(e, 'errno', None))
            if error_code == 10053 or "connection" in str(e).lower():
                logger.warning(f"SpyServer connection lost: {e}. Attempting to reconnect...")
                # Mark as disconnected and attempt reconnection
                self.connected = False
                if self._attempt_reconnection():
                    # Retry reading samples after successful reconnection
                    try:
                        if not self.spyserver_client.is_streaming:
                            self.spyserver_client.start_streaming()
                        samples = self.spyserver_client.read_samples(num_samples)
                        if len(samples) > 0:
                            return samples.astype(np.complex64)
                    except Exception as retry_e:
                        logger.error(f"Failed to read samples after reconnection: {retry_e}")
                return None
            else:
                logger.error(f"Error reading samples from SpyServer: {e}")
                return None
        except Exception as e:
            logger.error(f"Error reading samples from SpyServer: {e}")
            return None
    
    def read_samples_timeout(self, duration: float) -> Optional[np.ndarray]:
        """Read IQ samples for specified duration from SpyServer with automatic reconnection."""
        if not self.connected or not self.spyserver_client:
            logger.error("SpyServer not connected")
            return None
        
        try:
            # Start streaming if not already started
            if not self.spyserver_client.is_streaming:
                self.spyserver_client.start_streaming()
            
            # Read samples with timeout
            samples = self.spyserver_client.read_samples_timeout(duration)
            
            if len(samples) == 0:
                logger.warning("No samples received from SpyServer")
                return None
            
            return samples.astype(np.complex64)
            
        except (ConnectionError, OSError) as e:
            # Handle connection errors (including WinError 10053)
            error_code = getattr(e, 'winerror', getattr(e, 'errno', None))
            if error_code == 10053 or "connection" in str(e).lower():
                logger.warning(f"SpyServer connection lost during timeout read: {e}. Attempting to reconnect...")
                # Mark as disconnected and attempt reconnection
                self.connected = False
                if self._attempt_reconnection():
                    # Retry reading samples after successful reconnection
                    try:
                        if not self.spyserver_client.is_streaming:
                            self.spyserver_client.start_streaming()
                        samples = self.spyserver_client.read_samples_timeout(duration)
                        if len(samples) > 0:
                            return samples.astype(np.complex64)
                    except Exception as retry_e:
                        logger.error(f"Failed to read samples after reconnection: {retry_e}")
                return None
            else:
                logger.error(f"Error reading samples from SpyServer: {e}")
                return None
        except Exception as e:
            logger.error(f"Error reading samples from SpyServer: {e}")
            return None
    
    def read_samples_with_metadata(self, duration: float) -> Optional[tuple]:
        """Read IQ samples with timing metadata (SpyServer specific feature) with automatic reconnection."""
        if not self.connected or not self.spyserver_client:
            logger.error("SpyServer not connected")
            return None
        
        try:
            # Start streaming if not already started
            if not self.spyserver_client.is_streaming:
                self.spyserver_client.start_streaming()
            
            # Read samples with metadata
            samples, timestamps, latencies = self.spyserver_client.read_iq_samples_with_metadata(duration)
            
            if len(samples) == 0:
                logger.warning("No samples received from SpyServer")
                return None
            
            return samples.astype(np.complex64), timestamps, latencies
            
        except (ConnectionError, OSError) as e:
            # Handle connection errors (including WinError 10053)
            error_code = getattr(e, 'winerror', getattr(e, 'errno', None))
            if error_code == 10053 or "connection" in str(e).lower():
                logger.warning(f"SpyServer connection lost during metadata read: {e}. Attempting to reconnect...")
                # Mark as disconnected and attempt reconnection
                self.connected = False
                if self._attempt_reconnection():
                    # Retry reading samples after successful reconnection
                    try:
                        if not self.spyserver_client.is_streaming:
                            self.spyserver_client.start_streaming()
                        samples, timestamps, latencies = self.spyserver_client.read_iq_samples_with_metadata(duration)
                        if len(samples) > 0:
                            return samples.astype(np.complex64), timestamps, latencies
                    except Exception as retry_e:
                        logger.error(f"Failed to read samples with metadata after reconnection: {retry_e}")
                return None
            else:
                logger.error(f"Error reading samples with metadata from SpyServer: {e}")
                return None
        except Exception as e:
            logger.error(f"Error reading samples with metadata from SpyServer: {e}")
            return None
    
    def set_frequency(self, frequency: float) -> bool:
        """Set center frequency."""
        if not self.connected or not self.spyserver_client:
            logger.error("SpyServer not connected")
            return False
        
        try:
            self.spyserver_client.set_frequency(int(frequency))
            self.center_frequency = frequency
            logger.debug(f"SpyServer frequency set to {frequency} Hz")
            return True
        except Exception as e:
            logger.error(f"Error setting SpyServer frequency: {e}")
            return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate."""
        if not self.connected or not self.spyserver_client:
            logger.error("SpyServer not connected")
            return False
        
        try:
            self.spyserver_client.set_sample_rate(int(sample_rate))
            self.sample_rate = sample_rate
            logger.debug(f"SpyServer sample rate set to {sample_rate} Hz")
            return True
        except Exception as e:
            logger.error(f"Error setting SpyServer sample rate: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """Set RF gain."""
        if not self.connected or not self.spyserver_client:
            logger.error("SpyServer not connected")
            return False
        
        try:
            gain_value = int(gain) if gain is not None else None
            self.spyserver_client.set_gain(gain_value)
            self.gain = gain
            logger.debug(f"SpyServer gain set to {gain}")
            return True
        except Exception as e:
            logger.error(f"Error setting SpyServer gain: {e}")
            return False
    
    def set_bandwidth(self, bandwidth: float) -> bool:
        """Set bandwidth - SpyServer specific implementation."""
        if not self.connected or not self.spyserver_client:
            logger.error("SpyServer not connected")
            return False
        
        try:
            # SpyServer bandwidth is often tied to sample rate
            # For most SpyServer implementations, bandwidth = sample_rate
            # Some may support separate bandwidth setting
            if hasattr(self.spyserver_client, 'set_bandwidth'):
                self.spyserver_client.set_bandwidth(int(bandwidth))
                logger.debug(f"SpyServer bandwidth set to {bandwidth}")
            else:
                # Fallback: use sample rate as effective bandwidth
                logger.info(f"SpyServer bandwidth control via sample rate: {bandwidth}")
                self.set_sample_rate(bandwidth)
            return True
        except Exception as e:
            logger.error(f"Error setting SpyServer bandwidth: {e}")
            return False
    
    def start_streaming(self) -> bool:
        """Start IQ streaming."""
        if not self.connected or not self.spyserver_client:
            logger.error("SpyServer not connected")
            return False
        
        try:
            self.spyserver_client.start_streaming()
            logger.debug("SpyServer streaming started")
            return True
        except Exception as e:
            logger.error(f"Error starting SpyServer streaming: {e}")
            return False
    
    def stop_streaming(self) -> bool:
        """Stop IQ streaming."""
        if not self.connected or not self.spyserver_client:
            logger.error("SpyServer not connected")
            return False
        
        try:
            self.spyserver_client.stop_streaming()
            logger.debug("SpyServer streaming stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping SpyServer streaming: {e}")
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get SpyServer device information."""
        if not self.connected or not self.spyserver_client:
            return {
                "name": "SpyServer",
                "type": "spyserver",
                "connected": False,
                "host": self.host,
                "port": self.port
            }
        
        try:
            device_info = self.spyserver_client.get_device_info()
            return {
                "name": "SpyServer",
                "type": "spyserver", 
                "connected": True,
                "host": self.host,
                "port": self.port,
                "device_type": device_info.get("DeviceType", "Unknown"),
                "device_serial": device_info.get("DeviceSerial", "Unknown"),
                "max_sample_rate": device_info.get("MaximumSampleRate", 0),
                "max_bandwidth": device_info.get("MaximumBandwidth", 0),
                "max_gain_index": device_info.get("MaximumGainIndex", 0),
                "min_frequency": device_info.get("MinimumFrequency", 0),
                "max_frequency": device_info.get("MaximumFrequency", 0),
                "resolution": device_info.get("Resolution", 0)
            }
        except Exception as e:
            logger.error(f"Error getting SpyServer device info: {e}")
            return {
                "name": "SpyServer",
                "type": "spyserver",
                "connected": self.connected,
                "host": self.host,
                "port": self.port,
                "error": str(e)
            }
    
    def is_connection_healthy(self) -> bool:
        """Check if SpyServer connection is healthy."""
        if not self.connected or not self.spyserver_client:
            return False
        
        try:
            # Try to get device info as a health check
            self.spyserver_client.get_device_info()
            return True
        except (ConnectionError, OSError) as e:
            error_code = getattr(e, 'winerror', getattr(e, 'errno', None))
            if error_code == 10053 or "connection" in str(e).lower():
                logger.warning(f"Connection health check failed: {e}")
                self.connected = False
                return False
            else:
                logger.warning(f"Connection health check failed: {e}")
                return False
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False

    @staticmethod
    def detect_devices() -> List[SDRDevice]:
        """Detect available SpyServer connections."""
        if not SDRCONNECT_AVAILABLE:
            logger.warning("sdrconnect not available for SpyServer detection")
            return []
        
        # SpyServer devices need to be manually configured
        # Return a placeholder device that can be configured in GUI
        devices = [
            SDRDevice(
                device_type=SDRDeviceType.SPYSERVER,
                device_id="default",
                name="SpyServer (Configure in Settings)"
            )
        ]
        
        return devices
    
    def __enter__(self):
        """Context manager entry."""
        if self.connect():
            return self
        else:
            raise ConnectionError("Failed to connect to SpyServer")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()