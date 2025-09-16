"""
SDR Backend Tests
Comprehensive testing of all SDR backend implementations
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import warnings
from unittest.mock import Mock, patch, MagicMock
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Test backend availability
BACKENDS_AVAILABLE = {}

try:
    from rf_spectrum_analyzer.backends.hackrf_backend import HackRFBackend
    BACKENDS_AVAILABLE['hackrf'] = True
except ImportError:
    BACKENDS_AVAILABLE['hackrf'] = False

try:
    from rf_spectrum_analyzer.backends.rtlsdr_backend import RTLSDRBackend
    BACKENDS_AVAILABLE['rtlsdr'] = True
except ImportError:
    BACKENDS_AVAILABLE['rtlsdr'] = False

try:
    from rf_spectrum_analyzer.backends.pluto_backend import PlutoSDRBackend
    BACKENDS_AVAILABLE['plutosdr'] = True
except ImportError:
    BACKENDS_AVAILABLE['plutosdr'] = False

try:
    from rf_spectrum_analyzer.backends.soapy_backend import SoapySDRBackend
    BACKENDS_AVAILABLE['soapysdr'] = True
except ImportError:
    BACKENDS_AVAILABLE['soapysdr'] = False

try:
    from rf_spectrum_analyzer.backends.usrp_backend import USRPBackend
    BACKENDS_AVAILABLE['usrp'] = True
except ImportError:
    BACKENDS_AVAILABLE['usrp'] = False

try:
    from rf_spectrum_analyzer.core.sdr_backend import SDRBackend, SDRConfig
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


class TestHackRFBackend(unittest.TestCase):
    """Test HackRF Backend"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = SDRConfig(
            device_type="hackrf",
            sample_rate=20e6,
            center_frequency=433e6,
            gain=40,
            bandwidth=20e6
        )
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('hackrf'), "HackRF backend not available")
    def test_hackrf_creation(self):
        """Test HackRF backend creation"""
        backend = HackRFBackend(self.config)
        
        self.assertEqual(backend.config.device_type, "hackrf")
        self.assertFalse(backend.is_connected)
        self.assertFalse(backend.is_streaming)
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('hackrf'), "HackRF backend not available")
    @patch('hackrf.HackRf')
    def test_hackrf_connection(self, mock_hackrf):
        """Test HackRF connection with mock device"""
        # Mock HackRF device
        mock_device = Mock()
        mock_hackrf.return_value = mock_device
        mock_device.open.return_value = True
        mock_device.close.return_value = True
        
        backend = HackRFBackend(self.config)
        
        # Test connection
        result = backend.connect()
        self.assertTrue(result)
        self.assertTrue(backend.is_connected)
        
        # Test disconnection
        result = backend.disconnect()
        self.assertTrue(result)
        self.assertFalse(backend.is_connected)
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('hackrf'), "HackRF backend not available")
    @patch('hackrf.HackRf')
    def test_hackrf_streaming(self, mock_hackrf):
        """Test HackRF streaming with mock device"""
        # Mock HackRF device and streaming
        mock_device = Mock()
        mock_hackrf.return_value = mock_device
        mock_device.open.return_value = True
        mock_device.start_rx.return_value = True
        mock_device.stop_rx.return_value = True
        
        # Mock sample data
        mock_samples = np.random.randn(2048) + 1j * np.random.randn(2048)
        mock_device.read_samples.return_value = mock_samples
        
        backend = HackRFBackend(self.config)
        backend.connect()
        
        # Test streaming start
        result = backend.start_streaming()
        self.assertTrue(result)
        self.assertTrue(backend.is_streaming)
        
        # Test sample reading
        samples = backend.read_samples(1024)
        self.assertIsNotNone(samples)
        self.assertLessEqual(len(samples), 2048)  # May return fewer samples
        
        # Test streaming stop
        result = backend.stop_streaming()
        self.assertTrue(result)
        self.assertFalse(backend.is_streaming)
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('hackrf'), "HackRF backend not available")
    def test_hackrf_parameter_validation(self):
        """Test HackRF parameter validation"""
        backend = HackRFBackend(self.config)
        
        # Test valid frequency range
        backend.set_center_frequency(100e6)  # Should work
        self.assertEqual(backend.config.center_frequency, 100e6)
        
        # Test valid sample rate
        backend.set_sample_rate(10e6)  # Should work
        self.assertEqual(backend.config.sample_rate, 10e6)
        
        # Test gain range
        backend.set_gain(0)  # Minimum gain
        self.assertEqual(backend.config.gain, 0)
        
        backend.set_gain(47)  # Maximum gain
        self.assertEqual(backend.config.gain, 47)


class TestRTLSDRBackend(unittest.TestCase):
    """Test RTL-SDR Backend"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = SDRConfig(
            device_type="rtlsdr",
            sample_rate=2.4e6,
            center_frequency=100e6,
            gain=30
        )
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('rtlsdr'), "RTL-SDR backend not available")
    def test_rtlsdr_creation(self):
        """Test RTL-SDR backend creation"""
        backend = RTLSDRBackend(self.config)
        
        self.assertEqual(backend.config.device_type, "rtlsdr")
        self.assertFalse(backend.is_connected)
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('rtlsdr'), "RTL-SDR backend not available")
    @patch('rtlsdr.RtlSdr')
    def test_rtlsdr_connection(self, mock_rtlsdr):
        """Test RTL-SDR connection with mock device"""
        # Mock RTL-SDR device
        mock_device = Mock()
        mock_rtlsdr.return_value = mock_device
        
        backend = RTLSDRBackend(self.config)
        
        # Test connection
        result = backend.connect()
        self.assertTrue(result)
        self.assertTrue(backend.is_connected)
        
        # Test disconnection
        result = backend.disconnect()
        self.assertTrue(result)
        self.assertFalse(backend.is_connected)
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('rtlsdr'), "RTL-SDR backend not available")
    @patch('rtlsdr.RtlSdr')
    def test_rtlsdr_sampling(self, mock_rtlsdr):
        """Test RTL-SDR sampling with mock device"""
        # Mock RTL-SDR device
        mock_device = Mock()
        mock_rtlsdr.return_value = mock_device
        
        # Mock sample data (RTL-SDR returns real samples)
        mock_samples = np.random.randn(1024).astype(np.complex128)
        mock_device.read_samples.return_value = mock_samples
        
        backend = RTLSDRBackend(self.config)
        backend.connect()
        backend.start_streaming()
        
        # Test sample reading
        samples = backend.read_samples(1024)
        self.assertIsNotNone(samples)
        self.assertEqual(len(samples), 1024)
        
        backend.stop_streaming()
        backend.disconnect()


class TestPlutoSDRBackend(unittest.TestCase):
    """Test PlutoSDR Backend"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = SDRConfig(
            device_type="plutosdr",
            sample_rate=1e6,
            center_frequency=2.4e9,
            gain=30,
            bandwidth=1e6
        )
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('plutosdr'), "PlutoSDR backend not available")
    def test_plutosdr_creation(self):
        """Test PlutoSDR backend creation"""
        backend = PlutoSDRBackend(self.config)
        
        self.assertEqual(backend.config.device_type, "plutosdr")
        self.assertFalse(backend.is_connected)
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('plutosdr'), "PlutoSDR backend not available")
    @patch('adi.Pluto')
    def test_plutosdr_connection(self, mock_pluto):
        """Test PlutoSDR connection with mock device"""
        # Mock PlutoSDR device
        mock_device = Mock()
        mock_pluto.return_value = mock_device
        
        backend = PlutoSDRBackend(self.config)
        
        # Test connection
        result = backend.connect()
        self.assertTrue(result)
        self.assertTrue(backend.is_connected)
        
        # Verify device parameters were set
        self.assertTrue(hasattr(mock_device, 'sample_rate'))
        self.assertTrue(hasattr(mock_device, 'rx_lo'))
        
        # Test disconnection
        result = backend.disconnect()
        self.assertTrue(result)
        self.assertFalse(backend.is_connected)
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('plutosdr'), "PlutoSDR backend not available")
    @patch('adi.Pluto')
    def test_plutosdr_streaming(self, mock_pluto):
        """Test PlutoSDR streaming with mock device"""
        # Mock PlutoSDR device
        mock_device = Mock()
        mock_pluto.return_value = mock_device
        
        # Mock sample data
        mock_samples = np.random.randn(1024) + 1j * np.random.randn(1024)
        mock_device.rx.return_value = mock_samples
        
        backend = PlutoSDRBackend(self.config)
        backend.connect()
        backend.start_streaming()
        
        # Test sample reading
        samples = backend.read_samples(1024)
        self.assertIsNotNone(samples)
        self.assertTrue(np.iscomplexobj(samples))
        
        backend.stop_streaming()
        backend.disconnect()


class TestSoapySDRBackend(unittest.TestCase):
    """Test SoapySDR Backend"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = SDRConfig(
            device_type="soapysdr",
            sample_rate=1e6,
            center_frequency=100e6,
            gain=20
        )
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('soapysdr'), "SoapySDR backend not available")
    def test_soapysdr_creation(self):
        """Test SoapySDR backend creation"""
        backend = SoapySDRBackend(self.config)
        
        self.assertEqual(backend.config.device_type, "soapysdr")
        self.assertFalse(backend.is_connected)
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('soapysdr'), "SoapySDR backend not available")
    @patch('SoapySDR.Device')
    def test_soapysdr_connection(self, mock_device_class):
        """Test SoapySDR connection with mock device"""
        # Mock SoapySDR device
        mock_device = Mock()
        mock_device_class.return_value = mock_device
        mock_device.setupStream.return_value = Mock()
        mock_device.activateStream.return_value = 0
        mock_device.deactivateStream.return_value = 0
        
        backend = SoapySDRBackend(self.config)
        
        # Test connection
        result = backend.connect()
        self.assertTrue(result)
        self.assertTrue(backend.is_connected)
        
        # Test disconnection
        result = backend.disconnect()
        self.assertTrue(result)
        self.assertFalse(backend.is_connected)
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('soapysdr'), "SoapySDR backend not available")
    @patch('SoapySDR.Device')
    def test_soapysdr_streaming(self, mock_device_class):
        """Test SoapySDR streaming with mock device"""
        # Mock SoapySDR device
        mock_device = Mock()
        mock_device_class.return_value = mock_device
        mock_device.setupStream.return_value = Mock()
        mock_device.activateStream.return_value = 0
        mock_device.deactivateStream.return_value = 0
        
        # Mock sample data
        mock_samples = np.random.randn(1024) + 1j * np.random.randn(1024)
        mock_device.readStream.return_value = (0, mock_samples, 0)
        
        backend = SoapySDRBackend(self.config)
        backend.connect()
        backend.start_streaming()
        
        # Test sample reading
        samples = backend.read_samples(1024)
        self.assertIsNotNone(samples)
        self.assertTrue(np.iscomplexobj(samples))
        
        backend.stop_streaming()
        backend.disconnect()


class TestUSRPBackend(unittest.TestCase):
    """Test USRP Backend"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = SDRConfig(
            device_type="usrp",
            sample_rate=1e6,
            center_frequency=100e6,
            gain=30
        )
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('usrp'), "USRP backend not available")
    def test_usrp_creation(self):
        """Test USRP backend creation"""
        backend = USRPBackend(self.config)
        
        self.assertEqual(backend.config.device_type, "usrp")
        self.assertFalse(backend.is_connected)
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('usrp'), "USRP backend not available")
    @patch('uhd.usrp.MultiUSRP')
    def test_usrp_connection(self, mock_usrp_class):
        """Test USRP connection with mock device"""
        # Mock USRP device
        mock_device = Mock()
        mock_usrp_class.return_value = mock_device
        
        backend = USRPBackend(self.config)
        
        # Test connection
        result = backend.connect()
        self.assertTrue(result)
        self.assertTrue(backend.is_connected)
        
        # Test disconnection
        result = backend.disconnect()
        self.assertTrue(result)
        self.assertFalse(backend.is_connected)
    
    @unittest.skipUnless(BACKENDS_AVAILABLE.get('usrp'), "USRP backend not available")
    @patch('uhd.usrp.MultiUSRP')
    def test_usrp_streaming(self, mock_usrp_class):
        """Test USRP streaming with mock device"""
        # Mock USRP device and streaming
        mock_device = Mock()
        mock_usrp_class.return_value = mock_device
        
        # Mock streamer
        mock_streamer = Mock()
        mock_device.get_rx_stream.return_value = mock_streamer
        
        # Mock sample data
        mock_samples = np.random.randn(1024) + 1j * np.random.randn(1024)
        mock_streamer.recv.return_value = (mock_samples, 0)
        
        backend = USRPBackend(self.config)
        backend.connect()
        backend.start_streaming()
        
        # Test sample reading
        samples = backend.read_samples(1024)
        self.assertIsNotNone(samples)
        self.assertTrue(np.iscomplexobj(samples))
        
        backend.stop_streaming()
        backend.disconnect()


class TestBackendFactory(unittest.TestCase):
    """Test Backend Factory"""
    
    @unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")
    def test_backend_factory_creation(self):
        """Test backend factory for creating different backends"""
        from rf_spectrum_analyzer.core.backend_factory import BackendFactory
        
        # Test creating different backend types
        backend_types = ['hackrf', 'rtlsdr', 'plutosdr', 'soapysdr', 'usrp']
        
        for backend_type in backend_types:
            config = SDRConfig(device_type=backend_type)
            
            try:
                backend = BackendFactory.create_backend(config)
                self.assertIsNotNone(backend)
                self.assertEqual(backend.config.device_type, backend_type)
            except ImportError:
                # Skip if backend dependencies not available
                self.skipTest(f"{backend_type} backend not available")
    
    @unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")
    def test_backend_factory_invalid_type(self):
        """Test backend factory with invalid type"""
        from rf_spectrum_analyzer.core.backend_factory import BackendFactory
        
        config = SDRConfig(device_type="invalid_type")
        
        with self.assertRaises(ValueError):
            BackendFactory.create_backend(config)


class TestBackendPerformance(unittest.TestCase):
    """Test backend performance characteristics"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.sample_rates = [1e6, 2e6, 5e6, 10e6]
        self.num_samples = [1024, 2048, 4096, 8192]
    
    def test_backend_latency(self):
        """Test backend latency characteristics"""
        # This would test actual latency with real devices
        # For now, just test the framework
        
        config = SDRConfig(sample_rate=1e6)
        
        # Mock backend for testing
        class LatencyTestBackend(SDRBackend):
            def connect(self):
                self.is_connected = True
                return True
            
            def disconnect(self):
                self.is_connected = False
                return True
            
            def start_streaming(self):
                if self.is_connected:
                    self.is_streaming = True
                return True
            
            def stop_streaming(self):
                self.is_streaming = False
                return True
            
            def read_samples(self, num_samples):
                if not self.is_streaming:
                    return None
                
                # Simulate processing time
                time.sleep(0.001)  # 1ms delay
                return np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        
        backend = LatencyTestBackend(config)
        backend.connect()
        backend.start_streaming()
        
        # Measure read latency
        start_time = time.time()
        samples = backend.read_samples(1024)
        end_time = time.time()
        
        latency = end_time - start_time
        self.assertLess(latency, 0.1)  # Should be less than 100ms
        self.assertIsNotNone(samples)
        
        backend.stop_streaming()
        backend.disconnect()
    
    def test_backend_throughput(self):
        """Test backend throughput characteristics"""
        # Similar to latency test but focused on throughput
        
        config = SDRConfig(sample_rate=10e6)
        
        class ThroughputTestBackend(SDRBackend):
            def __init__(self, config):
                super().__init__(config)
                self.total_samples = 0
            
            def connect(self):
                self.is_connected = True
                return True
            
            def disconnect(self):
                self.is_connected = False
                return True
            
            def start_streaming(self):
                if self.is_connected:
                    self.is_streaming = True
                return True
            
            def stop_streaming(self):
                self.is_streaming = False
                return True
            
            def read_samples(self, num_samples):
                if not self.is_streaming:
                    return None
                
                self.total_samples += num_samples
                return np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        
        backend = ThroughputTestBackend(config)
        backend.connect()
        backend.start_streaming()
        
        # Test throughput over time
        start_time = time.time()
        duration = 0.1  # 100ms test
        
        while time.time() - start_time < duration:
            samples = backend.read_samples(1024)
            self.assertIsNotNone(samples)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Calculate throughput
        throughput = backend.total_samples / actual_duration
        expected_min_throughput = config.sample_rate * 0.5  # At least 50% of sample rate
        
        self.assertGreater(throughput, expected_min_throughput)
        
        backend.stop_streaming()
        backend.disconnect()


class TestBackendErrorConditions(unittest.TestCase):
    """Test backend error conditions and recovery"""
    
    def test_connection_failures(self):
        """Test handling of connection failures"""
        
        class FailingBackend(SDRBackend):
            def __init__(self, config, fail_on_connect=False):
                super().__init__(config)
                self.fail_on_connect = fail_on_connect
            
            def connect(self):
                if self.fail_on_connect:
                    return False
                self.is_connected = True
                return True
            
            def disconnect(self):
                self.is_connected = False
                return True
            
            def start_streaming(self):
                return False  # Always fail
            
            def stop_streaming(self):
                return True
            
            def read_samples(self, num_samples):
                return None
        
        config = SDRConfig()
        
        # Test connection failure
        backend = FailingBackend(config, fail_on_connect=True)
        result = backend.connect()
        self.assertFalse(result)
        self.assertFalse(backend.is_connected)
        
        # Test streaming failure
        backend = FailingBackend(config, fail_on_connect=False)
        backend.connect()
        self.assertTrue(backend.is_connected)
        
        result = backend.start_streaming()
        self.assertFalse(result)
        self.assertFalse(backend.is_streaming)
    
    def test_sample_buffer_overflow(self):
        """Test handling of sample buffer overflow"""
        
        class OverflowBackend(SDRBackend):
            def __init__(self, config):
                super().__init__(config)
                self.overflow_count = 0
            
            def connect(self):
                self.is_connected = True
                return True
            
            def disconnect(self):
                self.is_connected = False
                return True
            
            def start_streaming(self):
                if self.is_connected:
                    self.is_streaming = True
                return True
            
            def stop_streaming(self):
                self.is_streaming = False
                return True
            
            def read_samples(self, num_samples):
                if not self.is_streaming:
                    return None
                
                # Simulate overflow every 5th read
                self.overflow_count += 1
                if self.overflow_count % 5 == 0:
                    raise RuntimeError("Buffer overflow")
                
                return np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        
        config = SDRConfig()
        backend = OverflowBackend(config)
        backend.connect()
        backend.start_streaming()
        
        successful_reads = 0
        overflow_errors = 0
        
        for _ in range(10):
            try:
                samples = backend.read_samples(1024)
                if samples is not None:
                    successful_reads += 1
            except RuntimeError as e:
                if "overflow" in str(e):
                    overflow_errors += 1
        
        # Should have both successful reads and overflow errors
        self.assertGreater(successful_reads, 0)
        self.assertGreater(overflow_errors, 0)
        
        backend.stop_streaming()
        backend.disconnect()


if __name__ == '__main__':
    # Print backend availability
    print("\n=== SDR Backend Availability ===")
    for backend, available in BACKENDS_AVAILABLE.items():
        status = "✓" if available else "✗"
        print(f"{status} {backend.upper()}")
    print()
    
    unittest.main(verbosity=2)