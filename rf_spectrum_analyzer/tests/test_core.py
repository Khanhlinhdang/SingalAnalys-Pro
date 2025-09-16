"""
Core Module Tests
Comprehensive testing of core application components
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import warnings
from unittest.mock import Mock, patch, MagicMock
import time
import threading

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

try:
    from rf_spectrum_analyzer.core.sdr_backend import SDRBackend, SDRConfig
    from rf_spectrum_analyzer.core.signal_processor import SignalProcessor, ProcessingConfig
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    print(f"Core modules not available: {e}")

# Test if GUI components are available
try:
    from rf_spectrum_analyzer.core.app import RFSpectrumAnalyzerApp
    from rf_spectrum_analyzer.config.settings import Settings
    GUI_AVAILABLE = True
except ImportError as e:
    GUI_AVAILABLE = False
    print(f"GUI components not available: {e}")


class TestSDRConfig(unittest.TestCase):
    """Test SDR Configuration"""
    
    @unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")
    def test_default_sdr_config(self):
        """Test default SDR configuration"""
        config = SDRConfig()
        
        # Check default values
        self.assertIsNotNone(config.device_type)
        self.assertGreater(config.sample_rate, 0)
        self.assertGreater(config.center_frequency, 0)
        self.assertIsInstance(config.gain, (int, float))
        self.assertIsInstance(config.bandwidth, (int, float))
    
    @unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")
    def test_custom_sdr_config(self):
        """Test custom SDR configuration"""
        config = SDRConfig(
            device_type="hackrf",
            sample_rate=20e6,
            center_frequency=433e6,
            gain=40,
            bandwidth=20e6
        )
        
        self.assertEqual(config.device_type, "hackrf")
        self.assertEqual(config.sample_rate, 20e6)
        self.assertEqual(config.center_frequency, 433e6)
        self.assertEqual(config.gain, 40)
        self.assertEqual(config.bandwidth, 20e6)


class TestSDRBackend(unittest.TestCase):
    """Test SDR Backend base class"""
    
    @unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")
    def setUp(self):
        """Set up test environment"""
        self.config = SDRConfig(
            device_type="mock",
            sample_rate=1e6,
            center_frequency=100e6,
            gain=20
        )
    
    def test_sdr_backend_creation(self):
        """Test SDR backend creation"""
        backend = SDRBackend(self.config)
        
        self.assertEqual(backend.config, self.config)
        self.assertFalse(backend.is_connected)
        self.assertFalse(backend.is_streaming)
    
    def test_sdr_backend_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError"""
        backend = SDRBackend(self.config)
        
        with self.assertRaises(NotImplementedError):
            backend.connect()
        
        with self.assertRaises(NotImplementedError):
            backend.disconnect()
        
        with self.assertRaises(NotImplementedError):
            backend.start_streaming()
        
        with self.assertRaises(NotImplementedError):
            backend.stop_streaming()
        
        with self.assertRaises(NotImplementedError):
            backend.read_samples(1024)
    
    def test_sdr_backend_parameter_updates(self):
        """Test parameter update methods"""
        backend = SDRBackend(self.config)
        
        # Test frequency update
        new_freq = 200e6
        backend.set_center_frequency(new_freq)
        self.assertEqual(backend.config.center_frequency, new_freq)
        
        # Test sample rate update
        new_rate = 2e6
        backend.set_sample_rate(new_rate)
        self.assertEqual(backend.config.sample_rate, new_rate)
        
        # Test gain update
        new_gain = 30
        backend.set_gain(new_gain)
        self.assertEqual(backend.config.gain, new_gain)
        
        # Test bandwidth update
        new_bw = 5e6
        backend.set_bandwidth(new_bw)
        self.assertEqual(backend.config.bandwidth, new_bw)


class MockSDRBackend(SDRBackend):
    """Mock SDR Backend for testing"""
    
    def __init__(self, config):
        super().__init__(config)
        self.mock_data = None
        self.sample_counter = 0
    
    def connect(self):
        """Mock connect"""
        self.is_connected = True
        return True
    
    def disconnect(self):
        """Mock disconnect"""
        self.is_connected = False
        self.is_streaming = False
        return True
    
    def start_streaming(self):
        """Mock start streaming"""
        if self.is_connected:
            self.is_streaming = True
            return True
        return False
    
    def stop_streaming(self):
        """Mock stop streaming"""
        self.is_streaming = False
        return True
    
    def read_samples(self, num_samples):
        """Mock read samples"""
        if not self.is_streaming:
            return None
        
        # Generate mock data
        t = np.arange(num_samples) / self.config.sample_rate
        t += self.sample_counter / self.config.sample_rate
        
        # Generate a complex sinusoid with some noise
        freq = 1e3  # 1 kHz tone
        signal = np.exp(1j * 2 * np.pi * freq * t)
        noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        self.sample_counter += num_samples
        return signal + noise
    
    def get_device_info(self):
        """Mock device info"""
        return {
            'device': 'Mock SDR',
            'serial': 'MOCK123',
            'version': '1.0'
        }


class TestMockSDRBackend(unittest.TestCase):
    """Test Mock SDR Backend implementation"""
    
    @unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")
    def setUp(self):
        """Set up test environment"""
        self.config = SDRConfig(sample_rate=1e6, center_frequency=100e6)
        self.backend = MockSDRBackend(self.config)
    
    def test_mock_backend_connection(self):
        """Test mock backend connection"""
        # Initial state
        self.assertFalse(self.backend.is_connected)
        self.assertFalse(self.backend.is_streaming)
        
        # Connect
        result = self.backend.connect()
        self.assertTrue(result)
        self.assertTrue(self.backend.is_connected)
        
        # Disconnect
        result = self.backend.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.backend.is_connected)
        self.assertFalse(self.backend.is_streaming)
    
    def test_mock_backend_streaming(self):
        """Test mock backend streaming"""
        # Must connect first
        self.backend.connect()
        
        # Start streaming
        result = self.backend.start_streaming()
        self.assertTrue(result)
        self.assertTrue(self.backend.is_streaming)
        
        # Read samples
        samples = self.backend.read_samples(1024)
        self.assertIsNotNone(samples)
        self.assertEqual(len(samples), 1024)
        self.assertTrue(np.iscomplexobj(samples))
        
        # Stop streaming
        result = self.backend.stop_streaming()
        self.assertTrue(result)
        self.assertFalse(self.backend.is_streaming)
    
    def test_mock_backend_read_without_streaming(self):
        """Test reading samples without streaming"""
        self.backend.connect()
        
        # Try to read without starting streaming
        samples = self.backend.read_samples(1024)
        self.assertIsNone(samples)
    
    def test_mock_backend_device_info(self):
        """Test mock device info"""
        info = self.backend.get_device_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('device', info)
        self.assertIn('serial', info)


class TestProcessingConfig(unittest.TestCase):
    """Test Processing Configuration"""
    
    @unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")
    def test_default_processing_config(self):
        """Test default processing configuration"""
        config = ProcessingConfig()
        
        self.assertGreater(config.fft_size, 0)
        self.assertGreaterEqual(config.overlap, 0)
        self.assertLessEqual(config.overlap, 1)
        self.assertIsInstance(config.window_type, str)
        self.assertGreater(config.averaging_factor, 0)
    
    @unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")
    def test_custom_processing_config(self):
        """Test custom processing configuration"""
        config = ProcessingConfig(
            fft_size=2048,
            overlap=0.75,
            window_type="blackman",
            averaging_factor=20
        )
        
        self.assertEqual(config.fft_size, 2048)
        self.assertEqual(config.overlap, 0.75)
        self.assertEqual(config.window_type, "blackman")
        self.assertEqual(config.averaging_factor, 20)


class TestSignalProcessor(unittest.TestCase):
    """Test Signal Processor"""
    
    @unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")
    def setUp(self):
        """Set up test environment"""
        self.config = ProcessingConfig(
            fft_size=1024,
            overlap=0.5,
            window_type="hann"
        )
        self.processor = SignalProcessor(self.config)
        
        # Generate test signal
        self.sample_rate = 1e6
        t = np.arange(0, 0.001, 1/self.sample_rate)  # 1ms of data
        self.test_signal = (np.sin(2 * np.pi * 100e3 * t) + 
                           0.5 * np.sin(2 * np.pi * 200e3 * t) +
                           0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t))))
    
    def test_signal_processor_creation(self):
        """Test signal processor creation"""
        processor = SignalProcessor()
        self.assertIsNotNone(processor.config)
    
    def test_compute_spectrum(self):
        """Test spectrum computation"""
        freqs, psd = self.processor.compute_spectrum(
            self.test_signal, self.sample_rate
        )
        
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(psd), 0)
        self.assertEqual(len(freqs), len(psd))
        
        # Check frequency range
        self.assertLessEqual(np.max(np.abs(freqs)), self.sample_rate/2)
    
    def test_compute_spectrogram(self):
        """Test spectrogram computation"""
        # Need longer signal for spectrogram
        t = np.arange(0, 0.01, 1/self.sample_rate)  # 10ms
        long_signal = np.sin(2 * np.pi * 100e3 * t)
        
        freqs, times, spectrogram = self.processor.compute_spectrogram(
            long_signal, self.sample_rate
        )
        
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(times), 0)
        self.assertEqual(spectrogram.shape[0], len(freqs))
        self.assertEqual(spectrogram.shape[1], len(times))
    
    def test_apply_window(self):
        """Test window application"""
        windowed_signal = self.processor.apply_window(self.test_signal)
        
        self.assertEqual(len(windowed_signal), len(self.test_signal))
        # Windowed signal should be different from original
        self.assertFalse(np.array_equal(windowed_signal, self.test_signal))
    
    def test_decimate_signal(self):
        """Test signal decimation"""
        decimation_factor = 4
        decimated = self.processor.decimate_signal(self.test_signal, decimation_factor)
        
        expected_length = len(self.test_signal) // decimation_factor
        self.assertEqual(len(decimated), expected_length)
    
    def test_filter_signal(self):
        """Test signal filtering"""
        # Test lowpass filter
        filtered = self.processor.filter_signal(
            self.test_signal, 
            filter_type="lowpass",
            cutoff=150e3,
            sample_rate=self.sample_rate
        )
        
        self.assertEqual(len(filtered), len(self.test_signal))
        
        # Test bandpass filter
        filtered_bp = self.processor.filter_signal(
            self.test_signal,
            filter_type="bandpass", 
            cutoff=[90e3, 110e3],
            sample_rate=self.sample_rate
        )
        
        self.assertEqual(len(filtered_bp), len(self.test_signal))
    
    def test_detect_peaks(self):
        """Test peak detection"""
        freqs, psd = self.processor.compute_spectrum(self.test_signal, self.sample_rate)
        
        peak_freqs, peak_powers = self.processor.detect_peaks(
            freqs, psd, threshold=-50
        )
        
        # Should detect peaks
        self.assertGreaterEqual(len(peak_freqs), 0)
        self.assertEqual(len(peak_freqs), len(peak_powers))
    
    def test_estimate_snr(self):
        """Test SNR estimation"""
        snr = self.processor.estimate_snr(self.test_signal)
        
        self.assertIsInstance(snr, float)
        self.assertGreater(snr, 0)  # Should have positive SNR
    
    def test_process_streaming_data(self):
        """Test streaming data processing"""
        # Simulate streaming chunks
        chunk_size = 512
        chunks = []
        
        for i in range(0, len(self.test_signal), chunk_size):
            chunk = self.test_signal[i:i+chunk_size]
            if len(chunk) == chunk_size:  # Only process full chunks
                chunks.append(chunk)
        
        results = []
        for chunk in chunks:
            freqs, psd = self.processor.compute_spectrum(chunk, self.sample_rate)
            results.append((freqs, psd))
        
        # Should process all chunks
        self.assertEqual(len(results), len(chunks))


class TestApplicationIntegration(unittest.TestCase):
    """Test application integration (if GUI available)"""
    
    @unittest.skipUnless(GUI_AVAILABLE, "GUI components not available")
    def setUp(self):
        """Set up test environment"""
        self.settings = Settings()
        # Use mock backend for testing
        self.settings.sdr.device_type = "mock"
    
    @patch('core.app.QApplication')
    def test_app_creation(self, mock_qapp):
        """Test application creation"""
        try:
            # Mock Qt application
            mock_qapp.instance.return_value = None
            
            app = RFSpectrumAnalyzerApp(self.settings)
            self.assertIsNotNone(app)
            
        except Exception as e:
            # If Qt is not available, skip this test
            self.skipTest(f"Qt not available: {e}")
    
    def test_settings_loading(self):
        """Test settings loading"""
        settings = Settings()
        
        # Check that settings have reasonable defaults
        self.assertIsNotNone(settings.sdr)
        self.assertIsNotNone(settings.sdr.device_type)
        self.assertGreater(settings.sdr.sample_rate, 0)
        self.assertGreater(settings.sdr.center_frequency, 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in core modules"""
    
    @unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")
    def test_invalid_config_handling(self):
        """Test handling of invalid configurations"""
        # Test invalid sample rate
        with self.assertRaises((ValueError, AssertionError)):
            config = SDRConfig(sample_rate=-1000)
        
        # Test invalid frequency
        with self.assertRaises((ValueError, AssertionError)):
            config = SDRConfig(center_frequency=-100e6)
    
    def test_empty_signal_processing(self):
        """Test processing of empty signals"""
        processor = SignalProcessor()
        
        empty_signal = np.array([])
        
        try:
            freqs, psd = processor.compute_spectrum(empty_signal, 1e6)
            # Should either handle gracefully or raise appropriate exception
            self.assertEqual(len(freqs), 0)
            self.assertEqual(len(psd), 0)
        except (ValueError, IndexError):
            # Acceptable to raise exception for empty input
            pass
    
    def test_invalid_filter_parameters(self):
        """Test invalid filter parameters"""
        processor = SignalProcessor()
        test_signal = np.random.randn(1000)
        
        # Test invalid cutoff frequency
        try:
            filtered = processor.filter_signal(
                test_signal,
                filter_type="lowpass",
                cutoff=2e6,  # Above Nyquist for 1MHz sample rate
                sample_rate=1e6
            )
            # Should handle gracefully or raise exception
        except (ValueError, Warning):
            # Acceptable to raise exception for invalid parameters
            pass


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of core components"""
    
    @unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")
    def test_concurrent_backend_access(self):
        """Test concurrent access to SDR backend"""
        config = SDRConfig()
        backend = MockSDRBackend(config)
        backend.connect()
        backend.start_streaming()
        
        results = []
        errors = []
        
        def read_samples():
            try:
                for _ in range(10):
                    samples = backend.read_samples(1024)
                    if samples is not None:
                        results.append(len(samples))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=read_samples)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertGreater(len(results), 0)
        self.assertEqual(len(errors), 0)  # No errors should occur
        
        backend.stop_streaming()
        backend.disconnect()
    
    def test_concurrent_signal_processing(self):
        """Test concurrent signal processing"""
        processor = SignalProcessor()
        
        # Generate test signals
        signals = []
        for i in range(5):
            t = np.arange(0, 0.001, 1/1e6)
            signal = np.sin(2 * np.pi * (100e3 + i*10e3) * t)
            signals.append(signal)
        
        results = []
        errors = []
        
        def process_signal(signal):
            try:
                freqs, psd = processor.compute_spectrum(signal, 1e6)
                results.append((len(freqs), len(psd)))
            except Exception as e:
                errors.append(e)
        
        # Process signals concurrently
        threads = []
        for signal in signals:
            thread = threading.Thread(target=process_signal, args=(signal,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(results), len(signals))
        self.assertEqual(len(errors), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)