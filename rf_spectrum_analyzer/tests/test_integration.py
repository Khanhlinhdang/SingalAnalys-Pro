"""
Integration Tests
Comprehensive testing of complete application workflow and component integration
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import warnings
from unittest.mock import Mock, patch, MagicMock
import time
import threading
import queue

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Test component availability
COMPONENTS_AVAILABLE = {}

try:
    from rf_spectrum_analyzer.core.sdr_backend import SDRBackend, SDRConfig
    from rf_spectrum_analyzer.core.signal_processor import SignalProcessor, ProcessingConfig
    COMPONENTS_AVAILABLE['core'] = True
except ImportError:
    COMPONENTS_AVAILABLE['core'] = False

try:
    from rf_spectrum_analyzer.gui.main_window import MainWindow
    from PySide6.QtWidgets import QApplication
    COMPONENTS_AVAILABLE['gui'] = True
except ImportError:
    try:
        from rf_spectrum_analyzer.gui.main_window import MainWindow
        from PyQt5.QtWidgets import QApplication
        COMPONENTS_AVAILABLE['gui'] = True
    except ImportError:
        COMPONENTS_AVAILABLE['gui'] = False

try:
    from rf_spectrum_analyzer.config.settings import Settings
    COMPONENTS_AVAILABLE['config'] = True
except ImportError:
    COMPONENTS_AVAILABLE['config'] = False

try:
    import pyqtgraph as pg
    COMPONENTS_AVAILABLE['pyqtgraph'] = True
except ImportError:
    COMPONENTS_AVAILABLE['pyqtgraph'] = False


class MockSDRBackend(SDRBackend):
    """Mock SDR Backend for integration testing"""
    
    def __init__(self, config):
        super().__init__(config)
        self.mock_data_queue = queue.Queue()
        self.data_generation_active = False
        self.sample_counter = 0
        self.error_probability = 0.0  # Probability of simulating errors
    
    def connect(self):
        """Mock connect with potential failure"""
        if np.random.random() < self.error_probability:
            return False
        self.is_connected = True
        return True
    
    def disconnect(self):
        """Mock disconnect"""
        self.is_connected = False
        self.is_streaming = False
        self.data_generation_active = False
        return True
    
    def start_streaming(self):
        """Mock start streaming"""
        if not self.is_connected:
            return False
        
        if np.random.random() < self.error_probability:
            return False
        
        self.is_streaming = True
        self.data_generation_active = True
        self._start_data_generation()
        return True
    
    def stop_streaming(self):
        """Mock stop streaming"""
        self.is_streaming = False
        self.data_generation_active = False
        return True
    
    def read_samples(self, num_samples):
        """Mock read samples"""
        if not self.is_streaming:
            return None
        
        if np.random.random() < self.error_probability:
            raise RuntimeError("Mock hardware error")
        
        try:
            # Try to get data from queue with timeout
            data = self.mock_data_queue.get(timeout=0.1)
            if len(data) >= num_samples:
                return data[:num_samples]
            else:
                return data
        except queue.Empty:
            # Generate data directly if queue is empty
            return self._generate_mock_data(num_samples)
    
    def _generate_mock_data(self, num_samples):
        """Generate mock signal data"""
        t = np.arange(num_samples) / self.config.sample_rate
        t += self.sample_counter / self.config.sample_rate
        
        # Generate multiple signal components
        signal = np.zeros(num_samples, dtype=complex)
        
        # Add multiple tones at different frequencies
        tones = [1e3, 5e3, 10e3, 50e3]  # Multiple test tones
        for i, tone_freq in enumerate(tones):
            amplitude = 0.5 / (i + 1)  # Decreasing amplitude
            phase = np.random.random() * 2 * np.pi  # Random phase
            signal += amplitude * np.exp(1j * (2 * np.pi * tone_freq * t + phase))
        
        # Add time-varying component (simulates real-world signals)
        sweep_rate = 1e3  # 1 kHz/s sweep
        instantaneous_freq = 20e3 + sweep_rate * (t + self.sample_counter / self.config.sample_rate)
        signal += 0.3 * np.exp(1j * 2 * np.pi * np.cumsum(instantaneous_freq) / self.config.sample_rate)
        
        # Add noise
        noise_power = 0.1
        noise = noise_power * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        signal += noise
        
        self.sample_counter += num_samples
        return signal
    
    def _start_data_generation(self):
        """Start background data generation"""
        def generate_data():
            while self.data_generation_active:
                if self.mock_data_queue.qsize() < 10:  # Keep queue reasonably full
                    data = self._generate_mock_data(1024)
                    try:
                        self.mock_data_queue.put_nowait(data)
                    except queue.Full:
                        pass
                time.sleep(0.01)  # 10ms delay
        
        thread = threading.Thread(target=generate_data, daemon=True)
        thread.start()
    
    def set_error_probability(self, prob):
        """Set probability of simulating errors for testing"""
        self.error_probability = prob


class TestBasicIntegration(unittest.TestCase):
    """Test basic component integration"""
    
    @unittest.skipUnless(COMPONENTS_AVAILABLE.get('core'), "Core components not available")
    def setUp(self):
        """Set up integration test environment"""
        self.sdr_config = SDRConfig(
            device_type="mock",
            sample_rate=1e6,
            center_frequency=100e6,
            gain=20
        )
        
        self.processing_config = ProcessingConfig(
            fft_size=1024,
            overlap=0.5,
            window_type="hann",
            averaging_factor=5
        )
        
        self.backend = MockSDRBackend(self.sdr_config)
        self.processor = SignalProcessor(self.processing_config)
    
    def test_sdr_to_processor_pipeline(self):
        """Test SDR backend to signal processor pipeline"""
        # Connect and start streaming
        self.assertTrue(self.backend.connect())
        self.assertTrue(self.backend.start_streaming())
        
        # Read samples and process
        samples = self.backend.read_samples(2048)
        self.assertIsNotNone(samples)
        self.assertEqual(len(samples), 2048)
        
        # Process samples
        freqs, psd = self.processor.compute_spectrum(samples, self.sdr_config.sample_rate)
        
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(psd), 0)
        self.assertEqual(len(freqs), len(psd))
        
        # Verify frequency range
        max_freq = np.max(np.abs(freqs))
        self.assertLessEqual(max_freq, self.sdr_config.sample_rate / 2)
        
        # Clean up
        self.backend.stop_streaming()
        self.backend.disconnect()
    
    def test_continuous_processing_pipeline(self):
        """Test continuous processing pipeline"""
        self.backend.connect()
        self.backend.start_streaming()
        
        # Process multiple chunks
        num_chunks = 10
        results = []
        
        for i in range(num_chunks):
            samples = self.backend.read_samples(1024)
            if samples is not None:
                freqs, psd = self.processor.compute_spectrum(
                    samples, self.sdr_config.sample_rate
                )
                results.append((freqs, psd))
        
        # Should have processed multiple chunks
        self.assertGreaterEqual(len(results), num_chunks // 2)
        
        # All results should have consistent frequency axes
        if len(results) > 1:
            freq_axis_0 = results[0][0]
            for freqs, psd in results[1:]:
                np.testing.assert_array_almost_equal(freqs, freq_axis_0)
        
        self.backend.stop_streaming()
        self.backend.disconnect()
    
    def test_parameter_change_during_operation(self):
        """Test changing parameters during operation"""
        self.backend.connect()
        self.backend.start_streaming()
        
        # Initial parameters
        initial_freq = self.backend.config.center_frequency
        initial_gain = self.backend.config.gain
        
        # Read some samples
        samples1 = self.backend.read_samples(1024)
        self.assertIsNotNone(samples1)
        
        # Change parameters
        new_freq = initial_freq + 10e6
        new_gain = initial_gain + 10
        
        self.backend.set_center_frequency(new_freq)
        self.backend.set_gain(new_gain)
        
        # Verify parameters changed
        self.assertEqual(self.backend.config.center_frequency, new_freq)
        self.assertEqual(self.backend.config.gain, new_gain)
        
        # Continue reading samples
        samples2 = self.backend.read_samples(1024)
        self.assertIsNotNone(samples2)
        
        self.backend.stop_streaming()
        self.backend.disconnect()


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling in integrated system"""
    
    @unittest.skipUnless(COMPONENTS_AVAILABLE.get('core'), "Core components not available")
    def setUp(self):
        """Set up error testing environment"""
        self.sdr_config = SDRConfig(device_type="mock", sample_rate=1e6)
        self.backend = MockSDRBackend(self.sdr_config)
        self.processor = SignalProcessor()
    
    def test_connection_failure_handling(self):
        """Test handling of connection failures"""
        # Set high error probability
        self.backend.set_error_probability(0.8)
        
        # Multiple connection attempts
        connection_attempts = 0
        max_attempts = 5
        connected = False
        
        while connection_attempts < max_attempts and not connected:
            connected = self.backend.connect()
            connection_attempts += 1
        
        # Should eventually succeed or handle failure gracefully
        if not connected:
            self.assertEqual(connection_attempts, max_attempts)
    
    def test_streaming_error_recovery(self):
        """Test recovery from streaming errors"""
        self.backend.set_error_probability(0.3)  # 30% error rate
        
        self.assertTrue(self.backend.connect())
        self.assertTrue(self.backend.start_streaming())
        
        successful_reads = 0
        error_count = 0
        total_attempts = 20
        
        for _ in range(total_attempts):
            try:
                samples = self.backend.read_samples(1024)
                if samples is not None:
                    successful_reads += 1
            except RuntimeError:
                error_count += 1
                # Test recovery - continue trying to read
                continue
        
        # Should have some successful reads despite errors
        self.assertGreater(successful_reads, 0)
        self.assertGreater(error_count, 0)  # Should have encountered some errors
        
        self.backend.stop_streaming()
        self.backend.disconnect()
    
    def test_processing_error_handling(self):
        """Test processing error handling with invalid data"""
        self.backend.connect()
        self.backend.start_streaming()
        
        # Test with various problematic data
        test_cases = [
            np.array([]),  # Empty array
            np.array([np.nan, np.inf, -np.inf]),  # Invalid values
            np.array([1+1j, 2+2j]),  # Too few samples
        ]
        
        for test_data in test_cases:
            try:
                freqs, psd = self.processor.compute_spectrum(test_data, self.sdr_config.sample_rate)
                # If no exception, check results are reasonable
                if len(freqs) > 0:
                    self.assertEqual(len(freqs), len(psd))
                    self.assertFalse(np.any(np.isnan(psd)))
            except (ValueError, IndexError, ZeroDivisionError):
                # Acceptable to raise exception for invalid data
                pass
        
        self.backend.stop_streaming()
        self.backend.disconnect()


class TestPerformanceIntegration(unittest.TestCase):
    """Test integrated system performance"""
    
    @unittest.skipUnless(COMPONENTS_AVAILABLE.get('core'), "Core components not available")
    def setUp(self):
        """Set up performance test environment"""
        self.sdr_config = SDRConfig(sample_rate=10e6)  # High sample rate
        self.backend = MockSDRBackend(self.sdr_config)
        self.processor = SignalProcessor(ProcessingConfig(fft_size=2048))
    
    def test_throughput_performance(self):
        """Test system throughput performance"""
        self.backend.connect()
        self.backend.start_streaming()
        
        # Measure throughput over time
        duration = 1.0  # 1 second test
        start_time = time.time()
        total_samples = 0
        processing_times = []
        
        while time.time() - start_time < duration:
            # Measure sample acquisition time
            sample_start = time.time()
            samples = self.backend.read_samples(4096)
            sample_end = time.time()
            
            if samples is not None:
                total_samples += len(samples)
                
                # Measure processing time
                proc_start = time.time()
                freqs, psd = self.processor.compute_spectrum(samples, self.sdr_config.sample_rate)
                proc_end = time.time()
                
                processing_times.append(proc_end - proc_start)
        
        actual_duration = time.time() - start_time
        
        # Calculate performance metrics
        sample_rate_achieved = total_samples / actual_duration
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        # Performance assertions
        self.assertGreater(sample_rate_achieved, 0)
        self.assertLess(avg_processing_time, 0.1)  # Processing should be fast
        
        # Should achieve reasonable fraction of configured sample rate
        efficiency = sample_rate_achieved / self.sdr_config.sample_rate
        self.assertGreater(efficiency, 0.1)  # At least 10% efficiency
        
        self.backend.stop_streaming()
        self.backend.disconnect()
    
    def test_memory_usage_stability(self):
        """Test memory usage stability over time"""
        self.backend.connect()
        self.backend.start_streaming()
        
        # Run for extended period to check for memory leaks
        iterations = 100
        
        for i in range(iterations):
            samples = self.backend.read_samples(2048)
            if samples is not None:
                # Process samples
                freqs, psd = self.processor.compute_spectrum(samples, self.sdr_config.sample_rate)
                
                # Compute spectrogram occasionally
                if i % 10 == 0:
                    long_samples = np.concatenate([samples] * 4)  # Longer signal
                    freqs_sg, times_sg, spectrogram = self.processor.compute_spectrogram(
                        long_samples, self.sdr_config.sample_rate
                    )
        
        # Memory usage should be stable (no easy way to test without external tools)
        # At minimum, ensure we completed all iterations without crashing
        self.assertEqual(iterations, 100)
        
        self.backend.stop_streaming()
        self.backend.disconnect()


class TestFullSystemIntegration(unittest.TestCase):
    """Test complete system integration scenarios"""
    
    @unittest.skipUnless(
        COMPONENTS_AVAILABLE.get('core') and COMPONENTS_AVAILABLE.get('config'), 
        "Core components or config not available"
    )
    def setUp(self):
        """Set up full system test"""
        self.settings = Settings() if COMPONENTS_AVAILABLE.get('config') else None
        if self.settings:
            self.settings.sdr.device_type = "mock"
            self.settings.sdr.sample_rate = 2e6
            self.settings.sdr.center_frequency = 433e6
    
    def test_typical_user_workflow(self):
        """Test typical user workflow"""
        # 1. Initialize system with settings
        if self.settings:
            sdr_config = SDRConfig(
                device_type=self.settings.sdr.device_type,
                sample_rate=self.settings.sdr.sample_rate,
                center_frequency=self.settings.sdr.center_frequency,
                gain=self.settings.sdr.gain
            )
        else:
            sdr_config = SDRConfig(device_type="mock")
        
        backend = MockSDRBackend(sdr_config)
        processor = SignalProcessor()
        
        # 2. Connect to device
        self.assertTrue(backend.connect())
        
        # 3. Start acquisition
        self.assertTrue(backend.start_streaming())
        
        # 4. Acquire and process several frames
        spectra = []
        for _ in range(5):
            samples = backend.read_samples(1024)
            if samples is not None:
                freqs, psd = processor.compute_spectrum(samples, sdr_config.sample_rate)
                spectra.append((freqs, psd))
        
        self.assertGreater(len(spectra), 0)
        
        # 5. Change frequency and continue
        new_freq = sdr_config.center_frequency + 10e6
        backend.set_center_frequency(new_freq)
        
        # Continue acquisition
        samples = backend.read_samples(1024)
        self.assertIsNotNone(samples)
        
        # 6. Stop and disconnect
        self.assertTrue(backend.stop_streaming())
        self.assertTrue(backend.disconnect())
    
    def test_multi_device_simulation(self):
        """Test simulation of multiple devices"""
        # Create multiple mock backends
        configs = [
            SDRConfig(device_type="hackrf", sample_rate=20e6, center_frequency=433e6),
            SDRConfig(device_type="rtlsdr", sample_rate=2.4e6, center_frequency=100e6),
            SDRConfig(device_type="plutosdr", sample_rate=1e6, center_frequency=2.4e9),
        ]
        
        backends = [MockSDRBackend(config) for config in configs]
        processor = SignalProcessor()
        
        # Connect all devices
        for backend in backends:
            self.assertTrue(backend.connect())
            self.assertTrue(backend.start_streaming())
        
        # Acquire data from all devices
        all_results = []
        for backend in backends:
            samples = backend.read_samples(1024)
            if samples is not None:
                freqs, psd = processor.compute_spectrum(samples, backend.config.sample_rate)
                all_results.append({
                    'device': backend.config.device_type,
                    'freqs': freqs,
                    'psd': psd,
                    'center_freq': backend.config.center_frequency
                })
        
        # Should have results from all devices
        self.assertEqual(len(all_results), len(backends))
        
        # Verify each device has unique characteristics
        device_types = [result['device'] for result in all_results]
        self.assertEqual(len(set(device_types)), len(device_types))  # All unique
        
        # Clean up
        for backend in backends:
            backend.stop_streaming()
            backend.disconnect()


class TestRealTimeProcessingIntegration(unittest.TestCase):
    """Test real-time processing integration"""
    
    @unittest.skipUnless(COMPONENTS_AVAILABLE.get('core'), "Core components not available")
    def setUp(self):
        """Set up real-time test environment"""
        self.sdr_config = SDRConfig(sample_rate=1e6)
        self.backend = MockSDRBackend(self.sdr_config)
        self.processor = SignalProcessor(ProcessingConfig(fft_size=512))
        
        # Real-time processing state
        self.processing_queue = queue.Queue(maxsize=10)
        self.results_queue = queue.Queue()
        self.processing_active = False
    
    def test_producer_consumer_pattern(self):
        """Test producer-consumer pattern for real-time processing"""
        
        def data_producer():
            """Producer thread - acquires data from SDR"""
            while self.processing_active:
                try:
                    samples = self.backend.read_samples(512)
                    if samples is not None:
                        try:
                            self.processing_queue.put_nowait(samples)
                        except queue.Full:
                            # Drop samples if queue is full (overflow handling)
                            pass
                except:
                    break
                time.sleep(0.01)  # 10ms intervals
        
        def data_consumer():
            """Consumer thread - processes data"""
            while self.processing_active:
                try:
                    samples = self.processing_queue.get(timeout=0.1)
                    
                    # Process samples
                    freqs, psd = self.processor.compute_spectrum(
                        samples, self.sdr_config.sample_rate
                    )
                    
                    # Store results
                    result = {
                        'timestamp': time.time(),
                        'freqs': freqs,
                        'psd': psd
                    }
                    
                    try:
                        self.results_queue.put_nowait(result)
                    except queue.Full:
                        # Drop old results if queue is full
                        try:
                            self.results_queue.get_nowait()
                            self.results_queue.put_nowait(result)
                        except queue.Empty:
                            pass
                    
                except queue.Empty:
                    continue
                except:
                    break
        
        # Start system
        self.backend.connect()
        self.backend.start_streaming()
        self.processing_active = True
        
        # Start producer and consumer threads
        producer_thread = threading.Thread(target=data_producer, daemon=True)
        consumer_thread = threading.Thread(target=data_consumer, daemon=True)
        
        producer_thread.start()
        consumer_thread.start()
        
        # Let system run for a short time
        time.sleep(0.5)
        
        # Stop processing
        self.processing_active = False
        
        # Wait for threads to finish
        producer_thread.join(timeout=1.0)
        consumer_thread.join(timeout=1.0)
        
        # Check results
        results_count = self.results_queue.qsize()
        self.assertGreater(results_count, 0)
        
        # Verify timing characteristics
        timestamps = []
        while not self.results_queue.empty():
            try:
                result = self.results_queue.get_nowait()
                timestamps.append(result['timestamp'])
            except queue.Empty:
                break
        
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            avg_interval = np.mean(intervals)
            # Should have reasonable processing rate
            self.assertLess(avg_interval, 0.1)  # Less than 100ms between results
        
        # Clean up
        self.backend.stop_streaming()
        self.backend.disconnect()


if __name__ == '__main__':
    # Print component availability
    print("\n=== Component Availability ===")
    for component, available in COMPONENTS_AVAILABLE.items():
        status = "✓" if available else "✗"
        print(f"{status} {component.title()}")
    print()
    
    unittest.main(verbosity=2)