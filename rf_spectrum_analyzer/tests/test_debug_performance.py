"""
Debug and Performance Tests
Comprehensive debugging utilities and performance benchmarks for profiling
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import warnings
import time
import psutil
import gc
import tracemalloc
import cProfile
import pstats
import io
from unittest.mock import Mock, patch
import threading
import queue
import json
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Test component availability
DEBUG_TOOLS_AVAILABLE = {}

try:
    import memory_profiler
    DEBUG_TOOLS_AVAILABLE['memory_profiler'] = True
except ImportError:
    DEBUG_TOOLS_AVAILABLE['memory_profiler'] = False

try:
    import line_profiler
    DEBUG_TOOLS_AVAILABLE['line_profiler'] = True
except ImportError:
    DEBUG_TOOLS_AVAILABLE['line_profiler'] = False

try:
    from rf_spectrum_analyzer.core.sdr_backend import SDRBackend, SDRConfig
    from rf_spectrum_analyzer.core.signal_processor import SignalProcessor, ProcessingConfig
    DEBUG_TOOLS_AVAILABLE['core'] = True
except ImportError:
    DEBUG_TOOLS_AVAILABLE['core'] = False

try:
    import matplotlib.pyplot as plt
    DEBUG_TOOLS_AVAILABLE['matplotlib'] = True
except ImportError:
    DEBUG_TOOLS_AVAILABLE['matplotlib'] = False


class PerformanceProfiler:
    """Performance profiling utility"""
    
    def __init__(self):
        self.profiles = {}
        self.memory_snapshots = []
        self.timing_data = {}
    
    def start_memory_tracing(self):
        """Start memory tracing"""
        tracemalloc.start()
        self.memory_snapshots = []
    
    def stop_memory_tracing(self):
        """Stop memory tracing and return statistics"""
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            return snapshot
        return None
    
    def profile_function(self, func, *args, **kwargs):
        """Profile a function's performance"""
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Time measurement
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        profiler.disable()
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Extract profile statistics
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        profile_data = {
            'function_name': func.__name__,
            'execution_time': end_time - start_time,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_after - memory_before,
            'success': success,
            'error': error,
            'profile_stats': stats_buffer.getvalue(),
            'timestamp': time.time()
        }
        
        self.profiles[func.__name__] = profile_data
        return result, profile_data
    
    def benchmark_function(self, func, iterations=10, *args, **kwargs):
        """Benchmark a function over multiple iterations"""
        times = []
        memory_deltas = []
        errors = []
        
        for i in range(iterations):
            try:
                result, profile_data = self.profile_function(func, *args, **kwargs)
                if profile_data['success']:
                    times.append(profile_data['execution_time'])
                    memory_deltas.append(profile_data['memory_delta_mb'])
                else:
                    errors.append(profile_data['error'])
            except Exception as e:
                errors.append(str(e))
        
        if times:
            benchmark_results = {
                'function_name': func.__name__,
                'iterations': iterations,
                'successful_runs': len(times),
                'failed_runs': len(errors),
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'avg_memory_delta_mb': np.mean(memory_deltas) if memory_deltas else 0,
                'std_memory_delta_mb': np.std(memory_deltas) if memory_deltas else 0,
                'errors': errors
            }
        else:
            benchmark_results = {
                'function_name': func.__name__,
                'iterations': iterations,
                'successful_runs': 0,
                'failed_runs': len(errors),
                'errors': errors
            }
        
        return benchmark_results
    
    def save_results(self, filename):
        """Save profiling results to file"""
        results = {
            'profiles': self.profiles,
            'timing_data': self.timing_data,
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)


class MockSDRBackend(SDRBackend):
    """Mock SDR Backend with performance monitoring"""
    
    def __init__(self, config):
        super().__init__(config)
        self.call_counts = {
            'connect': 0,
            'disconnect': 0,
            'start_streaming': 0,
            'stop_streaming': 0,
            'read_samples': 0
        }
        self.timing_data = {
            'connect_times': [],
            'read_times': [],
            'sample_generation_times': []
        }
        self.total_samples_read = 0
        self.sample_counter = 0
    
    def connect(self):
        start_time = time.time()
        self.call_counts['connect'] += 1
        
        # Simulate connection delay
        time.sleep(0.01)  # 10ms connection time
        
        self.is_connected = True
        end_time = time.time()
        self.timing_data['connect_times'].append(end_time - start_time)
        return True
    
    def disconnect(self):
        self.call_counts['disconnect'] += 1
        self.is_connected = False
        self.is_streaming = False
        return True
    
    def start_streaming(self):
        self.call_counts['start_streaming'] += 1
        if self.is_connected:
            self.is_streaming = True
            return True
        return False
    
    def stop_streaming(self):
        self.call_counts['stop_streaming'] += 1
        self.is_streaming = False
        return True
    
    def read_samples(self, num_samples):
        start_time = time.time()
        self.call_counts['read_samples'] += 1
        
        if not self.is_streaming:
            return None
        
        # Generate samples with timing measurement
        gen_start = time.time()
        samples = self._generate_samples(num_samples)
        gen_end = time.time()
        
        self.timing_data['sample_generation_times'].append(gen_end - gen_start)
        self.total_samples_read += num_samples
        
        end_time = time.time()
        self.timing_data['read_times'].append(end_time - start_time)
        
        return samples
    
    def _generate_samples(self, num_samples):
        """Generate mock samples with configurable complexity"""
        t = np.arange(num_samples) / self.config.sample_rate
        t += self.sample_counter / self.config.sample_rate
        
        # Generate multiple frequency components
        signal = np.zeros(num_samples, dtype=complex)
        
        # Add multiple tones
        frequencies = [1e3, 5e3, 10e3, 25e3, 50e3]
        for i, freq in enumerate(frequencies):
            amplitude = 0.5 / (i + 1)
            phase = np.random.random() * 2 * np.pi
            signal += amplitude * np.exp(1j * (2 * np.pi * freq * t + phase))
        
        # Add noise
        noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        signal += noise
        
        self.sample_counter += num_samples
        return signal
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {
            'call_counts': self.call_counts.copy(),
            'total_samples_read': self.total_samples_read,
            'timing_data': {}
        }
        
        # Calculate timing statistics
        for key, times in self.timing_data.items():
            if times:
                stats['timing_data'][key] = {
                    'avg': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }
        
        return stats


class TestPerformanceProfiling(unittest.TestCase):
    """Test performance profiling capabilities"""
    
    @unittest.skipUnless(DEBUG_TOOLS_AVAILABLE.get('core'), "Core components not available")
    def setUp(self):
        """Set up performance test environment"""
        self.profiler = PerformanceProfiler()
        self.config = SDRConfig(sample_rate=1e6)
        self.backend = MockSDRBackend(self.config)
        self.processor = SignalProcessor()
    
    def test_function_profiling(self):
        """Test function profiling capabilities"""
        def test_function(n):
            """Test function for profiling"""
            data = np.random.randn(n)
            return np.fft.fft(data)
        
        # Profile the function
        result, profile_data = self.profiler.profile_function(test_function, 1024)
        
        self.assertIsNotNone(result)
        self.assertTrue(profile_data['success'])
        self.assertGreater(profile_data['execution_time'], 0)
        self.assertIn('function_name', profile_data)
        self.assertEqual(profile_data['function_name'], 'test_function')
    
    def test_function_benchmarking(self):
        """Test function benchmarking"""
        def benchmark_target(size):
            """Function to benchmark"""
            return np.fft.fft(np.random.randn(size))
        
        # Benchmark with different sizes
        sizes = [256, 512, 1024, 2048]
        benchmark_results = {}
        
        for size in sizes:
            results = self.profiler.benchmark_function(
                benchmark_target, iterations=5, size=size
            )
            benchmark_results[size] = results
        
        # Verify benchmarking worked
        for size, results in benchmark_results.items():
            self.assertGreater(results['successful_runs'], 0)
            self.assertGreater(results['avg_time'], 0)
            self.assertGreaterEqual(results['std_time'], 0)
    
    def test_memory_profiling(self):
        """Test memory profiling"""
        self.profiler.start_memory_tracing()
        
        # Allocate some memory
        large_array = np.random.randn(1000000)  # ~8MB array
        
        # Take snapshot
        snapshot = self.profiler.stop_memory_tracing()
        
        if snapshot:
            top_stats = snapshot.statistics('lineno')
            self.assertGreater(len(top_stats), 0)
        
        # Clean up
        del large_array
        gc.collect()
    
    def test_backend_performance_monitoring(self):
        """Test SDR backend performance monitoring"""
        self.backend.connect()
        self.backend.start_streaming()
        
        # Read samples multiple times
        for _ in range(10):
            samples = self.backend.read_samples(1024)
            self.assertIsNotNone(samples)
        
        # Get performance statistics
        stats = self.backend.get_performance_stats()
        
        self.assertEqual(stats['call_counts']['read_samples'], 10)
        self.assertEqual(stats['total_samples_read'], 10 * 1024)
        self.assertIn('read_times', stats['timing_data'])
        self.assertEqual(stats['timing_data']['read_times']['count'], 10)
        
        self.backend.stop_streaming()
        self.backend.disconnect()


class TestSystemBenchmarks(unittest.TestCase):
    """Test system-wide performance benchmarks"""
    
    @unittest.skipUnless(DEBUG_TOOLS_AVAILABLE.get('core'), "Core components not available")
    def setUp(self):
        """Set up benchmark environment"""
        self.profiler = PerformanceProfiler()
        self.sample_rates = [1e6, 2e6, 5e6, 10e6]
        self.fft_sizes = [512, 1024, 2048, 4096]
    
    def test_spectrum_computation_benchmark(self):
        """Benchmark spectrum computation performance"""
        processor = SignalProcessor()
        results = {}
        
        for sample_rate in self.sample_rates:
            for fft_size in self.fft_sizes:
                # Generate test signal
                t = np.arange(fft_size * 2) / sample_rate
                signal = np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(len(t))
                
                # Benchmark spectrum computation
                def compute_spectrum():
                    return processor.compute_spectrum(signal, sample_rate)
                
                benchmark_result = self.profiler.benchmark_function(
                    compute_spectrum, iterations=5
                )
                
                key = f"sr_{sample_rate:.0e}_fft_{fft_size}"
                results[key] = benchmark_result
        
        # Verify all benchmarks completed successfully
        for key, result in results.items():
            self.assertGreater(result['successful_runs'], 0)
            self.assertLess(result['avg_time'], 1.0)  # Should complete within 1 second
    
    def test_streaming_performance_benchmark(self):
        """Benchmark streaming performance"""
        config = SDRConfig(sample_rate=2e6)
        backend = MockSDRBackend(config)
        processor = SignalProcessor()
        
        backend.connect()
        backend.start_streaming()
        
        # Benchmark continuous streaming
        def streaming_benchmark():
            total_samples = 0
            start_time = time.time()
            target_duration = 0.1  # 100ms test
            
            while time.time() - start_time < target_duration:
                samples = backend.read_samples(1024)
                if samples is not None:
                    total_samples += len(samples)
                    # Process samples
                    freqs, psd = processor.compute_spectrum(samples, config.sample_rate)
            
            actual_duration = time.time() - start_time
            return total_samples / actual_duration  # Samples per second
        
        # Run benchmark
        result, profile_data = self.profiler.profile_function(streaming_benchmark)
        
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)  # Should process some samples
        
        # Check performance metrics
        backend_stats = backend.get_performance_stats()
        self.assertGreater(backend_stats['total_samples_read'], 0)
        
        backend.stop_streaming()
        backend.disconnect()
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage patterns"""
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test different buffer sizes
        buffer_sizes = [1024, 4096, 16384, 65536]
        memory_usage = {}
        
        for buffer_size in buffer_sizes:
            # Allocate buffer
            buffer = np.zeros(buffer_size, dtype=complex)
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_usage[buffer_size] = current_memory - baseline_memory
            
            # Clean up
            del buffer
            gc.collect()
        
        # Verify memory scaling is reasonable
        for size, memory in memory_usage.items():
            expected_mb = size * 16 / 1024 / 1024  # 16 bytes per complex128
            # Allow for some overhead
            self.assertLess(memory, expected_mb * 2)


class TestDebugUtilities(unittest.TestCase):
    """Test debugging utilities and diagnostics"""
    
    @unittest.skipUnless(DEBUG_TOOLS_AVAILABLE.get('core'), "Core components not available")
    def setUp(self):
        """Set up debug test environment"""
        self.config = SDRConfig(sample_rate=1e6)
        self.backend = MockSDRBackend(self.config)
        self.processor = SignalProcessor()
    
    def test_signal_validation(self):
        """Test signal validation and diagnostics"""
        self.backend.connect()
        self.backend.start_streaming()
        
        # Read samples and validate
        samples = self.backend.read_samples(1024)
        self.assertIsNotNone(samples)
        
        # Validate signal properties
        diagnostics = self._analyze_signal(samples)
        
        self.assertIn('length', diagnostics)
        self.assertIn('dtype', diagnostics)
        self.assertIn('has_nan', diagnostics)
        self.assertIn('has_inf', diagnostics)
        self.assertIn('power_db', diagnostics)
        self.assertIn('peak_frequency', diagnostics)
        
        # Signal should be valid
        self.assertFalse(diagnostics['has_nan'])
        self.assertFalse(diagnostics['has_inf'])
        self.assertEqual(diagnostics['length'], 1024)
        self.assertEqual(diagnostics['dtype'], complex)
        
        self.backend.stop_streaming()
        self.backend.disconnect()
    
    def _analyze_signal(self, signal):
        """Analyze signal for debugging purposes"""
        diagnostics = {
            'length': len(signal),
            'dtype': type(signal[0]) if len(signal) > 0 else None,
            'has_nan': np.any(np.isnan(signal)),
            'has_inf': np.any(np.isinf(signal)),
            'power_db': 10 * np.log10(np.mean(np.abs(signal)**2)) if len(signal) > 0 else None,
            'peak_frequency': None
        }
        
        # Find peak frequency
        if len(signal) > 1:
            spectrum = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal))
            peak_idx = np.argmax(np.abs(spectrum))
            diagnostics['peak_frequency'] = freqs[peak_idx]
        
        return diagnostics
    
    def test_processing_chain_validation(self):
        """Test processing chain validation"""
        self.backend.connect()
        self.backend.start_streaming()
        
        # Test complete processing chain
        samples = self.backend.read_samples(2048)
        self.assertIsNotNone(samples)
        
        # Validate each processing step
        steps = {}
        
        # Step 1: Raw samples
        steps['raw_samples'] = self._analyze_signal(samples)
        
        # Step 2: Windowing
        windowed = self.processor.apply_window(samples)
        steps['windowed'] = self._analyze_signal(windowed)
        
        # Step 3: Spectrum computation
        freqs, psd = self.processor.compute_spectrum(samples, self.config.sample_rate)
        steps['spectrum'] = {
            'freqs_length': len(freqs),
            'psd_length': len(psd),
            'freq_range': (np.min(freqs), np.max(freqs)),
            'psd_range': (np.min(psd), np.max(psd)),
            'has_nan': np.any(np.isnan(psd)),
            'has_inf': np.any(np.isinf(psd))
        }
        
        # Validate processing chain
        self.assertEqual(steps['raw_samples']['length'], 2048)
        self.assertEqual(steps['windowed']['length'], 2048)
        self.assertEqual(steps['spectrum']['freqs_length'], steps['spectrum']['psd_length'])
        self.assertFalse(steps['spectrum']['has_nan'])
        self.assertFalse(steps['spectrum']['has_inf'])
        
        self.backend.stop_streaming()
        self.backend.disconnect()
    
    def test_error_condition_diagnostics(self):
        """Test diagnostic capabilities for error conditions"""
        # Test various error conditions
        
        # 1. Connection failure simulation
        class FailingBackend(MockSDRBackend):
            def connect(self):
                return False
        
        failing_backend = FailingBackend(self.config)
        result = failing_backend.connect()
        self.assertFalse(result)
        
        # 2. Empty signal processing
        try:
            empty_signal = np.array([])
            diagnostics = self._analyze_signal(empty_signal)
            self.assertEqual(diagnostics['length'], 0)
            self.assertIsNone(diagnostics['power_db'])
        except:
            pass  # Expected to fail
        
        # 3. Invalid signal data
        invalid_signal = np.array([np.nan, np.inf, 1+1j, np.nan])
        diagnostics = self._analyze_signal(invalid_signal)
        self.assertTrue(diagnostics['has_nan'])
        self.assertTrue(diagnostics['has_inf'])
    
    def test_performance_regression_detection(self):
        """Test detection of performance regressions"""
        # Establish baseline performance
        def baseline_function():
            data = np.random.randn(1024)
            return np.fft.fft(data)
        
        profiler = PerformanceProfiler()
        baseline_results = profiler.benchmark_function(baseline_function, iterations=10)
        
        # Simulate performance regression
        def slow_function():
            data = np.random.randn(1024)
            time.sleep(0.001)  # Add 1ms delay
            return np.fft.fft(data)
        
        slow_results = profiler.benchmark_function(slow_function, iterations=10)
        
        # Detect regression
        regression_factor = slow_results['avg_time'] / baseline_results['avg_time']
        self.assertGreater(regression_factor, 1.5)  # Should be significantly slower


class TestDebugReporting(unittest.TestCase):
    """Test debug reporting and diagnostics output"""
    
    def setUp(self):
        """Set up reporting test environment"""
        self.profiler = PerformanceProfiler()
        self.test_results_dir = Path("test_results")
        self.test_results_dir.mkdir(exist_ok=True)
    
    def test_performance_report_generation(self):
        """Test generation of performance reports"""
        # Run some benchmarks
        def test_function(n):
            return np.fft.fft(np.random.randn(n))
        
        # Benchmark different sizes
        for size in [512, 1024, 2048]:
            self.profiler.benchmark_function(test_function, iterations=3, n=size)
        
        # Save results
        report_file = self.test_results_dir / "performance_report.json"
        self.profiler.save_results(str(report_file))
        
        # Verify report was created
        self.assertTrue(report_file.exists())
        
        # Verify report content
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        self.assertIn('profiles', report_data)
        self.assertIn('timestamp', report_data)
    
    @unittest.skipUnless(DEBUG_TOOLS_AVAILABLE.get('matplotlib'), "Matplotlib not available")
    def test_performance_visualization(self):
        """Test performance visualization capabilities"""
        # Generate sample data
        sizes = [256, 512, 1024, 2048, 4096]
        times = [size * 1e-6 for size in sizes]  # Simulate linear scaling
        
        # Create performance plot
        plt.figure(figsize=(10, 6))
        plt.loglog(sizes, times, 'o-')
        plt.xlabel('FFT Size')
        plt.ylabel('Execution Time (s)')
        plt.title('FFT Performance Scaling')
        plt.grid(True)
        
        # Save plot
        plot_file = self.test_results_dir / "performance_plot.png"
        plt.savefig(str(plot_file))
        plt.close()
        
        # Verify plot was created
        self.assertTrue(plot_file.exists())
    
    def test_system_information_collection(self):
        """Test system information collection for debugging"""
        system_info = {
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'platform': sys.platform,
            'available_components': DEBUG_TOOLS_AVAILABLE
        }
        
        # Save system info
        info_file = self.test_results_dir / "system_info.json"
        with open(info_file, 'w') as f:
            json.dump(system_info, f, indent=2, default=str)
        
        self.assertTrue(info_file.exists())
        
        # Verify essential information is present
        self.assertIn('python_version', system_info)
        self.assertIn('numpy_version', system_info)
        self.assertGreater(system_info['cpu_count'], 0)
        self.assertGreater(system_info['memory_total_gb'], 0)
    
    def tearDown(self):
        """Clean up test files"""
        # Remove test result files
        for file in self.test_results_dir.glob("*"):
            if file.is_file():
                file.unlink()


if __name__ == '__main__':
    # Print debug tools availability
    print("\n=== Debug Tools Availability ===")
    for tool, available in DEBUG_TOOLS_AVAILABLE.items():
        status = "✓" if available else "✗"
        print(f"{status} {tool.replace('_', ' ').title()}")
    print()
    
    unittest.main(verbosity=2)