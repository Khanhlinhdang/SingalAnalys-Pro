"""
DSP Analysis Module Tests
Comprehensive testing of signal analysis tools and algorithms
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

try:
    from rf_spectrum_analyzer.dsp.analysis import (
        AnalysisConfig, SpectrumAnalyzer, SignalDetector, 
        ParameterEstimator, InterferenceAnalyzer,
        analyze_spectrum, detect_signals, estimate_parameters
    )
    DSP_ANALYSIS_AVAILABLE = True
except ImportError as e:
    DSP_ANALYSIS_AVAILABLE = False
    print(f"DSP analysis not available: {e}")


class TestAnalysisConfig(unittest.TestCase):
    """Test AnalysisConfig dataclass"""
    
    @unittest.skipUnless(DSP_ANALYSIS_AVAILABLE, "DSP analysis not available")
    def test_default_config(self):
        """Test default analysis configuration"""
        config = AnalysisConfig()
        self.assertEqual(config.sample_rate, 1e6)
        self.assertEqual(config.window_size, 1024)
        self.assertEqual(config.overlap, 0.5)
        self.assertEqual(config.window_type, "hann")
        self.assertEqual(config.detection_threshold, -80.0)
    
    @unittest.skipUnless(DSP_ANALYSIS_AVAILABLE, "DSP analysis not available")
    def test_custom_config(self):
        """Test custom analysis configuration"""
        config = AnalysisConfig(
            sample_rate=2e6,
            window_size=2048,
            overlap=0.75,
            window_type="blackman",
            detection_threshold=-70.0
        )
        self.assertEqual(config.sample_rate, 2e6)
        self.assertEqual(config.window_size, 2048)
        self.assertEqual(config.overlap, 0.75)
        self.assertEqual(config.window_type, "blackman")
        self.assertEqual(config.detection_threshold, -70.0)


class TestSpectrumAnalyzer(unittest.TestCase):
    """Test Spectrum Analyzer implementation"""
    
    @unittest.skipUnless(DSP_ANALYSIS_AVAILABLE, "DSP analysis not available")
    def setUp(self):
        """Set up test environment"""
        self.config = AnalysisConfig(sample_rate=1000, window_size=256)
        self.analyzer = SpectrumAnalyzer(self.config)
        
        # Generate test signal: sum of sinusoids
        t = np.arange(0, 2, 1/1000)
        self.test_signal = (np.sin(2 * np.pi * 50 * t) + 
                           0.5 * np.sin(2 * np.pi * 150 * t) +
                           0.1 * np.random.randn(len(t)))
    
    def test_spectrum_analyzer_creation(self):
        """Test spectrum analyzer creation"""
        analyzer = SpectrumAnalyzer()
        self.assertIsNotNone(analyzer.config)
        self.assertEqual(analyzer.config.sample_rate, 1e6)
    
    def test_welch_psd(self):
        """Test Welch's method PSD calculation"""
        freqs, psd = self.analyzer.power_spectral_density(
            self.test_signal, method="welch"
        )
        
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(psd), 0)
        self.assertEqual(len(freqs), len(psd))
        
        # Check frequency range
        self.assertLessEqual(np.max(np.abs(freqs)), self.config.sample_rate/2)
    
    def test_periodogram_psd(self):
        """Test periodogram PSD calculation"""
        freqs, psd = self.analyzer.power_spectral_density(
            self.test_signal, method="periodogram"
        )
        
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(psd), 0)
        self.assertEqual(len(freqs), len(psd))
    
    def test_multitaper_psd(self):
        """Test multitaper PSD calculation"""
        freqs, psd = self.analyzer.power_spectral_density(
            self.test_signal, method="multitaper"
        )
        
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(psd), 0)
        self.assertEqual(len(freqs), len(psd))
    
    def test_spectrogram(self):
        """Test spectrogram calculation"""
        freqs, times, Sxx = self.analyzer.spectrogram(self.test_signal)
        
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(times), 0)
        self.assertEqual(Sxx.shape[0], len(freqs))
        self.assertEqual(Sxx.shape[1], len(times))
    
    def test_peak_detection(self):
        """Test peak detection in spectrum"""
        freqs, psd = self.analyzer.power_spectral_density(self.test_signal)
        peak_freqs, peak_powers = self.analyzer.peak_detection(
            freqs, psd, threshold=-60
        )
        
        # Should detect peaks at 50 Hz and 150 Hz
        self.assertGreater(len(peak_freqs), 0)
        self.assertEqual(len(peak_freqs), len(peak_powers))
        
        # Check if we found peaks near expected frequencies
        expected_freqs = [50, 150]
        for expected_freq in expected_freqs:
            min_distance = np.min(np.abs(peak_freqs - expected_freq))
            self.assertLess(min_distance, 20, 
                          f"No peak found near {expected_freq} Hz")
    
    def test_averaging(self):
        """Test spectrum averaging"""
        # Generate multiple PSD estimates
        freqs, psd1 = self.analyzer.power_spectral_density(self.test_signal)
        freqs, psd2 = self.analyzer.power_spectral_density(
            self.test_signal + 0.1 * np.random.randn(len(self.test_signal))
        )
        
        # Test averaging
        avg1 = self.analyzer.averaging(psd1)
        avg2 = self.analyzer.averaging(psd2)
        
        self.assertEqual(len(avg1), len(psd1))
        self.assertEqual(len(avg2), len(psd2))
        
        # Second average should be different from first
        self.assertFalse(np.array_equal(avg1, avg2))


class TestSignalDetector(unittest.TestCase):
    """Test Signal Detector implementation"""
    
    @unittest.skipUnless(DSP_ANALYSIS_AVAILABLE, "DSP analysis not available")
    def setUp(self):
        """Set up test environment"""
        self.config = AnalysisConfig(detection_threshold=-40)
        self.detector = SignalDetector(self.config)
        
        # Generate test signals with deterministic noise
        np.random.seed(42)  # Set seed for reproducible tests
        t = np.arange(0, 1, 1/1000)
        self.strong_signal = np.sin(2 * np.pi * 100 * t)  # Strong signal
        self.weak_signal = 0.01 * np.sin(2 * np.pi * 100 * t)  # Weak signal
        self.noise = 0.1 * np.random.randn(len(t))  # Noise only
    
    def test_signal_detector_creation(self):
        """Test signal detector creation"""
        detector = SignalDetector()
        self.assertIsNotNone(detector.config)
    
    def test_energy_detection_strong_signal(self):
        """Test energy detection with strong signal"""
        result = self.detector.energy_detection(self.strong_signal, threshold=-30)
        
        self.assertIsInstance(result, dict)
        self.assertIn('detected', result)
        self.assertIn('energy_db', result)
        self.assertIn('threshold_db', result)
        
        # Strong signal should be detected
        self.assertTrue(result['detected'])
        self.assertGreater(result['energy_db'], result['threshold_db'])
    
    def test_energy_detection_weak_signal(self):
        """Test energy detection with weak signal"""
        result = self.detector.energy_detection(self.weak_signal, threshold=-20)
        
        # Weak signal should not be detected with high threshold
        self.assertFalse(result['detected'])
        self.assertLess(result['energy_db'], result['threshold_db'])
    
    def test_energy_detection_noise_only(self):
        """Test energy detection with noise only"""
        result = self.detector.energy_detection(self.noise, threshold=-20)
        
        # Noise should not be detected
        self.assertFalse(result['detected'])
    
    def test_matched_filter_detection(self):
        """Test matched filter detection"""
        # Create template signal
        template = np.array([1, 1, -1, -1, 1, -1])
        
        # Create signal containing template
        signal_with_template = np.concatenate([
            np.random.randn(50),
            template,
            np.random.randn(50)
        ])
        
        result = self.detector.matched_filter_detection(
            signal_with_template, template
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('correlation', result)
        self.assertIn('peak_index', result)
        self.assertIn('peak_value', result)
        self.assertIn('snr_db', result)
        
        # Peak should be detected
        self.assertGreater(result['peak_value'], 0)
        self.assertGreater(result['snr_db'], 0)
    
    def test_cfar_detection(self):
        """Test CFAR detection"""
        # Create signal with target
        target_signal = np.random.randn(1000)
        target_signal[500:510] = 5  # Strong target at position 500-510
        
        result = self.detector.cfar_detection(
            target_signal, 
            guard_cells=2, 
            reference_cells=10,
            pfa=1e-4
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('detections', result)
        self.assertIn('thresholds', result)
        self.assertIn('detection_indices', result)
        
        # Should detect the target
        detection_indices = result['detection_indices']
        self.assertGreater(len(detection_indices), 0)
        
        # Detection should be near the target location
        target_detected = any(500 <= idx <= 510 for idx in detection_indices)
        self.assertTrue(target_detected)


class TestParameterEstimator(unittest.TestCase):
    """Test Parameter Estimator implementation"""
    
    @unittest.skipUnless(DSP_ANALYSIS_AVAILABLE, "DSP analysis not available")
    def setUp(self):
        """Set up test environment"""
        self.config = AnalysisConfig(sample_rate=1000)
        self.estimator = ParameterEstimator(self.config)
        
        # Generate test signal with known frequency
        self.test_frequency = 123.45
        t = np.arange(0, 2, 1/1000)
        self.test_signal = np.sin(2 * np.pi * self.test_frequency * t)
    
    def test_parameter_estimator_creation(self):
        """Test parameter estimator creation"""
        estimator = ParameterEstimator()
        self.assertIsNotNone(estimator.config)
    
    def test_fft_frequency_estimation(self):
        """Test FFT-based frequency estimation"""
        result = self.estimator.frequency_estimation(
            self.test_signal, method="fft"
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('frequency', result)
        self.assertIn('method', result)
        self.assertIn('confidence', result)
        
        # Estimated frequency should be close to actual
        estimated_freq = result['frequency']
        self.assertAlmostEqual(estimated_freq, self.test_frequency, delta=2.0)
    
    def test_argmax_frequency_estimation(self):
        """Test argmax-based frequency estimation"""
        result = self.estimator.frequency_estimation(
            self.test_signal, method="argmax"
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('frequency', result)
        self.assertIn('instantaneous_frequencies', result)
        
        # Should provide reasonable frequency estimate
        estimated_freq = result['frequency']
        self.assertAlmostEqual(estimated_freq, self.test_frequency, delta=10.0)
    
    def test_esprit_frequency_estimation(self):
        """Test ESPRIT frequency estimation"""
        result = self.estimator.frequency_estimation(
            self.test_signal, method="esprit"
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('frequencies', result)
        self.assertIn('method', result)
        
        # Should find at least one frequency
        frequencies = result['frequencies']
        self.assertGreater(len(frequencies), 0)
    
    def test_amplitude_estimation(self):
        """Test amplitude estimation"""
        result = self.estimator.amplitude_estimation(self.test_signal)
        
        self.assertIsInstance(result, dict)
        self.assertIn('rms', result)
        self.assertIn('peak', result)
        self.assertIn('crest_factor', result)
        self.assertIn('power_dbm', result)
        
        # RMS of sine wave should be peak/sqrt(2)
        expected_rms = 1.0 / np.sqrt(2)
        self.assertAlmostEqual(result['rms'], expected_rms, places=2)
        
        # Peak should be close to 1
        self.assertAlmostEqual(result['peak'], 1.0, places=1)
        
        # Crest factor should be sqrt(2) for sine wave
        self.assertAlmostEqual(result['crest_factor'], np.sqrt(2), places=1)
    
    def test_phase_estimation(self):
        """Test phase estimation"""
        # Create signal with known phase offset
        phase_offset = np.pi/4
        t = np.arange(0, 1, 1/1000)
        test_signal = np.sin(2 * np.pi * 100 * t + phase_offset)
        
        result = self.estimator.phase_estimation(test_signal)
        
        self.assertIsInstance(result, dict)
        self.assertIn('instantaneous_phase', result)
        self.assertIn('unwrapped_phase', result)
        self.assertIn('phase_offset', result)
        self.assertIn('frequency_estimate', result)
        
        # Phase offset should be close to expected
        estimated_offset = result['phase_offset']
        # Note: phase estimation might have ambiguity
        self.assertLess(abs(estimated_offset - phase_offset), np.pi/2)


class TestInterferenceAnalyzer(unittest.TestCase):
    """Test Interference Analyzer implementation"""
    
    @unittest.skipUnless(DSP_ANALYSIS_AVAILABLE, "DSP analysis not available")
    def setUp(self):
        """Set up test environment"""
        self.config = AnalysisConfig(sample_rate=1000)
        self.analyzer = InterferenceAnalyzer(self.config)
        
        # Generate baseline signal
        t = np.arange(0, 2, 1/1000)
        self.baseline_signal = 0.1 * np.random.randn(len(t))
        
        # Generate signal with interference
        interference = 0.5 * np.sin(2 * np.pi * 200 * t)
        self.interfered_signal = self.baseline_signal + interference
    
    def test_interference_analyzer_creation(self):
        """Test interference analyzer creation"""
        analyzer = InterferenceAnalyzer()
        self.assertIsNotNone(analyzer.config)
    
    def test_set_baseline(self):
        """Test setting baseline spectrum"""
        self.analyzer.set_baseline(self.baseline_signal)
        
        self.assertIsNotNone(self.analyzer.baseline_spectrum)
        freqs, psd = self.analyzer.baseline_spectrum
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(psd), 0)
    
    def test_detect_interference_without_baseline(self):
        """Test interference detection without baseline"""
        result = self.analyzer.detect_interference(self.interfered_signal)
        
        self.assertIsInstance(result, dict)
        self.assertIn('interference_detected', result)
        self.assertFalse(result['interference_detected'])
        self.assertIn('message', result)
    
    def test_detect_interference_with_baseline(self):
        """Test interference detection with baseline"""
        # Set baseline
        self.analyzer.set_baseline(self.baseline_signal)
        
        # Detect interference
        result = self.analyzer.detect_interference(
            self.interfered_signal, threshold_db=5.0
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('interference_detected', result)
        self.assertIn('interference_frequencies', result)
        self.assertIn('interference_levels_db', result)
        
        # Should detect interference
        self.assertTrue(result['interference_detected'])
        
        # Should find interference near 200 Hz
        interference_freqs = result['interference_frequencies']
        if len(interference_freqs) > 0:
            min_distance = np.min(np.abs(interference_freqs - 200))
            self.assertLess(min_distance, 50)
    
    def test_classify_interference(self):
        """Test interference classification"""
        # Simulate detected interference
        interference_freqs = np.array([100, 2450e6, 900e6])
        interference_levels = np.array([35, 20, 10])
        
        result = self.analyzer.classify_interference(
            interference_freqs, interference_levels
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('classifications', result)
        self.assertIn('total_interferences', result)
        
        classifications = result['classifications']
        self.assertEqual(len(classifications), 3)
        
        # Check classification structure
        for classification in classifications:
            self.assertIn('frequency', classification)
            self.assertIn('level_db', classification)
            self.assertIn('type', classification)
        
        # Check for potential source identification
        wifi_classified = any('WiFi' in c.get('potential_source', '') 
                            for c in classifications)
        cellular_classified = any('Cellular' in c.get('potential_source', '') 
                                for c in classifications)
        
        # Should identify WiFi and Cellular sources
        self.assertTrue(wifi_classified)
        self.assertTrue(cellular_classified)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience analysis functions"""
    
    @unittest.skipUnless(DSP_ANALYSIS_AVAILABLE, "DSP analysis not available")
    def setUp(self):
        """Set up test environment"""
        t = np.arange(0, 1, 1/1000)
        self.test_signal = np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(len(t))
    
    def test_analyze_spectrum(self):
        """Test spectrum analysis convenience function"""
        result = analyze_spectrum(self.test_signal, sample_rate=1000)
        
        self.assertIsInstance(result, dict)
        self.assertIn('frequencies', result)
        self.assertIn('psd_dbm', result)
        self.assertIn('peak_frequencies', result)
        self.assertIn('peak_powers', result)
        self.assertIn('sample_rate', result)
        self.assertIn('method', result)
        
        # Should find peak near 100 Hz
        peak_freqs = result['peak_frequencies']
        if len(peak_freqs) > 0:
            min_distance = np.min(np.abs(peak_freqs - 100))
            self.assertLess(min_distance, 20)
    
    def test_detect_signals(self):
        """Test signal detection convenience function"""
        result = detect_signals(self.test_signal, threshold_db=-50)
        
        self.assertIsInstance(result, dict)
        self.assertIn('energy_detection', result)
        self.assertIn('cfar_detection', result)
        
        # Energy detection should succeed for this signal
        energy_result = result['energy_detection']
        self.assertTrue(energy_result['detected'])
    
    def test_estimate_parameters(self):
        """Test parameter estimation convenience function"""
        result = estimate_parameters(self.test_signal, sample_rate=1000)
        
        self.assertIsInstance(result, dict)
        self.assertIn('frequency', result)
        self.assertIn('amplitude', result)
        self.assertIn('phase', result)
        
        # Frequency estimate should be close to 100 Hz
        freq_result = result['frequency']
        estimated_freq = freq_result['frequency']
        self.assertAlmostEqual(estimated_freq, 100, delta=5)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    @unittest.skipUnless(DSP_ANALYSIS_AVAILABLE, "DSP analysis not available")
    def test_empty_signal_analysis(self):
        """Test analysis with empty signal"""
        analyzer = SpectrumAnalyzer()
        
        empty_signal = np.array([])
        
        # Should handle empty signal gracefully
        try:
            freqs, psd = analyzer.power_spectral_density(empty_signal)
            # If it doesn't raise an exception, check the output
            self.assertEqual(len(freqs), 0)
            self.assertEqual(len(psd), 0)
        except ValueError:
            # It's also acceptable to raise an exception for empty input
            pass
    
    def test_single_sample_signal(self):
        """Test analysis with single sample"""
        analyzer = SpectrumAnalyzer()
        
        single_sample = np.array([1.0])
        
        # Should handle single sample gracefully
        try:
            freqs, psd = analyzer.power_spectral_density(single_sample)
        except ValueError:
            # Single sample might not be sufficient for analysis
            pass
    
    def test_complex_signal_analysis(self):
        """Test analysis with complex signals"""
        analyzer = SpectrumAnalyzer()
        
        # Generate complex signal
        t = np.arange(0, 1, 1/1000)
        complex_signal = np.exp(1j * 2 * np.pi * 100 * t)
        
        freqs, psd = analyzer.power_spectral_density(complex_signal)
        
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(psd), 0)
    
    def test_invalid_method_fallback(self):
        """Test fallback for invalid analysis methods"""
        analyzer = SpectrumAnalyzer()
        
        # Use invalid method - should fall back to default
        freqs, psd = analyzer.power_spectral_density(
            np.random.randn(1000), method="invalid_method"
        )
        
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(psd), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)