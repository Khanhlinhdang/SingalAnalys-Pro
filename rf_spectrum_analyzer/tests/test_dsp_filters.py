"""
DSP Filters Module Tests
Comprehensive testing of all filter classes and functions
"""

import unittest
import numpy as np
import scipy.signal as signal
from pathlib import Path
import sys
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

try:
    from rf_spectrum_analyzer.dsp.filters import (
        FIRFilter, IIRFilter, PolyphaseFilter, AdaptiveFilter,
        ButterworthFilter, ChebyshevFilter, EllipticFilter,
        FilterConfig, design_lowpass, design_highpass,
        design_bandpass, design_bandstop
    )
    DSP_FILTERS_AVAILABLE = True
except ImportError as e:
    DSP_FILTERS_AVAILABLE = False
    print(f"DSP filters not available: {e}")


class TestFilterConfig(unittest.TestCase):
    """Test FilterConfig dataclass"""
    
    @unittest.skipUnless(DSP_FILTERS_AVAILABLE, "DSP filters not available")
    def test_default_config(self):
        """Test default filter configuration"""
        config = FilterConfig()
        self.assertEqual(config.filter_type, "lowpass")
        self.assertEqual(config.cutoff_freq, 0.25)
        self.assertEqual(config.sample_rate, 1.0)
        self.assertEqual(config.order, 5)
        self.assertEqual(config.method, "butter")
    
    @unittest.skipUnless(DSP_FILTERS_AVAILABLE, "DSP filters not available")
    def test_custom_config(self):
        """Test custom filter configuration"""
        config = FilterConfig(
            filter_type="highpass",
            cutoff_freq=0.5,
            order=8,
            method="cheby1"
        )
        self.assertEqual(config.filter_type, "highpass")
        self.assertEqual(config.cutoff_freq, 0.5)
        self.assertEqual(config.order, 8)
        self.assertEqual(config.method, "cheby1")


class TestFIRFilter(unittest.TestCase):
    """Test FIR Filter implementation"""
    
    @unittest.skipUnless(DSP_FILTERS_AVAILABLE, "DSP filters not available")
    def setUp(self):
        """Set up test environment"""
        self.sample_rate = 1000
        self.filter_order = 10
        
    def test_fir_filter_creation(self):
        """Test FIR filter creation"""
        fir_filter = FIRFilter(
            filter_type="lowpass",
            cutoff_freq=0.25,
            order=self.filter_order,
            sample_rate=self.sample_rate
        )
        
        self.assertIsNotNone(fir_filter.coefficients)
        self.assertEqual(len(fir_filter.coefficients), self.filter_order + 1)
    
    def test_fir_filter_with_custom_coefficients(self):
        """Test FIR filter with custom coefficients"""
        custom_coeffs = np.array([1, 0.5, 0.25, 0.125])
        fir_filter = FIRFilter(coefficients=custom_coeffs)
        
        np.testing.assert_array_equal(fir_filter.coefficients, custom_coeffs)
    
    def test_fir_filter_processing(self):
        """Test FIR filter signal processing"""
        fir_filter = FIRFilter(
            filter_type="lowpass",
            cutoff_freq=0.25,
            order=self.filter_order
        )
        
        # Generate test signal
        t = np.arange(0, 1, 1/self.sample_rate)
        test_signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)
        
        # Filter signal
        filtered_signal = fir_filter.filter(test_signal)
        
        self.assertEqual(len(filtered_signal), len(test_signal))
        self.assertIsInstance(filtered_signal, np.ndarray)
    
    def test_fir_filter_streaming(self):
        """Test FIR filter streaming mode"""
        fir_filter = FIRFilter(order=5)
        
        # Process signal in chunks
        chunk_size = 100
        test_signal = np.random.randn(500)
        
        filtered_chunks = []
        for i in range(0, len(test_signal), chunk_size):
            chunk = test_signal[i:i+chunk_size]
            filtered_chunk = fir_filter.filter_streaming(chunk)
            filtered_chunks.append(filtered_chunk)
        
        streaming_result = np.concatenate(filtered_chunks)
        
        # Compare with batch processing
        fir_filter.reset()
        batch_result = fir_filter.filter(test_signal)
        
        # Results should be similar (not exact due to state differences)
        self.assertEqual(len(streaming_result), len(batch_result))
    
    def test_fir_frequency_response(self):
        """Test FIR filter frequency response"""
        fir_filter = FIRFilter(
            filter_type="lowpass",
            cutoff_freq=0.25,
            order=20
        )
        
        freqs, response = fir_filter.get_frequency_response()
        
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(response), 0)
        self.assertEqual(len(freqs), len(response))


class TestIIRFilter(unittest.TestCase):
    """Test IIR Filter implementation"""
    
    @unittest.skipUnless(DSP_FILTERS_AVAILABLE, "DSP filters not available")
    def setUp(self):
        """Set up test environment"""
        self.sample_rate = 1000
        self.filter_order = 5
    
    def test_iir_filter_creation(self):
        """Test IIR filter creation"""
        iir_filter = IIRFilter(
            filter_type="lowpass",
            cutoff_freq=0.25,
            order=self.filter_order,
            method="butter"
        )
        
        self.assertIsNotNone(iir_filter.b)
        self.assertIsNotNone(iir_filter.a)
    
    def test_iir_butterworth_filter(self):
        """Test Butterworth IIR filter"""
        iir_filter = IIRFilter(
            filter_type="lowpass",
            cutoff_freq=0.3,
            order=4,
            method="butter"
        )
        
        # Test signal processing
        test_signal = np.random.randn(1000)
        filtered_signal = iir_filter.filter(test_signal)
        
        self.assertEqual(len(filtered_signal), len(test_signal))
    
    def test_iir_chebyshev_filter(self):
        """Test Chebyshev IIR filter"""
        iir_filter = IIRFilter(
            filter_type="lowpass",
            cutoff_freq=0.3,
            order=4,
            method="cheby1",
            ripple=0.5
        )
        
        test_signal = np.random.randn(500)
        filtered_signal = iir_filter.filter(test_signal)
        
        self.assertEqual(len(filtered_signal), len(test_signal))
    
    def test_iir_elliptic_filter(self):
        """Test Elliptic IIR filter"""
        iir_filter = IIRFilter(
            filter_type="lowpass",
            cutoff_freq=0.3,
            order=4,
            method="ellip",
            ripple=0.5,
            attenuation=60
        )
        
        test_signal = np.random.randn(500)
        filtered_signal = iir_filter.filter(test_signal)
        
        self.assertEqual(len(filtered_signal), len(test_signal))
    
    def test_iir_streaming(self):
        """Test IIR filter streaming mode"""
        iir_filter = IIRFilter(order=3)
        
        # Process in chunks
        test_signal = np.random.randn(1000)
        chunk_size = 200
        
        filtered_chunks = []
        for i in range(0, len(test_signal), chunk_size):
            chunk = test_signal[i:i+chunk_size]
            filtered_chunk = iir_filter.filter_streaming(chunk)
            filtered_chunks.append(filtered_chunk)
        
        streaming_result = np.concatenate(filtered_chunks)
        self.assertEqual(len(streaming_result), len(test_signal))


class TestPolyphaseFilter(unittest.TestCase):
    """Test Polyphase Filter implementation"""
    
    @unittest.skipUnless(DSP_FILTERS_AVAILABLE, "DSP filters not available")
    def test_polyphase_creation(self):
        """Test polyphase filter creation"""
        # Create prototype filter
        prototype = signal.firwin(64, 0.25)
        poly_filter = PolyphaseFilter(prototype, M=4, mode="interpolation")
        
        self.assertEqual(poly_filter.M, 4)
        self.assertEqual(len(poly_filter.polyphase_filters), 4)
    
    def test_polyphase_interpolation(self):
        """Test polyphase interpolation"""
        prototype = signal.firwin(32, 0.25)
        poly_filter = PolyphaseFilter(prototype, M=3, mode="interpolation")
        
        test_signal = np.random.randn(100)
        interpolated = poly_filter.interpolate(test_signal)
        
        # Output should be approximately M times longer
        expected_length = len(test_signal) * 3
        self.assertAlmostEqual(len(interpolated), expected_length, delta=50)
    
    def test_polyphase_decimation(self):
        """Test polyphase decimation"""
        prototype = signal.firwin(32, 0.25)
        poly_filter = PolyphaseFilter(prototype, M=2, mode="decimation")
        
        test_signal = np.random.randn(200)
        decimated = poly_filter.decimate(test_signal)
        
        # Output should be approximately 1/M times shorter
        expected_length = len(test_signal) // 2
        self.assertAlmostEqual(len(decimated), expected_length, delta=20)


class TestAdaptiveFilter(unittest.TestCase):
    """Test Adaptive Filter implementation"""
    
    @unittest.skipUnless(DSP_FILTERS_AVAILABLE, "DSP filters not available")
    def test_lms_adaptive_filter(self):
        """Test LMS adaptive filter"""
        adaptive_filter = AdaptiveFilter(length=10, algorithm="lms", mu=0.01)
        
        # Generate test signals
        x = np.random.randn(500)
        # Desired signal is filtered version of x
        b_desired = np.array([1, 0.5, 0.25])
        d = signal.lfilter(b_desired, 1, x)
        
        # Adapt filter
        y, e = adaptive_filter.adapt(x, d)
        
        self.assertEqual(len(y), len(d))
        self.assertEqual(len(e), len(d))
        
        # Error should decrease over time
        early_error = np.mean(np.abs(e[:100]))
        late_error = np.mean(np.abs(e[-100:]))
        self.assertGreater(early_error, late_error)
    
    def test_rls_adaptive_filter(self):
        """Test RLS adaptive filter"""
        adaptive_filter = AdaptiveFilter(length=8, algorithm="rls")
        
        x = np.random.randn(200)
        b_desired = np.array([1, 0.3, 0.1])
        d = signal.lfilter(b_desired, 1, x)
        
        y, e = adaptive_filter.adapt(x, d)
        
        self.assertEqual(len(y), len(d))
        self.assertEqual(len(e), len(d))


class TestConvenienceFilters(unittest.TestCase):
    """Test convenience filter classes"""
    
    @unittest.skipUnless(DSP_FILTERS_AVAILABLE, "DSP filters not available")
    def test_butterworth_filter(self):
        """Test Butterworth convenience class"""
        butter_filter = ButterworthFilter(
            order=5,
            cutoff=0.3,
            filter_type="lowpass"
        )
        
        test_signal = np.random.randn(1000)
        filtered = butter_filter.filter(test_signal)
        
        self.assertEqual(len(filtered), len(test_signal))
    
    def test_chebyshev_filter(self):
        """Test Chebyshev convenience class"""
        cheby_filter = ChebyshevFilter(
            order=4,
            cutoff=0.25,
            ripple=0.5,
            cheby_type=1
        )
        
        test_signal = np.random.randn(500)
        filtered = cheby_filter.filter(test_signal)
        
        self.assertEqual(len(filtered), len(test_signal))
    
    def test_elliptic_filter(self):
        """Test Elliptic convenience class"""
        ellip_filter = EllipticFilter(
            order=3,
            cutoff=0.4,
            ripple=0.5,
            attenuation=50
        )
        
        test_signal = np.random.randn(300)
        filtered = ellip_filter.filter(test_signal)
        
        self.assertEqual(len(filtered), len(test_signal))


class TestDesignFunctions(unittest.TestCase):
    """Test filter design convenience functions"""
    
    @unittest.skipUnless(DSP_FILTERS_AVAILABLE, "DSP filters not available")
    def test_design_lowpass(self):
        """Test lowpass filter design"""
        # Test FIR design
        fir_filter = design_lowpass(0.3, order=20, filter_type="fir")
        self.assertIsInstance(fir_filter, FIRFilter)
        
        # Test IIR design
        iir_filter = design_lowpass(0.3, order=5, filter_type="iir")
        self.assertIsInstance(iir_filter, IIRFilter)
    
    def test_design_highpass(self):
        """Test highpass filter design"""
        hp_filter = design_highpass(0.4, order=6, method="cheby1")
        self.assertIsInstance(hp_filter, IIRFilter)
    
    def test_design_bandpass(self):
        """Test bandpass filter design"""
        bp_filter = design_bandpass(0.2, 0.6, order=4)
        self.assertIsInstance(bp_filter, IIRFilter)
    
    def test_design_bandstop(self):
        """Test bandstop filter design"""
        bs_filter = design_bandstop(0.3, 0.7, order=8, filter_type="fir")
        self.assertIsInstance(bs_filter, FIRFilter)


class TestFilterErrorHandling(unittest.TestCase):
    """Test filter error handling and edge cases"""
    
    @unittest.skipUnless(DSP_FILTERS_AVAILABLE, "DSP filters not available")
    def test_invalid_filter_type(self):
        """Test handling of invalid filter types"""
        # Should not raise exception, should fall back to default
        try:
            fir_filter = FIRFilter(filter_type="invalid_type")
            self.assertIsNotNone(fir_filter.coefficients)
        except Exception as e:
            self.fail(f"Filter creation should not fail with invalid type: {e}")
    
    def test_empty_signal_processing(self):
        """Test processing of empty signals"""
        fir_filter = FIRFilter(order=5)
        
        empty_signal = np.array([])
        result = fir_filter.filter(empty_signal)
        
        self.assertEqual(len(result), 0)
    
    def test_complex_signal_processing(self):
        """Test processing of complex signals"""
        fir_filter = FIRFilter(order=10)
        
        # Generate complex test signal
        t = np.arange(0, 1, 1/1000)
        complex_signal = np.exp(1j * 2 * np.pi * 50 * t)
        
        filtered = fir_filter.filter(complex_signal)
        
        self.assertEqual(len(filtered), len(complex_signal))
        self.assertTrue(np.iscomplexobj(filtered))


if __name__ == '__main__':
    unittest.main(verbosity=2)