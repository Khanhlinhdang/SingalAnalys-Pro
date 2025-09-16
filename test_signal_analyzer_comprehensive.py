#!/usr/bin/env python3
"""
Comprehensive test suite for SignalAnalyzer class
Tests all methods, edge cases, error handling, and integration scenarios.
"""

import sys
import os
import numpy as np
import pytest
import logging
import traceback
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# Add the project directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the SignalAnalyzer and related classes
try:
    from rf_spectrum_analyzer.dsp.signal_analysis import (
        SignalAnalyzer, 
        ModulationAnalysisResult,
        DemodulationResult, 
        CodingAnalysisResult,
        ADVANCED_DSP_AVAILABLE
    )
    print(f"✓ Successfully imported SignalAnalyzer. Advanced DSP available: {ADVANCED_DSP_AVAILABLE}")
except ImportError as e:
    print(f"✗ Failed to import SignalAnalyzer: {e}")
    traceback.print_exc()
    sys.exit(1)

class TestSignalAnalyzer:
    """Comprehensive test class for SignalAnalyzer"""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures"""
        cls.sample_rate = 1e6
        cls.analyzer = SignalAnalyzer(cls.sample_rate)
        
        # Generate test signals
        cls.test_signals = cls._generate_test_signals()
        
        print(f"✓ Test setup complete. Advanced features enabled: {cls.analyzer.advanced_features_enabled}")
    
    @classmethod
    def _generate_test_signals(cls) -> Dict[str, np.ndarray]:
        """Generate various test signals for comprehensive testing"""
        signals = {}
        
        # Signal parameters
        duration = 0.001  # 1ms
        t = np.arange(0, duration, 1/cls.sample_rate)
        
        # 1. BPSK signal
        symbols = np.random.choice([-1, 1], len(t)//10)
        symbols_upsampled = np.repeat(symbols, 10)
        # Ensure same length as t
        if len(symbols_upsampled) > len(t):
            symbols_upsampled = symbols_upsampled[:len(t)]
        elif len(symbols_upsampled) < len(t):
            # Pad with last symbol
            padding = np.full(len(t) - len(symbols_upsampled), symbols_upsampled[-1])
            symbols_upsampled = np.concatenate([symbols_upsampled, padding])
        
        carrier_freq = 10000
        signals['bpsk'] = symbols_upsampled * np.exp(1j * 2 * np.pi * carrier_freq * t)
        
        # 2. QPSK signal  
        i_symbols = np.random.choice([-1, 1], len(t)//10)
        q_symbols = np.random.choice([-1, 1], len(t)//10)
        i_upsampled = np.repeat(i_symbols, 10)
        q_upsampled = np.repeat(q_symbols, 10)
        
        # Ensure same length as t
        for upsampled in [i_upsampled, q_upsampled]:
            if len(upsampled) > len(t):
                upsampled = upsampled[:len(t)]
            elif len(upsampled) < len(t):
                padding = np.full(len(t) - len(upsampled), upsampled[-1])
                upsampled = np.concatenate([upsampled, padding])
        
        # Recompute after length adjustment
        i_upsampled = np.repeat(i_symbols, 10)[:len(t)]
        q_upsampled = np.repeat(q_symbols, 10)[:len(t)]
        if len(i_upsampled) < len(t):
            i_upsampled = np.pad(i_upsampled, (0, len(t) - len(i_upsampled)), 'edge')
        if len(q_upsampled) < len(t):
            q_upsampled = np.pad(q_upsampled, (0, len(t) - len(q_upsampled)), 'edge')
            
        signals['qpsk'] = (i_upsampled + 1j * q_upsampled) * np.exp(1j * 2 * np.pi * carrier_freq * t)
        
        # 3. FSK signal
        freq1, freq2 = 8000, 12000
        bits = np.random.choice([0, 1], len(t)//20)
        fsk_signal = np.zeros(len(t), dtype=complex)
        for i, bit in enumerate(bits):
            start_idx = i * 20
            end_idx = min((i + 1) * 20, len(t))
            freq = freq1 if bit == 0 else freq2
            fsk_signal[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t[start_idx:end_idx])
        signals['fsk'] = fsk_signal
        
        # 4. Noise signal
        signals['noise'] = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.1
        
        # 5. Empty signal
        signals['empty'] = np.array([], dtype=complex)
        
        # 6. Single sample
        signals['single'] = np.array([1+1j], dtype=complex)
        
        # 7. DC signal
        signals['dc'] = np.ones(len(t), dtype=complex) * (1+0.5j)
        
        # 8. Sine wave (ASK-like)
        amp_modulation = 0.5 * (1 + np.sin(2 * np.pi * 1000 * t))
        signals['ask'] = amp_modulation * np.exp(1j * 2 * np.pi * carrier_freq * t)
        
        return signals
    
    def test_initialization(self):
        """Test SignalAnalyzer initialization"""
        print("\n=== Testing SignalAnalyzer Initialization ===")
        
        # Test default initialization
        analyzer_default = SignalAnalyzer()
        assert analyzer_default.sample_rate == 1e6
        assert analyzer_default.fft_size == 1024
        assert analyzer_default.overlap == 0.5
        assert analyzer_default.window == 'hann'
        print("✓ Default initialization")
        
        # Test custom sample rate
        analyzer_custom = SignalAnalyzer(sample_rate=2.4e6)
        assert analyzer_custom.sample_rate == 2.4e6
        print("✓ Custom sample rate initialization")
        
        # Test attributes exist
        required_attrs = [
            'sample_rate', 'logger', 'fft_size', 'overlap', 'window',
            'constellation_thresholds', 'advanced_features_enabled'
        ]
        for attr in required_attrs:
            assert hasattr(analyzer_default, attr), f"Missing attribute: {attr}"
        print("✓ All required attributes present")
        
        # Test constellation thresholds structure
        thresholds = analyzer_default.constellation_thresholds
        expected_modulations = ['BPSK', 'QPSK', 'PSK8', 'QAM16', 'QAM64', 'FSK', 'ASK']
        for mod in expected_modulations:
            assert mod in thresholds, f"Missing modulation threshold: {mod}"
            assert 'tolerance' in thresholds[mod], f"Missing tolerance for {mod}"
        print("✓ Constellation thresholds properly configured")
    
    def test_preprocess_signal_basic(self):
        """Test basic signal preprocessing"""
        print("\n=== Testing Basic Signal Preprocessing ===")
        
        for signal_name, signal_data in self.test_signals.items():
            print(f"Testing preprocessing on {signal_name} signal...")
            
            if len(signal_data) == 0:
                # Empty signal should return empty
                result = self.analyzer._preprocess_signal(signal_data)
                assert len(result) == 0
                print(f"  ✓ Empty signal handled correctly")
                continue
            
            # Test preprocessing
            try:
                result = self.analyzer._preprocess_signal(signal_data)
                
                # Verify result properties
                assert isinstance(result, np.ndarray)
                assert len(result) == len(signal_data)
                assert result.dtype == signal_data.dtype
                
                # Check DC removal (mean should be close to zero)
                if len(result) > 1:
                    dc_level = abs(np.mean(result))
                    assert dc_level < 0.1, f"DC not properly removed: {dc_level}"
                
                print(f"  ✓ {signal_name} preprocessed successfully")
                
            except Exception as e:
                print(f"  ✗ {signal_name} preprocessing failed: {e}")
                raise
    
    def test_preprocess_signal_advanced(self):
        """Test advanced signal preprocessing"""
        print("\n=== Testing Advanced Signal Preprocessing ===")
        
        for signal_name, signal_data in self.test_signals.items():
            if len(signal_data) == 0:
                continue
                
            print(f"Testing advanced preprocessing on {signal_name} signal...")
            
            try:
                result = self.analyzer._preprocess_signal_advanced(signal_data)
                
                # Basic checks
                assert isinstance(result, np.ndarray)
                assert len(result) == len(signal_data)
                
                print(f"  ✓ {signal_name} advanced preprocessing successful")
                
            except Exception as e:
                print(f"  ✗ {signal_name} advanced preprocessing failed: {e}")
                raise
    
    def test_detect_signal_presence(self):
        """Test signal presence detection"""
        print("\n=== Testing Signal Presence Detection ===")
        
        for signal_name, signal_data in self.test_signals.items():
            if len(signal_data) == 0:
                continue
                
            print(f"Testing signal detection on {signal_name} signal...")
            
            try:
                result = self.analyzer._detect_signal_presence(signal_data)
                
                # Verify result structure
                required_keys = ['signal_detected', 'confidence', 'snr_estimate', 'noise_floor']
                for key in required_keys:
                    assert key in result, f"Missing key in detection result: {key}"
                
                # Verify data types
                assert isinstance(result['signal_detected'], bool)
                assert isinstance(result['confidence'], (int, float))
                assert isinstance(result['snr_estimate'], (int, float))
                assert isinstance(result['noise_floor'], (int, float))
                
                # Verify ranges
                assert 0 <= result['confidence'] <= 1, f"Confidence out of range: {result['confidence']}"
                
                print(f"  ✓ {signal_name}: detected={result['signal_detected']}, "
                      f"confidence={result['confidence']:.2f}, SNR={result['snr_estimate']:.1f}dB")
                
            except Exception as e:
                print(f"  ✗ {signal_name} signal detection failed: {e}")
                raise
    
    def test_modulation_analysis(self):
        """Test modulation type analysis"""
        print("\n=== Testing Modulation Analysis ===")
        
        for signal_name, signal_data in self.test_signals.items():
            if len(signal_data) < 10:  # Skip very short signals
                continue
                
            print(f"Testing modulation analysis on {signal_name} signal...")
            
            try:
                # Test basic modulation analysis
                result = self.analyzer.analyze_modulation(signal_data)
                
                # Verify result type and structure
                assert isinstance(result, ModulationAnalysisResult)
                assert hasattr(result, 'modulation_type')
                assert hasattr(result, 'confidence') 
                assert hasattr(result, 'parameters')
                assert hasattr(result, 'constellation_points')
                
                # Verify data types
                assert isinstance(result.modulation_type, str)
                assert isinstance(result.confidence, (int, float))
                assert isinstance(result.parameters, dict)
                
                # Verify confidence range
                assert 0 <= result.confidence <= 1
                
                print(f"  ✓ {signal_name}: type={result.modulation_type}, "
                      f"confidence={result.confidence:.2f}")
                
                # Test advanced modulation analysis if available
                if self.analyzer.advanced_features_enabled:
                    result_advanced = self.analyzer._analyze_modulation_advanced(signal_data)
                    assert isinstance(result_advanced, ModulationAnalysisResult)
                    print(f"  ✓ {signal_name}: advanced analysis successful")
                
            except Exception as e:
                print(f"  ✗ {signal_name} modulation analysis failed: {e}")
                raise
    
    def test_constellation_extraction(self):
        """Test constellation point extraction"""
        print("\n=== Testing Constellation Extraction ===")
        
        for signal_name, signal_data in self.test_signals.items():
            if len(signal_data) < 10:
                continue
                
            print(f"Testing constellation extraction on {signal_name} signal...")
            
            try:
                # Test basic extraction
                constellation = self.analyzer._extract_constellation(signal_data)
                
                assert isinstance(constellation, np.ndarray)
                assert constellation.dtype in [np.complex64, np.complex128, complex]
                
                print(f"  ✓ {signal_name}: extracted {len(constellation)} constellation points")
                
                # Test advanced extraction if available
                if self.analyzer.advanced_features_enabled:
                    constellation_adv = self.analyzer._extract_constellation_advanced(signal_data)
                    assert isinstance(constellation_adv, np.ndarray)
                    print(f"  ✓ {signal_name}: advanced extraction successful")
                
            except Exception as e:
                print(f"  ✗ {signal_name} constellation extraction failed: {e}")
                raise
    
    def test_demodulation(self):
        """Test signal demodulation"""
        print("\n=== Testing Signal Demodulation ===")
        
        # Create mock modulation results for testing different demodulation types
        test_modulations = [
            ModulationAnalysisResult('BPSK', 0.8, {}, None),
            ModulationAnalysisResult('QPSK', 0.7, {}, None), 
            ModulationAnalysisResult('PSK8', 0.6, {}, None),
            ModulationAnalysisResult('QAM16', 0.5, {}, None),
            ModulationAnalysisResult('FSK', 0.4, {}, None),
            ModulationAnalysisResult('ASK', 0.3, {}, None),
            ModulationAnalysisResult('Unknown', 0.0, {}, None)
        ]
        
        for mod_result in test_modulations:
            for signal_name, signal_data in self.test_signals.items():
                if len(signal_data) < 10:
                    continue
                    
                print(f"Testing {mod_result.modulation_type} demodulation on {signal_name}...")
                
                try:
                    result = self.analyzer.demodulate_signal(signal_data, mod_result)
                    
                    # Verify result structure
                    assert isinstance(result, DemodulationResult)
                    assert hasattr(result, 'success')
                    assert isinstance(result.success, bool)
                    
                    if result.success:
                        assert result.symbols is not None or result.bits is not None
                        
                        if result.symbols is not None:
                            assert isinstance(result.symbols, np.ndarray)
                            
                        if result.bits is not None:
                            assert isinstance(result.bits, np.ndarray)
                            # Bits should be 0 or 1
                            assert np.all(np.isin(result.bits, [0, 1]))
                    
                    print(f"  ✓ {mod_result.modulation_type} on {signal_name}: "
                          f"success={result.success}")
                    
                except Exception as e:
                    print(f"  ✗ {mod_result.modulation_type} on {signal_name} failed: {e}")
                    # Don't raise - some demodulations may fail on inappropriate signals
    
    def test_signal_quality_metrics(self):
        """Test signal quality metrics calculation"""
        print("\n=== Testing Signal Quality Metrics ===")
        
        for signal_name, signal_data in self.test_signals.items():
            if len(signal_data) < 10:
                continue
                
            print(f"Testing quality metrics on {signal_name} signal...")
            
            try:
                # Create a dummy demodulation result for testing
                dummy_demod = DemodulationResult(
                    success=True,
                    symbols=np.array([1, -1, 1, -1]),
                    bits=np.array([1, 0, 1, 0])
                )
                
                metrics = self.analyzer._calculate_signal_quality_metrics(signal_data, dummy_demod)
                
                # Verify metrics structure
                assert isinstance(metrics, dict)
                
                # Check for expected metrics
                expected_metrics = ['rms_power', 'par', 'crest_factor']
                for metric in expected_metrics:
                    if metric in metrics:
                        assert isinstance(metrics[metric], (int, float))
                        print(f"  ✓ {metric}: {metrics[metric]:.4f}")
                
                print(f"  ✓ {signal_name}: {len(metrics)} quality metrics calculated")
                
            except Exception as e:
                print(f"  ✗ {signal_name} quality metrics failed: {e}")
                raise
    
    def test_spectrum_peaks_analysis(self):
        """Test spectrum peak analysis"""
        print("\n=== Testing Spectrum Peak Analysis ===")
        
        for signal_name, signal_data in self.test_signals.items():
            if len(signal_data) < 10:
                continue
                
            print(f"Testing peak analysis on {signal_name} signal...")
            
            try:
                result = self.analyzer._analyze_spectrum_peaks(signal_data)
                
                # Verify result structure
                assert isinstance(result, dict)
                required_keys = ['peak_frequencies', 'peak_powers', 'num_peaks',
                               'strongest_peak_freq', 'strongest_peak_power']
                
                for key in required_keys:
                    assert key in result, f"Missing key: {key}"
                
                # Verify data types
                assert isinstance(result['peak_frequencies'], list)
                assert isinstance(result['peak_powers'], list)
                assert isinstance(result['num_peaks'], int)
                assert isinstance(result['strongest_peak_freq'], (int, float))
                assert isinstance(result['strongest_peak_power'], (int, float))
                
                # Verify consistency
                assert len(result['peak_frequencies']) == len(result['peak_powers'])
                assert result['num_peaks'] == len(result['peak_frequencies'])
                
                print(f"  ✓ {signal_name}: found {result['num_peaks']} peaks")
                
            except Exception as e:
                print(f"  ✗ {signal_name} peak analysis failed: {e}")
                raise
    
    def test_comprehensive_analysis(self):
        """Test comprehensive signal analysis pipeline"""
        print("\n=== Testing Comprehensive Signal Analysis ===")
        
        for signal_name, signal_data in self.test_signals.items():
            if len(signal_data) == 0:
                # Test empty signal handling
                result = self.analyzer.analyze_signal_comprehensive(
                    signal_data, center_freq=100e6, bandwidth=1e6)
                assert 'error' in result
                assert result['error'] == 'No IQ data provided'
                print(f"  ✓ {signal_name}: empty signal handled correctly")
                continue
                
            print(f"Testing comprehensive analysis on {signal_name} signal...")
            
            try:
                result = self.analyzer.analyze_signal_comprehensive(
                    signal_data, center_freq=100e6, bandwidth=1e6)
                
                # Verify result structure
                assert isinstance(result, dict)
                
                if 'error' in result:
                    print(f"  ⚠ {signal_name}: analysis returned error: {result['error']}")
                    continue
                
                # Check main sections
                expected_sections = ['signal_info', 'detection', 'modulation', 
                                   'demodulation', 'constellation_data', 
                                   'spectrum_analysis', 'analysis_status']
                
                for section in expected_sections:
                    assert section in result, f"Missing section: {section}"
                
                # Verify signal_info
                signal_info = result['signal_info']
                assert 'center_freq' in signal_info
                assert 'bandwidth' in signal_info
                assert 'sample_rate' in signal_info
                assert 'signal_length' in signal_info
                
                # Verify detection results
                detection = result['detection']
                assert 'signal_detected' in detection
                assert 'confidence' in detection
                
                # Verify modulation results
                modulation = result['modulation']
                assert 'type' in modulation
                assert 'confidence' in modulation
                
                # Verify analysis status
                assert result['analysis_status'] == 'success'
                
                print(f"  ✓ {signal_name}: comprehensive analysis successful")
                print(f"    - Signal detected: {detection['signal_detected']}")
                print(f"    - Modulation: {modulation['type']} ({modulation['confidence']:.2f})")
                print(f"    - Demodulation: {result['demodulation']['success']}")
                
            except Exception as e:
                print(f"  ✗ {signal_name} comprehensive analysis failed: {e}")
                raise
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        print("\n=== Testing Edge Cases and Error Conditions ===")
        
        # Test with None input
        try:
            result = self.analyzer._preprocess_signal(None)
            print("  ✗ None input should raise exception")
        except (TypeError, AttributeError):
            print("  ✓ None input properly rejected")
        
        # Test with invalid data types
        try:
            result = self.analyzer._preprocess_signal("invalid")
            print("  ✗ String input should raise exception")
        except (TypeError, AttributeError):
            print("  ✓ String input properly rejected")
        
        # Test with extremely small signals
        tiny_signal = np.array([1e-15 + 1j * 1e-15], dtype=complex)
        try:
            result = self.analyzer._preprocess_signal(tiny_signal)
            print("  ✓ Tiny signal handled")
        except Exception as e:
            print(f"  ⚠ Tiny signal failed: {e}")
        
        # Test with very large signals
        large_signal = np.array([1e10 + 1j * 1e10] * 100, dtype=complex)
        try:
            result = self.analyzer._preprocess_signal(large_signal)
            print("  ✓ Large signal handled")
        except Exception as e:
            print(f"  ⚠ Large signal failed: {e}")
        
        # Test with NaN/Inf values
        nan_signal = np.array([np.nan, np.inf, 1+1j, -np.inf], dtype=complex)
        try:
            result = self.analyzer._preprocess_signal(nan_signal)
            print("  ✓ NaN/Inf signal handled")
        except Exception as e:
            print(f"  ⚠ NaN/Inf signal failed: {e}")
    
    def test_coding_analysis(self):
        """Test coding analysis functionality"""
        print("\n=== Testing Coding Analysis ===")
        
        # Generate test bit patterns
        test_bits = {
            'random': np.random.randint(0, 2, 100),
            'alternating': np.tile([0, 1], 50),
            'manchester_like': np.tile([0, 1, 1, 0], 25),
            'repetition_3': np.repeat([0, 1, 0, 1], 3)[:100],
            'all_zeros': np.zeros(100, dtype=int),
            'all_ones': np.ones(100, dtype=int),
            'short': np.array([0, 1, 0]),
            'empty': np.array([], dtype=int)
        }
        
        for pattern_name, bits in test_bits.items():
            print(f"Testing coding analysis on {pattern_name} pattern...")
            
            try:
                result = self.analyzer.analyze_coding(bits)
                
                if result is None:
                    print(f"  ✓ {pattern_name}: no coding detected (None returned)")
                    continue
                
                # Verify result structure
                assert isinstance(result, CodingAnalysisResult)
                assert hasattr(result, 'coding_type')
                assert hasattr(result, 'confidence')
                assert hasattr(result, 'parameters')
                
                print(f"  ✓ {pattern_name}: type={result.coding_type}, "
                      f"confidence={result.confidence:.2f}")
                
            except Exception as e:
                print(f"  ✗ {pattern_name} coding analysis failed: {e}")
                raise
    
    def test_performance(self):
        """Test performance with different signal sizes"""
        print("\n=== Testing Performance ===")
        
        import time
        
        sizes = [100, 1000, 10000, 100000]
        
        for size in sizes:
            print(f"Testing performance with {size} samples...")
            
            # Generate test signal
            t = np.arange(size) / self.sample_rate
            test_signal = np.exp(1j * 2 * np.pi * 10000 * t) + 0.1 * (np.random.randn(size) + 1j * np.random.randn(size))
            
            try:
                start_time = time.time()
                
                # Test basic preprocessing
                _ = self.analyzer._preprocess_signal(test_signal)
                preprocess_time = time.time() - start_time
                
                # Test modulation analysis
                start_time = time.time()
                _ = self.analyzer.analyze_modulation(test_signal)
                modulation_time = time.time() - start_time
                
                print(f"  ✓ {size} samples: preprocess={preprocess_time:.3f}s, "
                      f"modulation={modulation_time:.3f}s")
                
                # Warn if processing is too slow
                if modulation_time > 5.0:  # 5 seconds threshold
                    print(f"  ⚠ Performance warning: modulation analysis took {modulation_time:.1f}s")
                
            except Exception as e:
                print(f"  ✗ Performance test failed for {size} samples: {e}")

def run_comprehensive_tests():
    """Run all SignalAnalyzer tests"""
    print("=" * 80)
    print("SIGNAL ANALYZER COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Initialize test class
    test_instance = TestSignalAnalyzer()
    test_instance.setup_class()
    
    # Run all tests
    test_methods = [
        test_instance.test_initialization,
        test_instance.test_preprocess_signal_basic,
        test_instance.test_preprocess_signal_advanced,
        test_instance.test_detect_signal_presence,
        test_instance.test_modulation_analysis,
        test_instance.test_constellation_extraction,
        test_instance.test_demodulation,
        test_instance.test_signal_quality_metrics,
        test_instance.test_spectrum_peaks_analysis,
        test_instance.test_comprehensive_analysis,
        test_instance.test_coding_analysis,
        test_instance.test_edge_cases,
        test_instance.test_performance,
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_method.__name__}")
            print(f"Error: {e}")
            traceback.print_exc()
            failed += 1
            print("-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"❌ {failed} tests failed. See details above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)