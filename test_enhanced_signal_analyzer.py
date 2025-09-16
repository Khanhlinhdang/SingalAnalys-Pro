#!/usr/bin/env python3
"""
Comprehensive test script for enhanced SignalAnalyzer class.
Tests all advanced DSP capabilities and integration.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import time

# Add the project directory to the path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Import the enhanced signal analyzer
from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer

def create_test_signals() -> Dict[str, np.ndarray]:
    """Create various test signals for comprehensive testing."""
    print("Creating test signals...")
    
    signals = {}
    
    # Parameters
    sample_rate = 1e6
    t = np.arange(0, 0.01, 1/sample_rate)  # 10ms
    
    try:
        # BPSK Signal
        print("  - Generating BPSK signal")
        bits = np.random.randint(0, 2, 1000)
        symbols_per_bit = len(t) // len(bits)
        bpsk_symbols = np.repeat(2 * bits - 1, symbols_per_bit)[:len(t)]  # Convert to +1/-1
        carrier = np.exp(1j * 2 * np.pi * 100e3 * t)
        noise = 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        signals['bpsk'] = bpsk_symbols * carrier + noise
        
        # QPSK Signal
        print("  - Generating QPSK signal")
        # Generate random bits and map to QPSK symbols
        qpsk_bits = np.random.randint(0, 2, 2000)
        qpsk_symbols = []
        for i in range(0, len(qpsk_bits), 2):
            if i+1 < len(qpsk_bits):
                bit_pair = qpsk_bits[i:i+2]
                if bit_pair[0] == 0 and bit_pair[1] == 0:
                    qpsk_symbols.append(1+1j)
                elif bit_pair[0] == 0 and bit_pair[1] == 1:
                    qpsk_symbols.append(-1+1j)
                elif bit_pair[0] == 1 and bit_pair[1] == 0:
                    qpsk_symbols.append(1-1j)
                else:
                    qpsk_symbols.append(-1-1j)
        
        symbols_per_qpsk = len(t) // len(qpsk_symbols)
        qpsk_signal = np.repeat(qpsk_symbols, symbols_per_qpsk)[:len(t)]
        carrier_qpsk = np.exp(1j * 2 * np.pi * 150e3 * t)
        noise_qpsk = 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        signals['qpsk'] = qpsk_signal * carrier_qpsk + noise_qpsk
        
        # FSK Signal (Binary FSK)
        print("  - Generating FSK signal")
        fsk_bits = np.random.randint(0, 2, 500)
        fsk_signal = np.zeros(len(t), dtype=complex)
        symbols_per_fsk = len(t) // len(fsk_bits)
        
        for i, bit in enumerate(fsk_bits):
            start_idx = i * symbols_per_fsk
            end_idx = min((i + 1) * symbols_per_fsk, len(t))
            if bit == 0:
                freq = 180e3  # Lower frequency
            else:
                freq = 220e3  # Higher frequency
            fsk_signal[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t[start_idx:end_idx])
        
        noise_fsk = 0.15 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        signals['fsk'] = fsk_signal + noise_fsk
        
        # 16-QAM Signal (simplified)
        print("  - Generating 16-QAM signal")
        qam_constellation = np.array([
            -3-3j, -3-1j, -3+1j, -3+3j,
            -1-3j, -1-1j, -1+1j, -1+3j,
            1-3j, 1-1j, 1+1j, 1+3j,
            3-3j, 3-1j, 3+1j, 3+3j
        ]) / np.sqrt(10)  # Normalize
        
        qam_symbols_idx = np.random.randint(0, 16, 250)
        qam_symbols = qam_constellation[qam_symbols_idx]
        symbols_per_qam = len(t) // len(qam_symbols)
        qam_signal = np.repeat(qam_symbols, symbols_per_qam)[:len(t)]
        carrier_qam = np.exp(1j * 2 * np.pi * 250e3 * t)
        noise_qam = 0.05 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        signals['qam16'] = qam_signal * carrier_qam + noise_qam
        
        # Noisy signal for detection testing
        print("  - Generating noisy signal")
        noise_power = 0.1
        signal_power = 1.0
        carrier_noisy = np.exp(1j * 2 * np.pi * 100e3 * t)
        noise_large = np.sqrt(noise_power/2) * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        signals['noisy'] = signal_power * carrier_noisy + noise_large
        
        # Pure noise for threshold testing
        print("  - Generating noise-only signal")
        signals['noise_only'] = np.sqrt(0.01/2) * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        
    except Exception as e:
        print(f"Error generating signals: {e}")
        # Fallback minimal signals
        t_short = np.arange(0, 0.001, 1/1e6)  # 1ms
        signals['bpsk'] = np.exp(1j * 2 * np.pi * 100e3 * t_short)
        signals['noise_only'] = 0.1 * (np.random.randn(len(t_short)) + 1j * np.random.randn(len(t_short)))
    
    print(f"Created {len(signals)} test signals")
    return signals

def test_signal_analyzer_initialization():
    """Test SignalAnalyzer initialization with advanced features."""
    print("\n=== Testing SignalAnalyzer Initialization ===")
    
    try:
        # Test basic initialization
        analyzer = SignalAnalyzer(sample_rate=1e6)
        print("+ Basic initialization successful")
        
        # Test with different parameters
        analyzer_custom = SignalAnalyzer(sample_rate=2.4e6)
        print("+ Custom parameters initialization successful")
        
        # Check if advanced features are available
        if hasattr(analyzer, 'advanced_features_enabled'):
            print(f"+ Advanced features enabled: {analyzer.advanced_features_enabled}")
        
        # Check for advanced engines
        advanced_components = []
        if hasattr(analyzer, 'demodulation_engine'):
            advanced_components.append('demodulation_engine')
        if hasattr(analyzer, 'signal_detector'):
            advanced_components.append('signal_detector')
        if hasattr(analyzer, 'modulation_analyzer'):
            advanced_components.append('modulation_analyzer')
        if hasattr(analyzer, 'decoding_engine'):
            advanced_components.append('decoding_engine')
            
        print(f"+ Advanced components available: {advanced_components}")
        
        return analyzer
        
    except Exception as e:
        print(f"- Initialization failed: {e}")
        return None

def test_signal_detection(analyzer: SignalAnalyzer, signals: Dict[str, np.ndarray]):
    """Test signal detection capabilities."""
    print("\n=== Testing Signal Detection ===")
    
    results = {}
    
    for signal_name, signal_data in signals.items():
        print(f"\nTesting signal: {signal_name}")
        
        try:
            # Test basic signal detection
            result = analyzer.analyze_signal_comprehensive(signal_data, 100e3, 500e3)
            
            if hasattr(analyzer, '_detect_signal_presence'):
                # Test advanced signal detection
                detection_result = analyzer._detect_signal_presence(signal_data)
                
                print(f"  Signal detected: {detection_result.get('signal_detected', 'Unknown')}")
                print(f"  Confidence: {detection_result.get('confidence', 0.0):.3f}")
                print(f"  SNR estimate: {detection_result.get('snr_estimate', 0.0):.2f} dB")
                print(f"  Noise floor: {detection_result.get('noise_floor', -100):.1f} dB")
                
                results[signal_name] = detection_result
            else:
                print(f"  Basic analysis result available: {result is not None}")
                results[signal_name] = {'basic_analysis': result is not None}
                
        except Exception as e:
            print(f"  - Detection failed: {e}")
            results[signal_name] = {'error': str(e)}
    
    return results

def test_modulation_analysis(analyzer: SignalAnalyzer, signals: Dict[str, np.ndarray]):
    """Test modulation analysis capabilities."""
    print("\n=== Testing Modulation Analysis ===")
    
    results = {}
    
    for signal_name, signal_data in signals.items():
        if signal_name == 'noise_only':  # Skip pure noise
            continue
            
        print(f"\nTesting modulation analysis for: {signal_name}")
        
        try:
            # Test modulation analysis
            mod_result = analyzer.analyze_modulation(signal_data)
            
            print(f"  Detected modulation: {mod_result.modulation_type}")
            print(f"  Confidence: {mod_result.confidence:.3f}")
            print(f"  Symbol rate: {mod_result.symbol_rate}")
            print(f"  Frequency offset: {mod_result.frequency_offset}")
            print(f"  Phase offset: {mod_result.phase_offset:.3f}")
            print(f"  Constellation points: {len(mod_result.constellation_points) if mod_result.constellation_points is not None else 0}")
            
            # Test advanced modulation analysis if available
            if hasattr(analyzer, '_analyze_modulation_advanced'):
                try:
                    advanced_result = analyzer._analyze_modulation_advanced(signal_data)
                    print(f"  Advanced analysis available: {advanced_result.modulation_type}")
                except Exception as e:
                    print(f"  Advanced analysis failed: {e}")
            
            results[signal_name] = {
                'modulation_type': mod_result.modulation_type,
                'confidence': mod_result.confidence,
                'symbol_rate': mod_result.symbol_rate,
                'constellation_size': len(mod_result.constellation_points) if mod_result.constellation_points is not None else 0
            }
            
        except Exception as e:
            print(f"  - Modulation analysis failed: {e}")
            results[signal_name] = {'error': str(e)}
    
    return results

def test_demodulation(analyzer: SignalAnalyzer, signals: Dict[str, np.ndarray]):
    """Test demodulation capabilities."""
    print("\n=== Testing Demodulation ===")
    
    results = {}
    
    for signal_name, signal_data in signals.items():
        if signal_name == 'noise_only':  # Skip pure noise
            continue
            
        print(f"\nTesting demodulation for: {signal_name}")
        
        try:
            # First analyze modulation
            mod_result = analyzer.analyze_modulation(signal_data)
            
            # Then demodulate
            demod_result = analyzer.demodulate_signal(signal_data, mod_result)
            
            print(f"  Demodulation success: {demod_result.success}")
            if demod_result.symbols is not None:
                print(f"  Symbols extracted: {len(demod_result.symbols)}")
            if demod_result.bits is not None:
                print(f"  Bits extracted: {len(demod_result.bits)}")
            print(f"  SNR: {demod_result.snr:.2f} dB" if demod_result.snr else "SNR: Unknown")
            print(f"  Error rate: {demod_result.error_rate:.4f}" if demod_result.error_rate else "Error rate: Unknown")
            
            # Test advanced demodulation if available
            if hasattr(analyzer, '_demodulate_signal_advanced'):
                try:
                    advanced_demod = analyzer._demodulate_signal_advanced(signal_data, mod_result)
                    print(f"  Advanced demodulation available: {advanced_demod.success}")
                except Exception as e:
                    print(f"  Advanced demodulation failed: {e}")
            
            results[signal_name] = {
                'success': demod_result.success,
                'symbols_count': len(demod_result.symbols) if demod_result.symbols is not None else 0,
                'bits_count': len(demod_result.bits) if demod_result.bits is not None else 0,
                'snr': demod_result.snr,
                'error_rate': demod_result.error_rate
            }
            
        except Exception as e:
            print(f"  - Demodulation failed: {e}")
            results[signal_name] = {'error': str(e)}
    
    return results

def test_comprehensive_analysis(analyzer: SignalAnalyzer, signals: Dict[str, np.ndarray]):
    """Test comprehensive signal analysis."""
    print("\n=== Testing Comprehensive Analysis ===")
    
    results = {}
    
    for signal_name, signal_data in signals.items():
        print(f"\nTesting comprehensive analysis for: {signal_name}")
        
        try:
            start_time = time.time()
            result = analyzer.analyze_signal_comprehensive(signal_data, 100e3, 500e3)
            analysis_time = time.time() - start_time
            
            print(f"  Analysis completed in {analysis_time:.3f} seconds")
            print(f"  Result type: {type(result)}")
            
            # Check result components
            if hasattr(result, 'signal_detected'):
                print(f"  Signal detected: {result.signal_detected}")
            if hasattr(result, 'modulation_result'):
                print(f"  Modulation: {result.modulation_result.modulation_type if result.modulation_result else 'None'}")
            if hasattr(result, 'demodulation_result'):
                print(f"  Demodulation success: {result.demodulation_result.success if result.demodulation_result else 'None'}")
            if hasattr(result, 'coding_result'):
                print(f"  Coding detected: {result.coding_result.coding_type if result.coding_result else 'None'}")
            
            results[signal_name] = {
                'analysis_time': analysis_time,
                'result_available': result is not None,
                'analysis_type': str(type(result))
            }
            
        except Exception as e:
            print(f"  - Comprehensive analysis failed: {e}")
            results[signal_name] = {'error': str(e)}
    
    return results

def test_quality_metrics(analyzer: SignalAnalyzer, signals: Dict[str, np.ndarray]):
    """Test signal quality metrics calculation."""
    print("\n=== Testing Signal Quality Metrics ===")
    
    results = {}
    
    # Test quality metrics if available
    if hasattr(analyzer, '_calculate_signal_quality_metrics'):
        for signal_name, signal_data in signals.items():
            print(f"\nTesting quality metrics for: {signal_name}")
            
            try:
                # Create a dummy demodulation result
                from rf_spectrum_analyzer.dsp.signal_analysis import DemodulationResult
                dummy_demod = DemodulationResult(
                    success=True,
                    symbols=np.random.random(100),
                    bits=np.random.randint(0, 2, 200),
                    snr=20.0,
                    error_rate=0.01
                )
                
                metrics = analyzer._calculate_signal_quality_metrics(signal_data, dummy_demod)
                
                print(f"  Quality metrics calculated: {len(metrics)} metrics")
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric_name}: {value:.4f}")
                    else:
                        print(f"    {metric_name}: {value}")
                
                results[signal_name] = metrics
                
            except Exception as e:
                print(f"  - Quality metrics failed: {e}")
                results[signal_name] = {'error': str(e)}
    else:
        print("  Quality metrics method not available")
    
    return results

def test_spectrum_analysis(analyzer: SignalAnalyzer, signals: Dict[str, np.ndarray]):
    """Test spectrum analysis and peak detection."""
    print("\n=== Testing Spectrum Analysis ===")
    
    results = {}
    
    if hasattr(analyzer, '_analyze_spectrum_peaks'):
        for signal_name, signal_data in signals.items():
            print(f"\nTesting spectrum analysis for: {signal_name}")
            
            try:
                spectrum_result = analyzer._analyze_spectrum_peaks(signal_data)
                
                print(f"  Peaks found: {spectrum_result.get('num_peaks', 0)}")
                if spectrum_result.get('strongest_peak_freq'):
                    print(f"  Strongest peak: {spectrum_result['strongest_peak_freq']:.0f} Hz")
                    print(f"  Peak power: {spectrum_result['strongest_peak_power']:.1f} dB")
                
                results[signal_name] = spectrum_result
                
            except Exception as e:
                print(f"  - Spectrum analysis failed: {e}")
                results[signal_name] = {'error': str(e)}
    else:
        print("  Spectrum analysis method not available")
    
    return results

def generate_test_report(all_results: Dict[str, Any]):
    """Generate a comprehensive test report."""
    print("\n" + "="*60)
    print("ENHANCED SIGNALANALYZER TEST REPORT")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, test_results in all_results.items():
        print(f"\n{test_name.upper()}:")
        
        if isinstance(test_results, dict):
            for signal_name, result in test_results.items():
                total_tests += 1
                if 'error' not in result:
                    passed_tests += 1
                    status = "+ PASS"
                else:
                    status = "- FAIL"
                print(f"  {signal_name}: {status}")
        else:
            total_tests += 1
            if test_results:
                passed_tests += 1
                print("  + PASS")
            else:
                print("  - FAIL")
    
    print(f"\nOVERALL RESULTS:")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
    
    # Summary of capabilities
    print(f"\nCAPABILITIES SUMMARY:")
    
    if 'initialization' in all_results and all_results['initialization']:
        analyzer = all_results['initialization']
        print("+ SignalAnalyzer initialization working")
        
        capabilities = []
        if hasattr(analyzer, 'advanced_features_enabled'):
            capabilities.append("Advanced features detection")
        if hasattr(analyzer, '_detect_signal_presence'):
            capabilities.append("Enhanced signal detection")
        if hasattr(analyzer, '_analyze_modulation_advanced'):
            capabilities.append("Advanced modulation analysis")
        if hasattr(analyzer, '_demodulate_signal_advanced'):
            capabilities.append("Advanced demodulation")
        if hasattr(analyzer, '_calculate_signal_quality_metrics'):
            capabilities.append("Signal quality metrics")
        if hasattr(analyzer, '_analyze_spectrum_peaks'):
            capabilities.append("Spectrum peak analysis")
        
        for capability in capabilities:
            print(f"+ {capability}")
    
    print("\n" + "="*60)

def main():
    """Main test function."""
    print("Enhanced SignalAnalyzer Comprehensive Test Suite")
    print("=" * 50)
    
    # Create test signals
    signals = create_test_signals()
    
    # Initialize analyzer
    analyzer = test_signal_analyzer_initialization()
    if not analyzer:
        print("Cannot continue without functional analyzer")
        return
    
    # Run all tests
    all_results = {}
    all_results['initialization'] = analyzer
    
    # Test signal detection
    all_results['signal_detection'] = test_signal_detection(analyzer, signals)
    
    # Test modulation analysis
    all_results['modulation_analysis'] = test_modulation_analysis(analyzer, signals)
    
    # Test demodulation
    all_results['demodulation'] = test_demodulation(analyzer, signals)
    
    # Test comprehensive analysis
    all_results['comprehensive_analysis'] = test_comprehensive_analysis(analyzer, signals)
    
    # Test quality metrics
    all_results['quality_metrics'] = test_quality_metrics(analyzer, signals)
    
    # Test spectrum analysis
    all_results['spectrum_analysis'] = test_spectrum_analysis(analyzer, signals)
    
    # Generate final report
    generate_test_report(all_results)

if __name__ == "__main__":
    main()