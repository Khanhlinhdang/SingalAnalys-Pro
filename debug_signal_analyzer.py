#!/usr/bin/env python3
"""
Debug script to identify specific issues in SignalAnalyzer
"""

import sys
import os
import numpy as np

# Add the project directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer

def debug_signal_detection():
    """Debug signal detection issues"""
    print("=== Debugging Signal Detection ===")
    
    analyzer = SignalAnalyzer(1e6)
    
    # Create test signal
    t = np.arange(1000) / 1e6
    test_signal = np.exp(1j * 2 * np.pi * 10000 * t)
    
    # Test signal detection
    try:
        result = analyzer._detect_signal_presence(test_signal)
        print(f"Detection result: {result}")
        print(f"Type of signal_detected: {type(result['signal_detected'])}")
        print(f"Value of signal_detected: {result['signal_detected']}")
        
        if hasattr(analyzer, 'signal_detector'):
            print("Advanced signal detector available")
            try:
                detection_result = analyzer.signal_detector.energy_detection(test_signal)
                print(f"Advanced detection result type: {type(detection_result)}")
                print(f"Advanced detection result: {detection_result}")
                if hasattr(detection_result, 'signal_detected'):
                    print(f"signal_detected type: {type(detection_result.signal_detected)}")
                    print(f"signal_detected value: {detection_result.signal_detected}")
            except Exception as e:
                print(f"Advanced detection error: {e}")
        else:
            print("No advanced signal detector")
    except Exception as e:
        print(f"Signal detection error: {e}")
        import traceback
        traceback.print_exc()

def debug_modulation_analysis():
    """Debug modulation analysis issues"""
    print("\n=== Debugging Modulation Analysis ===")
    
    analyzer = SignalAnalyzer(1e6)
    
    # Create simple BPSK signal
    symbols = np.random.choice([-1, 1], 100)
    symbols_upsampled = np.repeat(symbols, 10)
    t = np.arange(len(symbols_upsampled)) / 1e6
    test_signal = symbols_upsampled * np.exp(1j * 2 * np.pi * 10000 * t)
    
    print(f"Test signal length: {len(test_signal)}")
    print(f"Test signal range: {np.min(test_signal)} to {np.max(test_signal)}")
    
    # Extract constellation
    try:
        constellation = analyzer._extract_constellation(test_signal)
        print(f"Constellation length: {len(constellation)}")
        print(f"Constellation range: {np.min(constellation)} to {np.max(constellation)}")
        
        if len(constellation) > 0:
            # Test PSK analysis
            psk_scores = analyzer._analyze_psk_modulations(constellation)
            print(f"PSK scores: {psk_scores}")
            
            # Test QAM analysis  
            qam_scores = analyzer._analyze_qam_modulations(constellation)
            print(f"QAM scores: {qam_scores}")
            
    except Exception as e:
        print(f"Modulation analysis error: {e}")
        import traceback
        traceback.print_exc()

def debug_grid_score():
    """Debug grid score calculation"""
    print("\n=== Debugging Grid Score Calculation ===")
    
    analyzer = SignalAnalyzer(1e6)
    
    # Test with different value ranges
    test_cases = [
        np.array([1, 1, 1, 1]),  # All same values
        np.array([1, 2, 3, 4]),  # Distinct values
        np.array([]),  # Empty
        np.array([1.0]),  # Single value
        np.random.randn(100),  # Random values
    ]
    
    for i, values in enumerate(test_cases):
        print(f"\nTest case {i+1}: {len(values)} values")
        if len(values) > 0:
            print(f"  Range: {np.min(values):.3f} to {np.max(values):.3f}")
            print(f"  Unique values: {len(np.unique(values))}")
        
        try:
            score = analyzer._calculate_grid_score(values, 4)
            print(f"  Grid score: {score}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    debug_signal_detection()
    debug_modulation_analysis()
    debug_grid_score()