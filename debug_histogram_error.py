#!/usr/bin/env python3
"""
Debug the histogram bin error in modulation analysis
"""

import sys
import os
import numpy as np

# Add the project directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer

def debug_histogram_error():
    """Debug the histogram bin error"""
    print("=== Debugging Histogram Bin Error ===")
    
    analyzer = SignalAnalyzer(1e6)
    
    # Create test signals that might cause the error
    test_cases = [
        ("constant", np.ones(100, dtype=complex)),
        ("zero", np.zeros(100, dtype=complex)),
        ("very_small", np.ones(100, dtype=complex) * 1e-15),
        ("normal", np.random.randn(100) + 1j * np.random.randn(100)),
    ]
    
    for name, signal in test_cases:
        print(f"\n--- Testing {name} signal ---")
        print(f"Signal range: {np.min(signal)} to {np.max(signal)}")
        print(f"Signal std: {np.std(signal)}")
        print(f"Unique values: {len(np.unique(signal))}")
        
        try:
            # Test ASK analysis (this seems to cause the bin error)
            ask_score = analyzer._analyze_ask_modulation(signal)
            print(f"ASK score: {ask_score}")
        except Exception as e:
            print(f"ASK analysis error: {e}")
            
            # Debug the histogram call
            amplitudes = np.abs(signal)
            print(f"Amplitudes range: {np.min(amplitudes)} to {np.max(amplitudes)}")
            print(f"Amplitudes unique: {len(np.unique(amplitudes))}")
            
            try:
                hist, bins = np.histogram(amplitudes, bins=30)
                print(f"Histogram created successfully: {len(hist)} bins")
            except Exception as hist_e:
                print(f"Histogram error: {hist_e}")

if __name__ == "__main__":
    debug_histogram_error()