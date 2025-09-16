#!/usr/bin/env python3
"""
Debug the comprehensive analysis issues
"""

import sys
import os
import numpy as np

# Add the project directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer

def debug_comprehensive_analysis():
    """Debug why comprehensive analysis returns Unknown modulation"""
    print("=== Debugging Comprehensive Analysis ===")
    
    analyzer = SignalAnalyzer(1e6)
    
    # Create a simple BPSK signal
    duration = 0.001  # 1ms
    sample_rate = 1e6
    t = np.arange(0, duration, 1/sample_rate)
    
    # BPSK signal
    symbols = np.random.choice([-1, 1], len(t)//10)
    symbols_upsampled = np.repeat(symbols, 10)
    if len(symbols_upsampled) < len(t):
        symbols_upsampled = np.pad(symbols_upsampled, (0, len(t) - len(symbols_upsampled)), 'edge')
    elif len(symbols_upsampled) > len(t):
        symbols_upsampled = symbols_upsampled[:len(t)]
    
    carrier_freq = 10000
    bpsk_signal = symbols_upsampled * np.exp(1j * 2 * np.pi * carrier_freq * t)
    
    print(f"BPSK signal created: {len(bpsk_signal)} samples")
    
    # Test individual components
    print("\n--- Testing basic modulation analysis ---")
    basic_result = analyzer.analyze_modulation(bpsk_signal)
    print(f"Basic result: {basic_result.modulation_type}, confidence={basic_result.confidence}")
    
    print("\n--- Testing advanced modulation analysis ---")
    advanced_result = analyzer._analyze_modulation_advanced(bpsk_signal)
    print(f"Advanced result: {advanced_result.modulation_type}, confidence={advanced_result.confidence}")
    
    print("\n--- Testing comprehensive analysis ---")
    comp_result = analyzer.analyze_signal_comprehensive(bpsk_signal, 100e6, 1e6)
    print(f"Comprehensive modulation: {comp_result['modulation']['type']}, confidence={comp_result['modulation']['confidence']}")
    print(f"Signal detected: {comp_result['detection']['signal_detected']}")
    
    # Check if the issue is in the advanced analyzer
    if hasattr(analyzer, 'modulation_analyzer'):
        print("\n--- Testing modulation analyzer directly ---")
        try:
            mod_analyzer_result = analyzer.modulation_analyzer.detect_modulation(bpsk_signal)
            print(f"Modulation analyzer result: {mod_analyzer_result}")
        except Exception as e:
            print(f"Modulation analyzer error: {e}")

if __name__ == "__main__":
    debug_comprehensive_analysis()