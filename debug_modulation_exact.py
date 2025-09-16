#!/usr/bin/env python3
"""
Debug the exact modulation analysis error
"""

import sys
import os
import numpy as np

# Add the project directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer

def debug_modulation_error():
    """Debug the exact modulation analysis error from test"""
    print("=== Debugging Exact Modulation Analysis Error ===")
    
    analyzer = SignalAnalyzer(1e6)
    
    # Recreate the exact test signal that failed
    duration = 0.001  # 1ms
    sample_rate = 1e6
    t = np.arange(0, duration, 1/sample_rate)
    
    # BPSK signal
    symbols = np.random.choice([-1, 1], len(t)//10)
    symbols_upsampled = np.repeat(symbols, 10)
    # Fix length issue
    if len(symbols_upsampled) > len(t):
        symbols_upsampled = symbols_upsampled[:len(t)]
    elif len(symbols_upsampled) < len(t):
        symbols_upsampled = np.pad(symbols_upsampled, (0, len(t) - len(symbols_upsampled)), 'edge')
    
    carrier_freq = 10000
    bpsk_signal = symbols_upsampled * np.exp(1j * 2 * np.pi * carrier_freq * t)
    
    print(f"BPSK signal length: {len(bpsk_signal)}")
    print(f"Time array length: {len(t)}")
    print(f"Symbols upsampled length: {len(symbols_upsampled)}")
    
    try:
        result = analyzer.analyze_modulation(bpsk_signal)
        print(f"Modulation analysis successful: {result.modulation_type}, confidence={result.confidence}")
    except Exception as e:
        print(f"Modulation analysis error: {e}")
        import traceback
        traceback.print_exc()
        
        # Debug step by step
        print("\n--- Step by step debug ---")
        
        # Extract constellation
        try:
            constellation = analyzer._extract_constellation(bpsk_signal)
            print(f"Constellation extracted: {len(constellation)} points")
            
            # PSK analysis
            try:
                psk_scores = analyzer._analyze_psk_modulations(constellation)
                print(f"PSK scores: {psk_scores}")
            except Exception as psk_e:
                print(f"PSK analysis error: {psk_e}")
                
            # QAM analysis
            try:
                qam_scores = analyzer._analyze_qam_modulations(constellation)
                print(f"QAM scores: {qam_scores}")
            except Exception as qam_e:
                print(f"QAM analysis error: {qam_e}")
                
            # FSK analysis
            try:
                fsk_score = analyzer._analyze_fsk_modulation(bpsk_signal)
                print(f"FSK score: {fsk_score}")
            except Exception as fsk_e:
                print(f"FSK analysis error: {fsk_e}")
                
            # ASK analysis
            try:
                ask_score = analyzer._analyze_ask_modulation(bpsk_signal)
                print(f"ASK score: {ask_score}")
            except Exception as ask_e:
                print(f"ASK analysis error: {ask_e}")
                
        except Exception as const_e:
            print(f"Constellation extraction error: {const_e}")

if __name__ == "__main__":
    debug_modulation_error()