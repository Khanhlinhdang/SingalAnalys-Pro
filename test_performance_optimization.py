#!/usr/bin/env python3
"""
Performance Test for Optimized Spectrum Processing
Tests the pyFFTW optimizations and adaptive throttling.
"""

import sys
import os
import time
import numpy as np
import logging

# Add project path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rf_spectrum_analyzer.core.signal_processor import SignalProcessor
from rf_spectrum_analyzer.config.settings import Settings


def test_spectrum_performance():
    """Test spectrum computation performance."""
    print("=== RF Spectrum Analyzer Performance Test ===")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize settings
    settings = Settings()
    
    # Create signal processor
    processor = SignalProcessor(settings)
    
    # Generate test IQ data
    fft_size = settings.dsp.fft_size
    sample_rate = 5e6
    t = np.linspace(0, fft_size / sample_rate, fft_size)
    
    # Create complex test signal (multiple tones)
    test_signal = (
        0.5 * np.exp(2j * np.pi * 1e6 * t) +  # 1 MHz tone
        0.3 * np.exp(2j * np.pi * 2.5e6 * t) +  # 2.5 MHz tone
        0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))  # Noise
    ).astype(np.complex64)
    
    print(f"FFT Method: {processor.fft_method}")
    print(f"FFT Size: {fft_size}")
    print(f"Test Signal Length: {len(test_signal)}")
    print()
    
    # Test different performance modes
    modes = ['fast', 'balanced', 'quality']
    
    for mode in modes:
        print(f"--- Testing {mode} mode ---")
        processor.set_performance_mode(mode)
        
        # Warmup
        for _ in range(5):
            processor.compute_spectrum(test_signal)
        
        # Performance test
        start_time = time.time()
        spectra_computed = 0
        test_duration = 2.0  # seconds
        
        while (time.time() - start_time) < test_duration:
            spectrum = processor.compute_spectrum(test_signal)
            if spectrum is not None:
                spectra_computed += 1
        
        elapsed = time.time() - start_time
        fps = spectra_computed / elapsed
        
        stats = processor.get_performance_stats()
        
        print(f"  Spectra computed: {spectra_computed}")
        print(f"  Elapsed time: {elapsed:.2f}s")
        print(f"  FPS: {fps:.1f}")
        print(f"  Frames skipped: {stats.get('frames_skipped', 0)}")
        print()
    
    # Test averaging performance
    print("--- Testing Averaging Performance ---")
    processor.settings.dsp.averaging = 10  # Enable averaging
    processor._setup_fft()  # Reinitialize with new settings
    
    start_time = time.time()
    for i in range(100):
        spectrum = processor.compute_spectrum(test_signal)
    elapsed = time.time() - start_time
    
    print(f"  100 computations with 10x averaging: {elapsed:.2f}s")
    print(f"  Average time per spectrum: {elapsed/100*1000:.2f}ms")
    
    print("\n=== Performance Test Complete ===")


if __name__ == "__main__":
    try:
        test_spectrum_performance()
    except Exception as e:
        print(f"Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)