#!/usr/bin/env python3
"""
Simple pyFFTW performance test for Enhanced Signal Analysis
"""

import sys
import os
import time
import numpy as np
import logging

# Add project path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rf_spectrum_analyzer.dsp.enhanced_analysis import EnhancedSignalAnalysis


def test_basic_pyfftw_optimization():
    """Test basic pyFFTW optimization performance."""
    print("=== Enhanced Signal Analysis pyFFTW Performance Test ===")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test parameters
    sample_rate = 5e6
    fft_size = 1024
    
    # Create analyzer
    analyzer = EnhancedSignalAnalysis(sample_rate=sample_rate, fft_size=fft_size)
    
    # Get analysis info
    info = analyzer.get_analysis_info()
    print(f"FFT Method: {info['fft_method']}")
    print(f"pyFFTW Available: {info['pyfftw_available']}")
    print(f"SDRConnect Available: {info['sdrconnect_available']}")
    print()
    
    # Generate test signal with multiple tones
    t = np.linspace(0, fft_size / sample_rate, fft_size)
    test_signal = (
        0.8 * np.exp(2j * np.pi * 1.2e6 * t) +    # Strong 1.2 MHz tone
        0.4 * np.exp(2j * np.pi * 0.8e6 * t) +    # Weaker 0.8 MHz tone
        0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))  # Noise
    ).astype(np.complex64)
    
    print("Running basic analysis performance test...")
    
    # Warmup runs
    for _ in range(3):
        result = analyzer._basic_analysis(test_signal)  # Test basic analysis only
    
    # Performance test
    start_time = time.time()
    num_analyses = 100
    
    for i in range(num_analyses):
        result = analyzer._basic_analysis(test_signal)  # Use basic analysis to avoid sdrconnect
        
        if i == 0:  # Check first result
            print(f"First result validation:")
            print(f"  Peak Frequency: {result.peak_frequency/1e6:.2f} MHz")
            print(f"  Bandwidth: {result.bandwidth/1e3:.1f} kHz")
            print(f"  SNR Estimate: {result.snr_estimate:.1f} dB")
            print(f"  Analysis Method: {result.analysis_method}")
            print()
    
    elapsed = time.time() - start_time
    
    # Get performance statistics
    perf_stats = analyzer.get_performance_stats()
    
    print("Performance Results:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Analyses per second: {num_analyses/elapsed:.1f}")
    print(f"  Average per analysis: {elapsed/num_analyses*1000:.2f}ms")
    print(f"  Average FFT time: {perf_stats['average_computation_time_ms']:.2f}ms")
    print(f"  Min FFT time: {perf_stats['min_computation_time_ms']:.2f}ms")
    print(f"  Max FFT time: {perf_stats['max_computation_time_ms']:.2f}ms")
    print(f"  Threads used: {perf_stats['threads_used']}")
    print(f"  Measurements count: {perf_stats['measurements_count']}")
    
    # Test different FFT sizes
    print("\n--- Testing Different FFT Sizes ---")
    fft_sizes = [512, 1024, 2048]
    
    for fft_size in fft_sizes:
        print(f"\nFFT Size: {fft_size}")
        analyzer_test = EnhancedSignalAnalysis(sample_rate=sample_rate, fft_size=fft_size)
        
        # Generate test signal for this FFT size
        t = np.linspace(0, fft_size / sample_rate, fft_size)
        test_signal = (
            0.7 * np.exp(2j * np.pi * 1.0e6 * t) +
            0.2 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        ).astype(np.complex64)
        
        # Quick performance test
        start = time.time()
        for _ in range(20):
            analyzer_test._basic_analysis(test_signal)
        elapsed = time.time() - start
        
        stats = analyzer_test.get_performance_stats()
        print(f"  20 analyses: {elapsed:.3f}s ({elapsed/20*1000:.2f}ms avg)")
        print(f"  FFT time: {stats['average_computation_time_ms']:.2f}ms avg")
    
    print("\n=== Enhanced Analysis pyFFTW Test Complete ===")


if __name__ == "__main__":
    try:
        test_basic_pyfftw_optimization()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)