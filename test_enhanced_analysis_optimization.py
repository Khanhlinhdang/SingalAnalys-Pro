#!/usr/bin/env python3
"""
Test script for Enhanced Signal Analysis pyFFTW optimizations
"""

import sys
import os
import time
import numpy as np
import logging

# Add project path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rf_spectrum_analyzer.dsp.enhanced_analysis import EnhancedSignalAnalysis


def test_enhanced_analysis_performance():
    """Test the performance improvements in Enhanced Signal Analysis."""
    print("=== Enhanced Signal Analysis pyFFTW Optimization Test ===")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test different FFT sizes
    fft_sizes = [512, 1024, 2048]
    sample_rate = 5e6
    
    for fft_size in fft_sizes:
        print(f"\n--- Testing FFT Size: {fft_size} ---")
        
        # Create enhanced analyzer
        analyzer = EnhancedSignalAnalysis(sample_rate=sample_rate, fft_size=fft_size)
        
        # Get analysis info
        info = analyzer.get_analysis_info()
        print(f"FFT Method: {info['fft_method']}")
        print(f"pyFFTW Available: {info['pyfftw_available']}")
        print(f"SDRConnect Available: {info['sdrconnect_available']}")
        
        # Generate test signal
        t = np.linspace(0, fft_size / sample_rate, fft_size)
        test_signal = (
            0.7 * np.exp(2j * np.pi * 1.2e6 * t) +    # 1.2 MHz tone
            0.4 * np.exp(2j * np.pi * 0.8e6 * t) +    # 0.8 MHz tone
            0.2 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))  # Noise
        ).astype(np.complex64)
        
        # Warmup runs
        for _ in range(5):
            analyzer.analyze_iq_data(test_signal)
        
        # Performance test
        print("Running performance test...")
        start_time = time.time()
        num_analyses = 50
        
        for i in range(num_analyses):
            result = analyzer.analyze_iq_data(test_signal)
            
            # Validate result
            if i == 0:  # Check first result
                print(f"  Peak Frequency: {result.peak_frequency/1e6:.2f} MHz")
                print(f"  Bandwidth: {result.bandwidth/1e3:.1f} kHz")
                print(f"  SNR Estimate: {result.snr_estimate:.1f} dB")
                print(f"  Analysis Method: {result.analysis_method}")
        
        elapsed = time.time() - start_time
        
        # Get performance statistics
        perf_stats = analyzer.get_performance_stats()
        
        print(f"Performance Results:")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Analyses per second: {num_analyses/elapsed:.1f}")
        print(f"  Average FFT time: {perf_stats['average_computation_time_ms']:.2f}ms")
        print(f"  Min FFT time: {perf_stats['min_computation_time_ms']:.2f}ms")
        print(f"  Max FFT time: {perf_stats['max_computation_time_ms']:.2f}ms")
        print(f"  Threads used: {perf_stats['threads_used']}")
    
    print("\n=== Enhanced Analysis Optimization Test Complete ===")


def test_enhanced_vs_basic():
    """Compare enhanced vs basic analysis performance."""
    print("\n=== Enhanced vs Basic Analysis Comparison ===")
    
    sample_rate = 5e6
    fft_size = 1024
    
    # Create analyzer
    analyzer = EnhancedSignalAnalysis(sample_rate=sample_rate, fft_size=fft_size)
    
    # Generate complex test signal
    t = np.linspace(0, fft_size / sample_rate, fft_size)
    test_signal = (
        0.8 * np.exp(2j * np.pi * 1.5e6 * t) +     # Strong signal
        0.3 * np.exp(2j * np.pi * 0.5e6 * t) +     # Weaker signal
        0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))  # Noise
    ).astype(np.complex64)
    
    # Test enhanced analysis
    print("Testing Enhanced Analysis...")
    start_time = time.time()
    
    for _ in range(20):
        result = analyzer.analyze_iq_data(test_signal)
    
    enhanced_time = time.time() - start_time
    perf_stats = analyzer.get_performance_stats()
    
    print(f"Enhanced Analysis Results:")
    print(f"  Method: {result.analysis_method}")
    print(f"  Total time (20 runs): {enhanced_time:.3f}s")
    print(f"  Average per analysis: {enhanced_time/20*1000:.2f}ms")
    print(f"  FFT computation time: {perf_stats['average_computation_time_ms']:.2f}ms")
    print(f"  Peak frequency: {result.peak_frequency/1e6:.2f} MHz")
    print(f"  SNR estimate: {result.snr_estimate:.1f} dB")
    
    print("\n=== Comparison Test Complete ===")


if __name__ == "__main__":
    try:
        test_enhanced_analysis_performance()
        test_enhanced_vs_basic()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)