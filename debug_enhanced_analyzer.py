#!/usr/bin/env python3
"""
Quick debugging script for remaining issues in enhanced SignalAnalyzer.
Focuses on fixing spectrum peak analysis and advanced modulation analysis.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project directory to the path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer

def test_spectrum_peak_detection():
    """Debug spectrum peak detection issues."""
    print("=== Debugging Spectrum Peak Detection ===")
    
    # Create analyzer
    analyzer = SignalAnalyzer(sample_rate=1e6)
    
    # Create test signal with known peaks
    t = np.arange(0, 0.001, 1/1e6)  # 1ms
    
    # Create signal with multiple frequency components
    freq1, freq2, freq3 = 100e3, 200e3, 300e3
    signal = (np.exp(1j * 2 * np.pi * freq1 * t) + 
              0.5 * np.exp(1j * 2 * np.pi * freq2 * t) + 
              0.3 * np.exp(1j * 2 * np.pi * freq3 * t))
    
    # Add minimal noise
    noise = 0.01 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    test_signal = signal + noise
    
    print(f"Test signal length: {len(test_signal)}")
    print(f"Sample rate: {analyzer.sample_rate}")
    print(f"FFT size: {analyzer.fft_size}")
    
    try:
        # Test spectrum analysis method
        peaks_result = analyzer._analyze_spectrum_peaks(test_signal)
        
        print(f"Peak detection result: {peaks_result}")
        
        # Manual spectrum calculation for debugging
        fft_data = np.fft.fftshift(np.fft.fft(test_signal, analyzer.fft_size))
        freqs = np.fft.fftshift(np.fft.fftfreq(analyzer.fft_size, 1/analyzer.sample_rate))
        power_spectrum = 20 * np.log10(np.abs(fft_data) + 1e-12)
        
        print(f"Power spectrum range: {power_spectrum.min():.1f} to {power_spectrum.max():.1f} dB")
        print(f"Frequency range: {freqs.min()/1e3:.1f} to {freqs.max()/1e3:.1f} kHz")
        
        # Try scipy peak finding directly
        from scipy.signal import find_peaks
        noise_floor = np.percentile(power_spectrum, 25)
        threshold = noise_floor + 5  # Lower threshold
        peak_indices, props = find_peaks(power_spectrum, height=threshold, distance=3)
        
        print(f"Noise floor: {noise_floor:.1f} dB")
        print(f"Threshold: {threshold:.1f} dB")
        print(f"Found {len(peak_indices)} peaks with scipy")
        
        if len(peak_indices) > 0:
            peak_freqs = freqs[peak_indices]
            peak_powers = power_spectrum[peak_indices]
            for i, (freq, power) in enumerate(zip(peak_freqs, peak_powers)):
                print(f"  Peak {i+1}: {freq/1e3:.1f} kHz at {power:.1f} dB")
        
        # Plot for visual inspection
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(freqs/1e3, power_spectrum)
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.1f} dB')
        if len(peak_indices) > 0:
            plt.plot(freqs[peak_indices]/1e3, power_spectrum[peak_indices], 'ro', label='Detected peaks')
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Power (dB)')
        plt.title('Power Spectrum')
        plt.legend()
        plt.grid(True)
        
        # Time domain plot
        plt.subplot(1, 2, 2)
        plt.plot(t*1000, np.real(test_signal), label='Real')
        plt.plot(t*1000, np.imag(test_signal), label='Imag')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.title('Test Signal')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('spectrum_debug.png', dpi=150, bbox_inches='tight')
        print("Spectrum debug plot saved as 'spectrum_debug.png'")
        
    except Exception as e:
        print(f"Error in spectrum analysis: {e}")
        import traceback
        traceback.print_exc()

def test_modulation_analysis_advanced():
    """Debug advanced modulation analysis."""
    print("\n=== Debugging Advanced Modulation Analysis ===")
    
    analyzer = SignalAnalyzer(sample_rate=1e6)
    
    # Create clear BPSK signal
    t = np.arange(0, 0.005, 1/1e6)  # 5ms
    bits = np.random.randint(0, 2, 500)
    symbols_per_bit = len(t) // len(bits)
    bpsk_symbols = np.repeat(2 * bits - 1, symbols_per_bit)[:len(t)]
    carrier = np.exp(1j * 2 * np.pi * 100e3 * t)
    bpsk_signal = bpsk_symbols * carrier
    
    print(f"BPSK signal length: {len(bpsk_signal)}")
    
    try:
        # Test basic modulation analysis
        mod_result = analyzer.analyze_modulation(bpsk_signal)
        print(f"Basic analysis detected: {mod_result.modulation_type} (confidence: {mod_result.confidence:.3f})")
        
        # Test advanced modulation analysis if available
        if hasattr(analyzer, '_analyze_modulation_advanced'):
            try:
                advanced_result = analyzer._analyze_modulation_advanced(bpsk_signal)
                print(f"Advanced analysis detected: {advanced_result.modulation_type} (confidence: {advanced_result.confidence:.3f})")
            except Exception as e:
                print(f"Advanced modulation analysis error: {e}")
        
        # Test constellation extraction
        constellation = analyzer._extract_constellation(bpsk_signal, decimation=10)
        print(f"Constellation points extracted: {len(constellation)}")
        
        if len(constellation) > 0:
            # Plot constellation
            plt.figure(figsize=(8, 6))
            plt.scatter(np.real(constellation), np.imag(constellation), alpha=0.6, s=20)
            plt.xlabel('In-phase')
            plt.ylabel('Quadrature')
            plt.title(f'Constellation Diagram - {mod_result.modulation_type}')
            plt.grid(True)
            plt.axis('equal')
            plt.savefig('constellation_debug.png', dpi=150, bbox_inches='tight')
            print("Constellation plot saved as 'constellation_debug.png'")
        
    except Exception as e:
        print(f"Error in modulation analysis: {e}")
        import traceback
        traceback.print_exc()

def test_signal_quality_metrics():
    """Test signal quality metrics calculation."""
    print("\n=== Testing Signal Quality Metrics ===")
    
    analyzer = SignalAnalyzer(sample_rate=1e6)
    
    # Create test signal
    t = np.arange(0, 0.001, 1/1e6)
    clean_signal = np.exp(1j * 2 * np.pi * 100e3 * t)
    
    # Create dummy demodulation result
    from rf_spectrum_analyzer.dsp.signal_analysis import DemodulationResult
    dummy_demod = DemodulationResult(
        success=True,
        symbols=np.random.random(100) + 1j * np.random.random(100),
        bits=np.random.randint(0, 2, 200),
        snr=20.0,
        error_rate=0.01
    )
    
    try:
        if hasattr(analyzer, '_calculate_signal_quality_metrics'):
            metrics = analyzer._calculate_signal_quality_metrics(clean_signal, dummy_demod)
            
            print("Signal Quality Metrics:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
        else:
            print("Signal quality metrics method not available")
            
    except Exception as e:
        print(f"Error in quality metrics: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all debugging tests."""
    print("Enhanced SignalAnalyzer Debug Script")
    print("=" * 40)
    
    # Test spectrum peak detection
    test_spectrum_peak_detection()
    
    # Test modulation analysis
    test_modulation_analysis_advanced()
    
    # Test quality metrics
    test_signal_quality_metrics()
    
    print("\nDebugging complete. Check generated plots for visual analysis.")

if __name__ == "__main__":
    main()