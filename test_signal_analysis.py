"""
Test Signal Analysis Functionality
Tests the signal analysis system with synthetic signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer

def test_bpsk_analysis():
    """Test BPSK signal analysis."""
    print("Testing BPSK Signal Analysis...")
    
    # Generate BPSK signal
    sample_rate = 1e6
    duration = 0.1
    symbol_rate = 1000
    
    analyzer = SignalAnalyzer(sample_rate)
    
    # Generate synthetic BPSK data
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Generate random bits
    num_symbols = int(duration * symbol_rate)
    data_bits = np.random.randint(0, 2, num_symbols)
    symbols = 2 * data_bits - 1  # Convert to +1/-1
    
    # Upsample symbols
    samples_per_symbol = int(sample_rate / symbol_rate)
    upsampled = np.repeat(symbols, samples_per_symbol)[:num_samples]
    
    # Add carrier
    carrier_freq = 1000
    iq_data = upsampled * np.exp(1j * 2 * np.pi * carrier_freq * t)
    
    # Add noise
    noise_power = 0.1
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_power / 2)
    iq_data += noise
    
    # Analyze signal
    results = analyzer.analyze_signal_comprehensive(iq_data, 100e6, 10e3)
    
    print(f"Results: {results}")
    print(f"Detected modulation: {results['modulation']['type']}")
    print(f"Confidence: {results['modulation']['confidence']:.2f}")
    print(f"Demodulation success: {results['demodulation']['success']}")
    
    if results['demodulation']['snr']:
        print(f"SNR: {results['demodulation']['snr']:.1f} dB")
    
    return results

def test_qpsk_analysis():
    """Test QPSK signal analysis."""
    print("\nTesting QPSK Signal Analysis...")
    
    sample_rate = 1e6
    duration = 0.1
    symbol_rate = 500
    
    analyzer = SignalAnalyzer(sample_rate)
    
    # Generate synthetic QPSK data
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Generate random symbols (0-3)
    num_symbols = int(duration * symbol_rate)
    data_symbols = np.random.randint(0, 4, num_symbols)
    
    # Map to QPSK constellation
    constellation_map = {
        0: 1 + 1j,
        1: -1 + 1j, 
        2: -1 - 1j,
        3: 1 - 1j
    }
    
    symbols = np.array([constellation_map[s] for s in data_symbols])
    
    # Upsample symbols
    samples_per_symbol = int(sample_rate / symbol_rate)
    upsampled = np.repeat(symbols, samples_per_symbol)[:num_samples]
    
    # Add carrier
    carrier_freq = 2000
    iq_data = upsampled * np.exp(1j * 2 * np.pi * carrier_freq * t)
    
    # Add noise
    noise_power = 0.1
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_power / 2)
    iq_data += noise
    
    # Analyze signal
    results = analyzer.analyze_signal_comprehensive(iq_data, 200e6, 20e3)
    
    print(f"Detected modulation: {results['modulation']['type']}")
    print(f"Confidence: {results['modulation']['confidence']:.2f}")
    print(f"Demodulation success: {results['demodulation']['success']}")
    
    if results['demodulation']['snr']:
        print(f"SNR: {results['demodulation']['snr']:.1f} dB")
    
    return results

def test_fsk_analysis():
    """Test FSK signal analysis."""
    print("\nTesting FSK Signal Analysis...")
    
    sample_rate = 1e6
    duration = 0.1
    symbol_rate = 1200
    
    analyzer = SignalAnalyzer(sample_rate)
    
    # Generate synthetic FSK data
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Generate random bits
    num_symbols = int(duration * symbol_rate)
    data_bits = np.random.randint(0, 2, num_symbols)
    
    # FSK frequencies
    freq_0 = 1000  # Frequency for bit 0
    freq_1 = 2000  # Frequency for bit 1
    
    samples_per_symbol = int(sample_rate / symbol_rate)
    iq_data = np.zeros(num_samples, dtype=complex)
    
    for i, bit in enumerate(data_bits):
        start_idx = i * samples_per_symbol
        end_idx = min(start_idx + samples_per_symbol, num_samples)
        
        if end_idx <= start_idx:
            break
            
        freq = freq_1 if bit else freq_0
        t_symbol = t[start_idx:end_idx]
        iq_data[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t_symbol)
    
    # Add noise
    noise_power = 0.1
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_power / 2)
    iq_data += noise
    
    # Analyze signal
    results = analyzer.analyze_signal_comprehensive(iq_data, 400e6, 50e3)
    
    print(f"Detected modulation: {results['modulation']['type']}")
    print(f"Confidence: {results['modulation']['confidence']:.2f}")
    print(f"Demodulation success: {results['demodulation']['success']}")
    
    return results

def test_coding_analysis():
    """Test coding analysis."""
    print("\nTesting Coding Analysis...")
    
    analyzer = SignalAnalyzer(1e6)
    
    # Test Manchester encoding
    print("Testing Manchester encoding:")
    original_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    manchester_bits = []
    for bit in original_bits:
        if bit == 1:
            manchester_bits.extend([1, 0])
        else:
            manchester_bits.extend([0, 1])
    
    manchester_bits = np.array(manchester_bits)
    coding_result = analyzer.analyze_coding(manchester_bits)
    
    if coding_result:
        print(f"Detected coding: {coding_result.coding_type}")
        print(f"Confidence: {coding_result.confidence:.2f}")
        print(f"Original bits: {original_bits}")
        print(f"Decoded bits: {coding_result.decoded_bits}")
    
    # Test repetition coding
    print("\nTesting Repetition coding:")
    original_bits = np.array([1, 0, 1, 0])
    repetition_bits = []
    for bit in original_bits:
        repetition_bits.extend([bit, bit, bit])  # 3-bit repetition
    
    repetition_bits = np.array(repetition_bits)
    coding_result = analyzer.analyze_coding(repetition_bits)
    
    if coding_result:
        print(f"Detected coding: {coding_result.coding_type}")
        print(f"Confidence: {coding_result.confidence:.2f}")
        print(f"Original bits: {original_bits}")
        print(f"Decoded bits: {coding_result.decoded_bits}")

def plot_constellation(results, title):
    """Plot constellation diagram."""
    if 'constellation_data' in results and results['constellation_data']['points']:
        constellation = np.array(results['constellation_data']['points'])
        
        plt.figure(figsize=(8, 6))
        plt.scatter(np.real(constellation), np.imag(constellation), alpha=0.6, s=20)
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.title(f'{title} - Detected: {results["modulation"]["type"]} (Conf: {results["modulation"]["confidence"]:.2f})')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

def main():
    """Run all tests."""
    print("RF Signal Analysis Test Suite")
    print("=" * 40)
    
    try:
        # Test different modulation types
        bpsk_results = test_bpsk_analysis()
        qpsk_results = test_qpsk_analysis()
        fsk_results = test_fsk_analysis()
        
        # Test coding analysis
        test_coding_analysis()
        
        # Plot constellations if matplotlib is available
        try:
            plot_constellation(bpsk_results, "BPSK Signal")
            plot_constellation(qpsk_results, "QPSK Signal")
        except Exception as e:
            print(f"Could not plot constellations: {e}")
        
        print("\n" + "=" * 40)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()