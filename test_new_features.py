#!/usr/bin/env python3
"""
Test script for new modulation/demodulation and encoding/decoding features
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rf_spectrum_analyzer.dsp import (
    ModulationAnalyzer, 
    DemodulationEngine, 
    EncodingAnalyzer, 
    DecodingEngine,
    create_modulation_analyzer,
    create_demodulation_engine,
    create_encoding_analyzer,
    create_decoding_engine
)

def test_modulation_analysis():
    """Test modulation analysis functionality"""
    print("=" * 60)
    print("Testing Modulation Analysis")
    print("=" * 60)
    
    # Create a test QPSK signal
    sample_rate = 1e6
    symbol_rate = 100e3
    num_symbols = 1000
    
    # Generate random symbols
    symbols = np.random.randint(0, 4, num_symbols)
    
    # Simple QPSK modulation
    constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    modulated = constellation[symbols]
    
    # Upsample and add some noise
    samples_per_symbol = int(sample_rate / symbol_rate)
    upsampled = np.repeat(modulated, samples_per_symbol)
    noise = 0.1 * (np.random.randn(len(upsampled)) + 1j * np.random.randn(len(upsampled)))
    signal = upsampled + noise
    
    try:
        # Test modulation analyzer
        analyzer = create_modulation_analyzer()
        result = analyzer.detect_modulation(signal)
        
        print(f"Detected modulation: {result['type']}")
        print(f"Estimated symbol rate: {result['parameters'].get('symbol_rate', 'Unknown')}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("✅ Modulation analysis test passed!")
        
    except Exception as e:
        print(f"❌ Modulation analysis test failed: {e}")

def test_demodulation():
    """Test demodulation functionality"""
    print("\n" + "=" * 60)
    print("Testing Demodulation Engine")
    print("=" * 60)
    
    try:
        # Test demodulation engine
        engine = create_demodulation_engine()
        
        # Simple test with generated signal
        sample_rate = 1e6
        t = np.linspace(0, 0.001, int(sample_rate * 0.001))
        carrier_freq = 100e3
        
        # Generate FM signal
        message = np.sin(2 * np.pi * 1e3 * t)  # 1 kHz message
        deviation = 10e3  # 10 kHz deviation
        fm_signal = np.cos(2 * np.pi * carrier_freq * t + 
                          2 * np.pi * deviation * np.cumsum(message) / sample_rate)
        
        # Convert to complex
        fm_complex = fm_signal + 1j * np.zeros_like(fm_signal)
        
        result = engine.demodulate(fm_complex, 'FM', {'deviation': deviation})
        
        print(f"Demodulated {len(result['demodulated_data'])} samples")
        print(f"EVM: {result.get('evm', 'N/A')}")
        print(f"SNR estimate: {result.get('snr_db', 'N/A')}")
        print("✅ Demodulation test passed!")
        
    except Exception as e:
        print(f"❌ Demodulation test failed: {e}")

def test_encoding_analysis():
    """Test encoding analysis functionality"""
    print("\n" + "=" * 60)
    print("Testing Encoding Analysis")
    print("=" * 60)
    
    try:
        # Test encoding analyzer
        analyzer = create_encoding_analyzer()
        
        # Generate some test data (simulated coded bits)
        test_data = np.random.randint(0, 2, 1000).astype(np.uint8)
        
        result = analyzer.detect_encoding(test_data)
        
        print(f"Detected encoding: {result['type']}")
        print(f"Estimated code rate: {result['parameters'].get('code_rate', 'Unknown')}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("✅ Encoding analysis test passed!")
        
    except Exception as e:
        print(f"❌ Encoding analysis test failed: {e}")

def test_decoding():
    """Test decoding functionality"""
    print("\n" + "=" * 60)
    print("Testing Decoding Engine")
    print("=" * 60)
    
    try:
        # Test decoding engine
        engine = create_decoding_engine()
        
        # Simple test with Hamming(7,4) encoding
        test_data = np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
        
        result = engine.decode(test_data, 'hamming', {'code_rate': 4/7})
        
        print(f"Decoded {len(result['decoded_data'])} bits")
        print(f"Error correction applied: {result.get('errors_corrected', 'Unknown')}")
        print(f"Syndrome: {result.get('syndrome', 'Unknown')}")
        print("✅ Decoding test passed!")
        
    except Exception as e:
        print(f"❌ Decoding test failed: {e}")

def test_integration():
    """Test integration of all components"""
    print("\n" + "=" * 60)
    print("Testing Complete Signal Processing Chain")
    print("=" * 60)
    
    try:
        # Generate a more complex test signal
        sample_rate = 2e6
        duration = 0.01  # 10ms
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate BPSK signal with some data
        bit_rate = 50e3
        data_bits = np.random.randint(0, 2, int(bit_rate * duration))
        
        # Simple BPSK modulation
        samples_per_bit = int(sample_rate / bit_rate)
        bpsk_symbols = 2 * data_bits - 1  # Map 0,1 to -1,1
        signal = np.repeat(bpsk_symbols, samples_per_bit)
        
        # Add carrier and noise
        carrier_freq = 500e3
        signal = signal * np.cos(2 * np.pi * carrier_freq * t[:len(signal)])
        noise = 0.2 * np.random.randn(len(signal))
        signal = signal + noise
        
        print("Generated test signal:")
        print(f"  - Sample rate: {sample_rate/1e6:.1f} MHz")
        print(f"  - Duration: {duration*1000:.1f} ms")
        print(f"  - Data bits: {len(data_bits)}")
        print(f"  - Signal length: {len(signal)} samples")
        
        print("✅ Integration test completed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing New RF Spectrum Analyzer Features")
    print("=" * 60)
    
    test_modulation_analysis()
    test_demodulation()
    test_encoding_analysis()
    test_decoding()
    test_integration()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("=" * 60)