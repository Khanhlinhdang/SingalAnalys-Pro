#!/usr/bin/env python3
"""
Quick Test for Core Modulation Schemes
Kiểm tra nhanh các dạng điều chế cơ bản
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rf_spectrum_analyzer.core.signal_processor import SignalProcessor
from rf_spectrum_analyzer.config.settings import Settings

def quick_test():
    """Quick test for basic modulations."""
    print("QUICK MODULATION TEST")
    print("="*50)
    
    # Initialize
    settings = Settings()
    processor = SignalProcessor(settings)
    
    # Test parameters
    sample_rate = 2e6
    symbol_rate = 100e3
    num_bits = 256
    
    results = {}
    
    # Test QPSK (most reliable)
    print("\n1. Testing QPSK...")
    try:
        # Generate QPSK signal
        data_bits = np.random.randint(0, 2, num_bits)
        symbols = []
        for i in range(0, len(data_bits)-1, 2):
            bit_pair = data_bits[i:i+2]
            if np.array_equal(bit_pair, [0, 0]):
                symbols.append(1+1j)
            elif np.array_equal(bit_pair, [0, 1]):
                symbols.append(-1+1j)
            elif np.array_equal(bit_pair, [1, 1]):
                symbols.append(-1-1j)
            else:
                symbols.append(1-1j)
        
        symbols = np.array(symbols) / np.sqrt(2)
        samples_per_symbol = int(sample_rate / symbol_rate)
        upsampled = np.repeat(symbols, samples_per_symbol)
        noise = (np.random.randn(len(upsampled)) + 1j*np.random.randn(len(upsampled))) * 0.1
        iq_signal = upsampled + noise
        
        # Test processing with known symbol rate
        mod_analysis = processor.analyze_modulation(iq_signal)
        mod_analysis['parameters'] = {'symbol_rate': symbol_rate}  # Override
        
        demod_result = processor.demodulate_signal(
            iq_signal, 
            'QPSK',
            {'symbol_rate': symbol_rate}
        )
        
        demod_bits = demod_result.get('demodulated_data', np.array([]))
        
        print(f"✓ Generated {len(iq_signal)} IQ samples")
        print(f"✓ Modulation detected: {mod_analysis.get('type', 'Unknown')}")
        print(f"✓ Demodulated {len(demod_bits)} bits")
        print(f"✓ EVM: {demod_result.get('evm', 0):.1f}%")
        
        results['QPSK'] = {
            'success': True,
            'bits': len(demod_bits),
            'evm': demod_result.get('evm', 0)
        }
        
    except Exception as e:
        print(f"❌ QPSK test failed: {e}")
        results['QPSK'] = {'success': False, 'error': str(e)}
    
    # Test BPSK 
    print("\n2. Testing BPSK...")
    try:
        # Generate BPSK signal
        data_bits = np.random.randint(0, 2, num_bits)
        symbols = np.where(data_bits == 0, -1, 1)
        
        samples_per_symbol = int(sample_rate / symbol_rate)
        upsampled = np.repeat(symbols, samples_per_symbol)
        noise = np.random.randn(len(upsampled)) * 0.1
        iq_signal = upsampled + 1j*0  # Real signal for BPSK
        
        # Add some noise to I/Q
        iq_signal = iq_signal + (np.random.randn(len(iq_signal)) + 1j*np.random.randn(len(iq_signal))) * 0.05
        
        # Test processing
        mod_analysis = processor.analyze_modulation(iq_signal)
        
        demod_result = processor.demodulate_signal(
            iq_signal,
            'PSK',  # Will be detected as PSK/BPSK
            {'symbol_rate': symbol_rate}
        )
        
        demod_bits = demod_result.get('demodulated_data', np.array([]))
        
        print(f"✓ Generated {len(iq_signal)} IQ samples")
        print(f"✓ Modulation detected: {mod_analysis.get('type', 'Unknown')}")
        print(f"✓ Demodulated {len(demod_bits)} bits")
        print(f"✓ EVM: {demod_result.get('evm', 0):.1f}%")
        
        results['BPSK'] = {
            'success': True,
            'bits': len(demod_bits),
            'evm': demod_result.get('evm', 0)
        }
        
    except Exception as e:
        print(f"❌ BPSK test failed: {e}")
        results['BPSK'] = {'success': False, 'error': str(e)}
    
    # Test 8PSK
    print("\n3. Testing 8PSK...")
    try:
        # Generate 8PSK signal
        data_bits = np.random.randint(0, 2, num_bits)
        symbols = []
        
        for i in range(0, len(data_bits)-2, 3):
            bit_triplet = data_bits[i:i+3] if i+2 < len(data_bits) else np.pad(data_bits[i:], (0, 3-(len(data_bits)-i)), 'constant')
            # 8PSK mapping
            symbol_value = bit_triplet[0]*4 + bit_triplet[1]*2 + bit_triplet[2]
            angle = 2*np.pi*symbol_value/8
            symbols.append(np.exp(1j*angle))
        
        symbols = np.array(symbols)
        samples_per_symbol = int(sample_rate / symbol_rate)
        upsampled = np.repeat(symbols, samples_per_symbol)
        noise = (np.random.randn(len(upsampled)) + 1j*np.random.randn(len(upsampled))) * 0.15
        iq_signal = upsampled + noise
        
        # Test processing
        mod_analysis = processor.analyze_modulation(iq_signal)
        
        demod_result = processor.demodulate_signal(
            iq_signal,
            '8PSK',
            {'symbol_rate': symbol_rate}
        )
        
        demod_bits = demod_result.get('demodulated_data', np.array([]))
        
        print(f"✓ Generated {len(iq_signal)} IQ samples")
        print(f"✓ Modulation detected: {mod_analysis.get('type', 'Unknown')}")
        print(f"✓ Demodulated {len(demod_bits)} bits")
        print(f"✓ EVM: {demod_result.get('evm', 0):.1f}%")
        
        results['8PSK'] = {
            'success': True,
            'bits': len(demod_bits),
            'evm': demod_result.get('evm', 0)
        }
        
    except Exception as e:
        print(f"❌ 8PSK test failed: {e}")
        results['8PSK'] = {'success': False, 'error': str(e)}
    
    # Test complete processing chain
    print("\n4. Testing Complete Processing Chain...")
    try:
        # Use QPSK signal
        data_bits = np.random.randint(0, 2, 128)  # Smaller for speed
        symbols = []
        for i in range(0, len(data_bits)-1, 2):
            bit_pair = data_bits[i:i+2]
            if np.array_equal(bit_pair, [0, 0]):
                symbols.append(1+1j)
            elif np.array_equal(bit_pair, [0, 1]):
                symbols.append(-1+1j)
            elif np.array_equal(bit_pair, [1, 1]):
                symbols.append(-1-1j)
            else:
                symbols.append(1-1j)
        
        symbols = np.array(symbols) / np.sqrt(2)
        samples_per_symbol = int(sample_rate / symbol_rate)
        upsampled = np.repeat(symbols, samples_per_symbol)
        noise = (np.random.randn(len(upsampled)) + 1j*np.random.randn(len(upsampled))) * 0.08
        iq_signal = upsampled + noise
        
        # Complete processing chain
        result = processor.process_complete_chain(iq_signal)
        
        if result.get('success'):
            mod_analysis = result.get('modulation_analysis', {})
            demod_result = result.get('demodulation', {})
            enc_analysis = result.get('encoding_analysis', {})
            final_data = result.get('final_data', np.array([]))
            
            print(f"✓ Complete chain succeeded")
            print(f"✓ Detected modulation: {mod_analysis.get('type', 'Unknown')} (conf: {mod_analysis.get('confidence', 0):.2f})")
            print(f"✓ Demodulation: {'Success' if demod_result.get('success') else 'Failed'}")
            print(f"✓ Detected encoding: {enc_analysis.get('type', 'None')} (conf: {enc_analysis.get('confidence', 0):.2f})")
            print(f"✓ Final output: {len(final_data)} bits")
            
            results['Complete_Chain'] = {
                'success': True,
                'modulation': mod_analysis.get('type', 'Unknown'),
                'encoding': enc_analysis.get('type', 'None'),
                'final_bits': len(final_data)
            }
        else:
            print(f"❌ Complete chain failed: {result.get('error', 'Unknown')}")
            results['Complete_Chain'] = {'success': False}
            
    except Exception as e:
        print(f"❌ Complete chain test failed: {e}")
        results['Complete_Chain'] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "="*50)
    print("QUICK TEST SUMMARY")
    print("="*50)
    
    success_count = sum(1 for r in results.values() if r.get('success', False))
    total_count = len(results)
    
    print(f"Successful tests: {success_count}/{total_count}")
    
    for test_name, result in results.items():
        if result.get('success', False):
            if test_name == 'Complete_Chain':
                mod = result.get('modulation', 'Unknown')
                enc = result.get('encoding', 'None')
                bits = result.get('final_bits', 0)
                print(f"✓ {test_name}: {mod} → {enc} → {bits} bits")
            else:
                bits = result.get('bits', 0)
                evm = result.get('evm', 0)
                print(f"✓ {test_name}: {bits} bits, EVM: {evm:.1f}%")
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ {test_name}: {error}")
    
    if success_count == total_count:
        print("\n🎉 ALL QUICK TESTS PASSED!")
    elif success_count > 0:
        print(f"\n⚠️  {success_count}/{total_count} tests passed. Signal flow partially working.")
    else:
        print(f"\n❌ ALL TESTS FAILED. Signal flow needs debugging.")
    
    return success_count, total_count

if __name__ == "__main__":
    quick_test()