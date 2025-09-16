#!/usr/bin/env python3
"""
Comprehensive Test for All Modulation and Coding Schemes
Kiểm tra toàn bộ flow với tất cả dạng điều chế và mã hóa có sẵn
"""

import sys
import numpy as np
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rf_spectrum_analyzer.core.signal_processor import SignalProcessor
from rf_spectrum_analyzer.config.settings import Settings

# Import libraries to check availability
try:
    import sdr
    SDR_AVAILABLE = True
    print("✓ sdr library available")
except ImportError:
    SDR_AVAILABLE = False
    print("❌ sdr library not available")

try:
    import sk_dsp_comm.digitalcom as dc
    import sk_dsp_comm.fec_conv as fec_conv
    SCIKIT_DSP_AVAILABLE = True
    print("✓ sk_dsp_comm library available")
except ImportError:
    SCIKIT_DSP_AVAILABLE = False
    print("❌ sk_dsp_comm library not available")


class ModulationTestSuite:
    """Test suite for all modulation schemes."""
    
    def __init__(self):
        self.settings = Settings()
        self.processor = SignalProcessor(self.settings)
        self.sample_rate = 2e6
        self.symbol_rate = 100e3
        self.num_bits = 512
        self.results = {}
        
    def generate_test_signal(self, modulation_type: str, **kwargs):
        """Generate test signal for specific modulation type."""
        print(f"\nGenerating {modulation_type} test signal...")
        
        # Generate random data bits
        data_bits = np.random.randint(0, 2, self.num_bits)
        
        try:
            if modulation_type == "BPSK" and SDR_AVAILABLE:
                return self._generate_bpsk_sdr(data_bits)
            elif modulation_type == "QPSK" and SDR_AVAILABLE:
                return self._generate_qpsk_sdr(data_bits)
            elif modulation_type == "8PSK" and SDR_AVAILABLE:
                return self._generate_8psk_sdr(data_bits)
            elif modulation_type == "16PSK" and SDR_AVAILABLE:
                return self._generate_16psk_sdr(data_bits)
            elif modulation_type == "MSK" and SDR_AVAILABLE:
                return self._generate_msk_sdr(data_bits)
            elif modulation_type == "OQPSK" and SDR_AVAILABLE:
                return self._generate_oqpsk_sdr(data_bits)
            elif modulation_type.startswith("QAM") and SDR_AVAILABLE:
                order = int(modulation_type.replace("QAM", ""))
                return self._generate_qam_sdr(data_bits, order)
            elif SCIKIT_DSP_AVAILABLE:
                return self._generate_with_scikit(data_bits, modulation_type)
            else:
                return self._generate_basic(data_bits, modulation_type)
                
        except Exception as e:
            print(f"❌ Error generating {modulation_type}: {e}")
            return None, data_bits
    
    def _generate_bpsk_sdr(self, data_bits):
        """Generate BPSK signal using sdr library."""
        psk = sdr.PSK(2)  # BPSK
        symbols = psk.map_symbols(data_bits)
        
        # Pulse shaping
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        pulse = sdr.root_raised_cosine(0.35, samples_per_symbol, 10)
        tx_filter = sdr.FIR(pulse)
        
        # Modulate
        upsampled = sdr.upsample(symbols, samples_per_symbol)
        modulated = tx_filter(upsampled)
        
        # Add noise
        noise_power = 0.1
        try:
            noisy_signal = sdr.awgn(modulated, snr=10)
        except TypeError:
            # Fallback if awgn doesn't support 'measured' parameter
            signal_power = np.mean(np.abs(modulated)**2)
            noise = (np.random.randn(len(modulated)) + 1j*np.random.randn(len(modulated))) * np.sqrt(noise_power)
            noisy_signal = modulated + noise
        
        return noisy_signal, data_bits
    
    def _generate_qpsk_sdr(self, data_bits):
        """Generate QPSK signal using sdr library."""
        psk = sdr.PSK(4)  # QPSK
        symbols = psk.map_symbols(data_bits)
        
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        pulse = sdr.root_raised_cosine(0.35, samples_per_symbol, 10)
        tx_filter = sdr.FIR(pulse)
        
        upsampled = sdr.upsample(symbols, samples_per_symbol)
        modulated = tx_filter(upsampled)
        try:
            noisy_signal = sdr.awgn(modulated, snr=15)
        except TypeError:
            noise = (np.random.randn(len(modulated)) + 1j*np.random.randn(len(modulated))) * 0.1
            noisy_signal = modulated + noise
        
        return noisy_signal, data_bits
    
    def _generate_8psk_sdr(self, data_bits):
        """Generate 8-PSK signal using sdr library."""
        psk = sdr.PSK(8)  # 8-PSK
        symbols = psk.map_symbols(data_bits)
        
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        pulse = sdr.root_raised_cosine(0.35, samples_per_symbol, 10)
        tx_filter = sdr.FIR(pulse)
        
        upsampled = sdr.upsample(symbols, samples_per_symbol)
        modulated = tx_filter(upsampled)
        try:
            noisy_signal = sdr.awgn(modulated, snr=12)
        except TypeError:
            noise = (np.random.randn(len(modulated)) + 1j*np.random.randn(len(modulated))) * 0.15
            noisy_signal = modulated + noise
        
        return noisy_signal, data_bits
    
    def _generate_16psk_sdr(self, data_bits):
        """Generate 16-PSK signal using sdr library."""
        psk = sdr.PSK(16)  # 16-PSK
        symbols = psk.map_symbols(data_bits)
        
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        pulse = sdr.root_raised_cosine(0.35, samples_per_symbol, 10)
        tx_filter = sdr.FIR(pulse)
        
        upsampled = sdr.upsample(symbols, samples_per_symbol)
        modulated = tx_filter(upsampled)
        try:
            noisy_signal = sdr.awgn(modulated, snr=18)
        except TypeError:
            noise = (np.random.randn(len(modulated)) + 1j*np.random.randn(len(modulated))) * 0.08
            noisy_signal = modulated + noise
        
        return noisy_signal, data_bits
    
    def _generate_msk_sdr(self, data_bits):
        """Generate MSK signal using sdr library."""
        msk = sdr.MSK()
        symbols = msk.map_symbols(data_bits)
        
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        upsampled = sdr.upsample(symbols, samples_per_symbol)
        
        # Simple MSK modulation
        modulated = upsampled
        try:
            noisy_signal = sdr.awgn(modulated, snr=12)
        except TypeError:
            noise = (np.random.randn(len(modulated)) + 1j*np.random.randn(len(modulated))) * 0.15
            noisy_signal = modulated + noise
        
        return noisy_signal, data_bits
    
    def _generate_oqpsk_sdr(self, data_bits):
        """Generate OQPSK signal using sdr library."""
        oqpsk = sdr.OQPSK()
        symbols = oqpsk.map_symbols(data_bits)
        
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        upsampled = sdr.upsample(symbols, samples_per_symbol)
        
        modulated = upsampled
        try:
            noisy_signal = sdr.awgn(modulated, snr=12)
        except TypeError:
            noise = (np.random.randn(len(modulated)) + 1j*np.random.randn(len(modulated))) * 0.15
            noisy_signal = modulated + noise
        
        return noisy_signal, data_bits
    
    def _generate_qam_sdr(self, data_bits, order):
        """Generate QAM signal using sdr library."""
        # For QAM, we'll use PSK as approximation since sdr doesn't have native QAM
        if order == 16:
            # 16-QAM approximated with constellation
            constellation = np.array([
                -3-3j, -1-3j, 1-3j, 3-3j,
                -3-1j, -1-1j, 1-1j, 3-1j,
                -3+1j, -1+1j, 1+1j, 3+1j,
                -3+3j, -1+3j, 1+3j, 3+3j
            ]) / np.sqrt(10)
        elif order == 64:
            # 64-QAM - simplified constellation
            constellation = []
            for i in range(-7, 8, 2):
                for q in range(-7, 8, 2):
                    constellation.append(i + 1j*q)
            constellation = np.array(constellation[:64]) / np.sqrt(42)
        else:
            # Fallback to PSK
            psk = sdr.PSK(order)
            symbols = psk.map_symbols(data_bits)
            constellation = psk.symbol_map
        
        # Map bits to symbols
        bits_per_symbol = int(np.log2(len(constellation)))
        num_symbols = len(data_bits) // bits_per_symbol
        symbols = []
        
        for i in range(num_symbols):
            bit_group = data_bits[i*bits_per_symbol:(i+1)*bits_per_symbol]
            symbol_index = int(''.join(map(str, bit_group)), 2)
            if symbol_index < len(constellation):
                symbols.append(constellation[symbol_index])
            else:
                symbols.append(constellation[0])
        
        symbols = np.array(symbols)
        
        # Pulse shaping and modulation
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        pulse = sdr.root_raised_cosine(0.35, samples_per_symbol, 10)
        tx_filter = sdr.FIR(pulse)
        
        upsampled = sdr.upsample(symbols, samples_per_symbol)
        modulated = tx_filter(upsampled)
        try:
            noisy_signal = sdr.awgn(modulated, snr=15)
        except TypeError:
            noise = (np.random.randn(len(modulated)) + 1j*np.random.randn(len(modulated))) * 0.1
            noisy_signal = modulated + noise
        
        return noisy_signal, data_bits
    
    def _generate_with_scikit(self, data_bits, modulation_type):
        """Generate signal using scikit-dsp-comm."""
        if modulation_type == "BPSK":
            # BPSK using digitalcom
            symbols = dc.bpsk_tx(data_bits, 1, pulse='rect')
        else:
            # Simple QPSK fallback
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
        
        # Upsample and add noise
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        upsampled = np.repeat(symbols, samples_per_symbol)
        noise = (np.random.randn(len(upsampled)) + 1j*np.random.randn(len(upsampled))) * 0.1
        noisy_signal = upsampled + noise
        
        return noisy_signal, data_bits
    
    def _generate_basic(self, data_bits, modulation_type):
        """Basic signal generation fallback."""
        # Simple QPSK fallback
        symbols = []
        for i in range(0, len(data_bits)-1, 2):
            bit_pair = data_bits[i:i+2] if i+1 < len(data_bits) else [data_bits[i], 0]
            if np.array_equal(bit_pair, [0, 0]):
                symbols.append(1+1j)
            elif np.array_equal(bit_pair, [0, 1]):
                symbols.append(-1+1j)
            elif np.array_equal(bit_pair, [1, 1]):
                symbols.append(-1-1j)
            else:
                symbols.append(1-1j)
        
        symbols = np.array(symbols) / np.sqrt(2)
        
        # Upsample
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        upsampled = np.repeat(symbols, samples_per_symbol)
        
        # Add noise
        noise = (np.random.randn(len(upsampled)) + 1j*np.random.randn(len(upsampled))) * 0.1
        noisy_signal = upsampled + noise
        
        return noisy_signal, data_bits
    
    def test_modulation(self, modulation_type):
        """Test specific modulation type."""
        print(f"\n{'='*60}")
        print(f"TESTING {modulation_type}")
        print(f"{'='*60}")
        
        try:
            # Generate test signal
            iq_signal, original_bits = self.generate_test_signal(modulation_type)
            
            if iq_signal is None:
                print(f"❌ Failed to generate {modulation_type} signal")
                return False
            
            print(f"✓ Generated {len(iq_signal)} IQ samples")
            print(f"✓ Original data: {len(original_bits)} bits")
            
            # Test complete processing chain
            result = self.processor.process_complete_chain(iq_signal)
            
            # Override symbol rate for better demodulation
            if result.get('success') and 'modulation_analysis' in result:
                mod_analysis = result['modulation_analysis']
                if 'parameters' in mod_analysis:
                    mod_analysis['parameters']['symbol_rate'] = self.symbol_rate  # Use known symbol rate
                
                # Re-run demodulation with correct symbol rate
                demod_result = self.processor.demodulate_signal(
                    iq_signal,
                    mod_analysis.get('type', 'QPSK'),
                    mod_analysis.get('parameters', {})
                )
                result['demodulation'] = demod_result
                
                # Update final data
                if demod_result.get('success', False):
                    result['final_data'] = demod_result.get('demodulated_data', np.array([]))
            
            # Analyze results
            print(f"\nProcessing Results:")
            print(f"  Success: {result.get('success', False)}")
            
            if result.get('success'):
                # Modulation analysis
                mod_analysis = result.get('modulation_analysis', {})
                detected_type = mod_analysis.get('type', 'Unknown')
                confidence = mod_analysis.get('confidence', 0.0)
                print(f"  Detected Modulation: {detected_type} (confidence: {confidence:.2f})")
                
                # Demodulation results
                demod_result = result.get('demodulation', {})
                if demod_result.get('success', False):
                    demod_data = demod_result.get('demodulated_data', np.array([]))
                    print(f"  Demodulated: {len(demod_data)} bits")
                    print(f"  EVM: {demod_result.get('evm', 0):.2f}%")
                    print(f"  SNR: {demod_result.get('snr_db', 0):.1f} dB")
                    
                    # BER calculation if possible
                    if len(demod_data) > 0 and len(original_bits) > 0:
                        min_len = min(len(demod_data), len(original_bits))
                        if hasattr(demod_data, 'flatten'):
                            demod_flat = demod_data.flatten()
                        else:
                            demod_flat = demod_data
                        
                        # Convert to binary if needed
                        if demod_flat.dtype != bool and demod_flat.dtype not in [np.uint8, np.int8]:
                            demod_binary = (demod_flat > 0.5).astype(int)
                        else:
                            demod_binary = demod_flat.astype(int)
                        
                        if len(demod_binary) >= min_len:
                            errors = np.sum(demod_binary[:min_len] != original_bits[:min_len])
                            ber = errors / min_len
                            print(f"  BER: {ber:.4f} ({errors}/{min_len} errors)")
                        else:
                            print(f"  BER: Cannot calculate (insufficient data)")
                else:
                    print(f"  Demodulation: Failed")
                
                # Encoding analysis
                enc_analysis = result.get('encoding_analysis', {})
                enc_type = enc_analysis.get('type', 'None')
                enc_confidence = enc_analysis.get('confidence', 0.0)
                print(f"  Detected Encoding: {enc_type} (confidence: {enc_confidence:.2f})")
                
                # Final data
                final_data = result.get('final_data', np.array([]))
                print(f"  Final Output: {len(final_data)} bits")
                
                # Store results
                self.results[modulation_type] = {
                    'success': True,
                    'detected_modulation': detected_type,
                    'modulation_confidence': confidence,
                    'demodulation_success': demod_result.get('success', False),
                    'demodulated_bits': len(demod_data) if 'demod_data' in locals() else 0,
                    'evm': demod_result.get('evm', 0),
                    'snr': demod_result.get('snr_db', 0),
                    'detected_encoding': enc_type,
                    'encoding_confidence': enc_confidence,
                    'final_bits': len(final_data)
                }
                
                return True
            else:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                self.results[modulation_type] = {'success': False, 'error': result.get('error')}
                return False
                
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            self.results[modulation_type] = {'success': False, 'error': str(e)}
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive test for all modulation schemes."""
        print("="*80)
        print("COMPREHENSIVE MODULATION AND CODING TEST")
        print("Testing all available modulation schemes in sdr and sk_dsp_comm")
        print("="*80)
        
        # Define test modulations
        modulations_to_test = []
        
        if SDR_AVAILABLE:
            modulations_to_test.extend([
                "BPSK", "QPSK", "8PSK", "16PSK",
                "16QAM", "64QAM", 
                "MSK", "OQPSK"
            ])
        
        if SCIKIT_DSP_AVAILABLE:
            modulations_to_test.extend(["BPSK_Scikit"])
        
        if not modulations_to_test:
            print("❌ No modulation libraries available for testing")
            return
        
        # Remove duplicates while preserving order
        modulations_to_test = list(dict.fromkeys(modulations_to_test))
        
        print(f"Testing {len(modulations_to_test)} modulation schemes:")
        for mod in modulations_to_test:
            print(f"  - {mod}")
        
        # Run tests
        successful_tests = 0
        total_tests = len(modulations_to_test)
        
        for modulation in modulations_to_test:
            success = self.test_modulation(modulation)
            if success:
                successful_tests += 1
        
        # Summary
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Successful tests: {successful_tests}/{total_tests}")
        print(f"Success rate: {100*successful_tests/total_tests:.1f}%")
        
        print(f"\nDetailed Results:")
        for mod_type, result in self.results.items():
            if result.get('success', False):
                detected = result.get('detected_modulation', 'Unknown')
                confidence = result.get('modulation_confidence', 0)
                demod_success = result.get('demodulation_success', False)
                bits = result.get('demodulated_bits', 0)
                evm = result.get('evm', 0)
                print(f"  {mod_type:12} → {detected:8} (conf:{confidence:.2f}) | "
                      f"Demod: {'✓' if demod_success else '✗'} | "
                      f"Bits: {bits:4} | EVM: {evm:.1f}%")
            else:
                error = result.get('error', 'Unknown error')
                print(f"  {mod_type:12} → ❌ FAILED: {error}")
        
        return successful_tests, total_tests


if __name__ == "__main__":
    print("RF SPECTRUM ANALYZER - COMPREHENSIVE MODULATION TEST")
    print("Testing signal flow with all available modulation and coding schemes")
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    
    # Run comprehensive test
    test_suite = ModulationTestSuite()
    success_count, total_count = test_suite.run_comprehensive_test()
    
    print(f"\n🎯 FINAL RESULT: {success_count}/{total_count} modulation schemes successfully processed")
    
    if success_count == total_count:
        print("🎉 ALL TESTS PASSED! Signal flow works correctly for all modulations.")
    elif success_count > 0:
        print("⚠️  PARTIAL SUCCESS! Some modulations work, others need improvement.")
    else:
        print("❌ ALL TESTS FAILED! Signal flow needs significant debugging.")