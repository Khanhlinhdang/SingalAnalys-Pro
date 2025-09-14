
"""
Enhanced Signal Processor with Channel Coding Integration
Tích hợp channel coding detection và decoding vào signal processor
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks, welch, hilbert
import warnings
warnings.filterwarnings('ignore')

# Import channel coding module
try:
    from channel_coding import (ConvolutionalCoder, TurboCoder, LDPCCoder, 
                               PolarCoder, ReedSolomonCoder, ChannelCodingDetector,
                               generate_hamming_matrix, generate_random_ldpc_matrix)
except ImportError:
    print("Warning: channel_coding module not found. Channel coding features disabled.")
    ConvolutionalCoder = None


class EnhancedSignalProcessor:
    """Enhanced signal processor with comprehensive coding support"""

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate

        # Channel coding components
        if ConvolutionalCoder is not None:
            self.channel_coders = {
                'convolutional': None,
                'turbo': None,
                'ldpc': None,
                'polar': None,
                'reed_solomon': None
            }

            self.coding_detector = ChannelCodingDetector()

            # Initialize default coders
            self._initialize_default_coders()

        # Signal analysis components
        self.spectrum_history = []
        self.max_history = 100

        # Demodulation results cache
        self.last_demod_result = None
        self.last_coding_result = None

    def _initialize_default_coders(self):
        """Initialize default channel coders"""
        try:
            # Standard convolutional coder (WiFi 802.11a)
            self.channel_coders['convolutional'] = ConvolutionalCoder(
                constraint_length=7, code_rate=0.5, polynomials=[0o133, 0o171])

            # Turbo coder
            self.channel_coders['turbo'] = TurboCoder(
                constraint_length=3, interleaver_size=1024)

            # LDPC coder with Hamming matrix as example
            H = generate_hamming_matrix(3)
            self.channel_coders['ldpc'] = LDPCCoder(H)

            # Polar coder
            self.channel_coders['polar'] = PolarCoder(n=16, k=8)

            # Reed-Solomon coder
            self.channel_coders['reed_solomon'] = ReedSolomonCoder(n=15, k=11)

            print("✅ Channel coders initialized successfully")

        except Exception as e:
            print(f"Warning: Could not initialize channel coders: {e}")

    def detect_channel_coding(self, demodulated_bits):
        """Detect channel coding type from demodulated bits"""
        if ConvolutionalCoder is None or len(demodulated_bits) == 0:
            return "none", {}

        try:
            detected_type, scores = self.coding_detector.detect_coding_type(demodulated_bits)
            return detected_type, scores
        except Exception as e:
            print(f"Channel coding detection error: {e}")
            return "unknown", {}

    def decode_channel_coding(self, bits, coding_type, **params):
        """Decode channel coding based on detected or specified type"""
        if ConvolutionalCoder is None:
            return bits, False, "Channel coding not available"

        try:
            if coding_type == 'convolutional':
                return self._decode_convolutional(bits, **params)
            elif coding_type == 'turbo':
                return self._decode_turbo(bits, **params)
            elif coding_type == 'ldpc':
                return self._decode_ldpc(bits, **params)
            elif coding_type == 'polar':
                return self._decode_polar(bits, **params)
            elif coding_type == 'reed_solomon':
                return self._decode_reed_solomon(bits, **params)
            elif coding_type == 'hamming':
                return self._decode_hamming(bits, **params)
            else:
                return bits, False, f"Unsupported coding type: {coding_type}"

        except Exception as e:
            return bits, False, f"Decoding error: {str(e)}"

    def _decode_convolutional(self, bits, **params):
        """Decode convolutional code"""
        coder = self.channel_coders['convolutional']

        # Parameters
        soft_decision = params.get('soft_decision', False)
        constraint_length = params.get('constraint_length', 7)
        code_rate = params.get('code_rate', 0.5)
        polynomials = params.get('polynomials', [0o133, 0o171])

        # Create custom coder if parameters differ
        if (constraint_length != 7 or code_rate != 0.5 or 
            polynomials != [0o133, 0o171]):
            coder = ConvolutionalCoder(constraint_length, code_rate, polynomials)

        try:
            if soft_decision and 'soft_bits' in params:
                decoded_bits = coder.viterbi_decode(params['soft_bits'], is_hard_decision=False)
            else:
                decoded_bits = coder.viterbi_decode(bits, is_hard_decision=True)

            return decoded_bits, True, "Convolutional decoding successful"

        except Exception as e:
            return bits, False, f"Convolutional decoding failed: {str(e)}"

    def _decode_turbo(self, bits, **params):
        """Decode turbo code"""
        coder = self.channel_coders['turbo']

        # Parameters
        iterations = params.get('iterations', 8)
        snr_db = params.get('snr_db', 5)

        try:
            # Turbo decoding requires systematic and parity bits
            # Simple assumption: bits are in format [sys, par1, par2]
            n_bits = len(bits)
            if n_bits % 3 == 0:
                n_info = n_bits // 3
                systematic = bits[:n_info]
                parity1 = bits[n_info:2*n_info]
                parity2 = bits[2*n_info:]

                decoded_bits = coder.log_map_decode(systematic, parity1, parity2, 
                                                  iterations, snr_db)
                return decoded_bits[:n_info], True, "Turbo decoding successful"
            else:
                return bits, False, "Invalid turbo code format"

        except Exception as e:
            return bits, False, f"Turbo decoding failed: {str(e)}"

    def _decode_ldpc(self, bits, **params):
        """Decode LDPC code"""
        # Parameters
        max_iterations = params.get('max_iterations', 50)
        snr_db = params.get('snr_db', 5)
        algorithm = params.get('algorithm', 'sum_product')  # or 'min_sum'

        # Try to use appropriate LDPC matrix
        matrix_type = params.get('matrix_type', 'hamming')

        try:
            if matrix_type == 'hamming' and len(bits) in [7, 15, 31, 63, 127]:
                # Use Hamming matrix
                m = int(np.log2(len(bits) + 1))
                H = generate_hamming_matrix(m)
                coder = LDPCCoder(H)
            elif matrix_type == 'random':
                # Use random LDPC matrix
                n = len(bits)
                k = max(1, int(n * 0.5))  # Rate 1/2
                H = generate_random_ldpc_matrix(n, k)
                coder = LDPCCoder(H)
            else:
                # Use default coder
                coder = self.channel_coders['ldpc']

            # Convert bits to LLR for soft decoding
            if 'soft_bits' in params:
                received_llr = params['soft_bits']
            else:
                # Convert hard bits to LLR (simplified)
                noise_var = 1 / (10**(snr_db/10))
                received_llr = (2 * bits.astype(float) - 1) / noise_var

            if algorithm == 'min_sum':
                decoded_bits, iterations = coder.min_sum_decode(received_llr, max_iterations)
            else:
                decoded_bits, iterations = coder.sum_product_decode(received_llr, max_iterations)

            return decoded_bits, True, f"LDPC decoding successful ({iterations} iterations)"

        except Exception as e:
            return bits, False, f"LDPC decoding failed: {str(e)}"

    def _decode_polar(self, bits, **params):
        """Decode polar code"""
        # Parameters
        k = params.get('k', len(bits) // 2)  # Default rate 1/2
        design_snr_db = params.get('design_snr_db', 0)

        try:
            n = len(bits)

            # Ensure n is power of 2
            if n & (n - 1) != 0:
                # Find nearest power of 2
                n_new = 1
                while n_new < n:
                    n_new <<= 1

                # Pad bits
                padded_bits = np.zeros(n_new)
                padded_bits[:n] = bits
                bits = padded_bits
                n = n_new

            # Create polar coder
            coder = PolarCoder(n, k, design_snr_db)

            # Convert to LLR for SC decoding
            if 'soft_bits' in params:
                received_llr = params['soft_bits']
            else:
                # Convert hard bits to LLR
                received_llr = 2 * bits.astype(float) - 1

            decoded_bits = coder.sc_decode(received_llr)
            return decoded_bits, True, "Polar decoding successful"

        except Exception as e:
            return bits, False, f"Polar decoding failed: {str(e)}"

    def _decode_reed_solomon(self, bits, **params):
        """Decode Reed-Solomon code"""
        # Parameters
        symbol_size = params.get('symbol_size', 8)  # bits per symbol
        n = params.get('n', 255)  # codeword length in symbols
        k = params.get('k', 223)  # message length in symbols

        try:
            # Convert bits to symbols
            if len(bits) % symbol_size != 0:
                # Pad bits to symbol boundary
                pad_length = symbol_size - (len(bits) % symbol_size)
                bits = np.concatenate([bits, np.zeros(pad_length, dtype=int)])

            symbols = []
            for i in range(0, len(bits), symbol_size):
                symbol = 0
                for j in range(symbol_size):
                    if i + j < len(bits):
                        symbol |= bits[i + j] << j
                symbols.append(symbol)

            symbols = np.array(symbols)

            # Adjust to codeword length
            if len(symbols) != n:
                if len(symbols) < n:
                    # Pad with zeros
                    padded_symbols = np.zeros(n, dtype=int)
                    padded_symbols[:len(symbols)] = symbols
                    symbols = padded_symbols
                else:
                    # Truncate
                    symbols = symbols[:n]

            # Create RS coder and decode
            coder = ReedSolomonCoder(n, k)
            decoded_symbols, success = coder.berlekamp_massey_decode(symbols)

            if success:
                # Convert symbols back to bits
                decoded_bits = []
                for symbol in decoded_symbols:
                    for j in range(symbol_size):
                        decoded_bits.append((symbol >> j) & 1)

                return np.array(decoded_bits), True, "Reed-Solomon decoding successful"
            else:
                return bits, False, "Reed-Solomon decoding failed: too many errors"

        except Exception as e:
            return bits, False, f"Reed-Solomon decoding failed: {str(e)}"

    def _decode_hamming(self, bits, **params):
        """Decode Hamming code"""
        n = len(bits)

        # Check if valid Hamming code length
        if n not in [7, 15, 31, 63, 127]:
            return bits, False, f"Invalid Hamming code length: {n}"

        try:
            # Find m such that 2^m - 1 = n
            m = int(np.log2(n + 1))
            H = generate_hamming_matrix(m)

            # Syndrome calculation
            syndrome = (H @ bits) % 2

            if np.all(syndrome == 0):
                # No errors detected
                # Extract information bits (remove parity positions)
                info_positions = []
                for i in range(n):
                    if not (i + 1) & i:  # Not a power of 2 (parity position)
                        continue
                    info_positions.append(i)

                # All positions except powers of 2
                info_positions = [i for i in range(n) if not ((i + 1) & i == 0)]
                decoded_bits = bits[info_positions] if info_positions else bits

                return decoded_bits, True, "Hamming decoding successful (no errors)"
            else:
                # Find error position
                error_pos = 0
                for i in range(m):
                    if syndrome[i]:
                        error_pos += 2**i

                error_pos -= 1  # Convert to 0-based index

                if 0 <= error_pos < n:
                    # Correct single error
                    corrected_bits = bits.copy()
                    corrected_bits[error_pos] = 1 - corrected_bits[error_pos]

                    # Extract information bits
                    info_positions = [i for i in range(n) if not ((i + 1) & i == 0)]
                    decoded_bits = corrected_bits[info_positions] if info_positions else corrected_bits

                    return decoded_bits, True, f"Hamming decoding successful (corrected error at position {error_pos})"
                else:
                    return bits, False, "Hamming decoding failed: invalid error position"

        except Exception as e:
            return bits, False, f"Hamming decoding failed: {str(e)}"

    def comprehensive_signal_analysis(self, iq_data):
        """Comprehensive signal analysis including channel coding"""
        results = {
            'modulation': None,
            'channel_coding': None,
            'snr_estimate': None,
            'decoded_bits': None,
            'coding_success': False,
            'analysis': {}
        }

        try:
            # Basic signal analysis
            results['snr_estimate'] = self._estimate_snr(iq_data)
            results['analysis']['signal_power'] = np.mean(np.abs(iq_data)**2)
            results['analysis']['peak_power'] = np.max(np.abs(iq_data)**2)
            results['analysis']['papr'] = results['analysis']['peak_power'] / results['analysis']['signal_power']

            # Try to demodulate first (simplified - would need actual demodulation)
            # For demo purposes, create some test bits
            demodulated_bits = self._simple_demodulate(iq_data)

            if len(demodulated_bits) > 0:
                # Detect channel coding
                coding_type, coding_scores = self.detect_channel_coding(demodulated_bits)
                results['channel_coding'] = coding_type
                results['analysis']['coding_scores'] = coding_scores

                # Try to decode if coding detected
                if coding_type != 'none' and coding_type != 'unknown':
                    decoded_bits, success, message = self.decode_channel_coding(
                        demodulated_bits, coding_type, 
                        snr_db=results['snr_estimate'])

                    results['decoded_bits'] = decoded_bits
                    results['coding_success'] = success
                    results['analysis']['coding_message'] = message
                else:
                    results['decoded_bits'] = demodulated_bits

        except Exception as e:
            results['analysis']['error'] = str(e)

        return results

    def _estimate_snr(self, iq_data):
        """Estimate SNR from IQ data"""
        try:
            # Simple SNR estimation using signal and noise power
            signal_power = np.mean(np.abs(iq_data)**2)

            # Estimate noise power from high-frequency components
            analytic_signal = hilbert(np.real(iq_data))
            noise_estimate = np.std(np.diff(np.angle(analytic_signal)))
            noise_power = noise_estimate**2

            if noise_power > 0:
                snr_linear = signal_power / noise_power
                snr_db = 10 * np.log10(max(snr_linear, 1e-10))
                return max(-20, min(50, snr_db))  # Clamp between -20 and 50 dB
            else:
                return 30  # Default high SNR

        except:
            return 10  # Default SNR

    def _simple_demodulate(self, iq_data):
        """Simple demodulation for demonstration"""
        try:
            # For demonstration, create bits from signal phases
            # In practice, this would use actual demodulation algorithms
            phases = np.angle(iq_data)

            # Simple threshold detection
            bits = (phases > 0).astype(int)

            # Limit length for processing
            max_bits = min(len(bits), 1000)
            return bits[:max_bits]

        except:
            return np.array([])

    def generate_test_signals(self):
        """Generate test signals with different channel coding"""
        if ConvolutionalCoder is None:
            return {}

        test_signals = {}

        try:
            # Test data
            test_data = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1])

            # Convolutional encoded signal
            conv_coder = ConvolutionalCoder()
            conv_encoded = conv_coder.encode(test_data)
            test_signals['convolutional'] = self._bits_to_iq(conv_encoded)

            # LDPC encoded signal
            H = generate_hamming_matrix(3)
            ldpc_coder = LDPCCoder(H)
            ldpc_encoded = ldpc_coder.encode(test_data[:4])  # 4 info bits -> 7 code bits
            test_signals['ldpc'] = self._bits_to_iq(ldpc_encoded)

            # Polar encoded signal
            polar_coder = PolarCoder(n=16, k=8)
            polar_encoded = polar_coder.encode(test_data[:8])
            test_signals['polar'] = self._bits_to_iq(polar_encoded)

            # Reed-Solomon encoded signal
            rs_coder = ReedSolomonCoder(n=15, k=11)
            rs_message = np.arange(1, 12)  # 11 symbols
            rs_encoded = rs_coder.encode(rs_message)
            rs_bits = []
            for symbol in rs_encoded:
                for i in range(8):  # 8 bits per symbol
                    rs_bits.append((symbol >> i) & 1)
            test_signals['reed_solomon'] = self._bits_to_iq(np.array(rs_bits))

        except Exception as e:
            print(f"Test signal generation error: {e}")

        return test_signals

    def _bits_to_iq(self, bits):
        """Convert bits to IQ signal using BPSK"""
        # BPSK mapping: 0 -> -1, 1 -> +1
        symbols = 2 * bits.astype(float) - 1

        # Add some noise for realism
        noise = 0.1 * np.random.randn(len(symbols))
        symbols += noise

        # Convert to complex IQ (real part only for BPSK)
        iq_signal = symbols + 1j * 0.1 * np.random.randn(len(symbols))

        return iq_signal


# Example usage and testing
def test_enhanced_processor():
    """Test enhanced signal processor with channel coding"""
    print("Testing Enhanced Signal Processor with Channel Coding...")

    processor = EnhancedSignalProcessor()

    if ConvolutionalCoder is None:
        print("Channel coding not available - skipping tests")
        return

    # Generate test signals
    print("\n1. Generating test signals...")
    test_signals = processor.generate_test_signals()

    for coding_type, iq_signal in test_signals.items():
        print(f"\n2. Testing {coding_type} signal:")
        print(f"   Signal length: {len(iq_signal)} samples")

        # Comprehensive analysis
        results = processor.comprehensive_signal_analysis(iq_signal)

        print(f"   Detected coding: {results['channel_coding']}")
        print(f"   SNR estimate: {results['snr_estimate']:.1f} dB")
        print(f"   Decoding success: {results['coding_success']}")

        if 'coding_message' in results['analysis']:
            print(f"   Message: {results['analysis']['coding_message']}")

        if results['decoded_bits'] is not None:
            print(f"   Decoded bits: {len(results['decoded_bits'])} bits")

    print("\n3. Testing manual decoding:")

    # Test manual convolutional decoding
    test_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    decoded, success, message = processor.decode_channel_coding(
        test_bits, 'convolutional', 
        constraint_length=7, code_rate=0.5)

    print(f"   Convolutional: {success}, {message}")
    print(f"   Input length: {len(test_bits)}, Output length: {len(decoded)}")


if __name__ == "__main__":
    test_enhanced_processor()
