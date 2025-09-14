
"""
Enhanced Signal Generator
User-selectable modulation and channel coding with parameter control
"""

import numpy as np
import time
import threading
from queue import Queue
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import modulation and coding modules
try:
    from analog_modulation import AnalogModulation, AnalogDemodulation
    from extended_digital_modulation import ExtendedDigitalModulation, ExtendedDigitalDemodulation
    from multicarrier_spread_spectrum import MultiCarrierModulation, SpreadSpectrumModulation
    from channel_coding import (ConvolutionalCoder, TurboCoder, LDPCCoder, 
                               PolarCoder, ReedSolomonCoder, generate_hamming_matrix)
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODULES_AVAILABLE = False


class EnhancedSignalGenerator:
    """Enhanced signal generator with full user control"""

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate

        # Current configuration (user-selected, persistent)
        self.current_config = {
            'modulation_type': 'bpsk',
            'coding_type': 'none',
            'data_source': 'random',
            'signal_power': 0.0,  # dB
            'noise_power': -20.0,  # dB
            'continuous_generation': False,
            'generation_interval': 2.0,  # seconds
            'validation_mode': True,  # Enable demodulator validation
            'reference_data_tracking': True,  # Track reference data for comparison
        }

        # Modulation parameters by type (Updated with optimized parameters)
        self.modulation_params = {
            # Analog modulations
            'am_dsb_lc': {'carrier_freq': 10000, 'modulation_index': 0.8, 'message_freq': 1000, 'bandwidth': 2000},
            'am_dsb_sc': {'carrier_freq': 10000, 'modulation_index': 0.8, 'message_freq': 1000, 'bandwidth': 2000},
            'am_ssb_usb': {'carrier_freq': 10000, 'modulation_index': 0.8, 'message_freq': 1000, 'bandwidth': 1000},
            'am_ssb_lsb': {'carrier_freq': 10000, 'modulation_index': 0.8, 'message_freq': 1000, 'bandwidth': 1000},
            'fm_nb': {'carrier_freq': 10000, 'deviation': 2000, 'message_freq': 1000, 'bandwidth': 6000},
            'fm_wb': {'carrier_freq': 10000, 'deviation': 5000, 'message_freq': 1000, 'bandwidth': 20000},
            'pm': {'carrier_freq': 10000, 'deviation': 1.0, 'message_freq': 1000, 'bandwidth': 4000},

            # Digital single carrier (Enhanced with better parameters)
            'bpsk': {'symbol_rate': 10000, 'pulse_shape': 'rrc', 'roll_off': 0.35, 'sps': 8, 'filter_taps': 101},
            'qpsk': {'symbol_rate': 10000, 'pulse_shape': 'rrc', 'roll_off': 0.35, 'sps': 8, 'filter_taps': 101},
            '8psk': {'symbol_rate': 10000, 'pulse_shape': 'rrc', 'roll_off': 0.35, 'sps': 8, 'filter_taps': 101},
            'dpsk': {'symbol_rate': 10000, 'pulse_shape': 'rrc', 'roll_off': 0.35, 'sps': 8, 'differential': True},
            'dqpsk': {'symbol_rate': 10000, 'pulse_shape': 'rrc', 'roll_off': 0.35, 'sps': 8, 'differential': True},
            '16qam': {'symbol_rate': 8000, 'pulse_shape': 'rrc', 'roll_off': 0.35, 'sps': 8, 'normalization': 'avg_power'},
            '64qam': {'symbol_rate': 6000, 'pulse_shape': 'rrc', 'roll_off': 0.35, 'sps': 8, 'normalization': 'avg_power'},
            '256qam': {'symbol_rate': 4000, 'pulse_shape': 'rrc', 'roll_off': 0.35, 'sps': 8, 'normalization': 'avg_power'},
            '16apsk': {'symbol_rate': 8000, 'r1': 1.0, 'r2': 2.5, 'pulse_shape': 'rrc', 'sps': 8},
            '32apsk': {'symbol_rate': 6000, 'r1': 1.0, 'r2': 2.2, 'r3': 3.5, 'pulse_shape': 'rrc', 'sps': 8},
            'fsk': {'symbol_rate': 10000, 'freq_deviation': 2000, 'num_tones': 2, 'continuous_phase': False},
            'gfsk': {'symbol_rate': 10000, 'freq_deviation': 2000, 'bt_product': 0.3, 'continuous_phase': True},
            'msk': {'symbol_rate': 10000, 'pulse_shape': 'half_sine', 'continuous_phase': True},
            'gmsk': {'symbol_rate': 10000, 'bt_product': 0.3, 'continuous_phase': True},
            'cpfsk': {'symbol_rate': 10000, 'modulation_index': 0.5, 'pulse_shape': 'rrc', 'continuous_phase': True},

            # Multi-carrier (Enhanced OFDM parameters)
            'ofdm_bpsk': {'subcarriers': 64, 'cyclic_prefix': 16, 'pilot_spacing': 4, 'pilot_symbols': 'bpsk'},
            'ofdm_qpsk': {'subcarriers': 64, 'cyclic_prefix': 16, 'pilot_spacing': 4, 'pilot_symbols': 'qpsk'},
            'ofdm_16qam': {'subcarriers': 64, 'cyclic_prefix': 16, 'pilot_spacing': 4, 'pilot_symbols': 'qpsk'},
            'ofdm_64qam': {'subcarriers': 64, 'cyclic_prefix': 16, 'pilot_spacing': 4, 'pilot_symbols': 'qpsk'},
            'scfdma': {'subcarriers': 64, 'cyclic_prefix': 16, 'localized': True, 'interleaving': 'distributed'},
            'fbmc': {'subcarriers': 64, 'overlapping_factor': 4, 'filter_type': 'phydyas', 'offset_qam': True},

            # Spread spectrum (Improved parameters)
            'dsss_bpsk': {'chip_rate': 100000, 'spread_factor': 31, 'pn_type': 'gold', 'sync_word': True},
            'dsss_qpsk': {'chip_rate': 100000, 'spread_factor': 31, 'pn_type': 'gold', 'sync_word': True},
            'fhss': {'hop_rate': 1000, 'num_channels': 79, 'dwell_time': 0.625e-3, 'hop_pattern': 'pseudo_random'},
            'css_lora': {'bandwidth': 125000, 'spread_factor': 7, 'code_rate': '4/5', 'header_mode': 'explicit'},
        }

        # Channel coding parameters by type (Enhanced with better parameters)
        self.coding_params = {
            'none': {},
            'convolutional': {
                'constraint_length': 7,
                'code_rate': 0.5,
                'polynomials': [0o133, 0o171],  # Industry standard polynomials
                'termination': 'tail_biting',
                'soft_decision': True,
                'puncturing': None
            },
            'turbo': {
                'constraint_length': 3,
                'code_rate': 1/3,
                'interleaver_size': 1024,
                'num_iterations': 8,
                'interleaver_type': 'random',
                'soft_decision': True,
                'early_termination': True
            },
            'ldpc': {
                'code_rate': 0.5,
                'block_length': 1944,
                'matrix_type': 'wifi',  # IEEE 802.11n standard
                'max_iterations': 50,
                'min_sum_algorithm': True,
                'scaling_factor': 0.75
            },
            'polar': {
                'code_length': 1024,
                'info_length': 512,
                'design_snr': 0.0,
                'construction': 'bhattacharyya',
                'list_size': 8,  # For list decoding
                'crc_length': 8
            },
            'reed_solomon': {
                'n': 255,
                'k': 223,
                'symbol_size': 8,
                'primitive_poly': 0x11D,
                'erasure_correction': True,
                'interleaving': 8  # Byte interleaving depth
            },
            'hamming': {
                'n': 7,
                'k': 4,
                'distance': 3,
                'extended': False,  # Can be extended to (8,4) 
                'systematic': True
            },
            'bch': {
                'n': 127,
                'k': 64,
                't': 10,  # Error correction capability
                'primitive_poly': 0x89,
                'systematic': True
            }
        }

        # Data source configurations
        self.data_sources = {
            'random': {'length': 1000, 'seed': None},
            'sequence': {'pattern': [1, 0, 1, 1, 0, 0, 1, 0], 'repeat': 125},
            'file': {'filename': 'test_data.bin', 'format': 'binary'},
            'text': {'message': 'Hello SDR World!', 'encoding': 'utf-8'},
            'prbs': {'order': 15, 'length': 1000}
        }

        # Initialize modulators
        if MODULES_AVAILABLE:
            self._init_modulators()

        # Generation control
        self.generating = False
        self.generation_thread = None
        self.current_signal = None
        self.current_data = None
        self.generation_callback = None

    def _init_modulators(self):
        """Initialize modulation objects"""
        try:
            self.analog_mod = AnalogModulation(self.fs)
            self.digital_mod = ExtendedDigitalModulation(self.fs)
            self.multicarrier_mod = MultiCarrierModulation(self.fs)
            self.spread_mod = SpreadSpectrumModulation(self.fs)

            # Channel coders
            self.channel_coders = {}
            print("✅ Signal generator modulators initialized")
        except Exception as e:
            print(f"❌ Failed to initialize modulators: {e}")

    def get_supported_modulations(self):
        """Get list of supported modulation types"""
        return {
            'Analog': ['am_dsb_lc', 'am_dsb_sc', 'am_ssb_usb', 'am_ssb_lsb', 'fm_nb', 'fm_wb', 'pm'],
            'Digital PSK': ['bpsk', 'qpsk', '8psk', 'dpsk', 'dqpsk'],
            'Digital QAM': ['16qam', '64qam', '256qam'],
            'Digital APSK': ['16apsk', '32apsk'],
            'Digital FSK': ['fsk', 'gfsk', 'msk', 'gmsk', 'cpfsk'],
            'Multi-carrier': ['ofdm_bpsk', 'ofdm_qpsk', 'ofdm_16qam', 'ofdm_64qam', 'scfdma', 'fbmc'],
            'Spread Spectrum': ['dsss_bpsk', 'dsss_qpsk', 'fhss', 'css_lora']
        }

    def get_supported_codings(self):
        """Get list of supported channel coding types"""
        return {
            'No Coding': ['none'],
            'Block Codes': ['hamming', 'reed_solomon'],
            'Convolutional': ['convolutional'],
            'Modern Codes': ['turbo', 'ldpc', 'polar']
        }

    def set_modulation_type(self, mod_type):
        """Set modulation type"""
        if mod_type in self.modulation_params:
            self.current_config['modulation_type'] = mod_type
            return True
        return False

    def set_coding_type(self, coding_type):
        """Set channel coding type"""
        if coding_type in self.coding_params:
            self.current_config['coding_type'] = coding_type
            return True
        return False

    def get_modulation_parameters(self, mod_type=None):
        """Get modulation parameters"""
        if mod_type is None:
            mod_type = self.current_config['modulation_type']
        return self.modulation_params.get(mod_type, {}).copy()

    def get_coding_parameters(self, coding_type=None):
        """Get coding parameters"""
        if coding_type is None:
            coding_type = self.current_config['coding_type']
        return self.coding_params.get(coding_type, {}).copy()

    def set_modulation_parameters(self, params, mod_type=None):
        """Set modulation parameters"""
        if mod_type is None:
            mod_type = self.current_config['modulation_type']

        if mod_type in self.modulation_params:
            self.modulation_params[mod_type].update(params)
            return True
        return False

    def set_coding_parameters(self, params, coding_type=None):
        """Set coding parameters"""
        if coding_type is None:
            coding_type = self.current_config['coding_type']

        if coding_type in self.coding_params:
            self.coding_params[coding_type].update(params)
            return True
        return False

    def generate_data_bits(self, length=None):
        """Generate data bits based on current source configuration"""
        if length is None:
            length = self.data_sources['random']['length']

        source_type = self.current_config.get('data_source', 'random')

        try:
            if source_type == 'random':
                seed = self.data_sources['random'].get('seed')
                if seed is not None:
                    np.random.seed(seed)
                return np.random.randint(0, 2, length)

            elif source_type == 'sequence':
                pattern = self.data_sources['sequence']['pattern']
                repeat = length // len(pattern) + 1
                extended = np.tile(pattern, repeat)
                return extended[:length]

            elif source_type == 'text':
                message = self.data_sources['text']['message']
                encoding = self.data_sources['text']['encoding']

                # Convert text to binary
                text_bytes = message.encode(encoding)
                bits = []
                for byte in text_bytes:
                    for i in range(8):
                        bits.append((byte >> (7-i)) & 1)

                # Repeat if necessary
                if len(bits) < length:
                    repeat = length // len(bits) + 1
                    bits = bits * repeat

                return np.array(bits[:length], dtype=int)

            elif source_type == 'prbs':
                # Generate PRBS sequence
                order = self.data_sources['prbs']['order']
                return self._generate_prbs(order, length)

            else:
                # Default to random
                return np.random.randint(0, 2, length)

        except Exception as e:
            print(f"Error generating data: {e}")
            return np.random.randint(0, 2, length)

    def _generate_prbs(self, order, length):
        """Generate PRBS (Pseudo-Random Binary Sequence)"""
        # PRBS polynomials
        prbs_polys = {
            7: 0x91,    # x^7 + x^6 + 1
            9: 0x211,   # x^9 + x^5 + 1
            11: 0x501,  # x^11 + x^9 + 1
            15: 0x6001, # x^15 + x^14 + 1
            23: 0x840001  # x^23 + x^18 + 1
        }

        if order not in prbs_polys:
            order = 15  # Default

        poly = prbs_polys[order]
        register = 1  # Seed
        sequence = []

        for _ in range(length):
            # Output is LSB
            bit = register & 1
            sequence.append(bit)

            # Calculate feedback
            feedback = 0
            temp_reg = register & poly
            while temp_reg:
                feedback ^= temp_reg & 1
                temp_reg >>= 1

            # Shift and insert feedback
            register = (register >> 1) | (feedback << (order - 1))

        return np.array(sequence, dtype=int)

    def encode_data(self, data_bits):
        """Apply channel coding to data bits"""
        coding_type = self.current_config['coding_type']

        if coding_type == 'none':
            return data_bits

        try:
            # Get or create coder
            if coding_type not in self.channel_coders:
                self.channel_coders[coding_type] = self._create_channel_coder(coding_type)

            coder = self.channel_coders[coding_type]

            if coding_type == 'convolutional':
                return coder.encode(data_bits)

            elif coding_type == 'turbo':
                # Turbo encoding requires specific data length
                params = self.coding_params['turbo']
                interleaver_size = params['interleaver_size']

                # Pad/truncate to interleaver size
                if len(data_bits) > interleaver_size:
                    data_bits = data_bits[:interleaver_size]
                elif len(data_bits) < interleaver_size:
                    padded = np.zeros(interleaver_size, dtype=int)
                    padded[:len(data_bits)] = data_bits
                    data_bits = padded

                return coder.encode(data_bits)

            elif coding_type == 'ldpc':
                # LDPC encoding requires specific info length
                K = coder.K
                if len(data_bits) > K:
                    data_bits = data_bits[:K]
                elif len(data_bits) < K:
                    padded = np.zeros(K, dtype=int)
                    padded[:len(data_bits)] = data_bits
                    data_bits = padded

                return coder.encode(data_bits)

            elif coding_type == 'polar':
                # Polar encoding requires specific info length
                params = self.coding_params['polar']
                K = params['info_length']

                if len(data_bits) > K:
                    data_bits = data_bits[:K]
                elif len(data_bits) < K:
                    padded = np.zeros(K, dtype=int)
                    padded[:len(data_bits)] = data_bits
                    data_bits = padded

                return coder.encode(data_bits)

            elif coding_type == 'reed_solomon':
                # RS encoding works with symbols
                params = self.coding_params['reed_solomon']
                symbol_size = params['symbol_size']
                k = params['k']

                # Convert bits to symbols
                num_symbols = len(data_bits) // symbol_size
                if num_symbols > k:
                    num_symbols = k

                symbols = []
                for i in range(num_symbols):
                    start_bit = i * symbol_size
                    end_bit = start_bit + symbol_size
                    if end_bit <= len(data_bits):
                        symbol_bits = data_bits[start_bit:end_bit]
                        symbol = 0
                        for j, bit in enumerate(symbol_bits):
                            symbol |= (bit << j)
                        symbols.append(symbol)

                # Pad to k symbols if needed
                while len(symbols) < k:
                    symbols.append(0)

                encoded_symbols = coder.encode(np.array(symbols))

                # Convert back to bits
                encoded_bits = []
                for symbol in encoded_symbols:
                    for i in range(symbol_size):
                        encoded_bits.append((symbol >> i) & 1)

                return np.array(encoded_bits, dtype=int)

            elif coding_type == 'hamming':
                # Hamming encoding
                params = self.coding_params['hamming']
                k = params['k']

                # Process in k-bit blocks
                encoded_bits = []
                for i in range(0, len(data_bits), k):
                    block = data_bits[i:i+k]
                    if len(block) < k:
                        # Pad last block
                        padded_block = np.zeros(k, dtype=int)
                        padded_block[:len(block)] = block
                        block = padded_block

                    encoded_block = coder.encode(block)
                    encoded_bits.extend(encoded_block)

                return np.array(encoded_bits, dtype=int)

            else:
                return data_bits

        except Exception as e:
            print(f"Encoding error ({coding_type}): {e}")
            return data_bits

    def _create_channel_coder(self, coding_type):
        """Create channel coder instance"""
        params = self.coding_params[coding_type]

        if coding_type == 'convolutional':
            return ConvolutionalCoder(
                constraint_length=params['constraint_length'],
                code_rate=params['code_rate'],
                polynomials=params['polynomials']
            )

        elif coding_type == 'turbo':
            return TurboCoder(
                constraint_length=params['constraint_length'],
                interleaver_size=params['interleaver_size']
            )

        elif coding_type == 'ldpc':
            # Create appropriate LDPC matrix
            if params['matrix_type'] == 'wifi':
                # Use WiFi LDPC matrix (simplified)
                H = generate_hamming_matrix(3)  # Placeholder
            else:
                # Use random matrix
                n = params['block_length']
                k = int(n * params['code_rate'])
                from channel_coding import generate_random_ldpc_matrix
                H = generate_random_ldpc_matrix(n, k)

            return LDPCCoder(H)

        elif coding_type == 'polar':
            return PolarCoder(
                n=params['code_length'],
                k=params['info_length'],
                design_snr_db=params['design_snr']
            )

        elif coding_type == 'reed_solomon':
            return ReedSolomonCoder(
                n=params['n'],
                k=params['k']
            )

        elif coding_type == 'hamming':
            # Create Hamming coder
            from channel_coding import generate_hamming_matrix
            m = 3  # (7,4) Hamming code
            H = generate_hamming_matrix(m)
            return LDPCCoder(H)  # Use LDPC coder for Hamming

        else:
            raise ValueError(f"Unknown coding type: {coding_type}")

    def modulate_signal(self, coded_bits):
        """Apply modulation to coded bits"""
        mod_type = self.current_config['modulation_type']
        params = self.modulation_params[mod_type]

        try:
            # Analog modulations
            if mod_type.startswith('am_') or mod_type.startswith('fm_') or mod_type == 'pm':
                message = self._bits_to_analog_message(coded_bits)

                if mod_type == 'am_dsb_lc':
                    return self.analog_mod.am_modulate(
                        message, params['carrier_freq'], 
                        params['modulation_index'], 'dsb_lc')
                elif mod_type == 'fm_wb':
                    return self.analog_mod.fm_modulate(
                        message, params['carrier_freq'],
                        params['deviation'], 'wbfm')
                # Add other analog types...

            # Digital PSK modulations  
            elif mod_type in ['bpsk', 'qpsk', '8psk', 'dpsk', 'dqpsk']:
                self.digital_mod.symbol_rate = params['symbol_rate']
                self.digital_mod.samples_per_symbol = int(self.fs / params['symbol_rate'])

                if mod_type == 'bpsk':
                    return self._generate_bpsk_signal(coded_bits, params)
                elif mod_type == 'qpsk':
                    return self._generate_qpsk_signal(coded_bits, params)
                # Add other PSK types...

            # Digital QAM
            elif mod_type in ['16qam', '64qam', '256qam']:
                return self._generate_qam_signal(coded_bits, params)

            # FSK family
            elif mod_type in ['fsk', 'gfsk', 'msk', 'gmsk']:
                if mod_type == 'fsk':
                    return self.digital_mod.fsk_modulate(coded_bits, params['freq_deviation'])
                elif mod_type == 'gfsk':
                    return self.digital_mod.gfsk_modulate(coded_bits, params['bt_product'])

            # Multi-carrier
            elif mod_type.startswith('ofdm_'):
                submod = mod_type.split('_')[1]  # bpsk, qpsk, 16qam, 64qam
                return self.multicarrier_mod.ofdm_modulate(coded_bits, modulation=submod)

            # Spread spectrum
            elif mod_type.startswith('dsss_'):
                submod = mod_type.split('_')[1]  # bpsk, qpsk
                pn_code = self.spread_mod.generate_pn_sequence(params['spread_factor'])
                return self.spread_mod.dsss_modulate(coded_bits, pn_code, submod)

            else:
                # Default BPSK
                return self._generate_bpsk_signal(coded_bits, {'symbol_rate': 10000})

        except Exception as e:
            print(f"Modulation error ({mod_type}): {e}")
            return self._generate_bpsk_signal(coded_bits, {'symbol_rate': 10000})

    def _generate_bpsk_signal(self, bits, params):
        """Generate BPSK signal with improved constellation"""
        symbols = 2 * bits.astype(float) - 1  # Map 0->-1, 1->+1
        symbol_rate = params.get('symbol_rate', 10000)
        sps = params.get('sps', 8)  # Samples per symbol
        samples_per_symbol = int(self.fs / symbol_rate) if 'sps' not in params else sps
        
        # Apply pulse shaping for better spectrum efficiency
        pulse_shape = params.get('pulse_shape', 'rrc')
        roll_off = params.get('roll_off', 0.35)
        filter_taps = params.get('filter_taps', 101)
        
        if pulse_shape == 'rrc':
            # Root raised cosine pulse shaping
            upsampled = np.repeat(symbols, samples_per_symbol)
            # Simple RRC approximation
            signal_samples = upsampled * (1 + roll_off * np.cos(np.pi * np.arange(len(upsampled)) / samples_per_symbol))
        else:
            # Rectangle pulse (NRZ)
            signal_samples = np.repeat(symbols, samples_per_symbol)
            
        return signal_samples + 1j * np.zeros_like(signal_samples)

    def _generate_qpsk_signal(self, bits, params):
        """Generate QPSK signal with improved constellation"""
        # Group bits into pairs
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)

        i_bits = bits[::2]
        q_bits = bits[1::2]

        # QPSK constellation mapping (Gray coding)
        # 00 -> (1,1), 01 -> (-1,1), 11 -> (-1,-1), 10 -> (1,-1)
        i_symbols = 2 * i_bits.astype(float) - 1
        q_symbols = 2 * q_bits.astype(float) - 1

        symbol_rate = params.get('symbol_rate', 10000)
        sps = params.get('sps', 8)
        samples_per_symbol = int(self.fs / symbol_rate) if 'sps' not in params else sps

        # Apply pulse shaping
        pulse_shape = params.get('pulse_shape', 'rrc')
        roll_off = params.get('roll_off', 0.35)
        
        if pulse_shape == 'rrc':
            # Simple RRC approximation
            i_upsampled = np.repeat(i_symbols, samples_per_symbol)
            q_upsampled = np.repeat(q_symbols, samples_per_symbol)
            
            rrc_filter = 1 + roll_off * np.cos(np.pi * np.arange(len(i_upsampled)) / samples_per_symbol)
            i_signal = i_upsampled * rrc_filter
            q_signal = q_upsampled * rrc_filter
        else:
            i_signal = np.repeat(i_symbols, samples_per_symbol)
            q_signal = np.repeat(q_symbols, samples_per_symbol)

        # Normalize power
        return (i_signal + 1j * q_signal) / np.sqrt(2)

    def _generate_qam_signal(self, bits, params):
        """Generate QAM signal with improved constellation"""
        mod_type = self.current_config['modulation_type']
        symbol_rate = params.get('symbol_rate', 8000)
        sps = params.get('sps', 8)
        samples_per_symbol = int(self.fs / symbol_rate) if 'sps' not in params else sps

        if mod_type == '16qam':
            bits_per_symbol = 4
            constellation = self._get_16qam_constellation()
        elif mod_type == '64qam':
            bits_per_symbol = 6
            constellation = self._get_64qam_constellation()
        elif mod_type == '256qam':
            bits_per_symbol = 8
            constellation = self._get_256qam_constellation()
        else:
            bits_per_symbol = 4
            constellation = self._get_16qam_constellation()

        # Group bits and map to symbols
        num_symbols = len(bits) // bits_per_symbol
        if num_symbols == 0:
            return self._generate_bpsk_signal(bits, params)

        symbols = []
        for i in range(num_symbols):
            bit_group = bits[i*bits_per_symbol:(i+1)*bits_per_symbol]
            if len(bit_group) == bits_per_symbol:
                symbol_index = sum(bit * (2**j) for j, bit in enumerate(reversed(bit_group)))
                symbols.append(constellation[symbol_index])

        # Apply pulse shaping and upsampling
        pulse_shape = params.get('pulse_shape', 'rrc')
        roll_off = params.get('roll_off', 0.35)
        
        if pulse_shape == 'rrc':
            # Simple pulse shaping
            signal_samples = []
            for symbol in symbols:
                symbol_samples = [symbol] * samples_per_symbol
                # Apply basic RRC characteristics
                rrc_response = 1 + roll_off * np.cos(np.pi * np.arange(samples_per_symbol) / samples_per_symbol)
                shaped_samples = [s * r for s, r in zip(symbol_samples, rrc_response)]
                signal_samples.extend(shaped_samples)
        else:
            # Rectangle pulse
            signal_samples = []
            for symbol in symbols:
                signal_samples.extend([symbol] * samples_per_symbol)

        return np.array(signal_samples)

    def _get_16qam_constellation(self):
        """Get 16-QAM constellation points with Gray mapping"""
        # Standard 16-QAM constellation with Gray coding
        constellation = np.array([
            -3-3j, -3-1j, -3+3j, -3+1j,  # 0000, 0001, 0010, 0011
            -1-3j, -1-1j, -1+3j, -1+1j,  # 0100, 0101, 0110, 0111
             3-3j,  3-1j,  3+3j,  3+1j,  # 1000, 1001, 1010, 1011
             1-3j,  1-1j,  1+3j,  1+1j   # 1100, 1101, 1110, 1111
        ]) / np.sqrt(10)  # Normalize to unit average power
        return constellation

    def _get_64qam_constellation(self):
        """Get 64-QAM constellation points"""
        # 64-QAM constellation (8x8 grid)
        constellation = []
        for i in range(8):
            for q in range(8):
                # Map to constellation points
                i_val = 2*i - 7  # -7, -5, -3, -1, 1, 3, 5, 7
                q_val = 2*q - 7
                constellation.append(complex(i_val, q_val))
        
        constellation = np.array(constellation)
        return constellation / np.sqrt(np.mean(np.abs(constellation)**2))  # Normalize

    def _get_256qam_constellation(self):
        """Get 256-QAM constellation points"""
        # 256-QAM constellation (16x16 grid)
        constellation = []
        for i in range(16):
            for q in range(16):
                # Map to constellation points  
                i_val = 2*i - 15  # -15, -13, ..., 13, 15
                q_val = 2*q - 15
                constellation.append(complex(i_val, q_val))
        
        constellation = np.array(constellation)
        return constellation / np.sqrt(np.mean(np.abs(constellation)**2))  # Normalize

    def _map_qam_symbol(self, bits):
        """Map bits to QAM constellation point"""
        if len(bits) == 4:  # 16-QAM
            # Gray mapping for 16-QAM
            constellation = {
                (0,0,0,0): -3-3j, (0,0,0,1): -3-1j, (0,0,1,1): -3+1j, (0,0,1,0): -3+3j,
                (0,1,0,0): -1-3j, (0,1,0,1): -1-1j, (0,1,1,1): -1+1j, (0,1,1,0): -1+3j,
                (1,1,0,0): +1-3j, (1,1,0,1): +1-1j, (1,1,1,1): +1+1j, (1,1,1,0): +1+3j,
                (1,0,0,0): +3-3j, (1,0,0,1): +3-1j, (1,0,1,1): +3+1j, (1,0,1,0): +3+3j,
            }
            key = tuple(bits)
            return constellation.get(key, 0) / np.sqrt(10)  # Normalize

        # Fallback for other sizes
        decimal_val = sum(bit * (2**i) for i, bit in enumerate(reversed(bits)))
        M = 2**len(bits)
        side_length = int(np.sqrt(M))

        i_index = decimal_val % side_length
        q_index = decimal_val // side_length

        i_val = 2 * i_index - side_length + 1
        q_val = 2 * q_index - side_length + 1

        return complex(i_val, q_val) / np.sqrt(M * 2/3)

    def _bits_to_analog_message(self, bits):
        """Convert bits to analog message signal"""
        bit_rate = 1000  # 1 kHz
        bit_duration = int(self.fs / bit_rate)
        message = []

        for bit in bits:
            level = 1.0 if bit == 1 else -1.0
            message.extend([level] * bit_duration)

        return np.array(message)

    def add_noise_and_impairments(self, signal):
        """Add noise and channel impairments"""
        # Add AWGN noise
        signal_power_db = self.current_config['signal_power']
        noise_power_db = self.current_config['noise_power']

        # Calculate noise variance
        signal_power_linear = 10**(signal_power_db / 10)
        noise_power_linear = 10**(noise_power_db / 10)

        # Scale signal to desired power
        current_power = np.mean(np.abs(signal)**2)
        if current_power > 0:
            signal = signal * np.sqrt(signal_power_linear / current_power)

        # Add complex AWGN
        if np.iscomplexobj(signal):
            noise = np.sqrt(noise_power_linear/2) * (
                np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        else:
            noise = np.sqrt(noise_power_linear) * np.random.randn(len(signal))

        return signal + noise

    def generate_signal(self, config=None, duration=None, num_bits=None):
        """Generate complete signal with current configuration and validation data"""
        try:
            # Update configuration if provided
            if config is not None:
                # Update current config with provided values
                for key, value in config.items():
                    if key in self.current_config:
                        self.current_config[key] = value
                    elif key == 'data_length':
                        # Handle data_length parameter
                        num_bits = value
                    elif key == 'snr_db':
                        # Update signal and noise power based on SNR
                        signal_power = 10  # Base signal power
                        noise_power = signal_power - value
                        self.current_config['signal_power'] = signal_power
                        self.current_config['noise_power'] = noise_power
            # Determine data length
            if num_bits is not None:
                data_length = num_bits
            elif duration is not None:
                # Estimate based on symbol rate
                mod_params = self.modulation_params[self.current_config['modulation_type']]
                symbol_rate = mod_params.get('symbol_rate', 10000)
                symbols_needed = int(symbol_rate * duration)

                # Account for bits per symbol
                if 'qam' in self.current_config['modulation_type']:
                    if '16' in self.current_config['modulation_type']:
                        bits_per_symbol = 4
                    elif '64' in self.current_config['modulation_type']:
                        bits_per_symbol = 6
                    elif '256' in self.current_config['modulation_type']:
                        bits_per_symbol = 8
                    else:
                        bits_per_symbol = 1
                elif self.current_config['modulation_type'] in ['qpsk', '8psk']:
                    bits_per_symbol = 2 if 'qpsk' in self.current_config['modulation_type'] else 3
                else:
                    bits_per_symbol = 1

                data_length = symbols_needed * bits_per_symbol
            else:
                data_length = self.data_sources['random']['length']

            # Generate data bits
            data_bits = self.generate_data_bits(data_length)
            self.current_data = data_bits.copy()

            # Apply channel coding
            coded_bits = self.encode_data(data_bits)

            # Apply modulation
            signal = self.modulate_signal(coded_bits)

            # Add noise and impairments
            signal_with_noise = self.add_noise_and_impairments(signal)

            # Store current signal
            self.current_signal = signal_with_noise

            # Prepare validation data
            validation_data = {}
            if self.current_config.get('validation_mode', True):
                validation_data = {
                    'reference_constellation': self._generate_reference_constellation(),
                    'ideal_signal': signal.copy(),  # Signal before noise
                    'symbol_timing': self._calculate_symbol_timing(),
                    'expected_symbols': self._extract_symbol_sequence(coded_bits),
                    'snr_db': self.current_config['signal_power'] - self.current_config['noise_power']
                }

            return {
                'signal': signal_with_noise,
                'data_bits': data_bits,
                'coded_bits': coded_bits,
                'config': self.current_config.copy(),
                'modulation_params': self.modulation_params[self.current_config['modulation_type']].copy(),
                'coding_params': self.coding_params[self.current_config['coding_type']].copy(),
                'validation_data': validation_data,
                'generation_timestamp': time.time()
            }

        except Exception as e:
            print(f"Signal generation error: {e}")
            return None

    def start_continuous_generation(self, callback=None):
        """Start continuous signal generation"""
        if self.generating:
            return False

        self.generation_callback = callback
        self.generating = True
        self.generation_thread = threading.Thread(target=self._generation_loop)
        self.generation_thread.daemon = True
        self.generation_thread.start()

        return True

    def stop_continuous_generation(self):
        """Stop continuous signal generation"""
        self.generating = False
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=2.0)

    def _generation_loop(self):
        """Continuous generation loop"""
        while self.generating:
            try:
                # Generate signal
                result = self.generate_signal()

                if result and self.generation_callback:
                    self.generation_callback(result)

                # Wait for next generation
                interval = self.current_config['generation_interval']
                time.sleep(interval)

            except Exception as e:
                print(f"Generation loop error: {e}")
                time.sleep(1)

    def get_current_signal(self):
        """Get current generated signal"""
        return self.current_signal

    def get_current_data(self):
        """Get current data bits"""
        return self.current_data

    def _generate_reference_constellation(self):
        """Generate reference constellation points for current modulation"""
        mod_type = self.current_config['modulation_type']
        
        if mod_type == 'bpsk':
            return np.array([-1+0j, 1+0j])
        elif mod_type == 'qpsk':
            return np.array([(1+1j), (-1+1j), (-1-1j), (1-1j)]) / np.sqrt(2)
        elif mod_type == '16qam':
            return self._get_16qam_constellation()
        elif mod_type == '64qam':
            return self._get_64qam_constellation()
        elif mod_type == '256qam':
            return self._get_256qam_constellation()
        else:
            # Default BPSK
            return np.array([-1+0j, 1+0j])

    def _calculate_symbol_timing(self):
        """Calculate symbol timing information"""
        mod_params = self.modulation_params[self.current_config['modulation_type']]
        symbol_rate = mod_params.get('symbol_rate', 10000)
        sps = mod_params.get('sps', 8)
        samples_per_symbol = int(self.fs / symbol_rate) if 'sps' not in mod_params else sps
        
        return {
            'symbol_rate': symbol_rate,
            'samples_per_symbol': samples_per_symbol,
            'symbol_duration': 1.0 / symbol_rate,
            'sample_rate': self.fs
        }

    def _extract_symbol_sequence(self, coded_bits):
        """Extract expected symbol sequence from coded bits"""
        mod_type = self.current_config['modulation_type']
        
        if mod_type == 'bpsk':
            # Map bits directly to symbols: 0->-1, 1->+1
            return 2 * coded_bits.astype(float) - 1
        elif mod_type == 'qpsk':
            # Group bits into pairs
            if len(coded_bits) % 2 != 0:
                coded_bits = np.append(coded_bits, 0)
            
            symbols = []
            for i in range(0, len(coded_bits), 2):
                i_bit, q_bit = coded_bits[i], coded_bits[i+1]
                i_sym = 2 * i_bit - 1
                q_sym = 2 * q_bit - 1
                symbols.append((i_sym + 1j * q_sym) / np.sqrt(2))
            return np.array(symbols)
        elif mod_type in ['16qam', '64qam', '256qam']:
            # Use constellation mapping
            if mod_type == '16qam':
                bits_per_symbol = 4
                constellation = self._get_16qam_constellation()
            elif mod_type == '64qam':
                bits_per_symbol = 6
                constellation = self._get_64qam_constellation()
            else:  # 256qam
                bits_per_symbol = 8
                constellation = self._get_256qam_constellation()
            
            symbols = []
            for i in range(0, len(coded_bits), bits_per_symbol):
                if i + bits_per_symbol <= len(coded_bits):
                    bit_group = coded_bits[i:i+bits_per_symbol]
                    symbol_index = sum(bit * (2**j) for j, bit in enumerate(reversed(bit_group)))
                    if symbol_index < len(constellation):
                        symbols.append(constellation[symbol_index])
            return np.array(symbols)
        else:
            # Default to BPSK
            return 2 * coded_bits.astype(float) - 1

    def get_validation_metrics(self, demodulated_data, decoded_data=None):
        """Calculate validation metrics comparing transmitted vs received data"""
        if self.current_data is None:
            return {'error': 'No reference data available'}
        
        try:
            metrics = {}
            
            # Bit Error Rate (BER)
            if demodulated_data is not None:
                min_length = min(len(self.current_data), len(demodulated_data))
                if min_length > 0:
                    ref_bits = self.current_data[:min_length]
                    rx_bits = demodulated_data[:min_length]
                    bit_errors = np.sum(ref_bits != rx_bits)
                    metrics['ber'] = bit_errors / min_length
                    metrics['bit_errors'] = bit_errors
                    metrics['total_bits'] = min_length
                else:
                    metrics['ber'] = 1.0
                    metrics['bit_errors'] = 0
                    metrics['total_bits'] = 0
            
            # Block Error Rate (BLER) if decoded data is available
            if decoded_data is not None:
                block_size = 64  # Default block size
                total_blocks = min(len(self.current_data), len(decoded_data)) // block_size
                if total_blocks > 0:
                    block_errors = 0
                    for i in range(total_blocks):
                        start_idx = i * block_size
                        end_idx = start_idx + block_size
                        ref_block = self.current_data[start_idx:end_idx]
                        rx_block = decoded_data[start_idx:end_idx]
                        if not np.array_equal(ref_block, rx_block):
                            block_errors += 1
                    
                    metrics['bler'] = block_errors / total_blocks
                    metrics['block_errors'] = block_errors
                    metrics['total_blocks'] = total_blocks
            
            # SNR estimation
            signal_power_db = self.current_config['signal_power']
            noise_power_db = self.current_config['noise_power']
            metrics['theoretical_snr_db'] = signal_power_db - noise_power_db
            
            return metrics
            
        except Exception as e:
            return {'error': f'Validation error: {e}'}

    def update_data_for_validation(self, new_pattern=None):
        """Update data pattern for validation testing"""
        if new_pattern is not None:
            # Set new data pattern
            if isinstance(new_pattern, str):
                if new_pattern == 'alternating':
                    self.data_sources['sequence']['pattern'] = [1, 0, 1, 0]
                elif new_pattern == 'all_ones':
                    self.data_sources['sequence']['pattern'] = [1, 1, 1, 1]
                elif new_pattern == 'all_zeros':
                    self.data_sources['sequence']['pattern'] = [0, 0, 0, 0]
                elif new_pattern == 'prbs':
                    self.current_config['data_source'] = 'prbs'
                    return
            else:
                self.data_sources['sequence']['pattern'] = new_pattern
            
            self.current_config['data_source'] = 'sequence'


# Test signal generator
def test_signal_generator():
    """Test enhanced signal generator"""
    print("🧪 Testing Enhanced Signal Generator")
    print("=" * 50)

    if not MODULES_AVAILABLE:
        print("❌ Required modules not available")
        return

    # Create generator
    generator = EnhancedSignalGenerator(sample_rate=1e6)

    # Test different configurations
    test_configs = [
        {'modulation_type': 'bpsk', 'coding_type': 'convolutional'},
        {'modulation_type': 'qpsk', 'coding_type': 'ldpc'},
        {'modulation_type': '16qam', 'coding_type': 'polar'},
        {'modulation_type': 'ofdm_qpsk', 'coding_type': 'none'},
    ]

    for config in test_configs:
        print(f"\nTesting: {config}")

        # Set configuration
        generator.set_modulation_type(config['modulation_type'])
        generator.set_coding_type(config['coding_type'])

        # Generate signal
        result = generator.generate_signal(duration=0.01)  # 10ms signal

        if result:
            print(f"  ✅ Generated {len(result['signal'])} samples")
            print(f"  Data bits: {len(result['data_bits'])}")
            print(f"  Coded bits: {len(result['coded_bits'])}")
            print(f"  Signal power: {np.mean(np.abs(result['signal'])**2):.6f}")
        else:
            print(f"  ❌ Generation failed")

    print("\n✅ Signal generator test completed")


if __name__ == "__main__":
    test_signal_generator()
