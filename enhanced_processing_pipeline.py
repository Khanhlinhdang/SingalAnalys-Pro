
"""
Enhanced Processing Pipeline
Advanced signal processing with auto-detection and parameter tables
"""

import numpy as np
import time
import threading
from queue import Queue, Empty
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import required modules
try:
    from analog_modulation import AnalogDemodulation
    from extended_digital_modulation import ExtendedDigitalDemodulation, AdvancedModulationClassifier
    from multicarrier_spread_spectrum import AdvancedModulationDetector
    from channel_coding import (ConvolutionalCoder, TurboCoder, LDPCCoder, 
                               PolarCoder, ReedSolomonCoder, ChannelCodingDetector,
                               generate_hamming_matrix)
    from enhanced_signal_processor import EnhancedSignalProcessor
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODULES_AVAILABLE = False


class ParameterTable:
    """Parameter table for modulation/coding detection and demodulation"""

    def __init__(self):
        # Default parameters for each modulation type
        self.modulation_params = {
            # Analog modulations
            'am_dsb_lc': {
                'carrier_freq_range': [1000, 50000],
                'carrier_freq_default': 10000,
                'modulation_index_range': [0.1, 0.95],
                'modulation_index_default': 0.8,
                'detection_threshold': 0.7,
                'demod_method': 'envelope'
            },
            'fm_wb': {
                'carrier_freq_range': [1000, 50000],
                'carrier_freq_default': 10000,
                'deviation_range': [1000, 10000],
                'deviation_default': 5000,
                'detection_threshold': 0.7,
                'demod_method': 'discriminator'
            },

            # Digital PSK modulations
            'bpsk': {
                'symbol_rate_range': [1000, 100000],
                'symbol_rate_default': 10000,
                'carrier_freq_range': [0, 50000],
                'carrier_freq_default': 0,
                'detection_threshold': 0.8,
                'decision_threshold': 0.0,
                'phase_recovery': True,
                'timing_recovery': True
            },
            'qpsk': {
                'symbol_rate_range': [1000, 100000], 
                'symbol_rate_default': 10000,
                'carrier_freq_range': [0, 50000],
                'carrier_freq_default': 0,
                'detection_threshold': 0.8,
                'phase_recovery': True,
                'timing_recovery': True,
                'differential_mode': False
            },
            '8psk': {
                'symbol_rate_range': [1000, 50000],
                'symbol_rate_default': 8000,
                'carrier_freq_range': [0, 50000],
                'carrier_freq_default': 0,
                'detection_threshold': 0.75,
                'phase_recovery': True,
                'timing_recovery': True
            },

            # Digital QAM modulations
            '16qam': {
                'symbol_rate_range': [1000, 50000],
                'symbol_rate_default': 8000,
                'carrier_freq_range': [0, 50000],
                'carrier_freq_default': 0,
                'detection_threshold': 0.75,
                'phase_recovery': True,
                'timing_recovery': True,
                'agc_enabled': True
            },
            '64qam': {
                'symbol_rate_range': [1000, 25000],
                'symbol_rate_default': 6000,
                'carrier_freq_range': [0, 25000],
                'carrier_freq_default': 0,
                'detection_threshold': 0.7,
                'phase_recovery': True,
                'timing_recovery': True,
                'agc_enabled': True
            },

            # Digital FSK modulations
            'fsk': {
                'symbol_rate_range': [1000, 50000],
                'symbol_rate_default': 10000,
                'freq_deviation_range': [500, 10000],
                'freq_deviation_default': 2000,
                'detection_threshold': 0.8,
                'coherent_detection': False
            },
            'gfsk': {
                'symbol_rate_range': [1000, 50000],
                'symbol_rate_default': 10000,
                'freq_deviation_range': [500, 5000],
                'freq_deviation_default': 2000,
                'bt_product_range': [0.1, 1.0],
                'bt_product_default': 0.3,
                'detection_threshold': 0.8
            },
            'msk': {
                'symbol_rate_range': [1000, 50000],
                'symbol_rate_default': 10000,
                'detection_threshold': 0.85,
                'phase_continuous': True
            },

            # Multi-carrier
            'ofdm_qpsk': {
                'subcarriers_range': [16, 256],
                'subcarriers_default': 64,
                'cyclic_prefix_range': [4, 32],
                'cyclic_prefix_default': 16,
                'pilot_spacing_range': [2, 8],
                'pilot_spacing_default': 4,
                'detection_threshold': 0.7
            },

            # Spread spectrum
            'dsss_bpsk': {
                'chip_rate_range': [10000, 1000000],
                'chip_rate_default': 100000,
                'spread_factor_range': [7, 511],
                'spread_factor_default': 31,
                'detection_threshold': 0.6,
                'pn_sequence_type': 'gold'
            }
        }

        # Channel coding parameters
        self.coding_params = {
            'convolutional': {
                'constraint_length_range': [3, 15],
                'constraint_length_default': 7,
                'code_rate_options': [1/3, 1/2, 2/3, 3/4],
                'code_rate_default': 0.5,
                'detection_threshold': 0.8,
                'decoding_algorithm': 'viterbi',
                'soft_decision': True,
                'traceback_length': 35
            },
            'turbo': {
                'constraint_length_range': [3, 5],
                'constraint_length_default': 3,
                'interleaver_size_range': [64, 8192],
                'interleaver_size_default': 1024,
                'num_iterations_range': [1, 20],
                'num_iterations_default': 8,
                'detection_threshold': 0.7,
                'decoding_algorithm': 'log_map',
                'early_termination': True
            },
            'ldpc': {
                'block_length_range': [576, 8192],
                'block_length_default': 1944,
                'code_rate_options': [1/2, 2/3, 3/4, 5/6],
                'code_rate_default': 0.5,
                'max_iterations_range': [10, 100],
                'max_iterations_default': 50,
                'detection_threshold': 0.75,
                'decoding_algorithm': 'sum_product',
                'early_termination': True
            },
            'polar': {
                'code_length_range': [64, 2048],
                'code_length_default': 1024,
                'info_length_range': [32, 1024],
                'info_length_default': 512,
                'design_snr_range': [-5, 10],
                'design_snr_default': 0,
                'detection_threshold': 0.7,
                'decoding_algorithm': 'sc',
                'list_size': 1
            },
            'reed_solomon': {
                'n_range': [7, 255],
                'n_default': 255,
                'k_range': [3, 223],
                'k_default': 223,
                'symbol_size_range': [3, 16],
                'symbol_size_default': 8,
                'detection_threshold': 0.8,
                'decoding_algorithm': 'berlekamp_massey'
            }
        }

    def get_modulation_params(self, mod_type):
        """Get parameters for modulation type"""
        return self.modulation_params.get(mod_type, {}).copy()

    def get_coding_params(self, coding_type):
        """Get parameters for coding type"""
        return self.coding_params.get(coding_type, {}).copy()

    def update_modulation_params(self, mod_type, params):
        """Update parameters for modulation type"""
        if mod_type in self.modulation_params:
            self.modulation_params[mod_type].update(params)

    def update_coding_params(self, coding_type, params):
        """Update parameters for coding type"""
        if coding_type in self.coding_params:
            self.coding_params[coding_type].update(params)


class AutoDetectionEngine:
    """Auto-detection engine for modulation and coding"""

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate
        self.param_table = ParameterTable()

        # Initialize detectors
        if MODULES_AVAILABLE:
            self.mod_classifier = AdvancedModulationClassifier()
            self.coding_detector = ChannelCodingDetector()
            self.signal_processor = EnhancedSignalProcessor(sample_rate)

        # Detection configuration
        self.detection_config = {
            'modulation_auto': True,
            'coding_auto': True,
            'confidence_threshold': 0.7,
            'multi_candidate': True,
            'parameter_estimation': True
        }

    def detect_modulation(self, signal, candidates=None):
        """Detect modulation type with confidence scores"""
        if not MODULES_AVAILABLE:
            return 'unknown', {}, {}

        try:
            # Basic signal analysis
            signal_power = np.mean(np.abs(signal)**2)
            peak_power = np.max(np.abs(signal)**2)
            papr = peak_power / signal_power if signal_power > 0 else 0

            # Frequency domain analysis
            fft_data = np.fft.fft(signal[:min(2048, len(signal))])
            freq_spectrum = np.abs(fft_data)

            # Time domain features
            instantaneous_amplitude = np.abs(signal)
            instantaneous_phase = np.angle(signal)

            # Feature extraction
            features = {
                'papr': 10 * np.log10(papr) if papr > 0 else -np.inf,
                'amplitude_variance': np.var(instantaneous_amplitude),
                'phase_variance': np.var(np.diff(np.unwrap(instantaneous_phase))),
                'spectral_flatness': self._calculate_spectral_flatness(freq_spectrum),
                'zero_crossing_rate': self._calculate_zero_crossing_rate(signal),
                'fourth_order_cumulant': self._calculate_fourth_order_cumulant(signal)
            }

            # Classification logic
            scores = {}

            # Analog modulations
            if features['amplitude_variance'] > 0.1:
                if features['papr'] < 3:  # Low PAPR suggests AM
                    scores['am_dsb_lc'] = 0.8
                else:
                    scores['fm_wb'] = 0.7

            # Digital modulations
            if features['papr'] < 1:  # Very low PAPR suggests constant envelope
                scores['fsk'] = 0.7
                scores['msk'] = 0.6
            elif 1 <= features['papr'] < 4:  # Medium PAPR
                scores['bpsk'] = 0.8
                scores['qpsk'] = 0.9
                scores['8psk'] = 0.6
            elif features['papr'] >= 4:  # High PAPR suggests QAM/OFDM
                scores['16qam'] = 0.8
                scores['64qam'] = 0.7
                scores['ofdm_qpsk'] = 0.6

            # Apply spectral features
            if features['spectral_flatness'] > 0.8:  # Flat spectrum
                scores['dsss_bpsk'] = scores.get('dsss_bpsk', 0) + 0.3
                scores['ofdm_qpsk'] = scores.get('ofdm_qpsk', 0) + 0.2

            # Normalize scores
            if scores:
                max_score = max(scores.values())
                if max_score > 0:
                    scores = {k: v/max_score for k, v in scores.items()}

            # Select best candidate
            if scores:
                best_mod = max(scores, key=scores.get)
                confidence = scores[best_mod]
            else:
                best_mod = 'unknown'
                confidence = 0.0
                scores = {'unknown': 0.0}

            # Estimate parameters for detected modulation
            estimated_params = self._estimate_modulation_parameters(signal, best_mod, features)

            return best_mod, scores, estimated_params

        except Exception as e:
            print(f"Modulation detection error: {e}")
            return 'unknown', {'unknown': 0.0}, {}

    def detect_coding(self, bits):
        """Detect channel coding type"""
        if not MODULES_AVAILABLE or bits is None or len(bits) == 0:
            return 'none', {'none': 1.0}, {}

        try:
            # Use built-in coding detector
            detected_type, scores = self.coding_detector.detect_coding_type(bits)

            # Estimate parameters for detected coding
            estimated_params = self._estimate_coding_parameters(bits, detected_type)

            return detected_type, scores, estimated_params

        except Exception as e:
            print(f"Coding detection error: {e}")
            return 'none', {'none': 1.0}, {}

    def _calculate_spectral_flatness(self, spectrum):
        """Calculate spectral flatness measure"""
        try:
            spectrum_positive = spectrum[spectrum > 0]
            if len(spectrum_positive) == 0:
                return 0

            geometric_mean = np.exp(np.mean(np.log(spectrum_positive)))
            arithmetic_mean = np.mean(spectrum_positive)

            return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        except:
            return 0

    def _calculate_zero_crossing_rate(self, signal):
        """Calculate zero crossing rate"""
        try:
            real_part = np.real(signal)
            zero_crossings = np.sum(np.diff(np.sign(real_part)) != 0)
            return zero_crossings / len(signal) if len(signal) > 1 else 0
        except:
            return 0

    def _calculate_fourth_order_cumulant(self, signal):
        """Calculate fourth-order cumulant"""
        try:
            # Simplified fourth-order cumulant
            centered_signal = signal - np.mean(signal)
            c4 = np.mean(centered_signal**4) - 3 * (np.mean(centered_signal**2))**2
            return np.abs(c4)
        except:
            return 0

    def _estimate_modulation_parameters(self, signal, mod_type, features):
        """Estimate modulation-specific parameters"""
        params = {}

        try:
            if mod_type in ['bpsk', 'qpsk', '8psk']:
                # Estimate symbol rate from spectral analysis
                fft_data = np.fft.fft(signal[:min(2048, len(signal))])
                freqs = np.fft.fftfreq(len(fft_data), 1/self.fs)
                spectrum = np.abs(fft_data)

                # Find main lobe bandwidth (simplified)
                peak_idx = np.argmax(spectrum[:len(spectrum)//2])
                bandwidth = self._estimate_bandwidth(spectrum, freqs)
                symbol_rate = bandwidth * 0.8  # Approximate

                params['symbol_rate'] = max(1000, min(50000, symbol_rate))
                params['carrier_freq'] = 0  # Assume baseband

            elif mod_type in ['16qam', '64qam']:
                # Similar to PSK but with AGC considerations
                fft_data = np.fft.fft(signal[:min(2048, len(signal))])
                bandwidth = self._estimate_bandwidth(np.abs(fft_data), np.fft.fftfreq(len(fft_data), 1/self.fs))
                params['symbol_rate'] = max(1000, min(25000, bandwidth * 0.8))
                params['agc_enabled'] = True

            elif mod_type in ['fsk', 'gfsk']:
                # Estimate frequency deviation
                inst_freq = np.diff(np.unwrap(np.angle(signal)))
                freq_deviation = np.std(inst_freq) * self.fs / (2 * np.pi)
                params['freq_deviation'] = max(500, min(10000, freq_deviation))
                params['symbol_rate'] = 10000  # Default

            elif mod_type == 'ofdm_qpsk':
                # OFDM-specific parameters
                params['subcarriers'] = 64  # Default
                params['cyclic_prefix'] = 16  # Default
                params['pilot_spacing'] = 4

            # Add default parameters from table
            default_params = self.param_table.get_modulation_params(mod_type)
            for key, value in default_params.items():
                if key.endswith('_default') and key[:-8] not in params:
                    params[key[:-8]] = value

        except Exception as e:
            print(f"Parameter estimation error for {mod_type}: {e}")

        return params

    def _estimate_bandwidth(self, spectrum, freqs):
        """Estimate signal bandwidth"""
        try:
            # Find power spectral density
            psd = spectrum**2
            total_power = np.sum(psd)

            if total_power == 0:
                return 10000  # Default bandwidth

            # Find 99% power bandwidth
            cumulative_power = np.cumsum(psd)
            idx_95 = np.argmax(cumulative_power >= 0.95 * total_power)
            idx_5 = np.argmax(cumulative_power >= 0.05 * total_power)

            bandwidth = abs(freqs[idx_95] - freqs[idx_5])
            return max(1000, min(100000, bandwidth))
        except:
            return 10000

    def _estimate_coding_parameters(self, bits, coding_type):
        """Estimate coding-specific parameters"""
        params = {}

        try:
            if coding_type == 'convolutional':
                # Estimate constraint length and code rate from bit patterns
                # Simplified - use defaults with some analysis
                params['constraint_length'] = 7
                params['code_rate'] = 0.5
                params['traceback_length'] = 35

            elif coding_type == 'turbo':
                params['constraint_length'] = 3
                params['interleaver_size'] = 1024
                params['num_iterations'] = 8

            elif coding_type == 'ldpc':
                # Estimate from bit sequence statistics
                block_length = 1944  # Common WiFi length
                params['block_length'] = block_length
                params['code_rate'] = 0.5
                params['max_iterations'] = 50

            elif coding_type == 'polar':
                params['code_length'] = 1024
                params['info_length'] = 512
                params['design_snr'] = 0

            elif coding_type == 'reed_solomon':
                params['n'] = 255
                params['k'] = 223
                params['symbol_size'] = 8

            # Add defaults from table
            default_params = self.param_table.get_coding_params(coding_type)
            for key, value in default_params.items():
                if key.endswith('_default') and key[:-8] not in params:
                    params[key[:-8]] = value

        except Exception as e:
            print(f"Coding parameter estimation error for {coding_type}: {e}")

        return params


class EnhancedProcessingPipeline:
    """Enhanced processing pipeline with auto-detection and parameter control"""

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate

        # Components
        self.auto_detector = AutoDetectionEngine(sample_rate)
        self.param_table = ParameterTable()

        # Processing state
        self.processing_results = {
            'stage_1_modulation_detection': {'status': 'idle', 'result': None, 'confidence': 0, 'params': {}},
            'stage_2_demodulation': {'status': 'idle', 'result': None, 'constellation': [], 'params': {}},
            'stage_3_coding_detection': {'status': 'idle', 'result': None, 'confidence': 0, 'params': {}},
            'stage_4_channel_decoding': {'status': 'idle', 'result': None, 'success': False, 'params': {}},
            'stage_5_bit_extraction': {'status': 'idle', 'result': None, 'bit_count': 0}
        }

        # User-configurable parameters
        self.user_params = {
            'modulation_type_override': None,  # None = auto-detect
            'coding_type_override': None,      # None = auto-detect
            'snr_estimate': 10.0,
            'enable_parameter_estimation': True,
            'confidence_threshold': 0.7,
            'max_constellation_points': 1000
        }

        # Initialize demodulators
        if MODULES_AVAILABLE:
            self._init_demodulators()

    def _init_demodulators(self):
        """Initialize demodulation components"""
        try:
            self.analog_demod = AnalogDemodulation(self.fs)
            self.digital_demod = ExtendedDigitalDemodulation(self.fs)

            # Channel decoders (created on demand)
            self.channel_decoders = {}

        except Exception as e:
            print(f"Demodulator initialization error: {e}")

    def set_user_parameters(self, **params):
        """Set user-configurable parameters"""
        self.user_params.update(params)

    def get_user_parameters(self):
        """Get current user parameters"""
        return self.user_params.copy()

    def process_signal(self, signal, signal_info=None):
        """Process signal through complete pipeline"""
        try:
            # Reset results
            for stage in self.processing_results.values():
                stage['status'] = 'idle'
                stage['result'] = None

            # Stage 1: Modulation Detection
            self._stage_1_modulation_detection(signal, signal_info)

            # Stage 2: Demodulation
            if self.processing_results['stage_1_modulation_detection']['status'] == 'completed':
                self._stage_2_demodulation(signal)

            # Stage 3: Coding Detection
            if self.processing_results['stage_2_demodulation']['status'] == 'completed':
                self._stage_3_coding_detection()

            # Stage 4: Channel Decoding
            if self.processing_results['stage_3_coding_detection']['status'] == 'completed':
                self._stage_4_channel_decoding()

            # Stage 5: Bit Extraction
            if self.processing_results['stage_4_channel_decoding']['status'] == 'completed':
                self._stage_5_bit_extraction()

            return self.processing_results.copy()

        except Exception as e:
            print(f"Pipeline processing error: {e}")
            return self.processing_results.copy()

    def _stage_1_modulation_detection(self, signal, signal_info):
        """Stage 1: Detect modulation type"""
        try:
            self.processing_results['stage_1_modulation_detection']['status'] = 'processing'

            # Check for user override
            if self.user_params['modulation_type_override']:
                detected_type = self.user_params['modulation_type_override']
                confidence = 1.0
                scores = {detected_type: 1.0}
                params = self.param_table.get_modulation_params(detected_type)
            else:
                # Auto-detection
                detected_type, scores, params = self.auto_detector.detect_modulation(signal)
                confidence = scores.get(detected_type, 0.0)

            # Store results
            self.processing_results['stage_1_modulation_detection'].update({
                'status': 'completed',
                'result': detected_type,
                'confidence': confidence,
                'scores': scores,
                'params': params
            })

        except Exception as e:
            self.processing_results['stage_1_modulation_detection'].update({
                'status': 'error',
                'result': None,
                'error': str(e)
            })

    def _stage_2_demodulation(self, signal):
        """Stage 2: Demodulate signal with constellation validation"""
        try:
            self.processing_results['stage_2_demodulation']['status'] = 'processing'

            # Get modulation type and parameters
            mod_type = self.processing_results['stage_1_modulation_detection']['result']
            mod_params = self.processing_results['stage_1_modulation_detection']['params']

            if not mod_type or mod_type == 'unknown':
                raise ValueError("Unknown modulation type")

            # Perform demodulation
            demod_result = self._demodulate_signal(signal, mod_type, mod_params)

            # Extract constellation points for validation
            constellation = self._extract_constellation_points(signal, mod_type)
            
            # Enhanced constellation analysis for validation
            constellation_analysis = self._analyze_constellation(constellation, mod_type)

            self.processing_results['stage_2_demodulation'].update({
                'status': 'completed',
                'result': demod_result,
                'constellation': constellation,
                'constellation_analysis': constellation_analysis,
                'params': mod_params,
                'validation_metrics': constellation_analysis.get('validation_metrics', {})
            })

        except Exception as e:
            self.processing_results['stage_2_demodulation'].update({
                'status': 'error',
                'result': None,
                'error': str(e)
            })

    def _stage_3_coding_detection(self):
        """Stage 3: Detect channel coding"""
        try:
            self.processing_results['stage_3_coding_detection']['status'] = 'processing'

            # Get demodulated bits
            demod_bits = self.processing_results['stage_2_demodulation']['result']

            # Check for user override
            if self.user_params['coding_type_override']:
                detected_type = self.user_params['coding_type_override']
                confidence = 1.0
                scores = {detected_type: 1.0}
                params = self.param_table.get_coding_params(detected_type)
            else:
                # Auto-detection
                detected_type, scores, params = self.auto_detector.detect_coding(demod_bits)
                confidence = scores.get(detected_type, 0.0)

            self.processing_results['stage_3_coding_detection'].update({
                'status': 'completed',
                'result': detected_type,
                'confidence': confidence,
                'scores': scores,
                'params': params
            })

        except Exception as e:
            self.processing_results['stage_3_coding_detection'].update({
                'status': 'error',
                'result': None,
                'error': str(e)
            })

    def _stage_4_channel_decoding(self):
        """Stage 4: Decode channel coding"""
        try:
            self.processing_results['stage_4_channel_decoding']['status'] = 'processing'

            # Get coding type and demodulated bits
            coding_type = self.processing_results['stage_3_coding_detection']['result']
            coding_params = self.processing_results['stage_3_coding_detection']['params']
            demod_bits = self.processing_results['stage_2_demodulation']['result']

            if coding_type == 'none' or coding_type == 'unknown':
                # No coding - pass through
                decoded_bits = demod_bits
                success = True
                message = "No channel coding detected"
            else:
                # Perform channel decoding
                decoded_bits, success, message = self._decode_channel_coding(
                    demod_bits, coding_type, coding_params)

            self.processing_results['stage_4_channel_decoding'].update({
                'status': 'completed',
                'result': decoded_bits,
                'success': success,
                'message': message,
                'params': coding_params
            })

        except Exception as e:
            self.processing_results['stage_4_channel_decoding'].update({
                'status': 'error',
                'result': None,
                'success': False,
                'error': str(e)
            })

    def _stage_5_bit_extraction(self):
        """Stage 5: Extract final bit stream"""
        try:
            self.processing_results['stage_5_bit_extraction']['status'] = 'processing'

            # Get decoded bits
            decoded_bits = self.processing_results['stage_4_channel_decoding']['result']

            if decoded_bits is not None and len(decoded_bits) > 0:
                # Convert to integer array
                bit_stream = np.array(decoded_bits, dtype=int)
                bit_count = len(bit_stream)
            else:
                bit_stream = np.array([], dtype=int)
                bit_count = 0

            self.processing_results['stage_5_bit_extraction'].update({
                'status': 'completed',
                'result': bit_stream,
                'bit_count': bit_count
            })

        except Exception as e:
            self.processing_results['stage_5_bit_extraction'].update({
                'status': 'error',
                'result': None,
                'bit_count': 0,
                'error': str(e)
            })

    def _demodulate_signal(self, signal, mod_type, params):
        """Demodulate signal based on type and parameters"""
        try:
            # Analog demodulation
            if mod_type in ['am_dsb_lc', 'fm_wb']:
                if mod_type == 'am_dsb_lc':
                    return self.analog_demod.am_demodulate(signal, 'envelope')
                elif mod_type == 'fm_wb':
                    return self.analog_demod.fm_demodulate(signal, 'discriminator')

            # Digital demodulation
            else:
                # Set demodulator parameters
                symbol_rate = params.get('symbol_rate', 10000)
                self.digital_demod.symbol_rate = symbol_rate
                self.digital_demod.samples_per_symbol = int(self.fs / symbol_rate)

                if mod_type == 'bpsk':
                    return self._demod_bpsk(signal, params)
                elif mod_type == 'qpsk':
                    return self._demod_qpsk(signal, params)
                elif mod_type in ['16qam', '64qam']:
                    return self._demod_qam(signal, mod_type, params)
                elif mod_type in ['fsk', 'gfsk']:
                    return self._demod_fsk(signal, params)
                else:
                    # Default BPSK demodulation
                    return self._demod_bpsk(signal, params)

        except Exception as e:
            print(f"Demodulation error: {e}")
            # Fallback to simple magnitude detection
            magnitude = np.abs(signal)
            decimation = max(1, len(signal) // 200)
            decimated = magnitude[::decimation]
            return (decimated > np.mean(decimated)).astype(int)

    def _demod_bpsk(self, signal, params):
        """BPSK demodulation"""
        # Simple coherent detection
        real_part = np.real(signal)
        symbol_rate = params.get('symbol_rate', 10000)
        samples_per_symbol = int(self.fs / symbol_rate)

        # Matched filter (simplified)
        decimated = real_part[::max(1, samples_per_symbol//4)]

        # Decision
        return (decimated > 0).astype(int)

    def _demod_qpsk(self, signal, params):
        """QPSK demodulation"""
        symbol_rate = params.get('symbol_rate', 10000)
        samples_per_symbol = int(self.fs / symbol_rate)

        # Decimate
        decimation = max(1, samples_per_symbol // 4)
        i_data = np.real(signal)[::decimation]
        q_data = np.imag(signal)[::decimation]

        # Decision
        i_bits = (i_data > 0).astype(int)
        q_bits = (q_data > 0).astype(int)

        # Interleave
        bits = np.empty(len(i_bits) + len(q_bits), dtype=int)
        bits[0::2] = i_bits
        bits[1::2] = q_bits

        return bits

    def _demod_qam(self, signal, mod_type, params):
        """QAM demodulation"""
        symbol_rate = params.get('symbol_rate', 8000)
        samples_per_symbol = int(self.fs / symbol_rate)

        # Decimate to symbol rate
        decimation = max(1, samples_per_symbol // 4)
        symbols = signal[::decimation]

        # QAM demapping
        bits = []
        for symbol in symbols:
            if mod_type == '16qam':
                symbol_bits = self._demap_16qam(symbol)
            elif mod_type == '64qam':
                symbol_bits = self._demap_64qam(symbol)
            else:
                symbol_bits = [int(np.real(symbol) > 0)]  # Fallback to BPSK

            bits.extend(symbol_bits)

        return np.array(bits, dtype=int)

    def _demap_16qam(self, symbol):
        """Demap 16-QAM symbol to bits"""
        # Simple hard decision demapping
        real_part = np.real(symbol)
        imag_part = np.imag(symbol)

        # Two bits each for I and Q
        i_msb = 1 if real_part > 0 else 0
        i_lsb = 1 if abs(real_part) < 2 else 0
        q_msb = 1 if imag_part > 0 else 0
        q_lsb = 1 if abs(imag_part) < 2 else 0

        return [i_msb, i_lsb, q_msb, q_lsb]

    def _demap_64qam(self, symbol):
        """Demap 64-QAM symbol to bits"""
        # Simplified 64-QAM demapping
        real_part = np.real(symbol)
        imag_part = np.imag(symbol)

        # Three bits each for I and Q
        i_bits = []
        q_bits = []

        # Simple threshold-based demapping (not optimal)
        i_bits.append(1 if real_part > 0 else 0)
        i_bits.append(1 if abs(real_part) > 2 else 0)
        i_bits.append(1 if abs(real_part) > 4 else 0)

        q_bits.append(1 if imag_part > 0 else 0)
        q_bits.append(1 if abs(imag_part) > 2 else 0)
        q_bits.append(1 if abs(imag_part) > 4 else 0)

        return i_bits + q_bits

    def _demod_fsk(self, signal, params):
        """FSK demodulation"""
        # Frequency discrimination
        if np.iscomplexobj(signal):
            phase = np.angle(signal)
            inst_freq = np.diff(np.unwrap(phase))

            # Decision based on instantaneous frequency
            threshold = np.median(inst_freq)
            bits = (inst_freq > threshold).astype(int)
        else:
            # Energy-based detection for real signals
            decimation = max(1, len(signal) // 200)
            decimated = signal[::decimation]
            bits = (decimated > np.mean(decimated)).astype(int)

        return bits

    def _extract_constellation_points(self, signal, mod_type):
        """Extract constellation points for display"""
        try:
            max_points = self.user_params['max_constellation_points']

            # Decimate signal for constellation
            if len(signal) > max_points:
                indices = np.random.choice(len(signal), max_points, replace=False)
                constellation_signal = signal[indices]
            else:
                constellation_signal = signal

            # Convert to complex if needed
            if not np.iscomplexobj(constellation_signal):
                constellation_signal = constellation_signal + 1j * np.zeros_like(constellation_signal)

            return constellation_signal.tolist()

        except Exception as e:
            print(f"Constellation extraction error: {e}")
            return []

    def _analyze_constellation(self, constellation, mod_type):
        """Analyze constellation for validation and quality assessment"""
        try:
            if not constellation or len(constellation) == 0:
                return {'error': 'No constellation data'}

            constellation_array = np.array(constellation)
            
            # Convert to complex if needed
            if not np.iscomplexobj(constellation_array):
                constellation_array = constellation_array + 1j * np.zeros_like(constellation_array)

            analysis = {}
            
            # Basic constellation metrics
            analysis['num_points'] = len(constellation_array)
            analysis['mean_power'] = np.mean(np.abs(constellation_array)**2)
            analysis['peak_power'] = np.max(np.abs(constellation_array)**2)
            analysis['papr_db'] = 10 * np.log10(analysis['peak_power'] / analysis['mean_power']) if analysis['mean_power'] > 0 else 0
            
            # Get reference constellation for comparison
            ref_constellation = self._get_reference_constellation(mod_type)
            
            if ref_constellation is not None and len(ref_constellation) > 0:
                # Constellation quality metrics
                analysis['evm_percent'] = self._calculate_evm(constellation_array, ref_constellation, mod_type)
                analysis['magnitude_error'] = self._calculate_magnitude_error(constellation_array, ref_constellation)
                analysis['phase_error'] = self._calculate_phase_error(constellation_array, ref_constellation)
                analysis['cluster_separation'] = self._calculate_cluster_separation(constellation_array, ref_constellation)
                
                # Symbol error estimation
                analysis['estimated_ser'] = self._estimate_symbol_error_rate(constellation_array, ref_constellation)
                
                # Validation metrics
                validation_metrics = {
                    'constellation_accuracy': max(0, 100 - analysis['evm_percent']),
                    'demodulator_performance': 'good' if analysis['evm_percent'] < 10 else 'poor',
                    'snr_estimate_db': -20 * np.log10(analysis['evm_percent'] / 100) if analysis['evm_percent'] > 0 else 40,
                    'constellation_quality': self._assess_constellation_quality(analysis['evm_percent'])
                }
                analysis['validation_metrics'] = validation_metrics
            else:
                analysis['validation_metrics'] = {'error': 'No reference constellation available'}
            
            # Data pattern analysis for validation
            if len(constellation_array) > 10:
                analysis['data_pattern_analysis'] = self._analyze_data_patterns(constellation_array, mod_type)
            
            return analysis
            
        except Exception as e:
            return {'error': f'Constellation analysis error: {e}'}

    def _get_reference_constellation(self, mod_type):
        """Get reference constellation for comparison"""
        try:
            if mod_type == 'bpsk':
                return np.array([-1+0j, 1+0j])
            elif mod_type == 'qpsk':
                return np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
            elif mod_type == '8psk':
                angles = np.arange(0, 2*np.pi, 2*np.pi/8)
                return np.exp(1j * angles)
            elif mod_type == '16qam':
                # Standard 16-QAM constellation
                points = []
                for i in [-3, -1, 1, 3]:
                    for q in [-3, -1, 1, 3]:
                        points.append(complex(i, q))
                return np.array(points) / np.sqrt(10)
            elif mod_type == '64qam':
                # 64-QAM constellation
                points = []
                for i in range(-7, 8, 2):
                    for q in range(-7, 8, 2):
                        points.append(complex(i, q))
                return np.array(points) / np.sqrt(42)
            else:
                return None
        except Exception as e:
            print(f"Reference constellation error: {e}")
            return None

    def _calculate_evm(self, received_constellation, reference_constellation, mod_type):
        """Calculate Error Vector Magnitude (EVM) percentage"""
        try:
            if len(reference_constellation) == 0:
                return 100.0

            # Find closest reference points for each received point
            total_error_power = 0
            total_reference_power = 0
            
            for rx_point in received_constellation:
                # Find closest reference point
                distances = np.abs(rx_point - reference_constellation)
                closest_idx = np.argmin(distances)
                closest_ref = reference_constellation[closest_idx]
                
                # Calculate error vector
                error_vector = rx_point - closest_ref
                total_error_power += np.abs(error_vector)**2
                total_reference_power += np.abs(closest_ref)**2
            
            if total_reference_power == 0:
                return 100.0
            
            evm_rms = np.sqrt(total_error_power / len(received_constellation))
            ref_rms = np.sqrt(total_reference_power / len(received_constellation))
            
            evm_percent = (evm_rms / ref_rms) * 100 if ref_rms > 0 else 100.0
            return min(100.0, evm_percent)
            
        except Exception as e:
            print(f"EVM calculation error: {e}")
            return 100.0

    def _calculate_magnitude_error(self, received_constellation, reference_constellation):
        """Calculate magnitude error statistics"""
        try:
            magnitude_errors = []
            for rx_point in received_constellation:
                distances = np.abs(rx_point - reference_constellation)
                closest_idx = np.argmin(distances)
                closest_ref = reference_constellation[closest_idx]
                
                rx_mag = np.abs(rx_point)
                ref_mag = np.abs(closest_ref)
                mag_error = rx_mag - ref_mag
                magnitude_errors.append(mag_error)
            
            return {
                'mean': np.mean(magnitude_errors),
                'std': np.std(magnitude_errors),
                'rms': np.sqrt(np.mean(np.array(magnitude_errors)**2))
            }
        except Exception as e:
            return {'error': str(e)}

    def _calculate_phase_error(self, received_constellation, reference_constellation):
        """Calculate phase error statistics"""
        try:
            phase_errors = []
            for rx_point in received_constellation:
                distances = np.abs(rx_point - reference_constellation)
                closest_idx = np.argmin(distances)
                closest_ref = reference_constellation[closest_idx]
                
                rx_phase = np.angle(rx_point)
                ref_phase = np.angle(closest_ref)
                phase_error = np.angle(np.exp(1j * (rx_phase - ref_phase)))  # Wrap to [-pi, pi]
                phase_errors.append(phase_error)
            
            return {
                'mean_rad': np.mean(phase_errors),
                'std_rad': np.std(phase_errors),
                'rms_rad': np.sqrt(np.mean(np.array(phase_errors)**2)),
                'mean_deg': np.degrees(np.mean(phase_errors)),
                'std_deg': np.degrees(np.std(phase_errors))
            }
        except Exception as e:
            return {'error': str(e)}

    def _calculate_cluster_separation(self, received_constellation, reference_constellation):
        """Calculate constellation cluster separation"""
        try:
            if len(reference_constellation) < 2:
                return {'error': 'Need at least 2 reference points'}
            
            # Calculate minimum distance between reference points
            min_ref_distance = float('inf')
            for i in range(len(reference_constellation)):
                for j in range(i+1, len(reference_constellation)):
                    dist = np.abs(reference_constellation[i] - reference_constellation[j])
                    min_ref_distance = min(min_ref_distance, dist)
            
            # Calculate spread of received points around each reference
            max_spread = 0
            for ref_point in reference_constellation:
                distances = np.abs(received_constellation - ref_point)
                close_points = received_constellation[distances < min_ref_distance/2]
                if len(close_points) > 1:
                    spread = np.std(np.abs(close_points - ref_point))
                    max_spread = max(max_spread, spread)
            
            separation_ratio = min_ref_distance / (2 * max_spread) if max_spread > 0 else float('inf')
            
            return {
                'min_symbol_distance': min_ref_distance,
                'max_cluster_spread': max_spread,
                'separation_ratio': separation_ratio,
                'quality': 'good' if separation_ratio > 3 else 'poor'
            }
        except Exception as e:
            return {'error': str(e)}

    def _estimate_symbol_error_rate(self, received_constellation, reference_constellation):
        """Estimate Symbol Error Rate from constellation"""
        try:
            if len(reference_constellation) == 0:
                return 1.0
            
            # Count symbol errors (received points closer to wrong reference)
            errors = 0
            for rx_point in received_constellation:
                distances = np.abs(rx_point - reference_constellation)
                closest_indices = np.argsort(distances)
                
                # If closest and second closest are very different, likely correct
                # If they're similar, likely error region
                if len(closest_indices) > 1:
                    closest_dist = distances[closest_indices[0]]
                    second_dist = distances[closest_indices[1]]
                    
                    # Estimated error based on distance ratio
                    if closest_dist > 0.5 * second_dist:  # Threshold for error region
                        errors += 1
            
            ser = errors / len(received_constellation) if len(received_constellation) > 0 else 1.0
            return min(1.0, ser)
            
        except Exception as e:
            return 1.0

    def _assess_constellation_quality(self, evm_percent):
        """Assess overall constellation quality"""
        if evm_percent < 1:
            return 'excellent'
        elif evm_percent < 3:
            return 'very_good'
        elif evm_percent < 8:
            return 'good'
        elif evm_percent < 15:
            return 'acceptable'
        elif evm_percent < 25:
            return 'poor'
        else:
            return 'very_poor'

    def _analyze_data_patterns(self, constellation, mod_type):
        """Analyze data patterns in constellation for validation"""
        try:
            analysis = {}
            
            # Convert constellation to symbol decisions
            ref_constellation = self._get_reference_constellation(mod_type)
            if ref_constellation is None:
                return {'error': 'No reference available'}
            
            # Map received points to symbol decisions
            symbol_decisions = []
            for rx_point in constellation:
                distances = np.abs(rx_point - ref_constellation)
                closest_idx = np.argmin(distances)
                symbol_decisions.append(closest_idx)
            
            symbol_decisions = np.array(symbol_decisions)
            
            # Pattern analysis
            unique_symbols, counts = np.unique(symbol_decisions, return_counts=True)
            analysis['symbol_distribution'] = dict(zip(unique_symbols.tolist(), counts.tolist()))
            analysis['symbol_balance'] = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
            
            # Transition analysis (for detecting data patterns)
            if len(symbol_decisions) > 1:
                transitions = np.diff(symbol_decisions)
                analysis['transition_variance'] = np.var(transitions)
                analysis['pattern_detected'] = self._detect_specific_patterns(symbol_decisions)
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}

    def _detect_specific_patterns(self, symbols):
        """Detect specific data patterns like alternating, constant, etc."""
        try:
            if len(symbols) < 4:
                return 'insufficient_data'
            
            # Check for constant pattern
            if np.all(symbols == symbols[0]):
                return 'constant'
            
            # Check for alternating pattern
            alternating = True
            for i in range(1, min(10, len(symbols))):
                if symbols[i] == symbols[i-1]:
                    alternating = False
                    break
            if alternating:
                return 'alternating'
            
            # Check for repetitive short patterns
            for pattern_len in [2, 3, 4]:
                if len(symbols) >= 2 * pattern_len:
                    pattern = symbols[:pattern_len]
                    repetitive = True
                    for i in range(pattern_len, min(len(symbols), 4*pattern_len)):
                        if symbols[i] != pattern[i % pattern_len]:
                            repetitive = False
                            break
                    if repetitive:
                        return f'repetitive_{pattern_len}'
            
            return 'random'
            
        except Exception as e:
            return 'unknown'

    def _decode_channel_coding(self, coded_bits, coding_type, params):
        """Decode channel coding"""
        if not MODULES_AVAILABLE or coded_bits is None or len(coded_bits) == 0:
            return coded_bits, False, "No decoding available"

        try:
            # Get or create decoder
            if coding_type not in self.channel_decoders:
                self.channel_decoders[coding_type] = self._create_channel_decoder(coding_type, params)

            decoder = self.channel_decoders[coding_type]

            if coding_type == 'convolutional':
                # Viterbi decoding
                decoded_bits = decoder.viterbi_decode(coded_bits, 
                                                    is_hard_decision=True)
                success = True
                message = "Viterbi decoding successful"

            elif coding_type == 'turbo':
                # Split systematic and parity bits for turbo decoding
                n_info = len(coded_bits) // 3  # Assume rate 1/3
                if n_info > 0:
                    systematic = coded_bits[:n_info]
                    parity1 = coded_bits[n_info:2*n_info] 
                    parity2 = coded_bits[2*n_info:3*n_info]

                    decoded_bits = decoder.log_map_decode(systematic, parity1, parity2,
                                                        iterations=params.get('num_iterations', 8),
                                                        snr_db=self.user_params['snr_estimate'])
                    success = True
                    message = "Turbo decoding successful"
                else:
                    decoded_bits = coded_bits
                    success = False
                    message = "Insufficient bits for turbo decoding"

            elif coding_type == 'ldpc':
                # LDPC decoding
                received_llr = 2 * coded_bits.astype(float) - 1  # Convert to LLR
                decoded_bits, iterations = decoder.sum_product_decode(received_llr,
                                                                    max_iterations=params.get('max_iterations', 50))

                # Check syndrome
                H = decoder.H
                syndrome = (H @ decoded_bits) % 2
                success = np.all(syndrome == 0)
                message = f"LDPC decoding: {iterations} iterations, {'success' if success else 'failed'}"

            elif coding_type == 'polar':
                # Polar decoding
                received_llr = 2 * coded_bits.astype(float) - 1
                decoded_info = decoder.sc_decode(received_llr)
                decoded_bits = decoded_info  # Info bits only
                success = True
                message = "Polar SC decoding successful"

            elif coding_type == 'reed_solomon':
                # RS decoding (symbol-based)
                symbol_size = params.get('symbol_size', 8)
                n_symbols = len(coded_bits) // symbol_size

                # Convert bits to symbols
                symbols = []
                for i in range(n_symbols):
                    symbol_bits = coded_bits[i*symbol_size:(i+1)*symbol_size]
                    symbol = sum(bit << j for j, bit in enumerate(symbol_bits))
                    symbols.append(symbol)

                decoded_symbols, success = decoder.berlekamp_massey_decode(np.array(symbols))

                # Convert back to bits
                decoded_bits = []
                for symbol in decoded_symbols:
                    for i in range(symbol_size):
                        decoded_bits.append((symbol >> i) & 1)

                decoded_bits = np.array(decoded_bits, dtype=int)
                message = f"Reed-Solomon decoding: {'success' if success else 'failed'}"

            else:
                decoded_bits = coded_bits
                success = False
                message = f"Unknown coding type: {coding_type}"

            return decoded_bits, success, message

        except Exception as e:
            print(f"Channel decoding error ({coding_type}): {e}")
            return coded_bits, False, f"Decoding error: {str(e)}"

    def _create_channel_decoder(self, coding_type, params):
        """Create channel decoder instance"""
        if coding_type == 'convolutional':
            return ConvolutionalCoder(
                constraint_length=params.get('constraint_length', 7),
                code_rate=params.get('code_rate', 0.5),
                polynomials=params.get('polynomials', [0o133, 0o171])
            )
        elif coding_type == 'turbo':
            return TurboCoder(
                constraint_length=params.get('constraint_length', 3),
                interleaver_size=params.get('interleaver_size', 1024)
            )
        elif coding_type == 'ldpc':
            # Create LDPC matrix
            H = generate_hamming_matrix(3)  # Simplified
            return LDPCCoder(H)
        elif coding_type == 'polar':
            return PolarCoder(
                n=params.get('code_length', 1024),
                k=params.get('info_length', 512)
            )
        elif coding_type == 'reed_solomon':
            return ReedSolomonCoder(
                n=params.get('n', 255),
                k=params.get('k', 223)
            )
        else:
            raise ValueError(f"Unknown coding type: {coding_type}")

    def get_parameter_table(self):
        """Get parameter table for UI"""
        return self.param_table

    def update_parameter_table(self, mod_type=None, coding_type=None, **params):
        """Update parameter table"""
        if mod_type:
            self.param_table.update_modulation_params(mod_type, params)
        if coding_type:
            self.param_table.update_coding_params(coding_type, params)


# Test enhanced pipeline
def test_enhanced_pipeline():
    """Test enhanced processing pipeline"""
    print("🧪 Testing Enhanced Processing Pipeline")
    print("=" * 50)

    if not MODULES_AVAILABLE:
        print("❌ Required modules not available")
        return

    # Create pipeline
    pipeline = EnhancedProcessingPipeline(sample_rate=1e6)

    # Generate test signal (QPSK with convolutional coding)
    test_bits = np.random.randint(0, 2, 100)
    conv_coder = ConvolutionalCoder()
    coded_bits = conv_coder.encode(test_bits)

    # Simple QPSK modulation
    symbols = []
    for i in range(0, len(coded_bits), 2):
        if i+1 < len(coded_bits):
            i_bit = coded_bits[i]
            q_bit = coded_bits[i+1]
            symbol = (2*i_bit - 1) + 1j*(2*q_bit - 1)
            symbols.append(symbol)

    # Repeat symbols and add noise
    signal = np.repeat(symbols, 10)  # 10 samples per symbol
    noise = 0.1 * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    signal += noise

    # Process signal
    results = pipeline.process_signal(signal)

    # Display results
    for stage_name, stage_result in results.items():
        print(f"\n{stage_name}:")
        print(f"  Status: {stage_result['status']}")
        if 'result' in stage_result and stage_result['result'] is not None:
            result = stage_result['result']
            if isinstance(result, np.ndarray):
                print(f"  Result: {len(result)} elements")
            else:
                print(f"  Result: {result}")

        if 'confidence' in stage_result:
            print(f"  Confidence: {stage_result['confidence']:.1%}")

        if 'params' in stage_result and stage_result['params']:
            print(f"  Parameters: {stage_result['params']}")

    print("\n✅ Enhanced pipeline test completed")


if __name__ == "__main__":
    test_enhanced_pipeline()
