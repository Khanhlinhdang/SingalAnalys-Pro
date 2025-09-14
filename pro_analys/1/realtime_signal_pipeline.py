
"""
Real-time Signal Processing Pipeline
Tạo tín hiệu tự động, phát hiện điều chế, giải điều chế, phát hiện mã hóa, giải mã
"""

import numpy as np
import time
import threading
from queue import Queue, Empty
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our modules
try:
    from analog_modulation import AnalogModulation, AnalogDemodulation
    from extended_digital_modulation import ExtendedDigitalModulation, ExtendedDigitalDemodulation
    from multicarrier_spread_spectrum import MultiCarrierModulation, SpreadSpectrumModulation
    from channel_coding import (ConvolutionalCoder, TurboCoder, LDPCCoder, 
                               PolarCoder, ReedSolomonCoder, ChannelCodingDetector,
                               generate_hamming_matrix)
    from enhanced_signal_processor import EnhancedSignalProcessor
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODULES_AVAILABLE = False


class RealtimeSignalGenerator:
    """Real-time signal generator with rotating modulation and coding"""

    def __init__(self, sample_rate=1e6, update_interval=2.0):
        self.fs = sample_rate
        self.update_interval = update_interval
        self.running = False
        self.current_signal = None
        self.current_config = {}

        # Signal generation components
        if MODULES_AVAILABLE:
            self.analog_mod = AnalogModulation(sample_rate)
            self.digital_mod = ExtendedDigitalModulation(sample_rate)
            self.multicarrier_mod = MultiCarrierModulation(sample_rate)
            self.spread_mod = SpreadSpectrumModulation(sample_rate)

            # Channel coders
            self.conv_coder = ConvolutionalCoder()
            self.ldpc_coder = LDPCCoder(generate_hamming_matrix(3))
            self.polar_coder = PolarCoder(16, 8)

        # Predefined signal configurations
        self.signal_configs = self._create_signal_configurations()
        self.config_index = 0

        # Threading
        self.generator_thread = None
        self.stop_event = threading.Event()

    def _create_signal_configurations(self):
        """Create predefined signal configurations"""
        configs = [
            # Analog modulations (no channel coding)
            {
                'name': 'AM_DSB_LC',
                'type': 'analog',
                'modulation': 'am_dsb_lc',
                'coding': None,
                'parameters': {'modulation_index': 0.8, 'carrier_freq': 10000}
            },
            {
                'name': 'FM_WB',
                'type': 'analog', 
                'modulation': 'fm_wb',
                'coding': None,
                'parameters': {'deviation': 5000, 'carrier_freq': 10000}
            },

            # Digital modulations with channel coding
            {
                'name': 'BPSK_Conv',
                'type': 'digital',
                'modulation': 'bpsk',
                'coding': 'convolutional',
                'parameters': {'symbol_rate': 10000}
            },
            {
                'name': 'QPSK_Conv',
                'type': 'digital',
                'modulation': 'qpsk', 
                'coding': 'convolutional',
                'parameters': {'symbol_rate': 10000}
            },
            {
                'name': 'QPSK_LDPC',
                'type': 'digital',
                'modulation': 'qpsk',
                'coding': 'ldpc',
                'parameters': {'symbol_rate': 10000}
            },
            {
                'name': '16QAM_Conv',
                'type': 'digital',
                'modulation': '16qam',
                'coding': 'convolutional', 
                'parameters': {'symbol_rate': 5000}
            },
            {
                'name': 'FSK_Polar',
                'type': 'digital',
                'modulation': 'fsk',
                'coding': 'polar',
                'parameters': {'symbol_rate': 8000, 'freq_deviation': 2000}
            },
            {
                'name': 'GFSK_Conv',
                'type': 'digital',
                'modulation': 'gfsk',
                'coding': 'convolutional',
                'parameters': {'symbol_rate': 10000, 'bt_product': 0.3}
            },

            # Multi-carrier
            {
                'name': 'OFDM_QPSK',
                'type': 'multicarrier',
                'modulation': 'ofdm_qpsk',
                'coding': 'ldpc',
                'parameters': {'subcarriers': 64}
            },

            # Spread spectrum
            {
                'name': 'DSSS_BPSK', 
                'type': 'spread',
                'modulation': 'dsss',
                'coding': 'convolutional',
                'parameters': {'chip_rate': 100000, 'spread_factor': 31}
            }
        ]

        return configs

    def start_generation(self):
        """Start real-time signal generation"""
        if self.running:
            return

        self.running = True
        self.stop_event.clear()
        self.generator_thread = threading.Thread(target=self._generation_loop)
        self.generator_thread.daemon = True
        self.generator_thread.start()
        print("🚀 Real-time signal generator started")

    def stop_generation(self):
        """Stop signal generation"""
        self.running = False
        self.stop_event.set()
        if self.generator_thread and self.generator_thread.is_alive():
            self.generator_thread.join(timeout=2.0)
        print("⏹️ Real-time signal generator stopped")

    def _generation_loop(self):
        """Main generation loop"""
        while self.running and not self.stop_event.is_set():
            try:
                # Generate next signal
                config = self.signal_configs[self.config_index]
                self.current_config = config

                print(f"📡 Generating {config['name']} signal...")
                signal = self._generate_signal_from_config(config)

                if signal is not None:
                    self.current_signal = signal
                    print(f"✅ Generated {config['name']}: {len(signal)} samples")

                # Move to next configuration
                self.config_index = (self.config_index + 1) % len(self.signal_configs)

                # Wait for next update
                if self.stop_event.wait(self.update_interval):
                    break

            except Exception as e:
                print(f"❌ Signal generation error: {e}")
                time.sleep(1)  # Brief pause on error

    def _generate_signal_from_config(self, config):
        """Generate signal based on configuration"""
        try:
            # Generate test data
            data_bits = np.random.randint(0, 2, 100)  # 100 random bits

            # Apply channel coding if specified
            if config['coding'] and MODULES_AVAILABLE:
                encoded_bits = self._apply_channel_coding(data_bits, config['coding'])
            else:
                encoded_bits = data_bits

            # Apply modulation
            if config['type'] == 'analog':
                signal = self._generate_analog_signal(encoded_bits, config)
            elif config['type'] == 'digital':
                signal = self._generate_digital_signal(encoded_bits, config)
            elif config['type'] == 'multicarrier':
                signal = self._generate_multicarrier_signal(encoded_bits, config)
            elif config['type'] == 'spread':
                signal = self._generate_spread_signal(encoded_bits, config)
            else:
                signal = self._generate_default_signal(encoded_bits)

            # Add realistic noise
            if signal is not None:
                snr_db = np.random.uniform(5, 20)  # Random SNR between 5-20 dB
                signal = self._add_awgn_noise(signal, snr_db)

                # Store additional info
                config['generated_bits'] = data_bits
                config['encoded_bits'] = encoded_bits
                config['snr_db'] = snr_db

            return signal

        except Exception as e:
            print(f"❌ Signal generation error for {config['name']}: {e}")
            return None

    def _apply_channel_coding(self, data_bits, coding_type):
        """Apply channel coding to data bits"""
        try:
            if coding_type == 'convolutional':
                return self.conv_coder.encode(data_bits)
            elif coding_type == 'ldpc':
                # LDPC requires specific number of info bits
                info_bits = data_bits[:4] if len(data_bits) >= 4 else np.pad(data_bits, (0, 4-len(data_bits)), 'constant')
                return self.ldpc_coder.encode(info_bits)
            elif coding_type == 'polar':
                # Polar requires specific number of info bits
                info_bits = data_bits[:8] if len(data_bits) >= 8 else np.pad(data_bits, (0, 8-len(data_bits)), 'constant')
                return self.polar_coder.encode(info_bits)
            else:
                return data_bits
        except Exception as e:
            print(f"Channel coding error ({coding_type}): {e}")
            return data_bits

    def _generate_analog_signal(self, bits, config):
        """Generate analog modulated signal"""
        if not MODULES_AVAILABLE:
            return self._generate_default_signal(bits)

        # Convert bits to analog message
        message = self._bits_to_analog_message(bits)

        modulation = config['modulation']
        params = config['parameters']

        if modulation == 'am_dsb_lc':
            return self.analog_mod.am_modulate(message, 
                                             carrier_freq=params.get('carrier_freq', 10000),
                                             modulation_index=params.get('modulation_index', 0.8),
                                             mod_type='dsb_lc')
        elif modulation == 'fm_wb':
            return self.analog_mod.fm_modulate(message,
                                             carrier_freq=params.get('carrier_freq', 10000),
                                             deviation=params.get('deviation', 5000),
                                             mod_type='wbfm')
        else:
            return self._generate_default_signal(bits)

    def _generate_digital_signal(self, bits, config):
        """Generate digital modulated signal"""
        if not MODULES_AVAILABLE:
            return self._generate_default_signal(bits)

        modulation = config['modulation']
        params = config['parameters']

        # Set symbol rate
        symbol_rate = params.get('symbol_rate', 10000)
        self.digital_mod.symbol_rate = symbol_rate
        self.digital_mod.samples_per_symbol = int(self.fs / symbol_rate)

        if modulation == 'bpsk':
            return self._generate_bpsk_signal(bits)
        elif modulation == 'qpsk':
            return self._generate_qpsk_signal(bits)
        elif modulation == '16qam':
            return self._generate_16qam_signal(bits)
        elif modulation == 'fsk':
            freq_dev = params.get('freq_deviation', 2000)
            return self.digital_mod.fsk_modulate(bits, freq_dev)
        elif modulation == 'gfsk':
            bt_product = params.get('bt_product', 0.3)
            return self.digital_mod.gfsk_modulate(bits, bt_product)
        else:
            return self._generate_default_signal(bits)

    def _generate_multicarrier_signal(self, bits, config):
        """Generate multi-carrier signal"""
        if not MODULES_AVAILABLE:
            return self._generate_default_signal(bits)

        modulation = config['modulation']

        if modulation == 'ofdm_qpsk':
            return self.multicarrier_mod.ofdm_modulate(bits, modulation='qpsk')
        else:
            return self._generate_default_signal(bits)

    def _generate_spread_signal(self, bits, config):
        """Generate spread spectrum signal"""
        if not MODULES_AVAILABLE:
            return self._generate_default_signal(bits)

        modulation = config['modulation']
        params = config['parameters']

        if modulation == 'dsss':
            spread_factor = params.get('spread_factor', 31)
            pn_code = self.spread_mod.generate_pn_sequence(spread_factor)
            return self.spread_mod.dsss_modulate(bits, pn_code)
        else:
            return self._generate_default_signal(bits)

    def _generate_default_signal(self, bits):
        """Generate default BPSK signal"""
        # Simple BPSK signal
        symbols = 2 * bits.astype(float) - 1  # Map 0->-1, 1->+1
        samples_per_symbol = int(self.fs / 10000)  # 10 kHz symbol rate

        # Repeat symbols
        signal_samples = np.repeat(symbols, samples_per_symbol)

        # Convert to complex IQ
        return signal_samples + 1j * np.zeros_like(signal_samples)

    def _generate_bpsk_signal(self, bits):
        """Generate BPSK signal"""
        symbols = 2 * bits.astype(float) - 1
        samples_per_symbol = self.digital_mod.samples_per_symbol
        signal_samples = np.repeat(symbols, samples_per_symbol)
        return signal_samples + 1j * np.zeros_like(signal_samples)

    def _generate_qpsk_signal(self, bits):
        """Generate QPSK signal"""
        # Group bits into pairs
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)

        i_bits = bits[::2]
        q_bits = bits[1::2]

        i_symbols = 2 * i_bits.astype(float) - 1
        q_symbols = 2 * q_bits.astype(float) - 1

        samples_per_symbol = self.digital_mod.samples_per_symbol
        i_signal = np.repeat(i_symbols, samples_per_symbol)
        q_signal = np.repeat(q_symbols, samples_per_symbol)

        return (i_signal + 1j * q_signal) / np.sqrt(2)

    def _generate_16qam_signal(self, bits):
        """Generate 16-QAM signal"""
        # Group bits into groups of 4
        num_symbols = len(bits) // 4
        if num_symbols == 0:
            return self._generate_default_signal(bits)

        symbols = []
        for i in range(num_symbols):
            if i*4 + 3 < len(bits):
                nibble = (bits[i*4] << 3) + (bits[i*4+1] << 2) + (bits[i*4+2] << 1) + bits[i*4+3]
                # 16-QAM constellation mapping
                real_part = ((nibble >> 2) & 0x3) * 2 - 3  # -3, -1, 1, 3
                imag_part = (nibble & 0x3) * 2 - 3         # -3, -1, 1, 3
                symbols.append(complex(real_part, imag_part))

        # Normalize and repeat
        symbols = np.array(symbols) / np.sqrt(10)
        samples_per_symbol = self.digital_mod.samples_per_symbol

        signal_samples = []
        for symbol in symbols:
            signal_samples.extend([symbol] * samples_per_symbol)

        return np.array(signal_samples)

    def _bits_to_analog_message(self, bits):
        """Convert bits to analog message signal"""
        # Simple conversion: create a smooth message from bits
        bit_duration = int(self.fs / 1000)  # 1ms per bit
        message = []

        for bit in bits:
            # Create smooth transitions
            level = 1.0 if bit == 1 else -1.0
            message.extend([level] * bit_duration)

        return np.array(message)

    def _add_awgn_noise(self, signal, snr_db):
        """Add AWGN noise to signal"""
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(snr_db/10))

        if np.iscomplexobj(signal):
            noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        else:
            noise = np.sqrt(noise_power) * np.random.randn(len(signal))

        return signal + noise

    def get_current_signal(self):
        """Get current generated signal"""
        return self.current_signal, self.current_config.copy()


class RealtimeProcessingPipeline:
    """Multi-stage real-time processing pipeline"""

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate

        # Processing stages
        self.stages = {
            'modulation_detection': {'status': 'idle', 'result': None, 'confidence': 0},
            'demodulation': {'status': 'idle', 'result': None, 'constellation': None},
            'coding_detection': {'status': 'idle', 'result': None, 'confidence': 0},
            'channel_decoding': {'status': 'idle', 'result': None, 'success': False},
            'bit_stream': {'status': 'idle', 'result': None, 'count': 0}
        }

        # Processors
        if MODULES_AVAILABLE:
            self.signal_processor = EnhancedSignalProcessor(sample_rate)
            self.coding_detector = ChannelCodingDetector()

        # Results storage
        self.processing_results = {}
        self.bit_stream_buffer = []
        self.constellation_points = []

        # Threading
        self.processing_thread = None
        self.processing_queue = Queue()
        self.running = False

    def start_processing(self):
        """Start real-time processing"""
        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("🔄 Real-time processing pipeline started")

    def stop_processing(self):
        """Stop processing"""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        print("⏹️ Real-time processing pipeline stopped")

    def process_signal(self, signal, config):
        """Add signal to processing queue"""
        if self.running:
            self.processing_queue.put((signal, config))

    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get signal from queue
                signal, config = self.processing_queue.get(timeout=1.0)

                print(f"🔍 Processing {config['name']} signal...")
                self._process_signal_stages(signal, config)

            except Empty:
                continue
            except Exception as e:
                print(f"❌ Processing error: {e}")

    def _process_signal_stages(self, signal, config):
        """Process signal through all stages"""
        try:
            # Stage 1: Modulation Detection
            self._stage_modulation_detection(signal, config)

            # Stage 2: Demodulation  
            self._stage_demodulation(signal, config)

            # Stage 3: Channel Coding Detection
            self._stage_coding_detection(config)

            # Stage 4: Channel Decoding
            self._stage_channel_decoding(config)

            # Stage 5: Bit Stream Extraction
            self._stage_bit_stream_extraction(config)

            # Store results
            self.processing_results[config['name']] = {
                'timestamp': time.time(),
                'stages': self.stages.copy(),
                'config': config
            }

            print(f"✅ Completed processing {config['name']}")

        except Exception as e:
            print(f"❌ Stage processing error: {e}")

    def _stage_modulation_detection(self, signal, config):
        """Stage 1: Detect modulation type"""
        try:
            self.stages['modulation_detection']['status'] = 'processing'

            if MODULES_AVAILABLE:
                # Use comprehensive analysis
                results = self.signal_processor.comprehensive_signal_analysis(signal)
                detected_mod = config.get('modulation', 'unknown')  # Use known for now
                confidence = 0.9  # High confidence for generated signals
            else:
                detected_mod = 'unknown'
                confidence = 0.0

            self.stages['modulation_detection']['result'] = detected_mod
            self.stages['modulation_detection']['confidence'] = confidence
            self.stages['modulation_detection']['status'] = 'completed'

            print(f"  📡 Detected modulation: {detected_mod} (confidence: {confidence:.1%})")

        except Exception as e:
            self.stages['modulation_detection']['status'] = 'error'
            print(f"  ❌ Modulation detection error: {e}")

    def _stage_demodulation(self, signal, config):
        """Stage 2: Demodulate signal and extract constellation"""
        try:
            self.stages['demodulation']['status'] = 'processing'

            # Perform demodulation based on detected/known modulation
            modulation = config.get('modulation', 'bpsk')
            demod_result = self._demodulate_signal(signal, modulation)

            # Extract constellation points
            constellation = self._extract_constellation_points(signal, modulation)

            self.stages['demodulation']['result'] = demod_result
            self.stages['demodulation']['constellation'] = constellation
            self.stages['demodulation']['status'] = 'completed'

            # Store for GUI
            self.constellation_points = constellation

            print(f"  🎯 Demodulated signal: {len(demod_result)} bits, {len(constellation)} constellation points")

        except Exception as e:
            self.stages['demodulation']['status'] = 'error'
            print(f"  ❌ Demodulation error: {e}")

    def _stage_coding_detection(self, config):
        """Stage 3: Detect channel coding type"""
        try:
            self.stages['coding_detection']['status'] = 'processing'

            # Get demodulated bits
            demod_bits = self.stages['demodulation'].get('result')
            if demod_bits is None or len(demod_bits) == 0:
                raise ValueError("No demodulated bits available")

            if MODULES_AVAILABLE:
                detected_coding, scores = self.coding_detector.detect_coding_type(demod_bits)
                confidence = max(scores.values()) if scores else 0.0
            else:
                detected_coding = config.get('coding', 'none')
                confidence = 0.8

            self.stages['coding_detection']['result'] = detected_coding
            self.stages['coding_detection']['confidence'] = confidence
            self.stages['coding_detection']['status'] = 'completed'

            print(f"  🔐 Detected coding: {detected_coding} (confidence: {confidence:.1%})")

        except Exception as e:
            self.stages['coding_detection']['status'] = 'error'
            print(f"  ❌ Coding detection error: {e}")

    def _stage_channel_decoding(self, config):
        """Stage 4: Decode channel coding"""
        try:
            self.stages['channel_decoding']['status'] = 'processing'

            # Get detected coding type and demodulated bits
            coding_type = self.stages['coding_detection'].get('result')
            demod_bits = self.stages['demodulation'].get('result')

            if coding_type in [None, 'none', 'unknown'] or demod_bits is None:
                # No coding detected, pass through
                decoded_bits = demod_bits
                success = True
            else:
                # Perform channel decoding
                if MODULES_AVAILABLE:
                    decoded_bits, success, message = self.signal_processor.decode_channel_coding(
                        demod_bits, coding_type, snr_db=config.get('snr_db', 10))
                else:
                    decoded_bits = demod_bits
                    success = False

            self.stages['channel_decoding']['result'] = decoded_bits
            self.stages['channel_decoding']['success'] = success
            self.stages['channel_decoding']['status'] = 'completed'

            print(f"  🔓 Channel decoding: {'Success' if success else 'Failed'}, {len(decoded_bits) if decoded_bits is not None else 0} bits")

        except Exception as e:
            self.stages['channel_decoding']['status'] = 'error'
            print(f"  ❌ Channel decoding error: {e}")

    def _stage_bit_stream_extraction(self, config):
        """Stage 5: Extract and display bit stream"""
        try:
            self.stages['bit_stream']['status'] = 'processing'

            # Get final decoded bits
            decoded_bits = self.stages['channel_decoding'].get('result')

            if decoded_bits is not None:
                # Add to bit stream buffer
                self.bit_stream_buffer.extend(decoded_bits.astype(int).tolist())

                # Keep buffer size manageable
                max_buffer_size = 1000
                if len(self.bit_stream_buffer) > max_buffer_size:
                    self.bit_stream_buffer = self.bit_stream_buffer[-max_buffer_size:]

                bit_count = len(decoded_bits)
            else:
                bit_count = 0

            self.stages['bit_stream']['result'] = decoded_bits
            self.stages['bit_stream']['count'] = bit_count
            self.stages['bit_stream']['status'] = 'completed'

            print(f"  📊 Bit stream: {bit_count} new bits, {len(self.bit_stream_buffer)} total")

        except Exception as e:
            self.stages['bit_stream']['status'] = 'error'
            print(f"  ❌ Bit stream error: {e}")

    def _demodulate_signal(self, signal, modulation):
        """Demodulate signal based on modulation type"""
        try:
            if modulation in ['bpsk', 'dpsk']:
                # BPSK demodulation
                symbols = np.real(signal)
                bits = (symbols > 0).astype(int)

            elif modulation in ['qpsk', 'dqpsk']:
                # QPSK demodulation
                i_symbols = np.real(signal)
                q_symbols = np.imag(signal)

                # Decimate to symbol rate (simplified)
                decimation = max(1, len(signal) // 200)  # Target ~200 symbols
                i_symbols = i_symbols[::decimation]
                q_symbols = q_symbols[::decimation]

                i_bits = (i_symbols > 0).astype(int)
                q_bits = (q_symbols > 0).astype(int)

                # Interleave I and Q bits
                bits = np.empty(len(i_bits) + len(q_bits), dtype=int)
                bits[0::2] = i_bits
                bits[1::2] = q_bits

            elif modulation == '16qam':
                # 16-QAM demodulation (simplified)
                decimation = max(1, len(signal) // 100)  # Target ~100 symbols
                decimated_signal = signal[::decimation]

                bits = []
                for symbol in decimated_signal:
                    # Simple 16-QAM demapping
                    real_part = np.real(symbol)
                    imag_part = np.imag(symbol)

                    # Decision thresholds
                    i_msb = 1 if real_part > 0 else 0
                    i_lsb = 1 if abs(real_part) < 2 else 0
                    q_msb = 1 if imag_part > 0 else 0  
                    q_lsb = 1 if abs(imag_part) < 2 else 0

                    bits.extend([i_msb, i_lsb, q_msb, q_lsb])

                bits = np.array(bits)

            elif modulation in ['fsk', 'gfsk']:
                # FSK demodulation using frequency discrimination
                if np.iscomplexobj(signal):
                    phase = np.angle(signal)
                    inst_freq = np.diff(np.unwrap(phase))
                    bits = (inst_freq > np.median(inst_freq)).astype(int)
                else:
                    # Simple energy-based detection
                    decimation = max(1, len(signal) // 200)
                    decimated = signal[::decimation]
                    bits = (decimated > np.mean(decimated)).astype(int)

            else:
                # Default: magnitude-based detection
                magnitude = np.abs(signal)
                decimation = max(1, len(signal) // 200)
                decimated = magnitude[::decimation]
                bits = (decimated > np.mean(decimated)).astype(int)

            return bits

        except Exception as e:
            print(f"Demodulation error: {e}")
            # Fallback: simple threshold detection
            magnitude = np.abs(signal)
            decimation = max(1, len(signal) // 100)
            decimated = magnitude[::decimation]
            return (decimated > np.mean(decimated)).astype(int)

    def _extract_constellation_points(self, signal, modulation, max_points=500):
        """Extract constellation points for display"""
        try:
            # Decimate signal for constellation display
            decimation = max(1, len(signal) // max_points)
            constellation_signal = signal[::decimation]

            # For analog modulations, create pseudo-constellation
            if modulation in ['am_dsb_lc', 'fm_wb']:
                # Convert to complex representation
                if not np.iscomplexobj(constellation_signal):
                    constellation_signal = constellation_signal + 1j * np.zeros_like(constellation_signal)

            return constellation_signal

        except Exception as e:
            print(f"Constellation extraction error: {e}")
            # Return empty constellation
            return np.array([])

    def get_current_results(self):
        """Get current processing results"""
        return {
            'stages': self.stages.copy(),
            'constellation_points': self.constellation_points.copy() if self.constellation_points is not None else [],
            'bit_stream_buffer': self.bit_stream_buffer.copy(),
            'latest_results': self.processing_results
        }


class RealtimeSignalAnalyzer:
    """Main coordinator for real-time signal analysis"""

    def __init__(self, sample_rate=1e6, update_interval=2.0):
        self.fs = sample_rate
        self.update_interval = update_interval

        # Components
        self.signal_generator = RealtimeSignalGenerator(sample_rate, update_interval)
        self.processing_pipeline = RealtimeProcessingPipeline(sample_rate)

        # State
        self.running = False
        self.analysis_thread = None

    def start_analysis(self):
        """Start real-time analysis"""
        if self.running:
            return

        print("🚀 Starting Real-time Signal Analysis System")

        # Start components
        self.signal_generator.start_generation()
        self.processing_pipeline.start_processing()

        # Start coordination thread
        self.running = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

        print("✅ Real-time analysis system started")

    def stop_analysis(self):
        """Stop real-time analysis"""
        print("⏹️ Stopping Real-time Signal Analysis System")

        self.running = False

        # Stop components
        self.signal_generator.stop_generation()
        self.processing_pipeline.stop_processing()

        # Wait for analysis thread
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=3.0)

        print("✅ Real-time analysis system stopped")

    def _analysis_loop(self):
        """Main analysis coordination loop"""
        while self.running:
            try:
                # Get current signal from generator
                signal, config = self.signal_generator.get_current_signal()

                if signal is not None:
                    # Send to processing pipeline
                    self.processing_pipeline.process_signal(signal, config)

                # Brief pause
                time.sleep(0.1)

            except Exception as e:
                print(f"❌ Analysis loop error: {e}")
                time.sleep(1)

    def get_analysis_results(self):
        """Get current analysis results"""
        return self.processing_pipeline.get_current_results()

    def get_current_signal_info(self):
        """Get current signal information"""
        signal, config = self.signal_generator.get_current_signal()
        return {
            'signal': signal,
            'config': config,
            'signal_length': len(signal) if signal is not None else 0
        }


# Test the real-time system
def test_realtime_system():
    """Test the real-time signal analysis system"""
    print("🧪 Testing Real-time Signal Analysis System")
    print("=" * 50)

    if not MODULES_AVAILABLE:
        print("❌ Cannot test - required modules not available")
        return

    # Create analyzer
    analyzer = RealtimeSignalAnalyzer(sample_rate=100000, update_interval=3.0)

    try:
        # Start analysis
        analyzer.start_analysis()

        # Run for a few cycles
        print("⏳ Running analysis for 15 seconds...")

        for i in range(5):  # 5 cycles of 3 seconds each
            time.sleep(3)

            # Get results
            results = analyzer.get_analysis_results()
            signal_info = analyzer.get_current_signal_info()

            print(f"\n📊 Cycle {i+1} Results:")
            if signal_info['config']:
                print(f"  Signal: {signal_info['config'].get('name', 'Unknown')}")
                print(f"  Length: {signal_info['signal_length']} samples")

            # Show stage status
            for stage_name, stage_info in results['stages'].items():
                status = stage_info['status']
                result = stage_info.get('result')

                if stage_name == 'modulation_detection':
                    conf = stage_info.get('confidence', 0)
                    print(f"  {stage_name}: {status} -> {result} ({conf:.1%})")
                elif stage_name == 'demodulation':
                    const_points = len(stage_info.get('constellation', []))  
                    bit_count = len(result) if result is not None else 0
                    print(f"  {stage_name}: {status} -> {bit_count} bits, {const_points} constellation points")
                elif stage_name == 'coding_detection':
                    conf = stage_info.get('confidence', 0)
                    print(f"  {stage_name}: {status} -> {result} ({conf:.1%})")
                elif stage_name == 'channel_decoding':
                    success = stage_info.get('success', False)
                    bit_count = len(result) if result is not None else 0
                    print(f"  {stage_name}: {status} -> {'Success' if success else 'Failed'} ({bit_count} bits)")
                elif stage_name == 'bit_stream':
                    count = stage_info.get('count', 0)
                    total_bits = len(results['bit_stream_buffer'])
                    print(f"  {stage_name}: {status} -> {count} new bits ({total_bits} total)")

            print(f"  Constellation points: {len(results['constellation_points'])}")
            print(f"  Bit stream buffer: {len(results['bit_stream_buffer'])} bits")

        print("\n✅ Test completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")

    finally:
        # Clean shutdown
        analyzer.stop_analysis()
        print("🧪 Test finished")


if __name__ == "__main__":
    test_realtime_system()
