
"""
Multi-Carrier and Spread Spectrum Modulation Module
OFDM, OFDMA, SC-FDMA, DSSS, FHSS, CSS (LoRa), MIMO
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import chirp, correlate
import warnings
warnings.filterwarnings('ignore')


class MultiCarrierModulation:
    """Multi-carrier modulation techniques"""

    def __init__(self, sample_rate=1e6, subcarriers=64, cp_length=16):
        self.fs = sample_rate
        self.N = subcarriers  # Number of subcarriers
        self.cp_length = cp_length  # Cyclic prefix length
        self.pilot_spacing = 4  # Pilot every 4th subcarrier

    def ofdm_modulate(self, data_bits, modulation='qpsk'):
        """OFDM modulation with pilots"""
        # Map bits to symbols
        if modulation == 'bpsk':
            symbols = self._bpsk_map(data_bits)
        elif modulation == 'qpsk':
            symbols = self._qpsk_map(data_bits)
        elif modulation == '16qam':
            symbols = self._qam16_map(data_bits)
        elif modulation == '64qam':
            symbols = self._qam64_map(data_bits)
        else:
            raise ValueError(f"Unsupported modulation: {modulation}")

        # Create OFDM symbols
        ofdm_symbols = []
        data_subcarriers = self._get_data_subcarriers()
        symbols_per_ofdm = len(data_subcarriers)

        for i in range(0, len(symbols), symbols_per_ofdm):
            symbol_block = symbols[i:i+symbols_per_ofdm]
            if len(symbol_block) < symbols_per_ofdm:
                # Pad with zeros
                symbol_block = np.concatenate([symbol_block, 
                    np.zeros(symbols_per_ofdm - len(symbol_block))])

            # Create OFDM symbol
            ofdm_symbol = self._create_ofdm_symbol(symbol_block, data_subcarriers)
            ofdm_symbols.extend(ofdm_symbol)

        return np.array(ofdm_symbols)

    def _get_data_subcarriers(self):
        """Get indices of data subcarriers (excluding pilots and guard bands)"""
        all_subcarriers = np.arange(self.N)

        # Remove DC subcarrier
        all_subcarriers = all_subcarriers[all_subcarriers != self.N//2]

        # Remove pilot subcarriers
        pilot_indices = np.arange(0, self.N, self.pilot_spacing)
        data_subcarriers = np.setdiff1d(all_subcarriers, pilot_indices)

        # Remove guard bands (first and last few subcarriers)
        guard_band = self.N // 8
        data_subcarriers = data_subcarriers[(data_subcarriers >= guard_band) & 
                                          (data_subcarriers < self.N - guard_band)]

        return data_subcarriers

    def _create_ofdm_symbol(self, data_symbols, data_indices):
        """Create single OFDM symbol"""
        # Initialize frequency domain symbol
        freq_domain = np.zeros(self.N, dtype=complex)

        # Insert data symbols
        freq_domain[data_indices] = data_symbols

        # Insert pilots (known reference symbols)
        pilot_indices = np.arange(0, self.N, self.pilot_spacing)
        pilot_value = 1 + 1j  # Fixed pilot value
        freq_domain[pilot_indices] = pilot_value

        # IFFT to time domain
        time_domain = ifft(freq_domain) * np.sqrt(self.N)  # Normalization

        # Add cyclic prefix
        cp = time_domain[-self.cp_length:]
        ofdm_symbol = np.concatenate([cp, time_domain])

        return ofdm_symbol

    def ofdm_demodulate(self, received_signal, modulation='qpsk'):
        """OFDM demodulation"""
        symbol_length = self.N + self.cp_length
        num_symbols = len(received_signal) // symbol_length

        demodulated_bits = []
        data_indices = self._get_data_subcarriers()

        for i in range(num_symbols):
            start_idx = i * symbol_length
            end_idx = start_idx + symbol_length

            if end_idx <= len(received_signal):
                # Extract OFDM symbol
                ofdm_symbol = received_signal[start_idx:end_idx]

                # Remove cyclic prefix
                time_domain = ofdm_symbol[self.cp_length:]

                # FFT to frequency domain
                freq_domain = fft(time_domain) / np.sqrt(self.N)

                # Extract data subcarriers
                data_symbols = freq_domain[data_indices]

                # Demodulate symbols to bits
                if modulation == 'bpsk':
                    bits = self._bpsk_demap(data_symbols)
                elif modulation == 'qpsk':
                    bits = self._qpsk_demap(data_symbols)
                elif modulation == '16qam':
                    bits = self._qam16_demap(data_symbols)
                elif modulation == '64qam':
                    bits = self._qam64_demap(data_symbols)

                demodulated_bits.extend(bits)

        return np.array(demodulated_bits)

    def sc_fdma_modulate(self, data_bits, modulation='qpsk', mapping='localized'):
        """SC-FDMA modulation (LTE uplink)"""
        # Map bits to symbols
        if modulation == 'qpsk':
            symbols = self._qpsk_map(data_bits)
        elif modulation == '16qam':
            symbols = self._qam16_map(data_bits)
        else:
            symbols = self._qpsk_map(data_bits)

        sc_fdma_symbols = []
        M = min(self.N // 4, len(symbols))  # Number of symbols per SC-FDMA block

        for i in range(0, len(symbols), M):
            symbol_block = symbols[i:i+M]
            if len(symbol_block) < M:
                symbol_block = np.concatenate([symbol_block, 
                    np.zeros(M - len(symbol_block))])

            # DFT spreading
            dft_output = fft(symbol_block)

            # Subcarrier mapping
            freq_domain = np.zeros(self.N, dtype=complex)

            if mapping == 'localized':
                # Localized mapping (consecutive subcarriers)
                start_idx = self.N // 2 - M // 2
                freq_domain[start_idx:start_idx+M] = dft_output
            elif mapping == 'distributed':
                # Distributed mapping (equally spaced)
                step = self.N // M
                indices = np.arange(0, self.N, step)[:M]
                freq_domain[indices] = dft_output

            # IFFT to time domain
            time_domain = ifft(freq_domain) * np.sqrt(self.N)

            # Add cyclic prefix
            cp = time_domain[-self.cp_length:]
            sc_fdma_symbol = np.concatenate([cp, time_domain])

            sc_fdma_symbols.extend(sc_fdma_symbol)

        return np.array(sc_fdma_symbols)

    def fbmc_modulate(self, data_bits, prototype_filter='phydyas'):
        """Filter Bank Multi-Carrier modulation"""
        # Simplified FBMC implementation
        # Map bits to symbols
        symbols = self._qam16_map(data_bits)

        # Create prototype filter
        if prototype_filter == 'phydyas':
            K = 4  # Overlapping factor
            filter_length = K * self.N
            h = self._phydyas_filter(filter_length)
        else:
            h = np.ones(self.N)  # Rectangular filter

        # FBMC processing (simplified)
        fbmc_signal = []

        # Process symbols in blocks
        for i in range(0, len(symbols), self.N):
            symbol_block = symbols[i:i+self.N]
            if len(symbol_block) < self.N:
                symbol_block = np.concatenate([symbol_block, 
                    np.zeros(self.N - len(symbol_block))])

            # Apply IFFT
            time_block = ifft(symbol_block) * np.sqrt(self.N)

            # Apply prototype filter
            if len(h) == len(time_block):
                filtered_block = time_block * h
            else:
                filtered_block = time_block

            fbmc_signal.extend(filtered_block)

        return np.array(fbmc_signal)

    def _phydyas_filter(self, length):
        """Generate PHYDYAS prototype filter"""
        # Simplified PHYDYAS filter coefficients
        K = 4
        n = np.arange(length)
        filter_coeffs = np.zeros(length)

        for k in range(K):
            for i in range(length):
                if i >= k * length // K and i < (k+1) * length // K:
                    filter_coeffs[i] = 1 / np.sqrt(K) * np.cos(np.pi * (i - k * length // K) / (length // K))

        return filter_coeffs

    # Mapping functions for different modulations
    def _bpsk_map(self, bits):
        """Map bits to BPSK symbols"""
        return np.array([1 if bit == 1 else -1 for bit in bits], dtype=complex)

    def _qpsk_map(self, bits):
        """Map bits to QPSK symbols"""
        symbols = []
        for i in range(0, len(bits), 2):
            if i+1 < len(bits):
                dibits = (bits[i] << 1) + bits[i+1]
                if dibits == 0:    # 00
                    symbols.append(1+1j)
                elif dibits == 1:  # 01
                    symbols.append(1-1j)
                elif dibits == 2:  # 10
                    symbols.append(-1+1j)
                else:              # 11
                    symbols.append(-1-1j)
        return np.array(symbols) / np.sqrt(2)  # Normalize

    def _qam16_map(self, bits):
        """Map bits to 16-QAM symbols"""
        symbols = []
        for i in range(0, len(bits), 4):
            if i+3 < len(bits):
                nibble = (bits[i] << 3) + (bits[i+1] << 2) + (bits[i+2] << 1) + bits[i+3]
                # 16-QAM constellation mapping
                real_part = ((nibble >> 2) & 0x3) * 2 - 3  # -3, -1, 1, 3
                imag_part = (nibble & 0x3) * 2 - 3         # -3, -1, 1, 3
                symbols.append(complex(real_part, imag_part))

        # Normalize
        return np.array(symbols) / np.sqrt(10)

    def _qam64_map(self, bits):
        """Map bits to 64-QAM symbols"""
        symbols = []
        for i in range(0, len(bits), 6):
            if i+5 < len(bits):
                symbol_val = 0
                for j in range(6):
                    symbol_val += bits[i+j] * (2 ** (5-j))

                # 64-QAM constellation mapping
                real_part = ((symbol_val >> 3) & 0x7) * 2 - 7  # -7 to 7
                imag_part = (symbol_val & 0x7) * 2 - 7         # -7 to 7
                symbols.append(complex(real_part, imag_part))

        # Normalize
        return np.array(symbols) / np.sqrt(42)

    # Demapping functions
    def _bpsk_demap(self, symbols):
        """Demap BPSK symbols to bits"""
        return [1 if np.real(s) > 0 else 0 for s in symbols]

    def _qpsk_demap(self, symbols):
        """Demap QPSK symbols to bits"""
        bits = []
        for s in symbols:
            if np.real(s) > 0 and np.imag(s) > 0:
                bits.extend([0, 0])
            elif np.real(s) > 0 and np.imag(s) <= 0:
                bits.extend([0, 1])
            elif np.real(s) <= 0 and np.imag(s) > 0:
                bits.extend([1, 0])
            else:
                bits.extend([1, 1])
        return bits

    def _qam16_demap(self, symbols):
        """Demap 16-QAM symbols to bits"""
        bits = []
        for s in symbols:
            s_normalized = s * np.sqrt(10)  # Denormalize
            real_level = int(np.round((np.real(s_normalized) + 3) / 2))
            imag_level = int(np.round((np.imag(s_normalized) + 3) / 2))

            # Clip to valid range
            real_level = np.clip(real_level, 0, 3)
            imag_level = np.clip(imag_level, 0, 3)

            # Convert to 4 bits
            symbol_val = (real_level << 2) + imag_level
            for i in range(4):
                bits.append((symbol_val >> (3-i)) & 1)

        return bits

    def _qam64_demap(self, symbols):
        """Demap 64-QAM symbols to bits"""
        bits = []
        for s in symbols:
            s_normalized = s * np.sqrt(42)  # Denormalize
            real_level = int(np.round((np.real(s_normalized) + 7) / 2))
            imag_level = int(np.round((np.imag(s_normalized) + 7) / 2))

            # Clip to valid range
            real_level = np.clip(real_level, 0, 7)
            imag_level = np.clip(imag_level, 0, 7)

            # Convert to 6 bits
            symbol_val = (real_level << 3) + imag_level
            for i in range(6):
                bits.append((symbol_val >> (5-i)) & 1)

        return bits


class SpreadSpectrumModulation:
    """Spread spectrum modulation techniques"""

    def __init__(self, sample_rate=1e6, chip_rate=1000000):
        self.fs = sample_rate
        self.chip_rate = chip_rate
        self.samples_per_chip = int(self.fs / self.chip_rate)

    def dsss_modulate(self, data_bits, spreading_code):
        """Direct Sequence Spread Spectrum modulation"""
        # Ensure spreading code is numpy array
        spreading_code = np.array(spreading_code)
        spread_factor = len(spreading_code)

        # Spread each data bit
        spread_bits = []
        for bit in data_bits:
            # XOR data bit with spreading code
            if bit == 1:
                spread_sequence = spreading_code
            else:
                spread_sequence = 1 - spreading_code  # Invert for bit 0

            # Convert to +1/-1
            spread_sequence = 2 * spread_sequence - 1
            spread_bits.extend(spread_sequence)

        # Convert to samples
        samples = np.repeat(spread_bits, self.samples_per_chip)

        return samples

    def dsss_demodulate(self, received_signal, spreading_code):
        """DSSS demodulation using correlation"""
        spreading_code = np.array(spreading_code)
        spread_factor = len(spreading_code)

        # Convert spreading code to +1/-1
        spread_seq = 2 * spreading_code - 1
        spread_seq_samples = np.repeat(spread_seq, self.samples_per_chip)

        # Correlate received signal with spreading sequence
        correlation = correlate(received_signal, spread_seq_samples, mode='valid')

        # Sample at bit intervals
        bit_samples = len(spread_seq_samples)
        bits = []

        for i in range(0, len(correlation), bit_samples):
            if i < len(correlation):
                # Threshold decision
                bits.append(1 if correlation[i] > 0 else 0)

        return np.array(bits)

    def fhss_modulate(self, data_bits, hop_sequence, frequencies, hop_duration=0.001):
        """Frequency Hopping Spread Spectrum modulation"""
        samples_per_hop = int(self.fs * hop_duration)
        bits_per_hop = max(1, samples_per_hop // 1000)  # Assume 1000 samples per bit

        modulated_signal = []
        hop_index = 0

        for i in range(0, len(data_bits), bits_per_hop):
            # Get current hop frequency
            freq_index = hop_sequence[hop_index % len(hop_sequence)]
            carrier_freq = frequencies[freq_index % len(frequencies)]

            # Get data bits for this hop
            hop_bits = data_bits[i:i+bits_per_hop]
            if len(hop_bits) == 0:
                break

            # FSK modulation on current frequency
            t = np.arange(samples_per_hop) / self.fs

            # Simple FSK for bits in this hop
            hop_signal = np.zeros(samples_per_hop)
            samples_per_bit = samples_per_hop // len(hop_bits)

            for j, bit in enumerate(hop_bits):
                start_sample = j * samples_per_bit
                end_sample = min((j+1) * samples_per_bit, samples_per_hop)
                t_bit = t[start_sample:end_sample]

                if bit == 1:
                    freq_offset = 1000  # 1 kHz offset
                else:
                    freq_offset = -1000

                bit_signal = np.cos(2 * np.pi * (carrier_freq + freq_offset) * t_bit)
                hop_signal[start_sample:end_sample] = bit_signal

            modulated_signal.extend(hop_signal)
            hop_index += 1

        return np.array(modulated_signal)

    def fhss_demodulate(self, received_signal, hop_sequence, frequencies, hop_duration=0.001):
        """FHSS demodulation"""
        samples_per_hop = int(self.fs * hop_duration)
        num_hops = len(received_signal) // samples_per_hop

        demodulated_bits = []

        for hop_idx in range(num_hops):
            # Extract signal for current hop
            start_sample = hop_idx * samples_per_hop
            end_sample = min((hop_idx + 1) * samples_per_hop, len(received_signal))
            hop_signal = received_signal[start_sample:end_sample]

            # Get expected frequency for this hop
            freq_index = hop_sequence[hop_idx % len(hop_sequence)]
            carrier_freq = frequencies[freq_index % len(frequencies)]

            # Demodulate FSK (simplified energy detection)
            t = np.arange(len(hop_signal)) / self.fs

            # Mix with both FSK frequencies
            ref_high = np.cos(2 * np.pi * (carrier_freq + 1000) * t)
            ref_low = np.cos(2 * np.pi * (carrier_freq - 1000) * t)

            energy_high = np.sum(hop_signal * ref_high) ** 2
            energy_low = np.sum(hop_signal * ref_low) ** 2

            # Decide based on energy
            demodulated_bits.append(1 if energy_high > energy_low else 0)

        return np.array(demodulated_bits)

    def css_modulate(self, data_bits, sf=7, bandwidth=125000):
        """Chirp Spread Spectrum modulation (LoRa-like)"""
        # LoRa parameters
        symbols_per_second = bandwidth / (2**sf)
        samples_per_symbol = int(self.fs / symbols_per_second)

        # Group bits into symbols
        bits_per_symbol = sf
        css_signal = []

        for i in range(0, len(data_bits), bits_per_symbol):
            symbol_bits = data_bits[i:i+bits_per_symbol]
            if len(symbol_bits) < bits_per_symbol:
                # Pad with zeros
                symbol_bits = np.concatenate([symbol_bits, 
                    np.zeros(bits_per_symbol - len(symbol_bits))])

            # Convert bits to symbol value
            symbol_value = 0
            for j, bit in enumerate(symbol_bits):
                symbol_value += int(bit) * (2 ** (bits_per_symbol - 1 - j))

            # Generate chirp for this symbol
            symbol_chirp = self._generate_lora_chirp(symbol_value, sf, bandwidth, samples_per_symbol)
            css_signal.extend(symbol_chirp)

        return np.array(css_signal)

    def _generate_lora_chirp(self, symbol_value, sf, bandwidth, samples_per_symbol):
        """Generate LoRa chirp for given symbol"""
        t = np.arange(samples_per_symbol) / self.fs

        # Base chirp (upchirp)
        f0 = -bandwidth / 2
        f1 = bandwidth / 2
        chirp_signal = chirp(t, f0, t[-1], f1, method='linear')

        # Frequency shift based on symbol value
        freq_shift = symbol_value * bandwidth / (2**sf)
        shift_samples = int(freq_shift * samples_per_symbol / bandwidth)

        # Circular shift
        if shift_samples != 0:
            chirp_signal = np.roll(chirp_signal, shift_samples)

        return chirp_signal

    def css_demodulate(self, received_signal, sf=7, bandwidth=125000):
        """CSS demodulation using FFT correlation"""
        symbols_per_second = bandwidth / (2**sf)
        samples_per_symbol = int(self.fs / symbols_per_second)
        num_symbols = len(received_signal) // samples_per_symbol

        demodulated_bits = []

        for i in range(num_symbols):
            start_idx = i * samples_per_symbol
            end_idx = start_idx + samples_per_symbol

            if end_idx <= len(received_signal):
                symbol_signal = received_signal[start_idx:end_idx]

                # Generate reference chirps for all possible symbols
                correlations = []
                for symbol_val in range(2**sf):
                    ref_chirp = self._generate_lora_chirp(symbol_val, sf, bandwidth, samples_per_symbol)
                    correlation = np.abs(np.sum(symbol_signal * np.conj(ref_chirp)))
                    correlations.append(correlation)

                # Find symbol with highest correlation
                detected_symbol = np.argmax(correlations)

                # Convert symbol to bits
                for j in range(sf):
                    bit = (detected_symbol >> (sf - 1 - j)) & 1
                    demodulated_bits.append(bit)

        return np.array(demodulated_bits)

    def generate_pn_sequence(self, length, seed=1):
        """Generate pseudo-random sequence for spreading"""
        # Simple LFSR-based PN sequence
        lfsr = seed
        sequence = []

        for _ in range(length):
            # LFSR with polynomial x^4 + x + 1 (for example)
            bit = (lfsr ^ (lfsr >> 1)) & 1
            sequence.append(bit)
            lfsr = (lfsr >> 1) | (bit << 3)

        return np.array(sequence)

    def generate_gold_code(self, length, g1_seed=1, g2_seed=1):
        """Generate Gold code sequence"""
        # Two preferred sequences
        seq1 = self.generate_pn_sequence(length, g1_seed)
        seq2 = self.generate_pn_sequence(length, g2_seed)

        # XOR to create Gold code
        gold_code = seq1 ^ seq2
        return gold_code


class MIMOModulation:
    """MIMO and spatial modulation techniques"""

    def __init__(self, num_tx_antennas=2, num_rx_antennas=2):
        self.Nt = num_tx_antennas
        self.Nr = num_rx_antennas

    def alamouti_encode(self, symbols):
        """Alamouti Space-Time Block Code (2x1 MISO)"""
        if len(symbols) % 2 != 0:
            symbols = np.concatenate([symbols, [0]])  # Pad if odd length

        encoded_symbols = []
        for i in range(0, len(symbols), 2):
            s1, s2 = symbols[i], symbols[i+1]

            # Time slot 1: [s1, s2]
            # Time slot 2: [-s2*, s1*]
            time_slot_1 = [s1, s2]
            time_slot_2 = [-np.conj(s2), np.conj(s1)]

            encoded_symbols.append(time_slot_1)
            encoded_symbols.append(time_slot_2)

        return np.array(encoded_symbols)

    def alamouti_decode(self, received_symbols, channel_matrix):
        """Alamouti decoding"""
        decoded_symbols = []

        for i in range(0, len(received_symbols), 2):
            if i+1 < len(received_symbols):
                r1 = received_symbols[i]     # [r1, r2] at time 1
                r2 = received_symbols[i+1]   # [r1, r2] at time 2

                h11, h12 = channel_matrix[0, 0], channel_matrix[0, 1]  # Channel to rx1
                h21, h22 = channel_matrix[1, 0], channel_matrix[1, 1]  # Channel to rx2

                # Alamouti combining
                s1_est = (np.conj(h11) * r1[0] + h12 * np.conj(r2[1]) + 
                         np.conj(h21) * r1[1] + h22 * np.conj(r2[0])) /                         (abs(h11)**2 + abs(h12)**2 + abs(h21)**2 + abs(h22)**2)

                s2_est = (np.conj(h12) * r1[0] - h11 * np.conj(r2[1]) + 
                         np.conj(h22) * r1[1] - h21 * np.conj(r2[0])) /                         (abs(h11)**2 + abs(h12)**2 + abs(h21)**2 + abs(h22)**2)

                decoded_symbols.extend([s1_est, s2_est])

        return np.array(decoded_symbols)

    def spatial_multiplexing(self, symbols):
        """Spatial multiplexing (V-BLAST like)"""
        # Split symbols across transmit antennas
        symbols_per_antenna = len(symbols) // self.Nt

        tx_streams = []
        for i in range(self.Nt):
            start_idx = i * symbols_per_antenna
            end_idx = start_idx + symbols_per_antenna
            tx_streams.append(symbols[start_idx:end_idx])

        return np.array(tx_streams)

    def zero_forcing_detection(self, received_matrix, channel_matrix):
        """Zero-forcing detection for spatial multiplexing"""
        # Pseudo-inverse for ZF detection
        H_inv = np.linalg.pinv(channel_matrix)
        detected_symbols = H_inv @ received_matrix

        return detected_symbols

    def mmse_detection(self, received_matrix, channel_matrix, noise_variance=0.1):
        """MMSE detection for spatial multiplexing"""
        H = channel_matrix
        H_H = np.conj(H.T)
        I = np.eye(H.shape[1])

        # MMSE filter
        mmse_filter = np.linalg.inv(H_H @ H + noise_variance * I) @ H_H
        detected_symbols = mmse_filter @ received_matrix

        return detected_symbols


# Modulation classification for multi-carrier and spread spectrum
class AdvancedModulationDetector:
    """Detect multi-carrier and spread spectrum modulations"""

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate

    def detect_modulation_type(self, signal):
        """Detect advanced modulation types"""
        features = self._extract_advanced_features(signal)
        return self._classify_advanced_modulation(features)

    def _extract_advanced_features(self, signal):
        """Extract features for advanced modulation detection"""
        # Spectral features
        spectrum = np.abs(fft(signal))**2
        freq_bins = fftfreq(len(signal), 1/self.fs)

        # Peak-to-average power ratio
        papr = np.max(np.abs(signal)**2) / np.mean(np.abs(signal)**2)

        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-12)))
        arithmetic_mean = np.mean(spectrum)
        spectral_flatness = geometric_mean / arithmetic_mean

        # Autocorrelation properties
        autocorr = correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr_peaks = len(signal.find_peaks(autocorr, height=0.5*np.max(autocorr))[0])

        # Cyclostationary features
        cyclic_features = self._detect_cyclostationarity(signal)

        return {
            'papr': papr,
            'spectral_flatness': spectral_flatness,
            'autocorr_peaks': autocorr_peaks,
            'cyclic_features': cyclic_features
        }

    def _detect_cyclostationarity(self, signal):
        """Detect cyclostationary features"""
        # Simplified cyclostationarity detection
        N = len(signal)
        lags = np.arange(-N//4, N//4)

        # Spectral correlation function (simplified)
        correlation_sum = 0
        for lag in lags[::10]:  # Subsample for efficiency
            if 0 <= lag < N:
                shifted = np.roll(signal, lag)
                correlation_sum += np.abs(np.sum(signal * np.conj(shifted)))

        return correlation_sum / len(lags)

    def _classify_advanced_modulation(self, features):
        """Classify based on extracted features"""
        papr = features['papr']
        spectral_flatness = features['spectral_flatness']
        cyclic_features = features['cyclic_features']

        if papr > 8:  # High PAPR indicates OFDM
            if spectral_flatness > 0.8:
                return "OFDM"
            else:
                return "SC-FDMA"
        elif papr < 3 and cyclic_features > 1000:  # Spread spectrum characteristics
            if spectral_flatness > 0.9:
                return "DSSS"
            elif spectral_flatness < 0.3:
                return "FHSS"
            else:
                return "CSS/LoRa"
        else:
            return "Single Carrier"
