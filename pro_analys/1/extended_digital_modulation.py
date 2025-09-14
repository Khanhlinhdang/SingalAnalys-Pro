
"""
Extended Digital Modulation Module
Module mở rộng cho điều chế số: FSK, GFSK, MSK, GMSK, QAM cao cấp, APSK
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt, lfilter, hilbert
from scipy.special import erfc
import warnings
warnings.filterwarnings('ignore')


class ExtendedDigitalModulation:
    """Extended digital modulation techniques"""

    def __init__(self, sample_rate=1e6, symbol_rate=10000):
        self.fs = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(self.fs / self.symbol_rate)

    def ook_modulate(self, bits):
        """On-Off Keying modulation"""
        symbols = []
        for bit in bits:
            if bit == 1:
                symbol = np.ones(self.samples_per_symbol)
            else:
                symbol = np.zeros(self.samples_per_symbol)
            symbols.extend(symbol)
        return np.array(symbols)

    def ask_modulate(self, bits, M=4):
        """M-ary ASK modulation"""
        if M == 2:
            return self.ook_modulate(bits)

        # Convert bits to M-ary symbols
        bits_per_symbol = int(np.log2(M))
        symbols = []

        # Group bits into symbols
        for i in range(0, len(bits), bits_per_symbol):
            bit_group = bits[i:i+bits_per_symbol]
            if len(bit_group) == bits_per_symbol:
                symbol_value = 0
                for j, bit in enumerate(bit_group):
                    symbol_value += bit * (2 ** (bits_per_symbol - 1 - j))

                # Map to amplitude levels
                amplitude = (2 * symbol_value + 1 - M) / (M - 1)
                symbol_samples = amplitude * np.ones(self.samples_per_symbol)
                symbols.extend(symbol_samples)

        return np.array(symbols)

    def fsk_modulate(self, bits, freq_deviation=5000):
        """Binary FSK modulation"""
        t_symbol = np.arange(self.samples_per_symbol) / self.fs
        symbols = []

        for bit in bits:
            if bit == 1:
                freq = freq_deviation
            else:
                freq = -freq_deviation

            symbol = np.cos(2 * np.pi * freq * t_symbol)
            symbols.extend(symbol)

        return np.array(symbols)

    def mfsk_modulate(self, bits, M=4, freq_spacing=2000):
        """M-ary FSK modulation"""
        bits_per_symbol = int(np.log2(M))
        t_symbol = np.arange(self.samples_per_symbol) / self.fs
        symbols = []

        # Group bits into M-ary symbols
        for i in range(0, len(bits), bits_per_symbol):
            bit_group = bits[i:i+bits_per_symbol]
            if len(bit_group) == bits_per_symbol:
                symbol_value = 0
                for j, bit in enumerate(bit_group):
                    symbol_value += bit * (2 ** (bits_per_symbol - 1 - j))

                # Map to frequency
                freq = (symbol_value - (M-1)/2) * freq_spacing
                symbol = np.cos(2 * np.pi * freq * t_symbol)
                symbols.extend(symbol)

        return np.array(symbols)

    def gfsk_modulate(self, bits, bt_product=0.3, freq_deviation=5000):
        """Gaussian FSK modulation (Bluetooth, GSM)"""
        # Generate NRZ signal
        nrz = np.repeat(2*np.array(bits) - 1, self.samples_per_symbol)

        # Gaussian filter
        # BT = bt_product, T = symbol period
        T_symbol = 1 / self.symbol_rate
        sigma = T_symbol / (2 * np.pi * bt_product) * np.sqrt(np.log(2))

        # Create Gaussian filter
        filter_length = int(4 * sigma * self.fs)
        if filter_length % 2 == 0:
            filter_length += 1
        t_filter = np.arange(-(filter_length//2), filter_length//2 + 1) / self.fs
        gaussian_filter = np.exp(-0.5 * (t_filter / sigma) ** 2)
        gaussian_filter /= np.sum(gaussian_filter)

        # Filter the NRZ signal
        filtered_signal = np.convolve(nrz, gaussian_filter, mode='same')

        # Integrate for frequency modulation
        phase = 2 * np.pi * freq_deviation * np.cumsum(filtered_signal) / self.fs

        # Generate GFSK signal
        gfsk_signal = np.cos(phase)

        return gfsk_signal

    def cpfsk_modulate(self, bits, modulation_index=0.5):
        """Continuous Phase FSK modulation"""
        # Convert bits to +1/-1
        data = 2 * np.array(bits) - 1

        # Repeat each bit for samples_per_symbol
        repeated_data = np.repeat(data, self.samples_per_symbol)

        # Integrate for phase
        phase = np.cumsum(repeated_data) * np.pi * modulation_index / self.samples_per_symbol

        # Generate CPFSK signal
        cpfsk_signal = np.cos(phase)

        return cpfsk_signal

    def msk_modulate(self, bits):
        """Minimum Shift Keying (MSK) modulation"""
        # MSK is CPFSK with modulation index = 0.5
        return self.cpfsk_modulate(bits, modulation_index=0.5)

    def gmsk_modulate(self, bits, bt_product=0.3):
        """Gaussian MSK modulation (GSM)"""
        # GMSK is GFSK with modulation index = 0.5
        freq_deviation = self.symbol_rate / 4  # For MSK
        return self.gfsk_modulate(bits, bt_product, freq_deviation)

    def dpsk_modulate(self, bits, M=2):
        """Differential PSK modulation"""
        if M == 2:  # DBPSK
            # Differential encoding
            diff_bits = [0]  # Start with reference
            for i in range(len(bits)):
                diff_bits.append(diff_bits[-1] ^ bits[i])

            # BPSK modulation of differential bits
            symbols = []
            t_symbol = np.arange(self.samples_per_symbol) / self.fs

            for bit in diff_bits[1:]:  # Skip reference
                phase = 0 if bit == 0 else np.pi
                symbol = np.cos(2 * np.pi * 1000 * t_symbol + phase)  # 1kHz for illustration
                symbols.extend(symbol)

        elif M == 4:  # DQPSK
            # Group bits into 2-bit symbols
            symbols = []
            prev_phase = 0

            for i in range(0, len(bits), 2):
                if i + 1 < len(bits):
                    dibits = (bits[i] << 1) + bits[i+1]
                    # Phase changes: 00->0, 01->π/2, 10->π, 11->3π/2
                    phase_change = dibits * np.pi / 2
                    current_phase = prev_phase + phase_change

                    # Generate symbol
                    t_symbol = np.arange(self.samples_per_symbol) / self.fs
                    symbol = np.cos(2 * np.pi * 1000 * t_symbol + current_phase)
                    symbols.extend(symbol)

                    prev_phase = current_phase

        return np.array(symbols)

    def qam_modulate(self, bits, M=16):
        """High-order QAM modulation (16, 64, 256, 1024)"""
        bits_per_symbol = int(np.log2(M))

        # Define constellation points
        if M == 16:
            constellation = self._generate_16qam_constellation()
        elif M == 64:
            constellation = self._generate_64qam_constellation()
        elif M == 256:
            constellation = self._generate_256qam_constellation()
        elif M == 1024:
            constellation = self._generate_1024qam_constellation()
        else:
            raise ValueError(f"Unsupported QAM order: {M}")

        # Group bits and map to symbols
        symbols = []
        t_symbol = np.arange(self.samples_per_symbol) / self.fs
        carrier_freq = 10000  # 10 kHz carrier for illustration

        for i in range(0, len(bits), bits_per_symbol):
            bit_group = bits[i:i+bits_per_symbol]
            if len(bit_group) == bits_per_symbol:
                # Convert to decimal index
                symbol_index = 0
                for j, bit in enumerate(bit_group):
                    symbol_index += bit * (2 ** (bits_per_symbol - 1 - j))

                # Get constellation point
                complex_symbol = constellation[symbol_index]

                # Generate modulated symbol
                i_part = np.real(complex_symbol) * np.cos(2 * np.pi * carrier_freq * t_symbol)
                q_part = -np.imag(complex_symbol) * np.sin(2 * np.pi * carrier_freq * t_symbol)
                symbol = i_part + q_part

                symbols.extend(symbol)

        return np.array(symbols)

    def _generate_16qam_constellation(self):
        """Generate 16-QAM constellation points"""
        constellation = []
        for i in range(4):
            for q in range(4):
                real_part = 2*i - 3  # -3, -1, 1, 3
                imag_part = 2*q - 3  # -3, -1, 1, 3
                constellation.append(complex(real_part, imag_part))

        # Normalize
        constellation = np.array(constellation)
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        return constellation

    def _generate_64qam_constellation(self):
        """Generate 64-QAM constellation points"""
        constellation = []
        for i in range(8):
            for q in range(8):
                real_part = 2*i - 7  # -7, -5, -3, -1, 1, 3, 5, 7
                imag_part = 2*q - 7  # -7, -5, -3, -1, 1, 3, 5, 7
                constellation.append(complex(real_part, imag_part))

        # Normalize
        constellation = np.array(constellation)
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        return constellation

    def _generate_256qam_constellation(self):
        """Generate 256-QAM constellation points"""
        constellation = []
        for i in range(16):
            for q in range(16):
                real_part = 2*i - 15  # -15, -13, ..., 13, 15
                imag_part = 2*q - 15  # -15, -13, ..., 13, 15
                constellation.append(complex(real_part, imag_part))

        # Normalize
        constellation = np.array(constellation)
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        return constellation

    def _generate_1024qam_constellation(self):
        """Generate 1024-QAM constellation points"""
        constellation = []
        for i in range(32):
            for q in range(32):
                real_part = 2*i - 31  # -31, -29, ..., 29, 31
                imag_part = 2*q - 31  # -31, -29, ..., 29, 31
                constellation.append(complex(real_part, imag_part))

        # Normalize
        constellation = np.array(constellation)
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        return constellation

    def apsk_modulate(self, bits, M=16, ring_ratios=None):
        """Amplitude and Phase Shift Keying (DVB-S2/S2X)"""
        bits_per_symbol = int(np.log2(M))

        if M == 16:
            # 16-APSK: 4+12 constellation (inner ring: 4, outer ring: 12)
            if ring_ratios is None:
                ring_ratios = [1.0, 2.85]  # Typical ratio for 16-APSK
            constellation = self._generate_16apsk_constellation(ring_ratios)
        elif M == 32:
            # 32-APSK: 4+12+16 constellation
            if ring_ratios is None:
                ring_ratios = [1.0, 2.84, 5.27]
            constellation = self._generate_32apsk_constellation(ring_ratios)
        else:
            raise ValueError(f"Unsupported APSK order: {M}")

        # Map bits to symbols (similar to QAM)
        symbols = []
        t_symbol = np.arange(self.samples_per_symbol) / self.fs
        carrier_freq = 10000

        for i in range(0, len(bits), bits_per_symbol):
            bit_group = bits[i:i+bits_per_symbol]
            if len(bit_group) == bits_per_symbol:
                symbol_index = 0
                for j, bit in enumerate(bit_group):
                    symbol_index += bit * (2 ** (bits_per_symbol - 1 - j))

                complex_symbol = constellation[symbol_index]

                # Generate modulated symbol
                i_part = np.real(complex_symbol) * np.cos(2 * np.pi * carrier_freq * t_symbol)
                q_part = -np.imag(complex_symbol) * np.sin(2 * np.pi * carrier_freq * t_symbol)
                symbol = i_part + q_part

                symbols.extend(symbol)

        return np.array(symbols)

    def _generate_16apsk_constellation(self, ring_ratios):
        """Generate 16-APSK constellation (4+12)"""
        constellation = []

        # Inner ring (4 points)
        for i in range(4):
            phase = 2 * np.pi * i / 4 + np.pi/4  # 45° offset
            point = ring_ratios[0] * np.exp(1j * phase)
            constellation.append(point)

        # Outer ring (12 points)
        for i in range(12):
            phase = 2 * np.pi * i / 12
            point = ring_ratios[1] * np.exp(1j * phase)
            constellation.append(point)

        return np.array(constellation)

    def _generate_32apsk_constellation(self, ring_ratios):
        """Generate 32-APSK constellation (4+12+16)"""
        constellation = []

        # Inner ring (4 points)
        for i in range(4):
            phase = 2 * np.pi * i / 4 + np.pi/4
            point = ring_ratios[0] * np.exp(1j * phase)
            constellation.append(point)

        # Middle ring (12 points)
        for i in range(12):
            phase = 2 * np.pi * i / 12
            point = ring_ratios[1] * np.exp(1j * phase)
            constellation.append(point)

        # Outer ring (16 points)
        for i in range(16):
            phase = 2 * np.pi * i / 16
            point = ring_ratios[2] * np.exp(1j * phase)
            constellation.append(point)

        return np.array(constellation)


class ExtendedDigitalDemodulation:
    """Extended digital demodulation techniques"""

    def __init__(self, sample_rate=1e6, symbol_rate=10000):
        self.fs = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(self.fs / self.symbol_rate)

    def ook_demodulate(self, signal, threshold=None):
        """OOK demodulation with envelope detection"""
        # Envelope detection
        envelope = np.abs(hilbert(signal))

        # Threshold detection
        if threshold is None:
            threshold = np.mean(envelope)

        # Sample at symbol centers
        symbol_centers = np.arange(self.samples_per_symbol//2, len(envelope), self.samples_per_symbol)
        bits = []

        for center in symbol_centers:
            if center < len(envelope):
                bits.append(1 if envelope[center] > threshold else 0)

        return np.array(bits)

    def fsk_demodulate(self, signal, freq_deviation=5000, method='energy'):
        """FSK demodulation"""
        if method == 'energy':
            return self._fsk_energy_demodulate(signal, freq_deviation)
        elif method == 'pll':
            return self._fsk_pll_demodulate(signal, freq_deviation)
        else:
            return self._fsk_energy_demodulate(signal, freq_deviation)

    def _fsk_energy_demodulate(self, signal, freq_deviation):
        """FSK demodulation using energy detection"""
        # Create matched filters for both frequencies
        t_filter = np.arange(self.samples_per_symbol) / self.fs

        # Filter for frequency +freq_deviation (bit 1)
        filter_1 = np.cos(2 * np.pi * freq_deviation * t_filter)

        # Filter for frequency -freq_deviation (bit 0)
        filter_0 = np.cos(2 * np.pi * (-freq_deviation) * t_filter)

        # Correlate signal with both filters
        corr_1 = np.convolve(signal, filter_1[::-1], mode='valid')
        corr_0 = np.convolve(signal, filter_0[::-1], mode='valid')

        # Sample at symbol intervals
        symbol_indices = np.arange(0, len(corr_1), self.samples_per_symbol)
        bits = []

        for idx in symbol_indices:
            if idx < len(corr_1) and idx < len(corr_0):
                # Decide based on which correlation is larger
                if np.abs(corr_1[idx]) > np.abs(corr_0[idx]):
                    bits.append(1)
                else:
                    bits.append(0)

        return np.array(bits)

    def _fsk_pll_demodulate(self, signal, freq_deviation):
        """FSK demodulation using frequency discriminator"""
        # Convert to complex baseband
        analytic_signal = hilbert(signal)

        # Extract instantaneous frequency
        phase = np.angle(analytic_signal)
        unwrapped_phase = np.unwrap(phase)
        inst_freq = np.diff(unwrapped_phase) * self.fs / (2 * np.pi)

        # Pad to maintain length
        inst_freq = np.concatenate([[inst_freq[0]], inst_freq])

        # Sample at symbol centers and threshold
        symbol_centers = np.arange(self.samples_per_symbol//2, len(inst_freq), self.samples_per_symbol)
        bits = []

        for center in symbol_centers:
            if center < len(inst_freq):
                # Positive frequency deviation -> bit 1, negative -> bit 0
                bits.append(1 if inst_freq[center] > 0 else 0)

        return np.array(bits)

    def gfsk_demodulate(self, signal, bt_product=0.3):
        """GFSK demodulation"""
        # Use frequency discriminator approach
        analytic_signal = hilbert(signal)
        phase = np.angle(analytic_signal)
        unwrapped_phase = np.unwrap(phase)
        inst_freq = np.diff(unwrapped_phase) * self.fs / (2 * np.pi)
        inst_freq = np.concatenate([[inst_freq[0]], inst_freq])

        # Low-pass filter to remove noise
        T_symbol = 1 / self.symbol_rate
        cutoff_freq = 1 / (2 * bt_product * T_symbol)
        nyquist = self.fs / 2
        if cutoff_freq < nyquist:
            b, a = butter(4, cutoff_freq / nyquist, btype='low')
            inst_freq = filtfilt(b, a, inst_freq)

        # Sample and threshold
        symbol_centers = np.arange(self.samples_per_symbol//2, len(inst_freq), self.samples_per_symbol)
        bits = []

        for center in symbol_centers:
            if center < len(inst_freq):
                bits.append(1 if inst_freq[center] > 0 else 0)

        return np.array(bits)

    def msk_demodulate(self, signal):
        """MSK demodulation using OQPSK approach"""
        # MSK can be demodulated as OQPSK with half-symbol offset
        # Convert to complex baseband
        t = np.arange(len(signal)) / self.fs

        # Demodulate I and Q channels with offset
        local_osc_i = 2 * np.cos(2 * np.pi * 1000 * t)  # Assume 1kHz carrier
        local_osc_q = -2 * np.sin(2 * np.pi * 1000 * t)

        i_channel = signal * local_osc_i
        q_channel = signal * local_osc_q

        # Low-pass filter
        nyquist = self.fs / 2
        cutoff = (self.symbol_rate / 2) / nyquist
        if cutoff < 1:
            b, a = butter(4, cutoff, btype='low')
            i_channel = filtfilt(b, a, i_channel)
            q_channel = filtfilt(b, a, q_channel)

        # Sample I channel at even symbol times, Q at odd symbol times
        bits = []
        for i in range(len(signal) // self.samples_per_symbol):
            center = i * self.samples_per_symbol + self.samples_per_symbol // 2
            if center < len(i_channel):
                if i % 2 == 0:  # Even symbols from I channel
                    bits.append(1 if i_channel[center] > 0 else 0)
                else:  # Odd symbols from Q channel (offset)
                    offset_center = center - self.samples_per_symbol // 2
                    if offset_center >= 0 and offset_center < len(q_channel):
                        bits.append(1 if q_channel[offset_center] > 0 else 0)

        return np.array(bits)

    def qam_demodulate(self, signal, M=16, carrier_freq=10000):
        """High-order QAM demodulation"""
        t = np.arange(len(signal)) / self.fs

        # Coherent demodulation
        local_osc_i = 2 * np.cos(2 * np.pi * carrier_freq * t)
        local_osc_q = -2 * np.sin(2 * np.pi * carrier_freq * t)

        i_channel = signal * local_osc_i
        q_channel = signal * local_osc_q

        # Low-pass filtering
        nyquist = self.fs / 2
        cutoff = (self.symbol_rate) / nyquist
        if cutoff < 1:
            b, a = butter(6, cutoff, btype='low')
            i_channel = filtfilt(b, a, i_channel)
            q_channel = filtfilt(b, a, q_channel)

        # Sample at symbol centers
        symbol_centers = np.arange(self.samples_per_symbol//2, len(signal), self.samples_per_symbol)
        complex_symbols = []

        for center in symbol_centers:
            if center < len(i_channel) and center < len(q_channel):
                complex_symbols.append(complex(i_channel[center], q_channel[center]))

        # Get constellation and decode
        if M == 16:
            constellation = self._generate_16qam_constellation()
        elif M == 64:
            constellation = self._generate_64qam_constellation()
        elif M == 256:
            constellation = self._generate_256qam_constellation()
        else:
            raise ValueError(f"Unsupported QAM order: {M}")

        # Decode symbols to bits
        bits_per_symbol = int(np.log2(M))
        bits = []

        for symbol in complex_symbols:
            # Find closest constellation point
            distances = np.abs(constellation - symbol)
            closest_index = np.argmin(distances)

            # Convert index to bits
            bit_string = format(closest_index, f'0{bits_per_symbol}b')
            for bit_char in bit_string:
                bits.append(int(bit_char))

        return np.array(bits)

    def _generate_16qam_constellation(self):
        """Generate 16-QAM constellation for demodulation"""
        constellation = []
        for i in range(4):
            for q in range(4):
                real_part = 2*i - 3
                imag_part = 2*q - 3
                constellation.append(complex(real_part, imag_part))

        constellation = np.array(constellation)
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        return constellation

    def _generate_64qam_constellation(self):
        """Generate 64-QAM constellation for demodulation"""
        constellation = []
        for i in range(8):
            for q in range(8):
                real_part = 2*i - 7
                imag_part = 2*q - 7
                constellation.append(complex(real_part, imag_part))

        constellation = np.array(constellation)
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        return constellation

    def _generate_256qam_constellation(self):
        """Generate 256-QAM constellation for demodulation"""
        constellation = []
        for i in range(16):
            for q in range(16):
                real_part = 2*i - 15
                imag_part = 2*q - 15
                constellation.append(complex(real_part, imag_part))

        constellation = np.array(constellation)
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        return constellation


class AdvancedModulationClassifier:
    """Advanced classifier for extended digital modulations"""

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate

    def classify_modulation(self, iq_signal):
        """Classify modulation type from IQ signal"""
        features = self._extract_comprehensive_features(iq_signal)
        return self._advanced_classification(features)

    def _extract_comprehensive_features(self, iq_signal):
        """Extract comprehensive feature set"""
        # Amplitude features
        magnitude = np.abs(iq_signal)
        mag_var = np.var(magnitude)
        mag_kurt = self._kurtosis(magnitude)

        # Phase features
        phase = np.angle(iq_signal)
        phase_diff = np.diff(np.unwrap(phase))
        phase_var = np.var(phase_diff)

        # Higher-order moments
        c20 = np.mean(iq_signal**2)
        c21 = np.mean(iq_signal**2 * np.conj(iq_signal))
        c40 = np.mean(iq_signal**4) - 3*(np.mean(np.abs(iq_signal)**2))**2
        c41 = np.mean(iq_signal**3 * np.conj(iq_signal))
        c42 = np.mean(iq_signal**2 * (np.conj(iq_signal))**2) - (np.mean(np.abs(iq_signal)**2))**2

        # Spectral features
        spectrum = np.abs(fft(iq_signal))**2
        spectral_centroid = np.sum(np.arange(len(spectrum)) * spectrum) / np.sum(spectrum)

        # Constellation compactness
        constellation_compactness = self._constellation_compactness(iq_signal)

        return {
            'mag_var': mag_var,
            'mag_kurt': mag_kurt,
            'phase_var': phase_var,
            'c20': np.abs(c20),
            'c21': np.abs(c21),
            'c40': np.abs(c40),
            'c41': np.abs(c41), 
            'c42': np.abs(c42),
            'spectral_centroid': spectral_centroid,
            'constellation_compactness': constellation_compactness
        }

    def _kurtosis(self, x):
        """Calculate kurtosis"""
        mean_x = np.mean(x)
        std_x = np.std(x)
        if std_x == 0:
            return 0
        normalized = (x - mean_x) / std_x
        return np.mean(normalized**4) - 3

    def _constellation_compactness(self, iq_signal):
        """Measure constellation compactness"""
        # Cluster the constellation points
        from sklearn.cluster import KMeans

        try:
            # Try different numbers of clusters
            points = np.column_stack([np.real(iq_signal), np.imag(iq_signal)])
            best_score = float('inf')

            for n_clusters in [2, 4, 8, 16]:
                if len(iq_signal) >= n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(points)
                    score = kmeans.inertia_
                    if score < best_score:
                        best_score = score

            return best_score
        except:
            return 0

    def _advanced_classification(self, features):
        """Advanced rule-based classification"""
        mag_var = features['mag_var']
        c40 = features['c40']
        c42 = features['c42']
        constellation_compactness = features['constellation_compactness']

        # Classification logic based on cumulants and features
        if mag_var < 0.01:  # Constant envelope
            if c42 < 0.1:
                return "BPSK/DPSK"
            elif c42 < 0.5:
                return "QPSK/DQPSK"
            elif c40 < 0.1:
                return "MSK/GMSK"
            else:
                return "8PSK/CPFSK"
        else:  # Variable envelope
            if constellation_compactness < 50:
                if c42 > 0.8:
                    return "16QAM"
                elif c42 > 0.4:
                    return "64QAM" 
                else:
                    return "256QAM/1024QAM"
            else:
                return "APSK/ASK"
