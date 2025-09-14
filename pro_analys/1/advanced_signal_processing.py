
"""
Advanced Signal Processing Module
Các thuật toán tiên tiến cho phân tích tín hiệu và demodulation
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class AdvancedSignalProcessor:
    """Advanced signal processing algorithms"""

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate
        self.trained_classifier = None

    def carrier_frequency_estimation(self, iq_data, method='fft_peak'):
        """
        Estimate carrier frequency offset
        Methods: 'fft_peak', 'autocorr', 'cyclic'
        """
        if method == 'fft_peak':
            return self._fft_peak_estimation(iq_data)
        elif method == 'autocorr':
            return self._autocorr_estimation(iq_data)
        elif method == 'cyclic':
            return self._cyclic_estimation(iq_data)

    def _fft_peak_estimation(self, iq_data):
        """FFT-based frequency estimation"""
        spectrum = np.abs(fft(iq_data))
        peak_idx = np.argmax(spectrum)
        freqs = fftfreq(len(iq_data), 1/self.fs)
        return freqs[peak_idx]

    def _autocorr_estimation(self, iq_data):
        """Autocorrelation-based estimation"""
        autocorr = np.correlate(iq_data, iq_data, mode='full')
        peak_idx = np.argmax(np.abs(autocorr[len(autocorr)//2:]))
        if peak_idx > 0:
            return self.fs / peak_idx
        return 0

    def _cyclic_estimation(self, iq_data):
        """Cyclostationary-based estimation"""
        # Simplified cyclic frequency detection
        x4 = iq_data ** 4  # 4th power for PSK signals
        spectrum = np.abs(fft(x4))
        peak_idx = np.argmax(spectrum)
        freqs = fftfreq(len(x4), 1/self.fs)
        return freqs[peak_idx] / 4  # Divide by 4 for PSK

    def adaptive_threshold_detection(self, psd_db, false_alarm_rate=0.01):
        """
        Adaptive threshold for signal detection using CFAR
        """
        # Cell-Averaging CFAR
        n_guard = 2
        n_train = 10

        threshold = np.zeros_like(psd_db)

        for i in range(len(psd_db)):
            # Define training cells (avoiding guard cells)
            train_start = max(0, i - n_train - n_guard)
            train_end_left = max(0, i - n_guard)
            train_start_right = min(len(psd_db), i + n_guard + 1)
            train_end = min(len(psd_db), i + n_train + n_guard + 1)

            # Calculate noise floor from training cells
            train_cells = np.concatenate([
                psd_db[train_start:train_end_left],
                psd_db[train_start_right:train_end]
            ])

            if len(train_cells) > 0:
                noise_floor = np.mean(train_cells)
                noise_var = np.var(train_cells)
                # CFAR threshold
                k = -np.log(false_alarm_rate)  # Threshold factor
                threshold[i] = noise_floor + k * np.sqrt(noise_var)
            else:
                threshold[i] = np.mean(psd_db)

        # Detect signals above threshold
        detections = psd_db > threshold
        return detections, threshold

    def enhanced_modulation_classification(self, iq_data):
        """
        Enhanced automatic modulation classification using multiple features
        """
        features = self._extract_features(iq_data)

        if self.trained_classifier is None:
            # Use rule-based classification if no trained classifier
            return self._rule_based_classification(features)
        else:
            # Use trained ML classifier
            return self.trained_classifier.predict([features])[0]

    def _extract_features(self, iq_data):
        """Extract comprehensive feature set for modulation classification"""
        # Amplitude features
        magnitude = np.abs(iq_data)
        mag_mean = np.mean(magnitude)
        mag_var = np.var(magnitude)
        mag_std = np.std(magnitude)

        # Phase features  
        phase = np.angle(iq_data)
        phase_diff = np.diff(np.unwrap(phase))
        phase_var = np.var(phase_diff)
        phase_std = np.std(phase_diff)

        # Higher-order cumulants (C40, C41, C42)
        c40 = self._cumulant_40(iq_data)
        c41 = self._cumulant_41(iq_data) 
        c42 = self._cumulant_42(iq_data)

        # Spectral features
        psd = np.abs(fft(iq_data)) ** 2
        spectral_centroid = np.sum(np.arange(len(psd)) * psd) / np.sum(psd)
        spectral_spread = np.sqrt(np.sum(((np.arange(len(psd)) - spectral_centroid) ** 2) * psd) / np.sum(psd))

        # Peak-to-average power ratio
        papr = np.max(magnitude ** 2) / np.mean(magnitude ** 2)

        return [mag_var, phase_var, c40, c41, c42, spectral_centroid, 
                spectral_spread, papr, mag_std, phase_std]

    def _cumulant_40(self, x):
        """4th order cumulant C40"""
        x_centered = x - np.mean(x)
        c40 = np.mean(x_centered ** 4) - 3 * (np.mean(np.abs(x_centered) ** 2)) ** 2
        return np.abs(c40)

    def _cumulant_41(self, x):
        """4th order cumulant C41"""  
        x_centered = x - np.mean(x)
        c41 = np.mean(x_centered ** 3 * np.conj(x_centered))
        return np.abs(c41)

    def _cumulant_42(self, x):
        """4th order cumulant C42"""
        x_centered = x - np.mean(x)
        c42 = np.mean(x_centered ** 2 * np.conj(x_centered) ** 2) - (np.mean(np.abs(x_centered) ** 2)) ** 2
        return np.abs(c42)

    def _rule_based_classification(self, features):
        """Rule-based modulation classification"""
        mag_var, phase_var, c40, c41, c42, _, _, papr, _, _ = features

        # Classification rules based on cumulants and other features
        if mag_var < 0.05:  # Constant envelope
            if phase_var > 2.0:
                if c42 < 0.1:
                    return "BPSK"
                elif c42 < 0.5:
                    return "QPSK" 
                else:
                    return "8PSK"
            else:
                return "CW/Unmodulated"
        else:  # Variable envelope
            if papr > 3.0:
                return "QAM"
            else:
                return "ASK/PAM"

    def train_classifier(self, training_data, labels):
        """Train ML classifier for modulation recognition"""
        features_matrix = []
        for iq_samples in training_data:
            features = self._extract_features(iq_samples)
            features_matrix.append(features)

        features_matrix = np.array(features_matrix)

        # Use Random Forest classifier
        self.trained_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.trained_classifier.fit(features_matrix, labels)

        return self.trained_classifier

    def symbol_timing_recovery(self, iq_data, symbol_rate, method='gardner'):
        """
        Symbol timing recovery algorithms
        Methods: 'gardner', 'mueller_muller', 'early_late'
        """
        if method == 'gardner':
            return self._gardner_timing_recovery(iq_data, symbol_rate)
        elif method == 'mueller_muller':
            return self._mueller_muller_recovery(iq_data, symbol_rate)
        elif method == 'early_late':
            return self._early_late_recovery(iq_data, symbol_rate)

    def _gardner_timing_recovery(self, iq_data, symbol_rate):
        """Gardner timing recovery algorithm"""
        samples_per_symbol = int(self.fs / symbol_rate)

        # Interpolator (cubic spline)
        from scipy.interpolate import interp1d

        # Initialize timing loop
        timing_error = 0
        loop_filter = 0
        alpha = 0.1  # Loop bandwidth

        recovered_symbols = []
        sample_times = []

        mu = 0.5  # Fractional timing offset

        for k in range(1, len(iq_data) - samples_per_symbol):
            # Sample at symbol times
            if mu >= 1.0:
                # Get samples for Gardner algorithm
                current_idx = int(k + mu)
                if current_idx + 1 < len(iq_data):
                    y_k = iq_data[current_idx]
                    y_k_half = iq_data[current_idx - samples_per_symbol // 2]
                    y_k_minus1 = iq_data[current_idx - samples_per_symbol]

                    recovered_symbols.append(y_k)
                    sample_times.append(current_idx)

                    # Gardner timing error detector
                    error = np.real((y_k - y_k_minus1) * np.conj(y_k_half))

                    # Loop filter
                    loop_filter = loop_filter + alpha * error
                    mu = mu - loop_filter

                mu += 1.0 / samples_per_symbol
            else:
                mu += 1.0 / samples_per_symbol

        return np.array(recovered_symbols)

    def _mueller_muller_recovery(self, iq_data, symbol_rate):
        """Mueller and Muller timing recovery"""
        # Simplified implementation
        samples_per_symbol = int(self.fs / symbol_rate)
        decimated = iq_data[::samples_per_symbol]
        return decimated

    def _early_late_recovery(self, iq_data, symbol_rate):
        """Early-Late gate timing recovery"""
        # Simplified implementation
        samples_per_symbol = int(self.fs / symbol_rate)
        decimated = iq_data[::samples_per_symbol]
        return decimated

    def frequency_offset_correction(self, iq_data, freq_offset):
        """Correct for frequency offset"""
        t = np.arange(len(iq_data)) / self.fs
        correction = np.exp(-1j * 2 * np.pi * freq_offset * t)
        return iq_data * correction

    def phase_locked_loop(self, iq_data, loop_bandwidth=0.01):
        """
        Phase-locked loop for carrier phase recovery
        """
        phase_estimate = 0
        loop_filter = 0

        corrected_signal = np.zeros_like(iq_data)

        for i, sample in enumerate(iq_data):
            # Apply current phase correction
            corrected_sample = sample * np.exp(-1j * phase_estimate)
            corrected_signal[i] = corrected_sample

            # Phase error detector (simplified)
            phase_error = np.angle(corrected_sample)
            if phase_error > np.pi:
                phase_error -= 2 * np.pi
            elif phase_error < -np.pi:
                phase_error += 2 * np.pi

            # Loop filter (first-order)
            loop_filter += loop_bandwidth * phase_error
            phase_estimate += loop_filter

            # Keep phase in range [-π, π]
            if phase_estimate > np.pi:
                phase_estimate -= 2 * np.pi
            elif phase_estimate < -np.pi:
                phase_estimate += 2 * np.pi

        return corrected_signal

    def constellation_clustering(self, symbols, n_clusters='auto'):
        """
        Cluster constellation points to identify modulation type
        """
        if n_clusters == 'auto':
            # Try different number of clusters and use elbow method
            possible_clusters = [2, 4, 8, 16]
            inertias = []

            symbol_points = np.column_stack([np.real(symbols), np.imag(symbols)])

            for k in possible_clusters:
                if len(symbols) >= k:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(symbol_points)
                    inertias.append(kmeans.inertia_)
                else:
                    inertias.append(float('inf'))

            # Simple elbow method
            if len(inertias) > 1:
                diffs = np.diff(inertias)
                elbow_idx = np.argmax(diffs) if len(diffs) > 0 else 0
                n_clusters = possible_clusters[elbow_idx]
            else:
                n_clusters = 4  # Default

        # Perform clustering
        symbol_points = np.column_stack([np.real(symbols), np.imag(symbols)])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(symbol_points)
        centers = kmeans.cluster_centers_

        return labels, centers, n_clusters


class SpectrumScanner:
    """Spectrum scanning and signal detection"""

    def __init__(self, usrp_interface):
        self.usrp = usrp_interface
        self.scan_results = []

    def frequency_sweep(self, start_freq, end_freq, step_size=1e6, dwell_time=0.1):
        """
        Perform frequency sweep scan
        """
        frequencies = np.arange(start_freq, end_freq + step_size, step_size)
        scan_results = []

        for freq in frequencies:
            # Tune to frequency
            self.usrp.set_frequency(freq)
            time.sleep(dwell_time)

            # Get samples and compute spectrum
            if hasattr(self.usrp, 'get_samples'):
                samples = self.usrp.get_samples(1024)

                # Compute power spectral density
                f, psd = signal.welch(samples, fs=self.usrp.sample_rate)
                peak_power = np.max(psd)
                peak_freq = freq + f[np.argmax(psd)]

                scan_results.append({
                    'center_freq': freq,
                    'peak_freq': peak_freq,
                    'peak_power': peak_power,
                    'psd': psd,
                    'frequencies': freq + f
                })

        self.scan_results = scan_results
        return scan_results

    def detect_signals(self, threshold_db=-60):
        """Detect signals in scan results"""
        detected_signals = []

        for result in self.scan_results:
            psd_db = 10 * np.log10(result['psd'])

            # Find peaks above threshold
            peaks, properties = signal.find_peaks(
                psd_db, height=threshold_db, prominence=5
            )

            for peak in peaks:
                detected_signals.append({
                    'frequency': result['frequencies'][peak],
                    'power_db': psd_db[peak],
                    'bandwidth': self._estimate_bandwidth(result['psd'], peak),
                    'center_freq': result['center_freq']
                })

        return detected_signals

    def _estimate_bandwidth(self, psd, peak_idx, threshold_db=3):
        """Estimate signal bandwidth"""
        peak_power = psd[peak_idx]
        threshold = peak_power / (10 ** (threshold_db / 10))

        # Find -3dB points
        left_idx = peak_idx
        right_idx = peak_idx

        # Search left
        while left_idx > 0 and psd[left_idx] > threshold:
            left_idx -= 1

        # Search right  
        while right_idx < len(psd) - 1 and psd[right_idx] > threshold:
            right_idx += 1

        bandwidth = (right_idx - left_idx) * (self.usrp.sample_rate / len(psd))
        return bandwidth
