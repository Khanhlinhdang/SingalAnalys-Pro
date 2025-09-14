
"""
Analog Modulation and Demodulation Module
Module chuyên xử lý điều chế tương tự: AM, FM, PM và các biến thể
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt, lfilter, find_peaks, hilbert
import warnings
warnings.filterwarnings('ignore')


class AnalogModulation:
    """Class for analog modulation techniques"""

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate
        self.carrier_freq = 100e3  # Default carrier frequency

    def am_modulate(self, message, carrier_freq=None, modulation_index=0.5, mod_type='dsb_lc'):
        """
        Amplitude Modulation
        mod_type: 'dsb_lc', 'dsb_sc', 'ssb_usb', 'ssb_lsb', 'vsb'
        """
        if carrier_freq:
            self.carrier_freq = carrier_freq

        t = np.arange(len(message)) / self.fs
        carrier = np.cos(2 * np.pi * self.carrier_freq * t)

        if mod_type == 'dsb_lc':
            # Double Sideband Large Carrier (Standard AM)
            modulated = (1 + modulation_index * message) * carrier

        elif mod_type == 'dsb_sc':
            # Double Sideband Suppressed Carrier
            modulated = message * carrier

        elif mod_type == 'ssb_usb':
            # Single Sideband Upper Sideband
            modulated = self._ssb_modulate(message, carrier, 'usb')

        elif mod_type == 'ssb_lsb':
            # Single Sideband Lower Sideband  
            modulated = self._ssb_modulate(message, carrier, 'lsb')

        elif mod_type == 'vsb':
            # Vestigial Sideband
            modulated = self._vsb_modulate(message, carrier)

        return modulated

    def _ssb_modulate(self, message, carrier, sideband='usb'):
        """Single Sideband modulation using Hilbert transform"""
        t = np.arange(len(message)) / self.fs

        # Generate quadrature carrier
        carrier_cos = carrier
        carrier_sin = np.sin(2 * np.pi * self.carrier_freq * t)

        # Hilbert transform of message signal
        message_hilbert = hilbert(message)
        message_real = np.real(message_hilbert)
        message_imag = np.imag(message_hilbert)

        if sideband == 'usb':
            # Upper sideband
            modulated = message_real * carrier_cos - message_imag * carrier_sin
        else:
            # Lower sideband
            modulated = message_real * carrier_cos + message_imag * carrier_sin

        return modulated

    def _vsb_modulate(self, message, carrier):
        """Vestigial Sideband modulation"""
        # Start with DSB-SC
        dsb_sc = message * carrier

        # Apply VSB filter (simplified)
        # In practice, this would be a carefully designed filter
        nyquist = self.fs / 2
        low_cutoff = self.carrier_freq / nyquist
        high_cutoff = (self.carrier_freq + 5000) / nyquist  # 5kHz vestige

        if high_cutoff < 1.0:
            b, a = butter(6, [low_cutoff, high_cutoff], btype='band')
            modulated = filtfilt(b, a, dsb_sc)
        else:
            modulated = dsb_sc  # Fallback

        return modulated

    def fm_modulate(self, message, carrier_freq=None, deviation=5000, mod_type='wbfm'):
        """
        Frequency Modulation
        mod_type: 'nbfm' (narrow-band), 'wbfm' (wide-band)
        """
        if carrier_freq:
            self.carrier_freq = carrier_freq

        t = np.arange(len(message)) / self.fs

        # Integrate message for FM
        message_integrated = np.cumsum(message) / self.fs

        # FM modulated signal
        instantaneous_freq = self.carrier_freq + deviation * message
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.fs

        if mod_type == 'nbfm':
            # Narrow-band FM approximation
            modulated = np.cos(2 * np.pi * self.carrier_freq * t) -                        (deviation / self.carrier_freq) * message * np.sin(2 * np.pi * self.carrier_freq * t)
        else:
            # Wide-band FM
            modulated = np.cos(phase)

        return modulated

    def pm_modulate(self, message, carrier_freq=None, deviation=np.pi/4):
        """Phase Modulation"""
        if carrier_freq:
            self.carrier_freq = carrier_freq

        t = np.arange(len(message)) / self.fs

        # PM modulated signal
        modulated = np.cos(2 * np.pi * self.carrier_freq * t + deviation * message)

        return modulated


class AnalogDemodulation:
    """Class for analog demodulation techniques"""

    def __init__(self, sample_rate=1e6, carrier_freq=100e3):
        self.fs = sample_rate
        self.carrier_freq = carrier_freq

    def am_demodulate(self, signal, mod_type='dsb_lc', carrier_freq=None):
        """
        Amplitude Demodulation
        """
        if carrier_freq:
            self.carrier_freq = carrier_freq

        if mod_type == 'dsb_lc':
            # Envelope detection for standard AM
            demodulated = self._envelope_detect(signal)
            # Remove DC component
            demodulated = demodulated - np.mean(demodulated)

        elif mod_type in ['dsb_sc', 'ssb_usb', 'ssb_lsb', 'vsb']:
            # Coherent detection
            demodulated = self._coherent_demodulate(signal)

        return demodulated

    def _envelope_detect(self, signal):
        """Envelope detection using Hilbert transform"""
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        return envelope

    def _coherent_demodulate(self, signal):
        """Coherent demodulation for suppressed carrier signals"""
        t = np.arange(len(signal)) / self.fs

        # Mix with local oscillator
        local_osc = 2 * np.cos(2 * np.pi * self.carrier_freq * t)
        mixed = signal * local_osc

        # Low-pass filter
        nyquist = self.fs / 2
        cutoff = 10000 / nyquist  # 10kHz cutoff
        b, a = butter(6, cutoff, btype='low')
        demodulated = filtfilt(b, a, mixed)

        return demodulated

    def fm_demodulate(self, signal, method='phase_diff'):
        """
        FM demodulation
        methods: 'phase_diff', 'quadrature', 'pll'
        """
        if method == 'phase_diff':
            demodulated = self._fm_phase_diff(signal)
        elif method == 'quadrature':
            demodulated = self._fm_quadrature(signal)
        elif method == 'pll':
            demodulated = self._fm_pll(signal)
        else:
            demodulated = self._fm_phase_diff(signal)

        return demodulated

    def _fm_phase_diff(self, signal):
        """FM demodulation using phase differentiation"""
        # Convert to complex baseband
        analytic_signal = hilbert(signal)

        # Extract instantaneous phase
        phase = np.angle(analytic_signal)

        # Unwrap phase to avoid discontinuities
        unwrapped_phase = np.unwrap(phase)

        # Differentiate to get frequency
        freq_deviation = np.diff(unwrapped_phase) * self.fs / (2 * np.pi)

        # Pad to maintain length
        demodulated = np.concatenate([[freq_deviation[0]], freq_deviation])

        return demodulated

    def _fm_quadrature(self, signal):
        """FM demodulation using quadrature detector"""
        # Delay line discriminator
        delayed = np.concatenate([[0], signal[:-1]])

        # Quadrature multiplication
        i_component = signal * delayed
        q_component = signal * np.concatenate([[0, 0], signal[:-2]])

        # Discriminator output
        discriminator = np.arctan2(q_component, i_component)
        demodulated = np.diff(np.unwrap(discriminator))
        demodulated = np.concatenate([[0], demodulated])

        return demodulated

    def _fm_pll(self, signal, loop_bandwidth=0.01):
        """FM demodulation using Phase-Locked Loop"""
        phase_estimate = 0
        frequency_estimate = self.carrier_freq
        loop_filter = 0

        demodulated = np.zeros(len(signal))

        for i, sample in enumerate(signal):
            # VCO output
            vco_output = np.cos(phase_estimate)

            # Phase detector
            phase_error = sample * vco_output

            # Loop filter (first-order)
            loop_filter += loop_bandwidth * phase_error

            # VCO control
            frequency_estimate += loop_filter
            phase_estimate += 2 * np.pi * frequency_estimate / self.fs

            # Output is frequency deviation
            demodulated[i] = frequency_estimate - self.carrier_freq

        return demodulated

    def pm_demodulate(self, signal):
        """Phase demodulation"""
        # PM demodulation is similar to FM but without integration
        analytic_signal = hilbert(signal)
        phase = np.angle(analytic_signal)

        # Remove carrier phase
        t = np.arange(len(signal)) / self.fs
        carrier_phase = 2 * np.pi * self.carrier_freq * t

        demodulated = np.unwrap(phase) - carrier_phase

        return demodulated


class PulseAnalogModulation:
    """Class for pulse analog modulation: PAM, PWM, PPM"""

    def __init__(self, sample_rate=1e6, pulse_rate=1000):
        self.fs = sample_rate
        self.pulse_rate = pulse_rate
        self.samples_per_pulse = int(self.fs / self.pulse_rate)

    def pam_modulate(self, message):
        """Pulse Amplitude Modulation"""
        # Sample the message at pulse rate
        decimation_factor = max(1, len(message) // (len(message) * self.pulse_rate // self.fs))
        sampled_message = message[::decimation_factor]

        # Generate PAM signal
        pam_signal = np.zeros(len(message))
        pulse_indices = np.arange(0, len(message), self.samples_per_pulse)

        for i, idx in enumerate(pulse_indices):
            if i < len(sampled_message) and idx < len(pam_signal):
                # Rectangular pulse with amplitude proportional to message
                pulse_width = min(self.samples_per_pulse // 4, len(pam_signal) - idx)
                pam_signal[idx:idx+pulse_width] = sampled_message[i]

        return pam_signal

    def pwm_modulate(self, message, max_width_ratio=0.9):
        """Pulse Width Modulation"""
        # Normalize message to [0, max_width_ratio]
        normalized_message = (message - np.min(message)) / (np.max(message) - np.min(message))
        normalized_message *= max_width_ratio

        # Sample at pulse rate
        decimation_factor = max(1, len(message) // (len(message) * self.pulse_rate // self.fs))
        sampled_widths = normalized_message[::decimation_factor]

        # Generate PWM signal
        pwm_signal = np.zeros(len(message))
        pulse_indices = np.arange(0, len(message), self.samples_per_pulse)

        for i, idx in enumerate(pulse_indices):
            if i < len(sampled_widths) and idx < len(pwm_signal):
                # Variable width pulse
                pulse_width = int(self.samples_per_pulse * sampled_widths[i])
                end_idx = min(idx + pulse_width, len(pwm_signal))
                pwm_signal[idx:end_idx] = 1.0

        return pwm_signal

    def ppm_modulate(self, message, max_delay_ratio=0.5):
        """Pulse Position Modulation"""
        # Normalize message to [0, max_delay_ratio]
        normalized_message = (message - np.min(message)) / (np.max(message) - np.min(message))
        normalized_message *= max_delay_ratio

        # Sample at pulse rate
        decimation_factor = max(1, len(message) // (len(message) * self.pulse_rate // self.fs))
        sampled_delays = normalized_message[::decimation_factor]

        # Generate PPM signal
        ppm_signal = np.zeros(len(message))
        pulse_indices = np.arange(0, len(message), self.samples_per_pulse)

        for i, idx in enumerate(pulse_indices):
            if i < len(sampled_delays) and idx < len(ppm_signal):
                # Position-modulated pulse
                delay = int(self.samples_per_pulse * sampled_delays[i])
                pulse_pos = idx + delay
                pulse_width = self.samples_per_pulse // 8

                end_idx = min(pulse_pos + pulse_width, len(ppm_signal))
                if pulse_pos < len(ppm_signal):
                    ppm_signal[pulse_pos:end_idx] = 1.0

        return ppm_signal

    def pam_demodulate(self, signal):
        """PAM demodulation using sample and hold"""
        pulse_indices = np.arange(self.samples_per_pulse//2, len(signal), self.samples_per_pulse)
        demodulated = signal[pulse_indices]

        # Interpolate to original length
        from scipy.interpolate import interp1d
        if len(demodulated) > 1:
            f = interp1d(np.arange(len(demodulated)), demodulated, 
                        kind='linear', fill_value='extrapolate')
            demodulated_full = f(np.linspace(0, len(demodulated)-1, len(signal)))
        else:
            demodulated_full = np.full(len(signal), demodulated[0] if len(demodulated) > 0 else 0)

        return demodulated_full

    def pwm_demodulate(self, signal, threshold=0.5):
        """PWM demodulation by measuring pulse widths"""
        # Threshold to binary
        binary_signal = signal > threshold

        # Find rising and falling edges
        edges = np.diff(binary_signal.astype(int))
        rising_edges = np.where(edges == 1)[0]
        falling_edges = np.where(edges == -1)[0]

        # Measure pulse widths
        pulse_widths = []
        for i, rise in enumerate(rising_edges):
            # Find corresponding falling edge
            fall_candidates = falling_edges[falling_edges > rise]
            if len(fall_candidates) > 0:
                fall = fall_candidates[0]
                width = (fall - rise) / self.samples_per_pulse
                pulse_widths.append(width)

        # Convert to message values
        if len(pulse_widths) > 0:
            demodulated = np.array(pulse_widths)
            # Interpolate to original length
            from scipy.interpolate import interp1d
            f = interp1d(np.arange(len(demodulated)), demodulated, 
                        kind='linear', fill_value='extrapolate')
            demodulated_full = f(np.linspace(0, len(demodulated)-1, len(signal)))
        else:
            demodulated_full = np.zeros(len(signal))

        return demodulated_full

    def ppm_demodulate(self, signal, threshold=0.5):
        """PPM demodulation by measuring pulse positions"""
        # Threshold to binary
        binary_signal = signal > threshold

        # Find pulse positions (rising edges)
        edges = np.diff(binary_signal.astype(int))
        rising_edges = np.where(edges == 1)[0]

        # Calculate positions within each pulse period
        pulse_positions = []
        expected_times = np.arange(0, len(signal), self.samples_per_pulse)

        for expected_time in expected_times:
            # Find nearest rising edge
            distances = np.abs(rising_edges - expected_time)
            if len(distances) > 0:
                nearest_edge = rising_edges[np.argmin(distances)]
                position = (nearest_edge - expected_time) / self.samples_per_pulse
                pulse_positions.append(position)

        # Convert to message values
        if len(pulse_positions) > 0:
            demodulated = np.array(pulse_positions)
            # Interpolate to original length
            from scipy.interpolate import interp1d
            f = interp1d(np.arange(len(demodulated)), demodulated, 
                        kind='linear', fill_value='extrapolate')
            demodulated_full = f(np.linspace(0, len(demodulated)-1, len(signal)))
        else:
            demodulated_full = np.zeros(len(signal))

        return demodulated_full


class AnalogModulationClassifier:
    """Classifier for analog modulation types"""

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate

    def classify_analog_modulation(self, signal):
        """Classify analog modulation type"""
        features = self._extract_analog_features(signal)
        return self._classify_from_features(features)

    def _extract_analog_features(self, signal):
        """Extract features for analog modulation classification"""
        # Envelope variations
        envelope = np.abs(hilbert(signal))
        envelope_var = np.var(envelope)
        envelope_mean = np.mean(envelope)
        envelope_cv = envelope_var / (envelope_mean + 1e-12)  # Coefficient of variation

        # Frequency characteristics
        freqs, psd = signal.welch(signal, fs=self.fs)
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))

        # Phase characteristics
        analytic_signal = hilbert(signal)
        inst_phase = np.angle(analytic_signal)
        phase_diff = np.diff(np.unwrap(inst_phase))
        phase_var = np.var(phase_diff)

        # Instantaneous frequency
        inst_freq = self.fs * phase_diff / (2 * np.pi)
        freq_var = np.var(inst_freq)

        return {
            'envelope_cv': envelope_cv,
            'envelope_var': envelope_var,
            'spectral_spread': spectral_spread,
            'phase_var': phase_var,
            'freq_var': freq_var,
            'spectral_centroid': spectral_centroid
        }

    def _classify_from_features(self, features):
        """Rule-based classification from features"""
        env_cv = features['envelope_cv']
        freq_var = features['freq_var']
        phase_var = features['phase_var']

        # Classification rules based on signal characteristics
        if env_cv > 0.5:
            # High envelope variation indicates AM
            if env_cv > 1.0:
                return "AM (DSB-LC)"
            else:
                return "AM (DSB-SC/SSB/VSB)"

        elif freq_var > 1000:  # High frequency variation
            # Could be FM or PM
            if phase_var > 1.0:
                return "FM (Wide-band)"
            else:
                return "FM (Narrow-band)"

        elif phase_var > 0.5:
            # Phase modulation
            return "PM"

        else:
            # Low variation in all parameters
            return "Unmodulated Carrier"
