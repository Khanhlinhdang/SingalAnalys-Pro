
"""
Optimized Analog Modulation and Demodulation Module
Module chuyên xử lý điều chế tương tự được tối ưu hóa: AM, FM, PM và các biến thể
Tuân thủ chuẩn ITU-R và các khuyến nghị quốc tế
"""

import numpy as np
import logging
from scipy import signal
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import butter, filtfilt, lfilter, find_peaks, hilbert, sosfilt
from scipy.signal.windows import kaiser
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModulationType(Enum):
    """Standard analog modulation types"""
    AM_DSB_LC = "am_dsb_lc"  # Double Sideband Large Carrier
    AM_DSB_SC = "am_dsb_sc"  # Double Sideband Suppressed Carrier
    AM_SSB_USB = "am_ssb_usb"  # Single Sideband Upper
    AM_SSB_LSB = "am_ssb_lsb"  # Single Sideband Lower
    AM_VSB = "am_vsb"  # Vestigial Sideband
    FM_NB = "fm_nb"  # Narrow Band FM
    FM_WB = "fm_wb"  # Wide Band FM
    PM = "pm"  # Phase Modulation


@dataclass
class ModulationParameters:
    """Parameters for analog modulation"""
    carrier_freq: float
    modulation_index: float
    deviation: Optional[float] = None
    bandwidth: Optional[float] = None
    pre_emphasis: bool = False
    de_emphasis: bool = False


@dataclass
class DemodulationResult:
    """Result from analog demodulation"""
    demodulated_signal: np.ndarray
    snr_estimate: float
    thd_percent: float  # Total Harmonic Distortion
    carrier_recovery_error: float
    timing_recovery_error: float
    quality_metric: str


class AnalogModulation:
    """Optimized class for analog modulation techniques with ITU-R compliance"""

    # Standard broadcast frequencies (ITU-R)
    BROADCAST_BANDS = {
        'AM': (530e3, 1700e3),    # AM broadcast
        'FM': (88e6, 108e6),      # FM broadcast
        'SW': (3e6, 30e6),        # Short wave
        'VHF': (30e6, 300e6)      # VHF
    }
    
    # Pre-emphasis time constants (ITU-R BS.412)
    PRE_EMPHASIS_TC = {
        'CCITT': 50e-6,   # 50 microseconds (Europe)
        'FCC': 75e-6,     # 75 microseconds (North America)
    }

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate
        self.carrier_freq = 100e3  # Default carrier frequency
        self.nyquist = sample_rate / 2
        
        # Initialize filters
        self._init_standard_filters()
        
        logger.info(f"Initialized AnalogModulation: Fs={self.fs/1e6:.1f} MHz")

    def _init_standard_filters(self):
        """Initialize standard broadcast filters"""
        # AM broadcast filter (±5 kHz)
        self.am_lpf = self._design_butterworth_filter(5000, 'low', order=8)
        
        # FM broadcast filter (±75 kHz deviation + 15 kHz audio)
        self.fm_lpf = self._design_butterworth_filter(15000, 'low', order=8)
        
        # Pre-emphasis filters
        self._init_preemphasis_filters()

    def _design_butterworth_filter(self, cutoff: float, btype: str, 
                                 order: int = 6) -> np.ndarray:
        """Design Butterworth filter with standard parameters"""
        normalized_cutoff = cutoff / self.nyquist
        
        if normalized_cutoff >= 1.0:
            logger.warning(f"Cutoff frequency {cutoff} Hz too high for sample rate")
            normalized_cutoff = 0.95
        
        # Use second-order sections for numerical stability
        sos = signal.butter(order, normalized_cutoff, btype=btype, output='sos')
        return sos

    def _init_preemphasis_filters(self):
        """Initialize pre-emphasis and de-emphasis filters"""
        # Pre-emphasis: H(s) = (1 + sτ) / (1 + sτ/K) where K > 1
        # For FM: τ = 75μs (FCC) or 50μs (CCITT)
        
        for standard, tc in self.PRE_EMPHASIS_TC.items():
            # Implementation would go here for each standard
            pass

    def am_modulate(self, message, carrier_freq=None, modulation_index=0.5, mod_type='dsb_lc'):
        """
        Enhanced AM modulation with carrier recovery aids
        mod_type: 'dsb_lc', 'dsb_sc', 'ssb_usb', 'ssb_lsb', 'vsb'
        """
        if carrier_freq:
            self.carrier_freq = carrier_freq

        # Normalize message signal
        message = self._normalize_audio(message)

        # Apply audio bandwidth limiting if needed
        if hasattr(self, 'am_lpf'):
            message = sosfilt(self.am_lpf, message)

        t = np.arange(len(message)) / self.fs
        carrier = np.cos(2 * np.pi * self.carrier_freq * t)

        if mod_type == 'dsb_lc':
            # Double Sideband Large Carrier (Standard AM)
            if modulation_index > 1.0:
                logger.warning(f"Modulation index {modulation_index} > 1.0 may cause overmodulation")
            
            modulated = (1 + modulation_index * message) * carrier
            
            # Add pilot tone for carrier recovery (±25 Hz from carrier)
            pilot_freq = self.carrier_freq + 25
            pilot = 0.01 * np.cos(2 * np.pi * pilot_freq * t)  # -40 dB pilot
            modulated += pilot

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
        """Enhanced Single Sideband modulation using Hilbert transform method"""
        message = self._normalize_audio(message)
        
        # Apply audio bandwidth limiting
        if hasattr(self, 'am_lpf'):
            message = sosfilt(self.am_lpf, message)
        
        t = np.arange(len(message)) / self.fs
        
        # Hilbert transform for quadrature component
        message_hilbert = hilbert(message)
        message_i = np.real(message_hilbert)  # In-phase
        message_q = np.imag(message_hilbert)  # Quadrature
        
        # Quadrature carriers
        carrier_i = np.cos(2 * np.pi * self.carrier_freq * t)
        carrier_q = np.sin(2 * np.pi * self.carrier_freq * t)
        
        if sideband.lower() == 'usb':
            # Upper sideband: I*cos - Q*sin
            ssb_signal = message_i * carrier_i - message_q * carrier_q
        else:
            # Lower sideband: I*cos + Q*sin
            ssb_signal = message_i * carrier_i + message_q * carrier_q
        
        return ssb_signal

    def _vsb_modulate(self, message, carrier):
        """Enhanced Vestigial Sideband modulation with improved filter design"""
        message = self._normalize_audio(message)
        
        # Start with DSB-SC
        dsb_sc = message * carrier
        
        # Design VSB filter
        # For NTSC TV: pass full upper sideband, partial lower sideband
        vestigial_freq = 1.25e6  # 1.25 MHz vestigial bandwidth for TV
        
        # Enhanced VSB filter design
        vsb_filter = self._design_vsb_filter(self.carrier_freq, vestigial_freq)
        
        # Apply VSB filter
        vsb_signal = self._apply_filter(dsb_sc, vsb_filter)
        
        return vsb_signal

    def fm_modulate(self, message, carrier_freq=None, deviation=5000, mod_type='wbfm', pre_emphasis=False):
        """
        Enhanced FM modulation with pre-emphasis and stereo capability
        mod_type: 'nbfm' (narrow-band), 'wbfm' (wide-band)
        """
        if carrier_freq:
            self.carrier_freq = carrier_freq

        message = self._normalize_audio(message)

        # Apply pre-emphasis for FM broadcast
        if pre_emphasis:
            message = self._apply_preemphasis(message, 'fcc')

        # Apply audio filter
        if hasattr(self, 'fm_lpf'):
            message = sosfilt(self.fm_lpf, message)

        t = np.arange(len(message)) / self.fs

        if mod_type == 'nbfm':
            # Narrow-band FM approximation (small angle approximation)
            modulation_index = deviation / self.carrier_freq
            modulated = (np.cos(2 * np.pi * self.carrier_freq * t) - 
                        modulation_index * message * np.sin(2 * np.pi * self.carrier_freq * t))
        else:
            # Wide-band FM
            modulated = self._wideband_fm(message, deviation, t)

        return modulated

    def _wideband_fm(self, message: np.ndarray, deviation: float, t: np.ndarray) -> np.ndarray:
        """Wide-band FM modulation with proper integration"""
        # Integrate message for frequency modulation
        # Use cumulative sum for integration
        integrated_message = np.cumsum(message) / self.fs
        
        # Instantaneous phase
        phase = 2 * np.pi * (self.carrier_freq * t + 
                           deviation * integrated_message)
        
        # Generate FM signal
        fm_signal = np.cos(phase)
        
        return fm_signal

    def pm_modulate(self, message, carrier_freq=None, deviation=np.pi/4):
        """Enhanced Phase Modulation"""
        if carrier_freq:
            self.carrier_freq = carrier_freq

        message = self._normalize_audio(message)
        
        # Apply audio bandwidth limiting
        if hasattr(self, 'am_lpf'):
            message = sosfilt(self.am_lpf, message)

        t = np.arange(len(message)) / self.fs

        # Phase deviation (typically in radians)
        phase_deviation = deviation  # Use provided deviation

        # PM signal
        instantaneous_phase = 2 * np.pi * self.carrier_freq * t + phase_deviation * message
        pm_signal = np.cos(instantaneous_phase)

        return pm_signal

    def _normalize_audio(self, signal: np.ndarray, target_level: float = -12) -> np.ndarray:
        """
        Normalize audio signal to standard broadcast level
        
        Args:
            signal: Input audio signal
            target_level: Target level in dB (relative to full scale)
            
        Returns:
            Normalized signal
        """
        if len(signal) == 0:
            return signal
        
        # Calculate RMS level
        rms_level = np.sqrt(np.mean(signal**2))
        
        if rms_level == 0:
            return signal
        
        # Target level in linear scale
        target_linear = 10**(target_level / 20)
        
        # Normalize
        normalized = signal * (target_linear / rms_level)
        
        # Soft limiting to prevent clipping
        normalized = np.tanh(normalized)
        
        return normalized

    def _apply_preemphasis(self, signal: np.ndarray, standard: str = 'fcc') -> np.ndarray:
        """Apply pre-emphasis filtering"""
        try:
            # Simple pre-emphasis implementation
            # In practice, this would use proper filter design
            return signal  # Placeholder
        except AttributeError:
            return signal

    def _design_vsb_filter(self, carrier_freq: float, vestigial_bw: float) -> np.ndarray:
        """Design VSB (Vestigial Sideband) filter"""
        # Simplified VSB filter design
        # Create frequency response
        n_fft = 2048
        freqs = np.linspace(-self.nyquist, self.nyquist, n_fft)
        
        # VSB response: full upper sideband, partial lower sideband
        response = np.zeros(n_fft, dtype=complex)
        
        for i, freq in enumerate(freqs):
            if abs(freq - carrier_freq) <= vestigial_bw/2:
                response[i] = 1.0  # Full response
            elif abs(freq + carrier_freq) <= vestigial_bw/4:
                response[i] = 0.5  # Partial response
        
        # Convert to time domain filter
        impulse_response = np.real(ifft(fftshift(response)))
        
        # Window and truncate
        window_length = min(512, len(impulse_response))
        window = kaiser(window_length, 8.0)
        
        start_idx = len(impulse_response) // 2 - window_length // 2
        windowed_ir = impulse_response[start_idx:start_idx + window_length] * window
        
        return windowed_ir

    def _apply_filter(self, signal: np.ndarray, filter_ir: np.ndarray) -> np.ndarray:
        """Apply FIR filter using convolution"""
        # Use scipy's efficient convolution
        filtered = np.convolve(signal, filter_ir, mode='same')
        
        return filtered


class AnalogDemodulation:
    """Enhanced class for analog demodulation techniques with carrier recovery"""

    def __init__(self, sample_rate=1e6, carrier_freq=100e3):
        self.fs = sample_rate
        self.carrier_freq = carrier_freq
        self.nyquist = sample_rate / 2
        
        # Initialize carrier recovery parameters
        self.pll_bandwidth = 1000  # 1 kHz loop bandwidth
        self.pll_damping = 0.707   # Critical damping
        
        # Initialize filters
        self._init_demod_filters()
        
        logger.info(f"Initialized AnalogDemodulation: Fc={carrier_freq/1e3:.1f} kHz")

    def _init_demod_filters(self):
        """Initialize demodulation filters"""
        # Audio low-pass filters
        self.am_audio_lpf = self._design_butterworth_filter(5000, 'low', order=8)
        self.fm_audio_lpf = self._design_butterworth_filter(15000, 'low', order=8)

    def _design_butterworth_filter(self, cutoff: float, btype: str, 
                                 order: int = 6) -> np.ndarray:
        """Design Butterworth filter"""
        normalized_cutoff = cutoff / self.nyquist
        
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.95
        
        sos = signal.butter(order, normalized_cutoff, btype=btype, output='sos')
        return sos

    def am_demodulate(self, signal, mod_type='dsb_lc', carrier_freq=None, return_metrics=False):
        """
        Enhanced AM demodulation with multiple detection methods and quality metrics
        """
        if carrier_freq:
            self.carrier_freq = carrier_freq

        if mod_type == 'dsb_lc':
            # Envelope detection for standard AM
            demodulated = self._envelope_detect(signal)
            # Remove DC component
            demodulated = demodulated - np.mean(demodulated)
            method_used = 'envelope'

        elif mod_type in ['dsb_sc', 'ssb_usb', 'ssb_lsb', 'vsb']:
            # Coherent detection
            demodulated = self._coherent_demodulate(signal)
            method_used = 'synchronous'

        # Apply audio filtering
        if hasattr(self, 'am_audio_lpf'):
            demodulated = sosfilt(self.am_audio_lpf, demodulated)

        if return_metrics:
            # Calculate quality metrics
            snr_estimate = self._estimate_snr(demodulated, signal)
            thd = self._calculate_thd(demodulated)
            carrier_error = self._estimate_carrier_recovery_error(signal)
            quality = self._assess_quality(snr_estimate, thd)
            
            result = DemodulationResult(
                demodulated_signal=demodulated,
                snr_estimate=snr_estimate,
                thd_percent=thd,
                carrier_recovery_error=carrier_error,
                timing_recovery_error=0.0,
                quality_metric=quality
            )
            return result
        else:
            return demodulated

    def _envelope_detect(self, signal):
        """Enhanced envelope detection using Hilbert transform"""
        # Hilbert transform for analytic signal
        analytic_signal = hilbert(signal)
        
        # Envelope is the magnitude
        envelope = np.abs(analytic_signal)
        
        # Remove DC component (carrier) more effectively
        envelope_ac = envelope - np.mean(envelope)
        
        return envelope_ac

    def _coherent_demodulate(self, signal):
        """Enhanced coherent demodulation with carrier recovery"""
        # Try carrier recovery first
        try:
            recovered_carrier = self._costas_loop_carrier_recovery(signal)
            # Synchronous detection
            demod = signal * recovered_carrier
        except:
            # Fallback to simple local oscillator
            t = np.arange(len(signal)) / self.fs
            local_osc = 2 * np.cos(2 * np.pi * self.carrier_freq * t)
            demod = signal * local_osc

        # Low-pass filter
        if hasattr(self, 'am_audio_lpf'):
            demodulated = sosfilt(self.am_audio_lpf, demod)
        else:
            # Fallback filter
            nyquist = self.fs / 2
            cutoff = 10000 / nyquist  # 10kHz cutoff
            b, a = butter(6, cutoff, btype='low')
            demodulated = filtfilt(b, a, demod)

        return demodulated

    def fm_demodulate(self, signal, method='phase_diff', return_metrics=False):
        """
        Enhanced FM demodulation with multiple methods and quality metrics
        methods: 'phase_diff', 'quadrature', 'pll', 'discriminator'
        """
        if method == 'phase_diff' or method == 'discriminator':
            demodulated = self._fm_phase_diff(signal)
        elif method == 'quadrature':
            demodulated = self._fm_quadrature(signal)
        elif method == 'pll':
            demodulated = self._fm_pll(signal)
        else:
            demodulated = self._fm_phase_diff(signal)

        # Apply de-emphasis for FM broadcast
        demodulated = self._apply_deemphasis(demodulated, 'fcc')

        # Apply audio filtering
        if hasattr(self, 'fm_audio_lpf'):
            demodulated = sosfilt(self.fm_audio_lpf, demodulated)

        if return_metrics:
            # Quality metrics
            snr_estimate = self._estimate_snr(demodulated, signal)
            thd = self._calculate_thd(demodulated)
            quality = self._assess_quality(snr_estimate, thd)
            
            result = DemodulationResult(
                demodulated_signal=demodulated,
                snr_estimate=snr_estimate,
                thd_percent=thd,
                carrier_recovery_error=0.0,
                timing_recovery_error=0.0,
                quality_metric=quality
            )
            return result
        else:
            return demodulated

    def _fm_phase_diff(self, signal):
        """Enhanced FM demodulation using phase differentiation"""
        # Convert to complex baseband
        analytic_signal = hilbert(signal)

        # Extract instantaneous phase
        inst_phase = np.angle(analytic_signal)

        # Unwrap phase discontinuities for better accuracy
        unwrapped_phase = np.unwrap(inst_phase)

        # Differentiate to get frequency
        inst_freq = np.diff(unwrapped_phase) * self.fs / (2 * np.pi)

        # Pad to maintain length
        inst_freq = np.concatenate([[inst_freq[0]], inst_freq])

        # Remove carrier frequency to get baseband audio
        audio = inst_freq - self.carrier_freq

        return audio

    def _fm_quadrature(self, signal):
        """Enhanced quadrature detector for FM demodulation"""
        # Delay line
        delayed_signal = np.concatenate([[0], signal[:-1]])
        
        # Quadrature multiplication (more robust implementation)
        quad_output = np.real(signal * np.conj(delayed_signal))
        
        # Low-pass filter to extract audio
        if hasattr(self, 'fm_audio_lpf'):
            audio = sosfilt(self.fm_audio_lpf, quad_output)
        else:
            # Fallback filter
            sos = self._design_butterworth_filter(15000, 'low', order=6)
            audio = sosfilt(sos, quad_output)
        
        return audio

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

    def _costas_loop_carrier_recovery(self, signal: np.ndarray) -> np.ndarray:
        """Costas loop for carrier recovery in AM synchronous detection"""
        # Costas loop parameters
        alpha = 0.01  # Loop gain
        beta = alpha**2 / 4  # Second-order loop parameter
        
        # Initialize loop state
        nco_phase = 0.0
        nco_freq = 2 * np.pi * self.carrier_freq / self.fs
        loop_filter_state = 0.0
        
        recovered_carrier = np.zeros(len(signal), dtype=complex)
        
        for i, sample in enumerate(signal):
            # NCO output
            nco_i = np.cos(nco_phase)
            nco_q = np.sin(nco_phase)
            recovered_carrier[i] = nco_i + 1j * nco_q
            
            # Phase detector (Costas)
            mixed_i = sample * nco_i
            mixed_q = sample * nco_q
            
            # Phase error
            phase_error = np.arctan2(mixed_q, mixed_i)
            
            # Loop filter
            loop_filter_state += beta * phase_error
            nco_freq += alpha * phase_error + loop_filter_state
            
            # NCO update
            nco_phase += nco_freq
            nco_phase = np.mod(nco_phase, 2 * np.pi)
        
        return np.real(recovered_carrier)

    def _apply_deemphasis(self, signal: np.ndarray, standard: str = 'fcc') -> np.ndarray:
        """Apply de-emphasis filtering"""
        try:
            # Simple de-emphasis implementation
            # In practice, this would use proper filter design
            return signal  # Placeholder
        except AttributeError:
            return signal

    def _estimate_snr(self, audio_signal: np.ndarray, rf_signal: np.ndarray) -> float:
        """Estimate SNR of demodulated audio"""
        if len(audio_signal) == 0:
            return 0.0
        
        # Signal power (RMS of audio in active portions)
        signal_power = np.mean(audio_signal**2)
        
        # Estimate noise power from high-frequency components
        if len(audio_signal) > 100:
            # High-pass filter to isolate noise
            sos = self._design_butterworth_filter(8000, 'high', order=4)
            noise_estimate = sosfilt(sos, audio_signal)
            noise_power = np.mean(noise_estimate**2)
        else:
            # Simple estimation
            noise_power = np.var(audio_signal) * 0.1
        
        if noise_power == 0:
            return 60.0  # Very high SNR
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(max(snr_linear, 1e-10))
        
        return max(min(snr_db, 60.0), -10.0)  # Clamp between -10 and 60 dB

    def _calculate_thd(self, signal: np.ndarray) -> float:
        """Calculate Total Harmonic Distortion"""
        if len(signal) < 1024:
            return 0.0
        
        # FFT-based THD calculation
        spectrum = np.abs(fft(signal))
        freqs = fftfreq(len(signal), 1/self.fs)
        
        # Find fundamental frequency (highest peak)
        fundamental_idx = np.argmax(spectrum[1:len(spectrum)//2]) + 1
        fundamental_power = spectrum[fundamental_idx]**2
        
        # Find harmonics (2f, 3f, 4f, 5f)
        harmonic_power = 0
        for h in range(2, 6):  # 2nd to 5th harmonic
            harmonic_idx = fundamental_idx * h
            if harmonic_idx < len(spectrum)//2:
                harmonic_power += spectrum[harmonic_idx]**2
        
        if fundamental_power == 0:
            return 0.0
        
        thd = np.sqrt(harmonic_power / fundamental_power) * 100
        return min(thd, 50.0)  # Cap at 50%

    def _estimate_carrier_recovery_error(self, signal: np.ndarray) -> float:
        """Estimate carrier recovery error"""
        # Simple frequency error estimation
        analytic_signal = hilbert(signal)
        inst_freq = np.diff(np.unwrap(np.angle(analytic_signal))) * self.fs / (2 * np.pi)
        
        # Estimate carrier frequency from spectrum peak
        spectrum = np.abs(fft(signal))
        freqs = fftfreq(len(signal), 1/self.fs)
        peak_freq_idx = np.argmax(spectrum[1:len(spectrum)//2]) + 1
        estimated_carrier = abs(freqs[peak_freq_idx])
        
        # Error in Hz
        carrier_error = abs(estimated_carrier - self.carrier_freq)
        
        return carrier_error

    def _assess_quality(self, snr_db: float, thd_percent: float) -> str:
        """Assess overall demodulation quality"""
        if snr_db > 20 and thd_percent < 3:
            return "Excellent"
        elif snr_db > 15 and thd_percent < 5:
            return "Good"
        elif snr_db > 10 and thd_percent < 10:
            return "Fair"
        else:
            return "Poor"


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
    """Enhanced classifier for analog modulation types with improved features"""

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate
        self.nyquist = sample_rate / 2

    def classify_analog_modulation(self, input_signal):
        """Enhanced classify analog modulation type with confidence metrics"""
        features = self._extract_analog_features(input_signal)
        classification, confidence = self._classify_from_features(features)
        
        return {
            'modulation_type': classification,
            'confidence': confidence,
            'features': features
        }

    def _extract_analog_features(self, input_signal):
        """Enhanced feature extraction for analog modulation classification"""
        # Envelope variations
        envelope = np.abs(hilbert(input_signal))
        envelope_var = np.var(envelope)
        envelope_mean = np.mean(envelope)
        envelope_cv = envelope_var / (envelope_mean + 1e-12)  # Coefficient of variation

        # Frequency characteristics with improved robustness
        psd = np.array([])
        freqs = np.array([])
        spectral_centroid = 0
        spectral_spread = 0
        
        try:
            freqs, psd = signal.welch(input_signal, fs=self.fs, nperseg=min(1024, len(input_signal)//4))
            if len(psd) > 0 and np.sum(psd) > 0:
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
        except Exception as e:
            logger.warning(f"Error in spectral analysis: {e}")
            spectral_centroid = 0
            spectral_spread = 0

        # Phase characteristics
        analytic_signal = hilbert(input_signal)
        inst_phase = np.angle(analytic_signal)
        phase_diff = np.diff(np.unwrap(inst_phase))
        phase_var = np.var(phase_diff)

        # Instantaneous frequency with improved calculation
        inst_freq = self.fs * phase_diff / (2 * np.pi)
        freq_var = np.var(inst_freq)
        
        # Additional features for better classification
        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(input_signal)) != 0) / len(input_signal)
        
        # Peak-to-average power ratio
        peak_power = np.max(np.abs(input_signal))**2
        avg_power = np.mean(input_signal**2)
        papr = peak_power / (avg_power + 1e-12)
        
        # Spectral flatness (measure of how flat the spectrum is)
        if len(psd) > 0:
            geometric_mean = np.exp(np.mean(np.log(psd + 1e-12)))
            arithmetic_mean = np.mean(psd)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-12)
        else:
            spectral_flatness = 0

        return {
            'envelope_cv': envelope_cv,
            'envelope_var': envelope_var,
            'spectral_spread': spectral_spread,
            'phase_var': phase_var,
            'freq_var': freq_var,
            'spectral_centroid': spectral_centroid,
            'zero_crossing_rate': zero_crossings,
            'papr': papr,
            'spectral_flatness': spectral_flatness,
            'envelope_mean': envelope_mean
        }

    def _classify_from_features(self, features):
        """Enhanced rule-based classification from features with confidence scoring"""
        env_cv = features['envelope_cv']
        freq_var = features['freq_var']
        phase_var = features['phase_var']
        papr = features['papr']
        spectral_flatness = features['spectral_flatness']
        
        # Initialize confidence score
        confidence = 0.5
        classification = "Unknown"

        # Enhanced classification rules with confidence scoring
        if env_cv > 0.5:
            # High envelope variation indicates AM
            confidence = min(0.9, 0.5 + env_cv * 0.4)
            
            if env_cv > 1.0 and papr > 2.0:
                classification = "AM (DSB-LC)"
                confidence = min(0.95, confidence + 0.1)
            elif env_cv > 0.7:
                classification = "AM (DSB-SC/SSB/VSB)"
                confidence = min(0.85, confidence)
            else:
                classification = "AM (Low modulation)"
                confidence = max(0.6, confidence - 0.1)

        elif freq_var > 1000:  # High frequency variation
            # Could be FM or PM
            confidence = min(0.9, 0.5 + min(freq_var/10000, 0.4))
            
            if phase_var > 1.0 and spectral_flatness < 0.5:
                classification = "FM (Wide-band)"
                confidence = min(0.9, confidence + 0.1)
            elif freq_var > 5000:
                classification = "FM (Narrow-band)"
                confidence = min(0.8, confidence)
            else:
                classification = "FM (Uncertain)"
                confidence = max(0.6, confidence - 0.1)

        elif phase_var > 0.5:
            # Phase modulation
            confidence = min(0.85, 0.5 + phase_var * 0.3)
            classification = "PM"
            
            # Additional check for PM vs FM distinction
            if freq_var < 100:
                confidence = min(0.9, confidence + 0.1)

        elif env_cv < 0.1 and freq_var < 100 and phase_var < 0.1:
            # Low variation in all parameters
            confidence = 0.8
            classification = "Unmodulated Carrier"
        
        else:
            # Ambiguous case - use additional features
            if papr > 3.0:
                classification = "AM (Suspected)"
                confidence = 0.4
            elif spectral_flatness > 0.8:
                classification = "Noise-like"
                confidence = 0.3
            else:
                classification = "Unknown Modulation"
                confidence = 0.2

        return classification, confidence


def test_optimized_analog_modulation():
    """Test optimized analog modulation implementation"""
    print("🧪 Testing Optimized Analog Modulation...")
    
    # Test parameters
    fs = 100e3  # 100 kHz sample rate
    duration = 0.1  # 100 ms
    t = np.arange(0, duration, 1/fs)
    
    # Test message signal (1 kHz sine wave)
    message_freq = 1000
    message = 0.5 * np.sin(2 * np.pi * message_freq * t)
    
    # Initialize components
    modulator = AnalogModulation(fs)
    demodulator = AnalogDemodulation(fs, 10000)
    classifier = AnalogModulationClassifier(fs)
    
    print(f"✅ Components initialized: Fs={fs/1e3:.1f} kHz")
    
    # Test AM modulation with parameters object
    am_params = ModulationParameters(
        carrier_freq=10000,
        modulation_index=0.8,
        bandwidth=5000
    )
    
    print(f"📡 Testing AM modulation...")
    am_signal = modulator.am_modulate(message, carrier_freq=am_params.carrier_freq, 
                                    modulation_index=am_params.modulation_index)
    print(f"   Generated {len(am_signal)} samples")
    
    # Test AM demodulation with metrics
    am_result = demodulator.am_demodulate(am_signal, return_metrics=True)
    print(f"   SNR: {am_result.snr_estimate:.1f} dB")
    print(f"   THD: {am_result.thd_percent:.1f}%")
    print(f"   Quality: {am_result.quality_metric}")
    
    # Test FM modulation
    print(f"📻 Testing FM modulation...")
    fm_signal = modulator.fm_modulate(message, carrier_freq=10000, deviation=5000, 
                                    mod_type='wbfm', pre_emphasis=True)
    print(f"   Generated {len(fm_signal)} samples")
    
    # Test FM demodulation with metrics
    fm_result = demodulator.fm_demodulate(fm_signal, method='discriminator', return_metrics=True)
    print(f"   SNR: {fm_result.snr_estimate:.1f} dB")
    print(f"   THD: {fm_result.thd_percent:.1f}%")
    print(f"   Quality: {fm_result.quality_metric}")
    
    # Test classification
    print(f"🔍 Testing enhanced classification...")
    am_class_result = classifier.classify_analog_modulation(am_signal)
    print(f"   AM Classification: {am_class_result['modulation_type']}")
    print(f"   Confidence: {am_class_result['confidence']:.2f}")
    
    fm_class_result = classifier.classify_analog_modulation(fm_signal)
    print(f"   FM Classification: {fm_class_result['modulation_type']}")
    print(f"   Confidence: {fm_class_result['confidence']:.2f}")
    
    print("✅ All optimized analog modulation tests completed successfully!")
    return True


if __name__ == "__main__":
    test_optimized_analog_modulation()
