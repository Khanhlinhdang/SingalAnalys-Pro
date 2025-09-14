"""
Optimized Analog Modulation Module - Research-Based Implementation
Based on ITU-R recommendations and communication theory standards
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import butter, filtfilt, lfilter, hilbert, sosfilt
from scipy.signal.windows import kaiser
import warnings
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

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

class OptimizedAnalogModulation:
    """Research-based analog modulation with ITU-R compliance"""
    
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
    
    def __init__(self, sample_rate: float = 1e6):
        """
        Initialize analog modulation system
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.fs = sample_rate
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
            logger.warning(f"Cutoff frequency {cutoff} Hz too high for Nyquist {self.nyquist}")
            normalized_cutoff = 0.9
        
        # Use second-order sections for numerical stability
        sos = signal.butter(order, normalized_cutoff, btype=btype, output='sos')
        return sos
    
    def _init_preemphasis_filters(self):
        """Initialize pre-emphasis and de-emphasis filters"""
        # Pre-emphasis: H(s) = (1 + sτ) / (1 + sτ/K) where K > 1
        # For FM: τ = 75μs (FCC) or 50μs (CCITT)
        
        for standard, tc in self.PRE_EMPHASIS_TC.items():
            # Pre-emphasis filter (high-pass characteristic)
            omega_c = 1 / tc  # Corner frequency in rad/s
            freq_c = omega_c / (2 * np.pi)  # Corner frequency in Hz
            
            if freq_c < self.nyquist:
                # Create pre-emphasis filter
                num = [tc, 1]
                den = [tc/10, 1]  # K = 10 for standard pre-emphasis
                
                # Convert to digital filter
                system = signal.cont2discrete((num, den), 1/self.fs, method='bilinear')
                setattr(self, f'preemph_{standard.lower()}', system)
                
                # De-emphasis filter (inverse)
                system_de = signal.cont2discrete((den, num), 1/self.fs, method='bilinear')
                setattr(self, f'deemph_{standard.lower()}', system_de)
    
    def am_modulate(self, message: np.ndarray, params: ModulationParameters) -> np.ndarray:
        """
        AM modulation with carrier recovery aids
        
        Args:
            message: Message signal
            params: Modulation parameters
            
        Returns:
            AM modulated signal
        """
        # Normalize message signal
        message = self._normalize_audio(message)
        
        # Apply audio processing
        if params.bandwidth:
            message = self._apply_audio_filter(message, params.bandwidth)
        
        # Generate time vector
        t = np.arange(len(message)) / self.fs
        
        # Generate carrier
        carrier = np.cos(2 * np.pi * params.carrier_freq * t)
        
        # Modulation
        if params.modulation_index > 1.0:
            logger.warning("Modulation index > 1.0 may cause distortion")
        
        # AM-DSB-LC (Standard AM)
        modulated = (1 + params.modulation_index * message) * carrier
        
        # Add pilot tone for carrier recovery (±25 Hz from carrier)
        pilot_freq = params.carrier_freq + 25
        pilot = 0.01 * np.cos(2 * np.pi * pilot_freq * t)  # -40 dB pilot
        
        return modulated + pilot
    
    def am_dsb_sc_modulate(self, message: np.ndarray, params: ModulationParameters) -> np.ndarray:
        """DSB-SC (Double Sideband Suppressed Carrier) modulation"""
        message = self._normalize_audio(message)
        
        if params.bandwidth:
            message = self._apply_audio_filter(message, params.bandwidth)
        
        t = np.arange(len(message)) / self.fs
        carrier = np.cos(2 * np.pi * params.carrier_freq * t)
        
        # DSB-SC: direct multiplication (no DC offset)
        return message * carrier
    
    def ssb_modulate(self, message: np.ndarray, params: ModulationParameters,
                    sideband: str = 'usb') -> np.ndarray:
        """
        SSB (Single Sideband) modulation using Hilbert transform method
        
        Args:
            message: Message signal
            params: Modulation parameters
            sideband: 'usb' or 'lsb'
            
        Returns:
            SSB modulated signal
        """
        message = self._normalize_audio(message)
        
        if params.bandwidth:
            message = self._apply_audio_filter(message, params.bandwidth)
        
        t = np.arange(len(message)) / self.fs
        
        # Hilbert transform for quadrature component
        message_hilbert = hilbert(message)
        message_i = np.real(message_hilbert)  # In-phase
        message_q = np.imag(message_hilbert)  # Quadrature
        
        # Quadrature carriers
        carrier_i = np.cos(2 * np.pi * params.carrier_freq * t)
        carrier_q = np.sin(2 * np.pi * params.carrier_freq * t)
        
        if sideband.lower() == 'usb':
            # Upper sideband: I*cos - Q*sin
            ssb_signal = message_i * carrier_i - message_q * carrier_q
        else:
            # Lower sideband: I*cos + Q*sin
            ssb_signal = message_i * carrier_i + message_q * carrier_q
        
        return ssb_signal
    
    def vsb_modulate(self, message: np.ndarray, params: ModulationParameters) -> np.ndarray:
        """
        VSB (Vestigial Sideband) modulation
        Used in analog TV (NTSC/PAL)
        """
        message = self._normalize_audio(message)
        
        # Start with DSB-SC
        dsb_sc = self.am_dsb_sc_modulate(message, params)
        
        # Design VSB filter
        # For NTSC TV: pass full upper sideband, partial lower sideband
        vestigial_freq = 1.25e6  # 1.25 MHz vestigial bandwidth for TV
        
        # VSB filter design (asymmetric around carrier)
        vsb_filter = self._design_vsb_filter(params.carrier_freq, vestigial_freq)
        
        # Apply VSB filter
        vsb_signal = self._apply_filter(dsb_sc, vsb_filter)
        
        return vsb_signal
    
    def fm_modulate(self, message: np.ndarray, params: ModulationParameters,
                   mode: str = 'wbfm') -> np.ndarray:
        """
        FM modulation with pre-emphasis and stereo capability
        
        Args:
            message: Message signal
            params: Modulation parameters
            mode: 'nbfm' (narrow-band) or 'wbfm' (wide-band)
            
        Returns:
            FM modulated signal
        """
        message = self._normalize_audio(message)
        
        # Apply pre-emphasis for FM broadcast
        if params.pre_emphasis:
            message = self._apply_preemphasis(message, 'fcc')
        
        # Apply audio filter
        if params.bandwidth:
            message = self._apply_audio_filter(message, params.bandwidth)
        
        t = np.arange(len(message)) / self.fs
        
        if mode == 'nbfm':
            # Narrow-band FM (Carson's rule: BW ≈ 2 * deviation)
            deviation = params.deviation or 5000  # 5 kHz for NBFM
            
            # NBFM approximation: cos(ωt) - β*m(t)*sin(ωt)
            beta = deviation / self._estimate_message_bandwidth(message)
            
            if beta < 0.5:  # Valid for small modulation index
                carrier_cos = np.cos(2 * np.pi * params.carrier_freq * t)
                carrier_sin = np.sin(2 * np.pi * params.carrier_freq * t)
                
                fm_signal = carrier_cos - beta * message * carrier_sin
            else:
                # Use wideband method
                fm_signal = self._wideband_fm(message, params, t)
        else:
            # Wide-band FM
            fm_signal = self._wideband_fm(message, params, t)
        
        return fm_signal
    
    def _wideband_fm(self, message: np.ndarray, params: ModulationParameters,
                    t: np.ndarray) -> np.ndarray:
        """Wide-band FM modulation"""
        deviation = params.deviation or 75000  # 75 kHz for WBFM broadcast
        
        # Integrate message for frequency modulation
        # Use cumulative sum for integration
        integrated_message = np.cumsum(message) / self.fs
        
        # Instantaneous phase
        phase = 2 * np.pi * (params.carrier_freq * t + 
                           deviation * integrated_message)
        
        # Generate FM signal
        fm_signal = np.cos(phase)
        
        return fm_signal
    
    def pm_modulate(self, message: np.ndarray, params: ModulationParameters) -> np.ndarray:
        """
        Phase modulation
        
        Args:
            message: Message signal
            params: Modulation parameters
            
        Returns:
            PM modulated signal
        """
        message = self._normalize_audio(message)
        
        if params.bandwidth:
            message = self._apply_audio_filter(message, params.bandwidth)
        
        t = np.arange(len(message)) / self.fs
        
        # Phase deviation (typically in radians)
        phase_deviation = params.deviation or (np.pi / 4)  # 45 degrees max
        
        # PM signal
        instantaneous_phase = 2 * np.pi * params.carrier_freq * t + phase_deviation * message
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
    
    def _apply_audio_filter(self, signal: np.ndarray, bandwidth: float) -> np.ndarray:
        """Apply audio bandwidth limiting filter"""
        if bandwidth >= self.nyquist:
            return signal
        
        # Design low-pass filter for audio bandwidth
        sos = self._design_butterworth_filter(bandwidth, 'low', order=8)
        
        # Apply filter
        filtered = sosfilt(sos, signal)
        
        return filtered
    
    def _apply_preemphasis(self, signal: np.ndarray, standard: str = 'fcc') -> np.ndarray:
        """Apply pre-emphasis filtering"""
        try:
            filter_system = getattr(self, f'preemph_{standard.lower()}')
            b, a = filter_system[0], filter_system[1]
            
            # Apply pre-emphasis filter
            emphasized = lfilter(b, a, signal)
            
            return emphasized
        except AttributeError:
            logger.warning(f"Pre-emphasis standard '{standard}' not available")
            return signal
    
    def _estimate_message_bandwidth(self, message: np.ndarray) -> float:
        """Estimate message signal bandwidth"""
        # FFT-based bandwidth estimation
        spectrum = np.abs(fft(message))
        freqs = fftfreq(len(message), 1/self.fs)
        
        # Find 99% power bandwidth
        power_spectrum = spectrum**2
        total_power = np.sum(power_spectrum)
        
        cumulative_power = np.cumsum(power_spectrum)
        idx_99 = np.argmax(cumulative_power >= 0.99 * total_power)
        
        bandwidth = abs(freqs[idx_99])
        
        return max(bandwidth, 1000)  # Minimum 1 kHz
    
    def _design_vsb_filter(self, carrier_freq: float, vestigial_bw: float) -> np.ndarray:
        """Design VSB (Vestigial Sideband) filter"""
        # Simplified VSB filter design
        # In practice, this would be a carefully designed filter
        
        # Create frequency response
        n_fft = 2048
        freqs = np.linspace(-self.nyquist, self.nyquist, n_fft)
        
        # VSB response: full upper sideband, partial lower sideband
        response = np.zeros(n_fft, dtype=complex)
        
        for i, freq in enumerate(freqs):
            rel_freq = freq - carrier_freq
            
            if rel_freq > 0:  # Upper sideband - pass fully
                response[i] = 1.0
            elif rel_freq > -vestigial_bw:  # Vestigial part - partial pass
                response[i] = 0.5 * (1 + np.cos(np.pi * rel_freq / vestigial_bw))
            else:  # Lower sideband - reject
                response[i] = 0.0
        
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

class OptimizedAnalogDemodulation:
    """Research-based analog demodulation with carrier recovery"""
    
    def __init__(self, sample_rate: float = 1e6, carrier_freq: float = 100e3):
        """
        Initialize analog demodulation system
        
        Args:
            sample_rate: Sample rate in Hz
            carrier_freq: Expected carrier frequency in Hz
        """
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
        
        # De-emphasis filters
        self._init_deemphasis_filters()
    
    def _design_butterworth_filter(self, cutoff: float, btype: str, 
                                 order: int = 6) -> np.ndarray:
        """Design Butterworth filter"""
        normalized_cutoff = cutoff / self.nyquist
        
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.9
        
        sos = signal.butter(order, normalized_cutoff, btype=btype, output='sos')
        return sos
    
    def _init_deemphasis_filters(self):
        """Initialize de-emphasis filters"""
        # Standard de-emphasis time constants
        time_constants = {'fcc': 75e-6, 'ccitt': 50e-6}
        
        for standard, tc in time_constants.items():
            omega_c = 1 / tc
            freq_c = omega_c / (2 * np.pi)
            
            if freq_c < self.nyquist:
                # De-emphasis filter: H(s) = 1 / (1 + sτ)
                num = [1]
                den = [tc, 1]
                
                system = signal.cont2discrete((num, den), 1/self.fs, method='bilinear')
                setattr(self, f'deemph_{standard}', system)
    
    def am_demodulate(self, received_signal: np.ndarray, 
                     method: str = 'envelope') -> DemodulationResult:
        """
        AM demodulation with multiple detection methods
        
        Args:
            received_signal: Received AM signal
            method: 'envelope', 'synchronous', or 'product'
            
        Returns:
            Demodulation result with quality metrics
        """
        if method == 'envelope':
            demod_audio = self._envelope_detection(received_signal)
        elif method == 'synchronous':
            demod_audio = self._synchronous_detection(received_signal)
        elif method == 'product':
            demod_audio = self._product_detection(received_signal)
        else:
            logger.warning(f"Unknown AM demod method: {method}, using envelope")
            demod_audio = self._envelope_detection(received_signal)
        
        # Apply audio filtering
        demod_audio = sosfilt(self.am_audio_lpf, demod_audio)
        
        # Calculate quality metrics
        snr_estimate = self._estimate_snr(demod_audio, received_signal)
        thd = self._calculate_thd(demod_audio)
        
        # Carrier recovery error (for synchronous detection)
        carrier_error = self._estimate_carrier_recovery_error(received_signal)
        
        # Quality assessment
        quality = self._assess_quality(snr_estimate, thd)
        
        result = DemodulationResult(
            demodulated_signal=demod_audio,
            snr_estimate=snr_estimate,
            thd_percent=thd,
            carrier_recovery_error=carrier_error,
            timing_recovery_error=0.0,  # Not applicable for AM
            quality_metric=quality
        )
        
        return result
    
    def _envelope_detection(self, signal: np.ndarray) -> np.ndarray:
        """Envelope detection using Hilbert transform"""
        # Hilbert transform for analytic signal
        analytic_signal = hilbert(signal)
        
        # Envelope is the magnitude
        envelope = np.abs(analytic_signal)
        
        # Remove DC component (carrier)
        envelope_ac = envelope - np.mean(envelope)
        
        return envelope_ac
    
    def _synchronous_detection(self, signal: np.ndarray) -> np.ndarray:
        """Synchronous (coherent) detection with carrier recovery"""
        # Carrier recovery using Costas loop or PLL
        recovered_carrier = self._costas_loop_carrier_recovery(signal)
        
        # Synchronous detection
        demod = signal * recovered_carrier
        
        return demod
    
    def _product_detection(self, signal: np.ndarray) -> np.ndarray:
        """Product detection (simple coherent detection)"""
        t = np.arange(len(signal)) / self.fs
        
        # Local oscillator (assume perfect frequency knowledge)
        local_osc = 2 * np.cos(2 * np.pi * self.carrier_freq * t)
        
        # Product detection
        demod = signal * local_osc
        
        return demod
    
    def fm_demodulate(self, received_signal: np.ndarray,
                     method: str = 'discriminator') -> DemodulationResult:
        """
        FM demodulation with multiple methods
        
        Args:
            received_signal: Received FM signal
            method: 'discriminator', 'pll', or 'quadrature'
            
        Returns:
            Demodulation result
        """
        if method == 'discriminator':
            demod_audio = self._frequency_discriminator(received_signal)
        elif method == 'pll':
            demod_audio = self._pll_fm_demod(received_signal)
        elif method == 'quadrature':
            demod_audio = self._quadrature_detector(received_signal)
        else:
            logger.warning(f"Unknown FM demod method: {method}, using discriminator")
            demod_audio = self._frequency_discriminator(received_signal)
        
        # Apply de-emphasis for FM broadcast
        demod_audio = self._apply_deemphasis(demod_audio, 'fcc')
        
        # Apply audio filtering
        demod_audio = sosfilt(self.fm_audio_lpf, demod_audio)
        
        # Quality metrics
        snr_estimate = self._estimate_snr(demod_audio, received_signal)
        thd = self._calculate_thd(demod_audio)
        
        quality = self._assess_quality(snr_estimate, thd)
        
        result = DemodulationResult(
            demodulated_signal=demod_audio,
            snr_estimate=snr_estimate,
            thd_percent=thd,
            carrier_recovery_error=0.0,  # Not applicable for FM
            timing_recovery_error=0.0,   # Not applicable for FM
            quality_metric=quality
        )
        
        return result
    
    def _frequency_discriminator(self, signal: np.ndarray) -> np.ndarray:
        """Frequency discriminator using phase differentiation"""
        # Convert to complex baseband
        analytic_signal = hilbert(signal)
        
        # Extract instantaneous phase
        inst_phase = np.angle(analytic_signal)
        
        # Unwrap phase discontinuities
        unwrapped_phase = np.unwrap(inst_phase)
        
        # Differentiate to get frequency
        inst_freq = np.diff(unwrapped_phase) * self.fs / (2 * np.pi)
        
        # Pad to maintain length
        inst_freq = np.concatenate([[inst_freq[0]], inst_freq])
        
        # Remove carrier frequency to get baseband audio
        audio = inst_freq - self.carrier_freq
        
        return audio
    
    def _pll_fm_demod(self, signal: np.ndarray) -> np.ndarray:
        """PLL-based FM demodulation"""
        # PLL parameters
        Kv = 2 * np.pi * self.pll_bandwidth  # VCO gain
        Kd = 1.0  # Phase detector gain
        
        # Loop filter parameters
        wn = self.pll_bandwidth  # Natural frequency
        zeta = self.pll_damping  # Damping factor
        
        K = Kd * Kv
        tau1 = K / (wn**2)
        tau2 = 2 * zeta / wn - 1 / K
        
        # Initialize PLL state
        vco_phase = 0.0
        loop_filter_state = 0.0
        demod_output = np.zeros(len(signal))
        
        for i, sample in enumerate(signal):
            # Phase detector
            phase_error = np.angle(sample * np.exp(-1j * vco_phase))
            
            # Loop filter (lead-lag)
            loop_filter_state += (tau1 * phase_error - tau2 * loop_filter_state) / self.fs
            loop_filter_out = phase_error + loop_filter_state
            
            # VCO
            vco_freq = self.carrier_freq + Kv * loop_filter_out
            vco_phase += 2 * np.pi * vco_freq / self.fs
            
            # Demodulated output is the VCO control voltage
            demod_output[i] = loop_filter_out
        
        return demod_output
    
    def _quadrature_detector(self, signal: np.ndarray) -> np.ndarray:
        """Quadrature detector for FM demodulation"""
        # Delay line
        delayed_signal = np.concatenate([[0], signal[:-1]])
        
        # Quadrature multiplication
        quad_output = np.real(signal * np.conj(delayed_signal))
        
        # Low-pass filter to extract audio
        sos = self._design_butterworth_filter(15000, 'low', order=6)
        audio = sosfilt(sos, quad_output)
        
        return audio
    
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
            nco_out = np.exp(1j * nco_phase)
            recovered_carrier[i] = nco_out
            
            # Mix with received signal
            baseband = sample * np.conj(nco_out)
            
            # Phase detector (for BPSK/AM)
            phase_error = np.imag(baseband) * np.sign(np.real(baseband))
            
            # Loop filter
            loop_filter_state += beta * phase_error
            nco_freq += loop_filter_state + alpha * phase_error
            
            # Update NCO phase
            nco_phase += nco_freq
            
            # Keep phase in [-π, π]
            if nco_phase > np.pi:
                nco_phase -= 2 * np.pi
            elif nco_phase < -np.pi:
                nco_phase += 2 * np.pi
        
        return np.real(recovered_carrier)
    
    def _apply_deemphasis(self, signal: np.ndarray, standard: str = 'fcc') -> np.ndarray:
        """Apply de-emphasis filtering"""
        try:
            filter_system = getattr(self, f'deemph_{standard}')
            b, a = filter_system[0], filter_system[1]
            
            # Apply de-emphasis filter
            deemphasized = lfilter(b, a, signal)
            
            return deemphasized
        except AttributeError:
            logger.warning(f"De-emphasis standard '{standard}' not available")
            return signal
    
    def _estimate_snr(self, audio_signal: np.ndarray, rf_signal: np.ndarray) -> float:
        """Estimate SNR of demodulated audio"""
        if len(audio_signal) == 0:
            return -10.0
        
        # Signal power (RMS of audio in active portions)
        signal_power = np.mean(audio_signal**2)
        
        # Estimate noise power from high-frequency components
        # High-pass filter to isolate noise
        if len(audio_signal) > 100:
            sos = self._design_butterworth_filter(8000, 'high', order=4)
            noise_estimate = sosfilt(sos, audio_signal)
            noise_power = np.mean(noise_estimate**2)
        else:
            # Fallback noise estimation
            noise_power = np.var(np.diff(audio_signal))
        
        if noise_power == 0:
            return 40.0  # Very high SNR
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(max(snr_linear, 1e-10))
        
        return np.clip(snr_db, -20, 40)
    
    def _calculate_thd(self, signal: np.ndarray) -> float:
        """Calculate Total Harmonic Distortion"""
        if len(signal) < 1024:
            return 0.0
        
        # FFT for harmonic analysis
        spectrum = np.abs(fft(signal, n=4096))
        freqs = fftfreq(4096, 1/self.fs)
        
        # Find fundamental frequency (peak in spectrum)
        positive_freqs = freqs[:len(freqs)//2]
        positive_spectrum = spectrum[:len(spectrum)//2]
        
        # Look for fundamental in audio range (100 Hz - 4 kHz)
        audio_mask = (positive_freqs >= 100) & (positive_freqs <= 4000)
        if not np.any(audio_mask):
            return 0.0
        
        fund_idx = np.argmax(positive_spectrum[audio_mask])
        fund_freq = positive_freqs[audio_mask][fund_idx]
        fund_power = positive_spectrum[audio_mask][fund_idx]**2
        
        # Calculate harmonic powers
        harmonic_power = 0
        for harmonic in range(2, 6):  # 2nd to 5th harmonics
            harm_freq = harmonic * fund_freq
            if harm_freq < positive_freqs[-1]:
                # Find closest frequency bin
                harm_idx = np.argmin(np.abs(positive_freqs - harm_freq))
                harmonic_power += positive_spectrum[harm_idx]**2
        
        if fund_power == 0:
            return 0.0
        
        thd = np.sqrt(harmonic_power / fund_power) * 100
        return min(thd, 50)  # Cap at 50%
    
    def _estimate_carrier_recovery_error(self, signal: np.ndarray) -> float:
        """Estimate carrier recovery error for synchronous detection"""
        # Simple carrier frequency estimation
        spectrum = np.abs(fft(signal))
        freqs = fftfreq(len(signal), 1/self.fs)
        
        # Find peak near expected carrier frequency
        carrier_mask = (np.abs(freqs - self.carrier_freq) < 1000)  # ±1 kHz
        if np.any(carrier_mask):
            masked_spectrum = spectrum.copy()
            masked_spectrum[~carrier_mask] = 0
            
            peak_idx = np.argmax(masked_spectrum)
            estimated_carrier = freqs[peak_idx]
            
            error = abs(estimated_carrier - self.carrier_freq)
            return error
        
        return 1000.0  # Large error if no carrier found
    
    def _assess_quality(self, snr_db: float, thd_percent: float) -> str:
        """Assess overall demodulation quality"""
        if snr_db > 25 and thd_percent < 1:
            return "excellent"
        elif snr_db > 20 and thd_percent < 2:
            return "very_good"
        elif snr_db > 15 and thd_percent < 5:
            return "good"
        elif snr_db > 10 and thd_percent < 10:
            return "fair"
        else:
            return "poor"

# Testing functions
def test_analog_modulation():
    """Test analog modulation implementation"""
    print("Testing Analog Modulation...")
    
    # Test parameters
    fs = 100e3  # 100 kHz sample rate
    duration = 0.1  # 100 ms
    t = np.arange(0, duration, 1/fs)
    
    # Test message signal (1 kHz sine wave)
    message_freq = 1000
    message = 0.5 * np.sin(2 * np.pi * message_freq * t)
    
    # Test modulation
    modulator = OptimizedAnalogModulation(fs)
    
    # AM parameters
    am_params = ModulationParameters(
        carrier_freq=10000,
        modulation_index=0.8,
        bandwidth=5000
    )
    
    # Test AM modulation
    am_signal = modulator.am_modulate(message, am_params)
    print(f"AM signal generated: {len(am_signal)} samples")
    
    # FM parameters
    fm_params = ModulationParameters(
        carrier_freq=10000,
        modulation_index=0,  # Not used for FM
        deviation=5000,
        pre_emphasis=True
    )
    
    # Test FM modulation
    fm_signal = modulator.fm_modulate(message, fm_params, mode='wbfm')
    print(f"FM signal generated: {len(fm_signal)} samples")
    
    # Test demodulation
    demodulator = OptimizedAnalogDemodulation(fs, am_params.carrier_freq)
    
    # AM demodulation
    am_result = demodulator.am_demodulate(am_signal, method='envelope')
    print(f"AM SNR: {am_result.snr_estimate:.1f} dB, THD: {am_result.thd_percent:.1f}%, Quality: {am_result.quality_metric}")
    
    # FM demodulation
    fm_result = demodulator.fm_demodulate(fm_signal, method='discriminator')
    print(f"FM SNR: {fm_result.snr_estimate:.1f} dB, THD: {fm_result.thd_percent:.1f}%, Quality: {fm_result.quality_metric}")
    
    print("✅ Analog modulation tests completed")

if __name__ == "__main__":
    test_analog_modulation()