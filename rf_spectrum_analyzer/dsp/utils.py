"""
DSP Utilities and Helper Functions
Common utility functions for digital signal processing
"""

import numpy as np
import scipy.signal as signal
from scipy import special
from typing import Optional, Tuple, List, Union, Dict, Any, Callable
from dataclasses import dataclass
import warnings
import sys
import os

# Add paths for local libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'sdr'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'sk_dsp_comm'))

# Import from sdr library
try:
    import sdr
    from sdr._helper import *
    SDR_AVAILABLE = True
except ImportError:
    SDR_AVAILABLE = False
    warnings.warn("SDR library not available. Some features disabled.")

# Import from sk_dsp_comm
try:
    import sk_dsp_comm.sigsys as ss
    import sk_dsp_comm.fir_design_helper as fir_helper
    SK_DSP_AVAILABLE = True
except ImportError:
    SK_DSP_AVAILABLE = False
    warnings.warn("sk_dsp_comm library not available. Some features disabled.")

from rf_spectrum_analyzer.utils.logger import get_logger

logger = get_logger('dsp_utils')


# Window Functions
def create_window(window_type: str, length: int, **kwargs) -> np.ndarray:
    """Create various window functions"""
    
    if window_type.lower() == "hann" or window_type.lower() == "hanning":
        return signal.windows.hann(length)
    
    elif window_type.lower() == "hamming":
        return signal.windows.hamming(length)
    
    elif window_type.lower() == "blackman":
        return signal.windows.blackman(length)
    
    elif window_type.lower() == "bartlett":
        return signal.windows.bartlett(length)
    
    elif window_type.lower() == "kaiser":
        beta = kwargs.get('beta', 8.6)
        return signal.windows.kaiser(length, beta)
    
    elif window_type.lower() == "tukey":
        alpha = kwargs.get('alpha', 0.5)
        return signal.windows.tukey(length, alpha)
    
    elif window_type.lower() == "gaussian":
        std = kwargs.get('std', length/6)
        return signal.windows.gaussian(length, std)
    
    elif window_type.lower() == "cheby" or window_type.lower() == "chebyshev":
        at = kwargs.get('at', 100)
        return signal.windows.chebwin(length, at)
    
    elif window_type.lower() == "rectangular" or window_type.lower() == "rect":
        return np.ones(length)
    
    else:
        logger.warning(f"Unknown window type: {window_type}, using Hann")
        return signal.windows.hann(length)


def window_correction_factor(window: np.ndarray, correction_type: str = "amplitude") -> float:
    """Calculate window correction factors"""
    
    if correction_type == "amplitude":
        # Amplitude correction (coherent gain)
        return len(window) / np.sum(window)
    
    elif correction_type == "power":
        # Power correction (non-coherent gain)
        return len(window) / np.sqrt(np.sum(window**2))
    
    elif correction_type == "enbw":
        # Equivalent noise bandwidth
        return len(window) * np.sum(window**2) / (np.sum(window)**2)
    
    else:
        return 1.0


# Noise Generation
def generate_awgn(length: int, variance: float = 1.0, 
                 complex_valued: bool = True) -> np.ndarray:
    """Generate Additive White Gaussian Noise"""
    
    if complex_valued:
        # Complex AWGN with proper variance scaling
        noise = (np.random.normal(0, np.sqrt(variance/2), length) + 
                1j * np.random.normal(0, np.sqrt(variance/2), length))
    else:
        # Real AWGN
        noise = np.random.normal(0, np.sqrt(variance), length)
    
    return noise


def generate_colored_noise(length: int, psd_shape: str = "pink", 
                          exponent: float = -1.0) -> np.ndarray:
    """Generate colored noise with specified PSD shape"""
    
    # Generate white noise
    white_noise = np.random.normal(0, 1, length)
    
    # Create frequency domain representation
    fft_white = np.fft.fft(white_noise)
    freqs = np.fft.fftfreq(length)
    
    # Apply coloring filter
    if psd_shape == "pink":
        # Pink noise (1/f)
        filter_response = 1 / np.sqrt(np.abs(freqs) + 1e-10)
    elif psd_shape == "brown" or psd_shape == "red":
        # Brown noise (1/f^2)
        filter_response = 1 / (np.abs(freqs) + 1e-10)
    elif psd_shape == "blue":
        # Blue noise (f)
        filter_response = np.sqrt(np.abs(freqs))
    elif psd_shape == "violet":
        # Violet noise (f^2)
        filter_response = np.abs(freqs)
    else:
        # Custom exponent
        filter_response = np.abs(freqs)**exponent
    
    # Avoid division by zero at DC
    filter_response[0] = 1.0
    
    # Apply filter and transform back
    colored_fft = fft_white * filter_response
    colored_noise = np.real(np.fft.ifft(colored_fft))
    
    return colored_noise


def snr_to_noise_power(signal_power: float, snr_db: float) -> float:
    """Convert SNR in dB to noise power"""
    snr_linear = 10**(snr_db / 10)
    return signal_power / snr_linear


def add_noise(signal: np.ndarray, snr_db: float, 
              noise_type: str = "awgn") -> np.ndarray:
    """Add noise to signal at specified SNR"""
    
    # Calculate signal power
    signal_power = np.mean(np.abs(signal)**2)
    
    # Calculate required noise power
    noise_power = snr_to_noise_power(signal_power, snr_db)
    
    # Generate noise
    if noise_type == "awgn":
        noise = generate_awgn(len(signal), noise_power, 
                             complex_valued=np.iscomplexobj(signal))
    elif noise_type == "pink":
        noise = generate_colored_noise(len(signal), "pink")
        noise = noise * np.sqrt(noise_power / np.var(noise))
    elif noise_type == "uniform":
        if np.iscomplexobj(signal):
            noise = (np.random.uniform(-1, 1, len(signal)) + 
                    1j * np.random.uniform(-1, 1, len(signal)))
        else:
            noise = np.random.uniform(-1, 1, len(signal))
        noise = noise * np.sqrt(noise_power / np.var(noise))
    else:
        noise = generate_awgn(len(signal), noise_power, 
                             complex_valued=np.iscomplexobj(signal))
    
    return signal + noise


# Signal Generation
def generate_tone(frequency: float, duration: float, sample_rate: float,
                 amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Generate sinusoidal tone"""
    
    t = np.arange(0, duration, 1/sample_rate)
    return amplitude * np.exp(1j * (2 * np.pi * frequency * t + phase))


def generate_chirp(f_start: float, f_end: float, duration: float,
                  sample_rate: float, method: str = "linear") -> np.ndarray:
    """Generate frequency sweep (chirp) signal"""
    
    t = np.arange(0, duration, 1/sample_rate)
    
    if method == "linear":
        return signal.chirp(t, f_start, duration, f_end, method='linear')
    elif method == "quadratic":
        return signal.chirp(t, f_start, duration, f_end, method='quadratic')
    elif method == "logarithmic":
        return signal.chirp(t, f_start, duration, f_end, method='logarithmic')
    elif method == "hyperbolic":
        return signal.chirp(t, f_start, duration, f_end, method='hyperbolic')
    else:
        return signal.chirp(t, f_start, duration, f_end, method='linear')


def generate_multitone(frequencies: List[float], amplitudes: List[float],
                      phases: List[float], duration: float, 
                      sample_rate: float) -> np.ndarray:
    """Generate multitone signal"""
    
    if len(frequencies) != len(amplitudes) or len(frequencies) != len(phases):
        raise ValueError("Frequencies, amplitudes, and phases must have same length")
    
    t = np.arange(0, duration, 1/sample_rate)
    signal_sum = np.zeros(len(t), dtype=complex)
    
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        signal_sum += amp * np.exp(1j * (2 * np.pi * freq * t + phase))
    
    return signal_sum


def generate_pulse_train(pulse_width: float, pulse_period: float, 
                        duration: float, sample_rate: float,
                        amplitude: float = 1.0) -> np.ndarray:
    """Generate pulse train signal"""
    
    t = np.arange(0, duration, 1/sample_rate)
    signal_out = np.zeros(len(t))
    
    pulse_samples = int(pulse_width * sample_rate)
    period_samples = int(pulse_period * sample_rate)
    
    for i in range(0, len(t), period_samples):
        end_idx = min(i + pulse_samples, len(t))
        signal_out[i:end_idx] = amplitude
    
    return signal_out


# Timing and Synchronization
def find_peaks_advanced(x: np.ndarray, height: Optional[float] = None,
                       distance: Optional[int] = None, 
                       prominence: Optional[float] = None) -> Dict[str, Any]:
    """Advanced peak finding with multiple criteria"""
    
    peaks, properties = signal.find_peaks(
        np.abs(x), 
        height=height,
        distance=distance,
        prominence=prominence
    )
    
    return {
        'peak_indices': peaks,
        'peak_values': x[peaks],
        'properties': properties
    }


def estimate_delay(signal1: np.ndarray, signal2: np.ndarray,
                  max_delay: Optional[int] = None) -> Dict[str, Any]:
    """Estimate delay between two signals using cross-correlation"""
    
    # Cross-correlation
    correlation = signal.correlate(signal2, signal1, mode='full')
    
    # Find peak
    if max_delay is not None:
        center = len(correlation) // 2
        start_idx = max(0, center - max_delay)
        end_idx = min(len(correlation), center + max_delay + 1)
        search_correlation = correlation[start_idx:end_idx]
        peak_idx = np.argmax(np.abs(search_correlation)) + start_idx
    else:
        peak_idx = np.argmax(np.abs(correlation))
    
    # Calculate delay
    delay = peak_idx - (len(signal1) - 1)
    
    return {
        'delay_samples': delay,
        'correlation_peak': correlation[peak_idx],
        'correlation_full': correlation,
        'confidence': np.abs(correlation[peak_idx]) / np.max(np.abs(correlation))
    }


def phase_lock_loop(signal: np.ndarray, bandwidth: float, 
                   damping: float = 0.707) -> Dict[str, Any]:
    """Simple Phase Lock Loop implementation"""
    
    # PLL parameters
    Kp = 2 * bandwidth  # Proportional gain
    Ki = bandwidth**2   # Integral gain
    
    # Initialize variables
    phase_error = np.zeros(len(signal))
    vco_phase = np.zeros(len(signal))
    vco_output = np.zeros(len(signal), dtype=complex)
    integrator = 0
    
    for i in range(len(signal)):
        # Phase detector
        phase_error[i] = np.angle(signal[i] * np.conj(vco_output[i-1] if i > 0 else 1))
        
        # Loop filter
        integrator += Ki * phase_error[i]
        control_voltage = Kp * phase_error[i] + integrator
        
        # VCO
        if i > 0:
            vco_phase[i] = vco_phase[i-1] + control_voltage
        vco_output[i] = np.exp(1j * vco_phase[i])
    
    return {
        'phase_error': phase_error,
        'vco_phase': vco_phase,
        'vco_output': vco_output,
        'locked_signal': signal * np.conj(vco_output)
    }


# Resampling and Interpolation
def resample_signal(signal: np.ndarray, original_rate: float, 
                   target_rate: float, method: str = "scipy") -> np.ndarray:
    """Resample signal to different sample rate"""
    
    if SDR_AVAILABLE and method == "sdr":
        try:
            if hasattr(sdr, 'resample'):
                return sdr.resample(signal, original_rate, target_rate)
        except Exception as e:
            logger.warning(f"SDR resampling failed: {e}")
    
    # Scipy resampling
    num_samples = int(len(signal) * target_rate / original_rate)
    return signal.resample(signal, num_samples)


def interpolate_signal(x: np.ndarray, factor: int, 
                      filter_type: str = "linear") -> np.ndarray:
    """Interpolate signal by integer factor"""
    
    if filter_type == "linear":
        # Simple linear interpolation
        interpolated = np.zeros(len(x) * factor, dtype=x.dtype)
        interpolated[::factor] = x
        
        # Apply anti-aliasing filter (adjust order based on signal length)
        cutoff = 0.8 / factor
        filter_order = min(8, len(interpolated) // 6)  # Ensure filter is not too long
        if filter_order < 1:
            return interpolated
        b, a = signal.butter(filter_order, cutoff)
        return signal.filtfilt(b, a, interpolated)
    
    elif filter_type == "cubic":
        # Cubic spline interpolation
        from scipy import interpolate
        
        # Need at least 4 points for cubic interpolation
        if len(x) < 4:
            # Fall back to linear interpolation
            factor_array = np.ones(len(x) * factor, dtype=x.dtype)
            factor_array[::factor] = x
            return factor_array
        
        x_orig = np.arange(len(x))
        x_new = np.linspace(0, len(x)-1, len(x)*factor)
        
        if np.iscomplexobj(x):
            real_interp = interpolate.interp1d(x_orig, x.real, kind='linear')  # Use linear for robustness
            imag_interp = interpolate.interp1d(x_orig, x.imag, kind='linear')
            return real_interp(x_new) + 1j * imag_interp(x_new)
        else:
            interp_func = interpolate.interp1d(x_orig, x, kind='linear')  # Use linear for robustness
            return interp_func(x_new)
    
    else:
        # Zero-order hold
        return np.repeat(x, factor)


def decimate_signal(x: np.ndarray, factor: int, 
                   filter_order: int = 8) -> np.ndarray:
    """Decimate signal by integer factor with anti-aliasing"""
    
    # Design anti-aliasing filter
    cutoff = 0.8 / factor
    b, a = signal.butter(filter_order, cutoff)
    
    # Apply filter and downsample
    filtered = signal.filtfilt(b, a, x)
    return filtered[::factor]


# Mathematical Utilities
def db_to_linear(db_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert dB to linear scale"""
    return 10**(db_value / 10)


def linear_to_db(linear_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert linear to dB scale"""
    return 10 * np.log10(np.abs(linear_value) + 1e-12)  # Small epsilon to avoid log(0)


def dbm_to_watts(dbm_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert dBm to watts"""
    return 10**((dbm_value - 30) / 10)


def watts_to_dbm(watts: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert watts to dBm"""
    return 10 * np.log10(watts * 1000)


def rms_value(signal: np.ndarray) -> float:
    """Calculate RMS value of signal"""
    return np.sqrt(np.mean(np.abs(signal)**2))


def peak_to_average_ratio(signal: np.ndarray) -> float:
    """Calculate Peak-to-Average Power Ratio (PAPR)"""
    peak_power = np.max(np.abs(signal)**2)
    avg_power = np.mean(np.abs(signal)**2)
    return peak_power / avg_power if avg_power > 0 else float('inf')


def crest_factor(signal: np.ndarray) -> float:
    """Calculate crest factor (peak-to-RMS ratio)"""
    peak = np.max(np.abs(signal))
    rms = rms_value(signal)
    return peak / rms if rms > 0 else float('inf')


# Performance Metrics
def calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> Dict[str, Any]:
    """Calculate Bit Error Rate"""
    
    if len(tx_bits) != len(rx_bits):
        min_len = min(len(tx_bits), len(rx_bits))
        tx_bits = tx_bits[:min_len]
        rx_bits = rx_bits[:min_len]
    
    errors = np.sum(tx_bits != rx_bits)
    ber = errors / len(tx_bits) if len(tx_bits) > 0 else 0
    
    return {
        'ber': ber,
        'error_count': errors,
        'total_bits': len(tx_bits),
        'error_positions': np.where(tx_bits != rx_bits)[0]
    }


def calculate_evm(reference: np.ndarray, measured: np.ndarray) -> Dict[str, Any]:
    """Calculate Error Vector Magnitude"""
    
    if len(reference) != len(measured):
        min_len = min(len(reference), len(measured))
        reference = reference[:min_len]
        measured = measured[:min_len]
    
    error_vector = measured - reference
    error_power = np.mean(np.abs(error_vector)**2)
    reference_power = np.mean(np.abs(reference)**2)
    
    evm_rms = np.sqrt(error_power / reference_power) if reference_power > 0 else float('inf')
    evm_peak = np.max(np.abs(error_vector)) / np.sqrt(reference_power) if reference_power > 0 else float('inf')
    
    return {
        'evm_rms_percent': evm_rms * 100,
        'evm_peak_percent': evm_peak * 100,
        'evm_rms_db': 20 * np.log10(evm_rms) if evm_rms > 0 else float('-inf'),
        'error_vector': error_vector
    }


def calculate_snr(signal: np.ndarray, noise: Optional[np.ndarray] = None,
                 signal_bandwidth: Optional[float] = None) -> Dict[str, Any]:
    """Calculate Signal-to-Noise Ratio"""
    
    if noise is not None:
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = np.mean(np.abs(noise)**2)
    else:
        # Estimate noise from signal (simple approach)
        # Assume noise is the difference between signal and its smoothed version
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(np.abs(signal), sigma=5)
        noise_estimate = np.abs(signal) - smoothed
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = np.mean(noise_estimate**2)
    
    snr_linear = signal_power / noise_power if noise_power > 0 else float('inf')
    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else float('inf')
    
    return {
        'snr_db': snr_db,
        'snr_linear': snr_linear,
        'signal_power': signal_power,
        'noise_power': noise_power
    }


# Convenience wrapper functions
def apply_window(signal: np.ndarray, window_type: str = "hann", **kwargs) -> np.ndarray:
    """Apply window function to signal"""
    window = create_window(window_type, len(signal), **kwargs)
    return signal * window


def normalize_signal(signal: np.ndarray, method: str = "peak") -> np.ndarray:
    """Normalize signal"""
    
    if method == "peak":
        return signal / np.max(np.abs(signal))
    elif method == "rms":
        return signal / rms_value(signal)
    elif method == "energy":
        return signal / np.sqrt(np.sum(np.abs(signal)**2))
    else:
        return signal


def zero_pad(signal: np.ndarray, target_length: int, mode: str = "center") -> np.ndarray:
    """Zero-pad signal to target length"""
    
    if len(signal) >= target_length:
        return signal[:target_length]
    
    pad_length = target_length - len(signal)
    
    if mode == "center":
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        return np.pad(signal, (pad_left, pad_right), mode='constant')
    elif mode == "end":
        return np.pad(signal, (0, pad_length), mode='constant')
    elif mode == "start":
        return np.pad(signal, (pad_length, 0), mode='constant')
    else:
        return np.pad(signal, (0, pad_length), mode='constant')


def circular_shift(signal: np.ndarray, shift: int) -> np.ndarray:
    """Circular shift of signal"""
    return np.roll(signal, shift)


def time_reverse(signal: np.ndarray) -> np.ndarray:
    """Time-reverse signal"""
    return signal[::-1]


def complex_conjugate(signal: np.ndarray) -> np.ndarray:
    """Complex conjugate of signal"""
    return np.conj(signal)


# File I/O utilities (if needed for future integration)
def save_complex_signal(signal: np.ndarray, filename: str, format: str = "npy"):
    """Save complex signal to file"""
    if format == "npy":
        np.save(filename, signal)
    elif format == "txt":
        np.savetxt(filename, np.column_stack([signal.real, signal.imag]))
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_complex_signal(filename: str, format: str = "npy") -> np.ndarray:
    """Load complex signal from file"""
    if format == "npy":
        return np.load(filename)
    elif format == "txt":
        data = np.loadtxt(filename)
        return data[:, 0] + 1j * data[:, 1]
    else:
        raise ValueError(f"Unsupported format: {format}")


# Error handling and validation
def validate_signal(signal: np.ndarray, name: str = "signal") -> bool:
    """Validate signal array"""
    if not isinstance(signal, np.ndarray):
        logger.error(f"{name} must be numpy array")
        return False
    
    if signal.size == 0:
        logger.error(f"{name} is empty")
        return False
    
    if not np.all(np.isfinite(signal)):
        logger.warning(f"{name} contains non-finite values")
        return False
    
    return True


# PySDR-inspired utility functions for signal processing optimization
def add_awgn(samples: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Additive White Gaussian Noise (AWGN) at specified SNR.
    
    Based on PySDR best practices for noise addition.
    
    Args:
        samples: Input signal samples (real or complex)
        snr_db: Desired Signal-to-Noise Ratio in dB
        
    Returns:
        Signal with added noise
        
    Example:
        >>> signal = np.exp(1j * 2 * np.pi * 0.1 * np.arange(100))
        >>> noisy_signal = add_awgn(signal, snr_db=10)
    """
    # Calculate signal power
    signal_power = np.mean(np.abs(samples)**2)
    
    # Calculate noise power from SNR
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate complex AWGN with proper variance
    if np.iscomplexobj(samples):
        # Complex noise: split power between I and Q
        noise = np.random.normal(0, np.sqrt(noise_power/2), len(samples)) + \
                1j * np.random.normal(0, np.sqrt(noise_power/2), len(samples))
    else:
        # Real noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(samples))
    
    return samples + noise


def estimate_snr(samples: np.ndarray, method: str = "moment") -> float:
    """
    Estimate Signal-to-Noise Ratio from signal samples.
    
    Implements multiple SNR estimation methods from PySDR.
    
    Args:
        samples: Input signal samples
        method: Estimation method ("moment", "split_spectrum", "percentile")
        
    Returns:
        Estimated SNR in dB
        
    Example:
        >>> signal = add_awgn(np.ones(1000), snr_db=15)
        >>> snr_est = estimate_snr(signal, method="moment")
    """
    if len(samples) == 0:
        return 0.0
    
    if method == "moment":
        # Second and fourth moment method
        m2 = np.mean(np.abs(samples)**2)
        m4 = np.mean(np.abs(samples)**4)
        
        if m2 == 0:
            return 0.0
        
        # For complex AWGN: SNR = sqrt(2*m2^2 / (m4 - 2*m2^2)) - 1
        snr_linear = np.sqrt(2 * m2**2 / (m4 - 2 * m2**2 + 1e-12)) - 1
        snr_linear = max(snr_linear, 1e-12)  # Avoid log of negative
        return 10 * np.log10(snr_linear)
    
    elif method == "split_spectrum":
        # Split spectrum method: signal in center, noise at edges
        fft_result = np.fft.fft(samples)
        power_spectrum = np.abs(fft_result)**2
        
        n = len(power_spectrum)
        # Signal power (center 50%)
        center_start = n // 4
        center_end = 3 * n // 4
        signal_power = np.mean(power_spectrum[center_start:center_end])
        
        # Noise power (edges 25% each side)
        noise_power = (np.mean(power_spectrum[:n//8]) + 
                      np.mean(power_spectrum[-n//8:])) / 2
        
        if noise_power == 0:
            return 60.0
        
        snr_linear = signal_power / noise_power
        return 10 * np.log10(snr_linear)
    
    elif method == "percentile":
        # Percentile-based method
        abs_samples = np.abs(samples)
        
        # 90th percentile as signal level
        signal_level = np.percentile(abs_samples, 90)
        # 10th percentile as noise level
        noise_level = np.percentile(abs_samples, 10)
        
        if noise_level == 0:
            return 60.0
        
        snr_linear = (signal_level / noise_level)**2
        return 10 * np.log10(snr_linear)
    
    else:
        # Default to moment method
        return estimate_snr(samples, method="moment")


def remove_dc_offset(samples: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from IQ samples.
    
    Simple and efficient DC removal as used in PySDR examples.
    
    Args:
        samples: Input IQ samples
        
    Returns:
        Samples with DC removed
        
    Example:
        >>> samples = np.array([1+1j, 2+2j, 3+3j])
        >>> corrected = remove_dc_offset(samples)
    """
    # Simply subtract the mean - very efficient
    return samples - np.mean(samples)


def normalize_power(samples: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    """
    Normalize signal to unit average power.
    
    Ensures consistent power levels as recommended in PySDR.
    
    Args:
        samples: Input signal samples
        target_power: Target average power (default 1.0)
        
    Returns:
        Power-normalized samples
        
    Example:
        >>> samples = np.random.randn(1000) + 1j*np.random.randn(1000)
        >>> normalized = normalize_power(samples)
        >>> np.mean(np.abs(normalized)**2)  # Should be ~1.0
    """
    current_power = np.mean(np.abs(samples)**2)
    
    if current_power == 0:
        return samples
    
    scale_factor = np.sqrt(target_power / current_power)
    return samples * scale_factor


def frequency_shift(samples: np.ndarray, freq_offset: float, sample_rate: float) -> np.ndarray:
    """
    Shift signal in frequency domain.
    
    Efficient frequency shifting as used in PySDR for frequency correction.
    
    Args:
        samples: Input signal samples
        freq_offset: Frequency offset in Hz (positive = shift up)
        sample_rate: Sample rate in Hz
        
    Returns:
        Frequency-shifted samples
        
    Example:
        >>> # Shift signal by 1 kHz
        >>> signal = np.exp(1j * 2 * np.pi * 5000 * np.arange(1000) / 48000)
        >>> shifted = frequency_shift(signal, 1000, 48000)
    """
    # Generate complex exponential for frequency shift
    t = np.arange(len(samples)) / sample_rate
    shift_signal = np.exp(2j * np.pi * freq_offset * t)
    
    return samples * shift_signal


def apply_agc(samples: np.ndarray, reference: float = 1.0, 
              attack: float = 0.01, release: float = 0.1) -> np.ndarray:
    """
    Apply Automatic Gain Control (AGC) to signal.
    
    Simple AGC implementation for consistent signal levels.
    
    Args:
        samples: Input signal samples
        reference: Target reference level
        attack: Attack time constant (0-1)
        release: Release time constant (0-1)
        
    Returns:
        AGC-applied samples
    """
    output = np.zeros_like(samples)
    gain = 1.0
    
    for i, sample in enumerate(samples):
        # Measure instantaneous amplitude
        amplitude = np.abs(sample)
        
        # Compute desired gain
        if amplitude > 0:
            desired_gain = reference / amplitude
        else:
            desired_gain = 1.0
        
        # Smooth gain adjustment
        if desired_gain < gain:
            # Attack (reduce gain quickly)
            gain = gain * (1 - attack) + desired_gain * attack
        else:
            # Release (increase gain slowly)
            gain = gain * (1 - release) + desired_gain * release
        
        # Apply gain
        output[i] = sample * gain
    
    return output


def compute_spectrogram_efficient(samples: np.ndarray, fft_size: int = 1024, 
                                  overlap: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram efficiently using PySDR row-by-row pattern.
    
    This is optimized for real-time processing as shown in PySDR examples.
    
    Args:
        samples: Input signal samples
        fft_size: FFT size
        overlap: Number of overlapping samples
        
    Returns:
        Tuple of (spectrogram 2D array, time axis, frequency axis)
        
    Example:
        >>> samples = np.random.randn(10000) + 1j*np.random.randn(10000)
        >>> spec, t, f = compute_spectrogram_efficient(samples, fft_size=1024)
    """
    hop_size = fft_size - overlap
    num_rows = (len(samples) - fft_size) // hop_size + 1
    
    # Pre-allocate spectrogram array
    spectrogram = np.zeros((num_rows, fft_size), dtype=np.float32)
    
    # Window for better spectral resolution
    window = np.hanning(fft_size)
    
    # Compute row by row (PySDR pattern)
    for i in range(num_rows):
        start_idx = i * hop_size
        end_idx = start_idx + fft_size
        
        if end_idx > len(samples):
            break
        
        # Extract segment and apply window
        segment = samples[start_idx:end_idx] * window
        
        # Compute FFT and shift
        fft_result = np.fft.fftshift(np.fft.fft(segment))
        
        # Convert to dB
        spectrogram[i, :] = 10 * np.log10(np.abs(fft_result)**2 + 1e-12)
    
    # Create time and frequency axes
    time_axis = np.arange(num_rows) * hop_size / fft_size
    freq_axis = np.linspace(-0.5, 0.5, fft_size)
    
    return spectrogram, time_axis, freq_axis


def downsample_efficient(samples: np.ndarray, decimation_factor: int) -> np.ndarray:
    """
    Efficiently downsample signal with anti-aliasing.
    
    Uses scipy's decimate for proper anti-aliasing filtering.
    
    Args:
        samples: Input signal samples
        decimation_factor: Decimation factor
        
    Returns:
        Downsampled signal
    """
    if decimation_factor == 1:
        return samples
    
    # Use scipy's decimate with default 8th-order Chebyshev filter
    return signal.decimate(samples, decimation_factor, zero_phase=True)


def upsample_efficient(samples: np.ndarray, interpolation_factor: int) -> np.ndarray:
    """
    Efficiently upsample signal with anti-imaging.
    
    Args:
        samples: Input signal samples
        interpolation_factor: Interpolation factor
        
    Returns:
        Upsampled signal
    """
    if interpolation_factor == 1:
        return samples
    
    # Zero-stuff
    upsampled = np.zeros(len(samples) * interpolation_factor, dtype=samples.dtype)
    upsampled[::interpolation_factor] = samples
    
    # Design anti-imaging filter
    cutoff = 0.8 / interpolation_factor  # Cutoff at 80% of new Nyquist
    taps = signal.firwin(64, cutoff)
    
    # Apply filter and scale
    return signal.lfilter(taps, 1.0, upsampled) * interpolation_factor