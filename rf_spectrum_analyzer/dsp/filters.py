"""
Advanced Digital Filters
Comprehensive filter implementations using sdr and sk_dsp_comm libraries
"""

import numpy as np
import scipy.signal as signal
from scipy import linalg
from typing import Optional, Tuple, List, Union, Dict, Any
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
    from sdr._filter import *
    SDR_AVAILABLE = True
except ImportError:
    SDR_AVAILABLE = False
    warnings.warn("SDR library not available. Some features disabled.")

# Import from sk_dsp_comm
try:
    import sk_dsp_comm.fir_design_helper as fir_helper
    import sk_dsp_comm.iir_design_helper as iir_helper
    import sk_dsp_comm.sigsys as ss
    SK_DSP_AVAILABLE = True
except ImportError:
    SK_DSP_AVAILABLE = False
    warnings.warn("sk_dsp_comm library not available. Some features disabled.")

from rf_spectrum_analyzer.utils.logger import get_logger

logger = get_logger('filters')


@dataclass
class FilterConfig:
    """Filter configuration parameters"""
    filter_type: str = "lowpass"
    cutoff_freq: Union[float, List[float]] = 0.25
    sample_rate: float = 1.0
    order: int = 5
    ripple: float = 0.5
    attenuation: float = 60.0
    window: str = "hamming"
    method: str = "butter"


class FIRFilter:
    """Advanced FIR Filter implementation"""
    
    def __init__(self, coefficients: Optional[np.ndarray] = None, **kwargs):
        self.coefficients = coefficients
        self.delay_line = None
        self.config = FilterConfig(**kwargs)
        
        if coefficients is None:
            self.design_filter()
        
        self.reset()
    
    def design_filter(self):
        """Design FIR filter based on configuration"""
        config = self.config
        
        if SDR_AVAILABLE:
            self._design_with_sdr()
        elif SK_DSP_AVAILABLE:
            self._design_with_sk_dsp()
        else:
            self._design_with_scipy()
    
    def _design_with_sdr(self):
        """Design filter using SDR library"""
        config = self.config
        
        try:
            if hasattr(sdr, '_filter'):
                if config.filter_type == "lowpass":
                    self.coefficients = sdr._filter.design_fir.lowpass(
                        cutoff=config.cutoff_freq,
                        length=config.order + 1,
                        window=config.window
                    )
                elif config.filter_type == "highpass":
                    self.coefficients = sdr._filter.design_fir.highpass(
                        cutoff=config.cutoff_freq,
                        length=config.order + 1,
                        window=config.window
                    )
                elif config.filter_type == "bandpass":
                    if isinstance(config.cutoff_freq, (list, tuple)) and len(config.cutoff_freq) == 2:
                        self.coefficients = sdr._filter.design_fir.bandpass(
                            cutoff=config.cutoff_freq,
                            length=config.order + 1,
                            window=config.window
                        )
                elif config.filter_type == "bandstop":
                    if isinstance(config.cutoff_freq, (list, tuple)) and len(config.cutoff_freq) == 2:
                        self.coefficients = sdr._filter.design_fir.bandstop(
                            cutoff=config.cutoff_freq,
                            length=config.order + 1,
                            window=config.window
                        )
                else:
                    # Unknown filter type - fall back to scipy
                    self._design_with_scipy()
                    return
            else:
                self._design_with_scipy()
        except Exception as e:
            logger.warning(f"Error designing filter with SDR: {e}")
            self._design_with_scipy()
    
    def _design_with_sk_dsp(self):
        """Design filter using sk_dsp_comm library"""
        config = self.config
        
        try:
            if config.filter_type == "lowpass":
                self.coefficients = fir_helper.lowpass(
                    f_pass=config.cutoff_freq * 0.8,
                    f_stop=config.cutoff_freq * 1.2,
                    d_stop=10**(-config.attenuation/20),
                    fs=config.sample_rate,
                    win_type=config.window
                )
            elif config.filter_type == "bandpass":
                if isinstance(config.cutoff_freq, (list, tuple)) and len(config.cutoff_freq) == 2:
                    self.coefficients = fir_helper.bandpass(
                        f_pass=config.cutoff_freq,
                        f_stop=[config.cutoff_freq[0] * 0.8, config.cutoff_freq[1] * 1.2],
                        d_stop=10**(-config.attenuation/20),
                        fs=config.sample_rate,
                        win_type=config.window
                    )
            else:
                # Unknown filter type - fall back to scipy
                self._design_with_scipy()
                return
        except Exception as e:
            logger.warning(f"Error designing filter with sk_dsp_comm: {e}")
            self._design_with_scipy()
    
    def _design_with_scipy(self):
        """Design filter using scipy"""
        config = self.config
        
        try:
            if config.filter_type == "lowpass":
                self.coefficients = signal.firwin(
                    config.order + 1, 
                    config.cutoff_freq,
                    window=config.window,
                    fs=config.sample_rate
                )
            elif config.filter_type == "highpass":
                self.coefficients = signal.firwin(
                    config.order + 1, 
                    config.cutoff_freq,
                    pass_zero=False,
                    window=config.window,
                    fs=config.sample_rate
                )
            elif config.filter_type == "bandpass":
                if isinstance(config.cutoff_freq, (list, tuple)) and len(config.cutoff_freq) == 2:
                    self.coefficients = signal.firwin(
                        config.order + 1,
                        config.cutoff_freq,
                        pass_zero=False,
                        window=config.window,
                        fs=config.sample_rate
                    )
            elif config.filter_type == "bandstop":
                if isinstance(config.cutoff_freq, (list, tuple)) and len(config.cutoff_freq) == 2:
                    self.coefficients = signal.firwin(
                        config.order + 1,
                        config.cutoff_freq,
                        window=config.window,
                        fs=config.sample_rate
                    )
            else:
                # Invalid filter type - use default lowpass
                logger.warning(f"Unknown filter type '{config.filter_type}', using lowpass")
                self.coefficients = signal.firwin(config.order + 1, 0.25)
        except Exception as e:
            logger.error(f"Error designing filter with scipy: {e}")
            # Fallback to simple lowpass
            self.coefficients = signal.firwin(config.order + 1, 0.25)
    
    def reset(self):
        """Reset filter state"""
        if self.coefficients is not None:
            self.delay_line = np.zeros(len(self.coefficients) - 1, dtype=complex)
    
    def filter(self, x: np.ndarray) -> np.ndarray:
        """Filter input signal"""
        if self.coefficients is None:
            return x
        
        # Handle empty signals
        if len(x) == 0:
            return np.array([])
            
        return signal.lfilter(self.coefficients, 1, x)
    
    def filter_streaming(self, x: np.ndarray) -> np.ndarray:
        """Filter with state preservation for streaming"""
        if self.coefficients is None:
            return x
            
        y, self.delay_line = signal.lfilter(
            self.coefficients, 1, x, zi=self.delay_line
        )
        return y
    
    def get_frequency_response(self, n_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Get filter frequency response"""
        if self.coefficients is None:
            return np.array([]), np.array([])
            
        w, h = signal.freqz(self.coefficients, worN=n_points, fs=self.config.sample_rate)
        return w, h


class IIRFilter:
    """Advanced IIR Filter implementation"""
    
    def __init__(self, b: Optional[np.ndarray] = None, a: Optional[np.ndarray] = None, **kwargs):
        self.b = b  # Numerator coefficients
        self.a = a  # Denominator coefficients
        self.zi = None  # Filter state
        self.config = FilterConfig(**kwargs)
        
        if b is None or a is None:
            self.design_filter()
        
        self.reset()
    
    def design_filter(self):
        """Design IIR filter based on configuration"""
        config = self.config
        
        if SK_DSP_AVAILABLE:
            self._design_with_sk_dsp()
        else:
            self._design_with_scipy()
    
    def _design_with_sk_dsp(self):
        """Design filter using sk_dsp_comm library"""
        config = self.config
        
        try:
            if config.method == "butter":
                self.b, self.a = iir_helper.butter_ba(
                    N=config.order,
                    f_c=config.cutoff_freq,
                    btype=config.filter_type,
                    fs=config.sample_rate
                )
            elif config.method == "cheby1":
                self.b, self.a = iir_helper.cheby1_ba(
                    N=config.order,
                    r_p=config.ripple,
                    f_c=config.cutoff_freq,
                    btype=config.filter_type,
                    fs=config.sample_rate
                )
            elif config.method == "cheby2":
                self.b, self.a = iir_helper.cheby2_ba(
                    N=config.order,
                    r_s=config.attenuation,
                    f_c=config.cutoff_freq,
                    btype=config.filter_type,
                    fs=config.sample_rate
                )
            elif config.method == "ellip":
                self.b, self.a = iir_helper.ellip_ba(
                    N=config.order,
                    r_p=config.ripple,
                    r_s=config.attenuation,
                    f_c=config.cutoff_freq,
                    btype=config.filter_type,
                    fs=config.sample_rate
                )
        except Exception as e:
            logger.warning(f"Error designing IIR filter with sk_dsp_comm: {e}")
            self._design_with_scipy()
    
    def _design_with_scipy(self):
        """Design filter using scipy"""
        config = self.config
        
        try:
            if config.method == "butter":
                self.b, self.a = signal.butter(
                    config.order, config.cutoff_freq, 
                    btype=config.filter_type, fs=config.sample_rate
                )
            elif config.method == "cheby1":
                self.b, self.a = signal.cheby1(
                    config.order, config.ripple, config.cutoff_freq,
                    btype=config.filter_type, fs=config.sample_rate
                )
            elif config.method == "cheby2":
                self.b, self.a = signal.cheby2(
                    config.order, config.attenuation, config.cutoff_freq,
                    btype=config.filter_type, fs=config.sample_rate
                )
            elif config.method == "ellip":
                self.b, self.a = signal.elliptic(
                    config.order, config.ripple, config.attenuation, config.cutoff_freq,
                    btype=config.filter_type, fs=config.sample_rate
                )
        except Exception as e:
            logger.error(f"Error designing IIR filter with scipy: {e}")
            # Fallback to simple Butterworth lowpass
            self.b, self.a = signal.butter(2, 0.25)
    
    def reset(self):
        """Reset filter state"""
        if self.b is not None and self.a is not None:
            self.zi = signal.lfilter_zi(self.b, self.a)
    
    def filter(self, x: np.ndarray) -> np.ndarray:
        """Filter input signal"""
        if self.b is None or self.a is None:
            return x
            
        return signal.lfilter(self.b, self.a, x)
    
    def filter_streaming(self, x: np.ndarray) -> np.ndarray:
        """Filter with state preservation for streaming"""
        if self.b is None or self.a is None:
            return x
            
        y, self.zi = signal.lfilter(self.b, self.a, x, zi=self.zi)
        return y
    
    def get_frequency_response(self, n_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Get filter frequency response"""
        if self.b is None or self.a is None:
            return np.array([]), np.array([])
            
        w, h = signal.freqz(self.b, self.a, worN=n_points, fs=self.config.sample_rate)
        return w, h


class PolyphaseFilter:
    """Polyphase filter for multirate processing"""
    
    def __init__(self, prototype_filter: np.ndarray, M: int, mode: str = "interpolation"):
        """
        Initialize polyphase filter
        
        Args:
            prototype_filter: Prototype filter coefficients
            M: Factor for interpolation/decimation
            mode: "interpolation", "decimation", or "resampling"
        """
        self.prototype = prototype_filter
        self.M = M
        self.mode = mode
        self.polyphase_filters = self._create_polyphase_filters()
        self.delay_lines = [np.array([]) for _ in range(M)]
        
        if SDR_AVAILABLE:
            self._init_sdr_filters()
    
    def _create_polyphase_filters(self) -> List[np.ndarray]:
        """Create polyphase filter bank"""
        L = len(self.prototype)
        filters = []
        
        for m in range(self.M):
            # Extract every M-th coefficient starting from m
            poly_coeffs = self.prototype[m::self.M]
            filters.append(poly_coeffs)
            
        return filters
    
    def _init_sdr_filters(self):
        """Initialize SDR polyphase filters if available"""
        if not SDR_AVAILABLE:
            return
            
        try:
            # Use SDR library's multirate filters if available
            if hasattr(sdr, '_filter') and hasattr(sdr._filter, 'multirate'):
                if self.mode == "interpolation":
                    self.sdr_filter = sdr._filter.multirate.Interpolator(self.M, self.prototype)
                elif self.mode == "decimation":
                    self.sdr_filter = sdr._filter.multirate.Decimator(self.M, self.prototype)
                elif self.mode == "resampling":
                    self.sdr_filter = sdr._filter.multirate.Resampler(self.M, 1, self.prototype)
        except Exception as e:
            logger.warning(f"Error initializing SDR polyphase filter: {e}")
            self.sdr_filter = None
    
    def interpolate(self, x: np.ndarray) -> np.ndarray:
        """Interpolate signal by factor M"""
        if SDR_AVAILABLE and hasattr(self, 'sdr_filter') and self.mode == "interpolation":
            try:
                return self.sdr_filter(x)
            except:
                pass
        
        # Manual implementation
        return self._manual_interpolate(x)
    
    def decimate(self, x: np.ndarray) -> np.ndarray:
        """Decimate signal by factor M"""
        if SDR_AVAILABLE and hasattr(self, 'sdr_filter') and self.mode == "decimation":
            try:
                return self.sdr_filter(x)
            except:
                pass
        
        # Manual implementation
        return self._manual_decimate(x)
    
    def _manual_interpolate(self, x: np.ndarray) -> np.ndarray:
        """Manual interpolation implementation"""
        # Upsample by inserting zeros
        upsampled = np.zeros(len(x) * self.M, dtype=x.dtype)
        upsampled[::self.M] = x
        
        # Apply anti-aliasing filter
        return signal.lfilter(self.prototype * self.M, 1, upsampled)
    
    def _manual_decimate(self, x: np.ndarray) -> np.ndarray:
        """Manual decimation implementation"""
        # Apply anti-aliasing filter
        filtered = signal.lfilter(self.prototype, 1, x)
        
        # Downsample
        return filtered[::self.M]


class AdaptiveFilter:
    """Adaptive filter using LMS/RLS algorithms"""
    
    def __init__(self, length: int, algorithm: str = "lms", mu: float = 0.01):
        """
        Initialize adaptive filter
        
        Args:
            length: Filter length
            algorithm: "lms" or "rls"
            mu: Adaptation step size (for LMS)
        """
        self.length = length
        self.algorithm = algorithm
        self.mu = mu
        self.weights = np.zeros(length, dtype=complex)
        self.error_history = []
        
        # RLS-specific parameters
        if algorithm == "rls":
            self.lambda_rls = 0.99  # Forgetting factor
            self.P = np.eye(length) / 0.01  # Inverse correlation matrix
    
    def adapt(self, x: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adapt filter weights
        
        Args:
            x: Input signal
            d: Desired signal
            
        Returns:
            y: Filter output
            e: Error signal
        """
        y = np.zeros_like(d)
        e = np.zeros_like(d)
        
        for i in range(len(x) - self.length + 1):
            x_window = x[i:i + self.length]
            
            # Filter output
            y_i = np.dot(self.weights.conj(), x_window)
            y[i + self.length - 1] = y_i
            
            # Error
            e_i = d[i + self.length - 1] - y_i
            e[i + self.length - 1] = e_i
            
            # Update weights
            if self.algorithm == "lms":
                self.weights += self.mu * e_i.conj() * x_window
            elif self.algorithm == "rls":
                self._rls_update(x_window, e_i)
                
            self.error_history.append(abs(e_i))
        
        return y, e
    
    def _rls_update(self, x: np.ndarray, error: complex):
        """RLS weight update"""
        # Gain vector
        k = self.P @ x / (self.lambda_rls + x.conj().T @ self.P @ x)
        
        # Update weights
        self.weights += k * error.conj()
        
        # Update inverse correlation matrix
        self.P = (self.P - np.outer(k, x.conj().T @ self.P)) / self.lambda_rls


# Convenience classes for common filter types
class ButterworthFilter(IIRFilter):
    """Butterworth IIR Filter"""
    
    def __init__(self, order: int, cutoff: Union[float, List[float]], 
                 filter_type: str = "lowpass", sample_rate: float = 1.0):
        super().__init__(
            filter_type=filter_type,
            cutoff_freq=cutoff,
            order=order,
            method="butter",
            sample_rate=sample_rate
        )


class ChebyshevFilter(IIRFilter):
    """Chebyshev IIR Filter"""
    
    def __init__(self, order: int, cutoff: Union[float, List[float]], 
                 ripple: float = 0.5, filter_type: str = "lowpass", 
                 sample_rate: float = 1.0, cheby_type: int = 1):
        method = "cheby1" if cheby_type == 1 else "cheby2"
        super().__init__(
            filter_type=filter_type,
            cutoff_freq=cutoff,
            order=order,
            method=method,
            ripple=ripple,
            sample_rate=sample_rate
        )


class EllipticFilter(IIRFilter):
    """Elliptic IIR Filter"""
    
    def __init__(self, order: int, cutoff: Union[float, List[float]], 
                 ripple: float = 0.5, attenuation: float = 60.0,
                 filter_type: str = "lowpass", sample_rate: float = 1.0):
        super().__init__(
            filter_type=filter_type,
            cutoff_freq=cutoff,
            order=order,
            method="ellip",
            ripple=ripple,
            attenuation=attenuation,
            sample_rate=sample_rate
        )


# Convenience functions for common filter designs
def design_lowpass(cutoff: float, order: int = 5, method: str = "butter", 
                  filter_type: str = "iir", sample_rate: float = 1.0) -> Union[FIRFilter, IIRFilter]:
    """Design lowpass filter"""
    config = FilterConfig(
        filter_type="lowpass",
        cutoff_freq=cutoff,
        order=order,
        method=method,
        sample_rate=sample_rate
    )
    
    if filter_type == "fir":
        return FIRFilter(**config.__dict__)
    else:
        return IIRFilter(**config.__dict__)


def design_highpass(cutoff: float, order: int = 5, method: str = "butter",
                   filter_type: str = "iir", sample_rate: float = 1.0) -> Union[FIRFilter, IIRFilter]:
    """Design highpass filter"""
    config = FilterConfig(
        filter_type="highpass",
        cutoff_freq=cutoff,
        order=order,
        method=method,
        sample_rate=sample_rate
    )
    
    if filter_type == "fir":
        return FIRFilter(**config.__dict__)
    else:
        return IIRFilter(**config.__dict__)


def design_bandpass(low_cutoff: float, high_cutoff: float, order: int = 5,
                   method: str = "butter", filter_type: str = "iir",
                   sample_rate: float = 1.0) -> Union[FIRFilter, IIRFilter]:
    """Design bandpass filter"""
    config = FilterConfig(
        filter_type="bandpass",
        cutoff_freq=[low_cutoff, high_cutoff],
        order=order,
        method=method,
        sample_rate=sample_rate
    )
    
    if filter_type == "fir":
        return FIRFilter(**config.__dict__)
    else:
        return IIRFilter(**config.__dict__)


def design_bandstop(low_cutoff: float, high_cutoff: float, order: int = 5,
                   method: str = "butter", filter_type: str = "iir",
                   sample_rate: float = 1.0) -> Union[FIRFilter, IIRFilter]:
    """Design bandstop filter"""
    config = FilterConfig(
        filter_type="bandstop",
        cutoff_freq=[low_cutoff, high_cutoff],
        order=order,
        method=method,
        sample_rate=sample_rate
    )
    
    if filter_type == "fir":
        return FIRFilter(**config.__dict__)
    else:
        return IIRFilter(**config.__dict__)


def design_fm_deemphasis(sample_rate: float, time_constant: float = 75e-6,
                        region: str = "americas") -> IIRFilter:
    """
    Design FM de-emphasis filter for FM audio (PySDR best practice).
    
    De-emphasis filter compensates for pre-emphasis applied at transmitter.
    Standard time constants:
    - Americas/Korea: 75μs
    - Europe/Australia: 50μs
    
    Args:
        sample_rate: Sample rate in Hz
        time_constant: Time constant in seconds (75e-6 for Americas, 50e-6 for Europe)
        region: "americas" (75μs) or "europe" (50μs)
        
    Returns:
        IIRFilter configured as de-emphasis filter
        
    Example:
        >>> # For US FM broadcast
        >>> deemph = design_fm_deemphasis(48000, region="americas")
        >>> audio_out = deemph.filter(fm_demodulated)
    """
    # Set time constant based on region if not explicitly specified
    if time_constant == 75e-6:
        if region.lower() in ["europe", "australia"]:
            time_constant = 50e-6
    elif region.lower() == "americas":
        time_constant = 75e-6
    elif region.lower() in ["europe", "australia"]:
        time_constant = 50e-6
    
    # Design de-emphasis filter using bilinear transform
    # Transfer function: H(s) = 1 / (1 + s*tau)
    # where tau is the time constant
    from scipy.signal import bilinear
    
    # Analog filter coefficients
    b_analog = [1]
    a_analog = [time_constant, 1]
    
    # Convert to digital using bilinear transform
    b_digital, a_digital = bilinear(b_analog, a_analog, fs=sample_rate)
    
    # Create IIR filter
    filter_obj = IIRFilter(b=b_digital, a=a_digital, sample_rate=sample_rate)
    
    logger.info(f"Created FM de-emphasis filter: tau={time_constant*1e6:.0f}μs, fs={sample_rate/1e3:.1f}kHz")
    
    return filter_obj


def create_streaming_filter(b: np.ndarray, a: Union[float, np.ndarray] = 1.0) -> Dict[str, Any]:
    """
    Create streaming filter state for continuous operation (PySDR pattern).
    
    This uses scipy's lfilter_zi to maintain filter state between chunks,
    enabling continuous filtering without transient artifacts.
    
    Args:
        b: Numerator coefficients (FIR) or both for IIR
        a: Denominator coefficients (IIR), default 1.0 for FIR
        
    Returns:
        Dictionary with 'b', 'a', and 'zi' (filter state)
        
    Example:
        >>> # Create streaming filter
        >>> b = scipy.signal.firwin(64, 0.1)
        >>> stream_filter = create_streaming_filter(b)
        >>> 
        >>> # Process chunks
        >>> for chunk in data_chunks:
        >>>     filtered, stream_filter['zi'] = scipy.signal.lfilter(
        >>>         stream_filter['b'], stream_filter['a'], chunk, zi=stream_filter['zi'])
    """
    from scipy.signal import lfilter_zi
    
    # Ensure a is array
    if isinstance(a, (int, float)):
        a = np.array([a])
    else:
        a = np.array(a)
    
    b = np.array(b)
    
    # Initialize filter state
    zi = lfilter_zi(b, a)
    
    return {
        'b': b,
        'a': a,
        'zi': zi
    }


def apply_streaming_filter(stream_filter: Dict[str, Any], data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply streaming filter to data chunk and update state.
    
    Args:
        stream_filter: Filter dictionary from create_streaming_filter
        data: Input data chunk
        
    Returns:
        Tuple of (filtered_data, updated_filter_dict)
        
    Example:
        >>> b = scipy.signal.firwin(64, 0.1)
        >>> stream_filter = create_streaming_filter(b)
        >>> 
        >>> # Process first chunk
        >>> filtered1, stream_filter = apply_streaming_filter(stream_filter, chunk1)
        >>> 
        >>> # Process second chunk (maintains continuity)
        >>> filtered2, stream_filter = apply_streaming_filter(stream_filter, chunk2)
    """
    from scipy.signal import lfilter
    
    # Apply filter with state
    filtered, zi_new = lfilter(stream_filter['b'], stream_filter['a'], data, zi=stream_filter['zi'])
    
    # Update state
    stream_filter['zi'] = zi_new
    
    return filtered, stream_filter