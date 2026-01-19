"""
Signal Processor - Core DSP functionality
Integrates multiple DSP libraries for comprehensive signal analysis.
"""

import numpy as np
import scipy.signal
import logging
import os
import time
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

# Import DSP libraries based on requirements
try:
    import sdr
    SDR_AVAILABLE = True
except ImportError:
    SDR_AVAILABLE = False
    logging.warning("sdr library not available")

try:
    import sk_dsp_comm.sigsys as sigsys
    import sk_dsp_comm.digitalcom as digitalcom
    import sk_dsp_comm.fir_design_helper as fir_design_helper
    import sk_dsp_comm.iir_design_helper as iir_design_helper
    SCIKIT_DSP_AVAILABLE = True
except ImportError:
    SCIKIT_DSP_AVAILABLE = False
    logging.warning("scikit-dsp-comm library not available")

try:
    import pyfftw
    PYFFTW_AVAILABLE = True
except ImportError:
    PYFFTW_AVAILABLE = False
    logging.warning("pyfftw not available, using numpy FFT")

from rf_spectrum_analyzer.config.settings import Settings
from rf_spectrum_analyzer.dsp.modulation_analysis import create_modulation_analyzer, create_encoding_analyzer
from rf_spectrum_analyzer.dsp.demodulation_engine import create_demodulation_engine
from rf_spectrum_analyzer.dsp.decoding_engine import create_decoding_engine
from rf_spectrum_analyzer.dsp.signal_detection import create_signal_detector
from rf_spectrum_analyzer.dsp.tdma_detector import TDMABurstDetector
from rf_spectrum_analyzer.dsp.enhanced_analysis import EnhancedSignalAnalysis

logger = logging.getLogger(__name__)


@dataclass
class SpectrumData:
    """Container for spectrum analysis results."""
    frequencies: np.ndarray
    power_db: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    sample_rate: float
    center_frequency: float
    fft_size: int
    window_type: str


@dataclass
class SignalStats:
    """Container for signal statistics."""
    mean_power: float
    peak_power: float
    snr_estimate: float
    dc_offset: complex
    iq_imbalance: float
    rms_level: float


class SignalProcessor:
    """Main signal processing class integrating multiple DSP libraries."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Processing parameters
        self.fft_size = settings.dsp.fft_size
        self.window_type = settings.dsp.window_type
        self.overlap = settings.dsp.overlap
        self.averaging = settings.dsp.averaging
        
        # Window function
        self.window = self._create_window(self.window_type, self.fft_size)
        
        # FFT setup
        self._setup_fft()
        
        # Averaging buffers
        self.spectrum_history = []
        
        # Filter objects
        self.current_filter = None
        
        # Demodulator objects
        self.demodulator = None
        
        # Modulation analysis components
        self.modulation_analyzer = create_modulation_analyzer(settings.sdr.sample_rate)
        self.encoding_analyzer = create_encoding_analyzer()
        self.demodulation_engine = create_demodulation_engine(settings.sdr.sample_rate)
        self.decoding_engine = create_decoding_engine()
        
        # Signal detection components
        self.signal_detector = create_signal_detector(settings.sdr.sample_rate)
        self.tdma_detector = TDMABurstDetector(settings.sdr.sample_rate)
        
        # Enhanced analysis engine (with sdrconnect integration)
        self.enhanced_analyzer = EnhancedSignalAnalysis(
            sample_rate=settings.sdr.sample_rate,
            fft_size=self.fft_size
        )
        
        # Analysis results cache
        self.last_modulation_analysis = None
        self.last_encoding_analysis = None
        self.last_detection_result = None
        self.last_tdma_analysis = None
        self.last_enhanced_analysis = None
        
        # Detection state variables
        self._current_iq_data = None
        self._auto_detection_enabled = False
        self._advanced_analysis_enabled = False
        self._detection_threshold_dbm = -80.0
        self._detection_interval_ms = 100
        
        # Performance optimization variables
        self._last_spectrum_time = 0
        self._spectrum_skip_counter = 0
        self._adaptive_update_rate = 20  # Target updates per second
        self._min_update_interval = 1.0 / 30  # Maximum 30 FPS
        self._performance_mode = 'balanced'  # 'fast', 'balanced', 'quality'
        
        self.logger.info("Signal processor initialized")
    
    def _setup_fft(self):
        """Setup optimized FFT computation with pyFFTW and PySDR best practices."""
        # Create window
        self.window = self._create_window(self.settings.dsp.window_type, self.fft_size)
        
        # Calculate window correction factors (PySDR best practice)
        self.window_coherent_gain = len(self.window) / np.sum(self.window)
        self.window_enbw = len(self.window) * np.sum(self.window**2) / (np.sum(self.window)**2)
        
        # Check if FFT size is power of 2 for np.roll optimization
        self._use_roll_for_shift = (self.fft_size & (self.fft_size - 1)) == 0
        
        # Setup averaging with pre-allocated arrays
        self.averaging = self.settings.dsp.averaging
        self.spectrum_history = []
        self._spectrum_sum = None  # Pre-allocated sum array for averaging
        self._exponential_avg_buffer = None  # For exponential moving average
        self._ema_alpha = 2.0 / (self.averaging + 1)  # EMA smoothing factor
        
        # Setup optimized FFTW if available
        if PYFFTW_AVAILABLE:
            try:
                # Enable FFTW wisdom for optimized FFT plans
                pyfftw.config.NUM_THREADS = min(4, os.cpu_count() or 1)  # Limit threads for GUI responsiveness
                pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
                
                # Create cache-aligned arrays for optimal performance
                self.fft_input = pyfftw.empty_aligned(self.fft_size, dtype='complex64', 
                                                    n=pyfftw.simd_alignment)
                self.fft_output = pyfftw.empty_aligned(self.fft_size, dtype='complex64',
                                                     n=pyfftw.simd_alignment)
                
                # Pre-allocate windowed samples array
                self.windowed_samples = pyfftw.empty_aligned(self.fft_size, dtype='complex64',
                                                           n=pyfftw.simd_alignment)
                
                # Create optimized FFTW object with wisdom
                self.fft_object = pyfftw.FFTW(self.fft_input, 
                                            self.fft_output, 
                                            direction='FFTW_FORWARD', 
                                            flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'),
                                            threads=pyfftw.config.NUM_THREADS)
                
                # Pre-allocate power spectrum arrays
                self._power_spectrum = np.empty(self.fft_size, dtype=np.float32)
                self._power_db = np.empty(self.fft_size, dtype=np.float32)
                
                self.fft_method = 'pyfftw_optimized'
                self.logger.info(f"Using optimized FFTW with {pyfftw.config.NUM_THREADS} threads for FFT computation")
                
            except Exception as e:
                self.logger.warning(f"Failed to setup optimized FFTW: {e}, falling back to basic FFTW")
                try:
                    # Fallback to basic FFTW setup
                    self.fft_input = pyfftw.empty_aligned(self.fft_size, dtype='complex64')
                    self.fft_output = pyfftw.empty_aligned(self.fft_size, dtype='complex64')
                    
                    self.fft_object = pyfftw.FFTW(self.fft_input, 
                                                self.fft_output, 
                                                direction='FFTW_FORWARD', 
                                                flags=('FFTW_MEASURE',))
                    self.fft_method = 'pyfftw'
                    self.logger.info("Using basic FFTW for FFT computation")
                except Exception as e2:
                    self.logger.warning(f"Failed to setup basic FFTW: {e2}, falling back to numpy")
                    self.fft_method = 'numpy'
        else:
            self.fft_method = 'numpy'
            self.logger.info("Using numpy for FFT computation")
            self.logger.info("Using numpy for FFT computation")
    
    def _create_window(self, window_type: str, size: int) -> np.ndarray:
        """Create window function."""
        window_functions = {
            'hann': scipy.signal.windows.hann,
            'hamming': scipy.signal.windows.hamming,
            'blackman': scipy.signal.windows.blackman,
            'bartlett': scipy.signal.windows.bartlett,
            'kaiser': lambda n: scipy.signal.windows.kaiser(n, beta=8.6),
        }
        
        if window_type in window_functions:
            return window_functions[window_type](size)
        else:
            self.logger.warning(f"Unknown window type {window_type}, using Hann")
            return scipy.signal.windows.hann(size)
    
    def compute_spectrum(self, iq_samples: np.ndarray) -> Optional[np.ndarray]:
        """Compute power spectrum from IQ samples with optimized FFT and adaptive throttling."""
        try:
            # Check if we should skip this frame for performance
            current_time = time.time()
            time_since_last = current_time - self._last_spectrum_time
            
            # Adaptive throttling based on performance mode
            if self._performance_mode == 'fast':
                min_interval = 1.0 / 15  # 15 FPS max
            elif self._performance_mode == 'balanced':
                min_interval = 1.0 / 20  # 20 FPS max  
            else:  # quality mode
                min_interval = 1.0 / 30  # 30 FPS max
            
            # Skip frame if updating too frequently
            if time_since_last < min_interval:
                self._spectrum_skip_counter += 1
                return None
                
            self._last_spectrum_time = current_time
            if self._spectrum_skip_counter > 0:
                self.logger.debug(f"Skipped {self._spectrum_skip_counter} frames for performance")
                self._spectrum_skip_counter = 0
            
            if len(iq_samples) < self.fft_size:
                return None
            
            # Store IQ data for sequential workflow
            self._current_iq_data = iq_samples.copy()
            
            # Extract samples for FFT (ensure complex64 for optimal performance)
            samples = iq_samples[:self.fft_size].astype(np.complex64)
            
            # Compute FFT with optimized path
            if self.fft_method == 'pyfftw_optimized':
                # Optimized FFTW path with pre-allocated arrays
                np.multiply(samples, self.window, out=self.windowed_samples)
                self.fft_input[:] = self.windowed_samples
                self.fft_object()
                
                # Compute power spectrum in-place for better memory efficiency
                np.abs(self.fft_output, out=self._power_spectrum)
                np.square(self._power_spectrum, out=self._power_spectrum)
                
                # Convert to dB efficiently
                np.maximum(self._power_spectrum, 1e-12, out=self._power_spectrum)  # Prevent log(0)
                np.log10(self._power_spectrum, out=self._power_db)
                self._power_db *= 10.0
                
                # Use np.roll for power-of-2 FFT sizes (PySDR optimization)
                if self._use_roll_for_shift:
                    power_db_shifted = np.roll(self._power_db, self.fft_size // 2)
                else:
                    power_db_shifted = np.fft.fftshift(self._power_db)
                
            elif self.fft_method == 'pyfftw':
                # Basic FFTW path
                windowed_samples = samples * self.window
                self.fft_input[:] = windowed_samples
                self.fft_object()
                fft_result = self.fft_output.copy()
                
                # Compute power spectrum
                power_spectrum = np.abs(fft_result) ** 2
                power_db = 10 * np.log10(power_spectrum + 1e-12)
                power_db_shifted = np.fft.fftshift(power_db)
                
            else:
                # NumPy fallback path
                windowed_samples = samples * self.window
                fft_result = np.fft.fft(windowed_samples)
                power_spectrum = np.abs(fft_result) ** 2
                power_db = 10 * np.log10(power_spectrum + 1e-12)
                power_db_shifted = np.fft.fftshift(power_db)
            
            # Apply optimized averaging if enabled
            if self.averaging > 1:
                power_db_shifted = self._apply_optimized_averaging(power_db_shifted)
            
            # Validate the spectrum data to prevent GUI errors
            if not self._validate_spectrum_data(power_db_shifted):
                self.logger.warning("Invalid spectrum data detected, skipping frame")
                return None
            
            return power_db_shifted
            
        except Exception as e:
            self.logger.error(f"Error computing spectrum: {e}")
            return None
    
    def _validate_spectrum_data(self, spectrum: np.ndarray) -> bool:
        """Validate spectrum data to prevent GUI errors."""
        try:
            # Check for empty array
            if len(spectrum) == 0:
                return False
            
            # Check for NaN or infinite values
            if not np.isfinite(spectrum).all():
                # Replace invalid values with reasonable defaults
                spectrum[~np.isfinite(spectrum)] = -120.0  # Default noise floor
                
            # Check for reasonable range (spectrum should be in dB)
            min_val = np.min(spectrum)
            max_val = np.max(spectrum)
            
            # Reasonable spectrum range should be between -200 and +100 dB
            if min_val < -200 or max_val > 100:
                self.logger.debug(f"Spectrum values outside expected range: min={min_val:.1f}, max={max_val:.1f}")
                # Clamp values to reasonable range
                np.clip(spectrum, -200, 100, out=spectrum)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error validating spectrum data: {e}")
            return False
    
    def _apply_averaging(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply spectrum averaging."""
        self.spectrum_history.append(spectrum.copy())
        
        # Keep only required number of spectra
        if len(self.spectrum_history) > self.averaging:
            self.spectrum_history.pop(0)
        
        # Return averaged spectrum
        return np.mean(self.spectrum_history, axis=0)
    
    def _apply_optimized_averaging(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply optimized spectrum averaging with exponential moving average (PySDR pattern)."""
        # Use exponential moving average for better performance and memory efficiency
        if self._exponential_avg_buffer is None:
            # Initialize with first spectrum
            self._exponential_avg_buffer = spectrum.copy().astype(np.float32)
            return spectrum
        
        # Exponential moving average: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        # Where alpha = 2 / (N + 1) for N-point moving average equivalent
        alpha = self._ema_alpha
        self._exponential_avg_buffer = alpha * spectrum + (1 - alpha) * self._exponential_avg_buffer
        
        return self._exponential_avg_buffer.copy()
    
    def compute_detailed_spectrum(self, iq_samples: np.ndarray) -> Optional[SpectrumData]:
        """Compute detailed spectrum analysis."""
        try:
            if len(iq_samples) < self.fft_size:
                return None
            
            samples = iq_samples[:self.fft_size].astype(np.complex64)
            windowed_samples = samples * self.window
            
            # Compute FFT
            if self.fft_method == 'pyfftw':
                self.fft_input[:] = windowed_samples
                self.fft_object()
                fft_result = self.fft_output.copy()
            else:
                fft_result = np.fft.fft(windowed_samples)
            
            # Shift FFT result - use np.roll for power-of-2
            if self._use_roll_for_shift:
                fft_shifted = np.roll(fft_result, self.fft_size // 2)
            else:
                fft_shifted = np.fft.fftshift(fft_result)
            
            # Create frequency array
            freqs = np.fft.fftfreq(self.fft_size, 1/self.settings.sdr.sample_rate)
            freqs_shifted = np.fft.fftshift(freqs)
            
            # Compute magnitude and phase
            magnitude = np.abs(fft_shifted)
            phase = np.angle(fft_shifted)
            power_db = 10 * np.log10(magnitude**2 + 1e-12)
            
            return SpectrumData(
                frequencies=freqs_shifted + self.settings.sdr.center_frequency,
                power_db=power_db,
                magnitude=magnitude,
                phase=phase,
                sample_rate=self.settings.sdr.sample_rate,
                center_frequency=self.settings.sdr.center_frequency,
                fft_size=self.fft_size,
                window_type=self.window_type
            )
            
        except Exception as e:
            self.logger.error(f"Error computing detailed spectrum: {e}")
            return None
    
    def compute_spectrogram_efficient(self, iq_samples: np.ndarray, 
                                      overlap_factor: float = 0.5) -> Optional[np.ndarray]:
        """
        Compute spectrogram efficiently using PySDR row-by-row pattern.
        
        Args:
            iq_samples: Input IQ samples
            overlap_factor: Overlap factor (0-1), default 0.5 = 50% overlap
            
        Returns:
            Spectrogram as 2D numpy array (time x frequency)
        """
        try:
            overlap = int(self.fft_size * overlap_factor)
            hop_size = self.fft_size - overlap
            num_rows = (len(iq_samples) - self.fft_size) // hop_size + 1
            
            if num_rows <= 0:
                return None
            
            # Pre-allocate spectrogram array
            spectrogram = np.zeros((num_rows, self.fft_size), dtype=np.float32)
            
            # Compute row by row (PySDR pattern for efficiency)
            for i in range(num_rows):
                start_idx = i * hop_size
                end_idx = start_idx + self.fft_size
                
                if end_idx > len(iq_samples):
                    break
                
                # Extract segment and apply window
                segment = iq_samples[start_idx:end_idx].astype(np.complex64)
                windowed = segment * self.window
                
                # Compute FFT
                if self.fft_method == 'pyfftw_optimized':
                    self.fft_input[:] = windowed
                    self.fft_object()
                    fft_result = self.fft_output.copy()
                else:
                    fft_result = np.fft.fft(windowed)
                
                # Shift and convert to dB
                if self._use_roll_for_shift:
                    fft_shifted = np.roll(fft_result, self.fft_size // 2)
                else:
                    fft_shifted = np.fft.fftshift(fft_result)
                
                spectrogram[i, :] = 10 * np.log10(np.abs(fft_shifted)**2 + 1e-12)
            
            return spectrogram
            
        except Exception as e:
            self.logger.error(f"Error computing spectrogram: {e}")
            return None
    
    def compute_signal_stats(self, iq_samples: np.ndarray) -> Optional[SignalStats]:
        """Compute comprehensive signal statistics."""
        try:
            if len(iq_samples) == 0:
                return None
            
            # Basic power measurements
            mean_power = np.mean(np.abs(iq_samples)**2)
            peak_power = np.max(np.abs(iq_samples)**2)
            rms_level = np.sqrt(mean_power)
            
            # DC offset
            dc_offset = np.mean(iq_samples)
            
            # IQ imbalance estimation
            i_samples = np.real(iq_samples)
            q_samples = np.imag(iq_samples)
            i_power = np.var(i_samples)
            q_power = np.var(q_samples)
            iq_imbalance = np.abs(i_power - q_power) / (i_power + q_power) if (i_power + q_power) > 0 else 0
            
            # Simple SNR estimation (ratio of signal power to noise floor)
            spectrum = np.abs(np.fft.fft(iq_samples))**2
            sorted_spectrum = np.sort(spectrum)
            noise_floor = np.mean(sorted_spectrum[:len(spectrum)//4])  # Bottom 25%
            signal_power = np.mean(sorted_spectrum[-len(spectrum)//4:])  # Top 25%
            snr_estimate = 10 * np.log10(signal_power / noise_floor) if noise_floor > 0 else 0
            
            return SignalStats(
                mean_power=10 * np.log10(mean_power + 1e-12),
                peak_power=10 * np.log10(peak_power + 1e-12),
                snr_estimate=snr_estimate,
                dc_offset=dc_offset,
                iq_imbalance=iq_imbalance,
                rms_level=20 * np.log10(rms_level + 1e-12)
            )
            
        except Exception as e:
            self.logger.error(f"Error computing signal stats: {e}")
            return None
    
    def apply_filter(self, iq_samples: np.ndarray, filter_type: str = None) -> Optional[np.ndarray]:
        """Apply digital filter to IQ samples."""
        try:
            if filter_type is None:
                filter_type = self.settings.dsp.filter_type
            
            if SDR_AVAILABLE:
                return self._apply_sdr_filter(iq_samples, filter_type)
            else:
                return self._apply_scipy_filter(iq_samples, filter_type)
                
        except Exception as e:
            self.logger.error(f"Error applying filter: {e}")
            return iq_samples
    
    def _apply_sdr_filter(self, iq_samples: np.ndarray, filter_type: str) -> np.ndarray:
        """Apply filter using sdr library."""
        if not SDR_AVAILABLE:
            return iq_samples
        
        try:
            # Design filter based on type
            if filter_type == "lowpass":
                cutoff = self.settings.dsp.filter_cutoff_high
                h = sdr.lowpass_fir(cutoff, self.settings.dsp.filter_order)
            elif filter_type == "highpass":
                cutoff = self.settings.dsp.filter_cutoff_low
                h = sdr.highpass_fir(cutoff, self.settings.dsp.filter_order)
            elif filter_type == "bandpass":
                cutoffs = [self.settings.dsp.filter_cutoff_low, 
                          self.settings.dsp.filter_cutoff_high]
                h = sdr.bandpass_fir(cutoffs, self.settings.dsp.filter_order)
            elif filter_type == "bandstop":
                cutoffs = [self.settings.dsp.filter_cutoff_low, 
                          self.settings.dsp.filter_cutoff_high]
                h = sdr.bandstop_fir(cutoffs, self.settings.dsp.filter_order)
            else:
                return iq_samples
            
            # Create FIR filter object and apply
            fir_filter = sdr.FIR(h)
            filtered_samples = fir_filter(iq_samples)
            return filtered_samples
            
        except Exception as e:
            self.logger.error(f"Error with sdr filter: {e}")
            return iq_samples
    
    def _apply_scipy_filter(self, iq_samples: np.ndarray, filter_type: str) -> np.ndarray:
        """Apply filter using scipy."""
        try:
            nyquist = 0.5
            low = self.settings.dsp.filter_cutoff_low * nyquist
            high = self.settings.dsp.filter_cutoff_high * nyquist
            order = self.settings.dsp.filter_order
            
            if filter_type == "lowpass":
                b, a = scipy.signal.butter(order, high, btype='low')
            elif filter_type == "highpass":
                b, a = scipy.signal.butter(order, low, btype='high')
            elif filter_type == "bandpass":
                b, a = scipy.signal.butter(order, [low, high], btype='band')
            elif filter_type == "bandstop":
                b, a = scipy.signal.butter(order, [low, high], btype='bandstop')
            else:
                return iq_samples
            
            # Apply filter
            filtered_samples = scipy.signal.filtfilt(b, a, iq_samples)
            return filtered_samples.astype(np.complex64)
            
        except Exception as e:
            self.logger.error(f"Error with scipy filter: {e}")
            return iq_samples
    
    def resample_signal(self, iq_samples: np.ndarray, 
                       resample_factor: float = None) -> Optional[np.ndarray]:
        """Resample signal using advanced techniques."""
        try:
            if resample_factor is None:
                resample_factor = self.settings.dsp.resample_factor
            
            if abs(resample_factor - 1.0) < 1e-6:
                return iq_samples  # No resampling needed
            
            if SDR_AVAILABLE:
                return self._resample_with_sdr(iq_samples, resample_factor)
            else:
                return self._resample_with_scipy(iq_samples, resample_factor)
                
        except Exception as e:
            self.logger.error(f"Error resampling signal: {e}")
            return iq_samples
    
    def _resample_with_sdr(self, iq_samples: np.ndarray, factor: float) -> np.ndarray:
        """Resample using sdr library."""
        if not SDR_AVAILABLE:
            return iq_samples
        
        try:
            # Use sdr's resampling capabilities
            if factor > 1.0:
                # Interpolation
                int_factor = int(np.round(factor))
                interpolator = sdr.Interpolator(int_factor)
                return interpolator(iq_samples)
            elif factor < 1.0:
                # Decimation
                dec_factor = int(np.round(1.0 / factor))
                decimator = sdr.Decimator(dec_factor)
                return decimator(iq_samples)
            else:
                return iq_samples
                
        except Exception as e:
            self.logger.error(f"Error with sdr resampling: {e}")
            return iq_samples
    
    def _resample_with_scipy(self, iq_samples: np.ndarray, factor: float) -> np.ndarray:
        """Resample using scipy."""
        try:
            new_length = int(len(iq_samples) * factor)
            resampled = scipy.signal.resample(iq_samples, new_length)
            return resampled.astype(np.complex64)
        except Exception as e:
            self.logger.error(f"Error with scipy resampling: {e}")
            return iq_samples
    
    def demodulate_signal(self, iq_samples: np.ndarray, 
                         demod_type: str = None) -> Optional[np.ndarray]:
        """Demodulate signal based on specified type."""
        try:
            if demod_type is None:
                demod_type = self.settings.dsp.demod_type
            
            if demod_type == "none":
                return None
            
            if demod_type == "am":
                return self._demodulate_am(iq_samples)
            elif demod_type == "fm":
                return self._demodulate_fm(iq_samples)
            elif demod_type == "pm":
                return self._demodulate_pm(iq_samples)
            elif demod_type in ["qpsk", "bpsk"]:
                return self._demodulate_psk(iq_samples, demod_type)
            else:
                self.logger.warning(f"Unknown demodulation type: {demod_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error demodulating signal: {e}")
            return None
    
    def _demodulate_am(self, iq_samples: np.ndarray) -> np.ndarray:
        """AM demodulation."""
        return np.abs(iq_samples)
    
    def _demodulate_fm(self, iq_samples: np.ndarray) -> np.ndarray:
        """FM demodulation."""
        # Simple FM demodulation using phase difference
        phase = np.angle(iq_samples)
        phase_diff = np.diff(phase)
        
        # Unwrap phase
        phase_diff = np.unwrap(phase_diff)
        
        # Pad to maintain length
        fm_demod = np.concatenate([[0], phase_diff])
        return fm_demod
    
    def _demodulate_pm(self, iq_samples: np.ndarray) -> np.ndarray:
        """PM demodulation."""
        return np.angle(iq_samples)
    
    def _demodulate_psk(self, iq_samples: np.ndarray, psk_type: str) -> np.ndarray:
        """PSK demodulation."""
        # Simplified PSK demodulation
        if psk_type == "bpsk":
            # BPSK: decode based on real part sign
            return np.sign(np.real(iq_samples))
        elif psk_type == "qpsk":
            # QPSK: decode both I and Q
            i_bits = np.sign(np.real(iq_samples))
            q_bits = np.sign(np.imag(iq_samples))
            return i_bits + 1j * q_bits
        else:
            return iq_samples
    
    def detect_peaks(self, spectrum: np.ndarray) -> List[Tuple[int, float]]:
        """Detect peaks in spectrum."""
        try:
            threshold = self.settings.processing.peak_threshold
            min_distance = self.settings.processing.peak_min_distance
            
            # Convert threshold from dB to linear if needed
            threshold_linear = 10**(threshold/10) if threshold < 0 else threshold
            
            # Find peaks using scipy
            peaks, properties = scipy.signal.find_peaks(
                spectrum, 
                height=threshold_linear,
                distance=min_distance
            )
            
            # Return peak indices and heights
            peak_list = [(int(peak), float(spectrum[peak])) for peak in peaks]
            return sorted(peak_list, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error detecting peaks: {e}")
            return []
    
    # Configuration methods
    def set_fft_size(self, fft_size: int):
        """Update FFT size."""
        self.fft_size = fft_size
        self.window = self._create_window(self.window_type, self.fft_size)
        self._setup_fft()
        self.spectrum_history.clear()
    
    def set_window_function(self, window_type: str):
        """Update window function."""
        self.window_type = window_type
        self.window = self._create_window(window_type, self.fft_size)
    
    def set_averaging(self, averaging: int):
        """Update averaging count."""
        self.averaging = averaging
    
    # Modulation Analysis and Demodulation Methods
    def analyze_modulation(self, iq_samples: np.ndarray) -> Dict[str, Any]:
        """
        Analyze modulation type and parameters from IQ samples.
        
        Args:
            iq_samples: Complex IQ signal data
            
        Returns:
            Dictionary containing modulation analysis results
        """
        try:
            if len(iq_samples) < 1024:
                return {"type": "Unknown", "confidence": 0.0, "parameters": {}}
            
            # Perform modulation analysis
            modulation_result = self.modulation_analyzer.detect_modulation(iq_samples)
            
            # Cache the result
            self.last_modulation_analysis = modulation_result
            
            self.logger.debug(f"Detected modulation: {modulation_result['type']} "
                            f"(confidence: {modulation_result['confidence']:.2f})")
            
            return modulation_result
            
        except Exception as e:
            self.logger.error(f"Modulation analysis error: {e}")
            return {"type": "Unknown", "confidence": 0.0, "parameters": {}, "error": str(e)}
    
    def demodulate_signal(self, iq_samples: np.ndarray, 
                         modulation_type: str = None, 
                         parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Demodulate signal based on modulation type.
        
        Args:
            iq_samples: Complex IQ signal data
            modulation_type: Modulation type (auto-detect if None)
            parameters: Modulation parameters
            
        Returns:
            Dictionary containing demodulated data and metadata
        """
        try:
            # Auto-detect modulation if not specified
            if modulation_type is None:
                if self.last_modulation_analysis is None:
                    mod_analysis = self.analyze_modulation(iq_samples)
                else:
                    mod_analysis = self.last_modulation_analysis
                
                modulation_type = mod_analysis["type"]
                if parameters is None:
                    parameters = mod_analysis.get("parameters", {})
            
            # Demodulate using the appropriate engine
            demod_result = self.demodulation_engine.demodulate(
                iq_samples, modulation_type, parameters
            )
            
            self.logger.debug(f"Demodulated {modulation_type} signal, "
                            f"got {len(demod_result.get('demodulated_data', []))} samples")
            
            return demod_result
            
        except Exception as e:
            self.logger.error(f"Demodulation error: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_encoding(self, bit_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze channel coding from bit sequence.
        
        Args:
            bit_data: Binary data sequence
            
        Returns:
            Dictionary containing encoding analysis results
        """
        try:
            if len(bit_data) < 64:
                return {"type": "None", "confidence": 0.0, "parameters": {}}
            
            # Perform encoding analysis
            encoding_result = self.encoding_analyzer.detect_encoding(bit_data)
            
            # Cache the result
            self.last_encoding_analysis = encoding_result
            
            self.logger.debug(f"Detected encoding: {encoding_result['type']} "
                            f"(confidence: {encoding_result['confidence']:.2f})")
            
            return encoding_result
            
        except Exception as e:
            self.logger.error(f"Encoding analysis error: {e}")
            return {"type": "None", "confidence": 0.0, "parameters": {}, "error": str(e)}
    
    def decode_data(self, encoded_data: np.ndarray, 
                   coding_type: str = None, 
                   parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Decode channel-coded data.
        
        Args:
            encoded_data: Encoded bit sequence
            coding_type: Coding type (auto-detect if None)
            parameters: Coding parameters
            
        Returns:
            Dictionary containing decoded data and metadata
        """
        try:
            # Auto-detect encoding if not specified
            if coding_type is None:
                if self.last_encoding_analysis is None:
                    enc_analysis = self.analyze_encoding(encoded_data)
                else:
                    enc_analysis = self.last_encoding_analysis
                
                coding_type = enc_analysis["type"]
                if parameters is None:
                    parameters = enc_analysis.get("parameters", {})
            
            # Decode using the appropriate engine
            decode_result = self.decoding_engine.decode(
                encoded_data, coding_type, parameters
            )
            
            self.logger.debug(f"Decoded {coding_type} data, "
                            f"corrected {decode_result.get('corrected_errors', 0)} errors")
            
            return decode_result
            
        except Exception as e:
            self.logger.error(f"Decoding error: {e}")
            return {"success": False, "error": str(e)}
    
    # Configuration methods
    def set_fft_size(self, fft_size: int):
        """Update FFT size."""
        self.fft_size = fft_size
        self.window = self._create_window(self.window_type, self.fft_size)
        self._setup_fft()
        self.spectrum_history.clear()
    
    def set_window_function(self, window_type: str):
        """Update window function."""
        self.window_type = window_type
        self.window = self._create_window(window_type, self.fft_size)
    
    def set_averaging(self, averaging: int):
        """Update averaging count."""
        self.averaging = averaging
    
    # Modulation Analysis and Demodulation Methods
    def analyze_modulation(self, iq_samples: np.ndarray) -> Dict[str, Any]:
        """
        Analyze modulation type and parameters from IQ samples.
        
        Args:
            iq_samples: Complex IQ signal data
            
        Returns:
            Dictionary containing modulation analysis results
        """
        try:
            if len(iq_samples) < 1024:
                return {"type": "Unknown", "confidence": 0.0, "parameters": {}}
            
            # Perform modulation analysis
            modulation_result = self.modulation_analyzer.detect_modulation(iq_samples)
            
            # Cache the result
            self.last_modulation_analysis = modulation_result
            
            self.logger.debug(f"Detected modulation: {modulation_result['type']} "
                            f"(confidence: {modulation_result['confidence']:.2f})")
            
            return modulation_result
            
        except Exception as e:
            self.logger.error(f"Modulation analysis error: {e}")
            return {"type": "Unknown", "confidence": 0.0, "parameters": {}, "error": str(e)}
    
    def demodulate_signal(self, iq_samples: np.ndarray, 
                         modulation_type: str = None, 
                         parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Demodulate signal based on modulation type.
        
        Args:
            iq_samples: Complex IQ signal data
            modulation_type: Modulation type (auto-detect if None)
            parameters: Modulation parameters
            
        Returns:
            Dictionary containing demodulated data and metadata
        """
        try:
            # Auto-detect modulation if not specified
            if modulation_type is None:
                if self.last_modulation_analysis is None:
                    mod_analysis = self.analyze_modulation(iq_samples)
                else:
                    mod_analysis = self.last_modulation_analysis
                
                modulation_type = mod_analysis["type"]
                if parameters is None:
                    parameters = mod_analysis.get("parameters", {})
            
            # Fix symbol rate estimation if needed
            if parameters and "symbol_rate" in parameters:
                # Override low symbol rate estimates that are clearly wrong
                if parameters["symbol_rate"] < 10000:  # Less than 10 kHz is probably wrong
                    # Estimate based on sample rate and expected oversampling
                    expected_symbol_rate = self.settings.sdr.sample_rate / 20  # 20x oversampling
                    parameters["symbol_rate"] = expected_symbol_rate
                    self.logger.debug(f"Overriding symbol rate to {expected_symbol_rate:.0f} Hz")
            
            # Demodulate using the appropriate engine
            demod_result = self.demodulation_engine.demodulate(
                iq_samples, modulation_type, parameters
            )
            
            self.logger.debug(f"Demodulated {modulation_type} signal, "
                            f"got {len(demod_result.get('demodulated_data', []))} samples")
            
            return demod_result
            
        except Exception as e:
            self.logger.error(f"Demodulation error: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_encoding(self, bit_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze channel coding from bit sequence.
        
        Args:
            bit_data: Binary data sequence
            
        Returns:
            Dictionary containing encoding analysis results
        """
        try:
            if len(bit_data) < 64:
                return {"type": "None", "confidence": 0.0, "parameters": {}}
            
            # Perform encoding analysis
            encoding_result = self.encoding_analyzer.detect_encoding(bit_data)
            
            # Cache the result
            self.last_encoding_analysis = encoding_result
            
            self.logger.debug(f"Detected encoding: {encoding_result['type']} "
                            f"(confidence: {encoding_result['confidence']:.2f})")
            
            return encoding_result
            
        except Exception as e:
            self.logger.error(f"Encoding analysis error: {e}")
            return {"type": "None", "confidence": 0.0, "parameters": {}, "error": str(e)}
    
    def decode_data(self, encoded_data: np.ndarray, 
                   coding_type: str = None, 
                   parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Decode channel-coded data.
        
        Args:
            encoded_data: Encoded bit sequence
            coding_type: Coding type (auto-detect if None)
            parameters: Coding parameters
            
        Returns:
            Dictionary containing decoded data and metadata
        """
        try:
            # Auto-detect encoding if not specified
            if coding_type is None:
                if self.last_encoding_analysis is None:
                    enc_analysis = self.analyze_encoding(encoded_data)
                else:
                    enc_analysis = self.last_encoding_analysis
                
                coding_type = enc_analysis["type"]
                if parameters is None:
                    parameters = enc_analysis.get("parameters", {})
            
            # Decode using the appropriate engine
            decode_result = self.decoding_engine.decode(
                encoded_data, coding_type, parameters
            )
            
            self.logger.debug(f"Decoded {coding_type} data, "
                            f"corrected {decode_result.get('corrected_errors', 0)} errors")
            
            return decode_result
            
        except Exception as e:
            self.logger.error(f"Decoding error: {e}")
            return {"success": False, "error": str(e)}
    
    def process_complete_chain(self, iq_samples: np.ndarray) -> Dict[str, Any]:
        """
        Process complete signal chain: modulation analysis -> demodulation -> decoding.
        
        Args:
            iq_samples: Complex IQ samples
            
        Returns:
            Dictionary with complete processing results
        """
        try:
            result = {
                "success": False,
                "modulation_analysis": {},
                "demodulation": {},
                "encoding_analysis": {},
                "decoding": {},
                "final_data": np.array([])
            }
            
            if len(iq_samples) == 0:
                return result
            
            # Step 1: Modulation Analysis
            self.logger.debug("Step 1: Analyzing modulation...")
            mod_analysis = self.analyze_modulation(iq_samples)
            result["modulation_analysis"] = mod_analysis
            
            # Step 2: Demodulation
            mod_type = mod_analysis.get("type", "Unknown")
            self.logger.debug(f"Step 2: Demodulating {mod_type}...")
            
            if mod_type != "Unknown":
                demod_result = self.demodulate_signal(
                    iq_samples,
                    mod_type,
                    mod_analysis.get("parameters", {})
                )
                result["demodulation"] = demod_result
                
                # Extract demodulated data
                demod_data = demod_result.get("demodulated_data", np.array([]))
                
                # Ensure demod_data is a numpy array (handle tuple returns from sdr library)
                if isinstance(demod_data, tuple):
                    # If sdr library returns tuple, take the first element (usually the data)
                    self.logger.debug("Converting tuple demodulated data to numpy array")
                    demod_data = np.array(demod_data[0]) if len(demod_data) > 0 else np.array([])
                elif not isinstance(demod_data, np.ndarray):
                    # Convert any other type to numpy array
                    self.logger.debug(f"Converting {type(demod_data)} demodulated data to numpy array")
                    demod_data = np.array(demod_data)
                
                # Step 3: Encoding Analysis (for digital data)
                if demod_result.get("data_type") == "digital" and len(demod_data) > 0:
                    self.logger.debug("Step 3: Analyzing encoding...")
                    
                    # Convert to binary if needed
                    if demod_data.dtype == bool:
                        binary_data = demod_data.astype(np.uint8)
                    elif demod_data.dtype in [np.int8, np.int16, np.int32, np.int64]:
                        binary_data = np.clip(demod_data, 0, 1).astype(np.uint8)
                    else:
                        # Float data - threshold to binary
                        threshold = np.mean(demod_data) if len(demod_data) > 0 else 0.5
                        binary_data = (demod_data > threshold).astype(np.uint8)
                    
                    enc_analysis = self.analyze_encoding(binary_data)
                    result["encoding_analysis"] = enc_analysis
                    
                    # Step 4: Decoding
                    if enc_analysis["type"] != "None" and enc_analysis["confidence"] > 0.3:
                        self.logger.debug(f"Step 4: Decoding {enc_analysis['type']}...")
                        decode_result = self.decode_data(
                            binary_data,
                            enc_analysis["type"],
                            enc_analysis["parameters"]
                        )
                        result["decoding"] = decode_result
                        result["final_data"] = decode_result.get("decoded_data", binary_data)
                    else:
                        result["final_data"] = binary_data
                else:
                    # For analog data, final data is the demodulated output
                    result["final_data"] = demod_data
            else:
                # Unknown modulation - pass through raw IQ
                result["final_data"] = iq_samples
            
            result["success"] = True
            self.logger.debug("Complete processing chain completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Complete processing chain error: {e}")
            return {
                "success": False,
                "error": str(e),
                "modulation_analysis": {},
                "demodulation": {},
                "encoding_analysis": {},
                "decoding": {},
                "final_data": np.array([])
            }

    def update_sample_rate(self, sample_rate: float):
        """Update sample rate for modulation analysis engines."""
        try:
            self.modulation_analyzer = create_modulation_analyzer(sample_rate)
            self.demodulation_engine = create_demodulation_engine(sample_rate)
            self.signal_detector = create_signal_detector(sample_rate)
            self.tdma_detector = TDMABurstDetector(sample_rate)
            self.logger.info(f"Updated sample rate to {sample_rate} Hz")
        except Exception as e:
            self.logger.error(f"Error updating sample rate: {e}")
        self.spectrum_history.clear()
    
    def detect_signal(self, iq_samples: np.ndarray, method: str = "energy", 
                     p_fa: float = 1e-6) -> Dict[str, Any]:
        """
        Perform signal detection using sdr._detection algorithms.
        
        Args:
            iq_samples: Complex IQ samples
            method: Detection method ("energy", "correlation", "adaptive")
            p_fa: Probability of false alarm
            
        Returns:
            Detection result dictionary
        """
        try:
            from rf_spectrum_analyzer.dsp.signal_detection import DetectionMethod
            
            if len(iq_samples) == 0:
                return {"success": False, "error": "Empty signal"}
            
            # Choose detection method
            if method == "energy":
                result = self.signal_detector.energy_detection(iq_samples, p_fa)
            elif method == "correlation":
                # Need template for correlation detection
                templates = list(self.signal_detector.signal_templates.keys())
                if not templates:
                    self.logger.warning("No templates available for correlation detection, using energy")
                    result = self.signal_detector.energy_detection(iq_samples, p_fa)
                else:
                    result = self.signal_detector.correlation_detection(iq_samples, templates[0], p_fa)
            elif method == "adaptive":
                result = self.signal_detector.adaptive_detection(iq_samples)
            else:
                self.logger.warning(f"Unknown detection method {method}, using energy")
                result = self.signal_detector.energy_detection(iq_samples, p_fa)
            
            # Cache result
            self.last_detection_result = result
            
            # Convert to dictionary for easier handling
            detection_dict = {
                "success": True,
                "signal_detected": result.signal_detected,
                "detection_method": result.detection_method,
                "confidence": result.confidence,
                "snr_estimate": result.snr_estimate,
                "test_statistic": result.test_statistic,
                "threshold": result.threshold,
                "p_fa": result.p_fa,
                "p_d": result.p_d,
                "noise_variance": result.noise_variance
            }
            
            self.logger.debug(f"Signal detection: {result.signal_detected} "
                            f"(confidence: {result.confidence:.3f}, SNR: {result.snr_estimate:.1f} dB)")
            
            return detection_dict
            
        except Exception as e:
            self.logger.error(f"Signal detection error: {e}")
            return {"success": False, "error": str(e)}
    
    def detect_tdma_bursts(self, iq_samples: np.ndarray, 
                          sync_pattern: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect TDMA bursts in signal.
        
        Args:
            iq_samples: Complex IQ samples
            sync_pattern: Optional sync pattern for correlation detection
            
        Returns:
            TDMA detection result dictionary
        """
        try:
            if len(iq_samples) == 0:
                return {"success": False, "error": "Empty signal"}
            
            # Set sync pattern if provided
            if sync_pattern is not None:
                self.tdma_detector.set_sync_pattern(sync_pattern)
            
            # Detect bursts
            bursts = self.tdma_detector.detect_bursts(iq_samples)
            
            # Analyze timing if bursts found
            timing_analysis = None
            if bursts:
                timing_analysis = self.tdma_detector.analyze_timing(bursts)
            
            # Cache result
            self.last_tdma_analysis = {
                "bursts": bursts,
                "timing_analysis": timing_analysis
            }
            
            result = {
                "success": True,
                "burst_count": len(bursts),
                "bursts_detected": len(bursts) > 0,
                "bursts": bursts,
                "timing_analysis": timing_analysis
            }
            
            self.logger.debug(f"TDMA detection: {len(bursts)} bursts found")
            
            return result
            
        except Exception as e:
            self.logger.error(f"TDMA burst detection error: {e}")
            return {"success": False, "error": str(e)}
    
    def spectrum_sensing(self, iq_samples: np.ndarray, 
                        frequency_bands: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Perform spectrum sensing across multiple frequency bands.
        
        Args:
            iq_samples: Complex IQ samples
            frequency_bands: Dictionary of {band_name: (start_freq, stop_freq)}
            
        Returns:
            Spectrum sensing results
        """
        try:
            if len(iq_samples) == 0:
                return {"success": False, "error": "Empty signal"}
            
            # Perform spectrum sensing
            sensing_results = self.signal_detector.spectrum_sensing(iq_samples, frequency_bands)
            
            # Convert results to serializable format
            results_dict = {}
            for band_name, result in sensing_results.items():
                results_dict[band_name] = {
                    "frequency_band": result.frequency_band,
                    "center_frequency": result.center_frequency,
                    "bandwidth": result.bandwidth,
                    "signal_detected": result.signal_detected,
                    "occupancy_probability": result.occupancy_probability,
                    "signal_power": result.signal_power,
                    "noise_power": result.noise_power,
                    "snr_db": result.snr_db,
                    "detection_confidence": result.detection_confidence
                }
            
            return {
                "success": True,
                "band_results": results_dict,
                "total_bands": len(frequency_bands),
                "occupied_bands": sum(1 for r in sensing_results.values() if r.signal_detected)
            }
            
        except Exception as e:
            self.logger.error(f"Spectrum sensing error: {e}")
            return {"success": False, "error": str(e)}
    
    def calibrate_detector(self, noise_samples: np.ndarray, method: str = "robust") -> Dict[str, Any]:
        """
        Calibrate signal detector with noise samples.
        
        Args:
            noise_samples: Pure noise samples for calibration
            method: Calibration method ("robust", "simple", "adaptive")
            
        Returns:
            Calibration result
        """
        try:
            if len(noise_samples) == 0:
                return {"success": False, "error": "Empty noise samples"}
            
            noise_variance = self.signal_detector.calibrate_noise_floor(noise_samples, method)
            
            return {
                "success": True,
                "noise_variance": noise_variance,
                "method": method,
                "calibrated": self.signal_detector.calibrated
            }
            
        except Exception as e:
            self.logger.error(f"Detector calibration error: {e}")
            return {"success": False, "error": str(e)}
    
    def add_signal_template(self, name: str, template: np.ndarray):
        """
        Add signal template for correlation detection.
        
        Args:
            name: Template name
            template: Signal template (complex samples)
        """
        try:
            self.signal_detector.add_signal_template(name, template)
            self.logger.info(f"Added signal template '{name}'")
        except Exception as e:
            self.logger.error(f"Error adding signal template: {e}")
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get signal detection performance statistics."""
        try:
            return self.signal_detector.get_detection_statistics()
        except Exception as e:
            self.logger.error(f"Error getting detection statistics: {e}")
            return {"error": str(e)}
    
    # Detection control methods for GUI integration
    def detect_signals_manual(self) -> Optional[Dict[str, Any]]:
        """
        Manually trigger signal detection on current data buffer.
        
        Returns:
            Detection results or None if no data available
        """
        try:
            # Check if we have recent signal data
            if not hasattr(self, '_current_iq_data') or self._current_iq_data is None:
                self.logger.warning("No signal data available for manual detection")
                return None
            
            # Perform energy detection
            result = self.detect_signal(self._current_iq_data, method="energy")
            
            if result.get("success", False):
                return {
                    "detected": result.get("signal_detected", False),
                    "snr_db": result.get("snr_estimate", 0.0),
                    "confidence": result.get("confidence", 0.0),
                    "method": "manual_energy",
                    "timestamp": "manual_trigger"
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Manual detection error: {e}")
            return None
    
    def detect_tdma_bursts(self, iq_samples: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Detect TDMA bursts in signal data.
        
        Args:
            iq_samples: Optional IQ samples (uses current data if None)
            
        Returns:
            TDMA detection results or None if no data available
        """
        try:
            # Use provided samples or current data buffer
            if iq_samples is None:
                if not hasattr(self, '_current_iq_data') or self._current_iq_data is None:
                    self.logger.warning("No signal data available for TDMA detection")
                    return None
                iq_samples = self._current_iq_data
            
            # Perform TDMA burst detection using the tdma_detector
            result = self.tdma_detector.detect_bursts(iq_samples)
            
            if result:
                # Analyze timing if bursts found
                timing_analysis = None
                if result:
                    timing_analysis = self.tdma_detector.analyze_timing(result)
                
                return {
                    "detected": len(result) > 0,
                    "burst_count": len(result),
                    "bursts": result,
                    "timing_analysis": timing_analysis,
                    "method": "tdma_burst",
                    "timestamp": "manual_trigger"
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"TDMA detection error: {e}")
            return None
    
    def set_auto_detection(self, enabled: bool):
        """
        Enable or disable automatic signal detection.
        
        Args:
            enabled: True to enable auto detection, False to disable
        """
        try:
            self._auto_detection_enabled = enabled
            self.logger.info(f"Auto detection {'enabled' if enabled else 'disabled'}")
            
            # Update settings if available
            if hasattr(self.settings, 'detection'):
                self.settings.detection.auto_detection_enabled = enabled
                
        except Exception as e:
            self.logger.error(f"Error setting auto detection: {e}")
    
    def set_advanced_analysis(self, enabled: bool):
        """
        Enable or disable advanced analysis mode.
        
        Args:
            enabled: True to enable advanced analysis, False to disable
        """
        try:
            self._advanced_analysis_enabled = enabled
            self.logger.info(f"Advanced analysis {'enabled' if enabled else 'disabled'}")
            
            # Update settings if available
            if hasattr(self.settings, 'detection'):
                self.settings.detection.advanced_analysis_enabled = enabled
                
        except Exception as e:
            self.logger.error(f"Error setting advanced analysis: {e}")
    
    def set_detection_threshold(self, threshold_dbm: float):
        """
        Set signal detection threshold.
        
        Args:
            threshold_dbm: Detection threshold in dBm
        """
        try:
            self._detection_threshold_dbm = threshold_dbm
            self.logger.info(f"Detection threshold set to {threshold_dbm} dBm")
            
            # Update settings if available
            if hasattr(self.settings, 'detection'):
                self.settings.detection.energy_threshold_dbm = threshold_dbm
                
        except Exception as e:
            self.logger.error(f"Error setting detection threshold: {e}")
    
    def set_detection_interval(self, interval_ms: int):
        """
        Set periodic detection interval.
        
        Args:
            interval_ms: Detection interval in milliseconds
        """
        try:
            self._detection_interval_ms = interval_ms
            self.logger.info(f"Detection interval set to {interval_ms} ms")
            
            # Update settings if available
            if hasattr(self.settings, 'detection'):
                self.settings.detection.detection_interval_ms = interval_ms
                
        except Exception as e:
            self.logger.error(f"Error setting detection interval: {e}")
    
    def update_current_data(self, iq_samples: np.ndarray):
        """
        Update current IQ data buffer for detection operations.
        
        Args:
            iq_samples: New IQ samples to store
        """
        try:
            self._current_iq_data = iq_samples.copy() if iq_samples is not None else None
            
            # Perform auto detection if enabled
            if hasattr(self, '_auto_detection_enabled') and self._auto_detection_enabled:
                self._perform_auto_detection()
                
        except Exception as e:
            self.logger.error(f"Error updating current data: {e}")
    
    def _perform_auto_detection(self):
        """Perform automatic detection on current data."""
        try:
            if self._current_iq_data is None:
                return
            
            # Check signal power against threshold
            signal_power = np.mean(np.abs(self._current_iq_data)**2)
            power_dbm = 10 * np.log10(signal_power + 1e-12)
            
            threshold = getattr(self, '_detection_threshold_dbm', -80.0)
            
            if power_dbm > threshold:
                # Trigger detection
                result = self.detect_signal(self._current_iq_data, method="energy")
                if result.get("signal_detected", False):
                    self.logger.debug(f"Auto detection triggered: {power_dbm:.1f} dBm > {threshold} dBm")
                    
        except Exception as e:
            self.logger.debug(f"Auto detection error: {e}")
    
    def enhanced_analysis(self, iq_samples: np.ndarray) -> Dict[str, Any]:
        """
        Perform enhanced signal analysis using sdrconnect integration.
        
        Args:
            iq_samples: Complex IQ samples
            
        Returns:
            Enhanced analysis result dictionary
        """
        try:
            if len(iq_samples) == 0:
                return {"success": False, "error": "Empty signal"}
            
            # Perform enhanced analysis
            result = self.enhanced_analyzer.analyze_iq_data(iq_samples)
            
            # Cache result
            self.last_enhanced_analysis = result
            
            # Convert to dictionary for easier handling
            analysis_dict = {
                "success": True,
                "analysis_method": result.analysis_method,
                "sdrconnect_available": result.sdrconnect_available,
                
                # Basic spectrum data
                "power_spectrum": result.power_spectrum.tolist() if len(result.power_spectrum) > 0 else [],
                "frequency_axis": result.frequency_axis.tolist() if len(result.frequency_axis) > 0 else [],
                "peak_frequency": result.peak_frequency,
                "bandwidth": result.bandwidth,
                "snr_estimate": result.snr_estimate,
                
                # Enhanced data (if available)
                "has_enhanced_data": result.sdrconnect_available and result.analysis_method == "enhanced"
            }
            
            # Add enhanced metrics if available
            if result.sdrconnect_available and result.analysis_method == "enhanced":
                enhanced_metrics = {
                    "rms_power": result.rms_power,
                    "peak_power": result.peak_power,
                    "crest_factor": result.crest_factor,
                    "dc_offset_i": result.dc_offset_i,
                    "dc_offset_q": result.dc_offset_q,
                    "zero_crossings": result.zero_crossings,
                    "noise_floor": result.noise_floor,
                    "sinad": result.sinad,
                    "occupied_bandwidth": result.occupied_bandwidth,
                    "frequency_drift": result.frequency_drift,
                    "spur_frequencies": result.spur_frequencies,
                }
                
                # Add arrays if available
                if result.spectrogram is not None:
                    enhanced_metrics["spectrogram"] = result.spectrogram.tolist()
                if result.mean_psd is not None:
                    enhanced_metrics["mean_psd"] = result.mean_psd.tolist()
                if result.time_axis is not None:
                    enhanced_metrics["time_axis"] = result.time_axis.tolist()
                
                analysis_dict["enhanced_metrics"] = enhanced_metrics
            
            self.logger.debug(f"Enhanced analysis completed using {result.analysis_method} method")
            
            return analysis_dict
            
        except Exception as e:
            self.logger.error(f"Enhanced analysis error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_analysis_capabilities(self) -> Dict[str, Any]:
        """Get information about analysis capabilities."""
        return {
            "sdr_available": SDR_AVAILABLE,
            "scikit_dsp_available": SCIKIT_DSP_AVAILABLE,
            "pyfftw_available": PYFFTW_AVAILABLE,
            "enhanced_analysis": self.enhanced_analyzer.get_analysis_info(),
            "fft_method": getattr(self, 'fft_method', 'numpy'),
            "window_type": self.window_type,
            "fft_size": self.fft_size,
            "averaging": self.averaging
        }
    
    # Sequential Workflow Methods
    def set_analysis_frequency_range(self, f1: float, f2: float):
        """Set frequency range for focused analysis."""
        try:
            self.analysis_f1 = f1
            self.analysis_f2 = f2
            self.analysis_bandwidth = f2 - f1
            
            # Calculate frequency shift to center the range
            center_freq = (f1 + f2) / 2
            current_center = self.settings.sdr.center_frequency
            self.frequency_offset = center_freq - current_center
            
            self.logger.info(f"Analysis frequency range set: {f1/1e6:.3f} - {f2/1e6:.3f} MHz")
            self.logger.info(f"Analysis bandwidth: {self.analysis_bandwidth/1e6:.3f} MHz")
            
        except Exception as e:
            self.logger.error(f"Error setting analysis frequency range: {e}")
    
    def detect_and_demodulate(self) -> Optional[Dict[str, Any]]:
        """Detect modulation type and demodulate signal in frequency range."""
        try:
            if not hasattr(self, 'analysis_f1') or not hasattr(self, 'analysis_f2'):
                self.logger.warning("No frequency range set for analysis")
                return {"success": False, "error": "No frequency range set"}
            
            # Get current IQ data buffer
            if not hasattr(self, '_current_iq_data') or self._current_iq_data is None or len(self._current_iq_data) == 0:
                self.logger.warning("No IQ data available for analysis")
                return {"success": False, "error": "No IQ data available"}
            
            # Filter signal to analysis bandwidth
            filtered_iq = self._filter_to_frequency_range(self._current_iq_data)
            
            if len(filtered_iq) == 0:
                return {"success": False, "error": "No signal in frequency range"}
            
            # Detect modulation type
            modulation_result = self._detect_modulation_type(filtered_iq)
            
            if not modulation_result or not modulation_result.get('success', False):
                return {"success": False, "error": "Modulation detection failed"}
            
            modulation_type = modulation_result.get('modulation_type', 'unknown')
            
            # Demodulate signal
            demod_result = self.demodulate_signal(filtered_iq, modulation_type)
            
            if demod_result and demod_result.get('success', False):
                # Store demodulated data for decoding step
                self.demodulated_data = demod_result.get('demodulated_data', np.array([]))
                self.demodulated_metadata = {
                    'modulation_type': modulation_type,
                    'frequency_range': (self.analysis_f1, self.analysis_f2),
                    'demodulation_params': demod_result.get('parameters', {})
                }
                
                return {
                    "success": True,
                    "modulation_type": modulation_type,
                    "demodulated_data": self.demodulated_data,
                    "metadata": self.demodulated_metadata
                }
            else:
                return {"success": False, "error": "Demodulation failed"}
                
        except Exception as e:
            self.logger.error(f"Error in detect_and_demodulate: {e}")
            return {"success": False, "error": str(e)}
    
    def detect_and_decode(self) -> Optional[Dict[str, Any]]:
        """Detect coding type and decode demodulated signal."""
        try:
            if not self.has_demodulated_data():
                return {"success": False, "error": "No demodulated data available"}
            
            # Detect channel coding type
            coding_result = self._detect_coding_type(self.demodulated_data)
            
            if not coding_result or not coding_result.get('success', False):
                return {"success": False, "error": "Coding detection failed"}
            
            coding_type = coding_result.get('coding_type', 'unknown')
            
            # Decode data
            decode_result = self.decode_data(self.demodulated_data, coding_type)
            
            if decode_result and decode_result.get('success', False):
                decoded_data = decode_result.get('decoded_data', np.array([]))
                
                return {
                    "success": True,
                    "coding_type": coding_type,
                    "decoded_data": decoded_data,
                    "metadata": {
                        'original_modulation': getattr(self, 'demodulated_metadata', {}).get('modulation_type', 'unknown'),
                        'coding_params': decode_result.get('parameters', {})
                    }
                }
            else:
                return {"success": False, "error": "Decoding failed"}
                
        except Exception as e:
            self.logger.error(f"Error in detect_and_decode: {e}")
            return {"success": False, "error": str(e)}
    
    def has_demodulated_data(self) -> bool:
        """Check if demodulated data is available."""
        return (hasattr(self, 'demodulated_data') and 
                self.demodulated_data is not None and 
                len(self.demodulated_data) > 0)
    
    def _filter_to_frequency_range(self, iq_data: np.ndarray) -> np.ndarray:
        """Filter IQ data to analysis frequency range."""
        try:
            if not hasattr(self, 'frequency_offset'):
                return iq_data
            
            # Apply frequency shift to center the analysis range
            sample_rate = self.settings.sdr.sample_rate
            t = np.arange(len(iq_data)) / sample_rate
            freq_shift = np.exp(-2j * np.pi * self.frequency_offset * t)
            shifted_iq = iq_data * freq_shift
            
            # Low-pass filter to analysis bandwidth
            nyquist = sample_rate / 2
            cutoff = min(self.analysis_bandwidth / 2, nyquist * 0.9)
            normalized_cutoff = cutoff / nyquist
            
            # Design filter
            b, a = scipy.signal.butter(4, normalized_cutoff, btype='low')
            filtered_iq = scipy.signal.filtfilt(b, a, shifted_iq)
            
            return filtered_iq
            
        except Exception as e:
            self.logger.error(f"Error filtering to frequency range: {e}")
            return iq_data
    
    def _detect_modulation_type(self, iq_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect modulation type from IQ data."""
        try:
            # Use modulation analyzer if available
            if hasattr(self, 'modulation_analyzer') and self.modulation_analyzer:
                return self.modulation_analyzer.analyze_modulation(iq_data)
            
            # Simple modulation detection based on signal characteristics
            # Calculate signal statistics
            magnitude = np.abs(iq_data)
            phase = np.angle(iq_data)
            
            # Analyze magnitude variation (AM vs constant envelope)
            mag_std = np.std(magnitude)
            mag_mean = np.mean(magnitude)
            magnitude_variation = mag_std / mag_mean if mag_mean > 0 else 0
            
            # Analyze phase characteristics
            phase_diff = np.diff(np.unwrap(phase))
            phase_std = np.std(phase_diff)
            
            # Simple classification
            if magnitude_variation > 0.2:
                modulation_type = "AM"
            elif phase_std > 0.5:
                modulation_type = "FM"
            else:
                modulation_type = "PSK"
            
            return {
                "success": True,
                "modulation_type": modulation_type,
                "confidence": 0.7,  # Basic detection has lower confidence
                "parameters": {
                    "magnitude_variation": magnitude_variation,
                    "phase_std": phase_std
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting modulation type: {e}")
            return {"success": False, "error": str(e)}
    
    def _detect_coding_type(self, demod_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect channel coding type from demodulated data."""
        try:
            # Use encoding analyzer if available
            if hasattr(self, 'encoding_analyzer') and self.encoding_analyzer:
                return self.encoding_analyzer.analyze_encoding(demod_data)
            
            # Simple coding detection
            # Check for typical coding patterns
            data_len = len(demod_data)
            
            # Check for block code patterns (fixed length blocks)
            if data_len % 7 == 0:
                coding_type = "hamming_7_4"
            elif data_len % 15 == 0:
                coding_type = "bch_15_11"
            elif data_len > 100:  # Longer sequences might be convolutional
                coding_type = "convolutional_1_2"
            else:
                coding_type = "none"
            
            return {
                "success": True,
                "coding_type": coding_type,
                "confidence": 0.6,  # Basic detection has lower confidence
                "parameters": {
                    "data_length": data_len,
                    "estimated_rate": "1/2" if coding_type.startswith("convolutional") else "variable"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting coding type: {e}")
            return {"success": False, "error": str(e)}
    
    def set_performance_mode(self, mode: str):
        """Set performance mode for spectrum computation.
        
        Args:
            mode: 'fast' (15 FPS), 'balanced' (20 FPS), 'quality' (30 FPS)
        """
        if mode in ['fast', 'balanced', 'quality']:
            self._performance_mode = mode
            self.logger.info(f"Performance mode set to: {mode}")
        else:
            self.logger.warning(f"Invalid performance mode: {mode}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'fft_method': self.fft_method,
            'performance_mode': self._performance_mode,
            'frames_skipped': self._spectrum_skip_counter,
            'last_update_time': self._last_spectrum_time
        }