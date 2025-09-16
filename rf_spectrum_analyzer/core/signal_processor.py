"""
Signal Processor - Core DSP functionality
Integrates multiple DSP libraries for comprehensive signal analysis.
"""

import numpy as np
import scipy.signal
import logging
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
        
        # Analysis results cache
        self.last_modulation_analysis = None
        self.last_encoding_analysis = None
        self.last_detection_result = None
        self.last_tdma_analysis = None
        
        # Detection state variables
        self._current_iq_data = None
        self._auto_detection_enabled = False
        self._advanced_analysis_enabled = False
        self._detection_threshold_dbm = -80.0
        self._detection_interval_ms = 100
        
        self.logger.info("Signal processor initialized")
    
    def _setup_fft(self):
        """Setup FFT objects for optimal performance."""
        if PYFFTW_AVAILABLE:
            try:
                # Setup FFTW for better performance
                self.fft_input = pyfftw.empty_aligned(self.fft_size, dtype='complex64')
                self.fft_output = pyfftw.empty_aligned(self.fft_size, dtype='complex64')
                self.fft_object = pyfftw.FFTW(self.fft_input, self.fft_output, 
                                            direction='FFTW_FORWARD', 
                                            flags=('FFTW_MEASURE',))
                self.fft_method = 'pyfftw'
                self.logger.info("Using FFTW for FFT computation")
            except Exception as e:
                self.logger.warning(f"Failed to setup FFTW: {e}, falling back to numpy")
                self.fft_method = 'numpy'
        else:
            self.fft_method = 'numpy'
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
        """Compute power spectrum from IQ samples."""
        try:
            if len(iq_samples) < self.fft_size:
                return None
            
            # Extract samples for FFT
            samples = iq_samples[:self.fft_size].astype(np.complex64)
            
            # Apply window
            windowed_samples = samples * self.window
            
            # Compute FFT
            if self.fft_method == 'pyfftw':
                self.fft_input[:] = windowed_samples
                self.fft_object()
                fft_result = self.fft_output.copy()
            else:
                fft_result = np.fft.fft(windowed_samples)
            
            # Compute power spectrum
            power_spectrum = np.abs(fft_result) ** 2
            
            # Convert to dB
            power_db = 10 * np.log10(power_spectrum + 1e-12)
            
            # FFT shift to center DC
            power_db_shifted = np.fft.fftshift(power_db)
            
            # Apply averaging if enabled
            if self.averaging > 1:
                power_db_shifted = self._apply_averaging(power_db_shifted)
            
            return power_db_shifted
            
        except Exception as e:
            self.logger.error(f"Error computing spectrum: {e}")
            return None
    
    def _apply_averaging(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply spectrum averaging."""
        self.spectrum_history.append(spectrum.copy())
        
        # Keep only required number of spectra
        if len(self.spectrum_history) > self.averaging:
            self.spectrum_history.pop(0)
        
        # Return averaged spectrum
        return np.mean(self.spectrum_history, axis=0)
    
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
            
            # Shift FFT result
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