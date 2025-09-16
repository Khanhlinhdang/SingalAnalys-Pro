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
        
        # Analysis results cache
        self.last_modulation_analysis = None
        self.last_encoding_analysis = None
        
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
    
    def process_complete_chain(self, iq_samples: np.ndarray) -> Dict[str, Any]:
        """
        Complete processing chain: modulation detection -> demodulation -> 
        encoding detection -> decoding.
        
        Args:
            iq_samples: Complex IQ signal data
            
        Returns:
            Dictionary containing all processing results
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
            
            # Step 1: Modulation analysis
            mod_analysis = self.analyze_modulation(iq_samples)
            result["modulation_analysis"] = mod_analysis
            
            if mod_analysis["confidence"] < 0.3:
                result["error"] = "Low confidence in modulation detection"
                return result
            
            # Step 2: Demodulation
            demod_result = self.demodulate_signal(
                iq_samples, 
                mod_analysis["type"], 
                mod_analysis["parameters"]
            )
            result["demodulation"] = demod_result
            
            if not demod_result.get("success", False):
                result["error"] = "Demodulation failed"
                return result
            
            # Step 3: Encoding analysis (if we have digital data)
            demod_data = demod_result.get("demodulated_data", np.array([]))
            if demod_result.get("data_type") == "digital" and len(demod_data) > 0:
                
                # Convert to binary if needed
                if demod_data.dtype != bool and demod_data.dtype != int:
                    binary_data = (demod_data > np.mean(demod_data)).astype(int)
                else:
                    binary_data = demod_data.astype(int)
                
                enc_analysis = self.analyze_encoding(binary_data)
                result["encoding_analysis"] = enc_analysis
                
                # Step 4: Decoding
                if enc_analysis["type"] != "None" and enc_analysis["confidence"] > 0.3:
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
            
            result["success"] = True
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
            self.logger.info(f"Updated sample rate to {sample_rate} Hz")
        except Exception as e:
            self.logger.error(f"Error updating sample rate: {e}")
        self.spectrum_history.clear()