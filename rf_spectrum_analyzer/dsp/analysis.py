"""
Advanced Signal Analysis Tools
Comprehensive signal analysis using sdr and sk_dsp_comm libraries
"""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from scipy.fft import fft, fftfreq, fftshift
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
    from sdr._detection import *
    from sdr import *
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

logger = get_logger('analysis')


@dataclass
class AnalysisConfig:
    """Signal analysis configuration"""
    sample_rate: float = 1e6
    window_size: int = 1024
    overlap: float = 0.5
    window_type: str = "hann"
    freq_resolution: float = 1000.0
    detection_threshold: float = -80.0  # dBm
    averaging_factor: int = 10


class SpectrumAnalyzer:
    """Advanced spectrum analyzer with multiple algorithms"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.previous_psd = None
        self.averaging_buffer = []
        
        if SDR_AVAILABLE:
            self._init_sdr_analyzer()
    
    def _init_sdr_analyzer(self):
        """Initialize SDR spectrum analyzer if available"""
        try:
            if hasattr(sdr, 'spectrum'):
                self.sdr_analyzer = sdr.spectrum.SpectrogramAnalyzer(
                    sample_rate=self.config.sample_rate,
                    window_size=self.config.window_size
                )
        except Exception as e:
            logger.warning(f"Error initializing SDR spectrum analyzer: {e}")
            self.sdr_analyzer = None
    
    def power_spectral_density(self, x: np.ndarray, method: str = "welch") -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power spectral density"""
        if SDR_AVAILABLE and hasattr(self, 'sdr_analyzer'):
            try:
                return self.sdr_analyzer.psd(x, method=method)
            except:
                pass
        
        # Manual implementation
        if method == "welch":
            return self._welch_psd(x)
        elif method == "periodogram":
            return self._periodogram_psd(x)
        elif method == "multitaper":
            return self._multitaper_psd(x)
        else:
            return self._welch_psd(x)
    
    def _welch_psd(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Welch's method for PSD estimation"""
        nperseg = self.config.window_size
        noverlap = int(nperseg * self.config.overlap)
        
        freqs, psd = signal.welch(
            x, fs=self.config.sample_rate, 
            window=self.config.window_type,
            nperseg=nperseg, noverlap=noverlap,
            return_onesided=False
        )
        
        # Convert to dBm (assuming 50 ohm impedance)
        psd_dbm = 10 * np.log10(psd / 1e-3)
        
        return fftshift(freqs), fftshift(psd_dbm)
    
    def _periodogram_psd(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Periodogram method for PSD estimation"""
        freqs, psd = signal.periodogram(
            x, fs=self.config.sample_rate,
            window=self.config.window_type,
            return_onesided=False
        )
        
        psd_dbm = 10 * np.log10(psd / 1e-3)
        return fftshift(freqs), fftshift(psd_dbm)
    
    def _multitaper_psd(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Multitaper method for PSD estimation"""
        try:
            # Use scipy's multitaper if available
            from scipy.signal import windows
            
            # Create multiple tapers
            NW = 4  # Time-bandwidth product
            K = 2*NW - 1  # Number of tapers
            N = len(x)
            
            tapers = windows.dpss(N, NW, K)
            
            # Calculate PSD for each taper
            psds = []
            for taper in tapers:
                windowed = x * taper
                freqs, psd = signal.periodogram(
                    windowed, fs=self.config.sample_rate,
                    return_onesided=False
                )
                psds.append(psd)
            
            # Average PSDs
            avg_psd = np.mean(psds, axis=0)
            psd_dbm = 10 * np.log10(avg_psd / 1e-3)
            
            return fftshift(freqs), fftshift(psd_dbm)
        
        except ImportError:
            logger.warning("Multitaper method not available, falling back to Welch")
            return self._welch_psd(x)
    
    def spectrogram(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate spectrogram"""
        nperseg = self.config.window_size
        noverlap = int(nperseg * self.config.overlap)
        
        freqs, times, Sxx = signal.spectrogram(
            x, fs=self.config.sample_rate,
            window=self.config.window_type,
            nperseg=nperseg, noverlap=noverlap,
            return_onesided=False
        )
        
        # Convert to dBm
        Sxx_dbm = 10 * np.log10(Sxx / 1e-3)
        
        return fftshift(freqs), times, fftshift(Sxx_dbm, axes=0)
    
    def peak_detection(self, freqs: np.ndarray, psd: np.ndarray, 
                      threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Detect peaks in spectrum"""
        if threshold is None:
            threshold = self.config.detection_threshold
        
        # Find peaks above threshold
        peaks, properties = signal.find_peaks(
            psd, height=threshold, distance=10
        )
        
        peak_freqs = freqs[peaks]
        peak_powers = psd[peaks]
        
        return peak_freqs, peak_powers
    
    def averaging(self, psd: np.ndarray) -> np.ndarray:
        """Apply spectrum averaging"""
        self.averaging_buffer.append(psd)
        
        if len(self.averaging_buffer) > self.config.averaging_factor:
            self.averaging_buffer.pop(0)
        
        return np.mean(self.averaging_buffer, axis=0)


class SignalDetector:
    """Advanced signal detection algorithms"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        
        if SDR_AVAILABLE:
            self._init_sdr_detector()
    
    def _init_sdr_detector(self):
        """Initialize SDR signal detector if available"""
        try:
            if hasattr(sdr, '_detection'):
                self.sdr_detector = sdr._detection.EnergyDetector(
                    threshold=self.config.detection_threshold
                )
        except Exception as e:
            logger.warning(f"Error initializing SDR detector: {e}")
            self.sdr_detector = None
    
    def energy_detection(self, x: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Energy-based signal detection"""
        if SDR_AVAILABLE and hasattr(self, 'sdr_detector'):
            try:
                return self.sdr_detector.detect(x)
            except:
                pass
        
        # Manual implementation
        if threshold is None:
            threshold = self.config.detection_threshold
        
        # Calculate energy (RMS power)
        energy = np.mean(np.abs(x)**2)
        
        # Avoid log of zero by adding small epsilon
        energy = max(energy, 1e-12)
        energy_db = 10 * np.log10(energy)  # Power in dB relative to 1
        
        detected = energy_db > threshold
        
        return {
            'detected': detected,
            'energy_db': energy_db,
            'threshold_db': threshold,
            'snr_estimate': energy_db - threshold if detected else None
        }
    
    def matched_filter_detection(self, x: np.ndarray, template: np.ndarray) -> Dict[str, Any]:
        """Matched filter detection"""
        # Normalize template
        template_norm = template / np.sqrt(np.sum(np.abs(template)**2))
        
        # Cross-correlation
        correlation = np.correlate(x, template_norm, mode='full')
        
        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        peak_value = np.abs(correlation[peak_idx])
        
        # Estimate noise level
        noise_power = np.var(np.abs(correlation))
        snr = peak_value**2 / noise_power if noise_power > 0 else float('inf')
        
        return {
            'correlation': correlation,
            'peak_index': peak_idx,
            'peak_value': peak_value,
            'snr_linear': snr,
            'snr_db': 10 * np.log10(snr) if snr > 0 else float('-inf')
        }
    
    def cfar_detection(self, x: np.ndarray, guard_cells: int = 4, 
                      reference_cells: int = 16, pfa: float = 1e-6) -> Dict[str, Any]:
        """Constant False Alarm Rate (CFAR) detection"""
        # For CA-CFAR (Cell Averaging CFAR), the threshold scaling factor is:
        alpha = reference_cells * (pfa**(-1/reference_cells) - 1)
        
        power = np.abs(x)**2
        detections = np.zeros(len(power), dtype=bool)
        thresholds = np.zeros(len(power))
        
        # Pre-compute median power for outlier detection
        median_power = np.median(power)
        outlier_threshold = median_power * 5  # Samples > 5x median are likely targets
        
        for i in range(len(power)):
            # Define reference cells around the cell under test
            # Left reference window (before guard cells)
            left_start = max(0, i - guard_cells - reference_cells)
            left_end = i - guard_cells
            
            # Right reference window (after guard cells)
            right_start = i + guard_cells + 1
            right_end = min(len(power), i + guard_cells + reference_cells + 1)
            
            # Collect reference samples from both sides
            reference_samples = []
            
            # Add left reference samples (exclude outliers to avoid target contamination)
            if left_end > left_start and left_start >= 0 and left_end > 0:
                left_samples = power[left_start:left_end]
                # Filter out likely target contamination
                clean_left = left_samples[left_samples <= outlier_threshold]
                if len(clean_left) > 0:
                    reference_samples.extend(clean_left)
                    
            # Add right reference samples (exclude outliers)
            if right_start < len(power) and right_end > right_start and right_start >= 0:
                right_samples = power[right_start:right_end]
                # Filter out likely target contamination  
                clean_right = right_samples[right_samples <= outlier_threshold]
                if len(clean_right) > 0:
                    reference_samples.extend(clean_right)
            
            # Need sufficient reference samples for reliable detection
            if len(reference_samples) >= max(4, reference_cells // 4):  # More lenient requirement
                noise_estimate = np.mean(reference_samples)
                threshold = alpha * noise_estimate
                detected = power[i] > threshold
            else:
                threshold = np.inf  # Cannot detect at edges reliably
                detected = False
            
            detections[i] = detected
            thresholds[i] = threshold
        
        detection_indices = np.where(detections)[0]
        
        return {
            'detections': detections,
            'thresholds': thresholds,
            'test_statistic': power,
            'detection_indices': detection_indices
        }


class ParameterEstimator:
    """Signal parameter estimation algorithms"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        
        if SDR_AVAILABLE:
            self._init_sdr_estimator()
    
    def _init_sdr_estimator(self):
        """Initialize SDR parameter estimator if available"""
        try:
            if hasattr(sdr, '_estimation'):
                self.sdr_estimator = sdr._estimation.ParameterEstimator()
        except Exception as e:
            logger.warning(f"Error initializing SDR estimator: {e}")
            self.sdr_estimator = None
    
    def frequency_estimation(self, x: np.ndarray, method: str = "fft") -> Dict[str, Any]:
        """Estimate dominant frequency"""
        if SDR_AVAILABLE and hasattr(self, 'sdr_estimator'):
            try:
                return self.sdr_estimator.estimate_frequency(x, method=method)
            except:
                pass
        
        if method == "fft":
            return self._fft_frequency_estimation(x)
        elif method == "argmax":
            return self._argmax_frequency_estimation(x)
        elif method == "esprit":
            return self._esprit_frequency_estimation(x)
        else:
            return self._fft_frequency_estimation(x)
    
    def _fft_frequency_estimation(self, x: np.ndarray) -> Dict[str, Any]:
        """FFT-based frequency estimation"""
        N = len(x)
        freqs = fftfreq(N, 1/self.config.sample_rate)
        fft_x = fft(x)
        
        # Find peak
        peak_idx = np.argmax(np.abs(fft_x))
        estimated_freq = freqs[peak_idx]
        
        # Parabolic interpolation for better accuracy
        if 0 < peak_idx < N-1:
            y1, y2, y3 = np.abs(fft_x[peak_idx-1:peak_idx+2])
            a = (y1 - 2*y2 + y3) / 2
            b = (y3 - y1) / 2
            
            if a != 0:
                correction = -b / (2*a)
                estimated_freq = freqs[peak_idx] + correction * (freqs[1] - freqs[0])
        
        return {
            'frequency': estimated_freq,
            'method': 'fft',
            'confidence': np.abs(fft_x[peak_idx]) / np.sum(np.abs(fft_x))
        }
    
    def _argmax_frequency_estimation(self, x: np.ndarray) -> Dict[str, Any]:
        """Argmax-based frequency estimation"""
        # Instantaneous frequency estimation
        analytic_signal = signal.hilbert(x)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_freq = np.diff(instantaneous_phase) / (2*np.pi) * self.config.sample_rate
        
        # Most common frequency
        hist, bin_edges = np.histogram(instantaneous_freq, bins=100)
        peak_bin = np.argmax(hist)
        estimated_freq = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
        
        return {
            'frequency': estimated_freq,
            'method': 'argmax',
            'instantaneous_frequencies': instantaneous_freq
        }
    
    def _esprit_frequency_estimation(self, x: np.ndarray, model_order: int = 4) -> Dict[str, Any]:
        """ESPRIT algorithm for frequency estimation"""
        try:
            N = len(x)
            M = model_order
            
            # Construct Hankel matrix
            H = np.array([x[i:i+N-M+1] for i in range(M)])
            
            # SVD
            U, s, Vh = np.linalg.svd(H, full_matrices=False)
            
            # Signal subspace
            Us = U[:, :model_order//2]
            
            # Split into two parts
            Us1 = Us[:-1, :]
            Us2 = Us[1:, :]
            
            # Solve eigenvalue problem
            try:
                eigenvals = np.linalg.eigvals(np.linalg.pinv(Us1) @ Us2)
                frequencies = np.angle(eigenvals) * self.config.sample_rate / (2 * np.pi)
                
                return {
                    'frequencies': frequencies,
                    'method': 'esprit',
                    'model_order': model_order
                }
            except np.linalg.LinAlgError:
                return self._fft_frequency_estimation(x)
                
        except Exception as e:
            logger.warning(f"ESPRIT estimation failed: {e}")
            return self._fft_frequency_estimation(x)
    
    def amplitude_estimation(self, x: np.ndarray) -> Dict[str, Any]:
        """Estimate signal amplitude"""
        rms = np.sqrt(np.mean(np.abs(x)**2))
        peak = np.max(np.abs(x))
        
        return {
            'rms': rms,
            'peak': peak,
            'crest_factor': peak / rms if rms > 0 else float('inf'),
            'power_dbm': 10 * np.log10(rms**2 / 1e-3)
        }
    
    def phase_estimation(self, x: np.ndarray) -> Dict[str, Any]:
        """Estimate signal phase"""
        analytic_signal = signal.hilbert(x)
        instantaneous_phase = np.angle(analytic_signal)
        
        # Unwrapped phase
        unwrapped_phase = np.unwrap(instantaneous_phase)
        
        # Linear phase trend (frequency)
        if len(unwrapped_phase) > 1:
            phase_slope = np.polyfit(np.arange(len(unwrapped_phase)), unwrapped_phase, 1)[0]
            phase_offset = unwrapped_phase[0]
        else:
            phase_slope = 0
            phase_offset = instantaneous_phase[0] if len(instantaneous_phase) > 0 else 0
        
        return {
            'instantaneous_phase': instantaneous_phase,
            'unwrapped_phase': unwrapped_phase,
            'phase_offset': phase_offset,
            'frequency_estimate': phase_slope * self.config.sample_rate / (2 * np.pi)
        }


class InterferenceAnalyzer:
    """Interference detection and analysis"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.baseline_spectrum = None
    
    def set_baseline(self, x: np.ndarray):
        """Set baseline spectrum for comparison"""
        analyzer = SpectrumAnalyzer(self.config)
        freqs, psd = analyzer.power_spectral_density(x)
        self.baseline_spectrum = (freqs, psd)
    
    def detect_interference(self, x: np.ndarray, threshold_db: float = 10.0) -> Dict[str, Any]:
        """Detect interference by comparing to baseline"""
        analyzer = SpectrumAnalyzer(self.config)
        freqs, current_psd = analyzer.power_spectral_density(x)
        
        if self.baseline_spectrum is None:
            return {
                'interference_detected': False,
                'message': 'No baseline spectrum available'
            }
        
        baseline_freqs, baseline_psd = self.baseline_spectrum
        
        # Interpolate baseline to current frequency grid if needed
        if not np.array_equal(freqs, baseline_freqs):
            baseline_psd = np.interp(freqs, baseline_freqs, baseline_psd)
        
        # Calculate difference
        psd_diff = current_psd - baseline_psd
        
        # Find interference peaks
        interference_mask = psd_diff > threshold_db
        interference_freqs = freqs[interference_mask]
        interference_levels = psd_diff[interference_mask]
        
        return {
            'interference_detected': len(interference_freqs) > 0,
            'interference_frequencies': interference_freqs,
            'interference_levels_db': interference_levels,
            'psd_difference': psd_diff,
            'threshold_db': threshold_db
        }
    
    def classify_interference(self, interference_freqs: np.ndarray, 
                            interference_levels: np.ndarray) -> Dict[str, Any]:
        """Classify type of interference"""
        classifications = []
        
        for freq, level in zip(interference_freqs, interference_levels):
            classification = {
                'frequency': freq,
                'level_db': level,
                'type': 'unknown'
            }
            
            # Simple classification rules based on level
            if level > 30:
                classification['type'] = 'strong_narrowband'
            elif level > 15:
                classification['type'] = 'moderate_narrowband'
            elif level > 5:
                classification['type'] = 'weak_narrowband'
            
            # Check for known interference sources
            if 2400e6 <= freq <= 2485e6:
                classification['potential_source'] = 'WiFi/Bluetooth'
            elif (824e6 <= freq <= 960e6) or (1850e6 <= freq <= 1990e6):  # Extended cellular range
                classification['potential_source'] = 'Cellular'
            elif 88e6 <= freq <= 108e6:  # FM radio range
                classification['potential_source'] = 'FM Radio'
            else:
                # Add more specific ranges for better detection
                if 'potential_source' not in classification:
                    classification['potential_source'] = 'Unknown'
            
            classifications.append(classification)
        
        return {
            'classifications': classifications,
            'total_interferences': len(classifications)
        }


# Convenience functions
def analyze_spectrum(signal: np.ndarray, sample_rate: float = 1e6, 
                    method: str = "welch") -> Dict[str, Any]:
    """Convenient spectrum analysis function"""
    config = AnalysisConfig(sample_rate=sample_rate)
    analyzer = SpectrumAnalyzer(config)
    
    freqs, psd = analyzer.power_spectral_density(signal, method=method)
    peak_freqs, peak_powers = analyzer.peak_detection(freqs, psd)
    
    return {
        'frequencies': freqs,
        'psd_dbm': psd,
        'peak_frequencies': peak_freqs,
        'peak_powers': peak_powers,
        'sample_rate': sample_rate,
        'method': method
    }


def detect_signals(signal: np.ndarray, threshold_db: float = -80.0) -> Dict[str, Any]:
    """Convenient signal detection function"""
    config = AnalysisConfig(detection_threshold=threshold_db)
    detector = SignalDetector(config)
    
    energy_result = detector.energy_detection(signal, threshold_db)
    cfar_result = detector.cfar_detection(signal)
    
    return {
        'energy_detection': energy_result,
        'cfar_detection': cfar_result
    }


def estimate_parameters(signal: np.ndarray, sample_rate: float = 1e6) -> Dict[str, Any]:
    """Convenient parameter estimation function"""
    config = AnalysisConfig(sample_rate=sample_rate)
    estimator = ParameterEstimator(config)
    
    freq_est = estimator.frequency_estimation(signal)
    amp_est = estimator.amplitude_estimation(signal)
    phase_est = estimator.phase_estimation(signal)
    
    return {
        'frequency': freq_est,
        'amplitude': amp_est,
        'phase': phase_est
    }