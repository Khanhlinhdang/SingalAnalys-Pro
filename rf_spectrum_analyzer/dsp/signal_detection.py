"""
Signal Detection Engine using SDR Detection Module
Implements comprehensive signal detection capabilities using sdr._detection.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

try:
    import sdr
    SDR_AVAILABLE = True
except ImportError:
    SDR_AVAILABLE = False

logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Signal detection methods."""
    ENERGY = "energy"
    CORRELATION = "correlation" 
    COMBINED = "combined"
    ADAPTIVE = "adaptive"


@dataclass
class DetectionResult:
    """Signal detection result."""
    signal_detected: bool
    detection_method: str
    confidence: float
    snr_estimate: float
    test_statistic: float
    threshold: float
    p_fa: float
    p_d: float
    noise_variance: float


@dataclass
class SpectrumSensingResult:
    """Spectrum sensing result for cognitive radio."""
    frequency_band: str
    center_frequency: float
    bandwidth: float
    signal_detected: bool
    occupancy_probability: float
    signal_power: float
    noise_power: float
    snr_db: float
    detection_confidence: float


class SignalDetectionEngine:
    """
    Advanced signal detection engine using sdr._detection algorithms.
    Supports energy detection, correlation detection, and hybrid approaches.
    """
    
    def __init__(self, sample_rate: float):
        """
        Initialize signal detection engine.
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.noise_variance = 1.0
        self.calibrated = False
        
        # Detection parameters
        self.default_p_fa = 1e-6  # Probability of false alarm
        self.energy_integration_length = 1000  # Samples for energy integration
        self.correlation_threshold = 0.7
        
        # Known signal templates for correlation detection
        self.signal_templates = {}
        
        # Detection history for adaptive algorithms
        self.detection_history = []
        self.noise_history = []
        
        logger.info(f"Signal Detection Engine initialized with {sample_rate} Hz sample rate")
        logger.info(f"SDR library available: {SDR_AVAILABLE}")
    
    def calibrate_noise_floor(self, noise_samples: np.ndarray, method: str = 'robust') -> float:
        """
        Calibrate noise floor estimation.
        
        Args:
            noise_samples: Pure noise samples for calibration
            method: Calibration method ('robust', 'simple', 'adaptive')
            
        Returns:
            Estimated noise variance
        """
        try:
            if method == 'robust':
                # Robust estimation using percentiles
                power_samples = np.abs(noise_samples) ** 2
                # Use lower percentiles to avoid outliers
                noise_var = np.percentile(power_samples, 25)
            elif method == 'simple':
                # Simple variance calculation
                noise_var = np.var(noise_samples)
            elif method == 'adaptive':
                # Adaptive estimation with outlier rejection
                power_samples = np.abs(noise_samples) ** 2
                median_power = np.median(power_samples)
                mad = np.median(np.abs(power_samples - median_power))
                # Robust variance estimate
                noise_var = (1.4826 * mad) ** 2
            else:
                noise_var = np.var(noise_samples)
            
            self.noise_variance = max(noise_var, 1e-12)  # Avoid division by zero
            self.calibrated = True
            
            # Store in history
            self.noise_history.append(self.noise_variance)
            if len(self.noise_history) > 1000:
                self.noise_history.pop(0)
            
            logger.info(f"Noise floor calibrated: σ² = {self.noise_variance:.2e} ({method} method)")
            return self.noise_variance
            
        except Exception as e:
            logger.error(f"Noise calibration error: {e}")
            self.noise_variance = 1.0
            return self.noise_variance
    
    def add_signal_template(self, name: str, template: np.ndarray):
        """
        Add known signal template for correlation detection.
        
        Args:
            name: Template name/identifier
            template: Signal template (complex samples)
        """
        # Normalize template
        template_norm = template / np.sqrt(np.sum(np.abs(template) ** 2))
        self.signal_templates[name] = template_norm.astype(np.complex64)
        logger.info(f"Added signal template '{name}' with {len(template)} samples")
    
    def energy_detection(self, signal: np.ndarray, 
                        p_fa: Optional[float] = None,
                        integration_length: Optional[int] = None) -> DetectionResult:
        """
        Perform energy detection using sdr._detection.EnergyDetector.
        
        Args:
            signal: Input signal
            p_fa: Probability of false alarm (default: self.default_p_fa)
            integration_length: Number of samples to integrate (default: self.energy_integration_length)
            
        Returns:
            Detection result
        """
        try:
            if p_fa is None:
                p_fa = self.default_p_fa
            if integration_length is None:
                integration_length = self.energy_integration_length
            
            # Ensure we have enough samples
            N_samples = min(len(signal), integration_length)
            test_signal = signal[:N_samples]
            
            # Calculate test statistic (energy)
            test_statistic = np.sum(np.abs(test_signal) ** 2)
            
            # Use sdr._detection for theoretical calculations
            if SDR_AVAILABLE and hasattr(sdr, 'EnergyDetector'):
                try:
                    # Calculate detection threshold
                    threshold = sdr.EnergyDetector.threshold(
                        N_nc=N_samples,
                        p_fa=p_fa,
                        sigma2=self.noise_variance,
                        complex=True
                    )
                    
                    # Calculate SNR estimate
                    signal_power = test_statistic / N_samples
                    snr_linear = signal_power / self.noise_variance - 1
                    snr_db = 10 * np.log10(max(snr_linear, 1e-10))
                    
                    # Calculate probability of detection
                    p_d = sdr.EnergyDetector.p_d(
                        snr=snr_db,
                        N_nc=N_samples,
                        p_fa=p_fa,
                        complex=True
                    )
                    
                    # Make detection decision
                    signal_detected = test_statistic > threshold
                    confidence = float(p_d) if signal_detected else (1.0 - float(p_fa))
                    
                except Exception as e:
                    logger.warning(f"SDR energy detection failed: {e}, using fallback")
                    threshold, signal_detected, confidence, snr_db, p_d = self._fallback_energy_detection(
                        test_statistic, N_samples, p_fa
                    )
            else:
                threshold, signal_detected, confidence, snr_db, p_d = self._fallback_energy_detection(
                    test_statistic, N_samples, p_fa
                )
            
            result = DetectionResult(
                signal_detected=signal_detected,
                detection_method="energy",
                confidence=confidence,
                snr_estimate=snr_db,
                test_statistic=test_statistic,
                threshold=threshold,
                p_fa=p_fa,
                p_d=float(p_d),
                noise_variance=self.noise_variance
            )
            
            # Store in history
            self.detection_history.append(result)
            if len(self.detection_history) > 1000:
                self.detection_history.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Energy detection error: {e}")
            return DetectionResult(
                signal_detected=False,
                detection_method="energy",
                confidence=0.0,
                snr_estimate=-50.0,
                test_statistic=0.0,
                threshold=1.0,
                p_fa=p_fa or self.default_p_fa,
                p_d=0.0,
                noise_variance=self.noise_variance
            )
    
    def _fallback_energy_detection(self, test_statistic: float, N_samples: int, 
                                  p_fa: float) -> Tuple[float, bool, float, float, float]:
        """Fallback energy detection without sdr library."""
        try:
            import scipy.stats
            
            # Chi-squared threshold for complex signals (2*N degrees of freedom)
            threshold = self.noise_variance * scipy.stats.chi2.isf(p_fa, 2*N_samples) / 2
            
            signal_detected = test_statistic > threshold
            
            # Simple SNR estimate
            signal_power = test_statistic / N_samples
            snr_linear = max(signal_power / self.noise_variance - 1, 1e-10)
            snr_db = 10 * np.log10(snr_linear)
            
            # Simple P_D estimate
            if signal_detected:
                confidence = 0.9
                p_d = 0.9
            else:
                confidence = 1.0 - p_fa
                p_d = 0.1
            
            return threshold, signal_detected, confidence, snr_db, p_d
            
        except Exception as e:
            logger.error(f"Fallback energy detection error: {e}")
            return 1.0, False, 0.0, -50.0, 0.0
    
    def correlation_detection(self, signal: np.ndarray, 
                            template_name: str,
                            p_fa: Optional[float] = None) -> DetectionResult:
        """
        Perform correlation detection using sdr._detection.ReplicaCorrelator.
        
        Args:
            signal: Input signal
            template_name: Name of signal template to correlate with
            p_fa: Probability of false alarm
            
        Returns:
            Detection result
        """
        try:
            if template_name not in self.signal_templates:
                logger.error(f"Template '{template_name}' not found")
                return DetectionResult(
                    signal_detected=False,
                    detection_method="correlation",
                    confidence=0.0,
                    snr_estimate=-50.0,
                    test_statistic=0.0,
                    threshold=1.0,
                    p_fa=p_fa or self.default_p_fa,
                    p_d=0.0,
                    noise_variance=self.noise_variance
                )
            
            if p_fa is None:
                p_fa = self.default_p_fa
            
            template = self.signal_templates[template_name]
            
            # Cross-correlation
            correlation = np.correlate(signal, template, mode='full')
            
            # Find maximum correlation
            max_corr_idx = np.argmax(np.abs(correlation))
            test_statistic = np.real(correlation[max_corr_idx])
            
            # Template energy
            template_energy = np.sum(np.abs(template) ** 2)
            
            # Use sdr._detection for theoretical calculations
            if SDR_AVAILABLE and hasattr(sdr, 'ReplicaCorrelator'):
                try:
                    # Calculate ENR (Energy-to-Noise Ratio)
                    signal_energy = template_energy  # Received energy
                    enr_linear = signal_energy / self.noise_variance
                    enr_db = 10 * np.log10(max(enr_linear, 1e-10))
                    
                    # Calculate probability of detection
                    p_d = sdr.ReplicaCorrelator.p_d(
                        enr=enr_db,
                        p_fa=p_fa,
                        complex=True
                    )
                    
                    # Calculate threshold
                    threshold = sdr.ReplicaCorrelator.threshold(
                        p_fa=p_fa,
                        energy=signal_energy,
                        sigma2=self.noise_variance,
                        complex=True
                    )
                    
                    # Detection decision
                    signal_detected = test_statistic > threshold
                    confidence = float(p_d) if signal_detected else (1.0 - float(p_fa))
                    snr_db = enr_db
                    
                except Exception as e:
                    logger.warning(f"SDR correlation detection failed: {e}, using fallback")
                    threshold, signal_detected, confidence, snr_db, p_d = self._fallback_correlation_detection(
                        test_statistic, template_energy, p_fa
                    )
            else:
                threshold, signal_detected, confidence, snr_db, p_d = self._fallback_correlation_detection(
                    test_statistic, template_energy, p_fa
                )
            
            result = DetectionResult(
                signal_detected=signal_detected,
                detection_method="correlation",
                confidence=confidence,
                snr_estimate=snr_db,
                test_statistic=test_statistic,
                threshold=threshold,
                p_fa=p_fa,
                p_d=float(p_d),
                noise_variance=self.noise_variance
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Correlation detection error: {e}")
            return DetectionResult(
                signal_detected=False,
                detection_method="correlation",
                confidence=0.0,
                snr_estimate=-50.0,
                test_statistic=0.0,
                threshold=1.0,
                p_fa=p_fa or self.default_p_fa,
                p_d=0.0,
                noise_variance=self.noise_variance
            )
    
    def _fallback_correlation_detection(self, test_statistic: float, template_energy: float,
                                      p_fa: float) -> Tuple[float, bool, float, float, float]:
        """Fallback correlation detection without sdr library."""
        try:
            # Simple threshold based on correlation energy
            threshold = self.correlation_threshold * np.sqrt(template_energy * self.noise_variance)
            
            signal_detected = test_statistic > threshold
            
            # Simple SNR estimate
            snr_linear = max(test_statistic / (template_energy * self.noise_variance), 1e-10)
            snr_db = 10 * np.log10(snr_linear)
            
            # Simple confidence estimate
            if signal_detected:
                confidence = min(0.95, test_statistic / threshold * 0.7)
                p_d = confidence
            else:
                confidence = 1.0 - p_fa
                p_d = 0.1
            
            return threshold, signal_detected, confidence, snr_db, p_d
            
        except Exception as e:
            logger.error(f"Fallback correlation detection error: {e}")
            return 1.0, False, 0.0, -50.0, 0.0
    
    def spectrum_sensing(self, signal: np.ndarray, 
                        frequency_bands: Dict[str, Tuple[float, float]],
                        method: DetectionMethod = DetectionMethod.ENERGY) -> Dict[str, SpectrumSensingResult]:
        """
        Perform spectrum sensing across multiple frequency bands.
        
        Args:
            signal: Input signal (full bandwidth)
            frequency_bands: Dictionary of {band_name: (start_freq, stop_freq)}
            method: Detection method to use
            
        Returns:
            Dictionary of spectrum sensing results per band
        """
        try:
            results = {}
            
            for band_name, (f_start, f_stop) in frequency_bands.items():
                try:
                    # Extract frequency band (simplified - would need proper filtering)
                    center_freq = (f_start + f_stop) / 2
                    bandwidth = f_stop - f_start
                    
                    # For simplicity, use entire signal (in practice would filter to band)
                    band_signal = signal
                    
                    # Perform detection
                    if method == DetectionMethod.ENERGY:
                        detection_result = self.energy_detection(band_signal)
                    elif method == DetectionMethod.CORRELATION and self.signal_templates:
                        # Use first available template
                        template_name = list(self.signal_templates.keys())[0]
                        detection_result = self.correlation_detection(band_signal, template_name)
                    else:
                        detection_result = self.energy_detection(band_signal)
                    
                    # Calculate band-specific metrics
                    signal_power = np.mean(np.abs(band_signal) ** 2)
                    noise_power = self.noise_variance
                    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else -50
                    
                    # Occupancy probability (simplified)
                    occupancy_prob = detection_result.confidence if detection_result.signal_detected else 0.0
                    
                    sensing_result = SpectrumSensingResult(
                        frequency_band=band_name,
                        center_frequency=center_freq,
                        bandwidth=bandwidth,
                        signal_detected=detection_result.signal_detected,
                        occupancy_probability=occupancy_prob,
                        signal_power=signal_power,
                        noise_power=noise_power,
                        snr_db=snr_db,
                        detection_confidence=detection_result.confidence
                    )
                    
                    results[band_name] = sensing_result
                    
                except Exception as e:
                    logger.error(f"Spectrum sensing error for band {band_name}: {e}")
                    # Add failed result
                    results[band_name] = SpectrumSensingResult(
                        frequency_band=band_name,
                        center_frequency=(f_start + f_stop) / 2,
                        bandwidth=f_stop - f_start,
                        signal_detected=False,
                        occupancy_probability=0.0,
                        signal_power=0.0,
                        noise_power=self.noise_variance,
                        snr_db=-50.0,
                        detection_confidence=0.0
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Spectrum sensing error: {e}")
            return {}
    
    def adaptive_detection(self, signal: np.ndarray) -> DetectionResult:
        """
        Adaptive detection that chooses best method based on signal characteristics.
        
        Args:
            signal: Input signal
            
        Returns:
            Detection result using optimal method
        """
        try:
            # Analyze signal characteristics
            signal_power = np.mean(np.abs(signal) ** 2)
            signal_variance = np.var(np.abs(signal))
            
            # Choose detection method based on characteristics
            if len(self.signal_templates) > 0 and signal_variance > 0.1:
                # Use correlation if templates available and signal has structure
                template_name = list(self.signal_templates.keys())[0]
                result = self.correlation_detection(signal, template_name)
                result.detection_method = "adaptive_correlation"
            else:
                # Use energy detection for general case
                result = self.energy_detection(signal)
                result.detection_method = "adaptive_energy"
            
            return result
            
        except Exception as e:
            logger.error(f"Adaptive detection error: {e}")
            return self.energy_detection(signal)
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection performance statistics."""
        try:
            if not self.detection_history:
                return {"status": "No detection history available"}
            
            recent_detections = self.detection_history[-100:]  # Last 100 detections
            
            detection_rate = sum(1 for d in recent_detections if d.signal_detected) / len(recent_detections)
            avg_confidence = np.mean([d.confidence for d in recent_detections])
            avg_snr = np.mean([d.snr_estimate for d in recent_detections])
            
            stats = {
                "total_detections": len(self.detection_history),
                "recent_detection_rate": detection_rate,
                "average_confidence": avg_confidence,
                "average_snr_db": avg_snr,
                "noise_variance": self.noise_variance,
                "calibrated": self.calibrated,
                "available_templates": list(self.signal_templates.keys()),
                "sdr_library_available": SDR_AVAILABLE
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics calculation error: {e}")
            return {"error": str(e)}


def create_signal_detector(sample_rate: float) -> SignalDetectionEngine:
    """Factory function to create signal detection engine."""
    return SignalDetectionEngine(sample_rate)