"""
TDMA Burst Detector using SDR Detection Module
Implements burst detection for TDMA signals using sdr._detection algorithms.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    import sdr
    SDR_AVAILABLE = True
except ImportError:
    SDR_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BurstInfo:
    """Information about detected burst."""
    start_sample: int
    end_sample: int
    duration_samples: int
    energy: float
    snr_estimate: float
    sync_correlation: float
    confidence: float


@dataclass
class TDMAFrameInfo:
    """TDMA frame structure information."""
    frame_period_samples: int
    frame_period_seconds: float
    slot_duration_samples: int
    slot_duration_seconds: float
    guard_time_samples: int
    duty_cycle: float
    slots_per_frame: int


class TDMABurstDetector:
    """
    TDMA Burst Detector using sdr._detection module for accurate burst timing detection.
    Supports GSM, DECT, and other TDMA standards.
    """
    
    def __init__(self, sample_rate: float, 
                 expected_burst_length: Optional[int] = None,
                 expected_guard_time: Optional[int] = None):
        """
        Initialize TDMA burst detector.
        
        Args:
            sample_rate: Sample rate in Hz
            expected_burst_length: Expected burst length in samples (optional)
            expected_guard_time: Expected guard time in samples (optional)
        """
        self.sample_rate = sample_rate
        self.expected_burst_length = expected_burst_length
        self.expected_guard_time = expected_guard_time
        
        # Detection parameters
        self.energy_threshold = 0.0  # Will be set adaptively
        self.correlation_threshold = 0.7
        self.min_burst_length = int(0.1e-3 * sample_rate)  # 0.1ms minimum
        self.max_burst_length = int(10e-3 * sample_rate)   # 10ms maximum
        
        # Known sync patterns for different standards
        self.sync_patterns = {
            'GSM': self._generate_gsm_training_sequence(),
            'DECT': self._generate_dect_sync_pattern(),
            'GENERIC': self._generate_generic_sync_pattern()
        }
        
        self.current_sync_pattern = None
        self.noise_variance = 1.0
        
        logger.info(f"TDMA Burst Detector initialized for {sample_rate} Hz")
    
    def _generate_gsm_training_sequence(self) -> np.ndarray:
        """Generate GSM training sequence for burst detection."""
        # GSM Training Sequence Code 0 (most common)
        tsc0_bits = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1])
        # Convert to GMSK-like signal (simplified)
        tsc0_symbols = 2 * tsc0_bits - 1  # BPSK mapping
        # Upsample to match sample rate (simplified)
        samples_per_symbol = max(1, int(self.sample_rate / 270833))  # GSM symbol rate
        upsampled = np.repeat(tsc0_symbols, samples_per_symbol)
        return upsampled.astype(np.complex64)
    
    def _generate_dect_sync_pattern(self) -> np.ndarray:
        """Generate DECT sync pattern."""
        # DECT synchronization word (simplified)
        sync_bits = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1])
        sync_symbols = 2 * sync_bits - 1
        samples_per_symbol = max(1, int(self.sample_rate / 1152000))  # DECT symbol rate
        upsampled = np.repeat(sync_symbols, samples_per_symbol)
        return upsampled.astype(np.complex64)
    
    def _generate_generic_sync_pattern(self) -> np.ndarray:
        """Generate generic sync pattern for unknown TDMA systems."""
        # Barker sequence for good correlation properties
        barker11 = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1])
        samples_per_symbol = max(1, int(self.sample_rate / 100000))  # Generic rate
        upsampled = np.repeat(barker11, samples_per_symbol)
        return upsampled.astype(np.complex64)
    
    def set_sync_pattern(self, pattern_input, custom_pattern: Optional[np.ndarray] = None):
        """
        Set synchronization pattern for burst detection.
        
        Args:
            pattern_input: Either pattern type string ('GSM', 'DECT', 'GENERIC') or numpy array
            custom_pattern: Custom sync pattern (overrides pattern_input if provided)
        """
        try:
            if custom_pattern is not None:
                # Use provided custom pattern
                self.current_sync_pattern = custom_pattern.astype(np.complex64)
                logger.info("Custom sync pattern set")
            elif isinstance(pattern_input, np.ndarray):
                # Pattern provided as numpy array
                self.current_sync_pattern = pattern_input.astype(np.complex64)
                logger.info("Numpy array sync pattern set")
            elif isinstance(pattern_input, str) and pattern_input in self.sync_patterns:
                # Pattern type string
                self.current_sync_pattern = self.sync_patterns[pattern_input]
                logger.info(f"{pattern_input} sync pattern set")
            else:
                # Fallback to generic
                self.current_sync_pattern = self.sync_patterns['GENERIC']
                logger.warning(f"Unknown pattern input, using GENERIC")
                
        except Exception as e:
            logger.error(f"Error setting sync pattern: {e}")
            self.current_sync_pattern = self.sync_patterns['GENERIC']
    
    def detect_bursts(self, signal: np.ndarray, method: str = 'correlation') -> List[BurstInfo]:
        """
        Main burst detection method.
        
        Args:
            signal: Input IQ signal
            method: Detection method ('energy', 'correlation', 'combined')
            
        Returns:
            List of detected bursts
        """
        try:
            if method == 'energy':
                return self.detect_bursts_energy(signal)
            elif method == 'correlation':
                return self.detect_bursts_correlation(signal)
            elif method == 'combined':
                # Use both methods and combine results
                energy_bursts = self.detect_bursts_energy(signal)
                corr_bursts = self.detect_bursts_correlation(signal)
                
                # Merge results (prefer correlation results)
                return corr_bursts if len(corr_bursts) > 0 else energy_bursts
            else:
                logger.warning(f"Unknown detection method {method}, using correlation")
                return self.detect_bursts_correlation(signal)
                
        except Exception as e:
            logger.error(f"Burst detection error: {e}")
            return []
    
    def analyze_timing(self, bursts: List[BurstInfo]) -> Optional[Dict[str, Any]]:
        """
        Analyze timing characteristics of detected bursts.
        
        Args:
            bursts: List of detected bursts
            
        Returns:
            Timing analysis results
        """
        try:
            if len(bursts) < 2:
                return None
            
            # Calculate intervals between bursts
            intervals = []
            for i in range(1, len(bursts)):
                interval = bursts[i].start_sample - bursts[i-1].start_sample
                intervals.append(interval)
            
            if not intervals:
                return None
            
            # Statistical analysis
            avg_interval = np.mean(intervals)
            interval_std = np.std(intervals)
            
            # Frame period estimation
            frame_period = avg_interval  # Simple estimation
            
            # Convert to time units
            avg_interval_sec = avg_interval / self.sample_rate
            interval_std_sec = interval_std / self.sample_rate
            frame_period_sec = frame_period / self.sample_rate
            
            return {
                'avg_interval': avg_interval,
                'avg_interval_sec': avg_interval_sec,
                'interval_std': interval_std,
                'interval_std_sec': interval_std_sec,
                'frame_period': frame_period,
                'frame_period_sec': frame_period_sec,
                'num_intervals': len(intervals),
                'intervals': intervals
            }
            
        except Exception as e:
            logger.error(f"Timing analysis error: {e}")
            return None
    
    def extract_burst_data(self, signal: np.ndarray, burst: BurstInfo) -> Optional[np.ndarray]:
        """
        Extract data from a specific burst.
        
        Args:
            signal: Original signal
            burst: Burst information
            
        Returns:
            Extracted burst data
        """
        try:
            if burst.start_sample < 0 or burst.end_sample >= len(signal):
                return None
                
            return signal[burst.start_sample:burst.end_sample]
            
        except Exception as e:
            logger.error(f"Burst extraction error: {e}")
            return None
        """
        Estimate noise variance from signal.
        
        Args:
            signal: Input signal
            method: Estimation method ('percentile', 'minimum', 'adaptive')
            
        Returns:
            Estimated noise variance
        """
        try:
            signal_power = np.abs(signal) ** 2
            
            if method == 'percentile':
                # Use lower percentile as noise estimate
                noise_var = np.percentile(signal_power, 10)
            elif method == 'minimum':
                # Use minimum power regions
                noise_var = np.mean(np.sort(signal_power)[:len(signal_power)//10])
            elif method == 'adaptive':
                # Adaptive threshold based on signal statistics
                median_power = np.median(signal_power)
                mad = np.median(np.abs(signal_power - median_power))
                noise_var = median_power - 2 * mad
            else:
                noise_var = np.var(signal_power) * 0.1  # Fallback
            
            self.noise_variance = max(noise_var, 1e-10)  # Avoid division by zero
            return self.noise_variance
            
        except Exception as e:
            logger.warning(f"Noise estimation error: {e}")
            self.noise_variance = 1.0
            return self.noise_variance
    
    def detect_bursts_energy(self, signal: np.ndarray, 
                           p_fa: float = 1e-6) -> List[Tuple[int, int]]:
        """
        Detect bursts using energy detection method.
        
        Args:
            signal: Input IQ signal
            p_fa: Probability of false alarm
            
        Returns:
            List of (start_sample, end_sample) tuples
        """
        try:
            if not SDR_AVAILABLE:
                return self._fallback_energy_detection(signal)
            
            # Estimate noise variance
            self.estimate_noise_variance(signal)
            
            # Calculate energy
            energy = np.abs(signal) ** 2
            
            # Smooth energy with moving average
            window_size = max(10, int(0.1e-3 * self.sample_rate))  # 0.1ms window
            if len(energy) > window_size:
                energy_smooth = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
            else:
                energy_smooth = energy
            
            # Calculate detection threshold using sdr._detection
            N_samples = len(energy_smooth)
            
            # Use EnergyDetector to calculate threshold
            if hasattr(sdr, 'EnergyDetector'):
                threshold = sdr.EnergyDetector.threshold(
                    N=N_samples, 
                    p_fa=p_fa, 
                    sigma2=self.noise_variance,
                    complex=True
                )
            else:
                # Fallback calculation
                import scipy.stats
                threshold = self.noise_variance * scipy.stats.chi2.isf(p_fa, 2)
            
            # Apply threshold detection
            detections = energy_smooth > threshold
            
            # Find burst boundaries
            bursts = self._find_burst_boundaries(detections)
            
            # Filter bursts by length
            valid_bursts = []
            for start, end in bursts:
                duration = end - start
                if self.min_burst_length <= duration <= self.max_burst_length:
                    valid_bursts.append((start, end))
            
            logger.debug(f"Energy detection found {len(valid_bursts)} valid bursts")
            return valid_bursts
            
        except Exception as e:
            logger.error(f"Energy burst detection error: {e}")
            return self._fallback_energy_detection(signal)
    
    def _fallback_energy_detection(self, signal: np.ndarray) -> List[Tuple[int, int]]:
        """Fallback energy detection without sdr library."""
        try:
            energy = np.abs(signal) ** 2
            threshold = np.mean(energy) + 3 * np.std(energy)
            detections = energy > threshold
            return self._find_burst_boundaries(detections)
        except Exception as e:
            logger.error(f"Fallback energy detection error: {e}")
            return []
    
    def detect_bursts_correlation(self, signal: np.ndarray) -> List[BurstInfo]:
        """
        Detect bursts using correlation with sync pattern.
        
        Args:
            signal: Input IQ signal
            
        Returns:
            List of BurstInfo objects
        """
        try:
            if self.current_sync_pattern is None:
                self.set_sync_pattern('GENERIC')
            
            if not SDR_AVAILABLE:
                return self._fallback_correlation_detection(signal)
            
            # Cross-correlation with sync pattern
            correlation = np.correlate(signal, self.current_sync_pattern, mode='full')
            correlation_mag = np.abs(correlation)
            
            # Normalize correlation
            pattern_energy = np.sum(np.abs(self.current_sync_pattern) ** 2)
            if pattern_energy > 0:
                correlation_norm = correlation_mag / np.sqrt(pattern_energy)
            else:
                correlation_norm = correlation_mag
            
            # Find correlation peaks
            threshold = self.correlation_threshold * np.max(correlation_norm)
            
            # Use sdr detection if available
            try:
                if hasattr(sdr, 'ReplicaCorrelator'):
                    # Use replica correlator for better detection
                    p_fa = 1e-6
                    # Calculate ENR (Energy-to-Noise Ratio)
                    signal_energy = np.mean(np.abs(signal) ** 2)
                    enr_linear = signal_energy / self.noise_variance
                    enr_db = 10 * np.log10(enr_linear) if enr_linear > 0 else -50
                    
                    # Calculate detection probability
                    p_d = sdr.ReplicaCorrelator.p_d(enr_db, p_fa, complex=True)
                    
                    # Adjust threshold based on theoretical performance
                    if p_d > 0.5:  # Good detection probability
                        threshold *= 0.8  # Lower threshold
                    else:
                        threshold *= 1.2  # Higher threshold
            except Exception as e:
                logger.debug(f"Advanced correlation detection failed: {e}")
            
            # Find peaks above threshold
            peaks = []
            for i in range(1, len(correlation_norm) - 1):
                if (correlation_norm[i] > threshold and 
                    correlation_norm[i] > correlation_norm[i-1] and 
                    correlation_norm[i] > correlation_norm[i+1]):
                    peaks.append(i)
            
            # Convert correlation peaks to burst information
            bursts = []
            for peak_idx in peaks:
                # Adjust index for correlation offset
                actual_idx = peak_idx - len(self.current_sync_pattern) + 1
                if actual_idx < 0:
                    continue
                
                # Estimate burst boundaries
                start_idx = max(0, actual_idx)
                end_idx = min(len(signal), start_idx + (self.expected_burst_length or 1000))
                
                # Calculate burst metrics
                burst_signal = signal[start_idx:end_idx]
                energy = np.sum(np.abs(burst_signal) ** 2)
                snr_est = 10 * np.log10(energy / (self.noise_variance * len(burst_signal))) if self.noise_variance > 0 else 0
                correlation_val = correlation_norm[peak_idx]
                confidence = min(1.0, correlation_val / self.correlation_threshold)
                
                burst_info = BurstInfo(
                    start_sample=start_idx,
                    end_sample=end_idx,
                    duration_samples=end_idx - start_idx,
                    energy=energy,
                    snr_estimate=snr_est,
                    sync_correlation=correlation_val,
                    confidence=confidence
                )
                bursts.append(burst_info)
            
            logger.debug(f"Correlation detection found {len(bursts)} bursts")
            return bursts
            
        except Exception as e:
            logger.error(f"Correlation burst detection error: {e}")
            return self._fallback_correlation_detection(signal)
    
    def _fallback_correlation_detection(self, signal: np.ndarray) -> List[BurstInfo]:
        """Fallback correlation detection without sdr library."""
        try:
            if self.current_sync_pattern is None:
                return []
            
            correlation = np.correlate(signal, self.current_sync_pattern, mode='full')
            correlation_mag = np.abs(correlation)
            threshold = 0.7 * np.max(correlation_mag)
            
            bursts = []
            for i in range(1, len(correlation_mag) - 1):
                if (correlation_mag[i] > threshold and 
                    correlation_mag[i] > correlation_mag[i-1] and 
                    correlation_mag[i] > correlation_mag[i+1]):
                    
                    actual_idx = max(0, i - len(self.current_sync_pattern) + 1)
                    end_idx = min(len(signal), actual_idx + 1000)
                    
                    burst_info = BurstInfo(
                        start_sample=actual_idx,
                        end_sample=end_idx,
                        duration_samples=end_idx - actual_idx,
                        energy=float(np.sum(np.abs(signal[actual_idx:end_idx]) ** 2)),
                        snr_estimate=10.0,
                        sync_correlation=float(correlation_mag[i]),
                        confidence=0.7
                    )
                    bursts.append(burst_info)
            
            return bursts
            
        except Exception as e:
            logger.error(f"Fallback correlation detection error: {e}")
            return []
    
    def _find_burst_boundaries(self, detections: np.ndarray) -> List[Tuple[int, int]]:
        """Find burst start and end boundaries from detection array."""
        bursts = []
        start_idx = None
        
        for i, detected in enumerate(detections):
            if detected and start_idx is None:
                start_idx = i
            elif not detected and start_idx is not None:
                bursts.append((start_idx, i))
                start_idx = None
        
        # Handle case where signal ends during burst
        if start_idx is not None:
            bursts.append((start_idx, len(detections)))
        
        return bursts
    
    def analyze_tdma_structure(self, bursts: List[BurstInfo]) -> Optional[TDMAFrameInfo]:
        """
        Analyze TDMA frame structure from detected bursts.
        
        Args:
            bursts: List of detected bursts
            
        Returns:
            TDMA frame information or None if insufficient data
        """
        try:
            if len(bursts) < 3:
                logger.warning("Insufficient bursts for TDMA analysis")
                return None
            
            # Calculate burst timing statistics
            start_times = [burst.start_sample for burst in bursts]
            durations = [burst.duration_samples for burst in bursts]
            
            # Estimate frame period
            if len(start_times) >= 2:
                intervals = np.diff(start_times)
                # Find the most common interval (frame period)
                frame_period_samples = int(np.median(intervals))
                frame_period_seconds = frame_period_samples / self.sample_rate
            else:
                frame_period_samples = 0
                frame_period_seconds = 0
            
            # Estimate slot duration
            slot_duration_samples = int(np.median(durations))
            slot_duration_seconds = slot_duration_samples / self.sample_rate
            
            # Estimate guard time
            if frame_period_samples > slot_duration_samples:
                guard_time_samples = frame_period_samples - slot_duration_samples
            else:
                guard_time_samples = 0
            
            # Calculate duty cycle
            duty_cycle = slot_duration_samples / frame_period_samples if frame_period_samples > 0 else 0
            
            # Estimate slots per frame (simplified)
            slots_per_frame = max(1, int(1 / duty_cycle)) if duty_cycle > 0 else 1
            
            frame_info = TDMAFrameInfo(
                frame_period_samples=frame_period_samples,
                frame_period_seconds=frame_period_seconds,
                slot_duration_samples=slot_duration_samples,
                slot_duration_seconds=slot_duration_seconds,
                guard_time_samples=guard_time_samples,
                duty_cycle=duty_cycle,
                slots_per_frame=slots_per_frame
            )
            
            logger.info(f"TDMA analysis: {slots_per_frame} slots/frame, "
                       f"{frame_period_seconds*1000:.2f}ms period, "
                       f"{duty_cycle*100:.1f}% duty cycle")
            
            return frame_info
            
        except Exception as e:
            logger.error(f"TDMA structure analysis error: {e}")
            return None
    
    def detect_and_analyze(self, signal: np.ndarray, 
                          pattern_type: str = 'GENERIC') -> Dict[str, Any]:
        """
        Complete TDMA detection and analysis.
        
        Args:
            signal: Input IQ signal
            pattern_type: Sync pattern type ('GSM', 'DECT', 'GENERIC')
            
        Returns:
            Complete analysis results
        """
        try:
            # Set sync pattern
            self.set_sync_pattern(pattern_type)
            
            # Detect bursts using both methods
            energy_bursts = self.detect_bursts_energy(signal)
            correlation_bursts = self.detect_bursts_correlation(signal)
            
            # Analyze TDMA structure
            tdma_info = self.analyze_tdma_structure(correlation_bursts)
            
            # Calculate detection statistics
            total_samples = len(signal)
            total_burst_samples = sum([b.duration_samples for b in correlation_bursts])
            
            results = {
                'detection_method': 'sdr._detection',
                'pattern_type': pattern_type,
                'signal_length_samples': total_samples,
                'signal_duration_seconds': total_samples / self.sample_rate,
                'energy_bursts': len(energy_bursts),
                'correlation_bursts': len(correlation_bursts),
                'burst_details': correlation_bursts,
                'tdma_structure': tdma_info,
                'detection_statistics': {
                    'total_burst_time': total_burst_samples / self.sample_rate,
                    'burst_duty_cycle': total_burst_samples / total_samples,
                    'avg_snr_estimate': np.mean([b.snr_estimate for b in correlation_bursts]) if correlation_bursts else 0,
                    'avg_confidence': np.mean([b.confidence for b in correlation_bursts]) if correlation_bursts else 0
                },
                'noise_variance': self.noise_variance,
                'sdr_library_available': SDR_AVAILABLE
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Complete TDMA detection error: {e}")
            return {
                'detection_method': 'sdr._detection',
                'error': str(e),
                'sdr_library_available': SDR_AVAILABLE
            }


def create_tdma_detector(sample_rate: float) -> TDMABurstDetector:
    """Factory function to create TDMA burst detector."""
    return TDMABurstDetector(sample_rate)