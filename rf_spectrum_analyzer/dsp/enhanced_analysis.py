"""
Enhanced Signal Analysis Module
Integrates sdrconnect's analyze_signal functionality into RF Spectrum Analyzer
"""

import numpy as np
import logging
import os
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from sdrconnect import analyze_signal
    SDRCONNECT_AVAILABLE = True
except ImportError:
    SDRCONNECT_AVAILABLE = False

try:
    import pyfftw
    PYFFTW_AVAILABLE = True
except ImportError:
    PYFFTW_AVAILABLE = False

from rf_spectrum_analyzer.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnhancedAnalysisResult:
    """Enhanced analysis result combining existing and sdrconnect analysis."""
    
    # Basic analysis (existing)
    power_spectrum: np.ndarray
    frequency_axis: np.ndarray
    peak_frequency: float
    bandwidth: float
    snr_estimate: float
    
    # Enhanced analysis from sdrconnect (if available)
    spectrogram: Optional[np.ndarray] = None
    mean_psd: Optional[np.ndarray] = None
    time_axis: Optional[np.ndarray] = None
    
    # Advanced metrics from sdrconnect
    rms_power: Optional[float] = None
    peak_power: Optional[float] = None
    crest_factor: Optional[float] = None
    dc_offset_i: Optional[float] = None
    dc_offset_q: Optional[float] = None
    zero_crossings: Optional[int] = None
    noise_floor: Optional[float] = None
    sinad: Optional[float] = None
    occupied_bandwidth: Optional[float] = None
    frequency_drift: Optional[float] = None
    spur_frequencies: Optional[list] = None
    
    # Metadata
    analysis_method: str = "basic"
    sdrconnect_available: bool = False


class EnhancedSignalAnalysis:
    """Enhanced signal analysis combining existing capabilities with sdrconnect."""
    
    def __init__(self, sample_rate: float = 1e6, fft_size: int = 1024):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.sdrconnect_available = SDRCONNECT_AVAILABLE
        
        # Setup optimized FFT processing
        self._setup_optimized_fft()
        
        # Performance tracking
        self._fft_computation_times = []
        self._last_analysis_time = 0
        
        if not self.sdrconnect_available:
            logger.warning("sdrconnect not available. Using basic analysis only.")
            
        logger.info(f"Enhanced Signal Analysis initialized with FFT method: {self.fft_method}")
    
    def _setup_optimized_fft(self):
        """Setup optimized FFT computation with pyFFTW."""
        # Setup optimized FFTW if available
        if PYFFTW_AVAILABLE:
            try:
                # Configure FFTW for optimal performance
                pyfftw.config.NUM_THREADS = min(2, os.cpu_count() or 1)  # Limit threads for analysis
                pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
                
                # Create cache-aligned arrays for optimal performance
                self.fft_input = pyfftw.empty_aligned(self.fft_size, dtype='complex64', 
                                                    n=pyfftw.simd_alignment)
                self.fft_output = pyfftw.empty_aligned(self.fft_size, dtype='complex64',
                                                     n=pyfftw.simd_alignment)
                
                # Pre-allocate working arrays
                self._power_spectrum = np.empty(self.fft_size, dtype=np.float32)
                self._power_db = np.empty(self.fft_size, dtype=np.float32)
                
                # Create optimized FFTW object
                self.fft_object = pyfftw.FFTW(self.fft_input, 
                                            self.fft_output, 
                                            direction='FFTW_FORWARD', 
                                            flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'),
                                            threads=pyfftw.config.NUM_THREADS)
                
                self.fft_method = 'pyfftw_optimized'
                logger.info(f"Using optimized FFTW with {pyfftw.config.NUM_THREADS} threads for enhanced analysis")
                
            except Exception as e:
                logger.warning(f"Failed to setup optimized FFTW: {e}, falling back to numpy")
                self.fft_method = 'numpy'
        else:
            self.fft_method = 'numpy'
            logger.info("Using numpy for FFT computation in enhanced analysis")
    
    def analyze_iq_data(self, iq_data: np.ndarray) -> EnhancedAnalysisResult:
        """
        Perform enhanced analysis on IQ data.
        
        Args:
            iq_data: Complex IQ samples
            
        Returns:
            EnhancedAnalysisResult with comprehensive analysis
        """
        if len(iq_data) == 0:
            return self._create_empty_result()
        
        # Basic analysis (always available)
        basic_result = self._basic_analysis(iq_data)
        
        # Enhanced analysis with sdrconnect (if available)
        if self.sdrconnect_available and len(iq_data) >= self.fft_size:
            try:
                enhanced_result = self._enhanced_analysis(iq_data)
                return self._combine_results(basic_result, enhanced_result)
            except Exception as e:
                logger.error(f"Enhanced analysis failed: {e}")
                return basic_result
        else:
            return basic_result
    
    def _basic_analysis(self, iq_data: np.ndarray) -> EnhancedAnalysisResult:
        """Perform optimized basic analysis using pyFFTW."""
        start_time = time.time()
        
        # Extract samples for FFT (ensure complex64 for optimal performance)
        samples = iq_data[:self.fft_size].astype(np.complex64)
        
        # Compute FFT with optimized path
        if self.fft_method == 'pyfftw_optimized':
            # Optimized FFTW path
            self.fft_input[:] = samples
            self.fft_object()
            
            # Compute power spectrum in-place for better memory efficiency
            np.abs(self.fft_output, out=self._power_spectrum)
            np.square(self._power_spectrum, out=self._power_spectrum)
            
            # Convert to dB efficiently
            np.maximum(self._power_spectrum, 1e-12, out=self._power_spectrum)  # Prevent log(0)
            np.log10(self._power_spectrum, out=self._power_db)
            self._power_db *= 20.0  # 20*log10 for power spectrum
            
            # FFT shift to center DC
            power_spectrum = np.fft.fftshift(self._power_db)
            
        else:
            # NumPy fallback path
            fft_data = np.fft.fft(samples)
            power_spectrum = 20 * np.log10(np.abs(fft_data) + 1e-12)
            power_spectrum = np.fft.fftshift(power_spectrum)
        
        # Frequency axis
        frequency_axis = np.linspace(-self.sample_rate/2, self.sample_rate/2, self.fft_size)
        
        # Peak frequency
        peak_idx = np.argmax(power_spectrum)
        peak_frequency = frequency_axis[peak_idx]
        
        # Basic bandwidth estimation (3dB bandwidth)
        peak_power = power_spectrum[peak_idx]
        bandwidth_indices = np.where(power_spectrum >= (peak_power - 3))[0]
        if len(bandwidth_indices) > 1:
            bandwidth = (bandwidth_indices[-1] - bandwidth_indices[0]) * (self.sample_rate / self.fft_size)
        else:
            bandwidth = self.sample_rate / self.fft_size
        
        # Basic SNR estimation
        signal_power = np.mean(power_spectrum[bandwidth_indices])
        noise_power = np.median(power_spectrum)  # Median as noise floor estimate
        snr_estimate = signal_power - noise_power
        
        # Track performance
        computation_time = time.time() - start_time
        self._fft_computation_times.append(computation_time)
        if len(self._fft_computation_times) > 100:  # Keep last 100 measurements
            self._fft_computation_times.pop(0)
        
        return EnhancedAnalysisResult(
            power_spectrum=power_spectrum,
            frequency_axis=frequency_axis,
            peak_frequency=peak_frequency,
            bandwidth=bandwidth,
            snr_estimate=snr_estimate,
            analysis_method=f"basic_{self.fft_method}",
            sdrconnect_available=False
        )
    
    def _enhanced_analysis(self, iq_data: np.ndarray) -> Dict[str, Any]:
        """Perform enhanced analysis using sdrconnect."""
        
        # Use sdrconnect's analyze_signal - returns 5 values
        spectrogram, mean_psd, freq_axis, time_axis, metrics_dict = analyze_signal(
            iq_data, self.sample_rate, self.fft_size
        )
        
        # Calculate additional metrics
        amplitude = np.abs(iq_data)
        
        # Extract metrics from sdrconnect's analysis
        rms_power = metrics_dict.get('rms_power', np.sqrt(np.mean(amplitude**2)))
        peak_power = metrics_dict.get('peak_power', np.max(amplitude))
        crest_factor = metrics_dict.get('crest_factor', peak_power / rms_power if rms_power > 0 else 0)
        dc_offset_i = metrics_dict.get('dc_offset_i', np.mean(np.real(iq_data)))
        dc_offset_q = metrics_dict.get('dc_offset_q', np.mean(np.imag(iq_data)))
        zero_crossings = metrics_dict.get('zero_crossings', ((np.real(iq_data[:-1]) * np.real(iq_data[1:])) < 0).sum())
        noise_floor = metrics_dict.get('noise_floor', np.median(mean_psd))
        sinad = metrics_dict.get('sinad', 10 * np.log10(
            np.mean(amplitude**2) / (np.mean((amplitude - np.mean(amplitude))**2) + 1e-12)
        ))
        occupied_bandwidth = metrics_dict.get('occupied_bandwidth_khz', 0.0)
        frequency_drift = metrics_dict.get('frequency_drift_hz', 0.0)
        spur_frequencies = metrics_dict.get('spur_frequencies_mhz', [])
        
        return {
            'spectrogram': spectrogram,
            'mean_psd': mean_psd,
            'freq_axis': freq_axis,
            'time_axis': time_axis,
            'rms_power': float(rms_power),
            'peak_power': float(peak_power),
            'crest_factor': float(crest_factor),
            'dc_offset_i': float(dc_offset_i),
            'dc_offset_q': float(dc_offset_q),
            'zero_crossings': int(zero_crossings),
            'noise_floor': float(noise_floor),
            'sinad': float(sinad),
            'occupied_bandwidth': float(occupied_bandwidth),
            'frequency_drift': float(frequency_drift),
            'spur_frequencies': spur_frequencies
        }
    
    def _combine_results(self, basic_result: EnhancedAnalysisResult, 
                        enhanced_data: Dict[str, Any]) -> EnhancedAnalysisResult:
        """Combine basic and enhanced analysis results."""
        
        # Update basic result with enhanced data
        basic_result.spectrogram = enhanced_data['spectrogram']
        basic_result.mean_psd = enhanced_data['mean_psd']
        basic_result.time_axis = enhanced_data['time_axis']
        basic_result.rms_power = enhanced_data['rms_power']
        basic_result.peak_power = enhanced_data['peak_power']
        basic_result.crest_factor = enhanced_data['crest_factor']
        basic_result.dc_offset_i = enhanced_data['dc_offset_i']
        basic_result.dc_offset_q = enhanced_data['dc_offset_q']
        basic_result.zero_crossings = enhanced_data['zero_crossings']
        basic_result.noise_floor = enhanced_data['noise_floor']
        basic_result.sinad = enhanced_data['sinad']
        basic_result.occupied_bandwidth = enhanced_data['occupied_bandwidth']
        basic_result.frequency_drift = enhanced_data['frequency_drift']
        basic_result.spur_frequencies = enhanced_data['spur_frequencies']
        
        # Update metadata
        basic_result.analysis_method = "enhanced"
        basic_result.sdrconnect_available = True
        
        # Use enhanced frequency axis if available
        if enhanced_data['freq_axis'] is not None:
            basic_result.frequency_axis = enhanced_data['freq_axis'] * 1e6  # Convert MHz to Hz
        
        # Use enhanced PSD for better peak detection
        if enhanced_data['mean_psd'] is not None:
            peak_idx = np.argmax(enhanced_data['mean_psd'])
            basic_result.peak_frequency = enhanced_data['freq_axis'][peak_idx] * 1e6  # Convert MHz to Hz
        
        return basic_result
    
    def _create_empty_result(self) -> EnhancedAnalysisResult:
        """Create empty result for invalid input."""
        return EnhancedAnalysisResult(
            power_spectrum=np.array([]),
            frequency_axis=np.array([]),
            peak_frequency=0.0,
            bandwidth=0.0,
            snr_estimate=0.0,
            analysis_method="empty",
            sdrconnect_available=self.sdrconnect_available
        )
    
    def get_analysis_info(self) -> Dict[str, Any]:
        """Get information about analysis capabilities."""
        return {
            'sdrconnect_available': self.sdrconnect_available,
            'sample_rate': self.sample_rate,
            'fft_size': self.fft_size,
            'fft_method': self.fft_method,
            'pyfftw_available': PYFFTW_AVAILABLE,
            'enhanced_features': [
                'spectrogram',
                'advanced_metrics',
                'spur_detection',
                'occupied_bandwidth',
                'frequency_drift',
                'crest_factor',
                'dc_offset_analysis'
            ] if self.sdrconnect_available else ['basic_spectrum']
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for FFT operations."""
        if not self._fft_computation_times:
            return {
                'fft_method': self.fft_method,
                'average_computation_time_ms': 0.0,
                'measurements_count': 0
            }
        
        avg_time = np.mean(self._fft_computation_times) * 1000  # Convert to ms
        return {
            'fft_method': self.fft_method,
            'average_computation_time_ms': float(avg_time),
            'min_computation_time_ms': float(np.min(self._fft_computation_times) * 1000),
            'max_computation_time_ms': float(np.max(self._fft_computation_times) * 1000),
            'measurements_count': len(self._fft_computation_times),
            'pyfftw_available': PYFFTW_AVAILABLE,
            'threads_used': getattr(pyfftw.config, 'NUM_THREADS', 1) if PYFFTW_AVAILABLE else 1
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking statistics."""
        self._fft_computation_times = []
        logger.info("Enhanced analysis performance statistics reset")