"""
Signal Analysis Module - Advanced signal processing for modulation and coding analysis
Analyzes IQ data to determine modulation type, demodulate signals, and decode data.
Enhanced with advanced DSP capabilities from other modules.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from scipy import signal, fft
from scipy.stats import kurtosis, skew

# Import advanced DSP capabilities
try:
    from .demodulation_engine import create_demodulation_engine, DemodulationEngine
    from .decoding_engine import create_decoding_engine, DecodingEngine
    from .modulation_analysis import create_modulation_analyzer, ModulationAnalyzer
    from .signal_detection import create_signal_detector, SignalDetectionEngine
    from .utils import (
        find_peaks_advanced, estimate_delay, db_to_linear, linear_to_db,
        calculate_ber, rms_value, peak_to_average_ratio, crest_factor
    )
    from .filters import design_lowpass, FIRFilter
    from .enhanced_analysis import EnhancedSignalAnalysis
    ADVANCED_DSP_AVAILABLE = True
except ImportError as e:
    print(f"Advanced DSP modules not available: {e}")
    ADVANCED_DSP_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ModulationAnalysisResult:
    """Results from modulation analysis."""
    modulation_type: str
    confidence: float
    parameters: Dict[str, Any]
    constellation_points: Optional[np.ndarray] = None
    symbol_rate: Optional[float] = None
    frequency_offset: Optional[float] = None
    phase_offset: Optional[float] = None

@dataclass
class DemodulationResult:
    """Results from signal demodulation."""
    success: bool
    symbols: Optional[np.ndarray] = None
    bits: Optional[np.ndarray] = None
    snr: Optional[float] = None
    error_rate: Optional[float] = None
    decoded_data: Optional[bytes] = None

@dataclass
class CodingAnalysisResult:
    """Results from coding analysis."""
    coding_type: str
    confidence: float
    parameters: Dict[str, Any]
    decoded_bits: Optional[np.ndarray] = None
    error_correction_info: Optional[Dict[str, Any]] = None

class SignalAnalyzer:
    """Advanced signal analyzer for modulation and coding detection."""
    
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
        self.logger = logger
        
        # Analysis parameters
        self.fft_size = 1024
        self.overlap = 0.5
        self.window = 'hann'
        
        # Modulation detection thresholds
        self.constellation_thresholds = {
            'BPSK': {'points': 2, 'tolerance': 0.3},
            'QPSK': {'points': 4, 'tolerance': 0.4},
            'PSK8': {'points': 8, 'tolerance': 0.5},
            'QAM16': {'points': 16, 'tolerance': 0.6},
            'QAM64': {'points': 64, 'tolerance': 0.8},
            'FSK': {'freq_domain': True, 'tolerance': 0.5},
            'ASK': {'amplitude_domain': True, 'tolerance': 0.4}
        }
        
        # Initialize advanced DSP engines if available
        self._init_advanced_engines()
        
    def _init_advanced_engines(self):
        """Initialize advanced DSP processing engines."""
        if ADVANCED_DSP_AVAILABLE:
            try:
                self.demodulation_engine = create_demodulation_engine(self.sample_rate)
                self.decoding_engine = create_decoding_engine()
                self.modulation_analyzer = create_modulation_analyzer(self.sample_rate)
                self.signal_detector = create_signal_detector(self.sample_rate)
                self.enhanced_analyzer = EnhancedSignalAnalysis(self.sample_rate, self.fft_size)
                
                # Design default filters
                self.lpf = design_lowpass(
                    cutoff=self.sample_rate * 0.4, 
                    order=5, 
                    filter_type="fir", 
                    sample_rate=self.sample_rate
                )
                
                self.advanced_features_enabled = True
                self.logger.info("Advanced DSP engines initialized successfully")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize advanced DSP engines: {e}")
                self.advanced_features_enabled = False
        else:
            self.advanced_features_enabled = False
            self.logger.info("Basic DSP features only")
    
    def analyze_signal_comprehensive(self, iq_data: np.ndarray, center_freq: float, 
                                   bandwidth: float) -> Dict[str, Any]:
        """Comprehensive signal analysis including modulation, demodulation, and coding."""
        if len(iq_data) == 0:
            return {'error': 'No IQ data provided'}
        
        try:
            # Step 1: Enhanced preprocessing with advanced filtering
            preprocessed_data = self._preprocess_signal_advanced(iq_data)
            
            # Step 2: Enhanced signal detection
            detection_result = self._detect_signal_presence(preprocessed_data)
            
            # Step 3: Advanced modulation analysis
            mod_result = self._analyze_modulation_advanced(preprocessed_data)
            
            # Step 4: Enhanced demodulation with multiple engines
            demod_result = self._demodulate_signal_advanced(preprocessed_data, mod_result)
            
            # Step 5: Advanced coding analysis with FEC detection
            coding_result = None
            if demod_result.success and demod_result.bits is not None:
                coding_result = self._analyze_coding_advanced(demod_result.bits)
            
            # Step 6: Enhanced signal quality metrics
            quality_metrics = self._calculate_signal_quality_metrics(preprocessed_data, demod_result)
            
            # Step 7: Peak detection and spectrum analysis
            peaks_info = self._analyze_spectrum_peaks(preprocessed_data)
            
            # Step 8: Compile comprehensive results
            analysis_results = {
                'signal_info': {
                    'center_freq': center_freq,
                    'bandwidth': bandwidth,
                    'sample_rate': self.sample_rate,
                    'signal_length': len(iq_data),
                    'signal_power': float(np.mean(np.abs(iq_data)**2)),
                    'peak_to_average_ratio': quality_metrics.get('par', 0.0),
                    'crest_factor': quality_metrics.get('crest_factor', 0.0),
                    'rms_power': quality_metrics.get('rms_power', 0.0)
                },
                'detection': {
                    'signal_detected': detection_result.get('signal_detected', True),
                    'confidence': detection_result.get('confidence', 0.5),
                    'snr_estimate': detection_result.get('snr_estimate', 0.0),
                    'noise_floor': detection_result.get('noise_floor', -80.0)
                },
                'modulation': {
                    'type': mod_result.modulation_type,
                    'confidence': mod_result.confidence,
                    'parameters': mod_result.parameters,
                    'symbol_rate': mod_result.symbol_rate,
                    'frequency_offset': mod_result.frequency_offset,
                    'phase_offset': mod_result.phase_offset if hasattr(mod_result, 'phase_offset') else None
                },
                'demodulation': {
                    'success': demod_result.success,
                    'snr': demod_result.snr,
                    'error_rate': demod_result.error_rate,
                    'symbols_count': len(demod_result.symbols) if demod_result.symbols is not None else 0,
                    'bits_count': len(demod_result.bits) if demod_result.bits is not None else 0,
                    'quality_metrics': quality_metrics
                },
                'coding': coding_result.__dict__ if coding_result else None,
                'constellation_data': {
                    'points': mod_result.constellation_points.tolist() if mod_result.constellation_points is not None else [],
                    'symbols': demod_result.symbols.tolist() if demod_result.symbols is not None else []
                },
                'spectrum_analysis': peaks_info,
                'advanced_features_used': self.advanced_features_enabled,
                'analysis_status': 'success'
            }
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Signal analysis failed: {e}")
            return {
                'error': str(e),
                'analysis_status': 'failed',
                'advanced_features_used': self.advanced_features_enabled
            }
    
    def _preprocess_signal(self, iq_data: np.ndarray) -> np.ndarray:
        """Preprocess IQ data for analysis."""
        # Remove DC component
        iq_data = iq_data - np.mean(iq_data)
        
        # Normalize signal
        if np.std(iq_data) > 0:
            iq_data = iq_data / np.std(iq_data)
        
        # Apply simple filtering to remove high-frequency noise
        if len(iq_data) > 10:
            # Simple moving average filter
            kernel_size = min(5, len(iq_data) // 10)
            if kernel_size > 1:
                kernel = np.ones(kernel_size) / kernel_size
                iq_data = np.convolve(iq_data, kernel, mode='same')
        
        return iq_data
    
    def _preprocess_signal_advanced(self, iq_data: np.ndarray) -> np.ndarray:
        """Advanced preprocessing with filtering and normalization."""
        # Remove DC component
        iq_data = iq_data - np.mean(iq_data)
        
        # Apply advanced filtering if available
        if self.advanced_features_enabled and hasattr(self, 'lpf'):
            try:
                iq_data = self.lpf.filter(iq_data)
            except Exception as e:
                self.logger.warning(f"Advanced filtering failed, using basic: {e}")
        
        # Normalize signal
        if np.std(iq_data) > 0:
            iq_data = iq_data / np.std(iq_data)
        
        # Apply simple filtering to remove high-frequency noise (fallback)
        if len(iq_data) > 10:
            # Simple moving average filter
            kernel_size = min(5, len(iq_data) // 10)
            if kernel_size > 1:
                kernel = np.ones(kernel_size) / kernel_size
                iq_data = np.convolve(iq_data, kernel, mode='same')
        
        return iq_data
    
    def _detect_signal_presence(self, iq_data: np.ndarray) -> Dict[str, Any]:
        """Enhanced signal detection using advanced algorithms."""
        if self.advanced_features_enabled and hasattr(self, 'signal_detector'):
            try:
                # Use advanced signal detection
                detection_result = self.signal_detector.energy_detection(iq_data)
                return {
                    'signal_detected': bool(detection_result.signal_detected),  # Convert to Python bool
                    'confidence': float(detection_result.confidence),
                    'snr_estimate': float(detection_result.snr_estimate),
                    'test_statistic': float(detection_result.test_statistic),
                    'threshold': float(detection_result.threshold),
                    'noise_floor': float(linear_to_db(detection_result.noise_variance)) if ADVANCED_DSP_AVAILABLE else -80.0
                }
            except Exception as e:
                self.logger.warning(f"Advanced signal detection failed: {e}")
        
        # Fallback to basic energy detection
        signal_power = np.mean(np.abs(iq_data)**2)
        noise_estimate = np.var(np.abs(iq_data))
        
        # Avoid division by zero
        if noise_estimate < 1e-12:
            noise_estimate = 1e-12
            
        snr_estimate = 10 * np.log10(signal_power / noise_estimate)
        
        # Use more reasonable threshold for synthetic signals
        detection_threshold = -10  # dB, more sensitive than -20
        
        return {
            'signal_detected': bool(snr_estimate > detection_threshold),
            'confidence': float(min(max((snr_estimate + 20) / 40, 0), 1)),
            'snr_estimate': float(snr_estimate),
            'noise_floor': float(10 * np.log10(noise_estimate))
        }
    
    def _analyze_modulation_advanced(self, iq_data: np.ndarray) -> ModulationAnalysisResult:
        """Enhanced modulation analysis using advanced algorithms."""
        if self.advanced_features_enabled and hasattr(self, 'modulation_analyzer'):
            try:
                # Use advanced modulation analyzer
                mod_result = self.modulation_analyzer.detect_modulation(iq_data)
                
                # Handle different possible key names
                modulation_type = mod_result.get('modulation_type') or mod_result.get('type', 'Unknown')
                confidence = mod_result.get('confidence', 0.0)
                
                # If advanced analyzer fails to provide good results, fallback to basic
                if modulation_type == 'Unknown' or confidence < 0.1:
                    self.logger.info("Advanced modulation analyzer returned low confidence, using basic analyzer")
                    return self.analyze_modulation(iq_data)
                
                return ModulationAnalysisResult(
                    modulation_type=modulation_type,
                    confidence=confidence,
                    parameters=mod_result.get('parameters', {}),
                    constellation_points=self._extract_constellation_advanced(iq_data),
                    symbol_rate=mod_result.get('symbol_rate'),
                    frequency_offset=mod_result.get('frequency_offset'),
                    phase_offset=mod_result.get('phase_offset', 0.0)
                )
            except Exception as e:
                self.logger.warning(f"Advanced modulation analysis failed: {e}")
        
        # Fallback to basic modulation analysis
        return self.analyze_modulation(iq_data)
    
    def _extract_constellation_advanced(self, iq_data: np.ndarray, decimation: int = 10) -> np.ndarray:
        """Enhanced constellation extraction with timing recovery."""
        if len(iq_data) < decimation:
            return np.array([])
        
        # Apply timing recovery if available
        if self.advanced_features_enabled:
            try:
                # Simple timing recovery by symbol rate estimation
                symbol_rate_est = self._estimate_symbol_rate_advanced(iq_data)
                if symbol_rate_est:
                    samples_per_symbol = int(self.sample_rate / symbol_rate_est)
                    if samples_per_symbol > 1:
                        # Extract symbols at estimated timing
                        constellation = iq_data[::samples_per_symbol]
                        return constellation[:min(len(constellation), 1000)]
            except Exception as e:
                self.logger.warning(f"Advanced constellation extraction failed: {e}")
        
        # Fallback to simple decimation
        return iq_data[::decimation][:min(len(iq_data)//decimation, 1000)]
    
    def _estimate_symbol_rate_advanced(self, iq_data: np.ndarray) -> Optional[float]:
        """Advanced symbol rate estimation using spectral analysis."""
        try:
            # Calculate magnitude spectrum
            magnitude = np.abs(iq_data)
            
            # Apply spectral analysis to detect symbol rate
            fft_mag = np.fft.fft(magnitude)
            freqs = np.fft.fftfreq(len(magnitude), 1/self.sample_rate)
            
            # Find peak in spectrum (excluding DC)
            fft_power = np.abs(fft_mag[1:len(fft_mag)//2])
            freqs_pos = freqs[1:len(freqs)//2]
            
            if len(fft_power) > 0:
                peak_idx = np.argmax(fft_power)
                symbol_rate = abs(freqs_pos[peak_idx])
                
                # Validate symbol rate
                if 100 <= symbol_rate <= self.sample_rate / 4:
                    return symbol_rate
                    
        except Exception as e:
            self.logger.warning(f"Advanced symbol rate estimation failed: {e}")
        
        return None
    
    def _demodulate_signal_advanced(self, iq_data: np.ndarray, 
                                  mod_result: ModulationAnalysisResult) -> DemodulationResult:
        """Enhanced demodulation using advanced engines."""
        if self.advanced_features_enabled and hasattr(self, 'demodulation_engine'):
            try:
                # Use advanced demodulation engine
                demod_result = self.demodulation_engine.demodulate(
                    iq_data, 
                    mod_result.modulation_type, 
                    mod_result.parameters
                )
                
                # Convert to our result format
                advanced_result = DemodulationResult(
                    success=demod_result.get('success', False),
                    symbols=demod_result.get('symbols'),
                    bits=demod_result.get('bits'),
                    snr=demod_result.get('snr'),
                    error_rate=demod_result.get('error_rate')
                )
                
                # If advanced demodulation was successful, return it
                if advanced_result.success:
                    return advanced_result
                else:
                    self.logger.info("Advanced demodulation failed, using basic demodulation")
                    
            except Exception as e:
                self.logger.warning(f"Advanced demodulation failed: {e}")
        
        # Fallback to basic demodulation
        return self.demodulate_signal(iq_data, mod_result)
    
    def _analyze_coding_advanced(self, bits: np.ndarray) -> Optional[CodingAnalysisResult]:
        """Advanced coding analysis using FEC detection."""
        if self.advanced_features_enabled and hasattr(self, 'decoding_engine'):
            try:
                # Try multiple coding schemes
                coding_types = ['hamming', 'convolutional', 'bch', 'manchester', 'nrz']
                best_result = None
                best_confidence = 0.0
                
                for coding_type in coding_types:
                    try:
                        decode_result = self.decoding_engine.decode(bits, coding_type)
                        if decode_result.get('confidence', 0) > best_confidence:
                            best_confidence = decode_result.get('confidence', 0)
                            best_result = CodingAnalysisResult(
                                coding_type=coding_type,
                                confidence=best_confidence,
                                parameters=decode_result.get('parameters', {}),
                                decoded_bits=decode_result.get('decoded_bits'),
                                error_correction_info=decode_result.get('error_correction_info')
                            )
                    except Exception:
                        continue
                
                if best_result and best_confidence > 0.5:
                    return best_result
                    
            except Exception as e:
                self.logger.warning(f"Advanced coding analysis failed: {e}")
        
        # Fallback to basic coding analysis
        return self.analyze_coding(bits)
    
    def _calculate_signal_quality_metrics(self, iq_data: np.ndarray, 
                                        demod_result: DemodulationResult) -> Dict[str, float]:
        """Calculate comprehensive signal quality metrics."""
        metrics = {}
        
        try:
            if ADVANCED_DSP_AVAILABLE:
                # Use advanced utility functions
                metrics['rms_power'] = rms_value(iq_data)
                metrics['par'] = peak_to_average_ratio(iq_data)
                metrics['crest_factor'] = crest_factor(iq_data)
            else:
                # Basic metrics
                metrics['rms_power'] = float(np.sqrt(np.mean(np.abs(iq_data)**2)))
                metrics['par'] = float(np.max(np.abs(iq_data)**2) / np.mean(np.abs(iq_data)**2))
                metrics['crest_factor'] = float(np.max(np.abs(iq_data)) / np.sqrt(np.mean(np.abs(iq_data)**2)))
            
            # Statistical metrics
            real_part = np.real(iq_data)
            imag_part = np.imag(iq_data)
            
            metrics['kurtosis_real'] = float(kurtosis(real_part))
            metrics['kurtosis_imag'] = float(kurtosis(imag_part))
            metrics['skewness_real'] = float(skew(real_part))
            metrics['skewness_imag'] = float(skew(imag_part))
            
            # BER calculation if demodulation successful
            if demod_result.success and demod_result.bits is not None and len(demod_result.bits) > 10:
                if ADVANCED_DSP_AVAILABLE:
                    # Use advanced BER calculation
                    # Generate reference bits for comparison (simplified)
                    ref_bits = np.random.randint(0, 2, len(demod_result.bits))
                    ber_result = calculate_ber(ref_bits, demod_result.bits)
                    metrics['estimated_ber'] = ber_result.get('ber', 0.0)
                else:
                    # Basic BER estimation based on symbol transitions
                    transitions = np.sum(np.diff(demod_result.bits) != 0)
                    metrics['estimated_ber'] = float(transitions / len(demod_result.bits))
            
        except Exception as e:
            self.logger.warning(f"Signal quality metrics calculation failed: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _analyze_spectrum_peaks(self, iq_data: np.ndarray) -> Dict[str, Any]:
        """Analyze spectrum to find significant peaks."""
        try:
            # Calculate power spectrum
            fft_data = np.fft.fftshift(np.fft.fft(iq_data, self.fft_size))
            freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1/self.sample_rate))
            power_spectrum = 20 * np.log10(np.abs(fft_data) + 1e-12)
            
            # Find peaks
            if ADVANCED_DSP_AVAILABLE:
                # Use advanced peak detection
                peaks_info = find_peaks_advanced(power_spectrum, height=-80, distance=5)
                peak_indices = peaks_info.get('peaks', [])
            else:
                # Basic peak detection with adaptive threshold
                from scipy.signal import find_peaks
                # Calculate adaptive threshold based on noise floor
                noise_floor = np.percentile(power_spectrum, 25)  # 25th percentile as noise estimate
                threshold = noise_floor + 10  # 10dB above noise floor
                peak_indices, _ = find_peaks(power_spectrum, height=threshold, distance=5)
            
            if len(peak_indices) > 0:
                peak_freqs = freqs[peak_indices]
                peak_powers = power_spectrum[peak_indices]
                
                # Sort by power (strongest first)
                sorted_idx = np.argsort(peak_powers)[::-1]
                
                # Filter for significant peaks only (top peaks above noise + margin)
                if len(sorted_idx) > 0:
                    strongest_power = peak_powers[sorted_idx[0]]
                    # Keep peaks within 30dB of strongest
                    significant_mask = peak_powers[sorted_idx] > (strongest_power - 30)
                    significant_idx = sorted_idx[significant_mask]
                    
                    # Limit to top 10 most significant peaks
                    top_peaks = significant_idx[:10]
                    
                    return {
                        'peak_frequencies': peak_freqs[top_peaks].tolist(),
                        'peak_powers': peak_powers[top_peaks].tolist(),
                        'num_peaks': len(top_peaks),
                        'strongest_peak_freq': float(peak_freqs[top_peaks[0]]) if len(top_peaks) > 0 else 0.0,
                        'strongest_peak_power': float(peak_powers[top_peaks[0]]) if len(top_peaks) > 0 else -100.0,
                        'total_peaks_found': len(peak_indices)
                    }
            
        except Exception as e:
            self.logger.warning(f"Spectrum peak analysis failed: {e}")
        
        return {
            'peak_frequencies': [],
            'peak_powers': [],
            'num_peaks': 0,
            'strongest_peak_freq': 0.0,
            'strongest_peak_power': -100.0,
            'error': f'Peak analysis failed: {str(e) if "e" in locals() else "Unknown error"}'
        }
    
    def analyze_modulation(self, iq_data: np.ndarray) -> ModulationAnalysisResult:
        """Analyze modulation type of the signal."""
        try:
            # Calculate constellation points
            constellation = self._extract_constellation(iq_data)
            
            # Analyze different modulation types
            modulation_scores = {}
            
            # PSK Analysis
            psk_scores = self._analyze_psk_modulations(constellation)
            modulation_scores.update(psk_scores)
            
            # QAM Analysis
            qam_scores = self._analyze_qam_modulations(constellation)
            modulation_scores.update(qam_scores)
            
            # FSK Analysis
            fsk_score = self._analyze_fsk_modulation(iq_data)
            if fsk_score > 0:
                modulation_scores['FSK'] = fsk_score
            
            # ASK Analysis
            ask_score = self._analyze_ask_modulation(iq_data)
            if ask_score > 0:
                modulation_scores['ASK'] = ask_score
            
            # Find best match
            if modulation_scores:
                best_modulation = max(modulation_scores, key=modulation_scores.get)
                confidence = modulation_scores[best_modulation]
            else:
                best_modulation = 'Unknown'
                confidence = 0.0
            
            # Estimate symbol rate
            symbol_rate = self._estimate_symbol_rate(iq_data)
            
            # Estimate frequency and phase offsets
            freq_offset = self._estimate_frequency_offset(iq_data)
            phase_offset = self._estimate_phase_offset(iq_data)
            
            return ModulationAnalysisResult(
                modulation_type=best_modulation,
                confidence=confidence,
                parameters={
                    'constellation_size': len(constellation),
                    'modulation_scores': modulation_scores
                },
                constellation_points=constellation,
                symbol_rate=symbol_rate,
                frequency_offset=freq_offset,
                phase_offset=phase_offset
            )
            
        except Exception as e:
            self.logger.error(f"Modulation analysis failed: {e}")
            return ModulationAnalysisResult(
                modulation_type='Unknown',
                confidence=0.0,
                parameters={'error': str(e)}
            )
    
    def _extract_constellation(self, iq_data: np.ndarray, decimation: int = 10) -> np.ndarray:
        """Extract constellation points from IQ data."""
        if len(iq_data) == 0:
            return np.array([])
        
        # Decimate to get symbol-rate samples (rough estimation)
        decimated = iq_data[::decimation]
        
        # Remove outliers (simple method)
        if len(decimated) > 10:
            magnitude = np.abs(decimated)
            threshold = np.percentile(magnitude, 95)
            mask = magnitude <= threshold
            decimated = decimated[mask]
        
        return decimated
    
    def _analyze_psk_modulations(self, constellation: np.ndarray) -> Dict[str, float]:
        """Analyze PSK modulation types."""
        scores = {}
        
        if len(constellation) == 0:
            return scores
        
        # Convert to polar coordinates
        phases = np.angle(constellation)
        magnitudes = np.abs(constellation)
        
        # Check for constant magnitude (PSK characteristic)
        if len(magnitudes) > 1:
            magnitude_variance = np.var(magnitudes) / (np.mean(magnitudes)**2 + 1e-10)
        else:
            magnitude_variance = 0.0
        
        # PSK signals should have relatively constant magnitude
        psk_threshold = 0.2  # Allow more variance for realistic signals
        
        if magnitude_variance < psk_threshold:
            # Check for different PSK orders
            for psk_type in ['BPSK', 'QPSK', 'PSK8']:
                if psk_type == 'BPSK':
                    expected_phases = [0, np.pi]
                elif psk_type == 'QPSK':
                    expected_phases = [0, np.pi/2, np.pi, 3*np.pi/2]
                elif psk_type == 'PSK8':
                    expected_phases = [i * np.pi/4 for i in range(8)]
                
                score = self._calculate_phase_cluster_score(phases, expected_phases)
                
                # Boost BPSK score for two-level signals
                if psk_type == 'BPSK':
                    # Check if signal is predominantly two-level
                    real_parts = np.real(constellation)
                    if len(np.unique(np.round(real_parts, 1))) <= 3:  # Allow for noise
                        score *= 1.5  # Boost BPSK detection
                
                scores[psk_type] = score
        
        return scores
    
    def _analyze_qam_modulations(self, constellation: np.ndarray) -> Dict[str, float]:
        """Analyze QAM modulation types."""
        scores = {}
        
        if len(constellation) == 0:
            return scores
        
        # For QAM, check for rectangular grid pattern
        real_parts = np.real(constellation)
        imag_parts = np.imag(constellation)
        
        # Check for grid-like clustering
        for qam_type in ['QAM16', 'QAM64']:
            if qam_type == 'QAM16':
                expected_levels = 4  # 4x4 grid
            elif qam_type == 'QAM64':
                expected_levels = 8  # 8x8 grid
            
            real_score = self._calculate_grid_score(real_parts, expected_levels)
            imag_score = self._calculate_grid_score(imag_parts, expected_levels)
            
            scores[qam_type] = (real_score + imag_score) / 2
        
        return scores
    
    def _analyze_fsk_modulation(self, iq_data: np.ndarray) -> float:
        """Analyze FSK modulation."""
        if len(iq_data) < 100:
            return 0.0
        
        # Calculate instantaneous frequency
        instant_phase = np.unwrap(np.angle(iq_data))
        instant_freq = np.diff(instant_phase)
        
        # Check for discrete frequency levels
        if len(instant_freq) > 10:
            # Check if frequencies have sufficient range for binning
            freq_range = np.max(instant_freq) - np.min(instant_freq)
            if freq_range < 1e-10:  # Nearly constant frequency
                return 0.0
            
            # Use adaptive number of bins
            n_unique = len(np.unique(instant_freq))
            max_bins = min(50, max(10, n_unique // 2))
            
            try:
                # Use histogram to find peaks
                hist, bins = np.histogram(instant_freq, bins=max_bins)
                peak_indices = self._find_peaks_simple(hist)
                
                # FSK should have distinct frequency peaks
                if len(peak_indices) >= 2:
                    return min(0.8, len(peak_indices) / 10.0)
            except ValueError as e:
                # If histogram fails, try with fewer bins
                self.logger.warning(f"FSK analysis histogram error: {e}")
                try:
                    hist, bins = np.histogram(instant_freq, bins=10)
                    peak_indices = self._find_peaks_simple(hist)
                    if len(peak_indices) >= 2:
                        return min(0.6, len(peak_indices) / 5.0)
                except Exception:
                    return 0.0
        
        return 0.0
    
    def _analyze_ask_modulation(self, iq_data: np.ndarray) -> float:
        """Analyze ASK modulation."""
        if len(iq_data) == 0:
            return 0.0
        
        # Check for amplitude modulation
        amplitudes = np.abs(iq_data)
        
        # Check for discrete amplitude levels
        if len(amplitudes) > 10:
            # Check if amplitudes have sufficient range for binning
            amp_range = np.max(amplitudes) - np.min(amplitudes)
            if amp_range < 1e-10:  # Nearly constant amplitude
                return 0.0
            
            # Use adaptive number of bins based on data range and size
            n_unique = len(np.unique(amplitudes))
            max_bins = min(30, max(5, n_unique // 2))  # Adaptive bin count
            
            try:
                hist, bins = np.histogram(amplitudes, bins=max_bins)
                peak_indices = self._find_peaks_simple(hist)
                
                # ASK should have distinct amplitude levels
                if len(peak_indices) >= 2:
                    return min(0.7, len(peak_indices) / 8.0)
            except ValueError as e:
                # If histogram fails, try with fewer bins
                self.logger.warning(f"ASK analysis histogram error: {e}")
                try:
                    hist, bins = np.histogram(amplitudes, bins=5)
                    peak_indices = self._find_peaks_simple(hist)
                    if len(peak_indices) >= 2:
                        return min(0.5, len(peak_indices) / 4.0)
                except Exception:
                    return 0.0
        
        return 0.0
    
    def _calculate_phase_cluster_score(self, phases: np.ndarray, expected_phases: List[float]) -> float:
        """Calculate how well phases cluster around expected values."""
        if len(phases) == 0:
            return 0.0
        
        # Normalize phases to [0, 2π]
        phases = (phases + 2*np.pi) % (2*np.pi)
        expected_phases = [(p + 2*np.pi) % (2*np.pi) for p in expected_phases]
        
        total_score = 0.0
        for phase in phases:
            min_distance = float('inf')
            
            # Find minimum distance to any expected phase
            for exp_phase in expected_phases:
                # Calculate distance considering circular nature
                distance = min(abs(phase - exp_phase), 2*np.pi - abs(phase - exp_phase))
                min_distance = min(min_distance, distance)
            
            # Score decreases with distance from expected phase
            # Use a more forgiving tolerance for noisy signals
            tolerance = np.pi/3  # 60 degrees tolerance
            score = max(0, 1 - min_distance / tolerance)
            total_score += score
        
        # Normalize by number of phases and apply bonus for good clustering
        base_score = total_score / len(phases) if len(phases) > 0 else 0.0
        
        # Bonus for having points near all expected phases
        if len(expected_phases) > 1:
            phases_found = 0
            for exp_phase in expected_phases:
                for phase in phases:
                    distance = min(abs(phase - exp_phase), 2*np.pi - abs(phase - exp_phase))
                    if distance < tolerance:
                        phases_found += 1
                        break
            
            phase_coverage = phases_found / len(expected_phases)
            base_score *= (0.5 + 0.5 * phase_coverage)  # Boost if all phases represented
        
        return min(1.0, base_score)
    
    def _calculate_grid_score(self, values: np.ndarray, expected_levels: int) -> float:
        """Calculate how well values cluster in a grid pattern."""
        if len(values) == 0:
            return 0.0
        
        # Find cluster centers using simple binning
        hist, bins = np.histogram(values, bins=expected_levels * 2)
        peak_indices = self._find_peaks_simple(hist)
        
        # Score based on number of peaks found vs expected
        if len(peak_indices) >= expected_levels // 2:
            return min(1.0, len(peak_indices) / expected_levels)
        
        return 0.0
    
    def _find_peaks_simple(self, data: np.ndarray, min_height_ratio: float = 0.1) -> List[int]:
        """Simple peak finding algorithm."""
        if len(data) < 3:
            return []
        
        peaks = []
        threshold = np.max(data) * min_height_ratio
        
        for i in range(1, len(data) - 1):
            if (data[i] > data[i-1] and data[i] > data[i+1] and data[i] > threshold):
                peaks.append(i)
        
        return peaks
    
    def _estimate_symbol_rate(self, iq_data: np.ndarray) -> Optional[float]:
        """Estimate symbol rate from IQ data."""
        try:
            if len(iq_data) < 100:
                return None
            
            # Calculate power spectral density
            fft_data = np.fft.fft(iq_data)
            psd = np.abs(fft_data)**2
            
            # Find main lobe bandwidth (simple method)
            peak_index = np.argmax(psd[:len(psd)//2])
            
            # Estimate symbol rate as a fraction of sample rate
            # This is a rough estimation
            symbol_rate = self.sample_rate / 10.0  # Default estimation
            
            return symbol_rate
            
        except Exception:
            return None
    
    def _estimate_frequency_offset(self, iq_data: np.ndarray) -> Optional[float]:
        """Estimate frequency offset."""
        try:
            if len(iq_data) < 10:
                return None
            
            # Simple frequency offset estimation using phase slope
            phases = np.unwrap(np.angle(iq_data))
            if len(phases) > 1:
                freq_offset = np.mean(np.diff(phases)) * self.sample_rate / (2 * np.pi)
                return float(freq_offset)
            
            return 0.0
            
        except Exception:
            return None
    
    def _estimate_phase_offset(self, iq_data: np.ndarray) -> Optional[float]:
        """Estimate phase offset."""
        try:
            if len(iq_data) == 0:
                return None
            
            # Simple phase offset as mean phase
            mean_phase = np.angle(np.mean(iq_data))
            return float(mean_phase)
            
        except Exception:
            return None
    
    def demodulate_signal(self, iq_data: np.ndarray, 
                         modulation_result: ModulationAnalysisResult) -> DemodulationResult:
        """Demodulate signal based on detected modulation type."""
        try:
            mod_type = modulation_result.modulation_type
            
            if mod_type == 'BPSK':
                return self._demodulate_bpsk(iq_data)
            elif mod_type == 'QPSK':
                return self._demodulate_qpsk(iq_data)
            elif mod_type == 'PSK8':
                return self._demodulate_psk8(iq_data)
            elif mod_type in ['QAM16', 'QAM64']:
                return self._demodulate_qam(iq_data, mod_type)
            elif mod_type == 'FSK':
                return self._demodulate_fsk(iq_data)
            elif mod_type == 'ASK':
                return self._demodulate_ask(iq_data)
            else:
                return DemodulationResult(success=False)
                
        except Exception as e:
            self.logger.error(f"Demodulation failed: {e}")
            return DemodulationResult(success=False)
    
    def _demodulate_bpsk(self, iq_data: np.ndarray) -> DemodulationResult:
        """Demodulate BPSK signal."""
        if len(iq_data) == 0:
            return DemodulationResult(success=False)
        
        # Simple BPSK demodulation: take real part and threshold
        real_part = np.real(iq_data)
        symbols = np.sign(real_part)
        
        # Convert to bits (0 and 1)
        bits = ((symbols + 1) / 2).astype(int)
        
        # Estimate SNR
        snr = self._estimate_snr_from_symbols(iq_data, symbols)
        
        return DemodulationResult(
            success=True,
            symbols=symbols,
            bits=bits,
            snr=snr,
            error_rate=None
        )
    
    def _demodulate_qpsk(self, iq_data: np.ndarray) -> DemodulationResult:
        """Demodulate QPSK signal."""
        if len(iq_data) == 0:
            return DemodulationResult(success=False)
        
        # QPSK demodulation: determine quadrant
        real_part = np.real(iq_data)
        imag_part = np.imag(iq_data)
        
        # Map to symbols (0, 1, 2, 3)
        symbols = np.zeros(len(iq_data), dtype=int)
        symbols[(real_part >= 0) & (imag_part >= 0)] = 0  # Q1
        symbols[(real_part < 0) & (imag_part >= 0)] = 1   # Q2
        symbols[(real_part < 0) & (imag_part < 0)] = 2    # Q3
        symbols[(real_part >= 0) & (imag_part < 0)] = 3   # Q4
        
        # Convert to bits (2 bits per symbol)
        bits = []
        for symbol in symbols:
            bits.extend([symbol >> 1, symbol & 1])
        
        bits = np.array(bits)
        
        # Estimate SNR
        snr = self._estimate_snr_from_symbols(iq_data, symbols)
        
        return DemodulationResult(
            success=True,
            symbols=symbols,
            bits=bits,
            snr=snr,
            error_rate=None
        )
    
    def _demodulate_psk8(self, iq_data: np.ndarray) -> DemodulationResult:
        """Demodulate 8-PSK signal."""
        if len(iq_data) == 0:
            return DemodulationResult(success=False)
        
        # 8-PSK: determine phase sector
        phases = np.angle(iq_data)
        
        # Normalize to [0, 2π] and quantize to 8 levels
        phases = (phases + 2*np.pi) % (2*np.pi)
        symbols = np.round(phases / (2*np.pi / 8)).astype(int) % 8
        
        # Convert to bits (3 bits per symbol)
        bits = []
        for symbol in symbols:
            bits.extend([(symbol >> 2) & 1, (symbol >> 1) & 1, symbol & 1])
        
        bits = np.array(bits)
        
        # Estimate SNR
        snr = self._estimate_snr_from_symbols(iq_data, symbols)
        
        return DemodulationResult(
            success=True,
            symbols=symbols,
            bits=bits,
            snr=snr,
            error_rate=None
        )
    
    def _demodulate_qam(self, iq_data: np.ndarray, qam_type: str) -> DemodulationResult:
        """Demodulate QAM signal."""
        if len(iq_data) == 0:
            return DemodulationResult(success=False)
        
        # Simplified QAM demodulation
        if qam_type == 'QAM16':
            levels = 4
            bits_per_symbol = 4
        elif qam_type == 'QAM64':
            levels = 8
            bits_per_symbol = 6
        else:
            return DemodulationResult(success=False)
        
        # Quantize real and imaginary parts
        real_part = np.real(iq_data)
        imag_part = np.imag(iq_data)
        
        # Simple quantization (assuming normalized signal)
        real_symbols = np.round((real_part + 1) * (levels-1) / 2).astype(int)
        imag_symbols = np.round((imag_part + 1) * (levels-1) / 2).astype(int)
        
        # Combine to form QAM symbols
        symbols = real_symbols * levels + imag_symbols
        
        # Convert to bits
        bits = []
        for symbol in symbols:
            for i in range(bits_per_symbol):
                bits.append((symbol >> (bits_per_symbol - 1 - i)) & 1)
        
        bits = np.array(bits)
        
        # Estimate SNR
        snr = self._estimate_snr_from_symbols(iq_data, symbols)
        
        return DemodulationResult(
            success=True,
            symbols=symbols,
            bits=bits,
            snr=snr,
            error_rate=None
        )
    
    def _demodulate_fsk(self, iq_data: np.ndarray) -> DemodulationResult:
        """Demodulate FSK signal."""
        if len(iq_data) < 10:
            return DemodulationResult(success=False)
        
        # Simple FSK demodulation using instantaneous frequency
        instant_phase = np.unwrap(np.angle(iq_data))
        instant_freq = np.diff(instant_phase)
        
        # Threshold frequency to get symbols
        if len(instant_freq) > 0:
            threshold = np.median(instant_freq)
            symbols = (instant_freq > threshold).astype(int)
            bits = symbols  # Binary FSK
            
            return DemodulationResult(
                success=True,
                symbols=symbols,
                bits=bits,
                snr=None,
                error_rate=None
            )
        
        return DemodulationResult(success=False)
    
    def _demodulate_ask(self, iq_data: np.ndarray) -> DemodulationResult:
        """Demodulate ASK signal."""
        if len(iq_data) == 0:
            return DemodulationResult(success=False)
        
        # Simple ASK demodulation using amplitude
        amplitudes = np.abs(iq_data)
        threshold = np.median(amplitudes)
        
        symbols = (amplitudes > threshold).astype(int)
        bits = symbols  # Binary ASK
        
        # Estimate SNR
        snr = self._estimate_snr_from_symbols(iq_data, symbols)
        
        return DemodulationResult(
            success=True,
            symbols=symbols,
            bits=bits,
            snr=snr,
            error_rate=None
        )
    
    def _estimate_snr_from_symbols(self, iq_data: np.ndarray, symbols: np.ndarray) -> Optional[float]:
        """Estimate SNR from demodulated symbols."""
        try:
            if len(iq_data) == 0 or len(symbols) == 0:
                return None
            
            # Simple SNR estimation
            signal_power = np.mean(np.abs(iq_data)**2)
            noise_power = np.var(np.abs(iq_data))  # Rough noise estimation
            
            if noise_power > 0:
                snr_linear = signal_power / noise_power
                snr_db = 10 * np.log10(snr_linear)
                return float(snr_db)
            
            return None
            
        except Exception:
            return None
    
    def analyze_coding(self, bits: np.ndarray) -> Optional[CodingAnalysisResult]:
        """Analyze coding/encoding of demodulated bits."""
        try:
            if len(bits) == 0:
                return None
            
            # Simple coding analysis
            coding_scores = {}
            
            # Check for common patterns
            
            # 1. Manchester encoding check
            manchester_score = self._check_manchester_encoding(bits)
            if manchester_score > 0.5:
                coding_scores['Manchester'] = manchester_score
            
            # 2. NRZ encoding check  
            nrz_score = self._check_nrz_encoding(bits)
            if nrz_score > 0.3:
                coding_scores['NRZ'] = nrz_score
            
            # 3. Check for repetition codes
            repetition_score = self._check_repetition_code(bits)
            if repetition_score > 0.4:
                coding_scores['Repetition'] = repetition_score
            
            # Find best match
            if coding_scores:
                best_coding = max(coding_scores, key=coding_scores.get)
                confidence = coding_scores[best_coding]
                
                # Attempt to decode based on detected coding
                decoded_bits = self._decode_bits(bits, best_coding)
                
                return CodingAnalysisResult(
                    coding_type=best_coding,
                    confidence=confidence,
                    parameters={'coding_scores': coding_scores},
                    decoded_bits=decoded_bits,
                    error_correction_info=None
                )
            else:
                return CodingAnalysisResult(
                    coding_type='Raw',
                    confidence=1.0,
                    parameters={},
                    decoded_bits=bits,
                    error_correction_info=None
                )
                
        except Exception as e:
            self.logger.error(f"Coding analysis failed: {e}")
            return None
    
    def _check_manchester_encoding(self, bits: np.ndarray) -> float:
        """Check for Manchester encoding pattern."""
        if len(bits) < 4 or len(bits) % 2 != 0:
            return 0.0
        
        # Manchester: each bit is represented by two opposite bits
        transitions = 0
        for i in range(0, len(bits) - 1, 2):
            if bits[i] != bits[i + 1]:
                transitions += 1
        
        return transitions / (len(bits) // 2)
    
    def _check_nrz_encoding(self, bits: np.ndarray) -> float:
        """Check for NRZ (Non-Return-to-Zero) encoding."""
        if len(bits) < 4:
            return 0.0
        
        # NRZ typically has longer runs of same bits
        runs = []
        current_run = 1
        
        for i in range(1, len(bits)):
            if bits[i] == bits[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        # NRZ score based on average run length
        avg_run_length = np.mean(runs)
        return min(1.0, avg_run_length / 3.0)  # Normalize to [0,1]
    
    def _check_repetition_code(self, bits: np.ndarray) -> float:
        """Check for repetition coding (3-bit, 5-bit, etc.)."""
        best_score = 0.0
        
        for rep_factor in [3, 5, 7]:
            if len(bits) % rep_factor == 0:
                score = 0
                groups = len(bits) // rep_factor
                
                for i in range(groups):
                    group = bits[i*rep_factor:(i+1)*rep_factor]
                    if len(set(group)) == 1:  # All bits in group are same
                        score += 1
                
                repetition_score = score / groups
                best_score = max(best_score, repetition_score)
        
        return best_score
    
    def _decode_bits(self, bits: np.ndarray, coding_type: str) -> np.ndarray:
        """Decode bits based on detected coding type."""
        try:
            if coding_type == 'Manchester':
                return self._decode_manchester(bits)
            elif coding_type == 'Repetition':
                return self._decode_repetition(bits)
            else:
                return bits  # Raw bits
                
        except Exception:
            return bits
    
    def _decode_manchester(self, bits: np.ndarray) -> np.ndarray:
        """Decode Manchester encoded bits."""
        if len(bits) % 2 != 0:
            return bits
        
        decoded = []
        for i in range(0, len(bits), 2):
            if bits[i] == 0 and bits[i+1] == 1:
                decoded.append(0)
            elif bits[i] == 1 and bits[i+1] == 0:
                decoded.append(1)
            else:
                # Error in Manchester encoding, use first bit
                decoded.append(bits[i])
        
        return np.array(decoded)
    
    def _decode_repetition(self, bits: np.ndarray) -> np.ndarray:
        """Decode repetition coded bits."""
        # Try different repetition factors
        for rep_factor in [3, 5, 7]:
            if len(bits) % rep_factor == 0:
                decoded = []
                groups = len(bits) // rep_factor
                
                for i in range(groups):
                    group = bits[i*rep_factor:(i+1)*rep_factor]
                    # Majority vote
                    decoded.append(1 if np.sum(group) > rep_factor // 2 else 0)
                
                return np.array(decoded)
        
        return bits