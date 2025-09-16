"""
Modulation Analysis and Detection Module
Provides automatic modulation recognition, demodulation, and encoding detection
for supported modulation schemes in SDR and scikit-dsp-comm libraries.
"""

import numpy as np
import scipy.signal as signal
from scipy.stats import kurtosis, skew
from typing import Dict, Tuple, List, Optional, Any
import logging

# Try to import scikit-dsp-comm components
try:
    import sk_dsp_comm.digitalcom as dc
    import sk_dsp_comm.fec_conv as fec
    import sk_dsp_comm.synchronization as sync
    SCIKIT_DSP_AVAILABLE = True
except ImportError:
    SCIKIT_DSP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModulationAnalyzer:
    """
    Automatic modulation recognition and analysis.
    Supports PSK, QAM, FSK, and OFDM detection.
    """
    
    def __init__(self, sample_rate: float = 2e6):
        self.sample_rate = sample_rate
        self.constellation_threshold = 0.1
        self.frequency_deviation_threshold = 0.05
        
        # Feature extraction parameters
        self.window_size = 1024
        self.overlap = 0.5
        
    def detect_modulation(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect modulation type from IQ signal data.
        
        Args:
            signal_data: Complex IQ signal data
            
        Returns:
            Dictionary containing detected modulation info
        """
        if len(signal_data) < self.window_size:
            return {"type": "Unknown", "confidence": 0.0, "parameters": {}}
        
        try:
            # Extract modulation features
            features = self._extract_features(signal_data)
            
            # Classify based on features
            classification = self._classify_modulation(features)
            
            # Estimate modulation parameters
            parameters = self._estimate_parameters(signal_data, classification["type"])
            
            return {
                "type": classification["type"],
                "confidence": classification["confidence"],
                "parameters": parameters,
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Modulation detection error: {e}")
            return {"type": "Unknown", "confidence": 0.0, "parameters": {}}
    
    def _extract_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract statistical and spectral features from signal."""
        features = {}
        
        # Amplitude features
        amplitude = np.abs(signal_data)
        features["amplitude_mean"] = np.mean(amplitude)
        features["amplitude_std"] = np.std(amplitude)
        features["amplitude_kurtosis"] = kurtosis(amplitude)
        features["amplitude_skewness"] = skew(amplitude)
        
        # Phase features
        phase = np.angle(signal_data)
        unwrapped_phase = np.unwrap(phase)
        phase_diff = np.diff(unwrapped_phase)
        features["phase_variance"] = np.var(phase_diff)
        features["phase_kurtosis"] = kurtosis(phase_diff)
        
        # Frequency domain features
        fft_data = np.fft.fft(signal_data, n=self.window_size)
        psd = np.abs(fft_data) ** 2
        features["spectral_centroid"] = self._spectral_centroid(psd)
        features["spectral_bandwidth"] = self._spectral_bandwidth(psd)
        features["spectral_rolloff"] = self._spectral_rolloff(psd)
        
        # Constellation features
        features.update(self._constellation_features(signal_data))
        
        return features
    
    def _constellation_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract constellation-specific features."""
        # Normalize signal
        signal_norm = signal_data / np.std(signal_data)
        
        # Sample constellation points
        step = max(1, len(signal_norm) // 1000)
        constellation = signal_norm[::step]
        
        features = {}
        
        # Cluster analysis for constellation points
        real_parts = np.real(constellation)
        imag_parts = np.imag(constellation)
        
        # Variance in I and Q channels
        features["i_channel_var"] = np.var(real_parts)
        features["q_channel_var"] = np.var(imag_parts)
        
        # Circular variance
        radius = np.abs(constellation)
        features["radius_variance"] = np.var(radius)
        features["radius_mean"] = np.mean(radius)
        
        # Count potential constellation points
        features["constellation_points"] = self._estimate_constellation_size(constellation)
        
        return features
    
    def _estimate_constellation_size(self, constellation: np.ndarray) -> int:
        """Estimate number of constellation points."""
        try:
            from sklearn.cluster import KMeans
            
            # Reshape for clustering
            points = np.column_stack([np.real(constellation), np.imag(constellation)])
            
            # Try different cluster numbers
            inertias = []
            k_range = range(2, 17)  # Test 2 to 16 constellation points
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(points)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point
            diff = np.diff(inertias)
            diff2 = np.diff(diff)
            elbow = np.argmax(diff2) + 2
            
            return min(max(elbow, 2), 16)
            
        except ImportError:
            # Fallback without sklearn
            return self._simple_constellation_estimate(constellation)
    
    def _simple_constellation_estimate(self, constellation: np.ndarray) -> int:
        """Simple constellation size estimation without sklearn."""
        # Use amplitude quantization levels
        amplitude = np.abs(constellation)
        amplitude_hist, _ = np.histogram(amplitude, bins=50)
        
        # Find peaks in amplitude histogram
        peaks = []
        for i in range(1, len(amplitude_hist) - 1):
            if amplitude_hist[i] > amplitude_hist[i-1] and amplitude_hist[i] > amplitude_hist[i+1]:
                if amplitude_hist[i] > np.max(amplitude_hist) * 0.1:
                    peaks.append(i)
        
        # Common constellation sizes
        common_sizes = [2, 4, 8, 16, 32, 64, 256]
        estimated_size = len(peaks) ** 2 if peaks else 4
        
        # Find closest common size
        return min(common_sizes, key=lambda x: abs(x - estimated_size))
    
    def _classify_modulation(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Classify modulation type based on extracted features."""
        confidence = 0.0
        mod_type = "Unknown"
        
        # Rule-based classification
        constellation_points = features.get("constellation_points", 4)
        amplitude_var = features.get("amplitude_std", 0) / features.get("amplitude_mean", 1)
        phase_var = features.get("phase_variance", 0)
        radius_var = features.get("radius_variance", 0)
        
        # More robust classification with priority order
        
        # 1. PSK detection (constant amplitude, discrete phases)
        if amplitude_var < 0.3 and radius_var < 0.2:
            if constellation_points <= 2:
                mod_type = "BPSK"
                confidence = 0.9
            elif constellation_points <= 4:
                mod_type = "QPSK"  
                confidence = 0.9
            elif constellation_points <= 8:
                mod_type = "8PSK"
                confidence = 0.8
            elif constellation_points <= 16:
                mod_type = "16PSK"
                confidence = 0.7
        
        # 2. QAM detection (variable amplitude and phase, multiple rings)
        elif amplitude_var > 0.3 and radius_var > 0.2:
            if constellation_points <= 4:
                # Could be QPSK with noise, but check more carefully
                if phase_var > 0.2:
                    mod_type = "QPSK"
                    confidence = 0.6
                else:
                    mod_type = "QAM4"
                    confidence = 0.7
            elif constellation_points <= 16:
                mod_type = "QAM16"
                confidence = 0.8
            elif constellation_points <= 64:
                mod_type = "QAM64"
                confidence = 0.7
            else:
                mod_type = "QAM256"
                confidence = 0.6
        
        # 3. MSK/OQPSK detection (constant amplitude, continuous phase)
        elif amplitude_var < 0.2 and phase_var > 0.5:
            spectral_bandwidth = features.get("spectral_bandwidth", 0)
            if spectral_bandwidth < 0.5:
                mod_type = "MSK"
                confidence = 0.8
            else:
                mod_type = "OQPSK"
                confidence = 0.7
        
        # 4. FSK detection (frequency variations)
        elif features.get("spectral_bandwidth", 0) > 0.5:
            mod_type = "FSK"
            confidence = 0.7
            
            # Check for GFSK (smoother transitions)
            if features.get("phase_kurtosis", 0) < 2.5:
                mod_type = "GFSK"
                confidence = 0.8
        
        # 5. AM detection (amplitude variations with constant phase)
        elif amplitude_var > 0.4 and phase_var < 0.1:
            mod_type = "AM"
            confidence = 0.7
        
        # 6. FM detection (constant amplitude with phase variations)
        elif amplitude_var < 0.2 and phase_var > 1.5:
            mod_type = "FM"
            confidence = 0.7
        
        # 7. Fallback to QPSK if unsure but has digital characteristics
        elif phase_var > 0.1:
            mod_type = "QPSK"
            confidence = 0.5
        
        return {"type": mod_type, "confidence": confidence}
    
    def _estimate_parameters(self, signal_data: np.ndarray, mod_type: str) -> Dict[str, float]:
        """Estimate modulation parameters."""
        parameters = {}
        
        try:
            if mod_type in ["PSK", "QPSK", "8PSK", "QAM16", "QAM64", "QAM256"]:
                # Estimate symbol rate for digital modulations
                parameters["symbol_rate"] = self._estimate_symbol_rate(signal_data)
                parameters["snr_estimate"] = self._estimate_snr(signal_data)
                
            elif mod_type in ["FSK", "GFSK", "MSK"]:
                # Estimate frequency deviation
                parameters["frequency_deviation"] = self._estimate_frequency_deviation(signal_data)
                parameters["symbol_rate"] = self._estimate_symbol_rate(signal_data)
                
            elif mod_type == "AM":
                # Estimate modulation depth
                parameters["modulation_depth"] = self._estimate_am_depth(signal_data)
                
            elif mod_type == "FM":
                # Estimate frequency deviation
                parameters["frequency_deviation"] = self._estimate_frequency_deviation(signal_data)
                
        except Exception as e:
            logger.warning(f"Parameter estimation error for {mod_type}: {e}")
        
        return parameters
    
    def _estimate_symbol_rate(self, signal_data: np.ndarray) -> float:
        """Estimate symbol rate using autocorrelation."""
        try:
            # Use magnitude of signal for symbol timing
            magnitude = np.abs(signal_data)
            
            # Remove DC component
            magnitude = magnitude - np.mean(magnitude)
            
            # Autocorrelation
            autocorr = np.correlate(magnitude, magnitude, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr[1:], height=np.max(autocorr) * 0.1)
            
            if len(peaks) > 0:
                # Symbol period in samples
                symbol_period_samples = peaks[0] + 1
                symbol_rate = self.sample_rate / symbol_period_samples
                return float(symbol_rate)
            
        except Exception as e:
            logger.warning(f"Symbol rate estimation error: {e}")
        
        return 1000.0  # Default fallback
    
    def _estimate_snr(self, signal_data: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio."""
        try:
            # Simple SNR estimation using signal and noise power
            signal_power = np.mean(np.abs(signal_data) ** 2)
            
            # Estimate noise from high-frequency components
            filtered_signal = signal.butter(4, 0.8, output='sos')
            noise_estimate = signal.sosfilt(filtered_signal, signal_data) - signal_data
            noise_power = np.mean(np.abs(noise_estimate) ** 2)
            
            if noise_power > 0:
                snr_linear = signal_power / noise_power
                snr_db = 10 * np.log10(snr_linear)
                return float(snr_db)
                
        except Exception as e:
            logger.warning(f"SNR estimation error: {e}")
        
        return 10.0  # Default SNR
    
    def _estimate_frequency_deviation(self, signal_data: np.ndarray) -> float:
        """Estimate frequency deviation for FM/FSK signals."""
        try:
            # Instantaneous frequency using phase derivative
            phase = np.angle(signal_data)
            unwrapped_phase = np.unwrap(phase)
            inst_freq = np.diff(unwrapped_phase) * self.sample_rate / (2 * np.pi)
            
            # Frequency deviation is the standard deviation of instantaneous frequency
            freq_deviation = np.std(inst_freq)
            return float(freq_deviation)
            
        except Exception as e:
            logger.warning(f"Frequency deviation estimation error: {e}")
        
        return 1000.0  # Default deviation
    
    def _estimate_am_depth(self, signal_data: np.ndarray) -> float:
        """Estimate AM modulation depth."""
        try:
            amplitude = np.abs(signal_data)
            amplitude_max = np.max(amplitude)
            amplitude_min = np.min(amplitude)
            
            if amplitude_max > 0:
                modulation_depth = (amplitude_max - amplitude_min) / (amplitude_max + amplitude_min)
                return float(modulation_depth * 100)  # Return as percentage
                
        except Exception as e:
            logger.warning(f"AM depth estimation error: {e}")
        
        return 50.0  # Default depth
    
    def _spectral_centroid(self, psd: np.ndarray) -> float:
        """Calculate spectral centroid."""
        freqs = np.arange(len(psd))
        return float(np.sum(freqs * psd) / np.sum(psd))
    
    def _spectral_bandwidth(self, psd: np.ndarray) -> float:
        """Calculate spectral bandwidth."""
        centroid = self._spectral_centroid(psd)
        freqs = np.arange(len(psd))
        return float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd)))
    
    def _spectral_rolloff(self, psd: np.ndarray, rolloff_percent: float = 0.85) -> float:
        """Calculate spectral rolloff point."""
        cumsum_psd = np.cumsum(psd)
        total_energy = cumsum_psd[-1]
        rolloff_energy = rolloff_percent * total_energy
        
        rolloff_idx = np.where(cumsum_psd >= rolloff_energy)[0]
        if len(rolloff_idx) > 0:
            return float(rolloff_idx[0])
        return float(len(psd) - 1)


class EncodingAnalyzer:
    """
    Channel coding detection and analysis.
    Supports detection of common FEC schemes.
    """
    
    def __init__(self):
        self.block_sizes = {
            "Hamming": [7, 15, 31, 63, 127],
            "BCH": [15, 31, 63, 127, 255, 511],
            "Reed-Solomon": [15, 31, 63, 127, 255],
            "Convolutional": None,  # Variable length
            "Turbo": None,  # Variable length
            "LDPC": [576, 1152, 1944],  # Common LDPC block sizes
            "Polar": [128, 256, 512, 1024]
        }
    
    def detect_encoding(self, bit_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect channel coding from bit sequence.
        
        Args:
            bit_data: Binary data sequence
            
        Returns:
            Dictionary containing detected encoding info
        """
        if len(bit_data) < 64:
            return {"type": "None", "confidence": 0.0, "parameters": {}}
        
        try:
            # For demonstration, simulate some encoding detection
            # In real implementation, this would analyze bit patterns, parity checks, etc.
            
            # Check for block codes
            block_detection = self._detect_block_codes(bit_data)
            
            # Check for convolutional codes
            conv_detection = self._detect_convolutional(bit_data)
            
            # Simple heuristic: if we have regular patterns, likely encoded
            bit_patterns = self._analyze_bit_patterns(bit_data)
            
            if bit_patterns["regularity"] > 0.7:
                # High regularity suggests encoding
                if len(bit_data) % 7 == 0:
                    return {
                        "type": "Hamming",
                        "confidence": 0.8,
                        "parameters": {"block_size": 7, "estimated_rate": "4/7"}
                    }
                elif len(bit_data) % 15 == 0:
                    return {
                        "type": "BCH",
                        "confidence": 0.7,
                        "parameters": {"block_size": 15, "estimated_rate": "11/15"}
                    }
                else:
                    return {
                        "type": "Convolutional",
                        "confidence": 0.6,
                        "parameters": {"constraint_length": 7, "estimated_rate": "1/2"}
                    }
            
            # Choose best detection
            if block_detection["confidence"] > conv_detection["confidence"] and block_detection["confidence"] > 0.3:
                return block_detection
            elif conv_detection["confidence"] > 0.3:
                return conv_detection
            else:
                # No clear encoding detected
                return {"type": "None", "confidence": 0.0, "parameters": {}}
                
        except Exception as e:
            logger.error(f"Encoding detection error: {e}")
            return {"type": "None", "confidence": 0.0, "parameters": {}}
    
    def _analyze_bit_patterns(self, bit_data: np.ndarray) -> Dict[str, float]:
        """Analyze bit patterns for encoding indicators."""
        try:
            # Calculate run length statistics
            runs = []
            current_run = 1
            for i in range(1, len(bit_data)):
                if bit_data[i] == bit_data[i-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)
            
            # Calculate regularity metrics
            run_variance = np.var(runs) if len(runs) > 1 else 0
            bit_transitions = np.sum(np.diff(bit_data.astype(int)) != 0)
            transition_rate = bit_transitions / (len(bit_data) - 1) if len(bit_data) > 1 else 0
            
            # Encoded data often has more regular patterns
            regularity = 1.0 / (1.0 + run_variance) if run_variance > 0 else 0.5
            
            return {
                "regularity": float(regularity),
                "transition_rate": float(transition_rate),
                "avg_run_length": float(np.mean(runs)) if runs else 1.0
            }
            
        except Exception as e:
            logger.warning(f"Bit pattern analysis error: {e}")
            return {"regularity": 0.0, "transition_rate": 0.5, "avg_run_length": 1.0}
    
    def _detect_block_codes(self, bit_data: np.ndarray) -> Dict[str, Any]:
        """Detect block codes by analyzing block structure."""
        best_detection = {"type": "None", "confidence": 0.0, "parameters": {}}
        
        for code_type, block_sizes in self.block_sizes.items():
            if block_sizes is None:
                continue
                
            for block_size in block_sizes:
                if len(bit_data) < block_size * 3:  # Need at least 3 blocks
                    continue
                
                confidence = self._analyze_block_structure(bit_data, block_size, code_type)
                
                if confidence > best_detection["confidence"]:
                    best_detection = {
                        "type": code_type,
                        "confidence": confidence,
                        "parameters": {
                            "block_size": block_size,
                            "estimated_rate": self._estimate_code_rate(code_type, block_size)
                        }
                    }
        
        return best_detection
    
    def _detect_convolutional(self, bit_data: np.ndarray) -> Dict[str, Any]:
        """Detect convolutional codes using Viterbi-like analysis."""
        try:
            # Look for patterns typical of convolutional codes
            # This is a simplified detection based on autocorrelation
            
            # Calculate bit transitions
            transitions = np.diff(bit_data.astype(int))
            transition_rate = np.mean(np.abs(transitions))
            
            # Convolutional codes typically have more regular transition patterns
            if 0.3 < transition_rate < 0.7:
                return {
                    "type": "Convolutional",
                    "confidence": 0.6,
                    "parameters": {
                        "constraint_length": self._estimate_constraint_length(bit_data),
                        "estimated_rate": "1/2"  # Most common rate
                    }
                }
        except Exception as e:
            logger.warning(f"Convolutional detection error: {e}")
        
        return {"type": "None", "confidence": 0.0, "parameters": {}}
    
    def _analyze_block_structure(self, bit_data: np.ndarray, block_size: int, code_type: str) -> float:
        """Analyze if data fits expected block structure."""
        try:
            # Reshape into blocks
            num_complete_blocks = len(bit_data) // block_size
            if num_complete_blocks < 2:
                return 0.0
            
            blocks = bit_data[:num_complete_blocks * block_size].reshape(-1, block_size)
            
            # Check for systematic code structure (if applicable)
            if code_type in ["Hamming", "BCH"]:
                return self._check_systematic_structure(blocks, code_type)
            
            # General block consistency check
            block_weights = np.sum(blocks, axis=1)
            weight_variance = np.var(block_weights)
            
            # Lower variance suggests more structured (encoded) data
            if weight_variance < block_size * 0.25:
                return 0.7
            elif weight_variance < block_size * 0.5:
                return 0.5
            else:
                return 0.2
                
        except Exception as e:
            logger.warning(f"Block structure analysis error: {e}")
            return 0.0
    
    def _check_systematic_structure(self, blocks: np.ndarray, code_type: str) -> float:
        """Check for systematic code structure."""
        # This is a simplified check - real implementation would need
        # specific generator matrices for each code
        
        block_size = blocks.shape[1]
        
        # For Hamming codes, check if parity bits satisfy parity equations
        if code_type == "Hamming":
            if block_size == 7:  # (7,4) Hamming
                return self._check_hamming_7_4(blocks)
            elif block_size == 15:  # (15,11) Hamming
                return self._check_hamming_15_11(blocks)
        
        # Generic systematic check
        # Look for correlation between different bit positions
        correlations = []
        for i in range(block_size):
            for j in range(i+1, block_size):
                corr = np.corrcoef(blocks[:, i], blocks[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if correlations:
            avg_correlation = np.mean(correlations)
            # Systematic codes should have some correlation between information and parity bits
            if 0.2 < avg_correlation < 0.8:
                return 0.6
        
        return 0.3
    
    def _check_hamming_7_4(self, blocks: np.ndarray) -> float:
        """Check (7,4) Hamming code structure."""
        try:
            # Parity check matrix for (7,4) Hamming code
            # H = [1 1 1 0 1 0 0; 1 1 0 1 0 1 0; 1 0 1 1 0 0 1]
            H = np.array([
                [1, 1, 1, 0, 1, 0, 0],
                [1, 1, 0, 1, 0, 1, 0],
                [1, 0, 1, 1, 0, 0, 1]
            ])
            
            valid_blocks = 0
            for block in blocks:
                syndrome = np.dot(H, block) % 2
                if np.sum(syndrome) == 0:  # Valid codeword
                    valid_blocks += 1
            
            validity_ratio = valid_blocks / len(blocks)
            return validity_ratio
            
        except Exception as e:
            logger.warning(f"Hamming (7,4) check error: {e}")
            return 0.3
    
    def _check_hamming_15_11(self, blocks: np.ndarray) -> float:
        """Check (15,11) Hamming code structure (simplified)."""
        # Simplified check for (15,11) Hamming code
        # Real implementation would use the full parity check matrix
        
        # Count bit weights - Hamming codes have specific weight distributions
        weights = np.sum(blocks, axis=1)
        weight_hist, _ = np.histogram(weights, bins=16)
        
        # Hamming codes have specific weight enumerators
        # This is a simplified heuristic
        peak_weights = np.where(weight_hist > np.max(weight_hist) * 0.5)[0]
        
        if len(peak_weights) == 2 and abs(peak_weights[1] - peak_weights[0]) > 3:
            return 0.6
        
        return 0.3
    
    def _estimate_constraint_length(self, bit_data: np.ndarray) -> int:
        """Estimate constraint length for convolutional codes."""
        # Simple estimation based on autocorrelation length
        try:
            autocorr = np.correlate(bit_data, bit_data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find where autocorrelation drops significantly
            threshold = np.max(autocorr) * 0.1
            significant_indices = np.where(autocorr > threshold)[0]
            
            if len(significant_indices) > 1:
                constraint_length = significant_indices[-1]
                return min(max(constraint_length, 3), 9)  # Typical range 3-9
        except Exception as e:
            logger.warning(f"Constraint length estimation error: {e}")
        
        return 7  # Common default
    
    def _estimate_code_rate(self, code_type: str, block_size: int) -> str:
        """Estimate code rate for known code types."""
        rate_mappings = {
            "Hamming": {
                7: "4/7", 15: "11/15", 31: "26/31", 63: "57/63", 127: "120/127"
            },
            "BCH": {
                15: "7/15", 31: "21/31", 63: "45/63", 127: "92/127", 255: "223/255"
            },
            "Reed-Solomon": {
                15: "11/15", 31: "25/31", 63: "55/63", 127: "115/127", 255: "239/255"
            }
        }
        
        if code_type in rate_mappings and block_size in rate_mappings[code_type]:
            return rate_mappings[code_type][block_size]
        
        return "1/2"  # Default rate


def create_modulation_analyzer(sample_rate: float = 2e6) -> ModulationAnalyzer:
    """Factory function to create modulation analyzer."""
    return ModulationAnalyzer(sample_rate)


def create_encoding_analyzer() -> EncodingAnalyzer:
    """Factory function to create encoding analyzer."""
    return EncodingAnalyzer()