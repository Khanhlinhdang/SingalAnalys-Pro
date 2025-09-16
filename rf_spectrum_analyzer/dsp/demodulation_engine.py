"""
Demodulation Engine
Provides demodulation capabilities for various modulation schemes
detected by the modulation analyzer.
"""

import numpy as np
import scipy.signal as signal
from typing import Dict, Tuple, Optional, Any, List
import logging

# Try to import scikit-dsp-comm components
try:
    import sk_dsp_comm.digitalcom as dc
    import sk_dsp_comm.synchronization as sync
    import sk_dsp_comm.fsk as fsk
    SCIKIT_DSP_AVAILABLE = True
except ImportError:
    SCIKIT_DSP_AVAILABLE = False

logger = logging.getLogger(__name__)


class DemodulationEngine:
    """
    Multi-mode demodulation engine supporting various modulation schemes.
    """
    
    def __init__(self, sample_rate: float = 2e6):
        self.sample_rate = sample_rate
        self.demodulators = {}
        self._init_demodulators()
    
    def _init_demodulators(self):
        """Initialize demodulators for different modulation types."""
        self.demodulators = {
            "AM": AMDemodulator(self.sample_rate),
            "FM": FMDemodulator(self.sample_rate),
            "PSK": PSKDemodulator(self.sample_rate),
            "QPSK": QPSKDemodulator(self.sample_rate),
            "8PSK": PSK8Demodulator(self.sample_rate),
            "QAM16": QAM16Demodulator(self.sample_rate),
            "QAM64": QAM64Demodulator(self.sample_rate),
            "QAM256": QAM256Demodulator(self.sample_rate),
            "FSK": FSKDemodulator(self.sample_rate),
            "GFSK": GFSKDemodulator(self.sample_rate),
            "MSK": MSKDemodulator(self.sample_rate),
            "OFDM": OFDMDemodulator(self.sample_rate)
        }
    
    def demodulate(self, signal_data: np.ndarray, modulation_type: str, 
                   parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Demodulate signal based on detected modulation type.
        
        Args:
            signal_data: Complex IQ signal data
            modulation_type: Detected modulation type
            parameters: Modulation parameters from analyzer
            
        Returns:
            Dictionary containing demodulated data and metadata
        """
        if modulation_type not in self.demodulators:
            logger.warning(f"Unsupported modulation type: {modulation_type}")
            return {"success": False, "error": f"Unsupported modulation: {modulation_type}"}
        
        try:
            demodulator = self.demodulators[modulation_type]
            if parameters:
                demodulator.update_parameters(parameters)
            
            result = demodulator.demodulate(signal_data)
            result["modulation_type"] = modulation_type
            result["success"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Demodulation error for {modulation_type}: {e}")
            return {"success": False, "error": str(e)}


class BaseDemodulator:
    """Base class for all demodulators."""
    
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.symbol_rate = 1000.0
        self.center_frequency = 0.0
        
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update demodulator parameters."""
        if "symbol_rate" in parameters:
            self.symbol_rate = parameters["symbol_rate"]
        if "center_frequency" in parameters:
            self.center_frequency = parameters["center_frequency"]
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate signal data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement demodulate method")
    
    def _normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Normalize signal amplitude."""
        if np.std(signal_data) > 0:
            return signal_data / np.std(signal_data)
        return signal_data
    
    def _apply_agc(self, signal_data: np.ndarray, target_power: float = 1.0) -> np.ndarray:
        """Apply Automatic Gain Control."""
        try:
            # Simple AGC implementation
            power = np.mean(np.abs(signal_data) ** 2)
            if power > 0:
                gain = np.sqrt(target_power / power)
                return signal_data * gain
        except Exception as e:
            logger.warning(f"AGC error: {e}")
        
        return signal_data


class AMDemodulator(BaseDemodulator):
    """Amplitude Modulation demodulator."""
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate AM signal."""
        try:
            # Envelope detection
            amplitude = np.abs(signal_data)
            
            # Remove DC component
            demodulated = amplitude - np.mean(amplitude)
            
            # Low-pass filter to remove high-frequency components
            nyquist = self.sample_rate / 2
            cutoff = min(10000, nyquist * 0.4)  # 10 kHz or 40% of Nyquist
            sos = signal.butter(6, cutoff / nyquist, output='sos')
            demodulated = signal.sosfilt(sos, demodulated)
            
            return {
                "demodulated_data": demodulated,
                "sample_rate": self.sample_rate,
                "data_type": "audio",
                "bandwidth": cutoff,
                "modulation_depth": self._estimate_modulation_depth(amplitude)
            }
            
        except Exception as e:
            logger.error(f"AM demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    
    def _estimate_modulation_depth(self, amplitude: np.ndarray) -> float:
        """Estimate AM modulation depth."""
        try:
            max_amp = np.max(amplitude)
            min_amp = np.min(amplitude)
            if max_amp + min_amp > 0:
                return (max_amp - min_amp) / (max_amp + min_amp) * 100
        except Exception:
            pass
        return 0.0


class FMDemodulator(BaseDemodulator):
    """Frequency Modulation demodulator."""
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate FM signal."""
        try:
            # Frequency demodulation using phase derivative
            phase = np.angle(signal_data)
            unwrapped_phase = np.unwrap(phase)
            
            # Instantaneous frequency
            inst_freq = np.diff(unwrapped_phase) * self.sample_rate / (2 * np.pi)
            
            # Remove DC component
            demodulated = inst_freq - np.mean(inst_freq)
            
            # Low-pass filter for audio
            nyquist = self.sample_rate / 2
            cutoff = min(15000, nyquist * 0.4)  # 15 kHz for FM audio
            sos = signal.butter(6, cutoff / nyquist, output='sos')
            demodulated = signal.sosfilt(sos, demodulated)
            
            # De-emphasis filter (75 μs time constant for FM broadcast)
            tau = 75e-6  # 75 microseconds
            alpha = 1 / (1 + 2 * np.pi * tau * self.sample_rate)
            demodulated = signal.lfilter([alpha], [1, alpha - 1], demodulated)
            
            return {
                "demodulated_data": demodulated,
                "sample_rate": self.sample_rate,
                "data_type": "audio",
                "bandwidth": cutoff,
                "frequency_deviation": np.std(inst_freq)
            }
            
        except Exception as e:
            logger.error(f"FM demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}


class PSKDemodulator(BaseDemodulator):
    """Binary Phase Shift Keying demodulator."""
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate BPSK signal."""
        try:
            # Normalize signal
            signal_norm = self._normalize_signal(signal_data)
            
            # Symbol timing recovery (simplified)
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            
            # Matched filter (simple rectangular pulse)
            matched_filter = np.ones(samples_per_symbol) / samples_per_symbol
            filtered = np.convolve(signal_norm, matched_filter, mode='same')
            
            # Sample at symbol rate
            symbol_indices = np.arange(samples_per_symbol//2, len(filtered), samples_per_symbol)
            symbols = filtered[symbol_indices]
            
            # Phase detection for BPSK
            phases = np.angle(symbols)
            
            # Decision: 0 for negative real part, 1 for positive
            bits = (np.real(symbols) > 0).astype(int)
            
            return {
                "demodulated_data": bits,
                "symbols": symbols,
                "sample_rate": self.symbol_rate,
                "data_type": "digital",
                "constellation": symbols,
                "evm": self._calculate_evm(symbols, bits),
                "ber_estimate": self._estimate_ber(symbols, bits)
            }
            
        except Exception as e:
            logger.error(f"PSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    
    def _calculate_evm(self, received_symbols: np.ndarray, decided_bits: np.ndarray) -> float:
        """Calculate Error Vector Magnitude."""
        try:
            # Ideal BPSK constellation
            ideal_symbols = 2 * decided_bits - 1  # Map 0,1 to -1,1
            
            # Error vectors
            error_vectors = received_symbols - ideal_symbols
            
            # EVM calculation
            error_power = np.mean(np.abs(error_vectors) ** 2)
            signal_power = np.mean(np.abs(ideal_symbols) ** 2)
            
            if signal_power > 0:
                evm = np.sqrt(error_power / signal_power) * 100  # Percentage
                return float(evm)
        except Exception as e:
            logger.warning(f"EVM calculation error: {e}")
        
        return 0.0
    
    def _estimate_ber(self, symbols: np.ndarray, bits: np.ndarray) -> float:
        """Estimate Bit Error Rate."""
        try:
            # Simple BER estimation based on symbol reliability
            symbol_magnitudes = np.abs(np.real(symbols))
            threshold = 0.1  # Threshold for unreliable symbols
            
            unreliable_symbols = np.sum(symbol_magnitudes < threshold)
            total_symbols = len(symbols)
            
            estimated_ber = unreliable_symbols / total_symbols if total_symbols > 0 else 0.0
            return float(estimated_ber)
            
        except Exception as e:
            logger.warning(f"BER estimation error: {e}")
        
        return 0.0


class QPSKDemodulator(BaseDemodulator):
    """Quadrature Phase Shift Keying demodulator."""
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate QPSK signal."""
        try:
            # Normalize signal
            signal_norm = self._normalize_signal(signal_data)
            
            # Symbol timing recovery
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            
            # Matched filter
            matched_filter = np.ones(samples_per_symbol) / samples_per_symbol
            filtered = np.convolve(signal_norm, matched_filter, mode='same')
            
            # Sample at symbol rate
            symbol_indices = np.arange(samples_per_symbol//2, len(filtered), samples_per_symbol)
            symbols = filtered[symbol_indices]
            
            # QPSK decision
            bits = []
            for symbol in symbols:
                # Determine quadrant
                real_bit = 1 if np.real(symbol) > 0 else 0
                imag_bit = 1 if np.imag(symbol) > 0 else 0
                bits.extend([real_bit, imag_bit])
            
            return {
                "demodulated_data": np.array(bits),
                "symbols": symbols,
                "sample_rate": self.symbol_rate * 2,  # 2 bits per symbol
                "data_type": "digital",
                "constellation": symbols,
                "evm": self._calculate_qpsk_evm(symbols),
                "ber_estimate": self._estimate_ber(symbols)
            }
            
        except Exception as e:
            logger.error(f"QPSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    
    def _calculate_qpsk_evm(self, symbols: np.ndarray) -> float:
        """Calculate EVM for QPSK."""
        try:
            # Ideal QPSK constellation points
            ideal_points = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
            
            evm_sum = 0
            for symbol in symbols:
                # Find closest ideal point
                distances = np.abs(symbol - ideal_points)
                closest_ideal = ideal_points[np.argmin(distances)]
                
                # Error vector
                error = symbol - closest_ideal
                evm_sum += np.abs(error) ** 2
            
            # Average EVM
            if len(symbols) > 0:
                avg_evm = np.sqrt(evm_sum / len(symbols)) * 100
                return float(avg_evm)
                
        except Exception as e:
            logger.warning(f"QPSK EVM calculation error: {e}")
        
        return 0.0
    
    def _estimate_ber(self, symbols: np.ndarray) -> float:
        """Estimate BER for QPSK."""
        try:
            # Distance from ideal constellation points
            ideal_points = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
            
            total_errors = 0
            for symbol in symbols:
                distances = np.abs(symbol - ideal_points)
                min_distance = np.min(distances)
                
                # If symbol is far from any ideal point, consider it an error
                if min_distance > 0.5:  # Threshold
                    total_errors += 1
            
            ber = total_errors / len(symbols) if len(symbols) > 0 else 0.0
            return float(ber)
            
        except Exception as e:
            logger.warning(f"QPSK BER estimation error: {e}")
        
        return 0.0


class PSK8Demodulator(BaseDemodulator):
    """8-PSK demodulator."""
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate 8-PSK signal."""
        try:
            # Similar to QPSK but with 8 constellation points
            signal_norm = self._normalize_signal(signal_data)
            
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            matched_filter = np.ones(samples_per_symbol) / samples_per_symbol
            filtered = np.convolve(signal_norm, matched_filter, mode='same')
            
            symbol_indices = np.arange(samples_per_symbol//2, len(filtered), samples_per_symbol)
            symbols = filtered[symbol_indices]
            
            # 8-PSK constellation points
            ideal_points = np.exp(1j * 2 * np.pi * np.arange(8) / 8)
            
            bits = []
            for symbol in symbols:
                # Find closest constellation point
                distances = np.abs(symbol - ideal_points)
                closest_index = np.argmin(distances)
                
                # Convert to 3 bits
                bit_triplet = format(closest_index, '03b')
                bits.extend([int(b) for b in bit_triplet])
            
            return {
                "demodulated_data": np.array(bits),
                "symbols": symbols,
                "sample_rate": self.symbol_rate * 3,  # 3 bits per symbol
                "data_type": "digital",
                "constellation": symbols,
                "evm": self._calculate_psk8_evm(symbols),
                "ber_estimate": self._estimate_ber(symbols)
            }
            
        except Exception as e:
            logger.error(f"8-PSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    
    def _calculate_psk8_evm(self, symbols: np.ndarray) -> float:
        """Calculate EVM for 8-PSK."""
        ideal_points = np.exp(1j * 2 * np.pi * np.arange(8) / 8)
        
        try:
            evm_sum = 0
            for symbol in symbols:
                distances = np.abs(symbol - ideal_points)
                closest_ideal = ideal_points[np.argmin(distances)]
                error = symbol - closest_ideal
                evm_sum += np.abs(error) ** 2
            
            if len(symbols) > 0:
                avg_evm = np.sqrt(evm_sum / len(symbols)) * 100
                return float(avg_evm)
                
        except Exception as e:
            logger.warning(f"8-PSK EVM calculation error: {e}")
        
        return 0.0


class QAM16Demodulator(BaseDemodulator):
    """16-QAM demodulator."""
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate 16-QAM signal."""
        try:
            signal_norm = self._normalize_signal(signal_data)
            
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            matched_filter = np.ones(samples_per_symbol) / samples_per_symbol
            filtered = np.convolve(signal_norm, matched_filter, mode='same')
            
            symbol_indices = np.arange(samples_per_symbol//2, len(filtered), samples_per_symbol)
            symbols = filtered[symbol_indices]
            
            # 16-QAM constellation
            ideal_points = self._generate_qam16_constellation()
            
            bits = []
            for symbol in symbols:
                distances = np.abs(symbol - ideal_points)
                closest_index = np.argmin(distances)
                
                # Convert to 4 bits
                bit_quartet = format(closest_index, '04b')
                bits.extend([int(b) for b in bit_quartet])
            
            return {
                "demodulated_data": np.array(bits),
                "symbols": symbols,
                "sample_rate": self.symbol_rate * 4,  # 4 bits per symbol
                "data_type": "digital",
                "constellation": symbols,
                "evm": self._calculate_qam_evm(symbols, ideal_points),
                "ber_estimate": self._estimate_ber(symbols)
            }
            
        except Exception as e:
            logger.error(f"16-QAM demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    
    def _generate_qam16_constellation(self) -> np.ndarray:
        """Generate 16-QAM constellation points."""
        # Standard 16-QAM constellation
        points = []
        for i in [-3, -1, 1, 3]:
            for q in [-3, -1, 1, 3]:
                points.append(i + 1j * q)
        
        points = np.array(points)
        # Normalize average power to 1
        avg_power = np.mean(np.abs(points) ** 2)
        return points / np.sqrt(avg_power)
    
    def _calculate_qam_evm(self, symbols: np.ndarray, ideal_points: np.ndarray) -> float:
        """Calculate EVM for QAM."""
        try:
            evm_sum = 0
            for symbol in symbols:
                distances = np.abs(symbol - ideal_points)
                closest_ideal = ideal_points[np.argmin(distances)]
                error = symbol - closest_ideal
                evm_sum += np.abs(error) ** 2
            
            if len(symbols) > 0:
                avg_evm = np.sqrt(evm_sum / len(symbols)) * 100
                return float(avg_evm)
                
        except Exception as e:
            logger.warning(f"QAM EVM calculation error: {e}")
        
        return 0.0


class QAM64Demodulator(QAM16Demodulator):
    """64-QAM demodulator."""
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate 64-QAM signal."""
        try:
            signal_norm = self._normalize_signal(signal_data)
            
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            matched_filter = np.ones(samples_per_symbol) / samples_per_symbol
            filtered = np.convolve(signal_norm, matched_filter, mode='same')
            
            symbol_indices = np.arange(samples_per_symbol//2, len(filtered), samples_per_symbol)
            symbols = filtered[symbol_indices]
            
            ideal_points = self._generate_qam64_constellation()
            
            bits = []
            for symbol in symbols:
                distances = np.abs(symbol - ideal_points)
                closest_index = np.argmin(distances)
                
                # Convert to 6 bits
                bit_sextet = format(closest_index, '06b')
                bits.extend([int(b) for b in bit_sextet])
            
            return {
                "demodulated_data": np.array(bits),
                "symbols": symbols,
                "sample_rate": self.symbol_rate * 6,  # 6 bits per symbol
                "data_type": "digital",
                "constellation": symbols,
                "evm": self._calculate_qam_evm(symbols, ideal_points),
                "ber_estimate": self._estimate_ber(symbols)
            }
            
        except Exception as e:
            logger.error(f"64-QAM demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    
    def _generate_qam64_constellation(self) -> np.ndarray:
        """Generate 64-QAM constellation points."""
        points = []
        for i in [-7, -5, -3, -1, 1, 3, 5, 7]:
            for q in [-7, -5, -3, -1, 1, 3, 5, 7]:
                points.append(i + 1j * q)
        
        points = np.array(points)
        avg_power = np.mean(np.abs(points) ** 2)
        return points / np.sqrt(avg_power)


class QAM256Demodulator(QAM16Demodulator):
    """256-QAM demodulator."""
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate 256-QAM signal."""
        try:
            signal_norm = self._normalize_signal(signal_data)
            
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            matched_filter = np.ones(samples_per_symbol) / samples_per_symbol
            filtered = np.convolve(signal_norm, matched_filter, mode='same')
            
            symbol_indices = np.arange(samples_per_symbol//2, len(filtered), samples_per_symbol)
            symbols = filtered[symbol_indices]
            
            ideal_points = self._generate_qam256_constellation()
            
            bits = []
            for symbol in symbols:
                distances = np.abs(symbol - ideal_points)
                closest_index = np.argmin(distances)
                
                # Convert to 8 bits
                bit_octet = format(closest_index, '08b')
                bits.extend([int(b) for b in bit_octet])
            
            return {
                "demodulated_data": np.array(bits),
                "symbols": symbols,
                "sample_rate": self.symbol_rate * 8,  # 8 bits per symbol
                "data_type": "digital",
                "constellation": symbols,
                "evm": self._calculate_qam_evm(symbols, ideal_points),
                "ber_estimate": self._estimate_ber(symbols)
            }
            
        except Exception as e:
            logger.error(f"256-QAM demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    
    def _generate_qam256_constellation(self) -> np.ndarray:
        """Generate 256-QAM constellation points."""
        points = []
        for i in range(-15, 16, 2):
            for q in range(-15, 16, 2):
                points.append(i + 1j * q)
        
        points = np.array(points)
        avg_power = np.mean(np.abs(points) ** 2)
        return points / np.sqrt(avg_power)


class FSKDemodulator(BaseDemodulator):
    """Frequency Shift Keying demodulator."""
    
    def __init__(self, sample_rate: float):
        super().__init__(sample_rate)
        self.frequency_separation = 1000.0  # Hz
        self.mark_frequency = 1200.0  # Hz
        self.space_frequency = 2200.0  # Hz
    
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update FSK parameters."""
        super().update_parameters(parameters)
        if "frequency_deviation" in parameters:
            self.frequency_separation = parameters["frequency_deviation"] * 2
            self.mark_frequency = -parameters["frequency_deviation"]
            self.space_frequency = parameters["frequency_deviation"]
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate FSK signal."""
        try:
            # Non-coherent FSK demodulation using energy detection
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            
            # Generate local oscillators for mark and space frequencies
            t = np.arange(len(signal_data)) / self.sample_rate
            lo_mark = np.exp(-1j * 2 * np.pi * self.mark_frequency * t)
            lo_space = np.exp(-1j * 2 * np.pi * self.space_frequency * t)
            
            # Mix with local oscillators
            mark_mixed = signal_data * lo_mark
            space_mixed = signal_data * lo_space
            
            # Low-pass filter
            cutoff = self.symbol_rate / 2
            nyquist = self.sample_rate / 2
            sos = signal.butter(6, cutoff / nyquist, output='sos')
            
            mark_filtered = signal.sosfilt(sos, mark_mixed)
            space_filtered = signal.sosfilt(sos, space_mixed)
            
            # Energy detection
            mark_energy = np.abs(mark_filtered) ** 2
            space_energy = np.abs(space_filtered) ** 2
            
            # Symbol-rate sampling
            bits = []
            for i in range(0, len(mark_energy) - samples_per_symbol, samples_per_symbol):
                mark_sum = np.sum(mark_energy[i:i + samples_per_symbol])
                space_sum = np.sum(space_energy[i:i + samples_per_symbol])
                
                # Decision: mark = 1, space = 0
                bit = 1 if mark_sum > space_sum else 0
                bits.append(bit)
            
            return {
                "demodulated_data": np.array(bits),
                "sample_rate": self.symbol_rate,
                "data_type": "digital",
                "mark_frequency": self.mark_frequency,
                "space_frequency": self.space_frequency,
                "frequency_separation": self.frequency_separation
            }
            
        except Exception as e:
            logger.error(f"FSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}


class GFSKDemodulator(FSKDemodulator):
    """Gaussian Frequency Shift Keying demodulator."""
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate GFSK signal."""
        try:
            # GFSK is FSK with Gaussian pre-filtering
            # Use discriminator-based demodulation
            
            # Frequency discriminator
            delayed = np.concatenate([[0], signal_data[:-1]])
            discriminator_output = np.real(signal_data * np.conj(delayed))
            
            # Low-pass filter
            cutoff = self.symbol_rate
            nyquist = self.sample_rate / 2
            sos = signal.butter(6, cutoff / nyquist, output='sos')
            filtered = signal.sosfilt(sos, discriminator_output)
            
            # Symbol timing recovery and decision
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            
            bits = []
            for i in range(samples_per_symbol//2, len(filtered), samples_per_symbol):
                if i < len(filtered):
                    bit = 1 if filtered[i] > 0 else 0
                    bits.append(bit)
            
            return {
                "demodulated_data": np.array(bits),
                "sample_rate": self.symbol_rate,
                "data_type": "digital",
                "modulation_type": "GFSK"
            }
            
        except Exception as e:
            logger.error(f"GFSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}


class MSKDemodulator(BaseDemodulator):
    """Minimum Shift Keying demodulator."""
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate MSK signal."""
        try:
            # MSK can be demodulated as OQPSK with specific pulse shaping
            # Simplified approach using I/Q channels
            
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            
            # I and Q channel processing
            i_channel = np.real(signal_data)
            q_channel = np.imag(signal_data)
            
            # Correlate with reference waveforms
            # MSK has sinusoidal pulse shapes
            symbol_length = samples_per_symbol
            t_symbol = np.arange(symbol_length) / self.sample_rate
            
            # Reference waveforms for MSK
            cos_ref = np.cos(np.pi * t_symbol / (2 * symbol_length / self.sample_rate))
            sin_ref = np.sin(np.pi * t_symbol / (2 * symbol_length / self.sample_rate))
            
            bits = []
            for i in range(0, len(signal_data) - symbol_length, symbol_length):
                i_segment = i_channel[i:i + symbol_length]
                q_segment = q_channel[i:i + symbol_length]
                
                # Correlate with references
                i_corr = np.sum(i_segment * cos_ref)
                q_corr = np.sum(q_segment * sin_ref)
                
                # Make decisions
                i_bit = 1 if i_corr > 0 else 0
                q_bit = 1 if q_corr > 0 else 0
                
                bits.extend([i_bit, q_bit])
            
            return {
                "demodulated_data": np.array(bits),
                "sample_rate": self.symbol_rate * 2,
                "data_type": "digital",
                "modulation_type": "MSK"
            }
            
        except Exception as e:
            logger.error(f"MSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}


class OFDMDemodulator(BaseDemodulator):
    """OFDM demodulator (simplified)."""
    
    def __init__(self, sample_rate: float):
        super().__init__(sample_rate)
        self.num_subcarriers = 64
        self.cyclic_prefix_length = 16
        self.symbol_length = self.num_subcarriers + self.cyclic_prefix_length
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate OFDM signal."""
        try:
            # Simplified OFDM demodulation
            symbols_per_frame = len(signal_data) // self.symbol_length
            
            if symbols_per_frame == 0:
                return {"demodulated_data": np.array([]), "error": "Insufficient data for OFDM"}
            
            all_bits = []
            
            for sym_idx in range(symbols_per_frame):
                start_idx = sym_idx * self.symbol_length
                end_idx = start_idx + self.symbol_length
                
                if end_idx > len(signal_data):
                    break
                
                ofdm_symbol = signal_data[start_idx:end_idx]
                
                # Remove cyclic prefix
                data_part = ofdm_symbol[self.cyclic_prefix_length:]
                
                # FFT to get subcarriers
                fft_data = np.fft.fft(data_part, self.num_subcarriers)
                
                # Simple QPSK demodulation on each subcarrier
                for subcarrier in fft_data:
                    # QPSK decision
                    real_bit = 1 if np.real(subcarrier) > 0 else 0
                    imag_bit = 1 if np.imag(subcarrier) > 0 else 0
                    all_bits.extend([real_bit, imag_bit])
            
            return {
                "demodulated_data": np.array(all_bits),
                "sample_rate": self.symbol_rate * 2 * self.num_subcarriers,
                "data_type": "digital",
                "num_subcarriers": self.num_subcarriers,
                "cyclic_prefix_length": self.cyclic_prefix_length
            }
            
        except Exception as e:
            logger.error(f"OFDM demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}


def create_demodulation_engine(sample_rate: float = 2e6) -> DemodulationEngine:
    """Factory function to create demodulation engine."""
    return DemodulationEngine(sample_rate)