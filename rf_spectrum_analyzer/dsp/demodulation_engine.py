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
    import sk_dsp_comm.fec_conv as fec
    import sk_dsp_comm.sigsys as ss
    import sk_dsp_comm.fir_design_helper as fir
    import sk_dsp_comm.multirate_helper as mr
    SCIKIT_DSP_AVAILABLE = True
except ImportError:
    SCIKIT_DSP_AVAILABLE = False

# Try to import sdr library components
try:
    import sdr
    SDR_AVAILABLE = True
except ImportError:
    SDR_AVAILABLE = False

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
            "BPSK": PSKDemodulator(self.sample_rate),  # BPSK is the same as PSK
            "QPSK": QPSKDemodulator(self.sample_rate),
            "PSK8": PSK8Demodulator(self.sample_rate),  # Add PSK8 mapping
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
    
    def _ensure_numpy_array(self, data: Any) -> np.ndarray:
        """
        Ensure data is a proper numpy array, handling various return types from 
        different signal processing libraries.
        
        Args:
            data: Input data (could be tuple, list, array, complex array, etc.)
            
        Returns:
            numpy array with consistent data type
        """
        try:
            # Handle tuple returns (from sdr library)
            if isinstance(data, tuple):
                if len(data) == 0:
                    return np.array([])
                # Take the first element if it's a data tuple
                data = data[0]
            
            # Convert to numpy array
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Handle complex data - convert to real if all imaginary parts are zero
            if data.dtype == np.complex128 or data.dtype == np.complex64:
                if np.allclose(np.imag(data), 0):
                    data = np.real(data)
            
            # Ensure 1D array
            if data.ndim > 1:
                data = data.flatten()
            
            # Convert to appropriate integer type for digital data
            if data.dtype == np.float64 or data.dtype == np.float32:
                # Threshold floating point data to binary for digital signals
                if np.all((data >= 0) & (data <= 1)):
                    data = (data > 0.5).astype(int)
                elif np.all((data >= -1) & (data <= 1)):
                    data = (data > 0).astype(int)
            
            return data
            
        except Exception as e:
            logger.warning(f"Data type conversion error: {e}, returning empty array")
            return np.array([])
    
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

    def _estimate_cfo_hz(self, signal_data: np.ndarray) -> float:
        """Estimate carrier frequency offset from phase progression."""
        try:
            if signal_data is None or len(signal_data) < 2:
                return 0.0
            phase = np.unwrap(np.angle(signal_data))
            freq = np.diff(phase) * self.sample_rate / (2 * np.pi)
            if len(freq) == 0:
                return 0.0
            return float(np.median(freq))
        except Exception as e:
            logger.warning(f"CFO estimation error: {e}")
            return 0.0

    def _build_sync_quality(
        self,
        signal_data: np.ndarray,
        symbols: np.ndarray,
        samples_per_symbol: int,
        evm: Optional[float] = None,
        ber: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build stable sync/carrier lock telemetry for digital demodulators."""
        try:
            residual_symbol_fraction = 0.0
            if samples_per_symbol > 0 and signal_data is not None:
                residual_symbol_fraction = float((len(signal_data) % samples_per_symbol) / samples_per_symbol)

            cfo_hz = self._estimate_cfo_hz(signal_data)
            carrier_lock = bool(abs(cfo_hz) <= max(250.0, self.sample_rate * 0.0025))
            timing_lock = bool(residual_symbol_fraction <= 0.5 and symbols is not None and len(symbols) > 0)

            evm_value = float(evm) if evm is not None else 50.0
            ber_value = float(ber) if ber is not None else 0.5
            lock_confidence = 1.0
            lock_confidence -= min(0.45, abs(cfo_hz) / max(self.sample_rate, 1.0))
            lock_confidence -= min(0.35, evm_value / 200.0)
            lock_confidence -= min(0.20, ber_value)
            if timing_lock:
                lock_confidence += 0.1
            lock_confidence = float(max(0.0, min(1.0, lock_confidence)))

            snr_db = None
            if evm_value > 0:
                snr_db = float(20.0 * np.log10(100.0 / max(evm_value, 1e-6)))

            return {
                'cfo_hz': cfo_hz,
                'timing_error_rms': residual_symbol_fraction,
                'carrier_lock': carrier_lock,
                'timing_lock': timing_lock,
                'lock_confidence': lock_confidence,
                'snr_db': snr_db,
                'evm': evm_value,
                'ber_estimate': ber_value,
            }
        except Exception as e:
            logger.warning(f"Sync quality build error: {e}")
            return {
                'cfo_hz': 0.0,
                'timing_error_rms': 1.0,
                'carrier_lock': False,
                'timing_lock': False,
                'lock_confidence': 0.0,
                'snr_db': None,
                'evm': float(evm) if evm is not None else 50.0,
                'ber_estimate': float(ber) if ber is not None else 0.5,
            }


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
        """Demodulate BPSK signal using advanced libraries."""
        try:
            if SDR_AVAILABLE:
                return self._demodulate_with_sdr(signal_data)
            elif SCIKIT_DSP_AVAILABLE:
                return self._demodulate_with_scikit(signal_data)
            else:
                return self._demodulate_basic(signal_data)
                
        except Exception as e:
            logger.error(f"PSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    
    def _demodulate_with_sdr(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate BPSK using sdr library."""
        try:
            # Normalize signal
            signal_norm = self._normalize_signal(signal_data)
            
            # Create PSK modulator for reference constellation
            M = 2  # BPSK
            constellation = sdr.PSK(M)
            
            # Symbol timing recovery
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            
            # Pulse shaping (Root Raised Cosine)
            h_rrc = sdr.root_raised_cosine(0.35, samples_per_symbol, 10)
            
            # Matched filtering
            matched_filtered = np.convolve(signal_norm, np.conj(h_rrc[::-1]), mode='same')
            
            # Timing recovery using Mueller & Muller
            recovered_symbols = self._timing_recovery_mm(matched_filtered, samples_per_symbol)
            
            # Phase recovery using Costas loop
            phase_recovered = self._costas_loop(recovered_symbols, M)
            
            # Symbol decision with proper error handling
            try:
                # Try sdr library methods
                if hasattr(constellation, 'decide'):
                    symbols = constellation.decide(phase_recovered)
                    bits = constellation.demodulate(symbols)
                elif hasattr(constellation, 'demodulate'):
                    bits = constellation.demodulate(phase_recovered)
                    symbols = self._bpsk_decision(phase_recovered)
                else:
                    # Manual BPSK decision
                    symbols = self._bpsk_decision(phase_recovered)
                    bits = self._bpsk_to_bits(symbols)
                    
                # Ensure proper data types
                bits = self._ensure_numpy_array(bits)
                symbols = self._ensure_numpy_array(symbols)
                
            except Exception:
                # Fallback to manual BPSK decision
                symbols = self._bpsk_decision(phase_recovered)
                bits = self._bpsk_to_bits(symbols)
            
            # Calculate metrics
            evm = self._calculate_evm_advanced(phase_recovered, symbols)
            ber = self._estimate_ber_advanced(phase_recovered, symbols)
            sync_quality = self._build_sync_quality(signal_norm, phase_recovered, samples_per_symbol, evm, ber)
            
            return {
                "demodulated_data": bits,
                "symbols": symbols,
                "constellation_points": phase_recovered,
                "sample_rate": self.symbol_rate,
                "data_type": "digital",
                "evm": evm,
                "ber_estimate": ber,
                "snr_db": sync_quality['snr_db'],
                "snr": sync_quality['snr_db'],
                "cfo_hz": sync_quality['cfo_hz'],
                "timing_error_rms": sync_quality['timing_error_rms'],
                "carrier_lock": sync_quality['carrier_lock'],
                "timing_lock": sync_quality['timing_lock'],
                "lock_confidence": sync_quality['lock_confidence'],
                "quality_metrics": sync_quality,
            }
            
        except Exception as e:
            logger.warning(f"SDR library demodulation failed: {e}")
            return self._demodulate_basic(signal_data)
    
    def _demodulate_with_scikit(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate BPSK using scikit-dsp-comm."""
        try:
            # Normalize signal
            signal_norm = self._normalize_signal(signal_data)
            
            # Symbol timing recovery
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            
            # Create matched filter using FIR design helper
            h_matched = fir.firwin_kaiser_lpf(samples_per_symbol, 1/(2*samples_per_symbol), 60)
            
            # Apply matched filter
            filtered = signal.lfilter(h_matched, 1, signal_norm)
            
            # Timing recovery using scikit-dsp-comm
            recovered_signal, timing_error = sync.NDA_symb_sync(filtered, samples_per_symbol, 0.01, 2.0)
            
            # Phase recovery for BPSK
            phase_recovered = sync.DD_carrier_sync(recovered_signal, 2, 0.01, 2.0)
            
            # Symbol decision
            symbols = np.sign(np.real(phase_recovered))
            bits = (symbols > 0).astype(int)
            
            # Calculate EVM and BER
            evm = np.sqrt(np.mean(np.abs(phase_recovered - symbols)**2))
            ber = np.mean(bits != (np.real(phase_recovered) > 0))
            sync_quality = self._build_sync_quality(signal_norm, phase_recovered, samples_per_symbol, evm, ber)
            
            return {
                "demodulated_data": bits,
                "symbols": symbols,
                "constellation_points": phase_recovered,
                "sample_rate": self.symbol_rate,
                "data_type": "digital",
                "evm": evm,
                "ber_estimate": ber,
                "timing_error": timing_error,
                "snr_db": sync_quality['snr_db'],
                "snr": sync_quality['snr_db'],
                "cfo_hz": sync_quality['cfo_hz'],
                "timing_error_rms": sync_quality['timing_error_rms'],
                "carrier_lock": sync_quality['carrier_lock'],
                "timing_lock": sync_quality['timing_lock'],
                "lock_confidence": sync_quality['lock_confidence'],
                "quality_metrics": sync_quality,
            }
            
        except Exception as e:
            logger.warning(f"Scikit-DSP-Comm demodulation failed: {e}")
            return self._demodulate_basic(signal_data)
    
    def _demodulate_basic(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Basic BPSK demodulation fallback."""
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
            sync_quality = self._build_sync_quality(signal_norm, symbols, samples_per_symbol, self._calculate_evm(symbols, bits), self._estimate_ber(symbols, bits))
            
            return {
                "demodulated_data": bits,
                "symbols": symbols,
                "sample_rate": self.symbol_rate,
                "data_type": "digital",
                "constellation": symbols,
                "evm": self._calculate_evm(symbols, bits),
                "ber_estimate": self._estimate_ber(symbols, bits),
                "snr_db": sync_quality['snr_db'],
                "snr": sync_quality['snr_db'],
                "cfo_hz": sync_quality['cfo_hz'],
                "timing_error_rms": sync_quality['timing_error_rms'],
                "carrier_lock": sync_quality['carrier_lock'],
                "timing_lock": sync_quality['timing_lock'],
                "lock_confidence": sync_quality['lock_confidence'],
                "quality_metrics": sync_quality,
            }
            
        except Exception as e:
            logger.error(f"Basic PSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    
    def _timing_recovery_mm(self, signal_data: np.ndarray, sps: int) -> np.ndarray:
        """Mueller & Muller timing recovery."""
        try:
            # Simplified M&M algorithm
            mu = 0.0  # Timing offset
            out = []
            out_rail = []
            
            for i in range(len(signal_data)):
                if i % sps == 0:
                    out.append(signal_data[i])
                    # Simplified timing error detector
                    if len(out) > 1:
                        timing_error = np.real(out[-1] * np.conj(out[-2]))
                        mu += 0.01 * timing_error  # Update timing
            
            return np.array(out)
            
        except Exception:
            # Fallback to simple downsampling
            return signal_data[::sps]
    
    def _costas_loop(self, signal_data: np.ndarray, M: int) -> np.ndarray:
        """Costas loop for phase recovery."""
        try:
            phase = 0.0
            alpha = 0.01  # Loop gain
            output = np.zeros_like(signal_data)
            
            for i, sample in enumerate(signal_data):
                # Apply phase correction
                corrected = sample * np.exp(-1j * phase)
                output[i] = corrected
                
                # Phase error detector for BPSK (M=2)
                if M == 2:
                    error = np.imag(corrected) * np.sign(np.real(corrected))
                else:
                    error = np.imag(corrected * np.conj(corrected)**(M-1))
                
                # Update phase
                phase += alpha * error
                
            return output
            
        except Exception:
            return signal_data
            
        except Exception as e:
            logger.error(f"PSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    def _calculate_evm_advanced(self, received_symbols: np.ndarray, ideal_symbols: np.ndarray) -> float:
        """Calculate advanced Error Vector Magnitude with improved accuracy."""
        try:
            if len(received_symbols) == 0 or len(ideal_symbols) == 0:
                return 0.0
            
            # Ensure same length
            min_len = min(len(received_symbols), len(ideal_symbols))
            received = received_symbols[:min_len]
            ideal = ideal_symbols[:min_len]
            
            # Normalize both signals to have same average power
            received_norm = received / np.sqrt(np.mean(np.abs(received)**2))
            ideal_norm = ideal / np.sqrt(np.mean(np.abs(ideal)**2))
            
            # Calculate error vector
            error_vector = received_norm - ideal_norm
            
            # EVM as percentage
            error_power = np.mean(np.abs(error_vector)**2)
            signal_power = np.mean(np.abs(ideal_norm)**2)
            
            if signal_power > 0:
                evm_percent = np.sqrt(error_power / signal_power) * 100
                return float(min(evm_percent, 100.0))  # Cap at 100%
                
        except Exception as e:
            logger.warning(f"Advanced EVM calculation error: {e}")
        
        return 50.0  # Default moderate EVM
    
    def _estimate_ber_advanced(self, received_symbols: np.ndarray, ideal_symbols: np.ndarray) -> float:
        """Estimate advanced Bit Error Rate with improved accuracy."""
        try:
            if len(received_symbols) == 0 or len(ideal_symbols) == 0:
                return 0.0
            
            # Ensure same length
            min_len = min(len(received_symbols), len(ideal_symbols))
            received = received_symbols[:min_len]
            ideal = ideal_symbols[:min_len]
            
            # Convert symbols to bits for comparison
            received_bits = (np.real(received) > 0).astype(int)
            ideal_bits = (np.real(ideal) > 0).astype(int)
            
            errors = np.sum(received_bits != ideal_bits)
            total_bits = len(received_bits)
            
            return float(errors / total_bits) if total_bits > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Advanced BER estimation error: {e}")
        
        return 0.0
    
    def _calculate_evm(self, received_symbols: np.ndarray, decided_bits: np.ndarray) -> float:
        """Calculate Error Vector Magnitude with improved noise handling."""
        try:
            if len(received_symbols) == 0 or len(decided_bits) == 0:
                return 50.0
                
            # Create ideal BPSK constellation points
            ideal_symbols = 2 * decided_bits.astype(float) - 1  # Map 0,1 to -1,1
            
            # Ensure same length
            min_len = min(len(received_symbols), len(ideal_symbols))
            received = received_symbols[:min_len]
            ideal = ideal_symbols[:min_len]
            
            # Normalize signals for fair comparison
            received_power = np.mean(np.abs(received)**2)
            ideal_power = np.mean(np.abs(ideal)**2)
            
            if received_power > 0 and ideal_power > 0:
                # Scale received to match ideal power
                scale_factor = np.sqrt(ideal_power / received_power)
                received_scaled = received * scale_factor
                
                # Calculate error
                error_vectors = received_scaled - ideal
                error_power = np.mean(np.abs(error_vectors)**2)
                
                # EVM as percentage
                evm_percent = np.sqrt(error_power / ideal_power) * 100
                return float(min(evm_percent, 150.0))  # Cap at 150%
            
        except Exception as e:
            logger.warning(f"EVM calculation error: {e}")
        
        return 50.0  # Default moderate EVM
    
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
    
    def _bpsk_decision(self, symbols: np.ndarray) -> np.ndarray:
        """Make BPSK symbol decisions."""
        try:
            # BPSK decision: positive real = +1, negative real = -1
            decisions = np.sign(np.real(symbols))
            # Ensure no zeros (handle case where real part is exactly 0)
            decisions[decisions == 0] = 1
            return decisions
        except Exception as e:
            logger.warning(f"BPSK decision error: {e}")
            return np.ones(len(symbols))  # Fallback to all +1
    
    def _bpsk_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """Convert BPSK symbols to bits."""
        try:
            # Map +1 to bit 1, -1 to bit 0
            bits = (symbols > 0).astype(int)
            return bits
        except Exception as e:
            logger.warning(f"BPSK to bits conversion error: {e}")
            return np.zeros(len(symbols), dtype=int)  # Fallback to all 0s


class QPSKDemodulator(BaseDemodulator):
    """Quadrature Phase Shift Keying demodulator."""
    
    def _timing_recovery_mm(self, signal_data: np.ndarray, sps: int) -> np.ndarray:
        """Mueller & Muller timing recovery."""
        try:
            # Simplified M&M algorithm
            mu = 0.0  # Timing offset
            out = []
            
            for i in range(len(signal_data)):
                if i % sps == 0:
                    out.append(signal_data[i])
                    # Simplified timing error detector
                    if len(out) > 1:
                        timing_error = np.real(out[-1] * np.conj(out[-2]))
                        mu += 0.01 * timing_error  # Update timing
            
            return np.array(out) if out else np.array([])
            
        except Exception:
            # Fallback to simple downsampling
            return signal_data[::sps] if len(signal_data) >= sps else signal_data
    
    def _costas_loop(self, signal_data: np.ndarray, M: int) -> np.ndarray:
        """Costas loop for phase recovery."""
        try:
            phase = 0.0
            alpha = 0.01  # Loop gain
            output = np.zeros_like(signal_data)
            
            for i, sample in enumerate(signal_data):
                # Apply phase correction
                corrected = sample * np.exp(-1j * phase)
                output[i] = corrected
                
                # Phase error detector for QPSK (M=4)
                if M == 4:
                    error = np.imag(corrected**3) * np.real(corrected)
                else:
                    error = np.imag(corrected * np.conj(corrected)**(M-1))
                
                # Update phase
                phase += alpha * error
            
            return output
            
        except Exception:
            return signal_data
    
    def demodulate(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate QPSK signal using advanced libraries."""
        try:
            if SDR_AVAILABLE:
                return self._demodulate_qpsk_with_sdr(signal_data)
            elif SCIKIT_DSP_AVAILABLE:
                return self._demodulate_qpsk_with_scikit(signal_data)
            else:
                return self._demodulate_qpsk_basic(signal_data)
                
        except Exception as e:
            logger.error(f"QPSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update demodulator parameters."""
        if 'symbol_rate' in parameters:
            # Use estimated symbol rate but with reasonable bounds
            estimated_rate = parameters['symbol_rate']
            # Clamp to reasonable range (1kHz to 1MHz)
            self.symbol_rate = max(1000, min(1e6, estimated_rate))
            logger.debug(f"Updated QPSK symbol rate to {self.symbol_rate} Hz")
    
    def _demodulate_qpsk_with_sdr(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate QPSK using sdr library."""
        try:
            # Normalize signal
            signal_norm = self._normalize_signal(signal_data)
            
            # Create QPSK modulator for reference
            M = 4  # QPSK
            constellation = sdr.PSK(M)
            
            # Symbol timing recovery
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            
            # Root Raised Cosine filter
            h_rrc = sdr.root_raised_cosine(0.35, samples_per_symbol, 10)
            
            # Matched filtering
            matched_filtered = np.convolve(signal_norm, np.conj(h_rrc[::-1]), mode='same')
            
            # Timing recovery
            recovered_symbols = self._timing_recovery_mm(matched_filtered, samples_per_symbol)
            
            # Phase recovery using Costas loop for QPSK
            phase_recovered = self._costas_loop(recovered_symbols, M)
            
            # Symbol decision
            try:
                # Try sdr library method
                if hasattr(constellation, 'decide'):
                    decided_symbols = constellation.decide(phase_recovered)
                    bits = constellation.demodulate(decided_symbols)
                elif hasattr(constellation, 'demodulate'):
                    bits = constellation.demodulate(phase_recovered)
                    decided_symbols = self._qpsk_decision(phase_recovered)
                else:
                    # Manual QPSK decision
                    decided_symbols = self._qpsk_decision(phase_recovered)
                    bits = self._qpsk_to_bits(decided_symbols)
                    
                # Ensure proper data types
                bits = self._ensure_numpy_array(bits)
                decided_symbols = self._ensure_numpy_array(decided_symbols)
                
            except Exception:
                # Fallback to manual decision
                decided_symbols = self._qpsk_decision(phase_recovered)
                bits = self._qpsk_to_bits(decided_symbols)
            
            # Ensure numpy arrays - handle complex data properly
            if not isinstance(bits, np.ndarray):
                if hasattr(bits, '__iter__') and any(isinstance(x, complex) for x in bits):
                    # Handle complex list/array - convert to real first
                    real_bits = np.real(np.array(bits)).astype(int)
                    bits = real_bits
                else:
                    bits = np.array(bits, dtype=int)
            elif bits.dtype == complex:
                # Convert complex to real by taking real part
                bits = np.real(bits).astype(int)
            
            # Ensure bits is 1D array
            if bits.ndim > 1:
                bits = bits.flatten()
            
            decided_symbols = np.array(decided_symbols) if not isinstance(decided_symbols, np.ndarray) else decided_symbols
            phase_recovered = np.array(phase_recovered) if not isinstance(phase_recovered, np.ndarray) else phase_recovered
            
            # Ensure symbols are 1D
            if decided_symbols.ndim > 1:
                decided_symbols = decided_symbols.flatten()
            if phase_recovered.ndim > 1:
                phase_recovered = phase_recovered.flatten()
            
            # Ensure bits is numpy array
            if not isinstance(bits, np.ndarray):
                bits = np.array(bits) if bits is not None else np.array([])
            if not isinstance(decided_symbols, np.ndarray):
                decided_symbols = np.array(decided_symbols) if decided_symbols is not None else np.array([])
            
            # Calculate metrics
            evm = self._calculate_evm_advanced(phase_recovered, decided_symbols)
            ber = self._estimate_ber_qpsk(phase_recovered, decided_symbols)
            
            return {
                "demodulated_data": bits,
                "symbols": decided_symbols,
                "constellation_points": phase_recovered,
                "sample_rate": self.symbol_rate * 2,  # 2 bits per symbol
                "data_type": "digital",
                "evm": evm,
                "ber_estimate": ber,
                "snr_db": -20 * np.log10(evm) if evm > 0 else float('inf')
            }
            
        except Exception as e:
            logger.warning(f"SDR QPSK demodulation failed: {e}")
            return self._demodulate_qpsk_basic(signal_data)
    
    def _demodulate_qpsk_with_scikit(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate QPSK using scikit-dsp-comm."""
        try:
            # Normalize signal
            signal_norm = self._normalize_signal(signal_data)
            
            # Symbol timing recovery
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            
            # Create matched filter
            h_matched = fir.firwin_kaiser_lpf(samples_per_symbol, 1/(2*samples_per_symbol), 60)
            
            # Apply matched filter
            filtered = signal.lfilter(h_matched, 1, signal_norm)
            
            # Timing recovery
            recovered_signal, timing_error = sync.NDA_symb_sync(filtered, samples_per_symbol, 0.01, 2.0)
            
            # Carrier recovery for QPSK
            phase_recovered = sync.DD_carrier_sync(recovered_signal, 4, 0.01, 2.0)
            
            # QPSK symbol decision
            decided_symbols = self._qpsk_decision(phase_recovered)
            bits = self._qpsk_to_bits(decided_symbols)
            
            # Calculate metrics
            evm = np.sqrt(np.mean(np.abs(phase_recovered - decided_symbols)**2))
            ber = self._estimate_ber_qpsk(phase_recovered, decided_symbols)
            
            return {
                "demodulated_data": bits,
                "symbols": decided_symbols,
                "constellation_points": phase_recovered,
                "sample_rate": self.symbol_rate * 2,
                "data_type": "digital",
                "evm": evm,
                "ber_estimate": ber,
                "timing_error": timing_error
            }
            
        except Exception as e:
            logger.warning(f"Scikit QPSK demodulation failed: {e}")
            return self._demodulate_qpsk_basic(signal_data)
    
    def _demodulate_qpsk_basic(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Basic QPSK demodulation fallback."""
        try:
            # Normalize signal
            signal_norm = self._normalize_signal(signal_data)
            
            # Calculate symbol timing parameters
            samples_per_symbol = max(1, int(self.sample_rate / self.symbol_rate))
            
            # Matched filter (simple rectangular)
            if samples_per_symbol > 1:
                matched_filter = np.ones(samples_per_symbol) / np.sqrt(samples_per_symbol)
                filtered = np.convolve(signal_norm, matched_filter, mode='same')
            else:
                filtered = signal_norm
            
            # Sample at symbol rate with better timing
            # Start from a good offset to avoid transients
            start_offset = samples_per_symbol // 2
            symbol_indices = np.arange(start_offset, len(filtered), samples_per_symbol)
            
            # Ensure we don't go out of bounds
            symbol_indices = symbol_indices[symbol_indices < len(filtered)]
            symbols = filtered[symbol_indices]
            
            # Only proceed if we have enough symbols
            if len(symbols) < 2:
                self.logger.warning(f"Too few symbols recovered: {len(symbols)}")
                return {"demodulated_data": np.array([]), "data_type": "digital", "error": "Too few symbols"}
            
            # QPSK decision and bit extraction
            bits = []
            decided_symbols = []
            
            for symbol in symbols:
                # Make QPSK decision
                real_part = 1 if np.real(symbol) > 0 else -1
                imag_part = 1 if np.imag(symbol) > 0 else -1
                decided_symbol = (real_part + 1j * imag_part) / np.sqrt(2)
                decided_symbols.append(decided_symbol)
                
                # Convert to bits (Gray coding)
                if np.real(symbol) > 0 and np.imag(symbol) > 0:  # 00
                    bits.extend([0, 0])
                elif np.real(symbol) < 0 and np.imag(symbol) > 0:  # 01
                    bits.extend([0, 1])
                elif np.real(symbol) < 0 and np.imag(symbol) < 0:  # 11
                    bits.extend([1, 1])
                else:  # 10 - real > 0, imag < 0
                    bits.extend([1, 0])
            
            # Convert to numpy arrays
            bits_array = np.array(bits, dtype=np.uint8)
            symbols_array = np.array(symbols)
            decided_symbols_array = np.array(decided_symbols)

            # Calculate performance metrics
            evm = self._calculate_qpsk_evm(decided_symbols_array)
            ber = self._estimate_ber_qpsk(symbols_array, decided_symbols_array)

            self.logger.debug(f"QPSK Basic demod: {len(symbols)} symbols -> {len(bits)} bits")

            return {
                "demodulated_data": bits_array,
                "symbols": decided_symbols_array,
                "constellation_points": symbols_array,
                "sample_rate": self.symbol_rate * 2,  # 2 bits per symbol
                "data_type": "digital",
                "evm": evm,
                "ber_estimate": ber,
                "snr_db": -20 * np.log10(evm/100) if evm > 0 else float('inf'),
                "timing_offset": start_offset
            }
            
        except Exception as e:
            logger.error(f"Basic QPSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "data_type": "digital", "error": str(e)}
    
    def _qpsk_decision(self, symbols: np.ndarray) -> np.ndarray:
        """Make QPSK symbol decisions."""
        # QPSK constellation points: [1+1j, -1+1j, -1-1j, 1-1j] / sqrt(2)
        decided = np.zeros_like(symbols)
        
        for i, symbol in enumerate(symbols):
            real_part = 1 if np.real(symbol) > 0 else -1
            imag_part = 1 if np.imag(symbol) > 0 else -1
            decided[i] = (real_part + 1j * imag_part) / np.sqrt(2)
        
        return decided
    
    def _qpsk_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """Convert QPSK symbols to bits."""
        bits = []
        for symbol in symbols:
            # Gray coding for QPSK
            if np.real(symbol) > 0 and np.imag(symbol) > 0:  # 00
                bits.extend([0, 0])
            elif np.real(symbol) < 0 and np.imag(symbol) > 0:  # 01
                bits.extend([0, 1])
            elif np.real(symbol) < 0 and np.imag(symbol) < 0:  # 11
                bits.extend([1, 1])
            else:  # 10
                bits.extend([1, 0])
        
        return np.array(bits)
    
    def _estimate_ber_qpsk(self, received_symbols: np.ndarray, ideal_symbols: np.ndarray) -> float:
        """Estimate BER for QPSK."""
        try:
            if len(received_symbols) != len(ideal_symbols):
                return 0.0
            
            received_bits = self._qpsk_to_bits(self._qpsk_decision(received_symbols))
            ideal_bits = self._qpsk_to_bits(ideal_symbols)
            
            errors = np.sum(received_bits != ideal_bits)
            total_bits = len(received_bits)
            
            return errors / total_bits if total_bits > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"QPSK BER estimation error: {e}")
            return 0.0
    
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
            return 0.0
                
        except Exception as e:
            logger.warning(f"QPSK EVM calculation error: {e}")
            return 0.0
    
    def _calculate_evm_advanced(self, received_symbols: np.ndarray, ideal_symbols: np.ndarray) -> float:
        """Calculate advanced EVM."""
        try:
            if len(received_symbols) == 0 or len(ideal_symbols) == 0:
                return 0.0
            
            min_len = min(len(received_symbols), len(ideal_symbols))
            received = received_symbols[:min_len]
            ideal = ideal_symbols[:min_len]
            
            error_vector = received - ideal
            signal_power = np.mean(np.abs(ideal) ** 2)
            error_power = np.mean(np.abs(error_vector) ** 2)
            
            evm = np.sqrt(error_power / signal_power) * 100 if signal_power > 0 else 0.0
            return float(evm)
            
        except Exception as e:
            logger.warning(f"Advanced EVM calculation error: {e}")
            return 0.0
    
    def _estimate_ber(self, symbols: np.ndarray) -> float:
        """Estimate basic BER."""
        try:
            if len(symbols) == 0:
                return 0.0
            
            # Simple BER estimation based on EVM
            evm = self._calculate_qpsk_evm(symbols) / 100.0  # Convert to ratio
            
            # Rough BER approximation for QPSK
            if evm > 0:
                ber = 0.5 * np.exp(-1 / (2 * evm**2))
                return float(ber)
            return 0.0
            
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

            evm = self._calculate_psk8_evm(symbols)
            ber = self._estimate_ber(symbols)
            sync_quality = self._build_sync_quality(signal_norm, symbols, samples_per_symbol, evm, ber)
            
            return {
                "demodulated_data": np.array(bits),
                "symbols": symbols,
                "sample_rate": self.symbol_rate * 3,  # 3 bits per symbol
                "data_type": "digital",
                "constellation": symbols,
                "evm": evm,
                "ber_estimate": ber,
                "snr_db": sync_quality['snr_db'],
                "snr": sync_quality['snr_db'],
                "cfo_hz": sync_quality['cfo_hz'],
                "timing_error_rms": sync_quality['timing_error_rms'],
                "carrier_lock": sync_quality['carrier_lock'],
                "timing_lock": sync_quality['timing_lock'],
                "lock_confidence": sync_quality['lock_confidence'],
                "quality_metrics": sync_quality,
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

    def _estimate_ber(self, symbols: np.ndarray) -> float:
        """Estimate BER for 8-PSK using nearest-constellation decisions."""
        try:
            if symbols is None or len(symbols) == 0:
                return 0.0

            ideal_points = np.exp(1j * 2 * np.pi * np.arange(8) / 8)
            bit_errors = 0
            total_bits = 0

            for symbol in symbols:
                distances = np.abs(symbol - ideal_points)
                closest_index = int(np.argmin(distances))
                error_distance = float(np.abs(symbol - ideal_points[closest_index]))
                if error_distance > 0.5:
                    bit_errors += 1
                total_bits += 3

            if total_bits > 0:
                return float(bit_errors / total_bits)
        except Exception as e:
            logger.warning(f"8-PSK BER estimation error: {e}")

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
    
    def _estimate_ber(self, symbols: np.ndarray) -> float:
        """Estimate Bit Error Rate for QAM."""
        try:
            if len(symbols) == 0:
                return 0.0
            
            # Generate ideal constellation
            ideal_points = self._generate_qam16_constellation()
            
            error_count = 0
            total_bits = 0
            
            for symbol in symbols:
                # Find closest ideal point
                distances = np.abs(symbol - ideal_points)
                closest_index = np.argmin(distances)
                closest_ideal = ideal_points[closest_index]
                
                # Simple error estimation based on distance
                error_distance = np.abs(symbol - closest_ideal)
                
                # If distance is large, assume bit errors
                if error_distance > 0.5:  # Threshold for error detection
                    error_count += 1  # Assume 1 bit error per symbol for simplicity
                
                total_bits += 4  # 4 bits per 16-QAM symbol
            
            if total_bits > 0:
                return float(error_count / total_bits)
            
        except Exception as e:
            logger.warning(f"QAM BER estimation error: {e}")
        
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
    
    def _estimate_ber(self, symbols: np.ndarray) -> float:
        """Estimate Bit Error Rate for 64-QAM."""
        try:
            if len(symbols) == 0:
                return 0.0
            
            # Generate ideal constellation
            ideal_points = self._generate_qam64_constellation()
            
            error_count = 0
            total_bits = 0
            
            for symbol in symbols:
                # Find closest ideal point
                distances = np.abs(symbol - ideal_points)
                closest_index = np.argmin(distances)
                closest_ideal = ideal_points[closest_index]
                
                # Simple error estimation based on distance
                error_distance = np.abs(symbol - closest_ideal)
                
                # If distance is large, assume bit errors
                if error_distance > 0.5:  # Threshold for error detection
                    error_count += 1  # Assume 1 bit error per symbol
                
                total_bits += 6  # 6 bits per 64-QAM symbol
            
            if total_bits > 0:
                return float(error_count / total_bits)
            
        except Exception as e:
            logger.warning(f"64-QAM BER estimation error: {e}")
        
        return 0.0


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
        """Demodulate FSK signal using advanced libraries."""
        try:
            if SCIKIT_DSP_AVAILABLE:
                return self._demodulate_fsk_with_scikit(signal_data)
            elif SDR_AVAILABLE:
                return self._demodulate_fsk_with_sdr(signal_data)
            else:
                return self._demodulate_fsk_basic(signal_data)
                
        except Exception as e:
            logger.error(f"FSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    
    def _demodulate_fsk_with_scikit(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate FSK using scikit-dsp-comm."""
        try:
            # Use scikit-dsp-comm's FSK demodulation capabilities
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            
            # Ensure filter parameters are valid
            filter_length = int(filter_length) if 'filter_length' in locals() else int(101)  # Ensure integer
            nyquist_freq = self.sample_rate / 2
            
            # Calculate normalized frequencies (ensure they're in valid range)
            mark_low = max(0.01, (self.mark_frequency - self.symbol_rate/2) / nyquist_freq)
            mark_high = min(0.99, (self.mark_frequency + self.symbol_rate/2) / nyquist_freq)
            space_low = max(0.01, (self.space_frequency - self.symbol_rate/2) / nyquist_freq)
            space_high = min(0.99, (self.space_frequency + self.symbol_rate/2) / nyquist_freq)
            
            # Create bandpass filters for mark and space frequencies with proper parameters
            mark_filter = fir.firwin_kaiser_bpf(
                filter_length,  # N (integer filter length)
                mark_low,       # f_pass1 (normalized frequency)
                mark_high,      # f_pass2 (normalized frequency)
                60,             # d_pass (passband ripple in dB)
                80              # d_stop (stopband attenuation in dB)
            )
            
            space_filter = fir.firwin_kaiser_bpf(
                filter_length,  # N (integer filter length)
                space_low,      # f_pass1 (normalized frequency)  
                space_high,     # f_pass2 (normalized frequency)
                60,             # d_pass (passband ripple in dB)
                80              # d_stop (stopband attenuation in dB)
            )
            
            # Filter signals
            mark_filtered = signal.lfilter(mark_filter, 1, signal_data)
            space_filtered = signal.lfilter(space_filter, 1, signal_data)
            
            # Energy detection
            mark_energy = np.abs(mark_filtered)**2
            space_energy = np.abs(space_filtered)**2
            
            # Integrate over symbol periods
            mark_integrated = self._integrate_and_dump(mark_energy, samples_per_symbol)
            space_integrated = self._integrate_and_dump(space_energy, samples_per_symbol)
            
            # Make decisions
            bits = (mark_integrated > space_integrated).astype(int)
            
            # Calculate SNR
            signal_power = np.mean(np.maximum(mark_integrated, space_integrated))
            noise_power = np.mean(np.minimum(mark_integrated, space_integrated))
            snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
            return {
                "demodulated_data": bits,
                "sample_rate": self.symbol_rate,
                "data_type": "digital",
                "mark_energy": mark_integrated,
                "space_energy": space_integrated,
                "snr_db": snr_db,
                "frequency_separation": self.frequency_separation
            }
            
        except Exception as e:
            logger.warning(f"Scikit FSK demodulation failed: {e}")
            return self._demodulate_fsk_basic(signal_data)
    
    def _demodulate_fsk_with_sdr(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Demodulate FSK using sdr library."""
        try:
            # Use frequency discrimination method
            # Instantaneous frequency
            analytic_signal = signal.hilbert(np.real(signal_data))
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_freq = np.diff(instantaneous_phase) * self.sample_rate / (2 * np.pi)
            
            # Symbol timing recovery
            samples_per_symbol = int(self.sample_rate / self.symbol_rate)
            
            # Integrate and dump
            freq_integrated = self._integrate_and_dump(instantaneous_freq, samples_per_symbol)
            
            # Decision based on frequency threshold
            freq_threshold = (self.mark_frequency + self.space_frequency) / 2
            bits = (freq_integrated > freq_threshold).astype(int)
            
            return {
                "demodulated_data": bits,
                "sample_rate": self.symbol_rate,
                "data_type": "digital",
                "instantaneous_freq": instantaneous_freq,
                "frequency_threshold": freq_threshold
            }
            
        except Exception as e:
            logger.warning(f"SDR FSK demodulation failed: {e}")
            return self._demodulate_fsk_basic(signal_data)
    
    def _demodulate_fsk_basic(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Basic FSK demodulation fallback."""
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
            mark_energy = np.abs(mark_filtered)**2
            space_energy = np.abs(space_filtered)**2
            
            # Integrate and dump
            mark_integrated = self._integrate_and_dump(mark_energy, samples_per_symbol)
            space_integrated = self._integrate_and_dump(space_energy, samples_per_symbol)
            
            # Decision
            bits = (mark_integrated > space_integrated).astype(int)
            
            return {
                "demodulated_data": bits,
                "sample_rate": self.symbol_rate,
                "data_type": "digital",
                "mark_energy": mark_integrated,
                "space_energy": space_integrated
            }
            
        except Exception as e:
            logger.error(f"Basic FSK demodulation error: {e}")
            return {"demodulated_data": np.array([]), "error": str(e)}
    
    def _integrate_and_dump(self, data: np.ndarray, samples_per_symbol: int) -> np.ndarray:
        """Integrate and dump filter."""
        try:
            num_symbols = len(data) // samples_per_symbol
            integrated = np.zeros(num_symbols)
            
            for i in range(num_symbols):
                start_idx = i * samples_per_symbol
                end_idx = start_idx + samples_per_symbol
                integrated[i] = np.sum(data[start_idx:end_idx])
            
            return integrated
            
        except Exception as e:
            logger.warning(f"Integrate and dump error: {e}")
            return np.array([])
            
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