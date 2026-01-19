"""
Modulation and Demodulation Module

Implements various modulation schemes using mhostetter/sdr and scikit-dsp-comm libraries.
Supports PSK, QAM, FSK, MSK, and other digital modulation techniques.
"""

import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

# Core signal processing libraries
try:
    import sdr
    SDR_AVAILABLE = True
except ImportError:
    SDR_AVAILABLE = False
    sdr = None

try:
    import sk_dsp_comm.synchronization as synchronization
    SCIKIT_DSP_AVAILABLE = True
except ImportError:
    SCIKIT_DSP_AVAILABLE = False
    synchronization = None

from scipy import signal
from scipy.special import erfc


class ModulationType(Enum):
    """Supported modulation types"""
    PSK = "PSK"
    QPSK = "QPSK"
    BPSK = "BPSK"
    PSK8 = "8PSK"
    QAM16 = "16QAM"
    QAM64 = "64QAM"
    QAM256 = "256QAM"
    FSK = "FSK"
    MSK = "MSK" 
    GMSK = "GMSK"
    OQPSK = "OQPSK"
    CPM = "CPM"
    AM = "AM"
    FM = "FM"
    PM = "PM"


@dataclass
class ConstellationData:
    """Container for constellation diagram data"""
    symbols: np.ndarray           # Complex constellation points
    reference: np.ndarray         # Reference constellation
    evm_rms: float               # RMS Error Vector Magnitude
    evm_peak: float              # Peak EVM
    phase_error: np.ndarray      # Phase error per symbol
    magnitude_error: np.ndarray  # Magnitude error per symbol


@dataclass
class ModulationAnalysis:
    """Results of modulation analysis"""
    modulation_type: str
    symbol_rate: float
    carrier_frequency: float
    snr_estimate: float
    frequency_offset: float
    phase_offset: float
    constellation: Optional[ConstellationData]
    confidence: float  # Detection confidence 0-1


# Pre-computed constellation lookup tables (PySDR best practice - 5x faster than trig)
# These are computed once and reused for all modulation operations
CONSTELLATION_BPSK = np.array([1+0j, -1+0j], dtype=np.complex64)
CONSTELLATION_QPSK = np.exp(1j * np.pi/4 * np.array([1, 3, 5, 7], dtype=np.float32)).astype(np.complex64)
CONSTELLATION_8PSK = np.exp(1j * 2*np.pi/8 * np.arange(8, dtype=np.float32)).astype(np.complex64)
CONSTELLATION_16PSK = np.exp(1j * 2*np.pi/16 * np.arange(16, dtype=np.float32)).astype(np.complex64)

# QAM constellations (normalized to unit average power)
def _generate_qam_constellation(M: int) -> np.ndarray:
    """Generate square QAM constellation with unit average power."""
    sqrt_M = int(np.sqrt(M))
    if sqrt_M * sqrt_M != M:
        raise ValueError(f"M={M} must be a perfect square for QAM")
    
    # Create grid
    I = np.arange(-sqrt_M + 1, sqrt_M, 2, dtype=np.float32)
    Q = np.arange(-sqrt_M + 1, sqrt_M, 2, dtype=np.float32)
    
    # Create all combinations
    constellation = []
    for q in Q:
        for i in I:
            constellation.append(complex(i, q))
    
    constellation = np.array(constellation, dtype=np.complex64)
    
    # Normalize to unit average power
    avg_power = np.mean(np.abs(constellation)**2)
    return constellation / np.sqrt(avg_power)

CONSTELLATION_16QAM = _generate_qam_constellation(16)
CONSTELLATION_64QAM = _generate_qam_constellation(64)
CONSTELLATION_256QAM = _generate_qam_constellation(256)

# Constellation lookup dictionary for fast access
CONSTELLATION_LOOKUP = {
    'BPSK': CONSTELLATION_BPSK,
    'QPSK': CONSTELLATION_QPSK,
    '8PSK': CONSTELLATION_8PSK,
    '16PSK': CONSTELLATION_16PSK,
    '16QAM': CONSTELLATION_16QAM,
    '64QAM': CONSTELLATION_64QAM,
    '256QAM': CONSTELLATION_256QAM,
}


@dataclass 
class ModulationConfig:
    """Configuration for modulation schemes"""
    modulation_type: str = "bpsk"  # Changed from mod_type to modulation_type for test compatibility
    m: int = 4  # Constellation size
    constellation_size: int = 4  # Alternative name for m
    sample_rate: float = 10e6
    symbol_rate: float = 1e6
    pulse_shape: str = "rrc"  # "rect", "rc", "srrc"
    alpha: float = 0.35  # Roll-off factor for RC/SRRC
    sps: int = 8  # Samples per symbol
    span: int = 8  # Filter span in symbols
    frequency_separation: Optional[float] = None  # For FSK
    n_subcarriers: Optional[int] = None  # For OFDM
    cp_length: Optional[int] = None  # Cyclic prefix for OFDM
    
    # Legacy compatibility
    @property
    def mod_type(self):
        return self.modulation_type


class DigitalModulator:
    """Digital modulation engine using multiple libraries"""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

        # Modulator objects
        self.modulators = {}

        # Initialize modulators
        self.initialize_modulators()

    def initialize_modulators(self):
        """Initialize various modulator objects"""
        try:
            if SDR_AVAILABLE:
                # PSK modulators
                self.modulators['BPSK'] = sdr.PSK(2, pulse_shape="srrc", sps=8, span=8)
                self.modulators['QPSK'] = sdr.PSK(4, pulse_shape="srrc", sps=8, span=8)
                self.modulators['8PSK'] = sdr.PSK(8, pulse_shape="srrc", sps=8, span=8)

                # QAM modulators
                self.modulators['16QAM'] = sdr.QAM(16, pulse_shape="srrc", sps=8, span=8)
                self.modulators['64QAM'] = sdr.QAM(64, pulse_shape="srrc", sps=8, span=8)
                self.modulators['256QAM'] = sdr.QAM(256, pulse_shape="srrc", sps=8, span=8)

                # MSK modulator
                self.modulators['MSK'] = sdr.MSK(sps=8, span=8)

                # OQPSK modulator
                try:
                    self.modulators['OQPSK'] = sdr.OQPSK(pulse_shape="srrc", sps=8, span=8)
                except AttributeError:
                    # OQPSK might not be available in all versions
                    pass

            self.logger.info(f"Initialized {len(self.modulators)} digital modulators")

        except Exception as e:
            self.logger.error(f"Modulator initialization error: {e}")

    def modulate(self, bits: np.ndarray, mod_type: str = None, **kwargs) -> np.ndarray:
        """Modulate bit sequence using specified modulation"""
        try:
            # If no mod_type specified, use the modulator's default type
            if mod_type is None:
                if hasattr(self, 'mod_type'):
                    mod_type = self.mod_type
                else:
                    self.logger.error("No modulation type specified and modulator has no default")
                    return np.array([])
            
            if mod_type not in self.modulators:
                self.logger.error(f"Modulator '{mod_type}' not available")
                return np.array([])

            modulator = self.modulators[mod_type]

            # Convert bits to symbols if needed
            if hasattr(modulator, 'map'):
                symbols = modulator.map(bits)
            else:
                # For modulators that take bits directly
                symbols = bits

            # Modulate
            if hasattr(modulator, 'modulate'):
                return modulator.modulate(symbols)
            else:
                return modulator(symbols)

        except Exception as e:
            self.logger.error(f"Modulation error: {e}")
            return np.array([])

    def get_constellation(self, mod_type: str) -> Optional[np.ndarray]:
        """Get reference constellation for modulation type"""
        try:
            if mod_type not in self.modulators:
                return None

            modulator = self.modulators[mod_type]

            if hasattr(modulator, 'constellation'):
                return modulator.constellation
            elif hasattr(modulator, 'symbol_map'):
                return modulator.symbol_map
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error getting constellation: {e}")
            return None


class DigitalDemodulator:
    """Digital demodulation engine"""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

        # Demodulator objects
        self.demodulators = {}

        # Synchronization objects
        self.carrier_recovery = None
        self.symbol_timing_recovery = None

        self.initialize_demodulators()

    def initialize_demodulators(self):
        """Initialize demodulator objects"""
        try:
            if SDR_AVAILABLE:
                # Create demodulator instances (same as modulators for coherent demod)
                self.demodulators['BPSK'] = sdr.PSK(2, pulse_shape="srrc", sps=8, span=8)
                self.demodulators['QPSK'] = sdr.PSK(4, pulse_shape="srrc", sps=8, span=8)
                self.demodulators['8PSK'] = sdr.PSK(8, pulse_shape="srrc", sps=8, span=8)

                self.demodulators['16QAM'] = sdr.QAM(16, pulse_shape="srrc", sps=8, span=8)
                self.demodulators['64QAM'] = sdr.QAM(64, pulse_shape="srrc", sps=8, span=8)

                self.demodulators['MSK'] = sdr.MSK(sps=8, span=8)

            # Initialize synchronization components
            if SCIKIT_DSP_AVAILABLE and synchronization:
                # Carrier recovery PLL
                self.carrier_recovery = synchronization.NDA_PLL()

                # Symbol timing recovery
                # This would need proper initialization based on the specific API
                pass

            self.logger.info(f"Initialized {len(self.demodulators)} digital demodulators")

        except Exception as e:
            self.logger.error(f"Demodulator initialization error: {e}")

    def demodulate(
        self, 
        signal_samples: np.ndarray, 
        mod_type: str,
        symbol_rate: float = None,
        carrier_freq: float = 0,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Demodulate signal and return symbols and constellation"""
        try:
            if mod_type not in self.demodulators:
                self.logger.error(f"Demodulator '{mod_type}' not available")
                return np.array([]), np.array([])

            demodulator = self.demodulators[mod_type]

            # Apply carrier frequency correction if needed
            if carrier_freq != 0:
                signal_samples = self._apply_frequency_correction(signal_samples, carrier_freq)

            # Demodulate
            if hasattr(demodulator, 'demodulate'):
                symbols = demodulator.demodulate(signal_samples)
            else:
                symbols = demodulator(signal_samples)

            # Get constellation
            constellation = self.get_constellation_data(symbols, mod_type)

            return symbols, constellation

        except Exception as e:
            self.logger.error(f"Demodulation error: {e}")
            return np.array([]), np.array([])

    def get_constellation_data(self, symbols: np.ndarray, mod_type: str) -> np.ndarray:
        """Extract constellation points from demodulated symbols"""
        try:
            if mod_type not in self.demodulators:
                return symbols

            demodulator = self.demodulators[mod_type]

            # Get reference constellation
            if hasattr(demodulator, 'constellation'):
                reference_constellation = demodulator.constellation
            else:
                reference_constellation = None

            # For now, just return the symbols as constellation points
            # In a full implementation, this would include EVM calculation, etc.
            return symbols

        except Exception as e:
            self.logger.error(f"Error extracting constellation data: {e}")
            return symbols

    def _apply_frequency_correction(self, samples: np.ndarray, freq_offset: float) -> np.ndarray:
        """Apply frequency correction to samples"""
        try:
            t = np.arange(len(samples)) / self.sample_rate
            correction = np.exp(-1j * 2 * np.pi * freq_offset * t)
            return samples * correction
        except Exception as e:
            self.logger.error(f"Frequency correction error: {e}")
            return samples


class AnalogModulator:
    """Analog modulation (AM, FM, PM) implementation"""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

    def am_modulate(
        self, 
        message: np.ndarray, 
        carrier_freq: float, 
        modulation_index: float = 0.5
    ) -> np.ndarray:
        """Amplitude modulation"""
        try:
            t = np.arange(len(message)) / self.sample_rate
            carrier = np.cos(2 * np.pi * carrier_freq * t)

            # AM: (1 + m * message(t)) * cos(2πfct)
            modulated = (1 + modulation_index * message) * carrier

            return modulated

        except Exception as e:
            self.logger.error(f"AM modulation error: {e}")
            return np.zeros_like(message)

    def fm_modulate(
        self, 
        message: np.ndarray, 
        carrier_freq: float, 
        frequency_deviation: float
    ) -> np.ndarray:
        """Frequency modulation"""
        try:
            # Integrate message to get phase
            dt = 1 / self.sample_rate
            phase_integral = np.cumsum(message) * dt

            t = np.arange(len(message)) / self.sample_rate

            # FM: cos(2πfct + 2πkf∫m(τ)dτ)
            instantaneous_phase = 2 * np.pi * carrier_freq * t + 2 * np.pi * frequency_deviation * phase_integral

            modulated = np.cos(instantaneous_phase)

            return modulated

        except Exception as e:
            self.logger.error(f"FM modulation error: {e}")
            return np.zeros_like(message)

    def pm_modulate(
        self, 
        message: np.ndarray, 
        carrier_freq: float, 
        phase_deviation: float
    ) -> np.ndarray:
        """Phase modulation"""
        try:
            t = np.arange(len(message)) / self.sample_rate

            # PM: cos(2πfct + kp*m(t))
            instantaneous_phase = 2 * np.pi * carrier_freq * t + phase_deviation * message

            modulated = np.cos(instantaneous_phase)

            return modulated

        except Exception as e:
            self.logger.error(f"PM modulation error: {e}")
            return np.zeros_like(message)


class AnalogDemodulator:
    """Analog demodulation implementation"""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

    def am_demodulate(self, signal: np.ndarray, method: str = "envelope") -> np.ndarray:
        """AM demodulation"""
        try:
            if method == "envelope":
                # Envelope detection
                analytic_signal = signal.hilbert(signal)
                envelope = np.abs(analytic_signal)

                # Remove DC component
                message = envelope - np.mean(envelope)

                return message

            elif method == "coherent":
                # Coherent demodulation (requires carrier recovery)
                # This is simplified - real implementation would need PLL
                carrier_freq = self._estimate_carrier_frequency(signal)
                t = np.arange(len(signal)) / self.sample_rate
                local_oscillator = 2 * np.cos(2 * np.pi * carrier_freq * t)

                # Multiply and low-pass filter
                demodulated = signal * local_oscillator

                # Simple low-pass filter
                cutoff = carrier_freq / 10  # Rough estimate
                sos = signal.butter(6, cutoff, fs=self.sample_rate, output='sos')
                message = signal.sosfilt(sos, demodulated)

                return message

            else:
                self.logger.error(f"Unknown AM demodulation method: {method}")
                return np.zeros_like(signal)

        except Exception as e:
            self.logger.error(f"AM demodulation error: {e}")
            return np.zeros_like(signal)

    def fm_demodulate(self, signal: np.ndarray, method: str = "quadrature") -> np.ndarray:
        """
        FM demodulation using efficient PySDR quadrature technique.
        
        Args:
            signal: Complex IQ samples
            method: "quadrature" (fast PySDR method) or "phase" (traditional)
            
        Returns:
            Demodulated FM signal
        """
        try:
            if method == "quadrature":
                # Efficient quadrature FM demodulation (PySDR technique)
                # This is 2-3x faster than traditional phase differentiation
                # Formula: 0.5 * angle(x[n] * conj(x[n-1]))
                demod = 0.5 * np.angle(signal[1:] * np.conj(signal[:-1]))
                
                # Pad to maintain length
                return np.concatenate([[0], demod])
            
            else:  # Traditional phase method
                # Get analytic signal
                analytic_signal = signal.hilbert(signal)

                # Compute instantaneous phase
                instantaneous_phase = np.angle(analytic_signal)

                # Unwrap phase to avoid discontinuities
                instantaneous_phase = np.unwrap(instantaneous_phase)

                # Differentiate to get instantaneous frequency
                instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi) * self.sample_rate

                # The message is the deviation from carrier frequency
                # For simplicity, we'll high-pass filter to remove DC
                if len(instantaneous_freq) > 100:
                    sos = signal.butter(3, 100, btype='high', fs=self.sample_rate, output='sos')
                    message = signal.sosfilt(sos, instantaneous_freq)
                else:
                    message = instantaneous_freq - np.mean(instantaneous_freq)

                return message

        except Exception as e:
            self.logger.error(f"FM demodulation error: {e}")
            return np.zeros_like(signal)

    def pm_demodulate(self, signal: np.ndarray) -> np.ndarray:
        """PM demodulation"""
        try:
            # Get analytic signal
            analytic_signal = signal.hilbert(signal)

            # Compute instantaneous phase
            instantaneous_phase = np.angle(analytic_signal)

            # Unwrap phase
            instantaneous_phase = np.unwrap(instantaneous_phase)

            # Remove linear trend (carrier phase)
            t = np.arange(len(instantaneous_phase))
            p = np.polyfit(t, instantaneous_phase, 1)
            linear_trend = np.polyval(p, t)

            message = instantaneous_phase - linear_trend

            return message

        except Exception as e:
            self.logger.error(f"PM demodulation error: {e}")
            return np.zeros_like(signal)

    def _estimate_carrier_frequency(self, signal: np.ndarray) -> float:
        """Estimate carrier frequency from signal spectrum"""
        try:
            # Compute FFT
            fft_result = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), 1/self.sample_rate)

            # Find peak frequency (positive frequencies only)
            positive_freqs = freqs[freqs >= 0]
            positive_fft = np.abs(fft_result[freqs >= 0])

            peak_index = np.argmax(positive_fft)
            carrier_freq = positive_freqs[peak_index]

            return carrier_freq

        except Exception as e:
            self.logger.error(f"Carrier frequency estimation error: {e}")
            return 1e6  # Default 1 MHz


class ModulationClassifier:
    """Automatic modulation classification"""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

    def classify_modulation(self, signal: np.ndarray) -> ModulationAnalysis:
        """Classify modulation type of signal"""
        try:
            # Extract various signal features
            features = self._extract_features(signal)

            # Use feature-based classification
            mod_type, confidence = self._classify_from_features(features)

            # Estimate parameters
            symbol_rate = self._estimate_symbol_rate(signal, mod_type)
            carrier_freq = self._estimate_carrier_frequency(signal)
            snr = self._estimate_snr(signal)
            freq_offset = self._estimate_frequency_offset(signal)
            phase_offset = self._estimate_phase_offset(signal)

            return ModulationAnalysis(
                modulation_type=mod_type,
                symbol_rate=symbol_rate,
                carrier_frequency=carrier_freq,
                snr_estimate=snr,
                frequency_offset=freq_offset,
                phase_offset=phase_offset,
                constellation=None,  # Could be populated later
                confidence=confidence
            )

        except Exception as e:
            self.logger.error(f"Modulation classification error: {e}")
            return ModulationAnalysis(
                modulation_type="Unknown",
                symbol_rate=0,
                carrier_frequency=0,
                snr_estimate=0,
                frequency_offset=0,
                phase_offset=0,
                constellation=None,
                confidence=0
            )

    def _extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract signal features for classification"""
        try:
            features = {}

            # Amplitude features
            features['mean_amplitude'] = np.mean(np.abs(signal))
            features['std_amplitude'] = np.std(np.abs(signal))
            features['amplitude_variance'] = np.var(np.abs(signal))

            # Phase features
            phases = np.angle(signal)
            phase_diff = np.diff(np.unwrap(phases))
            features['mean_phase_diff'] = np.mean(phase_diff)
            features['std_phase_diff'] = np.std(phase_diff)

            # Frequency domain features
            fft_result = np.fft.fft(signal)
            power_spectrum = np.abs(fft_result) ** 2
            features['spectral_centroid'] = np.sum(np.arange(len(power_spectrum)) * power_spectrum) / np.sum(power_spectrum)
            features['spectral_rolloff'] = self._spectral_rolloff(power_spectrum)

            # Higher order statistics
            features['kurtosis'] = self._kurtosis(signal)
            features['skewness'] = self._skewness(signal)

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return {}

    def _classify_from_features(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Simple rule-based classification from features"""
        try:
            # This is a very simplified classification
            # Real implementation would use machine learning

            if not features:
                return "Unknown", 0.0

            std_amplitude = features.get('std_amplitude', 0)
            std_phase_diff = features.get('std_phase_diff', 0)
            amplitude_variance = features.get('amplitude_variance', 0)

            # Simple heuristic rules
            if amplitude_variance < 0.01:
                # Constant amplitude suggests PSK/FSK
                if std_phase_diff > 1.0:
                    return "PSK", 0.7
                else:
                    return "FSK", 0.6

            elif amplitude_variance > 0.1:
                # Variable amplitude suggests QAM/AM
                if std_phase_diff > 0.5:
                    return "QAM", 0.6
                else:
                    return "AM", 0.7

            else:
                return "Unknown", 0.3

        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return "Unknown", 0.0

    def _estimate_symbol_rate(self, signal: np.ndarray, mod_type: str) -> float:
        """Estimate symbol rate"""
        try:
            # Simplified symbol rate estimation
            # Real implementation would use more sophisticated methods

            # Use spectral analysis of envelope or phase
            if mod_type in ["PSK", "QAM"]:
                # For PSK/QAM, look at phase transitions
                phases = np.angle(signal)
                phase_diff = np.abs(np.diff(np.unwrap(phases)))

                # Find peaks in phase difference (symbol transitions)
                threshold = np.std(phase_diff) * 2
                transitions = phase_diff > threshold

                if np.sum(transitions) > 0:
                    avg_symbol_period = len(signal) / np.sum(transitions) / self.sample_rate
                    return 1 / avg_symbol_period

            # Default estimate
            return self.sample_rate / 8  # Assume 8 samples per symbol

        except Exception as e:
            self.logger.error(f"Symbol rate estimation error: {e}")
            return 0.0

    def _estimate_carrier_frequency(self, signal: np.ndarray) -> float:
        """Estimate carrier frequency"""
        try:
            fft_result = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), 1/self.sample_rate)

            # Find peak in positive frequencies
            positive_mask = freqs >= 0
            positive_freqs = freqs[positive_mask]
            positive_fft = np.abs(fft_result[positive_mask])

            peak_index = np.argmax(positive_fft)
            return positive_freqs[peak_index]

        except Exception as e:
            self.logger.error(f"Carrier frequency estimation error: {e}")
            return 0.0

    def _estimate_snr(self, signal: np.ndarray) -> float:
        """Estimate SNR in dB"""
        try:
            # Simple SNR estimation based on signal variance
            signal_power = np.mean(np.abs(signal) ** 2)

            # Estimate noise power (very simplified)
            # Real implementation would use more sophisticated methods
            high_freq_samples = signal[::10]  # Decimate to get high frequency components
            noise_power = np.var(np.abs(high_freq_samples))

            if noise_power > 0:
                snr_linear = signal_power / noise_power
                return 10 * np.log10(snr_linear)
            else:
                return 60.0  # High SNR if no noise detected

        except Exception as e:
            self.logger.error(f"SNR estimation error: {e}")
            return 0.0

    def _estimate_frequency_offset(self, signal: np.ndarray) -> float:
        """Estimate frequency offset"""
        # Simplified - return 0 for now
        return 0.0

    def _estimate_phase_offset(self, signal: np.ndarray) -> float:
        """Estimate phase offset"""
        # Simplified - return 0 for now
        return 0.0

    def _spectral_rolloff(self, power_spectrum: np.ndarray, rolloff_threshold: float = 0.85) -> float:
        """Calculate spectral rolloff point"""
        total_energy = np.sum(power_spectrum)
        threshold_energy = rolloff_threshold * total_energy

        cumulative_energy = np.cumsum(power_spectrum)
        rolloff_index = np.where(cumulative_energy >= threshold_energy)[0]

        if len(rolloff_index) > 0:
            return rolloff_index[0] / len(power_spectrum)
        else:
            return 1.0

    def _kurtosis(self, signal: np.ndarray) -> float:
        """Calculate kurtosis of signal"""
        real_part = np.real(signal)
        mean_val = np.mean(real_part)
        std_val = np.std(real_part)

        if std_val > 0:
            normalized = (real_part - mean_val) / std_val
            return np.mean(normalized ** 4) - 3
        else:
            return 0.0

    def _skewness(self, signal: np.ndarray) -> float:
        """Calculate skewness of signal"""
        real_part = np.real(signal)
        mean_val = np.mean(real_part)
        std_val = np.std(real_part)

        if std_val > 0:
            normalized = (real_part - mean_val) / std_val
            return np.mean(normalized ** 3)
        else:
            return 0.0


# Test functions
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    sample_rate = 1e6  # 1 MHz

    # Test digital modulation
    digital_mod = DigitalModulator(sample_rate)
    digital_demod = DigitalDemodulator(sample_rate)

    # Generate test bits
    test_bits = np.random.randint(0, 2, 100)

    if SDR_AVAILABLE and 'QPSK' in digital_mod.modulators:
        # Test QPSK modulation
        modulated_signal = digital_mod.modulate(test_bits, 'QPSK')
        print(f"QPSK modulated signal length: {len(modulated_signal)}")

        # Test demodulation
        symbols, constellation = digital_demod.demodulate(modulated_signal, 'QPSK')
        print(f"Demodulated symbols: {len(symbols)}")

    # Test analog modulation
    analog_mod = AnalogModulator(sample_rate)
    analog_demod = AnalogDemodulator(sample_rate)

    # Generate test message
    test_message = np.sin(2 * np.pi * 1000 * np.arange(1000) / sample_rate)  # 1 kHz tone

    # Test AM modulation/demodulation
    am_signal = analog_mod.am_modulate(test_message, 100e3, 0.5)  # 100 kHz carrier
    demod_message = analog_demod.am_demodulate(am_signal)

    print(f"AM signal length: {len(am_signal)}, Demodulated: {len(demod_message)}")

    # Test modulation classification
    classifier = ModulationClassifier(sample_rate)
    classification = classifier.classify_modulation(am_signal)

    print(f"Classified modulation: {classification.modulation_type} (confidence: {classification.confidence:.2f})")
    print(f"Estimated parameters: SNR={classification.snr_estimate:.1f} dB, "
          f"Symbol rate={classification.symbol_rate:.0f} Hz")


# Backward compatibility aliases for testing
BaseModulator = DigitalModulator  # For test compatibility
BaseDemodulator = DigitalDemodulator  # For test compatibility
PSKModulator = DigitalModulator  
QAMModulator = DigitalModulator
FSKModulator = DigitalModulator
OFDMModulator = DigitalModulator
PSKDemodulator = DigitalDemodulator


# Specific modulator classes for better API
class PSKModulator(DigitalModulator):
    """PSK-specific modulator"""
    def __init__(self, m: int = 4, **kwargs):
        super().__init__(kwargs.get('sample_rate', 1e6))
        self.m = m
        self.mod_type = f"{m}PSK" if m > 2 else "BPSK"
        
        # Generate constellation points
        self._constellation = self._generate_constellation()
        
        # Pulse shaping parameters
        self.samples_per_symbol = kwargs.get('sps', 8)
        self.alpha = kwargs.get('alpha', 0.35)
        self.span = kwargs.get('span', 8)
        self.pulse_shape = kwargs.get('pulse_shape', 'srrc')
        
        # Create pulse filter
        self.pulse_filter = self._create_pulse_filter()
    
    def _generate_constellation(self):
        """Generate PSK constellation points"""
        angles = 2 * np.pi * np.arange(self.m) / self.m
        return np.exp(1j * angles)
    
    def _create_pulse_filter(self):
        """Create pulse shaping filter"""
        # Simple RRC filter implementation
        if self.pulse_shape in ['rrc', 'srrc']:
            # Generate RRC filter taps
            beta = self.alpha
            t = np.arange(-self.span * self.samples_per_symbol // 2, 
                         self.span * self.samples_per_symbol // 2 + 1) / self.samples_per_symbol
            
            # RRC formula with edge case handling
            h = np.zeros_like(t)
            for i, time_val in enumerate(t):
                if time_val == 0:
                    h[i] = (1 + beta * (4/np.pi - 1))
                elif abs(time_val) == 1/(4*beta) and beta != 0:
                    h[i] = (beta/np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*beta)) + 
                                               (1 - 2/np.pi) * np.cos(np.pi/(4*beta)))
                else:
                    numerator = np.sin(np.pi * time_val * (1 - beta)) + \
                               4 * beta * time_val * np.cos(np.pi * time_val * (1 + beta))
                    denominator = np.pi * time_val * (1 - (4 * beta * time_val)**2)
                    h[i] = numerator / denominator
            
            return h / np.sqrt(np.sum(h**2))
        else:
            # Rectangular pulse
            return np.ones(self.samples_per_symbol) / np.sqrt(self.samples_per_symbol)
    
    @property
    def constellation(self):
        """Get constellation points"""
        return self._constellation
    
    def get_constellation(self):
        """Get constellation points (method version)"""
        return self._constellation
    
    def generate_symbols(self, bits: np.ndarray) -> np.ndarray:
        """Generate symbols from input bits"""
        if len(bits) == 0:
            return np.array([])
        
        # Convert bits to symbols
        bits_per_symbol = int(np.log2(self.m))
        
        # Pad bits if necessary
        n_pad = (bits_per_symbol - (len(bits) % bits_per_symbol)) % bits_per_symbol
        padded_bits = np.append(bits, np.zeros(n_pad, dtype=int))
        
        # Group bits into symbols
        bit_groups = padded_bits.reshape(-1, bits_per_symbol)
        
        # Convert bit groups to symbol indices
        symbol_indices = np.array([np.packbits(bg, bitorder='big')[0] >> (8 - bits_per_symbol) 
                                  for bg in bit_groups])
        
        # Map to constellation
        return self._constellation[symbol_indices]
    
    def modulate(self, data: np.ndarray) -> np.ndarray:
        """Modulate input data (bits or symbols) to PSK signal"""
        if len(data) == 0:
            return np.array([])
        
        # Check if input is complex (symbols) or real (bits)
        if np.iscomplexobj(data):
            # Input is already symbols
            symbols = data
        else:
            # Input is bits, generate symbols
            symbols = self.generate_symbols(data)
        
        # Upsample by inserting zeros
        upsampled = np.zeros(len(symbols) * self.samples_per_symbol, dtype=complex)
        upsampled[::self.samples_per_symbol] = symbols
        
        # Apply pulse shaping filter
        if len(self.pulse_filter) > 0:
            modulated = np.convolve(upsampled, self.pulse_filter, mode='same')
        else:
            modulated = upsampled
        
        return modulated


class QAMModulator(DigitalModulator):
    """QAM-specific modulator"""
    def __init__(self, m: int = 16, **kwargs):
        if not (m > 0 and (m & (m-1)) == 0):  # Check if m is power of 2
            # Check if perfect square for QAM
            sqrt_m = int(np.sqrt(m))
            if sqrt_m * sqrt_m != m:
                raise ValueError(f"QAM constellation size {m} must be a perfect square")
        super().__init__(kwargs.get('sample_rate', 1e6))
        self.m = m
        self.mod_type = f"{m}QAM"
        
        # Generate constellation points
        self._constellation = self._generate_constellation()
    
    def _generate_constellation(self):
        """Generate QAM constellation points"""
        sqrt_m = int(np.sqrt(self.m))
        
        # Create square QAM constellation
        # Map to {-sqrt_m+1, -sqrt_m+3, ..., sqrt_m-3, sqrt_m-1}
        I = np.arange(-sqrt_m + 1, sqrt_m, 2)
        Q = np.arange(-sqrt_m + 1, sqrt_m, 2)
        
        # Create all combinations
        constellation = []
        for q in Q:
            for i in I:
                constellation.append(complex(i, q))
        
        constellation = np.array(constellation)
        
        # Normalize to unit average power
        avg_power = np.mean(np.abs(constellation)**2)
        constellation = constellation / np.sqrt(avg_power)
        
        return constellation
    
    @property
    def constellation(self):
        """Get constellation points"""
        return self._constellation
    
    def get_constellation(self):
        """Get constellation points (method version)"""
        return self._constellation
    
    def generate_symbols(self, bits: np.ndarray) -> np.ndarray:
        """Generate symbols from input bits"""
        if len(bits) == 0:
            return np.array([])
        
        # Convert bits to symbols
        bits_per_symbol = int(np.log2(self.m))
        
        # Pad bits if necessary
        n_pad = (bits_per_symbol - (len(bits) % bits_per_symbol)) % bits_per_symbol
        padded_bits = np.append(bits, np.zeros(n_pad, dtype=int))
        
        # Group bits into symbols
        bit_groups = padded_bits.reshape(-1, bits_per_symbol)
        
        # Convert bit groups to symbol indices
        symbol_indices = np.array([np.packbits(bg, bitorder='big')[0] >> (8 - bits_per_symbol) 
                                  for bg in bit_groups])
        
        # Map to constellation
        return self._constellation[symbol_indices]
    
    def modulate(self, data: np.ndarray) -> np.ndarray:
        """Modulate input data (bits or symbols) to QAM signal"""
        if len(data) == 0:
            return np.array([])
        
        # Check if input is complex (symbols) or real (bits)
        if np.iscomplexobj(data):
            # Input is already symbols
            symbols = data
        else:
            # Input is bits, generate symbols
            symbols = self.generate_symbols(data)
        
        # Simple upsampling without pulse shaping for now
        samples_per_symbol = 8
        upsampled = np.zeros(len(symbols) * samples_per_symbol, dtype=complex)
        upsampled[::samples_per_symbol] = symbols
        
        return upsampled


class FSKModulator(DigitalModulator):
    """FSK-specific modulator"""
    def __init__(self, m: int = 2, frequency_separation: float = 1000, **kwargs):
        super().__init__(kwargs.get('sample_rate', 1e6))
        self.m = m
        self.frequency_separation = frequency_separation
        self.mod_type = f"{m}FSK" if m > 2 else "FSK"
        
        # Generate frequency array
        self.frequencies = self._generate_frequencies()
    
    def _generate_frequencies(self):
        """Generate FSK frequency array"""
        # Center frequencies around 0
        start_freq = -(self.m - 1) * self.frequency_separation / 2
        return np.array([start_freq + i * self.frequency_separation for i in range(self.m)])
    
    def generate_symbols(self, bits: np.ndarray) -> np.ndarray:
        """Generate symbols from input bits"""
        if len(bits) == 0:
            return np.array([])
        
        # Convert bits to symbol indices
        bits_per_symbol = int(np.log2(self.m))
        
        # Pad bits if necessary
        n_pad = (bits_per_symbol - (len(bits) % bits_per_symbol)) % bits_per_symbol
        padded_bits = np.append(bits, np.zeros(n_pad, dtype=int))
        
        # Group bits into symbols
        bit_groups = padded_bits.reshape(-1, bits_per_symbol)
        
        # Convert bit groups to symbol indices
        symbol_indices = np.array([np.packbits(bg, bitorder='big')[0] >> (8 - bits_per_symbol) 
                                  for bg in bit_groups])
        
        # Return frequency indices (symbols are frequency indices for FSK)
        return symbol_indices
    
    def modulate(self, data: np.ndarray) -> np.ndarray:
        """Modulate input data (bits or symbol indices) to FSK signal"""
        if len(data) == 0:
            return np.array([])
        
        # Check if input is already symbol indices or bits
        if np.issubdtype(data.dtype, np.integer) and np.max(data) < self.m:
            # Input is already symbol indices
            symbol_indices = data
        else:
            # Input is bits, generate symbol indices
            symbol_indices = self.generate_symbols(data)
        
        # Generate FSK signal
        samples_per_symbol = 100  # samples per symbol
        t_symbol = samples_per_symbol / self.sample_rate
        
        signal = []
        for freq_idx in symbol_indices:
            freq = self.frequencies[freq_idx]
            t = np.linspace(0, t_symbol, samples_per_symbol, endpoint=False)
            symbol_signal = np.exp(1j * 2 * np.pi * freq * t)
            signal.extend(symbol_signal)
        
        return np.array(signal)


class OFDMModulator(DigitalModulator):
    """OFDM-specific modulator"""
    def __init__(self, n_subcarriers: int = 64, cp_length: int = 16, 
                 subcarrier_modulation: str = "qpsk", **kwargs):
        super().__init__(kwargs.get('sample_rate', 1e6))
        self.n_subcarriers = n_subcarriers
        self.cp_length = cp_length
        self.mod_type = "OFDM"
        self.subcarrier_modulation = subcarrier_modulation
        
        # Create subcarrier modulator
        if subcarrier_modulation.lower() == "qpsk":
            self.subcarrier_mod = PSKModulator(m=4, **kwargs)
        elif subcarrier_modulation.lower() == "16qam":
            self.subcarrier_mod = QAMModulator(m=16, **kwargs)
        else:
            self.subcarrier_mod = PSKModulator(m=4, **kwargs)  # Default to QPSK
    
    def modulate(self, data: np.ndarray) -> np.ndarray:
        """Modulate input data (bits or symbols) to OFDM signal"""
        if len(data) == 0:
            return np.array([])
        
        # Check if input is complex (symbols) or real (bits)
        if np.iscomplexobj(data):
            # Input is already symbols
            symbols = data
        else:
            # Input is bits, generate symbols using subcarrier modulation
            symbols = self.subcarrier_mod.generate_symbols(data)
        
        # Simple OFDM implementation - assign symbols to subcarriers
        n_symbols_per_block = min(len(symbols), self.n_subcarriers)
        
        if len(symbols) < self.n_subcarriers:
            # Pad with zeros if needed
            padded_symbols = np.zeros(self.n_subcarriers, dtype=complex)
            padded_symbols[:len(symbols)] = symbols
        else:
            # Take first n_subcarriers symbols
            padded_symbols = symbols[:self.n_subcarriers]
        
        # Apply IFFT
        ofdm_block = np.fft.ifft(padded_symbols)
        
        # Add cyclic prefix
        cp = ofdm_block[-self.cp_length:]
        ofdm_symbol = np.concatenate([cp, ofdm_block])
        
        return ofdm_symbol


class PSKDemodulator(DigitalDemodulator):
    """PSK-specific demodulator"""
    def __init__(self, m: int = 4, **kwargs):
        super().__init__(kwargs.get('sample_rate', 1e6))
        self.m = m
        self.mod_type = f"{m}PSK" if m > 2 else "BPSK"
        # Generate constellation points for demodulation
        self.constellation = np.exp(1j * 2 * np.pi * np.arange(m) / m)
    
    def detect_symbols(self, signal: np.ndarray) -> np.ndarray:
        """Detect symbols from received signal"""
        if len(signal) == 0:
            return np.array([])
        
        # Simple symbol detection by finding closest constellation point
        detected_indices = []
        for symbol in signal:
            # Calculate distances to all constellation points
            distances = np.abs(symbol - self.constellation)
            # Find closest constellation point
            closest_idx = np.argmin(distances)
            detected_indices.append(closest_idx)
        
        return np.array(detected_indices)


# Convenience functions
def create_psk_modulator(m: int = 4, **kwargs) -> PSKModulator:
    """Create PSK modulator with given parameters"""
    return PSKModulator(m=m, **kwargs)


def create_qam_modulator(m: int = 16, symbol_rate: float = 1e6, sample_rate: float = 10e6) -> QAMModulator:
    """Create QAM modulator with given parameters"""
    return QAMModulator(m=m, sample_rate=sample_rate, symbol_rate=symbol_rate)


def create_fsk_modulator(m: int = 2, symbol_rate: float = 1e6, sample_rate: float = 10e6) -> FSKModulator:
    """Create FSK modulator with given parameters"""
    freq_sep = symbol_rate / 2  # Default frequency separation
    return FSKModulator(m=m, frequency_separation=freq_sep, sample_rate=sample_rate)


def create_ofdm_modulator(n_subcarriers: int = 64, cp_length: int = 16) -> OFDMModulator:
    """Create OFDM modulator with given parameters"""
    return OFDMModulator(n_subcarriers=n_subcarriers, cp_length=cp_length)


# Utility functions  
def calculate_evm(reference: np.ndarray, measured: np.ndarray) -> float:
    """Calculate Error Vector Magnitude (EVM)"""
    if len(reference) != len(measured):
        # Handle length mismatch by truncating to shorter length
        min_len = min(len(reference), len(measured))
        reference = reference[:min_len]
        measured = measured[:min_len]
    
    if len(reference) == 0:
        return float('inf')
    
    # Normalize power
    ref_power = np.mean(np.abs(reference)**2)
    meas_power = np.mean(np.abs(measured)**2)
    
    if ref_power == 0 or meas_power == 0:
        return float('inf')
    
    # Calculate error vector
    error = measured - reference
    error_power = np.mean(np.abs(error)**2)
    
    # EVM as percentage
    evm = np.sqrt(error_power / ref_power) * 100
    return float(evm)


def estimate_snr(symbols: np.ndarray, constellation: np.ndarray) -> float:
    """Estimate SNR from received symbols and reference constellation"""
    if len(symbols) == 0 or len(constellation) == 0:
        return 0.0
    
    # Find closest constellation points
    distances = np.abs(symbols[:, np.newaxis] - constellation[np.newaxis, :])
    closest_indices = np.argmin(distances, axis=1)
    closest_symbols = constellation[closest_indices]
    
    # Calculate signal and noise power
    signal_power = np.mean(np.abs(closest_symbols)**2)
    noise_power = np.mean(np.abs(symbols - closest_symbols)**2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    return float(snr_db)


def plot_constellation(symbols: np.ndarray, title: str = "Constellation"):
    """Plot constellation diagram (stub for testing)"""
    # Simple return for testing - real implementation would use matplotlib
    return {"symbols": symbols, "title": title}
