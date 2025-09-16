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
    import sk_dsp_comm.digitalcomm as digitalcomm
    import sk_dsp_comm.synchronization as synchronization
    SCIKIT_DSP_AVAILABLE = True
except ImportError:
    SCIKIT_DSP_AVAILABLE = False
    digitalcomm = None
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

    def modulate(self, bits: np.ndarray, mod_type: str, **kwargs) -> np.ndarray:
        """Modulate bit sequence using specified modulation"""
        try:
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

    def fm_demodulate(self, signal: np.ndarray) -> np.ndarray:
        """FM demodulation using instantaneous frequency"""
        try:
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
