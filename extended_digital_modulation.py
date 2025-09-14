"""
Optimized Extended Digital Modulation Module - Research-Based Implementation
Based on IEEE 802.11, DVB-S2, LTE standards and digital communication theory
Module mở rộng cho điều chế số: FSK, GFSK, MSK, GMSK, QAM cao cấp, APSK
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt, lfilter, hilbert, sosfilt
from scipy.signal.windows import hann, hamming, blackmanharris
from scipy.special import erfc, erf
import warnings
from typing import Tuple, Optional, Union, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DigitalModulationType(Enum):
    """Standard digital modulation types"""
    BPSK = "bpsk"
    QPSK = "qpsk"
    OQPSK = "oqpsk"
    PSK8 = "8psk"
    DPSK = "dpsk"
    DQPSK = "dqpsk"
    QAM16 = "16qam"
    QAM64 = "64qam"
    QAM256 = "256qam"
    QAM1024 = "1024qam"
    APSK16 = "16apsk"
    APSK32 = "32apsk"
    FSK = "fsk"
    GFSK = "gfsk"
    MSK = "msk"
    GMSK = "gmsk"
    CPFSK = "cpfsk"
    OOK = "ook"
    ASK = "ask"

@dataclass
class DigitalModulationParams:
    """Parameters for digital modulation"""
    symbol_rate: float
    samples_per_symbol: int
    pulse_shape: str = 'rrc'  # 'rrc', 'rc', 'gaussian', 'rect'
    roll_off: float = 0.35
    bt_product: Optional[float] = None  # For GFSK/GMSK
    freq_deviation: Optional[float] = None  # For FSK variants
    constellation_scaling: str = 'avg_power'  # 'avg_power' or 'peak_power'
    phase_offset: float = 0.0
    differential: bool = False

@dataclass
class ConstellationPoint:
    """Constellation point with bit mapping"""
    complex_value: complex
    bit_sequence: List[int]
    symbol_index: int

@dataclass 
class DemodulationResult:
    """Result from digital demodulation"""
    demodulated_bits: np.ndarray
    soft_bits: Optional[np.ndarray]
    constellation_points: np.ndarray
    evm_rms: float  # Error Vector Magnitude
    evm_peak: float
    snr_estimate: float
    symbol_error_rate: float
    timing_error: float
    frequency_error: float
    phase_error: float
    quality_assessment: str


class OptimizedDigitalModulation:
    """Research-based digital modulation with standard compliance"""
    
    # Standard constellation mappings (Gray coding)
    GRAY_MAPPING = {
        4: [0b00, 0b01, 0b11, 0b10],      # QPSK Gray mapping
        8: [0b000, 0b001, 0b011, 0b010, 0b110, 0b111, 0b101, 0b100], # 8-PSK
        16: [0b0000, 0b0001, 0b0011, 0b0010, 0b0110, 0b0111, 0b0101, 0b0100,
             0b1100, 0b1101, 0b1111, 0b1110, 0b1010, 0b1011, 0b1001, 0b1000]  # 16-QAM
    }
    
    # DVB-S2 APSK constellation parameters
    APSK_PARAMS = {
        16: {'rings': [4, 12], 'radii': [1.0, 2.85], 'phases': [np.pi/4, 0]},
        32: {'rings': [4, 12, 16], 'radii': [1.0, 2.84, 5.27], 'phases': [np.pi/4, 0, np.pi/16]}
    }
    
    def __init__(self, sample_rate: float = 1e6):
        """
        Initialize digital modulation system
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.fs = sample_rate
        self.nyquist = sample_rate / 2
        
        # Pulse shaping filters cache
        self._pulse_filters = {}
        
        logger.info(f"Initialized DigitalModulation: Fs={self.fs/1e6:.1f} MHz")
    
    def generate_constellation(self, mod_type: str, M: int = None) -> List[ConstellationPoint]:
        """
        Generate standard constellation with Gray mapping
        
        Args:
            mod_type: Modulation type
            M: Constellation size (auto-detected if None)
            
        Returns:
            List of constellation points with bit mappings
        """
        constellation_points = []
        
        if mod_type == 'bpsk':
            points = [complex(-1, 0), complex(1, 0)]
            bits = [[0], [1]]
            
        elif mod_type == 'qpsk':
            # Standard QPSK constellation (Gray mapped)
            points = [complex(1, 1), complex(-1, 1), complex(-1, -1), complex(1, -1)]
            points = [p / np.sqrt(2) for p in points]  # Normalize to unit average power
            bits = [[0, 0], [0, 1], [1, 1], [1, 0]]
            
        elif mod_type == 'oqpsk':
            # OQPSK uses same constellation as QPSK but with offset
            points = [complex(1, 1), complex(-1, 1), complex(-1, -1), complex(1, -1)]
            points = [p / np.sqrt(2) for p in points]
            bits = [[0, 0], [0, 1], [1, 1], [1, 0]]
            
        elif mod_type == '8psk':
            # 8-PSK constellation with Gray mapping
            points = []
            bits_per_symbol = 3
            for i in range(8):
                angle = 2 * np.pi * i / 8 + np.pi/8  # 22.5° offset for optimal Gray mapping
                points.append(np.exp(1j * angle))
            
            # Gray code mapping for 8-PSK
            gray_bits = [[0,0,0], [0,0,1], [0,1,1], [0,1,0], 
                        [1,1,0], [1,1,1], [1,0,1], [1,0,0]]
            bits = gray_bits
            
        elif mod_type in ['16qam', '64qam', '256qam', '1024qam']:
            M = int(mod_type.replace('qam', ''))
            points, bits = self._generate_qam_constellation(M)
            
        elif mod_type in ['16apsk', '32apsk']:
            M = int(mod_type.replace('apsk', ''))
            points, bits = self._generate_apsk_constellation(M)
            
        elif mod_type == 'ook':
            # On-Off Keying
            points = [complex(0, 0), complex(1, 0)]
            bits = [[0], [1]]
            
        elif mod_type == 'ask':
            # Default 4-ASK
            M = M or 4
            points = []
            bits = []
            bits_per_symbol = int(np.log2(M))
            for i in range(M):
                amplitude = (2 * i + 1 - M) / (M - 1)
                points.append(complex(amplitude, 0))
                bit_seq = [(i >> bit) & 1 for bit in range(bits_per_symbol-1, -1, -1)]
                bits.append(bit_seq)
        
        elif mod_type in ['fsk', 'gfsk', 'msk', 'gmsk', 'cpfsk']:
            # FSK variants use binary symbols for frequency modulation
            points = [complex(-1, 0), complex(1, 0)]  # Binary symbols
            bits = [[0], [1]]
            
        else:
            raise ValueError(f"Unsupported modulation type: {mod_type}")
        
        # Create constellation point objects
        for i, (point, bit_seq) in enumerate(zip(points, bits)):
            constellation_points.append(ConstellationPoint(
                complex_value=point,
                bit_sequence=bit_seq,
                symbol_index=i
            ))
        
        return constellation_points
    
    def _generate_qam_constellation(self, M: int) -> Tuple[List[complex], List[List[int]]]:
        """Generate square QAM constellation with Gray mapping"""
        if M not in [16, 64, 256, 1024]:
            raise ValueError(f"Unsupported QAM order: {M}")
        
        bits_per_symbol = int(np.log2(M))
        side_length = int(np.sqrt(M))
        
        points = []
        bit_sequences = []
        
        # Generate constellation points
        for i in range(side_length):
            for q in range(side_length):
                # Map to symmetric constellation
                real_part = 2 * i - (side_length - 1)
                imag_part = 2 * q - (side_length - 1)
                
                point = complex(real_part, imag_part)
                points.append(point)
                
                # Gray code mapping
                gray_index = self._binary_to_gray(i) * side_length + self._binary_to_gray(q)
                bit_seq = [(gray_index >> bit) & 1 for bit in range(bits_per_symbol-1, -1, -1)]
                bit_sequences.append(bit_seq)
        
        # Normalize to unit average power
        avg_power = np.mean([abs(p)**2 for p in points])
        points = [p / np.sqrt(avg_power) for p in points]
        
        return points, bit_sequences
    
    def _generate_apsk_constellation(self, M: int) -> Tuple[List[complex], List[List[int]]]:
        """Generate APSK constellation (DVB-S2 standard)"""
        if M not in [16, 32]:
            raise ValueError(f"Unsupported APSK order: {M}")
        
        params = self.APSK_PARAMS[M]
        points = []
        bit_sequences = []
        
        symbol_idx = 0
        bits_per_symbol = int(np.log2(M))
        
        # Generate points for each ring
        for ring_idx, (num_points, radius, phase_offset) in enumerate(
            zip(params['rings'], params['radii'], params['phases'])):
            
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points + phase_offset
                point = radius * np.exp(1j * angle)
                points.append(point)
                
                # Gray code mapping
                gray_index = self._binary_to_gray(symbol_idx)
                bit_seq = [(gray_index >> bit) & 1 for bit in range(bits_per_symbol-1, -1, -1)]
                bit_sequences.append(bit_seq)
                
                symbol_idx += 1
        
        # Normalize outer ring to unit average power
        avg_power = np.mean([abs(p)**2 for p in points])
        points = [p / np.sqrt(avg_power) for p in points]
        
        return points, bit_sequences
    
    def _binary_to_gray(self, n: int) -> int:
        """Convert binary to Gray code"""
        return n ^ (n >> 1)
    
    def generate_pulse_shape_filter(self, params: DigitalModulationParams) -> np.ndarray:
        """
        Generate pulse shaping filter (RRC, RC, Gaussian)
        
        Args:
            params: Modulation parameters
            
        Returns:
            Filter impulse response
        """
        # Cache key for filter reuse
        cache_key = (params.pulse_shape, params.samples_per_symbol, 
                    params.roll_off, params.bt_product)
        
        if cache_key in self._pulse_filters:
            return self._pulse_filters[cache_key]
        
        sps = params.samples_per_symbol
        
        if params.pulse_shape == 'rrc':
            # Root Raised Cosine (Nyquist filter)
            span = 10  # Filter span in symbols
            h = self._rrc_filter(sps, params.roll_off, span)
            
        elif params.pulse_shape == 'rc':
            # Raised Cosine
            span = 10
            h = self._rc_filter(sps, params.roll_off, span)
            
        elif params.pulse_shape == 'gaussian':
            # Gaussian filter for GFSK/GMSK
            if params.bt_product is None:
                params.bt_product = 0.3  # Default BT
            h = self._gaussian_filter(sps, params.bt_product)
            
        elif params.pulse_shape == 'rect':
            # Rectangular (NRZ) pulse
            h = np.ones(sps)
            
        else:
            raise ValueError(f"Unknown pulse shape: {params.pulse_shape}")
        
        # Normalize filter
        h = h / np.sqrt(np.sum(h**2))
        
        # Cache the filter
        self._pulse_filters[cache_key] = h
        
        return h
    
    def _rrc_filter(self, sps: int, roll_off: float, span: int) -> np.ndarray:
        """Root Raised Cosine filter (matched filter pair)"""
        # Time vector
        t = np.arange(-span*sps//2, span*sps//2 + 1) / sps
        
        # Handle special cases
        h = np.zeros(len(t))
        
        for i, time in enumerate(t):
            if time == 0:
                h[i] = 1 - roll_off + (4 * roll_off / np.pi)
            elif abs(time) == 1 / (4 * roll_off):
                h[i] = (roll_off / np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*roll_off)) + 
                                                 (1 - 2/np.pi) * np.cos(np.pi/(4*roll_off)))
            else:
                numerator = np.sin(np.pi * time * (1 - roll_off)) + \
                           4 * roll_off * time * np.cos(np.pi * time * (1 + roll_off))
                denominator = np.pi * time * (1 - (4 * roll_off * time)**2)
                h[i] = numerator / denominator
        
        return h
    
    def _rc_filter(self, sps: int, roll_off: float, span: int) -> np.ndarray:
        """Raised Cosine filter"""
        # Time vector
        t = np.arange(-span*sps//2, span*sps//2 + 1) / sps
        
        h = np.zeros(len(t))
        
        for i, time in enumerate(t):
            if time == 0:
                h[i] = 1
            elif abs(time) == 1 / (2 * roll_off):
                h[i] = (np.pi / 4) * np.sinc(1 / (2 * roll_off))
            else:
                h[i] = np.sinc(time) * np.cos(np.pi * roll_off * time) / (1 - (2 * roll_off * time)**2)
        
        return h
    
    def _gaussian_filter(self, sps: int, bt_product: float) -> np.ndarray:
        """Gaussian filter for GFSK/GMSK"""
        # Time vector (4 symbol periods)
        span = 4
        t = np.arange(-span*sps//2, span*sps//2 + 1) / sps
        
        # Gaussian filter parameter
        alpha = np.sqrt(np.log(2) / 2) / bt_product
        
        # Gaussian impulse response
        h = np.exp(-2 * (np.pi * alpha * t)**2)
        
        return h
    
    def modulate_signal(self, bits: np.ndarray, mod_type: str, 
                       params: DigitalModulationParams) -> np.ndarray:
        """
        Modulate digital signal with specified parameters
        
        Args:
            bits: Input bit sequence
            mod_type: Modulation type
            params: Modulation parameters
            
        Returns:
            Modulated complex signal
        """
        # Generate constellation
        constellation = self.generate_constellation(mod_type)
        bits_per_symbol = len(constellation[0].bit_sequence)
        
        # Group bits into symbols
        padded_bits = self._pad_bits(bits, bits_per_symbol)
        symbols = []
        
        for i in range(0, len(padded_bits), bits_per_symbol):
            bit_group = padded_bits[i:i+bits_per_symbol].tolist()
            
            # Find matching constellation point
            for point in constellation:
                if point.bit_sequence == bit_group:
                    symbols.append(point.complex_value)
                    break
            else:
                # Default to first symbol if no match (shouldn't happen)
                symbols.append(constellation[0].complex_value)
        
        symbols = np.array(symbols)
        
        # Apply differential encoding if required
        if params.differential:
            symbols = self._differential_encode(symbols)
        
        # Apply phase offset
        if params.phase_offset != 0:
            symbols *= np.exp(1j * params.phase_offset)
        
        # Upsample and pulse shape
        upsampled = self._upsample_symbols(symbols, params.samples_per_symbol)
        
        # Apply pulse shaping filter
        if params.pulse_shape != 'rect':
            pulse_filter = self.generate_pulse_shape_filter(params)
            modulated = np.convolve(upsampled, pulse_filter, mode='same')
        else:
            modulated = upsampled
        
        # Handle special modulation types
        if mod_type in ['fsk', 'gfsk', 'msk', 'gmsk', 'cpfsk']:
            modulated = self._apply_frequency_modulation(modulated, mod_type, params)
        
        return modulated
    
    def _pad_bits(self, bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
        """Pad bit sequence to make it divisible by bits_per_symbol"""
        remainder = len(bits) % bits_per_symbol
        if remainder != 0:
            padding = np.zeros(bits_per_symbol - remainder, dtype=int)
            return np.concatenate([bits, padding])
        return bits
    
    def _differential_encode(self, symbols: np.ndarray) -> np.ndarray:
        """Apply differential encoding to symbols"""
        diff_symbols = np.zeros_like(symbols)
        diff_symbols[0] = symbols[0]  # Reference symbol
        
        for i in range(1, len(symbols)):
            diff_symbols[i] = diff_symbols[i-1] * symbols[i]
        
        return diff_symbols
    
    def _upsample_symbols(self, symbols: np.ndarray, sps: int) -> np.ndarray:
        """Upsample symbols by inserting zeros"""
        upsampled = np.zeros(len(symbols) * sps, dtype=complex)
        upsampled[::sps] = symbols
        return upsampled
    
    def _apply_frequency_modulation(self, signal: np.ndarray, mod_type: str, 
                                  params: DigitalModulationParams) -> np.ndarray:
        """Apply frequency modulation for FSK variants"""
        if mod_type == 'fsk':
            # Binary FSK
            return self._fsk_modulate(signal, params)
        elif mod_type == 'gfsk':
            # Gaussian FSK
            return self._gfsk_modulate(signal, params)
        elif mod_type in ['msk', 'gmsk']:
            # MSK/GMSK (continuous phase)
            return self._msk_modulate(signal, params)
        elif mod_type == 'cpfsk':
            # Continuous Phase FSK
            return self._cpfsk_modulate(signal, params)
        else:
            return signal
    
    def _fsk_modulate(self, symbols: np.ndarray, params: DigitalModulationParams) -> np.ndarray:
        """Binary FSK modulation"""
        freq_dev = params.freq_deviation or (params.symbol_rate / 4)
        
        # Convert symbols to frequency deviation
        real_symbols = np.real(symbols)  # Assume BPSK symbols (-1, +1)
        freq_deviations = real_symbols * freq_dev
        
        # Generate FSK signal
        t = np.arange(len(symbols)) / self.fs
        phase = 2 * np.pi * np.cumsum(freq_deviations) / self.fs
        
        fsk_signal = np.exp(1j * phase)
        
        return fsk_signal
    
    def _gfsk_modulate(self, symbols: np.ndarray, params: DigitalModulationParams) -> np.ndarray:
        """Gaussian FSK modulation"""
        freq_dev = params.freq_deviation or (params.symbol_rate / 4)
        bt_product = params.bt_product or 0.3
        
        # Convert to NRZ
        nrz = np.real(symbols)
        
        # Apply Gaussian filter
        gaussian_filter = self._gaussian_filter(params.samples_per_symbol, bt_product)
        filtered_nrz = np.convolve(nrz, gaussian_filter, mode='same')
        
        # FM modulation
        phase = 2 * np.pi * freq_dev * np.cumsum(filtered_nrz) / self.fs
        gfsk_signal = np.exp(1j * phase)
        
        return gfsk_signal
    
    def _msk_modulate(self, symbols: np.ndarray, params: DigitalModulationParams) -> np.ndarray:
        """MSK/GMSK modulation"""
        # MSK is CPFSK with modulation index h = 0.5
        modulation_index = 0.5
        
        # Convert symbols to phase changes
        real_symbols = np.real(symbols)
        phase_changes = real_symbols * np.pi * modulation_index
        
        # Cumulative phase (continuous phase)
        cumulative_phase = np.cumsum(phase_changes)
        
        # Generate MSK signal
        msk_signal = np.exp(1j * cumulative_phase)
        
        # Apply Gaussian filtering for GMSK
        if params.pulse_shape == 'gaussian':
            bt_product = params.bt_product or 0.3
            gaussian_filter = self._gaussian_filter(params.samples_per_symbol, bt_product)
            
            # Filter the phase trajectory
            filtered_phase = np.convolve(np.real(np.log(msk_signal + 1e-12)), 
                                       gaussian_filter, mode='same')
            msk_signal = np.exp(1j * filtered_phase)
        
        return msk_signal
    
    def _cpfsk_modulate(self, symbols: np.ndarray, params: DigitalModulationParams) -> np.ndarray:
        """Continuous Phase FSK modulation"""
        modulation_index = 0.5  # Default modulation index
        
        # Convert symbols to phase increments
        real_symbols = np.real(symbols)
        phase_increments = real_symbols * np.pi * modulation_index
        
        # Generate continuous phase
        phase = np.cumsum(phase_increments)
        
        # CPFSK signal
        cpfsk_signal = np.exp(1j * phase)
        
        return cpfsk_signal


class OptimizedDigitalDemodulation:
    """Research-based digital demodulation with carrier/timing recovery"""
    
    def __init__(self, sample_rate: float = 1e6):
        """Initialize digital demodulation system"""
        self.fs = sample_rate
        self.nyquist = sample_rate / 2
        
        # Carrier recovery parameters
        self.costas_loop_bw = 0.01  # Loop bandwidth
        self.timing_loop_bw = 0.01  # Timing recovery bandwidth
        
        # Backward compatibility attributes
        self.symbol_rate = 10000  # Default symbol rate
        self.samples_per_symbol = int(sample_rate / self.symbol_rate)
        
        logger.info(f"Initialized DigitalDemodulation: Fs={self.fs/1e6:.1f} MHz")
    
    def demodulate_signal(self, received_signal: np.ndarray, mod_type: str,
                         params: DigitalModulationParams,
                         enable_carrier_recovery: bool = True,
                         enable_timing_recovery: bool = True) -> DemodulationResult:
        """
        Demodulate digital signal with carrier and timing recovery
        
        Args:
            received_signal: Received complex signal
            mod_type: Modulation type
            params: Demodulation parameters
            enable_carrier_recovery: Enable carrier frequency/phase recovery
            enable_timing_recovery: Enable symbol timing recovery
            
        Returns:
            Demodulation result with quality metrics
        """
        # Matched filtering (if pulse shaped)
        if params.pulse_shape != 'rect':
            pulse_filter = OptimizedDigitalModulation(self.fs).generate_pulse_shape_filter(params)
            matched_filtered = np.convolve(received_signal, np.conj(pulse_filter[::-1]), mode='same')
        else:
            matched_filtered = received_signal
        
        # Carrier recovery
        if enable_carrier_recovery:
            recovered_signal, freq_error, phase_error = self._costas_loop(matched_filtered, mod_type)
        else:
            recovered_signal = matched_filtered
            freq_error = 0.0
            phase_error = 0.0
        
        # Timing recovery
        if enable_timing_recovery:
            symbols, timing_error = self._timing_recovery(recovered_signal, params)
        else:
            # Simple downsampling at expected symbol times
            symbols = recovered_signal[::params.samples_per_symbol]
            timing_error = 0.0
        
        # Generate reference constellation
        constellation = OptimizedDigitalModulation(self.fs).generate_constellation(mod_type)
        
        # Demodulate symbols to bits
        demod_bits, soft_bits = self._symbol_to_bits(symbols, constellation)
        
        # Calculate quality metrics
        evm_rms, evm_peak = self._calculate_evm(symbols, constellation)
        snr_estimate = self._estimate_snr_from_constellation(symbols, constellation)
        ser = self._estimate_symbol_error_rate(symbols, constellation)
        
        # Quality assessment
        quality = self._assess_demodulation_quality(evm_rms, snr_estimate, ser)
        
        result = DemodulationResult(
            demodulated_bits=demod_bits,
            soft_bits=soft_bits,
            constellation_points=symbols,
            evm_rms=evm_rms,
            evm_peak=evm_peak,
            snr_estimate=snr_estimate,
            symbol_error_rate=ser,
            timing_error=timing_error,
            frequency_error=freq_error,
            phase_error=phase_error,
            quality_assessment=quality
        )
        
        return result
    
    def _costas_loop(self, signal: np.ndarray, mod_type: str) -> Tuple[np.ndarray, float, float]:
        """Costas loop for carrier recovery"""
        # Loop parameters
        alpha = self.costas_loop_bw
        beta = alpha**2 / 4
        
        # Initialize loop state
        nco_phase = 0.0
        nco_freq = 0.0
        loop_filter_state = 0.0
        
        recovered_signal = np.zeros_like(signal)
        freq_errors = []
        phase_errors = []
        
        for i, sample in enumerate(signal):
            # NCO output
            nco_out = np.exp(-1j * nco_phase)
            recovered_sample = sample * nco_out
            recovered_signal[i] = recovered_sample
            
            # Phase detector (depends on modulation type)
            if mod_type in ['bpsk', 'qpsk']:
                # BPSK/QPSK Costas loop
                if mod_type == 'bpsk':
                    phase_error = np.imag(recovered_sample) * np.sign(np.real(recovered_sample))
                else:  # QPSK
                    phase_error = np.imag(recovered_sample) * np.sign(np.real(recovered_sample)) + \
                                 np.real(recovered_sample) * np.sign(np.imag(recovered_sample))
            else:
                # Generic phase detector for higher-order modulations
                # Use 4th power method for QAM
                raised_signal = recovered_sample**4
                phase_error = np.imag(raised_signal)
            
            phase_errors.append(phase_error)
            
            # Loop filter (2nd order)
            loop_filter_state += beta * phase_error
            nco_freq = loop_filter_state + alpha * phase_error
            freq_errors.append(nco_freq)
            
            # NCO update
            nco_phase += nco_freq
            
            # Keep phase in [-π, π]
            while nco_phase > np.pi:
                nco_phase -= 2 * np.pi
            while nco_phase < -np.pi:
                nco_phase += 2 * np.pi
        
        avg_freq_error = np.mean(freq_errors) * self.fs / (2 * np.pi)  # Convert to Hz
        avg_phase_error = np.mean(np.abs(phase_errors))
        
        return recovered_signal, avg_freq_error, avg_phase_error
    
    def _timing_recovery(self, signal: np.ndarray, 
                        params: DigitalModulationParams) -> Tuple[np.ndarray, float]:
        """Symbol timing recovery using Mueller & Müller algorithm"""
        sps = params.samples_per_symbol
        
        # Initialize timing recovery
        mu = 0.0  # Timing offset
        mu_step = 1.0 / sps
        timing_gain = self.timing_loop_bw
        
        symbols = []
        timing_errors = []
        sample_idx = sps // 2  # Start at middle of first symbol
        
        prev_symbol = 0
        prev_sample = 0
        
        while sample_idx < len(signal) - sps:
            # Interpolate current sample
            interp_idx = int(sample_idx)
            frac = sample_idx - interp_idx
            
            if interp_idx + 1 < len(signal):
                current_sample = signal[interp_idx] * (1 - frac) + signal[interp_idx + 1] * frac
            else:
                current_sample = signal[interp_idx]
            
            symbols.append(current_sample)
            
            # Mueller & Müller timing error detector
            if len(symbols) > 1:
                timing_error = np.real((current_sample - prev_sample) * np.conj(prev_symbol))
                timing_errors.append(timing_error)
                
                # Update timing
                mu += timing_gain * timing_error
                
                # Clamp mu to prevent runaway
                mu = np.clip(mu, -0.5, 0.5)
            
            prev_symbol = current_sample
            prev_sample = signal[int(sample_idx)]
            
            # Advance to next symbol
            sample_idx += sps + mu
        
        avg_timing_error = np.mean(np.abs(timing_errors)) if timing_errors else 0.0
        
        return np.array(symbols), avg_timing_error
    
    def _symbol_to_bits(self, symbols: np.ndarray, 
                       constellation: List[ConstellationPoint]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert symbols to bits using ML detection"""
        demod_bits = []
        soft_bits = []
        
        # Extract constellation points and bit mappings
        constellation_points = np.array([point.complex_value for point in constellation])
        bit_mappings = [point.bit_sequence for point in constellation]
        bits_per_symbol = len(bit_mappings[0])
        
        for symbol in symbols:
            # Calculate distances to all constellation points
            distances = np.abs(symbol - constellation_points)**2
            
            # Hard decision (minimum distance)
            min_idx = np.argmin(distances)
            demod_bits.extend(bit_mappings[min_idx])
            
            # Soft decision (log-likelihood ratios)
            for bit_pos in range(bits_per_symbol):
                # Separate constellation points by bit value at this position
                bit_0_indices = [i for i, mapping in enumerate(bit_mappings) 
                               if mapping[bit_pos] == 0]
                bit_1_indices = [i for i, mapping in enumerate(bit_mappings) 
                               if mapping[bit_pos] == 1]
                
                # Calculate log-likelihood ratio
                if bit_0_indices and bit_1_indices:
                    min_dist_0 = np.min(distances[bit_0_indices])
                    min_dist_1 = np.min(distances[bit_1_indices])
                    
                    # LLR = log(P(bit=0)/P(bit=1)) ∝ (dist_1 - dist_0)
                    llr = min_dist_1 - min_dist_0
                    soft_bits.append(llr)
                else:
                    soft_bits.append(0.0)  # No information
        
        return np.array(demod_bits), np.array(soft_bits)
    
    def _calculate_evm(self, symbols: np.ndarray, 
                      constellation: List[ConstellationPoint]) -> Tuple[float, float]:
        """Calculate Error Vector Magnitude"""
        constellation_points = np.array([point.complex_value for point in constellation])
        
        error_magnitudes = []
        
        for symbol in symbols:
            # Find closest constellation point
            distances = np.abs(symbol - constellation_points)
            closest_idx = np.argmin(distances)
            closest_point = constellation_points[closest_idx]
            
            # Error vector magnitude
            error_magnitude = np.abs(symbol - closest_point)
            error_magnitudes.append(error_magnitude)
        
        if not error_magnitudes:
            return 0.0, 0.0
        
        # Calculate reference power (average constellation power)
        reference_power = np.mean(np.abs(constellation_points)**2)
        
        # EVM as percentage
        error_magnitudes = np.array(error_magnitudes)
        evm_rms = np.sqrt(np.mean(error_magnitudes**2)) / np.sqrt(reference_power) * 100
        evm_peak = np.max(error_magnitudes) / np.sqrt(reference_power) * 100
        
        return evm_rms, evm_peak
    
    def _estimate_snr_from_constellation(self, symbols: np.ndarray,
                                       constellation: List[ConstellationPoint]) -> float:
        """Estimate SNR from constellation analysis"""
        if len(symbols) == 0:
            return 0.0
        
        constellation_points = np.array([point.complex_value for point in constellation])
        
        # Calculate signal power (average constellation power)
        signal_power = np.mean(np.abs(constellation_points)**2)
        
        # Estimate noise power from constellation spread
        noise_powers = []
        
        for symbol in symbols:
            distances = np.abs(symbol - constellation_points)
            closest_idx = np.argmin(distances)
            error = symbol - constellation_points[closest_idx]
            noise_powers.append(np.abs(error)**2)
        
        if not noise_powers:
            return 30.0  # High SNR default
        
        noise_power = np.mean(noise_powers)
        
        if noise_power == 0:
            return 40.0  # Very high SNR
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(max(snr_linear, 1e-10))
        
        return np.clip(snr_db, -10, 40)
    
    def _estimate_symbol_error_rate(self, symbols: np.ndarray,
                                   constellation: List[ConstellationPoint]) -> float:
        """Estimate symbol error rate"""
        if len(symbols) == 0:
            return 1.0
        
        constellation_points = np.array([point.complex_value for point in constellation])
        
        # For SER estimation, we need to know the transmitted symbols
        # This is a simplified estimation based on decision regions
        
        errors = 0
        total_symbols = len(symbols)
        
        # Simple SER estimation: symbols that are far from any constellation point
        error_threshold = np.mean([np.min(np.abs(cp - constellation_points)) 
                                 for cp in constellation_points]) * 2
        
        for symbol in symbols:
            distances = np.abs(symbol - constellation_points)
            min_distance = np.min(distances)
            
            if min_distance > error_threshold:
                errors += 1
        
        ser = errors / total_symbols if total_symbols > 0 else 1.0
        return np.clip(ser, 0.0, 1.0)
    
    def _assess_demodulation_quality(self, evm_rms: float, snr_db: float, ser: float) -> str:
        """Assess overall demodulation quality"""
        if evm_rms < 2 and snr_db > 25 and ser < 0.01:
            return "excellent"
        elif evm_rms < 5 and snr_db > 20 and ser < 0.05:
            return "very_good"
        elif evm_rms < 10 and snr_db > 15 and ser < 0.1:
            return "good"
        elif evm_rms < 20 and snr_db > 10 and ser < 0.2:
            return "fair"
        else:
            return "poor"


# Backward compatibility aliases
ExtendedDigitalModulation = OptimizedDigitalModulation
ExtendedDigitalDemodulation = OptimizedDigitalDemodulation


class AdvancedModulationClassifier:
    """Advanced classifier for extended digital modulations"""

    def __init__(self, sample_rate=1e6):
        self.fs = sample_rate

    def classify_modulation(self, iq_signal):
        """Classify modulation type from IQ signal"""
        try:
            features = self._extract_comprehensive_features(iq_signal)
            return self._advanced_classification(features)
        except Exception as e:
            print(f"Modulation classification error: {e}")
            return "unknown"

    def _extract_comprehensive_features(self, iq_signal):
        """Extract comprehensive feature set"""
        # Amplitude features
        magnitude = np.abs(iq_signal)
        mag_var = np.var(magnitude)
        mag_kurt = self._kurtosis(magnitude)

        # Phase features
        phase = np.angle(iq_signal)
        phase_diff = np.diff(np.unwrap(phase))
        phase_var = np.var(phase_diff)

        # Higher-order moments
        c20 = np.mean(iq_signal**2)
        c21 = np.mean(iq_signal**2 * np.conj(iq_signal))
        c40 = np.mean(iq_signal**4) - 3*(np.mean(np.abs(iq_signal)**2))**2
        c41 = np.mean(iq_signal**3 * np.conj(iq_signal))
        c42 = np.mean(iq_signal**2 * (np.conj(iq_signal))**2) - (np.mean(np.abs(iq_signal)**2))**2

        # Spectral features
        spectrum = np.abs(fft(iq_signal))**2
        spectral_centroid = np.sum(np.arange(len(spectrum)) * spectrum) / np.sum(spectrum)

        # Constellation compactness
        constellation_compactness = self._constellation_compactness(iq_signal)

        return {
            'magnitude_variance': mag_var,
            'magnitude_kurtosis': mag_kurt,
            'phase_variance': phase_var,
            'c20': abs(c20),
            'c21': abs(c21),
            'c40': abs(c40),
            'c41': abs(c41),
            'c42': abs(c42),
            'spectral_centroid': spectral_centroid,
            'constellation_compactness': constellation_compactness
        }

    def _advanced_classification(self, features):
        """Advanced classification based on features"""
        # Simplified classification logic
        mag_var = features['magnitude_variance']
        phase_var = features['phase_variance']
        c40 = features['c40']
        
        # Classification thresholds (simplified)
        if mag_var < 0.01 and phase_var > 1.0:
            return 'psk'
        elif mag_var > 0.1 and c40 > 0.1:
            return 'qam'
        elif phase_var > 2.0:
            return 'fsk'
        else:
            return 'bpsk'

    def _kurtosis(self, data):
        """Calculate kurtosis"""
        if len(data) == 0:
            return 0
        mean = np.mean(data)
        var = np.var(data)
        if var == 0:
            return 0
        return np.mean(((data - mean) / np.sqrt(var))**4) - 3

    def _constellation_compactness(self, iq_signal):
        """Calculate constellation compactness"""
        # Downsample to get constellation points
        decimation = max(1, len(iq_signal) // 1000)
        constellation_points = iq_signal[::decimation]
        
        if len(constellation_points) == 0:
            return 0
        
        # Calculate spread
        center = np.mean(constellation_points)
        distances = np.abs(constellation_points - center)
        return np.std(distances)


# Testing functions
def test_optimized_digital_modulation():
    """Test optimized digital modulation implementation"""
    print("Testing Optimized Digital Modulation...")
    
    # Test parameters
    fs = 1e6  # 1 MHz sample rate
    symbol_rate = 100e3  # 100 kHz symbol rate
    sps = int(fs / symbol_rate)  # Samples per symbol
    
    # Test bit sequence
    test_bits = np.random.randint(0, 2, 1000)
    
    # Modulation parameters
    params = DigitalModulationParams(
        symbol_rate=symbol_rate,
        samples_per_symbol=sps,
        pulse_shape='rrc',
        roll_off=0.35
    )
    
    # Test modulation
    modulator = OptimizedDigitalModulation(fs)
    
    # Test QPSK
    qpsk_signal = modulator.modulate_signal(test_bits, 'qpsk', params)
    print(f"QPSK signal generated: {len(qpsk_signal)} samples")
    
    # Test 16-QAM
    qam16_signal = modulator.modulate_signal(test_bits, '16qam', params)
    print(f"16-QAM signal generated: {len(qam16_signal)} samples")
    
    # Test GFSK
    gfsk_params = DigitalModulationParams(
        symbol_rate=symbol_rate,
        samples_per_symbol=sps,
        pulse_shape='gaussian',
        bt_product=0.3,
        freq_deviation=symbol_rate/4
    )
    gfsk_signal = modulator.modulate_signal(test_bits, 'gfsk', gfsk_params)
    print(f"GFSK signal generated: {len(gfsk_signal)} samples")
    
    # Add some noise
    snr_db = 20
    noise_power = 10**(-snr_db/10)
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(qpsk_signal)) + 
                                     1j * np.random.randn(len(qpsk_signal)))
    noisy_qpsk = qpsk_signal + noise
    
    # Test demodulation
    demodulator = OptimizedDigitalDemodulation(fs)
    
    # QPSK demodulation
    qpsk_result = demodulator.demodulate_signal(noisy_qpsk, 'qpsk', params)
    print(f"QPSK Demod - EVM: {qpsk_result.evm_rms:.2f}%, SNR: {qpsk_result.snr_estimate:.1f} dB")
    print(f"Quality: {qpsk_result.quality_assessment}")
    
    # Test constellation generation
    qpsk_constellation = modulator.generate_constellation('qpsk')
    print(f"QPSK constellation points: {len(qpsk_constellation)}")
    
    apsk_constellation = modulator.generate_constellation('16apsk')
    print(f"16-APSK constellation points: {len(apsk_constellation)}")
    
    print("✅ Optimized digital modulation tests completed!")

if __name__ == "__main__":
    test_optimized_digital_modulation()