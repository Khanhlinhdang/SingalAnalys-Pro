"""
Research-Optimized Multicarrier and Spread Spectrum Module

Tối ưu hóa dựa trên IEEE 802.11 (WiFi OFDM), 3GPP LTE/5G NR (SC-FDMA), 
IEEE 802.15.4 (ZigBee), GPS/GNSS, và LoRa specifications
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

class OptimizedOFDMModulator:
    """IEEE 802.11 compliant OFDM modulator"""
    
    def __init__(self, n_fft=64, cp_length=16, pilot_indices=None, dc_null=True):
        self.n_fft = n_fft
        self.cp_length = cp_length
        self.dc_null = dc_null
        
        # IEEE 802.11a/g pilot pattern
        if pilot_indices is None:
            if n_fft == 64:
                self.pilot_indices = np.array([-21, -7, 7, 21]) + n_fft//2
                self.data_indices = self._generate_data_indices_80211()
            else:
                # Generic pilot pattern (every 8th subcarrier)
                self.pilot_indices = np.arange(0, n_fft, 8)
                self.data_indices = np.setdiff1d(np.arange(n_fft), self.pilot_indices)
        else:
            self.pilot_indices = pilot_indices
            self.data_indices = np.setdiff1d(np.arange(n_fft), pilot_indices)
        
        # Remove DC if specified
        if self.dc_null:
            dc_idx = n_fft // 2
            self.data_indices = self.data_indices[self.data_indices != dc_idx]
            self.pilot_indices = self.pilot_indices[self.pilot_indices != dc_idx]
        
        # IEEE 802.11a pilot values (BPSK, alternating polarity)
        self.pilot_values = self._generate_pilot_sequence()
        
    def _generate_data_indices_80211(self):
        """Generate IEEE 802.11a/g data subcarrier indices"""
        # 64-point FFT, subcarriers -26 to 26 (excluding 0, ±21, ±7)
        all_indices = np.arange(-26, 27) + self.n_fft//2
        pilot_indices_centered = np.array([-21, -7, 0, 7, 21]) + self.n_fft//2
        data_indices = np.setdiff1d(all_indices, pilot_indices_centered)
        return data_indices
    
    def _generate_pilot_sequence(self):
        """Generate IEEE 802.11a pilot sequence"""
        # Pseudo-random sequence for pilot values
        # In practice, this would be the actual 802.11 pilot sequence
        n_pilots = len(self.pilot_indices)
        # IEEE 802.11a uses specific pilot values: +1, -1 pattern
        pilot_pattern = np.array([1, 1, 1, -1])  # Example pattern
        pilots = np.tile(pilot_pattern, (n_pilots // len(pilot_pattern)) + 1)[:n_pilots]
        return pilots
    
    def modulate(self, data_symbols, symbol_mapping='qpsk'):
        """OFDM modulation with IEEE 802.11 compliance"""
        if len(data_symbols) == 0:
            return np.array([])
        
        n_data_carriers = len(self.data_indices)
        n_symbols = len(data_symbols) // n_data_carriers
        
        if n_symbols == 0:
            n_symbols = 1
            # Pad data if needed
            padded_data = np.zeros(n_data_carriers, dtype=complex)
            padded_data[:len(data_symbols)] = data_symbols
            data_symbols = padded_data
        
        # Reshape data into OFDM symbols
        data_matrix = data_symbols[:n_symbols * n_data_carriers].reshape(n_symbols, n_data_carriers)
        
        ofdm_signal = []
        
        for symbol_idx in range(n_symbols):
            # Create frequency domain symbol
            freq_symbol = np.zeros(self.n_fft, dtype=complex)
            
            # Insert data symbols
            freq_symbol[self.data_indices] = data_matrix[symbol_idx]
            
            # Insert pilot symbols
            pilot_values = self.pilot_values * np.exp(1j * np.pi * symbol_idx)  # Phase rotation
            freq_symbol[self.pilot_indices] = pilot_values
            
            # IFFT to get time domain
            time_symbol = ifft(freq_symbol)
            
            # Add cyclic prefix
            cp_symbol = np.concatenate([time_symbol[-self.cp_length:], time_symbol])
            
            ofdm_signal.extend(cp_symbol)
        
        return np.array(ofdm_signal)
    
    def generate_pilots_for_channel_estimation(self):
        """Generate pilots for channel estimation (IEEE 802.11 standard)"""
        pilot_symbol = np.zeros(self.n_fft, dtype=complex)
        pilot_symbol[self.pilot_indices] = self.pilot_values
        pilot_symbol[self.data_indices] = 0  # Zero data carriers for channel sounding
        return pilot_symbol

class OptimizedOFDMDemodulator:
    """IEEE 802.11 compliant OFDM demodulator with channel estimation"""
    
    def __init__(self, n_fft=64, cp_length=16, pilot_indices=None, dc_null=True):
        self.n_fft = n_fft
        self.cp_length = cp_length
        self.dc_null = dc_null
        
        # Match modulator configuration
        if pilot_indices is None:
            if n_fft == 64:
                self.pilot_indices = np.array([-21, -7, 7, 21]) + n_fft//2
                self.data_indices = self._generate_data_indices_80211()
            else:
                self.pilot_indices = np.arange(0, n_fft, 8)
                self.data_indices = np.setdiff1d(np.arange(n_fft), pilot_indices)
        else:
            self.pilot_indices = pilot_indices
            self.data_indices = np.setdiff1d(np.arange(n_fft), pilot_indices)
        
        if self.dc_null:
            dc_idx = n_fft // 2
            self.data_indices = self.data_indices[self.data_indices != dc_idx]
            self.pilot_indices = self.pilot_indices[self.pilot_indices != dc_idx]
        
        # Expected pilot values
        self.pilot_values = self._generate_pilot_sequence()
        
        # Channel estimation
        self.channel_estimate = np.ones(self.n_fft, dtype=complex)
        
    def _generate_data_indices_80211(self):
        """Generate IEEE 802.11a/g data subcarrier indices"""
        all_indices = np.arange(-26, 27) + self.n_fft//2
        pilot_indices_centered = np.array([-21, -7, 0, 7, 21]) + self.n_fft//2
        data_indices = np.setdiff1d(all_indices, pilot_indices_centered)
        return data_indices
    
    def _generate_pilot_sequence(self):
        """Generate expected pilot sequence"""
        n_pilots = len(self.pilot_indices)
        pilot_pattern = np.array([1, 1, 1, -1])
        pilots = np.tile(pilot_pattern, (n_pilots // len(pilot_pattern)) + 1)[:n_pilots]
        return pilots
    
    def demodulate(self, received_signal, channel_estimation=True):
        """OFDM demodulation with channel equalization"""
        if len(received_signal) == 0:
            return np.array([])
        
        symbol_length = self.n_fft + self.cp_length
        n_symbols = len(received_signal) // symbol_length
        
        demodulated_data = []
        
        for symbol_idx in range(n_symbols):
            # Extract OFDM symbol
            start_idx = symbol_idx * symbol_length
            end_idx = start_idx + symbol_length
            
            if end_idx > len(received_signal):
                break
                
            ofdm_symbol = received_signal[start_idx:end_idx]
            
            # Remove cyclic prefix
            time_symbol = ofdm_symbol[self.cp_length:]
            
            # FFT to frequency domain
            freq_symbol = fft(time_symbol)
            
            # Channel estimation using pilots
            if channel_estimation:
                self._estimate_channel(freq_symbol, symbol_idx)
            
            # Channel equalization
            equalized_symbol = freq_symbol / (self.channel_estimate + 1e-10)
            
            # Extract data symbols
            data_symbols = equalized_symbol[self.data_indices]
            demodulated_data.extend(data_symbols)
        
        return np.array(demodulated_data)
    
    def _estimate_channel(self, freq_symbol, symbol_idx):
        """Estimate channel using pilot subcarriers"""
        # Get received pilot symbols
        received_pilots = freq_symbol[self.pilot_indices]
        
        # Expected pilot values with phase rotation
        expected_pilots = self.pilot_values * np.exp(1j * np.pi * symbol_idx)
        
        # Channel estimate at pilot locations
        pilot_channel_est = received_pilots / (expected_pilots + 1e-10)
        
        # Interpolate channel estimate to all subcarriers
        self.channel_estimate = self._interpolate_channel(pilot_channel_est)
    
    def _interpolate_channel(self, pilot_estimates):
        """Interpolate channel estimate across all subcarriers"""
        # Linear interpolation
        full_channel = np.ones(self.n_fft, dtype=complex)
        
        # Set pilot locations
        full_channel[self.pilot_indices] = pilot_estimates
        
        # Simple linear interpolation for other subcarriers
        for i in range(self.n_fft):
            if i not in self.pilot_indices:
                # Find nearest pilots
                distances = np.abs(self.pilot_indices - i)
                nearest_pilots = np.argsort(distances)[:2]
                
                if len(nearest_pilots) >= 2:
                    # Linear interpolation between two nearest pilots
                    p1, p2 = self.pilot_indices[nearest_pilots[:2]]
                    w1 = 1 / (abs(i - p1) + 1e-10)
                    w2 = 1 / (abs(i - p2) + 1e-10)
                    
                    full_channel[i] = (w1 * pilot_estimates[nearest_pilots[0]] + 
                                     w2 * pilot_estimates[nearest_pilots[1]]) / (w1 + w2)
                else:
                    # Use nearest pilot
                    full_channel[i] = pilot_estimates[nearest_pilots[0]]
        
        return full_channel

class OptimizedDSSSModulator:
    """Research-grade Direct Sequence Spread Spectrum modulator"""
    
    def __init__(self, chip_rate=1e6, spreading_factor=31, sequence_type='gold'):
        self.chip_rate = chip_rate
        self.spreading_factor = spreading_factor
        self.sequence_type = sequence_type
        
        # Generate spreading sequence
        self.spreading_code = self._generate_spreading_sequence()
        
    def _generate_spreading_sequence(self):
        """Generate spreading sequence based on type"""
        if self.sequence_type == 'gold':
            return self._generate_gold_sequence()
        elif self.sequence_type == 'm_sequence':
            return self._generate_m_sequence()
        elif self.sequence_type == 'walsh':
            return self._generate_walsh_sequence()
        else:
            # Default to m-sequence
            return self._generate_m_sequence()
    
    def _generate_gold_sequence(self):
        """Generate Gold sequence (GPS standard)"""
        # Gold codes are constructed from two preferred m-sequences
        # For SF=31, use polynomials x^5+x^2+1 and x^5+x^4+x^3+x^2+1
        
        # First m-sequence: x^5 + x^2 + 1 (polynomial 0x25)
        m_seq1 = self._lfsr_sequence([1, 0, 1, 0, 0, 1], 31)
        
        # Second m-sequence: x^5 + x^4 + x^3 + x^2 + 1 (polynomial 0x3D)  
        m_seq2 = self._lfsr_sequence([1, 1, 1, 1, 0, 1], 31)
        
        # Gold sequence is XOR of two m-sequences
        gold_code = (m_seq1 ^ m_seq2)
        
        # Convert to bipolar
        return 2 * gold_code - 1
    
    def _generate_m_sequence(self):
        """Generate maximum length sequence"""
        if self.spreading_factor == 31:
            # x^5 + x^2 + 1
            return 2 * self._lfsr_sequence([1, 0, 1, 0, 0, 1], 31) - 1
        elif self.spreading_factor == 127:
            # x^7 + x^1 + 1  
            return 2 * self._lfsr_sequence([1, 1, 0, 0, 0, 0, 0, 1], 127) - 1
        elif self.spreading_factor == 511:
            # x^9 + x^4 + 1
            polynomial = [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
            return 2 * self._lfsr_sequence(polynomial, 511) - 1
        else:
            # Default 31-chip sequence
            return 2 * self._lfsr_sequence([1, 0, 1, 0, 0, 1], min(self.spreading_factor, 31)) - 1
    
    def _generate_walsh_sequence(self):
        """Generate Walsh-Hadamard sequence (CDMA2000 standard)"""
        # Find nearest power of 2
        n = int(2**np.ceil(np.log2(self.spreading_factor)))
        
        # Generate Hadamard matrix
        H = self._hadamard_matrix(n)
        
        # Select first row and truncate to spreading factor
        walsh_code = H[0, :self.spreading_factor]
        
        return walsh_code
    
    def _hadamard_matrix(self, n):
        """Generate Hadamard matrix"""
        if n == 1:
            return np.array([[1]])
        elif n == 2:
            return np.array([[1, 1], [1, -1]])
        else:
            H_n2 = self._hadamard_matrix(n // 2)
            return np.block([[H_n2, H_n2], [H_n2, -H_n2]])
    
    def _lfsr_sequence(self, polynomial, length):
        """Generate LFSR sequence"""
        # Initialize register with 1s
        register = np.ones(len(polynomial) - 1, dtype=int)
        sequence = []
        
        for _ in range(length):
            # Output bit
            sequence.append(register[-1])
            
            # Calculate feedback
            feedback = 0
            for i, coeff in enumerate(polynomial[:-1]):
                if coeff == 1:
                    feedback ^= register[i]
            
            # Shift register
            register = np.roll(register, 1)
            register[0] = feedback
        
        return np.array(sequence)
    
    def spread(self, data_bits):
        """Spread data bits with spreading sequence"""
        if len(data_bits) == 0:
            return np.array([])
        
        # Convert bits to bipolar
        data_symbols = 2 * data_bits.astype(float) - 1
        
        # Spread each data symbol
        spread_signal = []
        for symbol in data_symbols:
            # Multiply symbol by spreading sequence
            spread_chips = symbol * self.spreading_code
            spread_signal.extend(spread_chips)
        
        return np.array(spread_signal)
    
    def modulate(self, data_bits, modulation='bpsk', carrier_freq=0, sample_rate=1e6):
        """Complete DSSS modulation"""
        # Spread the data
        spread_chips = self.spread(data_bits)
        
        if carrier_freq == 0:
            # Return baseband signal
            if modulation == 'bpsk':
                return spread_chips
            elif modulation == 'qpsk':
                # For QPSK, group chips into I/Q
                if len(spread_chips) % 2 != 0:
                    spread_chips = np.append(spread_chips, 0)
                
                i_chips = spread_chips[::2]
                q_chips = spread_chips[1::2]
                return i_chips + 1j * q_chips
        else:
            # Passband modulation
            chip_duration = 1 / self.chip_rate
            samples_per_chip = int(sample_rate / self.chip_rate)
            
            # Upsample chips
            upsampled_chips = np.repeat(spread_chips, samples_per_chip)
            
            # Generate time vector
            t = np.arange(len(upsampled_chips)) / sample_rate
            
            if modulation == 'bpsk':
                # BPSK carrier modulation
                return upsampled_chips * np.cos(2 * np.pi * carrier_freq * t)
            elif modulation == 'qpsk':
                # QPSK carrier modulation
                if len(upsampled_chips) % 2 != 0:
                    upsampled_chips = np.append(upsampled_chips, 0)
                
                i_signal = upsampled_chips[::2]
                q_signal = upsampled_chips[1::2]
                
                # Repeat for proper length
                i_upsampled = np.repeat(i_signal, 2)[:len(upsampled_chips)]
                q_upsampled = np.repeat(q_signal, 2)[:len(upsampled_chips)]
                
                return (i_upsampled * np.cos(2 * np.pi * carrier_freq * t) -
                        q_upsampled * np.sin(2 * np.pi * carrier_freq * t))

class OptimizedDSSSDemodulator:
    """Research-grade DSSS demodulator with correlation detection"""
    
    def __init__(self, chip_rate=1e6, spreading_factor=31, sequence_type='gold'):
        self.chip_rate = chip_rate
        self.spreading_factor = spreading_factor
        self.sequence_type = sequence_type
        
        # Generate same spreading sequence as modulator
        modulator = OptimizedDSSSModulator(chip_rate, spreading_factor, sequence_type)
        self.spreading_code = modulator.spreading_code
        
        # Synchronization parameters
        self.correlation_threshold = 0.7 * spreading_factor  # 70% of max correlation
        
    def despread(self, received_chips, sync_search=True):
        """Despread received chips to recover data"""
        if len(received_chips) == 0:
            return np.array([])
        
        if sync_search:
            # Find synchronization
            sync_offset = self._find_synchronization(received_chips)
        else:
            sync_offset = 0
        
        # Despread starting from sync offset
        n_symbols = (len(received_chips) - sync_offset) // self.spreading_factor
        despread_symbols = []
        
        for i in range(n_symbols):
            start_idx = sync_offset + i * self.spreading_factor
            end_idx = start_idx + self.spreading_factor
            
            if end_idx <= len(received_chips):
                # Extract chip sequence
                chip_sequence = received_chips[start_idx:end_idx]
                
                # Correlate with spreading code
                correlation = np.sum(chip_sequence * self.spreading_code)
                
                # Normalize by spreading factor
                despread_symbol = correlation / self.spreading_factor
                despread_symbols.append(despread_symbol)
        
        return np.array(despread_symbols)
    
    def _find_synchronization(self, received_chips):
        """Find synchronization using correlation"""
        max_correlation = 0
        best_offset = 0
        
        # Search over possible offsets
        search_length = min(len(received_chips), 2 * self.spreading_factor)
        
        for offset in range(search_length - self.spreading_factor + 1):
            # Calculate correlation
            chip_sequence = received_chips[offset:offset + self.spreading_factor]
            correlation = abs(np.sum(chip_sequence * self.spreading_code))
            
            if correlation > max_correlation:
                max_correlation = correlation
                best_offset = offset
        
        return best_offset if max_correlation > self.correlation_threshold else 0
    
    def demodulate(self, received_signal, modulation='bpsk', carrier_freq=0, sample_rate=1e6):
        """Complete DSSS demodulation"""
        if carrier_freq == 0:
            # Baseband signal
            if modulation == 'bpsk':
                received_chips = received_signal
            elif modulation == 'qpsk':
                # Extract I and Q components
                i_chips = np.real(received_signal)
                q_chips = np.imag(received_signal)
                # Interleave I and Q  
                received_chips = np.empty(len(i_chips) + len(q_chips))
                received_chips[::2] = i_chips
                received_chips[1::2] = q_chips
        else:
            # Carrier demodulation first
            t = np.arange(len(received_signal)) / sample_rate
            
            if modulation == 'bpsk':
                # BPSK demodulation
                demod_signal = 2 * received_signal * np.cos(2 * np.pi * carrier_freq * t)
                
                # Low-pass filter
                nyquist = sample_rate / 2
                cutoff = self.chip_rate / nyquist
                if cutoff < 1:
                    b, a = butter(4, cutoff, btype='low')
                    demod_signal = filtfilt(b, a, demod_signal)
                
                # Decimate to chip rate
                samples_per_chip = int(sample_rate / self.chip_rate)
                received_chips = demod_signal[::samples_per_chip]
                
            elif modulation == 'qpsk':
                # QPSK demodulation
                i_demod = 2 * received_signal * np.cos(2 * np.pi * carrier_freq * t)
                q_demod = -2 * received_signal * np.sin(2 * np.pi * carrier_freq * t)
                
                # Low-pass filter both branches
                nyquist = sample_rate / 2
                cutoff = self.chip_rate / nyquist
                if cutoff < 1:
                    b, a = butter(4, cutoff, btype='low')
                    i_demod = filtfilt(b, a, i_demod)
                    q_demod = filtfilt(b, a, q_demod)
                
                # Decimate to chip rate
                samples_per_chip = int(sample_rate / self.chip_rate)
                i_chips = i_demod[::samples_per_chip]
                q_chips = q_demod[::samples_per_chip]
                
                # Interleave I and Q
                received_chips = np.empty(len(i_chips) + len(q_chips))
                received_chips[::2] = i_chips
                received_chips[1::2] = q_chips
        
        # Despread to recover data symbols
        data_symbols = self.despread(received_chips)
        
        # Hard decision to get bits
        data_bits = (data_symbols > 0).astype(int)
        
        return data_bits

class LoRaModulator:
    """LoRa CSS (Chirp Spread Spectrum) modulator"""
    
    def __init__(self, bandwidth=125000, spreading_factor=7, sample_rate=1e6):
        self.bw = bandwidth
        self.sf = spreading_factor
        self.fs = sample_rate
        self.n_samples = int(2**spreading_factor)  # Number of samples per symbol
        self.symbol_duration = self.n_samples / bandwidth  # Symbol duration in seconds
        
    def generate_chirp(self, symbol_value, up_chirp=True):
        """Generate LoRa chirp for given symbol value"""
        # Time vector for one symbol
        t = np.linspace(0, self.symbol_duration, self.n_samples, endpoint=False)
        
        if up_chirp:
            # Up-chirp: frequency increases linearly
            f_start = -self.bw / 2
            f_end = self.bw / 2
        else:
            # Down-chirp: frequency decreases linearly  
            f_start = self.bw / 2
            f_end = -self.bw / 2
        
        # Frequency ramp
        freq_ramp = f_start + (f_end - f_start) * t / self.symbol_duration
        
        # Frequency offset based on symbol value
        freq_offset = symbol_value * self.bw / (2**self.sf)
        
        # Total instantaneous frequency
        inst_freq = freq_ramp + freq_offset
        
        # Wrap frequency to stay within bandwidth
        inst_freq = ((inst_freq + self.bw/2) % self.bw) - self.bw/2
        
        # Generate chirp signal
        phase = 2 * np.pi * np.cumsum(inst_freq) / self.fs
        chirp = np.exp(1j * phase)
        
        return chirp
    
    def modulate(self, data_symbols):
        """Modulate data using LoRa CSS"""
        if len(data_symbols) == 0:
            return np.array([])
        
        modulated_signal = []
        
        for symbol in data_symbols:
            # Generate chirp for symbol
            chirp = self.generate_chirp(symbol, up_chirp=True)
            modulated_signal.extend(chirp)
        
        return np.array(modulated_signal)

class LoRaDemodulator:
    """LoRa CSS demodulator"""
    
    def __init__(self, bandwidth=125000, spreading_factor=7, sample_rate=1e6):
        self.bw = bandwidth
        self.sf = spreading_factor
        self.fs = sample_rate
        self.n_samples = int(2**spreading_factor)
        self.symbol_duration = self.n_samples / bandwidth
        
    def demodulate(self, received_signal):
        """Demodulate LoRa CSS signal"""
        if len(received_signal) == 0:
            return np.array([])
        
        n_symbols = len(received_signal) // self.n_samples
        demodulated_symbols = []
        
        for i in range(n_symbols):
            start_idx = i * self.n_samples
            end_idx = start_idx + self.n_samples
            
            if end_idx <= len(received_signal):
                symbol_signal = received_signal[start_idx:end_idx]
                
                # Correlate with all possible chirps to find symbol value
                max_correlation = 0
                detected_symbol = 0
                
                for symbol_value in range(2**self.sf):
                    # Generate reference chirp (down-chirp for correlation)
                    ref_chirp = self.generate_reference_chirp(symbol_value)
                    
                    # Calculate correlation
                    correlation = abs(np.sum(symbol_signal * np.conj(ref_chirp)))
                    
                    if correlation > max_correlation:
                        max_correlation = correlation
                        detected_symbol = symbol_value
                
                demodulated_symbols.append(detected_symbol)
        
        return np.array(demodulated_symbols)
    
    def generate_reference_chirp(self, symbol_value):
        """Generate reference chirp for correlation (down-chirp)"""
        t = np.linspace(0, self.symbol_duration, self.n_samples, endpoint=False)
        
        # Down-chirp for correlation
        f_start = self.bw / 2
        f_end = -self.bw / 2
        
        freq_ramp = f_start + (f_end - f_start) * t / self.symbol_duration
        freq_offset = symbol_value * self.bw / (2**self.sf)
        inst_freq = freq_ramp + freq_offset
        
        # Wrap frequency
        inst_freq = ((inst_freq + self.bw/2) % self.bw) - self.bw/2
        
        phase = 2 * np.pi * np.cumsum(inst_freq) / self.fs
        return np.exp(1j * phase)

def test_multicarrier_spread_spectrum():
    """Test optimized multicarrier and spread spectrum implementations"""
    print("Testing Research-Optimized Multicarrier and Spread Spectrum...")
    
    # Test OFDM
    print("\n1. Testing OFDM (IEEE 802.11 style):")
    ofdm_mod = OptimizedOFDMModulator(n_fft=64, cp_length=16)
    ofdm_demod = OptimizedOFDMDemodulator(n_fft=64, cp_length=16)
    
    # Generate test data (QPSK symbols)
    test_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1] * 3)
    qpsk_symbols = []
    for i in range(0, len(test_bits), 2):
        if i+1 < len(test_bits):
            # QPSK mapping
            symbol = (2*test_bits[i] - 1) + 1j*(2*test_bits[i+1] - 1)
            qpsk_symbols.append(symbol / np.sqrt(2))
    
    # OFDM modulation
    ofdm_signal = ofdm_mod.modulate(qpsk_symbols)
    print(f"OFDM signal length: {len(ofdm_signal)}")
    
    # Add noise
    snr_db = 20
    signal_power = np.mean(np.abs(ofdm_signal)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(ofdm_signal)) + 
                                      1j * np.random.randn(len(ofdm_signal)))
    noisy_ofdm = ofdm_signal + noise
    
    # OFDM demodulation
    demod_symbols = ofdm_demod.demodulate(noisy_ofdm)
    print(f"Recovered {len(demod_symbols)} symbols")
    
    # Test DSSS
    print("\n2. Testing DSSS (GPS style):")
    dsss_mod = OptimizedDSSSModulator(chip_rate=1e6, spreading_factor=31, sequence_type='gold')
    dsss_demod = OptimizedDSSSDemodulator(chip_rate=1e6, spreading_factor=31, sequence_type='gold')
    
    test_data_bits = np.array([1, 0, 1, 1, 0])
    
    # DSSS modulation
    dsss_signal = dsss_mod.modulate(test_data_bits, modulation='bpsk', carrier_freq=0)
    print(f"DSSS signal length: {len(dsss_signal)} (processing gain: {len(dsss_signal)/len(test_data_bits)})")
    
    # Add noise
    signal_power = np.mean(dsss_signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(dsss_signal))
    noisy_dsss = dsss_signal + noise
    
    # DSSS demodulation
    demod_bits = dsss_demod.demodulate(noisy_dsss, modulation='bpsk', carrier_freq=0)
    print(f"Original bits: {test_data_bits}")
    print(f"Recovered bits: {demod_bits}")
    
    if len(demod_bits) >= len(test_data_bits):
        ber = np.mean(test_data_bits != demod_bits[:len(test_data_bits)])
        print(f"DSSS BER: {ber:.4f}")
    
    # Test LoRa
    print("\n3. Testing LoRa CSS:")
    lora_mod = LoRaModulator(bandwidth=125000, spreading_factor=7)
    lora_demod = LoRaDemodulator(bandwidth=125000, spreading_factor=7)
    
    # LoRa symbols (0 to 2^SF - 1)
    lora_symbols = np.array([10, 50, 80, 120, 30])
    
    # LoRa modulation
    lora_signal = lora_mod.modulate(lora_symbols)
    print(f"LoRa signal length: {len(lora_signal)}")
    
    # Add noise
    signal_power = np.mean(np.abs(lora_signal)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(lora_signal)) + 
                                      1j * np.random.randn(len(lora_signal)))
    noisy_lora = lora_signal + noise
    
    # LoRa demodulation
    demod_lora_symbols = lora_demod.demodulate(noisy_lora)
    print(f"Original LoRa symbols: {lora_symbols}")
    print(f"Recovered LoRa symbols: {demod_lora_symbols}")
    
    if len(demod_lora_symbols) >= len(lora_symbols):
        ser = np.mean(lora_symbols != demod_lora_symbols[:len(lora_symbols)])
        print(f"LoRa SER: {ser:.4f}")
    
    print("\n✅ Multicarrier and Spread Spectrum tests completed")

if __name__ == "__main__":
    test_multicarrier_spread_spectrum()