# Create a comprehensive burst detector and demodulator module for the SDR project
# Based on gr-iridium architecture and algorithms

burst_detector_code = '''
"""
Burst Detector and Demodulator Module

Tương thích với gr-iridium - Phát hiện và giải điều chế burst tín hiệu

Dựa trên nghiên cứu gr-iridium: https://github.com/muccc/gr-iridium
"""

import numpy as np
import threading
import time
from queue import Queue, Empty
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
warnings.filterwarnings('ignore')

class FFTBurstTagger:
    """FFT-based burst detection engine tương thự gr-iridium"""
    
    def __init__(self, sample_rate=1e6, fft_size=1024, threshold_db=18, 
                 history_size=100, lookahead=5, burst_pre_len=10, burst_post_len=10):
        self.fs = sample_rate
        self.fft_size = fft_size
        self.threshold_db = threshold_db
        self.history_size = history_size
        self.lookahead = lookahead
        self.burst_pre_len = burst_pre_len
        self.burst_post_len = burst_post_len
        
        # Internal state
        self.noise_floor = np.full(fft_size, -100.0)  # dB
        self.noise_history = []
        self.active_bursts = {}  # burst_id -> burst_info
        self.burst_counter = 0
        self.fft_counter = 0
        
        # Burst tracking
        self.potential_bursts = {}  # bin -> frames_above_threshold
        self.burst_bins = set()  # Currently active burst bins
        
        # Detection parameters
        self.hysteresis_factor = 0.5  # 50% hysteresis for end detection
        self.min_burst_width = 3  # Minimum burst width in FFT bins
        
        print("✅ FFT Burst Tagger initialized")
        print(f"   FFT Size: {fft_size}")
        print(f"   Threshold: {threshold_db} dB")
        print(f"   Sample Rate: {sample_rate/1e6:.1f} MS/s")
    
    def process_samples(self, samples):
        """Process samples and detect bursts"""
        tagged_samples = []
        burst_tags = []
        
        # Process in FFT-sized chunks
        for i in range(0, len(samples), self.fft_size):
            chunk = samples[i:i+self.fft_size]
            if len(chunk) == self.fft_size:
                tags = self._process_fft_chunk(chunk, i)
                tagged_samples.extend(chunk)
                burst_tags.extend(tags)
        
        return np.array(tagged_samples), burst_tags
    
    def _process_fft_chunk(self, samples, sample_offset):
        """Process one FFT chunk for burst detection"""
        tags = []
        
        # Compute FFT
        fft_data = fft(samples * np.hanning(len(samples)))
        power_spectrum = 20 * np.log10(np.abs(fft_data) + 1e-12)
        freqs = fftfreq(len(samples), 1/self.fs)
        
        # Update noise floor
        self._update_noise_floor(power_spectrum)
        
        # Detect energy above threshold
        threshold_spectrum = self.noise_floor + self.threshold_db
        above_threshold = power_spectrum > threshold_spectrum
        
        # Process each bin for burst detection
        for bin_idx in range(len(above_threshold)):
            freq = freqs[bin_idx]
            
            if above_threshold[bin_idx]:
                # Bin is above threshold
                if bin_idx not in self.potential_bursts:
                    self.potential_bursts[bin_idx] = 1
                else:
                    self.potential_bursts[bin_idx] += 1
                
                # Check if burst should be declared
                if (bin_idx not in self.burst_bins and 
                    self.potential_bursts[bin_idx] >= self.lookahead):
                    
                    # Start new burst
                    burst_id = self._generate_burst_id()
                    burst_info = {
                        'burst_id': burst_id,
                        'center_freq': freq,
                        'start_sample': sample_offset - self.burst_pre_len * self.fft_size,
                        'magnitude': power_spectrum[bin_idx],
                        'noise_floor': self.noise_floor[bin_idx],
                        'bandwidth_est': self._estimate_bandwidth(power_spectrum, bin_idx),
                        'sample_rate': self.fs
                    }
                    
                    self.active_bursts[burst_id] = burst_info
                    self.burst_bins.add(bin_idx)
                    
                    # Create start tag
                    tag = {
                        'type': 'new_burst',
                        'offset': sample_offset,
                        'burst_info': burst_info
                    }
                    tags.append(tag)
            
            else:
                # Bin is below threshold
                if bin_idx in self.potential_bursts:
                    self.potential_bursts[bin_idx] -= 1
                    if self.potential_bursts[bin_idx] <= 0:
                        del self.potential_bursts[bin_idx]
                
                # Check if active burst should end
                if bin_idx in self.burst_bins:
                    # Use hysteresis threshold
                    hysteresis_threshold = (self.noise_floor[bin_idx] + 
                                          (self.threshold_db * self.hysteresis_factor))
                    
                    if power_spectrum[bin_idx] < hysteresis_threshold:
                        # End burst
                        self.burst_bins.remove(bin_idx)
                        
                        # Find corresponding burst
                        burst_to_end = None
                        for bid, binfo in self.active_bursts.items():
                            if abs(binfo['center_freq'] - freq) < self.fs / self.fft_size:
                                burst_to_end = bid
                                break
                        
                        if burst_to_end:
                            burst_info = self.active_bursts[burst_to_end]
                            burst_info['end_sample'] = sample_offset + self.burst_post_len * self.fft_size
                            
                            # Create end tag
                            tag = {
                                'type': 'end_burst', 
                                'offset': sample_offset,
                                'burst_info': burst_info
                            }
                            tags.append(tag)
                            
                            del self.active_bursts[burst_to_end]
        
        self.fft_counter += 1
        return tags
    
    def _update_noise_floor(self, power_spectrum):
        """Update noise floor estimate"""
        # Only update bins that are not currently in bursts
        for bin_idx in range(len(power_spectrum)):
            if bin_idx not in self.burst_bins:
                # Exponential moving average
                alpha = 0.01  # Slow adaptation
                if self.fft_counter == 0:
                    self.noise_floor[bin_idx] = power_spectrum[bin_idx]
                else:
                    self.noise_floor[bin_idx] = (
                        (1 - alpha) * self.noise_floor[bin_idx] + 
                        alpha * power_spectrum[bin_idx]
                    )
    
    def _estimate_bandwidth(self, power_spectrum, center_bin):
        """Estimate burst bandwidth"""
        # Find -3dB points around center
        center_power = power_spectrum[center_bin]
        threshold_3db = center_power - 3.0
        
        # Search left
        left_bin = center_bin
        for i in range(center_bin - 1, max(0, center_bin - 50), -1):
            if power_spectrum[i] < threshold_3db:
                break
            left_bin = i
        
        # Search right  
        right_bin = center_bin
        for i in range(center_bin + 1, min(len(power_spectrum), center_bin + 50)):
            if power_spectrum[i] < threshold_3db:
                break
            right_bin = i
        
        # Convert bins to bandwidth
        bin_width = self.fs / self.fft_size
        bandwidth = (right_bin - left_bin + 1) * bin_width
        
        return max(bandwidth, 25000)  # Minimum 25kHz
    
    def _generate_burst_id(self):
        """Generate unique burst ID"""
        self.burst_counter += 1
        return f"{int(time.time())}{self.burst_counter:06d}"

class TaggedBurstToPDU:
    """Convert tagged bursts to PDUs (Protocol Data Units)"""
    
    def __init__(self, max_burst_length=100000):
        self.max_burst_length = max_burst_length
        self.active_bursts = {}
        self.completed_pdus = Queue()
        
        print("✅ Tagged Burst to PDU converter initialized")
    
    def process_tagged_samples(self, samples, tags):
        """Process tagged samples and extract PDUs"""
        pdus = []
        
        for tag in tags:
            if tag['type'] == 'new_burst':
                # Start collecting samples for this burst
                burst_id = tag['burst_info']['burst_id']
                self.active_bursts[burst_id] = {
                    'info': tag['burst_info'],
                    'samples': [],
                    'start_offset': tag['offset']
                }
            
            elif tag['type'] == 'end_burst':
                # Complete the PDU for this burst
                burst_info = tag['burst_info']
                burst_id = burst_info['burst_id']
                
                if burst_id in self.active_bursts:
                    burst_data = self.active_bursts[burst_id]
                    
                    # Extract samples for this burst
                    start_offset = burst_data['start_offset']
                    end_offset = tag['offset']
                    
                    if end_offset > start_offset:
                        burst_samples = samples[start_offset:end_offset]
                        
                        pdu = {
                            'samples': burst_samples,
                            'metadata': burst_info,
                            'burst_id': burst_id
                        }
                        pdus.append(pdu)
                    
                    del self.active_bursts[burst_id]
        
        return pdus

class BurstDownmix:
    """Frequency correction and filtering for burst PDUs"""
    
    def __init__(self, sample_rate=1e6, target_rate=25000, filter_taps=101):
        self.fs = sample_rate
        self.target_rate = target_rate
        self.filter_taps = filter_taps
        self.decimation = int(sample_rate / target_rate)
        
        # Design anti-aliasing filter
        nyquist = sample_rate / 2
        cutoff = target_rate / 2 / nyquist
        self.aa_filter = signal.firwin(filter_taps, cutoff, window='hamming')
        
        print(f"✅ Burst Downmix initialized")
        print(f"   Decimation: {self.decimation}")
        print(f"   Output Rate: {target_rate} S/s")
    
    def process_pdu(self, pdu):
        """Process PDU: frequency correction, filtering, decimation"""
        samples = pdu['samples']
        metadata = pdu['metadata']
        
        # Frequency correction
        center_freq = metadata['center_freq']
        corrected_samples = self._frequency_correct(samples, center_freq)
        
        # Anti-aliasing filter
        filtered_samples = signal.lfilter(self.aa_filter, 1, corrected_samples)
        
        # Decimation
        decimated_samples = filtered_samples[::self.decimation]
        
        # Root raised cosine filter for pulse shaping
        rrc_samples = self._apply_rrc_filter(decimated_samples)
        
        # Update metadata
        updated_metadata = metadata.copy()
        updated_metadata['sample_rate'] = self.target_rate
        updated_metadata['corrected_freq'] = 0  # Now at baseband
        
        return {
            'samples': rrc_samples,
            'metadata': updated_metadata,
            'burst_id': pdu['burst_id']
        }
    
    def _frequency_correct(self, samples, freq_offset):
        """Correct frequency offset to move signal to baseband"""
        if abs(freq_offset) < 1:  # Already at baseband
            return samples
        
        t = np.arange(len(samples)) / self.fs
        correction = np.exp(-1j * 2 * np.pi * freq_offset * t)
        return samples * correction
    
    def _apply_rrc_filter(self, samples):
        """Apply Root Raised Cosine filter"""
        # Simplified RRC filter
        # In practice, would use proper RRC implementation
        alpha = 0.35  # Roll-off factor
        symbol_rate = 25000
        samples_per_symbol = int(self.target_rate / symbol_rate)
        
        # Generate RRC filter
        filter_len = 8 * samples_per_symbol + 1
        t = np.arange(-filter_len//2, filter_len//2 + 1) / self.target_rate
        
        # Simplified RRC approximation
        rrc_filter = np.sinc(t * symbol_rate) * np.cos(np.pi * alpha * t * symbol_rate)
        rrc_filter /= np.sum(rrc_filter**2)**0.5
        
        # Apply filter
        filtered = signal.lfilter(rrc_filter, 1, samples)
        
        return filtered

class IridiumQPSKDemodulator:
    """QPSK demodulator specifically for Iridium signals"""
    
    def __init__(self, symbol_rate=25000, sample_rate=25000):
        self.symbol_rate = symbol_rate
        self.sample_rate = sample_rate
        self.samples_per_symbol = int(sample_rate / symbol_rate)
        
        # Iridium sync word (12 symbols, BPSK)
        self.sync_word = np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
        self.sync_threshold = 0.7
        
        print(f"✅ Iridium QPSK Demodulator initialized")
        print(f"   Symbol Rate: {symbol_rate} sym/s")
        print(f"   Samples/Symbol: {self.samples_per_symbol}")
    
    def demodulate_pdu(self, pdu):
        """Demodulate QPSK PDU to bits"""
        samples = pdu['samples']
        metadata = pdu['metadata']
        
        # Phase and timing recovery
        recovered_samples = self._phase_recovery(samples)
        
        # Symbol timing recovery
        symbol_samples = self._timing_recovery(recovered_samples)
        
        # Find sync word
        sync_offset = self._find_sync(symbol_samples)
        
        if sync_offset is None:
            return None  # No sync found
        
        # Extract data starting after sync
        data_symbols = symbol_samples[sync_offset + len(self.sync_word):]
        
        # QPSK demodulation
        bits = self._qpsk_demodulate(data_symbols)
        
        # Calculate confidence
        confidence = self._calculate_confidence(symbol_samples, sync_offset)
        
        return {
            'bits': bits,
            'confidence': confidence,
            'sync_offset': sync_offset,
            'signal_level': np.mean(np.abs(symbol_samples)),
            'metadata': metadata,
            'burst_id': pdu['burst_id']
        }
    
    def _phase_recovery(self, samples):
        """Carrier phase recovery using Costa's loop"""
        # Simplified phase recovery
        # In practice would use proper PLL implementation
        
        # Rough phase correction using 4th power method
        phase_samples = samples**4
        avg_phase = np.angle(np.mean(phase_samples)) / 4
        
        # Correct phase
        phase_correction = np.exp(-1j * avg_phase)
        corrected = samples * phase_correction
        
        return corrected
    
    def _timing_recovery(self, samples):
        """Symbol timing recovery"""
        # Gardner timing recovery (simplified)
        if len(samples) < self.samples_per_symbol * 2:
            return samples[::max(1, self.samples_per_symbol)]
        
        # Simple symbol sampling at estimated symbol times
        symbol_indices = np.arange(self.samples_per_symbol//2, 
                                 len(samples), 
                                 self.samples_per_symbol)
        
        return samples[symbol_indices]
    
    def _find_sync(self, symbol_samples):
        """Find Iridium sync word in symbol stream"""
        if len(symbol_samples) < len(self.sync_word):
            return None
        
        # Convert sync word to expected symbol constellation
        sync_symbols = 2 * self.sync_word - 1  # BPSK: 0->-1, 1->+1
        
        # Correlate to find sync
        correlations = []
        for i in range(len(symbol_samples) - len(sync_symbols) + 1):
            window = symbol_samples[i:i+len(sync_symbols)]
            # Use real part for BPSK correlation
            corr = np.abs(np.corrcoef(np.real(window), sync_symbols)[0,1])
            correlations.append(corr if not np.isnan(corr) else 0)
        
        correlations = np.array(correlations)
        
        # Find best correlation
        best_idx = np.argmax(correlations)
        best_corr = correlations[best_idx]
        
        if best_corr > self.sync_threshold:
            return best_idx
        
        return None
    
    def _qpsk_demodulate(self, symbols):
        """Demodulate QPSK symbols to bits"""
        bits = []
        
        for symbol in symbols:
            # QPSK decision regions
            i_bit = 1 if np.real(symbol) > 0 else 0
            q_bit = 1 if np.imag(symbol) > 0 else 0
            
            bits.extend([i_bit, q_bit])
        
        return np.array(bits)
    
    def _calculate_confidence(self, symbol_samples, sync_offset):
        """Calculate demodulation confidence"""
        if sync_offset is None or len(symbol_samples) < 10:
            return 0.0
        
        # Use constellation dispersion as confidence metric
        data_symbols = symbol_samples[sync_offset:]
        
        if len(data_symbols) == 0:
            return 0.0
        
        # Expected QPSK constellation points
        constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        
        # Calculate average distance to nearest constellation point
        total_error = 0
        for symbol in data_symbols:
            distances = np.abs(symbol - constellation)
            min_distance = np.min(distances)
            total_error += min_distance
        
        avg_error = total_error / len(data_symbols)
        
        # Convert to confidence (0-1)
        confidence = np.exp(-avg_error * 5)  # Exponential mapping
        return np.clip(confidence, 0, 1)

class IridiumFrameProcessor:
    """Process demodulated bits into Iridium frames"""
    
    def __init__(self):
        self.frame_types = {
            'IRA': 'Iridium Ring Alert',
            'IBC': 'Iridium Broadcast Control', 
            'IMS': 'Iridium Message Signal',
            'TLC': 'Time and Location'
        }
        
        print("✅ Iridium Frame Processor initialized")
    
    def process_bits(self, demod_result):
        """Process demodulated bits into frame"""
        if demod_result is None:
            return None
        
        bits = demod_result['bits']
        
        # Basic frame structure analysis
        frame_info = {
            'raw_bits': bits,
            'bit_count': len(bits),
            'confidence': demod_result['confidence'],
            'signal_level': demod_result['signal_level'],
            'burst_id': demod_result['burst_id'],
            'timestamp': time.time()
        }
        
        # Attempt frame type detection
        frame_type = self._detect_frame_type(bits)
        frame_info['frame_type'] = frame_type
        
        # Extract frame data based on type
        if frame_type and len(bits) > 20:
            frame_info['parsed_data'] = self._parse_frame(bits, frame_type)
        
        return frame_info
    
    def _detect_frame_type(self, bits):
        """Detect Iridium frame type from bit pattern"""
        if len(bits) < 20:
            return 'UNKNOWN'
        
        # Check bit patterns for different frame types
        # This is a simplified detection - real implementation would be more complex
        
        # Look for typical Iridium patterns
        bit_str = ''.join(map(str, bits[:20]))
        
        if '1100' in bit_str:
            return 'IRA'
        elif '1010' in bit_str:
            return 'IBC'  
        elif '0011' in bit_str:
            return 'TLC'
        else:
            return 'DATA'
    
    def _parse_frame(self, bits, frame_type):
        """Parse frame data based on type"""
        parsed = {
            'header_bits': bits[:16] if len(bits) >= 16 else bits,
            'data_bits': bits[16:] if len(bits) > 16 else []
        }
        
        if frame_type == 'IRA':
            parsed['description'] = 'Ring Alert Message'
        elif frame_type == 'TLC':
            parsed['description'] = 'Time and Location Data'
        else:
            parsed['description'] = f'{frame_type} Frame'
        
        return parsed

class CompleteBurstDetector:
    """Complete burst detection and demodulation system"""
    
    def __init__(self, sample_rate=1e6, **kwargs):
        self.fs = sample_rate
        
        # Initialize pipeline components
        self.fft_tagger = FFTBurstTagger(sample_rate=sample_rate, **kwargs)
        self.burst_to_pdu = TaggedBurstToPDU()
        self.downmix = BurstDownmix(sample_rate=sample_rate)
        self.demodulator = IridiumQPSKDemodulator()
        self.frame_processor = IridiumFrameProcessor()
        
        # Statistics
        self.stats = {
            'samples_processed': 0,
            'bursts_detected': 0,
            'frames_decoded': 0,
            'decode_success_rate': 0.0
        }
        
        print("🎯 Complete Burst Detector System Ready")
    
    def process_iq_stream(self, iq_samples):
        """Process IQ samples through complete pipeline"""
        results = []
        
        try:
            # Stage 1: FFT Burst Detection
            tagged_samples, burst_tags = self.fft_tagger.process_samples(iq_samples)
            
            if not burst_tags:
                return results
            
            # Stage 2: Convert to PDUs
            pdus = self.burst_to_pdu.process_tagged_samples(tagged_samples, burst_tags)
            
            self.stats['bursts_detected'] += len(pdus)
            
            # Stage 3-5: Process each PDU
            for pdu in pdus:
                try:
                    # Downmix and filter
                    processed_pdu = self.downmix.process_pdu(pdu)
                    
                    # QPSK demodulation  
                    demod_result = self.demodulator.demodulate_pdu(processed_pdu)
                    
                    if demod_result and demod_result['confidence'] > 0.3:
                        # Frame processing
                        frame_info = self.frame_processor.process_bits(demod_result)
                        
                        if frame_info:
                            results.append(frame_info)
                            self.stats['frames_decoded'] += 1
                
                except Exception as e:
                    print(f"PDU processing error: {e}")
                    continue
            
            # Update statistics
            self.stats['samples_processed'] += len(iq_samples)
            if self.stats['bursts_detected'] > 0:
                self.stats['decode_success_rate'] = (
                    self.stats['frames_decoded'] / self.stats['bursts_detected']
                )
        
        except Exception as e:
            print(f"Pipeline error: {e}")
        
        return results
    
    def get_statistics(self):
        """Get processing statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.stats = {
            'samples_processed': 0,
            'bursts_detected': 0, 
            'frames_decoded': 0,
            'decode_success_rate': 0.0
        }

# Demo and test functions
def test_burst_detector():
    """Test the burst detection system"""
    print("🧪 Testing Burst Detection System")
    print("=" * 50)
    
    # Create test system
    detector = CompleteBurstDetector(
        sample_rate=1e6,
        fft_size=1024,
        threshold_db=15
    )
    
    # Generate test signal with bursts
    print("Generating test signal...")
    
    # Background noise
    noise_samples = 4000
    noise = 0.1 * (np.random.randn(noise_samples) + 1j * np.random.randn(noise_samples))
    
    # Add some burst signals
    burst_signals = []
    
    # Burst 1: QPSK at 100kHz offset
    burst1_len = 500
    t1 = np.arange(burst1_len) / 1e6
    burst1_freq = 100e3
    burst1_data = np.random.randint(0, 2, 50)  # Random bits
    
    # Simple QPSK modulation
    qpsk_symbols = []
    for i in range(0, len(burst1_data), 2):
        if i + 1 < len(burst1_data):
            i_bit, q_bit = burst1_data[i], burst1_data[i+1]
            symbol = (2*i_bit - 1) + 1j*(2*q_bit - 1)
            qpsk_symbols.append(symbol)
    
    # Upconvert to RF
    qpsk_samples = np.repeat(qpsk_symbols, burst1_len // len(qpsk_symbols))
    burst1 = qpsk_samples * np.exp(1j * 2 * np.pi * burst1_freq * t1[:len(qpsk_samples)])
    
    # Combine signals
    test_signal = np.concatenate([
        noise[:1000],
        burst1 * 2,  # Strong signal
        noise[1000:2000],
        burst1 * 0.5,  # Weaker signal  
        noise[2000:]
    ])
    
    print(f"Test signal: {len(test_signal)} samples")
    
    # Process through detector
    results = detector.process_iq_stream(test_signal)
    
    print(f"\n📊 Results:")
    print(f"Frames detected: {len(results)}")
    
    stats = detector.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    if results:
        print(f"\n🎯 Sample Frame:")
        frame = results[0]
        for key, value in frame.items():
            if key == 'raw_bits':
                print(f"{key}: {len(value)} bits")
            else:
                print(f"{key}: {value}")
    
    print("\n✅ Burst detector test completed")

if __name__ == "__main__":
    test_burst_detector()
'''

# Write the burst detector module to file
with open('burst_detector_demodulator.py', 'w', encoding='utf-8') as f:
    f.write(burst_detector_code)

print("✅ Created burst_detector_demodulator.py")
print("📁 File size:", len(burst_detector_code), "characters")