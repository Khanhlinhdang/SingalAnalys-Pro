# Create core/signal_processor.py - Main Signal Processing Engine
signal_processor_content = '''"""
RF Signal Processing Engine

This module integrates the three main libraries:
- pyspectrum: Real-time FFT spectrum analysis
- mhostetter/sdr: Digital signal processing and filtering  
- scikit-dsp-comm: Advanced DSP algorithms and synchronization

Provides comprehensive signal processing pipeline for RF analysis.
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from collections import deque
import queue

# Core signal processing libraries
try:
    import sdr
except ImportError:
    sdr = None
    logging.warning("mhostetter/sdr library not available")

try:
    import sk_dsp_comm.sigsys as sigsys
    import sk_dsp_comm.digitalcomm as digitalcomm
    import sk_dsp_comm.fir_design_helper as fir_design
    import sk_dsp_comm.iir_design_helper as iir_design
    import sk_dsp_comm.synchronization as sync
    import sk_dsp_comm.fec_conv as fec
    import sk_dsp_comm.multirate as multirate
except ImportError:
    logging.warning("scikit-dsp-comm library not available")

from scipy import signal
from scipy.fft import fft, fftshift, fftfreq
import scipy.signal.windows as windows

from config.settings import AppSettings, DSPConfig
from utils.logger import get_performance_logger


@dataclass  
class SpectrumData:
    """Container for spectrum analysis results"""
    frequencies: np.ndarray
    magnitudes: np.ndarray  # dB
    phases: np.ndarray      # radians
    timestamp: float
    center_freq: float
    sample_rate: float
    fft_size: int
    window_type: str


@dataclass
class SignalAnalysis:
    """Container for signal analysis results"""
    power: float            # Average power (dB)
    peak_freq: float        # Peak frequency (Hz)
    peak_power: float       # Peak power (dB)
    bandwidth: float        # Occupied bandwidth (Hz)
    snr_estimate: float     # SNR estimate (dB)
    modulation_type: str    # Detected modulation
    symbol_rate: float      # Symbol rate if digital
    constellation: Optional[np.ndarray] = None


class FilterBank:
    """Collection of digital filters using sdr and scikit-dsp-comm"""
    
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
        # Filter objects
        self.filters = {}
        self.current_filter = None
        
        # Initialize filters
        self.initialize_filters()
    
    def initialize_filters(self):
        """Initialize various filter types"""
        try:
            # FIR Filters using scikit-dsp-comm
            if 'fir_design' in globals():
                # Low-pass filter
                self.filters['lowpass_fir'] = self.create_fir_lowpass(
                    cutoff=self.sample_rate/4, 
                    transition_width=self.sample_rate/20
                )
                
                # High-pass filter
                self.filters['highpass_fir'] = self.create_fir_highpass(
                    cutoff=self.sample_rate/8,
                    transition_width=self.sample_rate/40
                )
                
                # Band-pass filter
                self.filters['bandpass_fir'] = self.create_fir_bandpass(
                    f_low=self.sample_rate/8,
                    f_high=self.sample_rate/4,
                    transition_width=self.sample_rate/40
                )
            
            # IIR Filters using scikit-dsp-comm
            if 'iir_design' in globals():
                self.filters['butterworth_lp'] = self.create_iir_butterworth(
                    cutoff=self.sample_rate/4,
                    order=8,
                    filter_type='low'
                )
                
                self.filters['chebyshev_bp'] = self.create_iir_chebyshev(
                    f_low=self.sample_rate/8,
                    f_high=self.sample_rate/4,
                    order=6,
                    ripple=0.5
                )
            
            # SDR library filters
            if sdr is not None:
                # Moving average filter
                self.filters['moving_average'] = sdr.MovingAverage(21)
                
                # Polyphase filters for resampling
                self.filters['decimator_2'] = sdr.Decimator(2)
                self.filters['interpolator_2'] = sdr.Interpolator(2)
                
                # Advanced resampler
                self.filters['resampler'] = sdr.Resampler(3, 2)  # 3/2 resampling
            
            self.logger.info(f"Initialized {len(self.filters)} filters")
            
        except Exception as e:
            self.logger.error(f"Filter initialization error: {e}")
    
    def create_fir_lowpass(self, cutoff: float, transition_width: float, order: int = 51):
        """Create FIR lowpass filter using scikit-dsp-comm"""
        try:
            f_pass = cutoff
            f_stop = cutoff + transition_width
            return fir_design.fir_remez_lpf(f_pass, f_stop, 0.1, 80, self.sample_rate, order)
        except:
            # Fallback to scipy
            return signal.firwin(order, cutoff, fs=self.sample_rate)
    
    def create_fir_highpass(self, cutoff: float, transition_width: float, order: int = 51):
        """Create FIR highpass filter"""
        try:
            f_stop = cutoff - transition_width
            f_pass = cutoff
            return fir_design.fir_remez_hpf(f_stop, f_pass, 80, 0.1, self.sample_rate, order)
        except:
            # Fallback to scipy
            return signal.firwin(order, cutoff, fs=self.sample_rate, pass_zero=False)
    
    def create_fir_bandpass(self, f_low: float, f_high: float, transition_width: float, order: int = 51):
        """Create FIR bandpass filter"""
        try:
            f_stop1 = f_low - transition_width
            f_pass1 = f_low
            f_pass2 = f_high
            f_stop2 = f_high + transition_width
            return fir_design.fir_remez_bpf(f_stop1, f_pass1, f_pass2, f_stop2, 80, 0.1, 0.1, 80, self.sample_rate, order)
        except:
            # Fallback to scipy
            return signal.firwin(order, [f_low, f_high], fs=self.sample_rate, pass_zero=False)
    
    def create_iir_butterworth(self, cutoff: float, order: int, filter_type: str = 'low'):
        """Create IIR Butterworth filter"""
        try:
            if filter_type == 'low':
                return iir_design.iir_d(order, cutoff, 0, self.sample_rate, ftype='butter')
            elif filter_type == 'high':
                return iir_design.iir_d(order, cutoff, 0, self.sample_rate, ftype='butter', btype='high')
        except:
            # Fallback to scipy
            sos = signal.butter(order, cutoff, btype=filter_type, fs=self.sample_rate, output='sos')
            return sos
    
    def create_iir_chebyshev(self, f_low: float, f_high: float, order: int, ripple: float):
        """Create IIR Chebyshev bandpass filter"""
        try:
            return iir_design.iir_d(order, [f_low, f_high], ripple, self.sample_rate, 
                                  ftype='cheby1', btype='band')
        except:
            # Fallback to scipy
            sos = signal.cheby1(order, ripple, [f_low, f_high], btype='band', 
                              fs=self.sample_rate, output='sos')
            return sos
    
    def apply_filter(self, data: np.ndarray, filter_name: str) -> np.ndarray:
        """Apply specified filter to data"""
        if filter_name not in self.filters:
            self.logger.warning(f"Filter '{filter_name}' not found")
            return data
        
        try:
            filter_obj = self.filters[filter_name]
            
            # Handle different filter types
            if hasattr(filter_obj, '__call__'):
                # SDR library filter objects
                return filter_obj(data)
            elif isinstance(filter_obj, np.ndarray):
                # FIR coefficients
                return signal.lfilter(filter_obj, 1, data)
            else:
                # IIR SOS format
                return signal.sosfilt(filter_obj, data)
                
        except Exception as e:
            self.logger.error(f"Filter application error: {e}")
            return data


class ModulationAnalyzer:
    """Modulation detection and demodulation using sdr library"""
    
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
        # Demodulators
        self.demodulators = {}
        self.initialize_demodulators()
    
    def initialize_demodulators(self):
        """Initialize various demodulators"""
        if sdr is None:
            return
        
        try:
            # PSK demodulators
            self.demodulators['bpsk'] = sdr.PSK(2, pulse_shape="srrc", sps=4)
            self.demodulators['qpsk'] = sdr.PSK(4, pulse_shape="srrc", sps=4) 
            self.demodulators['8psk'] = sdr.PSK(8, pulse_shape="srrc", sps=4)
            
            # QAM demodulators
            self.demodulators['16qam'] = sdr.QAM(16, pulse_shape="srrc", sps=4)
            self.demodulators['64qam'] = sdr.QAM(64, pulse_shape="srrc", sps=4)
            
            # CPM demodulators
            self.demodulators['msk'] = sdr.MSK(sps=4)
            
            self.logger.info(f"Initialized {len(self.demodulators)} demodulators")
            
        except Exception as e:
            self.logger.error(f"Demodulator initialization error: {e}")
    
    def detect_modulation(self, signal_data: np.ndarray) -> str:
        """Detect modulation type of signal"""
        try:
            # Simple modulation detection based on signal characteristics
            # This is a simplified approach - real detection would be more complex
            
            # Calculate signal statistics
            signal_power = np.mean(np.abs(signal_data)**2)
            signal_std = np.std(np.abs(signal_data))
            
            # Phase characteristics
            phases = np.angle(signal_data)
            phase_diff = np.diff(np.unwrap(phases))
            phase_std = np.std(phase_diff)
            
            # Simple classification rules
            if phase_std < 0.1:
                return "FM or unmodulated carrier"
            elif phase_std < 0.5:
                return "PSK"
            elif signal_std/np.mean(np.abs(signal_data)) > 0.3:
                return "QAM"
            else:
                return "Unknown"
                
        except Exception as e:
            self.logger.error(f"Modulation detection error: {e}")
            return "Unknown"
    
    def demodulate_signal(self, signal_data: np.ndarray, mod_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Demodulate signal and return symbols and constellation"""
        if mod_type not in self.demodulators:
            return np.array([]), np.array([])
        
        try:
            demod = self.demodulators[mod_type]
            
            # Demodulate signal
            symbols = demod.demodulate(signal_data)
            
            # Generate constellation points
            constellation = demod.constellation
            
            return symbols, constellation
            
        except Exception as e:
            self.logger.error(f"Demodulation error: {e}")
            return np.array([]), np.array([])


class SignalProcessor:
    """Main signal processing engine"""
    
    def __init__(self, settings: AppSettings, backend_manager=None):
        self.settings = settings
        self.backend_manager = backend_manager
        self.logger = logging.getLogger(__name__)
        self.perf_logger = get_performance_logger()
        
        # Processing components
        self.filter_bank = None
        self.modulation_analyzer = None
        
        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.sample_buffer = deque(maxlen=10000)  # Circular buffer
        
        # Results storage
        self.latest_spectrum = None
        self.latest_analysis = None
        self.spectrum_history = deque(maxlen=1000)
        
        # Performance tracking
        self.samples_processed = 0
        self.processing_time = 0
        
        # Initialize components
        self.initialize_processors()
    
    def initialize_processors(self):
        """Initialize signal processing components"""
        try:
            sample_rate = self.settings.sdr.sample_rate
            
            # Initialize filter bank
            self.filter_bank = FilterBank(sample_rate)
            
            # Initialize modulation analyzer
            self.modulation_analyzer = ModulationAnalyzer(sample_rate)
            
            self.logger.info("Signal processors initialized")
            
        except Exception as e:
            self.logger.error(f"Processor initialization error: {e}")
    
    def start(self):
        """Start signal processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Signal processing started")
    
    def stop(self):
        """Stop signal processing"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        self.logger.info("Signal processing stopped")
    
    def _processing_loop(self):
        """Main processing loop (runs in separate thread)"""
        while self.is_running:
            try:
                # Get samples from backend
                if self.backend_manager and self.backend_manager.current_backend:
                    samples = self.backend_manager.read_samples(
                        self.settings.dsp.fft_size
                    )
                    
                    if samples is not None and len(samples) > 0:
                        # Process samples
                        self._process_samples(samples)
                        
                        # Add to buffer for GUI
                        self.sample_buffer.extend(samples)
                
                # Small delay to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                time.sleep(0.1)
    
    def _process_samples(self, samples: np.ndarray):
        """Process a chunk of samples"""
        self.perf_logger.start_timer("sample_processing")
        
        try:
            # Apply filtering if enabled
            if self.settings.dsp.enable_filtering:
                samples = self.apply_filtering(samples)
            
            # Compute spectrum
            spectrum_data = self.compute_spectrum(samples)
            self.latest_spectrum = spectrum_data
            self.spectrum_history.append(spectrum_data)
            
            # Perform signal analysis
            analysis = self.analyze_signal(samples, spectrum_data)
            self.latest_analysis = analysis
            
            # Update counters
            self.samples_processed += len(samples)
            
        except Exception as e:
            self.logger.error(f"Sample processing error: {e}")
        
        finally:
            processing_time = self.perf_logger.end_timer("sample_processing")
            if processing_time:
                self.processing_time += processing_time
    
    def apply_filtering(self, samples: np.ndarray) -> np.ndarray:
        """Apply digital filtering to samples"""
        if not self.filter_bank:
            return samples
        
        try:
            filter_type = self.settings.dsp.filter_type
            return self.filter_bank.apply_filter(samples, filter_type)
        except Exception as e:
            self.logger.error(f"Filtering error: {e}")
            return samples
    
    def compute_spectrum(self, samples: np.ndarray) -> SpectrumData:
        """Compute FFT spectrum from samples"""
        try:
            fft_size = self.settings.dsp.fft_size
            window_type = self.settings.dsp.window
            
            # Ensure we have enough samples
            if len(samples) < fft_size:
                # Zero-pad if necessary
                padded_samples = np.zeros(fft_size, dtype=samples.dtype)
                padded_samples[:len(samples)] = samples
                samples = padded_samples
            elif len(samples) > fft_size:
                # Truncate if necessary
                samples = samples[:fft_size]
            
            # Apply window
            if window_type == 'hanning':
                window = windows.hann(fft_size)
            elif window_type == 'hamming':
                window = windows.hamming(fft_size)
            elif window_type == 'blackman':
                window = windows.blackman(fft_size)
            else:
                window = np.ones(fft_size)  # Rectangular
            
            windowed_samples = samples * window
            
            # Compute FFT
            spectrum = fft(windowed_samples, n=fft_size)
            spectrum_shifted = fftshift(spectrum)
            
            # Compute frequencies
            frequencies = fftshift(fftfreq(fft_size, 1/self.settings.sdr.sample_rate))
            frequencies += self.settings.sdr.center_freq  # Shift to actual frequencies
            
            # Compute magnitudes in dB
            magnitudes = 20 * np.log10(np.abs(spectrum_shifted) + 1e-10)
            
            # Compute phases
            phases = np.angle(spectrum_shifted)
            
            return SpectrumData(
                frequencies=frequencies,
                magnitudes=magnitudes,
                phases=phases,
                timestamp=time.time(),
                center_freq=self.settings.sdr.center_freq,
                sample_rate=self.settings.sdr.sample_rate,
                fft_size=fft_size,
                window_type=window_type
            )
            
        except Exception as e:
            self.logger.error(f"Spectrum computation error: {e}")
            return None
    
    def analyze_signal(self, samples: np.ndarray, spectrum_data: SpectrumData) -> SignalAnalysis:
        """Perform comprehensive signal analysis"""
        try:
            # Basic power measurements
            power_avg = 10 * np.log10(np.mean(np.abs(samples)**2) + 1e-10)
            
            # Find peak in spectrum
            peak_idx = np.argmax(spectrum_data.magnitudes)
            peak_freq = spectrum_data.frequencies[peak_idx]
            peak_power = spectrum_data.magnitudes[peak_idx]
            
            # Estimate occupied bandwidth (power above -20dB from peak)
            threshold = peak_power - 20
            above_threshold = spectrum_data.magnitudes > threshold
            bandwidth_bins = np.sum(above_threshold)
            bandwidth = bandwidth_bins * (self.settings.sdr.sample_rate / self.settings.dsp.fft_size)
            
            # Simple SNR estimate
            noise_floor = np.median(spectrum_data.magnitudes)
            snr_estimate = peak_power - noise_floor
            
            # Modulation detection
            if self.modulation_analyzer:
                mod_type = self.modulation_analyzer.detect_modulation(samples)
                symbols, constellation = self.modulation_analyzer.demodulate_signal(samples, 'qpsk')
            else:
                mod_type = "Unknown"
                symbols = np.array([])
                constellation = None
            
            # Symbol rate estimation (simplified)
            symbol_rate = 0.0
            if len(symbols) > 0:
                # Very basic symbol rate estimation
                symbol_rate = self.settings.sdr.sample_rate / 4  # Assume 4 samples per symbol
            
            return SignalAnalysis(
                power=power_avg,
                peak_freq=peak_freq,
                peak_power=peak_power,
                bandwidth=bandwidth,
                snr_estimate=snr_estimate,
                modulation_type=mod_type,
                symbol_rate=symbol_rate,
                constellation=constellation
            )
            
        except Exception as e:
            self.logger.error(f"Signal analysis error: {e}")
            return SignalAnalysis(
                power=0, peak_freq=0, peak_power=0, bandwidth=0,
                snr_estimate=0, modulation_type="Error", symbol_rate=0
            )
    
    def process_chunk(self) -> Optional[Dict[str, Any]]:
        """Process one chunk of data and return results"""
        try:
            if not self.backend_manager or not self.backend_manager.current_backend:
                return None
            
            # Get samples
            samples = self.backend_manager.read_samples(self.settings.dsp.fft_size)
            if samples is None or len(samples) == 0:
                return None
            
            # Process samples
            self._process_samples(samples)
            
            # Return results
            result = {
                'samples': samples,
                'timestamp': time.time()
            }
            
            if self.latest_spectrum:
                result['spectrum'] = self.latest_spectrum
            
            if self.latest_analysis:
                result['analysis'] = self.latest_analysis
            
            return result
            
        except Exception as e:
            self.logger.error(f"Chunk processing error: {e}")
            return None
    
    def get_latest_spectrum(self) -> Optional[SpectrumData]:
        """Get latest spectrum data"""
        return self.latest_spectrum
    
    def get_latest_analysis(self) -> Optional[SignalAnalysis]:
        """Get latest signal analysis"""
        return self.latest_analysis
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        return {
            'samples_processed': self.samples_processed,
            'processing_time': self.processing_time,
            'avg_processing_time': self.processing_time / max(1, self.samples_processed / 1000),
            'is_running': self.is_running
        }
    
    def update_settings(self, new_settings: AppSettings):
        """Update processing settings"""
        self.settings = new_settings
        
        # Reinitialize processors if sample rate changed
        if (self.filter_bank and 
            abs(self.filter_bank.sample_rate - new_settings.sdr.sample_rate) > 1.0):
            self.initialize_processors()
    
    def run_headless(self, config: Dict[str, Any]):
        """Run processing in headless mode"""
        self.logger.info("Running in headless mode")
        
        try:
            # Configure from config dict
            if 'freq' in config:
                self.settings.sdr.center_freq = config['freq']
            if 'samplerate' in config:
                self.settings.sdr.sample_rate = config['samplerate']
            
            # Initialize backend
            if not self.backend_manager.initialize():
                raise RuntimeError("Failed to initialize backend")
            
            # Start processing
            self.start()
            
            # Process for specified time or until interrupted
            start_time = time.time()
            while self.is_running:
                time.sleep(1.0)
                
                # Print status every 10 seconds
                elapsed = time.time() - start_time
                if elapsed % 10 < 1.0:
                    stats = self.get_performance_stats()
                    print(f"Processed {stats['samples_processed']:,} samples in {elapsed:.1f}s")
                    
                    if self.latest_analysis:
                        analysis = self.latest_analysis
                        print(f"Power: {analysis.power:.1f} dBm, "
                              f"Peak: {analysis.peak_freq/1e6:.3f} MHz @ {analysis.peak_power:.1f} dB, "
                              f"BW: {analysis.bandwidth/1e3:.1f} kHz, "
                              f"SNR: {analysis.snr_estimate:.1f} dB")
        
        except KeyboardInterrupt:
            self.logger.info("Headless processing interrupted")
        except Exception as e:
            self.logger.error(f"Headless processing error: {e}")
        finally:
            self.stop()
'''

with open("rf_spectrum_analyzer/core/signal_processor.py", "w") as f:
    f.write(signal_processor_content)

print("Created core/signal_processor.py")