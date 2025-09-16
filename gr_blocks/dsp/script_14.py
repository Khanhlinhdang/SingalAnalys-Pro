# Create dsp/filters.py - Advanced DSP Filters Module
dsp_filters_content = '''"""
Advanced DSP Filters Module

Implements various digital filters using mhostetter/sdr and scikit-dsp-comm libraries.
Provides FIR, IIR, polyphase, and adaptive filtering capabilities.
"""

import numpy as np
import logging
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Core signal processing libraries
try:
    import sdr
    SDR_AVAILABLE = True
except ImportError:
    SDR_AVAILABLE = False
    sdr = None

try:
    import sk_dsp_comm.fir_design_helper as fir_design
    import sk_dsp_comm.iir_design_helper as iir_design  
    import sk_dsp_comm.multirate as multirate
    SCIKIT_DSP_AVAILABLE = True
except ImportError:
    SCIKIT_DSP_AVAILABLE = False
    fir_design = None
    iir_design = None
    multirate = None

from scipy import signal
from scipy.signal import windows


@dataclass
class FilterResponse:
    """Container for filter frequency response"""
    frequencies: np.ndarray
    magnitude: np.ndarray     # dB
    phase: np.ndarray        # radians
    group_delay: np.ndarray  # samples


class FIRFilterBank:
    """Collection of FIR filters using multiple libraries"""
    
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
        # Available filters
        self.filters = {}
        
        # Initialize filters
        self.initialize_filters()
    
    def initialize_filters(self):
        """Initialize various FIR filters"""
        try:
            # Standard lowpass filters
            self.filters['lowpass_1MHz'] = self.design_lowpass_fir(1e6, 100e3)
            self.filters['lowpass_5MHz'] = self.design_lowpass_fir(5e6, 500e3)
            self.filters['lowpass_10MHz'] = self.design_lowpass_fir(10e6, 1e6)
            
            # Standard highpass filters
            self.filters['highpass_1MHz'] = self.design_highpass_fir(1e6, 100e3)
            self.filters['highpass_5MHz'] = self.design_highpass_fir(5e6, 500e3)
            
            # Standard bandpass filters
            self.filters['bandpass_narrow'] = self.design_bandpass_fir(10e6, 12e6, 500e3)
            self.filters['bandpass_wide'] = self.design_bandpass_fir(5e6, 15e6, 1e6)
            
            # Standard bandstop filters
            self.filters['bandstop_narrow'] = self.design_bandstop_fir(9e6, 11e6, 200e3)
            
            # Special purpose filters
            self.filters['anti_aliasing'] = self.design_anti_aliasing_filter()
            self.filters['reconstruction'] = self.design_reconstruction_filter()
            self.filters['matched_pulse'] = self.design_matched_filter()
            
            # SDR library filters (if available)
            if SDR_AVAILABLE:
                self.initialize_sdr_filters()
            
            self.logger.info(f"Initialized {len(self.filters)} FIR filters")
            
        except Exception as e:
            self.logger.error(f"Filter initialization error: {e}")
    
    def initialize_sdr_filters(self):
        """Initialize SDR library specific filters"""
        if not SDR_AVAILABLE:
            return
        
        try:
            # Root raised cosine filters for digital communications
            self.filters['rrc_0p5'] = sdr.root_raised_cosine(0.5, 8, 21)  # β=0.5, 8 sps, 21 taps
            self.filters['rrc_0p25'] = sdr.root_raised_cosine(0.25, 8, 21) # β=0.25, 8 sps, 21 taps
            
            # Raised cosine filters
            self.filters['rc_0p5'] = sdr.raised_cosine(0.5, 8, 21)
            
            # Moving average filters
            self.filters['moving_avg_5'] = np.ones(5) / 5
            self.filters['moving_avg_21'] = np.ones(21) / 21
            self.filters['moving_avg_51'] = np.ones(51) / 51
            
            # Differentiator
            self.filters['differentiator'] = sdr.differentiator(21, self.sample_rate)
            
        except Exception as e:
            self.logger.warning(f"SDR filter initialization error: {e}")
    
    def design_lowpass_fir(
        self, 
        cutoff: float, 
        transition_width: float, 
        order: int = None,
        window: str = 'kaiser'
    ) -> np.ndarray:
        """Design FIR lowpass filter"""
        try:
            # Use scikit-dsp-comm if available for optimal design
            if SCIKIT_DSP_AVAILABLE and fir_design:
                f_pass = cutoff
                f_stop = cutoff + transition_width
                ripple_pass = 0.1  # dB
                ripple_stop = 80   # dB
                
                if order is None:
                    order = fir_design.fir_remez_lpf_length(f_pass, f_stop, ripple_pass, ripple_stop, self.sample_rate)
                
                # Use Remez exchange algorithm
                return fir_design.fir_remez_lpf(f_pass, f_stop, ripple_pass, ripple_stop, self.sample_rate, order)
            
            else:
                # Fallback to scipy
                if order is None:
                    order = self._estimate_fir_order(transition_width)
                
                if window == 'kaiser':
                    beta = signal.kaiser_beta(80)  # 80 dB stopband attenuation
                    return signal.firwin(order, cutoff, fs=self.sample_rate, window=('kaiser', beta))
                else:
                    return signal.firwin(order, cutoff, fs=self.sample_rate, window=window)
        
        except Exception as e:
            self.logger.error(f"Lowpass filter design error: {e}")
            # Return simple rectangular window filter as fallback
            order = order or 51
            return signal.firwin(order, cutoff, fs=self.sample_rate)
    
    def design_highpass_fir(
        self, 
        cutoff: float, 
        transition_width: float, 
        order: int = None,
        window: str = 'kaiser'
    ) -> np.ndarray:
        """Design FIR highpass filter"""
        try:
            if SCIKIT_DSP_AVAILABLE and fir_design:
                f_stop = cutoff - transition_width
                f_pass = cutoff
                ripple_pass = 0.1
                ripple_stop = 80
                
                if order is None:
                    order = fir_design.fir_remez_hpf_length(f_stop, f_pass, ripple_stop, ripple_pass, self.sample_rate)
                
                return fir_design.fir_remez_hpf(f_stop, f_pass, ripple_stop, ripple_pass, self.sample_rate, order)
            
            else:
                if order is None:
                    order = self._estimate_fir_order(transition_width)
                
                if window == 'kaiser':
                    beta = signal.kaiser_beta(80)
                    return signal.firwin(order, cutoff, fs=self.sample_rate, pass_zero=False, window=('kaiser', beta))
                else:
                    return signal.firwin(order, cutoff, fs=self.sample_rate, pass_zero=False, window=window)
        
        except Exception as e:
            self.logger.error(f"Highpass filter design error: {e}")
            order = order or 51
            return signal.firwin(order, cutoff, fs=self.sample_rate, pass_zero=False)
    
    def design_bandpass_fir(
        self, 
        f_low: float, 
        f_high: float, 
        transition_width: float,
        order: int = None,
        window: str = 'kaiser'
    ) -> np.ndarray:
        """Design FIR bandpass filter"""
        try:
            if SCIKIT_DSP_AVAILABLE and fir_design:
                f_stop1 = f_low - transition_width
                f_pass1 = f_low
                f_pass2 = f_high
                f_stop2 = f_high + transition_width
                ripple_pass = 0.1
                ripple_stop = 80
                
                if order is None:
                    order = fir_design.fir_remez_bpf_length(
                        f_stop1, f_pass1, f_pass2, f_stop2,
                        ripple_stop, ripple_pass, ripple_pass, ripple_stop,
                        self.sample_rate
                    )
                
                return fir_design.fir_remez_bpf(
                    f_stop1, f_pass1, f_pass2, f_stop2,
                    ripple_stop, ripple_pass, ripple_pass, ripple_stop,
                    self.sample_rate, order
                )
            
            else:
                if order is None:
                    order = self._estimate_fir_order(transition_width)
                
                if window == 'kaiser':
                    beta = signal.kaiser_beta(80)
                    return signal.firwin(order, [f_low, f_high], fs=self.sample_rate, pass_zero=False, window=('kaiser', beta))
                else:
                    return signal.firwin(order, [f_low, f_high], fs=self.sample_rate, pass_zero=False, window=window)
        
        except Exception as e:
            self.logger.error(f"Bandpass filter design error: {e}")
            order = order or 51
            return signal.firwin(order, [f_low, f_high], fs=self.sample_rate, pass_zero=False)
    
    def design_bandstop_fir(
        self, 
        f_low: float, 
        f_high: float, 
        transition_width: float,
        order: int = None,
        window: str = 'kaiser'
    ) -> np.ndarray:
        """Design FIR bandstop filter"""
        try:
            if SCIKIT_DSP_AVAILABLE and fir_design:
                f_pass1 = f_low - transition_width
                f_stop1 = f_low
                f_stop2 = f_high
                f_pass2 = f_high + transition_width
                ripple_pass = 0.1
                ripple_stop = 80
                
                if order is None:
                    order = fir_design.fir_remez_bsf_length(
                        f_pass1, f_stop1, f_stop2, f_pass2,
                        ripple_pass, ripple_stop, ripple_stop, ripple_pass,
                        self.sample_rate
                    )
                
                return fir_design.fir_remez_bsf(
                    f_pass1, f_stop1, f_stop2, f_pass2,
                    ripple_pass, ripple_stop, ripple_stop, ripple_pass,
                    self.sample_rate, order
                )
            
            else:
                if order is None:
                    order = self._estimate_fir_order(transition_width)
                
                if window == 'kaiser':
                    beta = signal.kaiser_beta(80)
                    return signal.firwin(order, [f_low, f_high], fs=self.sample_rate, window=('kaiser', beta))
                else:
                    return signal.firwin(order, [f_low, f_high], fs=self.sample_rate, window=window)
        
        except Exception as e:
            self.logger.error(f"Bandstop filter design error: {e}")
            order = order or 51
            return signal.firwin(order, [f_low, f_high], fs=self.sample_rate)
    
    def design_anti_aliasing_filter(self) -> np.ndarray:
        """Design anti-aliasing filter"""
        # Design for Nyquist frequency with steep rolloff
        cutoff = self.sample_rate * 0.45  # 90% of Nyquist
        transition_width = self.sample_rate * 0.05
        return self.design_lowpass_fir(cutoff, transition_width, window='kaiser')
    
    def design_reconstruction_filter(self) -> np.ndarray:
        """Design reconstruction filter for DAC"""
        # Similar to anti-aliasing but optimized for reconstruction
        cutoff = self.sample_rate * 0.4
        transition_width = self.sample_rate * 0.1
        return self.design_lowpass_fir(cutoff, transition_width, window='kaiser')
    
    def design_matched_filter(self, pulse_shape: str = 'rectangular', symbol_period: float = None) -> np.ndarray:
        """Design matched filter for given pulse shape"""
        try:
            if symbol_period is None:
                symbol_period = 8 / self.sample_rate  # 8 samples per symbol default
            
            symbol_samples = int(symbol_period * self.sample_rate)
            
            if pulse_shape == 'rectangular':
                return np.ones(symbol_samples) / symbol_samples
            
            elif pulse_shape == 'raised_cosine' and SDR_AVAILABLE:
                return sdr.raised_cosine(0.5, symbol_samples, symbol_samples * 2 + 1)
            
            elif pulse_shape == 'gaussian':
                # Gaussian pulse
                t = np.arange(-symbol_samples, symbol_samples + 1) / self.sample_rate
                sigma = symbol_period / 4  # Adjust as needed
                return np.exp(-0.5 * (t / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            
            else:
                # Default to rectangular
                return np.ones(symbol_samples) / symbol_samples
                
        except Exception as e:
            self.logger.error(f"Matched filter design error: {e}")
            return np.ones(8) / 8  # Simple 8-tap moving average
    
    def _estimate_fir_order(self, transition_width: float) -> int:
        """Estimate required FIR filter order"""
        # Kaiser formula approximation
        stopband_attenuation = 80  # dB
        normalized_transition = transition_width / self.sample_rate
        
        if normalized_transition <= 0:
            return 51  # Default
        
        order = int(np.ceil((stopband_attenuation - 7.95) / (14.36 * normalized_transition)))
        
        # Ensure odd order for linear phase
        if order % 2 == 0:
            order += 1
        
        # Reasonable limits
        return max(21, min(order, 1001))
    
    def apply_filter(self, data: np.ndarray, filter_name: str, mode: str = 'same') -> np.ndarray:
        """Apply specified filter to data"""
        if filter_name not in self.filters:
            self.logger.warning(f"Filter '{filter_name}' not found")
            return data
        
        try:
            filter_coeffs = self.filters[filter_name]
            
            # Handle different filter types
            if isinstance(filter_coeffs, np.ndarray):
                # Standard FIR coefficients
                return signal.convolve(data, filter_coeffs, mode=mode)
            elif hasattr(filter_coeffs, '__call__'):
                # SDR library filter object
                return filter_coeffs(data)
            else:
                self.logger.warning(f"Unknown filter type for '{filter_name}'")
                return data
                
        except Exception as e:
            self.logger.error(f"Error applying filter '{filter_name}': {e}")
            return data
    
    def get_filter_response(self, filter_name: str, npoints: int = 1024) -> Optional[FilterResponse]:
        """Get frequency response of specified filter"""
        if filter_name not in self.filters:
            return None
        
        try:
            filter_coeffs = self.filters[filter_name]
            
            if isinstance(filter_coeffs, np.ndarray):
                # Compute frequency response
                w, h = signal.freqz(filter_coeffs, worN=npoints, fs=self.sample_rate)
                
                # Compute group delay
                _, gd = signal.group_delay((filter_coeffs, 1), fs=self.sample_rate)
                
                return FilterResponse(
                    frequencies=w,
                    magnitude=20 * np.log10(np.abs(h) + 1e-12),
                    phase=np.angle(h),
                    group_delay=gd
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error computing filter response: {e}")
            return None
    
    def list_available_filters(self) -> List[str]:
        """List all available filters"""
        return list(self.filters.keys())


class IIRFilterBank:
    """Collection of IIR filters using scikit-dsp-comm"""
    
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
        # Available filters (stored as SOS format)
        self.filters = {}
        
        # Filter states for stateful filtering
        self.filter_states = {}
        
        self.initialize_filters()
    
    def initialize_filters(self):
        """Initialize various IIR filters"""
        try:
            # Butterworth filters
            self.filters['butter_lp_1MHz'] = self.design_butterworth_lp(1e6, 8)
            self.filters['butter_hp_1MHz'] = self.design_butterworth_hp(1e6, 6)
            self.filters['butter_bp_10_15MHz'] = self.design_butterworth_bp(10e6, 15e6, 6)
            
            # Chebyshev Type I filters
            self.filters['cheby1_lp_1MHz'] = self.design_chebyshev1_lp(1e6, 6, 0.5)
            self.filters['cheby1_bp_5_10MHz'] = self.design_chebyshev1_bp(5e6, 10e6, 4, 0.5)
            
            # Chebyshev Type II filters
            self.filters['cheby2_lp_2MHz'] = self.design_chebyshev2_lp(2e6, 6, 60)
            
            # Elliptic filters
            self.filters['elliptic_lp_1MHz'] = self.design_elliptic_lp(1e6, 6, 0.5, 60)
            self.filters['elliptic_bp_8_12MHz'] = self.design_elliptic_bp(8e6, 12e6, 4, 0.5, 60)
            
            # Special purpose filters
            self.filters['dc_blocker'] = self.design_dc_blocker()
            self.filters['notch_60Hz'] = self.design_notch_filter(60, 5)  # 60 Hz notch
            self.filters['notch_50Hz'] = self.design_notch_filter(50, 5)  # 50 Hz notch
            
            # Initialize filter states
            for filter_name in self.filters:
                self.filter_states[filter_name] = None
            
            self.logger.info(f"Initialized {len(self.filters)} IIR filters")
            
        except Exception as e:
            self.logger.error(f"IIR filter initialization error: {e}")
    
    def design_butterworth_lp(self, cutoff: float, order: int) -> np.ndarray:
        """Design Butterworth lowpass filter"""
        try:
            if SCIKIT_DSP_AVAILABLE and iir_design:
                return iir_design.iir_d(order, cutoff, 0, self.sample_rate, ftype='butter')
            else:
                return signal.butter(order, cutoff, btype='low', fs=self.sample_rate, output='sos')
        except Exception as e:
            self.logger.error(f"Butterworth LP design error: {e}")
            return signal.butter(6, cutoff, btype='low', fs=self.sample_rate, output='sos')
    
    def design_butterworth_hp(self, cutoff: float, order: int) -> np.ndarray:
        """Design Butterworth highpass filter"""
        try:
            if SCIKIT_DSP_AVAILABLE and iir_design:
                return iir_design.iir_d(order, cutoff, 0, self.sample_rate, ftype='butter', btype='high')
            else:
                return signal.butter(order, cutoff, btype='high', fs=self.sample_rate, output='sos')
        except Exception as e:
            self.logger.error(f"Butterworth HP design error: {e}")
            return signal.butter(6, cutoff, btype='high', fs=self.sample_rate, output='sos')
    
    def design_butterworth_bp(self, f_low: float, f_high: float, order: int) -> np.ndarray:
        """Design Butterworth bandpass filter"""
        try:
            if SCIKIT_DSP_AVAILABLE and iir_design:
                return iir_design.iir_d(order, [f_low, f_high], 0, self.sample_rate, ftype='butter', btype='band')
            else:
                return signal.butter(order, [f_low, f_high], btype='band', fs=self.sample_rate, output='sos')
        except Exception as e:
            self.logger.error(f"Butterworth BP design error: {e}")
            return signal.butter(4, [f_low, f_high], btype='band', fs=self.sample_rate, output='sos')
    
    def design_chebyshev1_lp(self, cutoff: float, order: int, ripple: float) -> np.ndarray:
        """Design Chebyshev Type I lowpass filter"""
        try:
            if SCIKIT_DSP_AVAILABLE and iir_design:
                return iir_design.iir_d(order, cutoff, ripple, self.sample_rate, ftype='cheby1')
            else:
                return signal.cheby1(order, ripple, cutoff, btype='low', fs=self.sample_rate, output='sos')
        except Exception as e:
            self.logger.error(f"Chebyshev1 LP design error: {e}")
            return signal.cheby1(6, ripple, cutoff, btype='low', fs=self.sample_rate, output='sos')
    
    def design_chebyshev1_bp(self, f_low: float, f_high: float, order: int, ripple: float) -> np.ndarray:
        """Design Chebyshev Type I bandpass filter"""
        try:
            return signal.cheby1(order, ripple, [f_low, f_high], btype='band', fs=self.sample_rate, output='sos')
        except Exception as e:
            self.logger.error(f"Chebyshev1 BP design error: {e}")
            return signal.cheby1(4, ripple, [f_low, f_high], btype='band', fs=self.sample_rate, output='sos')
    
    def design_chebyshev2_lp(self, cutoff: float, order: int, stopband_attenuation: float) -> np.ndarray:
        """Design Chebyshev Type II lowpass filter"""
        try:
            return signal.cheby2(order, stopband_attenuation, cutoff, btype='low', fs=self.sample_rate, output='sos')
        except Exception as e:
            self.logger.error(f"Chebyshev2 LP design error: {e}")
            return signal.cheby2(6, stopband_attenuation, cutoff, btype='low', fs=self.sample_rate, output='sos')
    
    def design_elliptic_lp(self, cutoff: float, order: int, ripple: float, stopband_attenuation: float) -> np.ndarray:
        """Design elliptic lowpass filter"""
        try:
            return signal.ellip(order, ripple, stopband_attenuation, cutoff, btype='low', fs=self.sample_rate, output='sos')
        except Exception as e:
            self.logger.error(f"Elliptic LP design error: {e}")
            return signal.ellip(6, ripple, stopband_attenuation, cutoff, btype='low', fs=self.sample_rate, output='sos')
    
    def design_elliptic_bp(self, f_low: float, f_high: float, order: int, ripple: float, stopband_attenuation: float) -> np.ndarray:
        """Design elliptic bandpass filter"""
        try:
            return signal.ellip(order, ripple, stopband_attenuation, [f_low, f_high], btype='band', fs=self.sample_rate, output='sos')
        except Exception as e:
            self.logger.error(f"Elliptic BP design error: {e}")
            return signal.ellip(4, ripple, stopband_attenuation, [f_low, f_high], btype='band', fs=self.sample_rate, output='sos')
    
    def design_dc_blocker(self, pole_radius: float = 0.995) -> np.ndarray:
        """Design DC blocking filter"""
        try:
            # Simple DC blocker: H(z) = (1 - z^-1) / (1 - r*z^-1)
            # where r is close to 1 (e.g., 0.995)
            b = [1, -1]
            a = [1, -pole_radius]
            return signal.tf2sos(b, a)
        except Exception as e:
            self.logger.error(f"DC blocker design error: {e}")
            # Fallback: simple differentiator
            b = [1, -1]
            a = [1, 0]
            return signal.tf2sos(b, a)
    
    def design_notch_filter(self, notch_freq: float, q_factor: float) -> np.ndarray:
        """Design notch filter for specific frequency"""
        try:
            return signal.iirnotch(notch_freq, q_factor, fs=self.sample_rate)
        except Exception as e:
            self.logger.error(f"Notch filter design error: {e}")
            # Simple notch using bandstop Butterworth
            f_low = notch_freq - notch_freq / (2 * q_factor)
            f_high = notch_freq + notch_freq / (2 * q_factor)
            return signal.butter(4, [f_low, f_high], btype='bandstop', fs=self.sample_rate, output='sos')
    
    def apply_filter(self, data: np.ndarray, filter_name: str, use_state: bool = False) -> np.ndarray:
        """Apply IIR filter to data"""
        if filter_name not in self.filters:
            self.logger.warning(f"IIR filter '{filter_name}' not found")
            return data
        
        try:
            sos = self.filters[filter_name]
            
            if use_state:
                # Stateful filtering
                if self.filter_states[filter_name] is None:
                    # Initialize filter state
                    self.filter_states[filter_name] = signal.sosfilt_zi(sos) * data[0]
                
                filtered_data, self.filter_states[filter_name] = signal.sosfilt(
                    sos, data, zi=self.filter_states[filter_name]
                )
                return filtered_data
            else:
                # Stateless filtering
                return signal.sosfilt(sos, data)
                
        except Exception as e:
            self.logger.error(f"Error applying IIR filter '{filter_name}': {e}")
            return data
    
    def reset_filter_state(self, filter_name: str):
        """Reset filter state for stateful filtering"""
        if filter_name in self.filter_states:
            self.filter_states[filter_name] = None
    
    def reset_all_filter_states(self):
        """Reset all filter states"""
        for filter_name in self.filter_states:
            self.filter_states[filter_name] = None
    
    def get_filter_response(self, filter_name: str, npoints: int = 1024) -> Optional[FilterResponse]:
        """Get frequency response of IIR filter"""
        if filter_name not in self.filters:
            return None
        
        try:
            sos = self.filters[filter_name]
            w, h = signal.sosfreqz(sos, worN=npoints, fs=self.sample_rate)
            
            # Compute group delay
            _, gd = signal.group_delay(sos, fs=self.sample_rate)
            
            return FilterResponse(
                frequencies=w,
                magnitude=20 * np.log10(np.abs(h) + 1e-12),
                phase=np.angle(h),
                group_delay=gd
            )
            
        except Exception as e:
            self.logger.error(f"Error computing IIR filter response: {e}")
            return None
    
    def list_available_filters(self) -> List[str]:
        """List all available IIR filters"""
        return list(self.filters.keys())


# Test functions
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test FIR filters
    sample_rate = 10e6  # 10 MHz
    fir_bank = FIRFilterBank(sample_rate)
    
    print(f"Available FIR filters: {fir_bank.list_available_filters()}")
    
    # Test filter application
    test_signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    filtered_signal = fir_bank.apply_filter(test_signal, 'lowpass_1MHz')
    print(f"Original signal length: {len(test_signal)}, Filtered: {len(filtered_signal)}")
    
    # Test IIR filters
    iir_bank = IIRFilterBank(sample_rate)
    
    print(f"Available IIR filters: {iir_bank.list_available_filters()}")
    
    # Test IIR filter application
    iir_filtered = iir_bank.apply_filter(test_signal, 'butter_lp_1MHz')
    print(f"IIR filtered signal length: {len(iir_filtered)}")
    
    # Test filter responses
    fir_response = fir_bank.get_filter_response('lowpass_1MHz')
    if fir_response:
        print(f"FIR filter response computed: {len(fir_response.frequencies)} points")
    
    iir_response = iir_bank.get_filter_response('butter_lp_1MHz')
    if iir_response:
        print(f"IIR filter response computed: {len(iir_response.frequencies)} points")
'''

with open("rf_spectrum_analyzer/dsp/filters.py", "w") as f:
    f.write(dsp_filters_content)

print("Created dsp/filters.py")