"""
Helper utilities for RF Spectrum Analyzer
Contains various utility functions used throughout the application
"""

import numpy as np
import math
import time
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import queue
from collections import deque

from rf_spectrum_analyzer.utils.logger import get_logger

logger = get_logger('helpers')

# Unit conversion utilities
class FrequencyUnit(Enum):
    """Frequency unit enumeration"""
    HZ = 'Hz'
    KHZ = 'kHz'
    MHZ = 'MHz'
    GHZ = 'GHz'

class PowerUnit(Enum):
    """Power unit enumeration"""
    DB = 'dB'
    DBM = 'dBm'
    WATTS = 'W'
    MILLIWATTS = 'mW'

def convert_frequency(value: float, from_unit: FrequencyUnit, to_unit: FrequencyUnit) -> float:
    """Convert frequency between different units"""
    # Convert to Hz first
    hz_value = value
    if from_unit == FrequencyUnit.KHZ:
        hz_value = value * 1e3
    elif from_unit == FrequencyUnit.MHZ:
        hz_value = value * 1e6
    elif from_unit == FrequencyUnit.GHZ:
        hz_value = value * 1e9
    
    # Convert from Hz to target unit
    if to_unit == FrequencyUnit.HZ:
        return hz_value
    elif to_unit == FrequencyUnit.KHZ:
        return hz_value / 1e3
    elif to_unit == FrequencyUnit.MHZ:
        return hz_value / 1e6
    elif to_unit == FrequencyUnit.GHZ:
        return hz_value / 1e9
    
    return hz_value

def format_frequency(freq_hz: float, auto_unit: bool = True) -> str:
    """Format frequency with appropriate unit"""
    if auto_unit:
        if abs(freq_hz) >= 1e9:
            return f"{freq_hz/1e9:.3f} GHz"
        elif abs(freq_hz) >= 1e6:
            return f"{freq_hz/1e6:.3f} MHz"
        elif abs(freq_hz) >= 1e3:
            return f"{freq_hz/1e3:.3f} kHz"
        else:
            return f"{freq_hz:.1f} Hz"
    else:
        return f"{freq_hz:.0f} Hz"

def format_power(power_db: float, unit: PowerUnit = PowerUnit.DB) -> str:
    """Format power with specified unit"""
    if unit == PowerUnit.DB:
        return f"{power_db:.1f} dB"
    elif unit == PowerUnit.DBM:
        return f"{power_db:.1f} dBm"
    elif unit == PowerUnit.WATTS:
        watts = 10**(power_db/10) / 1000  # Assuming dBm input
        return f"{watts:.6f} W"
    elif unit == PowerUnit.MILLIWATTS:
        mw = 10**(power_db/10)  # Assuming dBm input
        return f"{mw:.3f} mW"
    
    return f"{power_db:.1f} dB"

def db_to_linear(db_value: float) -> float:
    """Convert dB to linear scale"""
    return 10**(db_value / 10)

def linear_to_db(linear_value: float) -> float:
    """Convert linear scale to dB"""
    return 10 * np.log10(linear_value)

def dbm_to_watts(dbm_value: float) -> float:
    """Convert dBm to watts"""
    return 10**((dbm_value - 30) / 10)

def watts_to_dbm(watts_value: float) -> float:
    """Convert watts to dBm"""
    return 10 * np.log10(watts_value) + 30

# Signal processing utilities
def next_power_of_2(n: int) -> int:
    """Find the next power of 2 greater than or equal to n"""
    return 1 << (n - 1).bit_length()

def normalize_array(data: np.ndarray, method: str = 'max') -> np.ndarray:
    """
    Normalize array using different methods
    
    Args:
        data: Input array
        method: Normalization method ('max', 'minmax', 'std', 'rms')
    
    Returns:
        Normalized array
    """
    if method == 'max':
        return data / np.max(np.abs(data))
    elif method == 'minmax':
        min_val, max_val = np.min(data), np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif method == 'std':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'rms':
        return data / np.sqrt(np.mean(data**2))
    else:
        return data

def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Calculate moving average of data"""
    if window_size <= 1:
        return data
    
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

def exponential_smoothing(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply exponential smoothing to data"""
    if len(data) == 0:
        return data
    
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed

def find_peaks(data: np.ndarray, threshold: float = None, min_distance: int = 1) -> List[int]:
    """
    Find peaks in data
    
    Args:
        data: Input data array
        threshold: Minimum peak height (auto if None)
        min_distance: Minimum distance between peaks
    
    Returns:
        List of peak indices
    """
    if threshold is None:
        threshold = np.mean(data) + np.std(data)
    
    peaks = []
    for i in range(1, len(data) - 1):
        if (data[i] > data[i-1] and 
            data[i] > data[i+1] and 
            data[i] > threshold):
            
            # Check minimum distance
            if not peaks or i - peaks[-1] >= min_distance:
                peaks.append(i)
    
    return peaks

def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio"""
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)

def estimate_noise_floor(spectrum: np.ndarray, percentile: float = 10) -> float:
    """Estimate noise floor from spectrum"""
    return np.percentile(spectrum, percentile)

# Data validation utilities
def validate_frequency_range(freq_start: float, freq_stop: float) -> bool:
    """Validate frequency range parameters"""
    return (freq_start < freq_stop and 
            freq_start >= 0 and 
            freq_stop > 0)

def validate_sample_rate(sample_rate: float, max_rate: float = 100e6) -> bool:
    """Validate sample rate parameter"""
    return 0 < sample_rate <= max_rate

def validate_fft_size(fft_size: int) -> bool:
    """Validate FFT size (should be power of 2)"""
    return fft_size > 0 and (fft_size & (fft_size - 1)) == 0

def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to specified range"""
    return max(min_val, min(max_val, value))

# Threading utilities
class ThreadSafeCounter:
    """Thread-safe counter"""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
    
    def decrement(self) -> int:
        with self._lock:
            self._value -= 1
            return self._value
    
    def get_value(self) -> int:
        with self._lock:
            return self._value
    
    def set_value(self, value: int):
        with self._lock:
            self._value = value

class CircularBuffer:
    """Thread-safe circular buffer for storing recent data"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def append(self, item: Any):
        with self.lock:
            self.buffer.append(item)
    
    def get_all(self) -> List[Any]:
        with self.lock:
            return list(self.buffer)
    
    def get_latest(self, n: int = 1) -> List[Any]:
        with self.lock:
            return list(self.buffer)[-n:] if len(self.buffer) >= n else list(self.buffer)
    
    def clear(self):
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        with self.lock:
            return len(self.buffer)

class RateLimiter:
    """Rate limiter for controlling function call frequency"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = threading.Lock()
    
    def can_proceed(self) -> bool:
        """Check if operation can proceed without exceeding rate limit"""
        with self.lock:
            now = time.time()
            
            # Remove old calls outside the time window
            while self.calls and now - self.calls[0] > self.time_window:
                self.calls.popleft()
            
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            
            return False
    
    def wait_time(self) -> float:
        """Get time to wait before next call can proceed"""
        with self.lock:
            if len(self.calls) < self.max_calls:
                return 0.0
            
            now = time.time()
            oldest_call = self.calls[0]
            return max(0.0, self.time_window - (now - oldest_call))

# Performance monitoring utilities
@dataclass
class PerformanceStats:
    """Container for performance statistics"""
    min_time: float
    max_time: float
    avg_time: float
    total_calls: int
    total_time: float

class PerformanceTracker:
    """Track performance metrics for functions"""
    
    def __init__(self):
        self.stats = {}
        self.lock = threading.Lock()
    
    def record_execution(self, function_name: str, execution_time: float):
        """Record execution time for a function"""
        with self.lock:
            if function_name not in self.stats:
                self.stats[function_name] = {
                    'times': [],
                    'total_calls': 0,
                    'total_time': 0.0
                }
            
            stats = self.stats[function_name]
            stats['times'].append(execution_time)
            stats['total_calls'] += 1
            stats['total_time'] += execution_time
            
            # Keep only recent measurements (last 1000)
            if len(stats['times']) > 1000:
                removed_time = stats['times'].pop(0)
                stats['total_time'] -= removed_time
                stats['total_calls'] -= 1
    
    def get_stats(self, function_name: str) -> Optional[PerformanceStats]:
        """Get performance statistics for a function"""
        with self.lock:
            if function_name not in self.stats:
                return None
            
            times = self.stats[function_name]['times']
            if not times:
                return None
            
            return PerformanceStats(
                min_time=min(times),
                max_time=max(times),
                avg_time=sum(times) / len(times),
                total_calls=self.stats[function_name]['total_calls'],
                total_time=self.stats[function_name]['total_time']
            )
    
    def get_all_stats(self) -> Dict[str, PerformanceStats]:
        """Get performance statistics for all tracked functions"""
        with self.lock:
            result = {}
            for func_name in self.stats:
                stats = self.get_stats(func_name)
                if stats:
                    result[func_name] = stats
            return result

def timing_decorator(tracker: PerformanceTracker = None):
    """Decorator to measure function execution time"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                if tracker:
                    tracker.record_execution(func.__name__, execution_time)
                logger.debug(f"{func.__name__} executed in {execution_time:.4f}s")
        return wrapper
    return decorator

# Mathematical utilities
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that handles division by zero"""
    return numerator / denominator if denominator != 0 else default

def interpolate_linear(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Linear interpolation for resampling data"""
    return np.interp(x_new, x, y)

def calculate_bandwidth(frequencies: np.ndarray, power: np.ndarray, 
                       db_down: float = 3.0) -> Tuple[float, float, float]:
    """
    Calculate signal bandwidth at specified dB down point
    
    Returns:
        Tuple of (bandwidth, center_frequency, peak_frequency)
    """
    # Find peak
    peak_idx = np.argmax(power)
    peak_power = power[peak_idx]
    peak_freq = frequencies[peak_idx]
    
    # Find points at db_down below peak
    threshold = peak_power - db_down
    above_threshold = power >= threshold
    
    if not np.any(above_threshold):
        return 0.0, peak_freq, peak_freq
    
    # Find bandwidth edges
    indices = np.where(above_threshold)[0]
    start_freq = frequencies[indices[0]]
    stop_freq = frequencies[indices[-1]]
    
    bandwidth = stop_freq - start_freq
    center_freq = (start_freq + stop_freq) / 2
    
    return bandwidth, center_freq, peak_freq

# Global performance tracker instance
global_performance_tracker = PerformanceTracker()

# Commonly used rate limiters
ui_update_limiter = RateLimiter(max_calls=30, time_window=1.0)  # 30 FPS
processing_limiter = RateLimiter(max_calls=100, time_window=1.0)  # 100 Hz processing