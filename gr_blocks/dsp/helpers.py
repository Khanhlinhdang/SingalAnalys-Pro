"""
Utility Helper Functions

Common utility functions used throughout the RF Spectrum Analyzer application.
Includes signal processing utilities, file I/O helpers, and general utilities.
"""

import numpy as np
import logging
import os
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import tempfile
import hashlib

# Optional imports for extended functionality
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def db_to_linear(db_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert dB value to linear scale"""
    return 10.0 ** (db_value / 10.0)


def linear_to_db(linear_value: Union[float, np.ndarray], min_db: float = -120.0) -> Union[float, np.ndarray]:
    """Convert linear value to dB scale with minimum floor"""
    with np.errstate(divide='ignore', invalid='ignore'):
        db_value = 10.0 * np.log10(np.abs(linear_value) + 1e-12)

    # Apply minimum dB floor
    if isinstance(db_value, np.ndarray):
        db_value = np.maximum(db_value, min_db)
    else:
        db_value = max(db_value, min_db)

    return db_value


def power_to_dbm(power_watts: Union[float, np.ndarray], impedance: float = 50.0) -> Union[float, np.ndarray]:
    """Convert power in watts to dBm (referenced to 1mW)"""
    power_mw = power_watts * 1000.0  # Convert to milliwatts
    return linear_to_db(power_mw)


def dbm_to_power(dbm_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert dBm to power in watts"""
    power_mw = db_to_linear(dbm_value)
    return power_mw / 1000.0  # Convert to watts


def rms_to_peak(rms_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert RMS value to peak value (assumes sinusoidal)"""
    return rms_value * np.sqrt(2)


def peak_to_rms(peak_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert peak value to RMS value (assumes sinusoidal)"""
    return peak_value / np.sqrt(2)


def frequency_to_wavelength(frequency: Union[float, np.ndarray], speed_of_light: float = 299792458.0) -> Union[float, np.ndarray]:
    """Convert frequency to wavelength"""
    return speed_of_light / frequency


def wavelength_to_frequency(wavelength: Union[float, np.ndarray], speed_of_light: float = 299792458.0) -> Union[float, np.ndarray]:
    """Convert wavelength to frequency"""
    return speed_of_light / wavelength


def format_frequency(frequency: float, precision: int = 3) -> str:
    """Format frequency with appropriate units"""
    if frequency >= 1e9:
        return f"{frequency/1e9:.{precision}f} GHz"
    elif frequency >= 1e6:
        return f"{frequency/1e6:.{precision}f} MHz"
    elif frequency >= 1e3:
        return f"{frequency/1e3:.{precision}f} kHz"
    else:
        return f"{frequency:.{precision}f} Hz"


def format_sample_rate(sample_rate: float, precision: int = 1) -> str:
    """Format sample rate with appropriate units"""
    if sample_rate >= 1e9:
        return f"{sample_rate/1e9:.{precision}f} GSps"
    elif sample_rate >= 1e6:
        return f"{sample_rate/1e6:.{precision}f} MSps"
    elif sample_rate >= 1e3:
        return f"{sample_rate/1e3:.{precision}f} kSps"
    else:
        return f"{sample_rate:.{precision}f} Sps"


def format_power(power_dbm: float, precision: int = 1) -> str:
    """Format power in dBm"""
    return f"{power_dbm:.{precision}f} dBm"


def format_bandwidth(bandwidth: float, precision: int = 1) -> str:
    """Format bandwidth with appropriate units"""
    if bandwidth >= 1e9:
        return f"{bandwidth/1e9:.{precision}f} GHz"
    elif bandwidth >= 1e6:
        return f"{bandwidth/1e6:.{precision}f} MHz"
    elif bandwidth >= 1e3:
        return f"{bandwidth/1e3:.{precision}f} kHz"
    else:
        return f"{bandwidth:.{precision}f} Hz"


def next_power_of_2(n: int) -> int:
    """Find next power of 2 greater than or equal to n"""
    return int(2 ** np.ceil(np.log2(n)))


def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0


def circular_buffer(buffer: np.ndarray, new_data: np.ndarray) -> np.ndarray:
    """Add new data to circular buffer, removing oldest data if necessary"""
    if len(new_data) >= len(buffer):
        # New data is larger than buffer, just keep the latest part
        return new_data[-len(buffer):]
    else:
        # Shift buffer and add new data
        buffer = np.roll(buffer, -len(new_data))
        buffer[-len(new_data):] = new_data
        return buffer


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute moving average with specified window size"""
    if window_size <= 1:
        return data

    # Use convolution for efficiency
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')


def exponential_moving_average(data: np.ndarray, alpha: float) -> np.ndarray:
    """Compute exponential moving average with smoothing factor alpha"""
    if len(data) == 0:
        return data

    ema = np.zeros_like(data)
    ema[0] = data[0]

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

    return ema


def find_peaks_simple(data: np.ndarray, threshold: float = None, min_distance: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Simple peak finding algorithm"""
    if threshold is None:
        threshold = np.mean(data) + np.std(data)

    peaks = []
    values = []

    for i in range(1, len(data) - 1):
        if (data[i] > data[i-1] and 
            data[i] > data[i+1] and 
            data[i] > threshold):

            # Check minimum distance from previous peaks
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
                values.append(data[i])

    return np.array(peaks), np.array(values)


def calculate_snr(signal: np.ndarray, noise: np.ndarray = None) -> float:
    """Calculate Signal-to-Noise Ratio in dB"""
    try:
        if noise is not None:
            signal_power = np.mean(np.abs(signal) ** 2)
            noise_power = np.mean(np.abs(noise) ** 2)
        else:
            # Estimate noise from signal (simple method)
            signal_power = np.mean(np.abs(signal) ** 2)
            # Assume noise is in the higher frequencies
            fft_signal = np.fft.fft(signal)
            noise_samples = fft_signal[-len(fft_signal)//4:]  # Last quarter of spectrum
            noise_power = np.mean(np.abs(noise_samples) ** 2)

        if noise_power > 0:
            snr_linear = signal_power / noise_power
            return linear_to_db(snr_linear)
        else:
            return 60.0  # High SNR if no noise

    except Exception:
        return 0.0


def calculate_thd(signal: np.ndarray, fundamental_freq: float, sample_rate: float, harmonics: int = 5) -> float:
    """Calculate Total Harmonic Distortion in dB"""
    try:
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)

        # Find fundamental peak
        fundamental_bin = int(fundamental_freq * len(signal) / sample_rate)
        fundamental_power = np.abs(fft_signal[fundamental_bin]) ** 2

        # Calculate harmonic powers
        harmonic_power = 0
        for h in range(2, harmonics + 1):
            harmonic_freq = h * fundamental_freq
            harmonic_bin = int(harmonic_freq * len(signal) / sample_rate)
            if harmonic_bin < len(fft_signal) // 2:
                harmonic_power += np.abs(fft_signal[harmonic_bin]) ** 2

        if fundamental_power > 0:
            thd_linear = harmonic_power / fundamental_power
            return linear_to_db(thd_linear)
        else:
            return -60.0  # Low THD if no fundamental

    except Exception:
        return 0.0


def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration file (JSON or YAML)"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                if YAML_AVAILABLE:
                    return yaml.safe_load(f)
                else:
                    raise ImportError("PyYAML not available for YAML files")
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                # Try JSON first, then YAML
                content = f.read()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    if YAML_AVAILABLE:
                        return yaml.safe_load(content)
                    else:
                        raise ValueError("Unknown configuration file format")

    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading config file {config_path}: {e}")
        return {}


def save_config_file(config_data: Dict[str, Any], config_path: Union[str, Path], format: str = 'auto') -> bool:
    """Save configuration file"""
    config_path = Path(config_path)

    try:
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'auto':
            format = config_path.suffix.lower().lstrip('.')

        with open(config_path, 'w', encoding='utf-8') as f:
            if format in ['yaml', 'yml']:
                if YAML_AVAILABLE:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    raise ImportError("PyYAML not available for YAML files")
            elif format == 'json':
                json.dump(config_data, f, indent=2)
            else:
                # Default to JSON
                json.dump(config_data, f, indent=2)

        return True

    except Exception as e:
        logging.getLogger(__name__).error(f"Error saving config file {config_path}: {e}")
        return False


def create_temp_file(suffix: str = '', prefix: str = 'rfa_', dir: Optional[str] = None) -> str:
    """Create a temporary file and return its path"""
    with tempfile.NamedTemporaryFile(suffix=suffix, prefix=prefix, dir=dir, delete=False) as f:
        return f.name


def cleanup_temp_files(pattern: str = 'rfa_*'):
    """Clean up temporary files matching pattern"""
    try:
        temp_dir = Path(tempfile.gettempdir())
        for temp_file in temp_dir.glob(pattern):
            try:
                temp_file.unlink()
            except:
                pass  # Ignore errors
    except Exception:
        pass  # Ignore errors


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """Calculate hash of file contents"""
    file_path = Path(file_path)

    if algorithm.lower() == 'md5':
        hash_obj = hashlib.md5()
    elif algorithm.lower() == 'sha256':
        hash_obj = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    except Exception as e:
        logging.getLogger(__name__).error(f"Error calculating file hash: {e}")
        return ""


def ensure_directory(directory: Union[str, Path]) -> bool:
    """Ensure directory exists, create if necessary"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Error creating directory {directory}: {e}")
        return False


def safe_division(numerator: Union[float, np.ndarray], 
                 denominator: Union[float, np.ndarray], 
                 default: float = 0.0) -> Union[float, np.ndarray]:
    """Safe division with default value for division by zero"""
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)

        # Replace inf and nan with default
        if isinstance(result, np.ndarray):
            result = np.where(np.isfinite(result), result, default)
        else:
            if not np.isfinite(result):
                result = default

        return result

    except Exception:
        if isinstance(numerator, np.ndarray):
            return np.full_like(numerator, default)
        else:
            return default


def validate_sample_rate(sample_rate: float) -> bool:
    """Validate sample rate value"""
    return (isinstance(sample_rate, (int, float)) and 
            sample_rate > 0 and 
            sample_rate <= 100e9)  # Max 100 GHz


def validate_frequency(frequency: float) -> bool:
    """Validate frequency value"""
    return (isinstance(frequency, (int, float)) and 
            frequency >= 0 and 
            frequency <= 300e9)  # Max 300 GHz


def validate_gain(gain: float) -> bool:
    """Validate gain value"""
    return (isinstance(gain, (int, float)) and 
            gain >= -20 and 
            gain <= 120)  # Typical gain range


def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to specified range"""
    return max(min_val, min(value, max_val))


def interpolate_spectrum(frequencies: np.ndarray, 
                        magnitudes: np.ndarray, 
                        new_frequencies: np.ndarray) -> np.ndarray:
    """Interpolate spectrum data to new frequency grid"""
    try:
        return np.interp(new_frequencies, frequencies, magnitudes)
    except Exception as e:
        logging.getLogger(__name__).error(f"Spectrum interpolation error: {e}")
        return np.zeros_like(new_frequencies)


def decimate_spectrum(frequencies: np.ndarray, 
                     magnitudes: np.ndarray, 
                     decimation_factor: int) -> Tuple[np.ndarray, np.ndarray]:
    """Decimate spectrum data by given factor"""
    try:
        if decimation_factor <= 1:
            return frequencies, magnitudes

        decimated_freq = frequencies[::decimation_factor]
        decimated_mag = magnitudes[::decimation_factor]

        return decimated_freq, decimated_mag

    except Exception as e:
        logging.getLogger(__name__).error(f"Spectrum decimation error: {e}")
        return frequencies, magnitudes


class PerformanceTimer:
    """Simple performance timer context manager"""

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        self.logger.debug(f"{self.name} took {duration:.3f} seconds")

    def get_duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


# Test functions
if __name__ == "__main__":
    # Test utility functions
    print("Testing RF Spectrum Analyzer Utilities")

    # Test conversions
    print(f"20 dB = {db_to_linear(20):.1f} linear")
    print(f"100 linear = {linear_to_db(100):.1f} dB")

    # Test formatting
    print(f"433.92e6 Hz = {format_frequency(433.92e6)}")
    print(f"2.4e6 Sps = {format_sample_rate(2.4e6)}")
    print(f"1e6 Hz BW = {format_bandwidth(1e6)}")

    # Test signal processing
    test_signal = np.sin(2 * np.pi * np.linspace(0, 1, 1000)) + 0.1 * np.random.randn(1000)

    # Moving average
    smoothed = moving_average(test_signal, 10)
    print(f"Original signal std: {np.std(test_signal):.3f}, Smoothed: {np.std(smoothed):.3f}")

    # Peak finding
    peaks, values = find_peaks_simple(np.abs(test_signal), threshold=0.5)
    print(f"Found {len(peaks)} peaks")

    # Performance timer test
    with PerformanceTimer("Test operation"):
        # Simulate some work
        np.fft.fft(np.random.randn(10000))

    print("Utility tests completed")
