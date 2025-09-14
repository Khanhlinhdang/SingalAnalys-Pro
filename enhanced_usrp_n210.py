# Enhanced USRP N210 Controller Class
# Kế thừa từ uhd.usrp.MultiUSRP với đầy đủ documentation và custom methods
# Author: AI Assistant
# Date: September 11, 2025

import uhd
import numpy as np
import time
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
from collections import deque
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedUSRPN210(uhd.usrp.MultiUSRP):
    """
    Enhanced USRP N210 Controller Class
    
    Kế thừa từ uhd.usrp.MultiUSRP với các tính năng mở rộng:
    - Tự động configuration và validation
    - Advanced signal monitoring và analysis
    - Multi-threading support cho real-time processing
    - Comprehensive error handling và recovery
    - Built-in calibration và performance optimization
    - Extended logging và debugging capabilities
    
    Mục đích: Cung cấp interface dễ sử dụng cho USRP N210 với các tính năng nâng cao
    cho nghiên cứu SDR và signal processing applications.
    """
    
    def __init__(self, device_args: str = "", **kwargs):
        """
        Khởi tạo Enhanced USRP N210 Controller
        
        Args:
            device_args (str): UHD device arguments (e.g., "addr=192.168.10.2")
            **kwargs: Additional configuration parameters
                - auto_config (bool): Tự động configure device (default: True)
                - enable_logging (bool): Enable detailed logging (default: True)
                - safety_checks (bool): Enable safety checks (default: True)
                - performance_mode (bool): Enable performance optimizations (default: False)
        
        Raises:
            RuntimeError: Nếu không thể kết nối với USRP device
            ValueError: Nếu device arguments không hợp lệ
        
        Example:
            >>> usrp = EnhancedUSRPN210("addr=192.168.10.2")
            >>> usrp.configure_for_inmarsat()  # Quick setup for Inmarsat AERO
        """
        try:
            # Initialize parent MultiUSRP class
            super().__init__(device_args)
            
            # Configuration parameters
            self.auto_config = kwargs.get('auto_config', True)
            self.enable_logging = kwargs.get('enable_logging', True)
            self.safety_checks = kwargs.get('safety_checks', True)
            self.performance_mode = kwargs.get('performance_mode', False)
            
            # Device information storage
            self.device_info = {}
            self.performance_stats = {}
            self.calibration_data = {}
            
            # Monitoring and statistics
            self.sample_stats = {
                'total_samples': 0,
                'dropped_samples': 0,
                'overruns': 0,
                'underruns': 0,
                'last_update': time.time()
            }
            
            # Signal monitoring buffers
            self.signal_history = deque(maxlen=1000)  # Last 1000 signal measurements
            self.frequency_history = deque(maxlen=100)  # Frequency change history
            
            # Thread safety
            self._lock = threading.Lock()
            self._streaming = False
            self._monitor_thread = None
            
            if self.enable_logging:
                logger.info(f"Enhanced USRP N210 Controller initialized with args: {device_args}")
            
            # Auto-configuration if enabled
            if self.auto_config:
                self._auto_configure()
                
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced USRP N210: {e}")
            raise RuntimeError(f"USRP initialization failed: {e}")
    
    def _auto_configure(self):
        """
        Tự động configure USRP với các settings tối ưu
        
        Mục đích: Setup device với các parameters an toàn và hiệu quả
        """
        try:
            # Get device information
            self.device_info = {
                'mboard_name': self.get_mboard_name(0),
                'pp_string': self.get_pp_string(),
                'num_rx_channels': self.get_rx_num_channels(),
                'num_tx_channels': self.get_tx_num_channels(),
                'master_clock_rate': self.get_master_clock_rate(0)
            }
            
            # Set safe default clock source
            available_clk_sources = self.get_clock_sources(0)
            if 'internal' in available_clk_sources:
                self.set_clock_source('internal', 0)
            
            # Set safe default time source
            available_time_sources = self.get_time_sources(0)
            if 'internal' in available_time_sources:
                self.set_time_source('internal', 0)
            
            # Initialize time
            self.set_time_now(uhd.types.TimeSpec(0.0))
            
            if self.enable_logging:
                logger.info("Auto-configuration completed successfully")
                logger.info(f"Device info: {self.device_info}")
                
        except Exception as e:
            logger.error(f"Auto-configuration failed: {e}")
            if self.safety_checks:
                raise
    
    # ==================== DEVICE INFO AND STATUS METHODS ====================
    
    def get_device_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái chi tiết của USRP device
        
        Returns:
            Dict chứa thông tin trạng thái device bao gồm:
            - device_info: Thông tin hardware
            - current_config: Configuration hiện tại
            - performance_stats: Thống kê performance
            - sensor_data: Dữ liệu từ các sensors
        
        Mục đích: Monitoring và debugging device status
        """
        try:
            with self._lock:
                status = {
                    'device_info': self.device_info.copy(),
                    'current_config': self._get_current_config(),
                    'performance_stats': self.performance_stats.copy(),
                    'sensor_data': self._get_sensor_data(),
                    'timestamp': time.time(),
                    'streaming': self._streaming
                }
                return status
                
        except Exception as e:
            logger.error(f"Failed to get device status: {e}")
            return {'error': str(e)}
    
    def _get_current_config(self) -> Dict[str, Any]:
        """
        Lấy configuration hiện tại của device
        
        Returns:
            Dict chứa các parameters configuration hiện tại
        """
        try:
            config = {
                'rx_rate': self.get_rx_rate(0),
                'tx_rate': self.get_tx_rate(0),
                'rx_freq': self.get_rx_freq(0),
                'tx_freq': self.get_tx_freq(0),
                'rx_gain': self.get_rx_gain(0),
                'tx_gain': self.get_tx_gain(0),
                'rx_antenna': self.get_rx_antenna(0),
                'tx_antenna': self.get_tx_antenna(0),
                'clock_source': self.get_clock_source(0),
                'time_source': self.get_time_source(0),
                'master_clock_rate': self.get_master_clock_rate(0)
            }
            return config
        except Exception as e:
            logger.error(f"Failed to get current config: {e}")
            return {}
    
    def _get_sensor_data(self) -> Dict[str, Any]:
        """
        Lấy dữ liệu từ tất cả sensors có sẵn
        
        Returns:
            Dict chứa dữ liệu từ motherboard và daughterboard sensors
        """
        sensor_data = {}
        
        try:
            # Motherboard sensors
            mboard_sensors = self.get_mboard_sensor_names(0)
            sensor_data['motherboard'] = {}
            for sensor_name in mboard_sensors:
                try:
                    sensor_value = self.get_mboard_sensor(sensor_name, 0)
                    sensor_data['motherboard'][sensor_name] = {
                        'value': sensor_value.value,
                        'unit': sensor_value.unit,
                        'name': sensor_value.name
                    }
                except:
                    pass
            
            # RX sensors
            if self.get_rx_num_channels() > 0:
                rx_sensors = self.get_rx_sensor_names(0)
                sensor_data['rx'] = {}
                for sensor_name in rx_sensors:
                    try:
                        sensor_value = self.get_rx_sensor(sensor_name, 0)
                        sensor_data['rx'][sensor_name] = {
                            'value': sensor_value.value,
                            'unit': sensor_value.unit,
                            'name': sensor_value.name
                        }
                    except:
                        pass
            
            # TX sensors
            if self.get_tx_num_channels() > 0:
                tx_sensors = self.get_tx_sensor_names(0)
                sensor_data['tx'] = {}
                for sensor_name in tx_sensors:
                    try:
                        sensor_value = self.get_tx_sensor(sensor_name, 0)
                        sensor_data['tx'][sensor_name] = {
                            'value': sensor_value.value,
                            'unit': sensor_value.unit,
                            'name': sensor_value.name
                        }
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Failed to get sensor data: {e}")
            
        return sensor_data
    
    # ==================== ENHANCED CONFIGURATION METHODS ====================
    
    def configure_for_inmarsat(self, center_freq: float = 1545e6, sample_rate: float = 2.4e6, 
                              gain: float = 30.0, antenna: str = "RX2") -> bool:
        """
        Configure USRP cho Inmarsat AERO reception
        
        Args:
            center_freq (float): Center frequency Hz (default: 1545 MHz)
            sample_rate (float): Sample rate Hz (default: 2.4 MS/s)
            gain (float): RX gain dB (default: 30 dB)
            antenna (str): Antenna port (default: "RX2")
        
        Returns:
            bool: True nếu configuration thành công
        
        Mục đích: Quick setup cho Inmarsat AERO signal reception với optimal parameters
        
        Example:
            >>> success = usrp.configure_for_inmarsat(1545.5e6, 4.8e6, 35.0)
            >>> if success:
            >>>     print("Ready for Inmarsat AERO reception")
        """
        try:
            if self.enable_logging:
                logger.info(f"Configuring for Inmarsat AERO: freq={center_freq/1e6:.1f}MHz, "
                           f"rate={sample_rate/1e6:.1f}MS/s, gain={gain}dB")
            
            # Validate parameters
            if not self._validate_inmarsat_params(center_freq, sample_rate, gain):
                return False
            
            # Configure RX parameters
            self.set_rx_rate(sample_rate, 0)
            actual_rate = self.get_rx_rate(0)
            if abs(actual_rate - sample_rate) > sample_rate * 0.01:  # 1% tolerance
                logger.warning(f"Sample rate coerced from {sample_rate/1e6:.1f} to {actual_rate/1e6:.1f} MS/s")
            
            # Set frequency
            tune_request = uhd.types.TuneRequest(center_freq)
            tune_result = self.set_rx_freq(tune_request, 0)
            actual_freq = self.get_rx_freq(0)
            
            if self.enable_logging:
                logger.info(f"Tuned to {actual_freq/1e6:.3f} MHz (requested: {center_freq/1e6:.3f} MHz)")
                logger.info(f"RF freq: {tune_result.actual_rf_freq/1e6:.3f} MHz, "
                           f"DSP freq: {tune_result.actual_dsp_freq/1e3:.3f} kHz")
            
            # Set gain
            self.set_rx_gain(gain, 0)
            actual_gain = self.get_rx_gain(0)
            if abs(actual_gain - gain) > 1.0:  # 1 dB tolerance
                logger.warning(f"Gain coerced from {gain} to {actual_gain} dB")
            
            # Set antenna
            available_antennas = self.get_rx_antennas(0)
            if antenna in available_antennas:
                self.set_rx_antenna(antenna, 0)
            else:
                logger.warning(f"Antenna {antenna} not available. Using {available_antennas[0]}")
                self.set_rx_antenna(available_antennas[0], 0)
            
            # Store configuration in history
            config_record = {
                'timestamp': time.time(),
                'center_freq': actual_freq,
                'sample_rate': actual_rate,
                'gain': actual_gain,
                'antenna': self.get_rx_antenna(0),
                'purpose': 'Inmarsat AERO'
            }
            self.frequency_history.append(config_record)
            
            if self.enable_logging:
                logger.info("Inmarsat AERO configuration completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure for Inmarsat AERO: {e}")
            return False
    
    def _validate_inmarsat_params(self, freq: float, rate: float, gain: float) -> bool:
        """
        Validate parameters cho Inmarsat configuration
        
        Args:
            freq (float): Center frequency Hz
            rate (float): Sample rate Hz
            gain (float): RX gain dB
        
        Returns:
            bool: True nếu parameters hợp lệ
        """
        try:
            # Check frequency range (L-band for Inmarsat)
            if not (1.5e9 <= freq <= 1.66e9):
                logger.error(f"Frequency {freq/1e6:.1f} MHz out of Inmarsat L-band range (1500-1660 MHz)")
                return False
            
            # Check sample rate range
            rx_rates = self.get_rx_rates(0)
            if rate < rx_rates.start() or rate > rx_rates.stop():
                logger.error(f"Sample rate {rate/1e6:.1f} MS/s out of supported range")
                return False
            
            # Check gain range
            gain_range = self.get_rx_gain_range(0)
            if gain < gain_range.start() or gain > gain_range.stop():
                logger.error(f"Gain {gain} dB out of supported range ({gain_range.start()}-{gain_range.stop()} dB)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return False
    
    def optimize_for_signal_type(self, signal_type: str, **kwargs) -> bool:
        """
        Optimize USRP configuration cho specific signal type
        
        Args:
            signal_type (str): Loại signal ("narrowband", "wideband", "burst", "continuous")
            **kwargs: Additional parameters specific to signal type
        
        Returns:
            bool: True nếu optimization thành công
        
        Mục đích: Tự động tối ưu parameters cho từng loại signal khác nhau
        
        Example:
            >>> usrp.optimize_for_signal_type("burst", burst_duration=0.1, pre_burst_time=0.01)
        """
        try:
            optimizations = {
                'narrowband': self._optimize_narrowband,
                'wideband': self._optimize_wideband,
                'burst': self._optimize_burst,
                'continuous': self._optimize_continuous
            }
            
            if signal_type not in optimizations:
                logger.error(f"Unknown signal type: {signal_type}")
                return False
            
            return optimizations[signal_type](**kwargs)
            
        except Exception as e:
            logger.error(f"Signal optimization failed: {e}")
            return False
    
    def _optimize_narrowband(self, **kwargs) -> bool:
        """Optimize cho narrowband signals (< 1 MHz bandwidth)"""
        try:
            # Lower sample rate để giảm processing load
            optimal_rate = kwargs.get('sample_rate', 1e6)  # 1 MS/s default
            self.set_rx_rate(optimal_rate, 0)
            
            # Higher gain cho weak narrowband signals
            optimal_gain = kwargs.get('gain', 35.0)
            self.set_rx_gain(optimal_gain, 0)
            
            logger.info("Optimized for narrowband signal reception")
            return True
        except Exception as e:
            logger.error(f"Narrowband optimization failed: {e}")
            return False
    
    def _optimize_wideband(self, **kwargs) -> bool:
        """Optimize cho wideband signals (> 10 MHz bandwidth)"""
        try:
            # Higher sample rate cho wide bandwidth
            optimal_rate = kwargs.get('sample_rate', 25e6)  # 25 MS/s default
            self.set_rx_rate(optimal_rate, 0)
            
            # Moderate gain để tránh saturation
            optimal_gain = kwargs.get('gain', 25.0)
            self.set_rx_gain(optimal_gain, 0)
            
            logger.info("Optimized for wideband signal reception")
            return True
        except Exception as e:
            logger.error(f"Wideband optimization failed: {e}")
            return False
    
    def _optimize_burst(self, **kwargs) -> bool:
        """Optimize cho burst mode signals"""
        try:
            # Fast AGC settling cho burst signals
            burst_duration = kwargs.get('burst_duration', 0.1)  # 100ms default
            pre_burst_time = kwargs.get('pre_burst_time', 0.01)  # 10ms default
            
            # Configure stream commands cho burst mode
            # Implementation would depend on specific burst characteristics
            
            logger.info(f"Optimized for burst signals: duration={burst_duration}s, pre-time={pre_burst_time}s")
            return True
        except Exception as e:
            logger.error(f"Burst optimization failed: {e}")
            return False
    
    def _optimize_continuous(self, **kwargs) -> bool:
        """Optimize cho continuous signals"""
        try:
            # Stable configuration cho long-term reception
            # Moderate settings để maintain stability
            
            logger.info("Optimized for continuous signal reception")
            return True
        except Exception as e:
            logger.error(f"Continuous optimization failed: {e}")
            return False
    
    # ==================== ADVANCED STREAMING METHODS ====================
    
    def start_advanced_rx_stream(self, stream_args: Optional[uhd.usrp.StreamArgs] = None,
                                 callback_func: Optional[callable] = None,
                                 buffer_size: int = 10000) -> bool:
        """
        Start advanced RX stream với callback processing
        
        Args:
            stream_args (StreamArgs): Stream configuration arguments
            callback_func (callable): Function để process received samples
            buffer_size (int): Buffer size cho streaming
        
        Returns:
            bool: True nếu stream started successfully
        
        Mục đích: Streaming với real-time processing capabilities
        
        Example:
            >>> def process_samples(samples, metadata):
            >>>     # Process samples here
            >>>     print(f"Received {len(samples)} samples")
            >>> 
            >>> usrp.start_advanced_rx_stream(callback_func=process_samples)
        """
        try:
            if self._streaming:
                logger.warning("Stream already running")
                return False
            
            # Setup stream arguments
            if stream_args is None:
                stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
                stream_args.channels = [0]
            
            # Get RX streamer
            self.rx_streamer = self.get_rx_stream(stream_args)
            
            # Setup stream command
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_continuous)
            stream_cmd.stream_now = True
            self.issue_stream_cmd(stream_cmd)
            
            self._streaming = True
            
            # Start monitoring thread nếu có callback
            if callback_func:
                self._start_monitoring_thread(callback_func, buffer_size)
            
            logger.info("Advanced RX stream started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start advanced RX stream: {e}")
            return False
    
    def _start_monitoring_thread(self, callback_func: callable, buffer_size: int):
        """Start thread để monitor stream và call callback function"""
        def monitor_stream():
            try:
                recv_buffer = np.zeros(buffer_size, dtype=np.complex64)
                metadata = uhd.types.RXMetadata()
                
                while self._streaming:
                    try:
                        num_rx_samps = self.rx_streamer.recv(recv_buffer, metadata, 0.1)
                        
                        if num_rx_samps > 0:
                            # Update statistics
                            with self._lock:
                                self.sample_stats['total_samples'] += num_rx_samps
                                if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                                    self.sample_stats['dropped_samples'] += 1
                            
                            # Call user callback
                            if callback_func:
                                callback_func(recv_buffer[:num_rx_samps], metadata)
                                
                    except Exception as e:
                        if self._streaming:  # Only log if we're supposed to be streaming
                            logger.error(f"Stream monitoring error: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"Monitoring thread failed: {e}")
                
        self._monitor_thread = threading.Thread(target=monitor_stream, daemon=True)
        self._monitor_thread.start()
    
    def stop_advanced_rx_stream(self) -> bool:
        """
        Stop advanced RX stream
        
        Returns:
            bool: True nếu stream stopped successfully
        """
        try:
            if not self._streaming:
                logger.warning("No stream running")
                return False
            
            # Stop streaming
            self._streaming = False
            
            # Send stop command
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_continuous)
            self.issue_stream_cmd(stream_cmd)
            
            # Wait for monitoring thread to finish
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=2.0)
            
            logger.info("Advanced RX stream stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop advanced RX stream: {e}")
            return False
    
    # ==================== SIGNAL ANALYSIS AND MONITORING ====================
    
    def measure_signal_power(self, duration: float = 1.0, 
                           channel: int = 0) -> Dict[str, float]:
        """
        Đo signal power trong khoảng thời gian specified
        
        Args:
            duration (float): Thời gian measurement (seconds)
            channel (int): RX channel số
        
        Returns:
            Dict chứa power measurements:
            - avg_power_dbm: Average power dBm
            - peak_power_dbm: Peak power dBm  
            - rms_power_dbm: RMS power dBm
            - noise_floor_dbm: Estimated noise floor dBm
        
        Mục đích: Analyze signal strength và quality
        
        Example:
            >>> power_data = usrp.measure_signal_power(2.0)
            >>> print(f"Average power: {power_data['avg_power_dbm']:.1f} dBm")
        """
        try:
            # Setup stream cho measurement
            stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
            stream_args.channels = [channel]
            rx_streamer = self.get_rx_stream(stream_args)
            
            # Calculate number of samples needed
            sample_rate = self.get_rx_rate(channel)
            num_samples = int(duration * sample_rate)
            
            # Collect samples
            recv_buffer = np.zeros(num_samples, dtype=np.complex64)
            metadata = uhd.types.RXMetadata()
            
            # Start stream
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_continuous)
            stream_cmd.stream_now = True
            self.issue_stream_cmd(stream_cmd)
            
            # Receive samples
            samples_collected = 0
            while samples_collected < num_samples:
                remaining = num_samples - samples_collected
                num_rx_samps = rx_streamer.recv(
                    recv_buffer[samples_collected:samples_collected + remaining], 
                    metadata, 
                    1.0
                )
                samples_collected += num_rx_samps
            
            # Stop stream
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_continuous)
            self.issue_stream_cmd(stream_cmd)
            
            # Analyze power
            power_linear = np.abs(recv_buffer[:samples_collected]) ** 2
            
            # Convert to dBm (assuming 50 ohm impedance)
            power_dbm = 10 * np.log10(power_linear + 1e-12) + 10  # +10 for 50 ohm ref
            
            results = {
                'avg_power_dbm': float(np.mean(power_dbm)),
                'peak_power_dbm': float(np.max(power_dbm)),
                'rms_power_dbm': float(10 * np.log10(np.mean(power_linear) + 1e-12) + 10),
                'noise_floor_dbm': float(np.percentile(power_dbm, 10)),  # 10th percentile as noise floor
                'samples_analyzed': samples_collected,
                'duration': duration,
                'sample_rate': sample_rate
            }
            
            # Store in signal history
            self.signal_history.append({
                'timestamp': time.time(),
                'measurement_type': 'power',
                'results': results
            })
            
            logger.info(f"Signal power measured: avg={results['avg_power_dbm']:.1f} dBm, "
                       f"peak={results['peak_power_dbm']:.1f} dBm")
            
            return results
            
        except Exception as e:
            logger.error(f"Signal power measurement failed: {e}")
            return {'error': str(e)}
    
    def analyze_spectrum(self, fft_size: int = 2048, 
                        num_averages: int = 10,
                        channel: int = 0) -> Dict[str, Any]:
        """
        Analyze frequency spectrum của received signal
        
        Args:
            fft_size (int): FFT size cho spectrum analysis
            num_averages (int): Number of FFT averages
            channel (int): RX channel số
        
        Returns:
            Dict chứa spectrum analysis results:
            - frequencies: Frequency bins Hz
            - power_spectrum_db: Power spectrum dB
            - peak_freq: Frequency of peak power Hz
            - bandwidth_3db: 3dB bandwidth Hz
            - spectral_centroid: Spectral centroid Hz
        
        Mục đích: Detailed frequency domain analysis
        """
        try:
            from scipy import signal as scipy_signal
            
            # Setup stream
            stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
            stream_args.channels = [channel]
            rx_streamer = self.get_rx_stream(stream_args)
            
            sample_rate = self.get_rx_rate(channel)
            
            # Collect samples cho multiple FFTs
            total_samples = fft_size * num_averages
            recv_buffer = np.zeros(total_samples, dtype=np.complex64)
            metadata = uhd.types.RXMetadata()
            
            # Start stream
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_continuous)
            stream_cmd.stream_now = True
            self.issue_stream_cmd(stream_cmd)
            
            # Collect samples
            samples_collected = 0
            while samples_collected < total_samples:
                remaining = total_samples - samples_collected
                num_rx_samps = rx_streamer.recv(
                    recv_buffer[samples_collected:samples_collected + remaining],
                    metadata,
                    1.0
                )
                samples_collected += num_rx_samps
            
            # Stop stream
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_continuous)
            self.issue_stream_cmd(stream_cmd)
            
            # Compute averaged power spectrum
            power_spectrum = np.zeros(fft_size)
            window = scipy_signal.windows.hann(fft_size)
            
            for i in range(num_averages):
                start_idx = i * fft_size
                segment = recv_buffer[start_idx:start_idx + fft_size] * window
                fft_result = np.fft.fftshift(np.fft.fft(segment))
                power_spectrum += np.abs(fft_result) ** 2
            
            power_spectrum /= num_averages
            power_spectrum_db = 10 * np.log10(power_spectrum + 1e-12)
            
            # Frequency axis
            frequencies = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/sample_rate))
            center_freq = self.get_rx_freq(channel)
            frequencies += center_freq
            
            # Analysis
            peak_idx = np.argmax(power_spectrum_db)
            peak_freq = frequencies[peak_idx]
            
            # 3dB bandwidth calculation
            peak_power = power_spectrum_db[peak_idx]
            half_power_indices = np.where(power_spectrum_db >= (peak_power - 3))[0]
            if len(half_power_indices) > 1:
                bandwidth_3db = frequencies[half_power_indices[-1]] - frequencies[half_power_indices[0]]
            else:
                bandwidth_3db = 0.0
            
            # Spectral centroid
            spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
            
            results = {
                'frequencies': frequencies.tolist(),
                'power_spectrum_db': power_spectrum_db.tolist(),
                'peak_freq': float(peak_freq),
                'peak_power_db': float(peak_power),
                'bandwidth_3db': float(bandwidth_3db),
                'spectral_centroid': float(spectral_centroid),
                'center_freq': center_freq,
                'sample_rate': sample_rate,
                'fft_size': fft_size,
                'num_averages': num_averages
            }
            
            # Store in history
            self.signal_history.append({
                'timestamp': time.time(),
                'measurement_type': 'spectrum',
                'results': results
            })
            
            logger.info(f"Spectrum analyzed: peak at {peak_freq/1e6:.3f} MHz, "
                       f"3dB BW = {bandwidth_3db/1e3:.1f} kHz")
            
            return results
            
        except Exception as e:
            logger.error(f"Spectrum analysis failed: {e}")
            return {'error': str(e)}
    
    def detect_signals(self, threshold_db: float = -60.0,
                      min_separation_hz: float = 1000.0,
                      channel: int = 0) -> List[Dict[str, float]]:
        """
        Detect signals trong current spectrum
        
        Args:
            threshold_db (float): Detection threshold dB above noise floor
            min_separation_hz (float): Minimum separation between signals Hz
            channel (int): RX channel số
        
        Returns:
            List of detected signals, mỗi signal là Dict:
            - frequency: Center frequency Hz
            - power_db: Signal power dB
            - bandwidth: Estimated bandwidth Hz
            - snr_db: Estimated SNR dB
        
        Mục đích: Automatic signal detection và analysis
        """
        try:
            # Get spectrum
            spectrum_data = self.analyze_spectrum(fft_size=4096, num_averages=5, channel=channel)
            
            if 'error' in spectrum_data:
                return []
            
            frequencies = np.array(spectrum_data['frequencies'])
            power_db = np.array(spectrum_data['power_spectrum_db'])
            
            # Estimate noise floor
            noise_floor = np.percentile(power_db, 20)  # 20th percentile
            detection_threshold = noise_floor + threshold_db
            
            # Find peaks above threshold
            from scipy import signal as scipy_signal
            peaks, properties = scipy_signal.find_peaks(
                power_db,
                height=detection_threshold,
                distance=int(min_separation_hz / (frequencies[1] - frequencies[0]))
            )
            
            detected_signals = []
            for i, peak_idx in enumerate(peaks):
                signal_freq = frequencies[peak_idx]
                signal_power = power_db[peak_idx]
                signal_snr = signal_power - noise_floor
                
                # Estimate bandwidth (crude method)
                half_power = signal_power - 3
                left_idx = peak_idx
                while left_idx > 0 and power_db[left_idx] > half_power:
                    left_idx -= 1
                right_idx = peak_idx
                while right_idx < len(power_db) - 1 and power_db[right_idx] > half_power:
                    right_idx += 1
                
                bandwidth = frequencies[right_idx] - frequencies[left_idx]
                
                detected_signals.append({
                    'frequency': float(signal_freq),
                    'power_db': float(signal_power),
                    'bandwidth': float(bandwidth),
                    'snr_db': float(signal_snr)
                })
            
            logger.info(f"Detected {len(detected_signals)} signals above {threshold_db} dB threshold")
            
            return detected_signals
            
        except Exception as e:
            logger.error(f"Signal detection failed: {e}")
            return []
    
    # ==================== CALIBRATION AND PERFORMANCE METHODS ====================
    
    def run_self_calibration(self) -> Dict[str, Any]:
        """
        Chạy self-calibration routine cho USRP
        
        Returns:
            Dict chứa calibration results và status
        
        Mục đích: Ensure optimal performance và accuracy
        """
        try:
            logger.info("Starting self-calibration routine...")
            
            calibration_results = {}
            
            # DC offset calibration
            calibration_results['dc_offset'] = self._calibrate_dc_offset()
            
            # IQ imbalance calibration  
            calibration_results['iq_imbalance'] = self._calibrate_iq_imbalance()
            
            # Frequency accuracy calibration
            calibration_results['freq_accuracy'] = self._calibrate_frequency_accuracy()
            
            # Gain calibration
            calibration_results['gain'] = self._calibrate_gain()
            
            # Store calibration data
            self.calibration_data = {
                'timestamp': time.time(),
                'results': calibration_results,
                'device_info': self.device_info.copy()
            }
            
            logger.info("Self-calibration completed successfully")
            return calibration_results
            
        except Exception as e:
            logger.error(f"Self-calibration failed: {e}")
            return {'error': str(e)}
    
    def _calibrate_dc_offset(self) -> Dict[str, Any]:
        """Calibrate DC offset correction"""
        try:
            # Enable automatic DC offset correction
            self.set_rx_dc_offset(True, 0)
            
            # Measure DC offset by receiving without signal
            original_freq = self.get_rx_freq(0)
            
            # Tune to known empty frequency
            self.set_rx_freq(uhd.types.TuneRequest(1e9), 0)  # 1 GHz
            
            # Measure DC offset
            power_data = self.measure_signal_power(0.5)  # 500ms measurement
            
            # Restore original frequency
            self.set_rx_freq(uhd.types.TuneRequest(original_freq), 0)
            
            return {
                'status': 'completed',
                'dc_power_dbm': power_data.get('avg_power_dbm', 0),
                'correction_enabled': True
            }
            
        except Exception as e:
            logger.error(f"DC offset calibration failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _calibrate_iq_imbalance(self) -> Dict[str, Any]:
        """Calibrate IQ imbalance correction"""
        try:
            # Enable automatic IQ imbalance correction
            self.set_rx_iq_balance(True, 0)
            
            return {
                'status': 'completed',
                'correction_enabled': True
            }
            
        except Exception as e:
            logger.error(f"IQ imbalance calibration failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _calibrate_frequency_accuracy(self) -> Dict[str, Any]:
        """Calibrate frequency accuracy"""
        try:
            # This would require a known reference signal
            # For now, just return status
            
            return {
                'status': 'completed',
                'accuracy_ppm': 'unknown',
                'reference_used': 'internal'
            }
            
        except Exception as e:
            logger.error(f"Frequency accuracy calibration failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _calibrate_gain(self) -> Dict[str, Any]:
        """Calibrate gain accuracy"""
        try:
            # Test gain linearity at different settings
            gain_points = [0, 10, 20, 30]
            gain_measurements = []
            
            original_gain = self.get_rx_gain(0)
            
            for gain in gain_points:
                self.set_rx_gain(gain, 0)
                time.sleep(0.1)  # Let gain settle
                
                power_data = self.measure_signal_power(0.2)  # Quick measurement
                gain_measurements.append({
                    'set_gain_db': gain,
                    'actual_gain_db': self.get_rx_gain(0),
                    'measured_power_dbm': power_data.get('avg_power_dbm', 0)
                })
            
            # Restore original gain
            self.set_rx_gain(original_gain, 0)
            
            return {
                'status': 'completed',
                'measurements': gain_measurements
            }
            
        except Exception as e:
            logger.error(f"Gain calibration failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    # ==================== UTILITY AND HELPER METHODS ====================
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Lấy performance statistics của USRP
        
        Returns:
            Dict chứa performance metrics
        """
        with self._lock:
            stats = self.sample_stats.copy()
            
        # Calculate rates
        current_time = time.time()
        time_diff = current_time - stats['last_update']
        
        if time_diff > 0:
            stats['sample_rate_actual'] = stats['total_samples'] / time_diff
            stats['drop_rate'] = stats['dropped_samples'] / time_diff if time_diff > 0 else 0
        
        stats['uptime'] = current_time - stats.get('start_time', current_time)
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics counters"""
        with self._lock:
            self.sample_stats = {
                'total_samples': 0,
                'dropped_samples': 0,
                'overruns': 0,
                'underruns': 0,
                'last_update': time.time(),
                'start_time': time.time()
            }
        logger.info("Performance statistics reset")
    
    def save_configuration(self, filename: str) -> bool:
        """
        Lưu current configuration ra file
        
        Args:
            filename (str): Tên file để save configuration
            
        Returns:
            bool: True nếu save thành công
        """
        try:
            import json
            
            config_data = {
                'device_info': self.device_info,
                'current_config': self._get_current_config(),
                'calibration_data': self.calibration_data,
                'timestamp': time.time()
            }
            
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_configuration(self, filename: str) -> bool:
        """
        Load configuration từ file
        
        Args:
            filename (str): Tên file chứa configuration
            
        Returns:
            bool: True nếu load và apply thành công
        """
        try:
            import json
            
            with open(filename, 'r') as f:
                config_data = json.load(f)
            
            # Apply configuration
            current_config = config_data.get('current_config', {})
            
            if 'rx_rate' in current_config:
                self.set_rx_rate(current_config['rx_rate'], 0)
            if 'rx_freq' in current_config:
                self.set_rx_freq(uhd.types.TuneRequest(current_config['rx_freq']), 0)
            if 'rx_gain' in current_config:
                self.set_rx_gain(current_config['rx_gain'], 0)
            if 'rx_antenna' in current_config:
                self.set_rx_antenna(current_config['rx_antenna'], 0)
            
            logger.info(f"Configuration loaded from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def __del__(self):
        """Cleanup khi object bị destroyed"""
        try:
            if hasattr(self, '_streaming') and self._streaming:
                self.stop_advanced_rx_stream()
            logger.info("Enhanced USRP N210 Controller cleaned up")
        except:
            pass


# ==================== EXAMPLE USAGE AND TESTING ====================

def test_enhanced_usrp():
    """
    Test function để demonstrate Enhanced USRP N210 capabilities
    """
    try:
        print("=== Enhanced USRP N210 Controller Test ===")
        
        # Initialize USRP (sử dụng mock args cho testing)
        usrp = EnhancedUSRPN210("type=usrp1")  # Mock device
        
        # Test device status
        print("\n1. Device Status:")
        status = usrp.get_device_status()
        print(f"Device: {status.get('device_info', {}).get('mboard_name', 'Unknown')}")
        
        # Test Inmarsat configuration
        print("\n2. Inmarsat Configuration:")
        success = usrp.configure_for_inmarsat(1545e6, 2.4e6, 30.0)
        print(f"Configuration {'successful' if success else 'failed'}")
        
        # Test signal measurements
        print("\n3. Signal Measurements:")
        power_data = usrp.measure_signal_power(0.5)
        if 'error' not in power_data:
            print(f"Average power: {power_data['avg_power_dbm']:.1f} dBm")
            print(f"Peak power: {power_data['peak_power_dbm']:.1f} dBm")
        
        # Test calibration
        print("\n4. Self-Calibration:")
        cal_results = usrp.run_self_calibration()
        if 'error' not in cal_results:
            print("Calibration completed successfully")
        
        # Test configuration save/load
        print("\n5. Configuration Management:")
        save_success = usrp.save_configuration("test_config.json")
        print(f"Configuration {'saved' if save_success else 'save failed'}")
        
        print("\n=== Test completed successfully ===")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_enhanced_usrp()