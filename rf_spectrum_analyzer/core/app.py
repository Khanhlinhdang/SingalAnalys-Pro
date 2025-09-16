"""
Main Application Class for RF Spectrum Analyzer
Coordinates GUI, SDR backend, and signal processing components.
"""

import sys
import logging
import numpy as np
from typing import Optional, Dict, Any
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtCore import QTimer, QThread, Signal, QObject
from PySide6.QtGui import QCloseEvent

from rf_spectrum_analyzer.gui.main_window import MainWindow
from rf_spectrum_analyzer.core.sdr_backend import SDRBackendManager
from rf_spectrum_analyzer.core.signal_processor import SignalProcessor
from rf_spectrum_analyzer.config.settings import Settings
from rf_spectrum_analyzer.utils.logger import get_logger

logger = get_logger(__name__)


class DataAcquisitionThread(QThread):
    """Background thread for continuous data acquisition from SDR."""
    
    data_ready = Signal(np.ndarray)  # Emitted when new IQ data is available
    error_occurred = Signal(str)     # Emitted when an error occurs
    
    def __init__(self, sdr_manager: SDRBackendManager, settings: Settings):
        super().__init__()
        self.sdr_manager = sdr_manager
        self.settings = settings
        self.running = False
        self.buffer_size = settings.dsp.fft_size * 2  # Double buffer for overlap
    
    def run(self):
        """Main acquisition loop."""
        try:
            logger.info("Starting data acquisition thread")
            self.running = True
            
            while self.running:
                if self.sdr_manager.is_connected():
                    # Read IQ samples from SDR
                    samples = self.sdr_manager.read_samples(self.buffer_size)
                    if samples is not None and len(samples) > 0:
                        self.data_ready.emit(samples)
                    else:
                        self.msleep(10)  # Brief pause if no data available
                else:
                    self.msleep(100)  # Wait longer if not connected
                    
        except Exception as e:
            logger.error(f"Data acquisition error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            logger.info("Data acquisition thread stopped")
    
    def stop(self):
        """Stop the acquisition thread."""
        self.running = False
        self.wait()


class RFSpectrumAnalyzerApp(QObject):
    """Main application class that coordinates all components."""
    
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # Core components
        self.sdr_manager = None
        self.signal_processor = None
        self.main_window = None
        self.acquisition_thread = None
        
        # Data buffers
        self.iq_buffer = np.array([], dtype=np.complex64)
        self.spectrum_data = np.array([])
        self.waterfall_data = []
        self.constellation_data = {
            'iq_samples': np.array([], dtype=np.complex64),
            'symbols': np.array([], dtype=np.complex64),
            'modulation_info': {}
        }
        self.bitstream_data = np.array([], dtype=np.uint8)
        
        # Demo mode
        self.demo_mode = getattr(settings, 'demo_mode', False)
        self.demo_timer = QTimer() if self.demo_mode else None
        self.demo_counter = 0
        
        # Timers for GUI updates
        self.spectrum_timer = QTimer()
        self.waterfall_timer = QTimer()
        self.constellation_timer = QTimer()
        self.bitstream_timer = QTimer()
        
        # Performance monitoring
        self.frame_count = 0
        self.last_fps_update = 0
        
        self.initialize_application()
    
    def initialize_application(self):
        """Initialize all application components."""
        try:
            self.logger.info("Initializing RF Spectrum Analyzer...")
            
            # Initialize signal processor
            self.signal_processor = SignalProcessor(self.settings)
            
            # Initialize SDR backend manager
            self.sdr_manager = SDRBackendManager(self.settings)
            
            # Initialize main window
            self.main_window = MainWindow(self.settings, self)
            self.main_window.show()
            
            # Setup window properties
            self.setup_window()
            
            # Connect signals
            self.connect_signals()
            
            # Setup timers
            self.setup_timers()
            
            # Initialize acquisition thread
            self.acquisition_thread = DataAcquisitionThread(
                self.sdr_manager, self.settings
            )
            self.acquisition_thread.data_ready.connect(self.on_new_data)
            self.acquisition_thread.error_occurred.connect(self.on_acquisition_error)
            
            # Setup demo mode if enabled
            if self.demo_mode:
                self.setup_demo_mode()
            
            self.logger.info("Application initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            raise
    
    def setup_window(self):
        """Setup main window properties."""
        if self.main_window:
            self.main_window.setWindowTitle("RF Spectrum Analyzer")
            self.main_window.resize(
                self.settings.gui.window_width,
                self.settings.gui.window_height
            )
            self.main_window.move(
                self.settings.gui.window_x,
                self.settings.gui.window_y
            )
    
    def connect_signals(self):
        """Connect signals between components."""
        if self.main_window:
            # Connect control signals
            self.main_window.start_requested.connect(self.start_acquisition)
            self.main_window.stop_requested.connect(self.stop_acquisition)
            self.main_window.device_changed.connect(self.change_device)
            self.main_window.frequency_changed.connect(self.change_frequency)
            self.main_window.sample_rate_changed.connect(self.change_sample_rate)
            self.main_window.gain_changed.connect(self.change_gain)
            
            # Connect processing signals
            self.main_window.fft_size_changed.connect(self.change_fft_size)
            self.main_window.window_changed.connect(self.change_window_function)
            self.main_window.averaging_changed.connect(self.change_averaging)
    
    def setup_timers(self):
        """Setup GUI update timers."""
        # Spectrum update timer
        spectrum_interval = int(1000 / self.settings.gui.spectrum_update_rate)
        self.spectrum_timer.timeout.connect(self.update_spectrum_display)
        self.spectrum_timer.start(spectrum_interval)
        
        # Waterfall update timer
        waterfall_interval = int(1000 / self.settings.gui.waterfall_update_rate)
        self.waterfall_timer.timeout.connect(self.update_waterfall_display)
        self.waterfall_timer.start(waterfall_interval)
        
        # Constellation update timer
        self.constellation_timer.timeout.connect(self.update_constellation_display)
        self.constellation_timer.start(75)  # ~13 FPS
        
        # Bitstream update timer
        self.bitstream_timer.timeout.connect(self.update_bitstream_display)
        self.bitstream_timer.start(100)  # 10 FPS for bitstream
    
    def start_acquisition(self):
        """Start SDR data acquisition."""
        try:
            self.logger.info("Starting SDR acquisition...")
            
            # Connect to SDR device
            if not self.sdr_manager.connect():
                self.logger.error("Failed to connect to SDR device")
                return False
            
            # Configure device
            device_settings = self.settings.get_device_settings()
            if not self.sdr_manager.configure(device_settings):
                self.logger.error("Failed to configure SDR device")
                return False
            
            # Start acquisition thread
            self.acquisition_thread.start()
            
            # Update UI state
            if self.main_window:
                self.main_window.set_acquisition_state(True)
            
            self.logger.info("SDR acquisition started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start acquisition: {e}")
            return False
    
    def stop_acquisition(self):
        """Stop SDR data acquisition."""
        try:
            self.logger.info("Stopping SDR acquisition...")
            
            # Stop acquisition thread
            if self.acquisition_thread and self.acquisition_thread.isRunning():
                self.acquisition_thread.stop()
            
            # Disconnect SDR
            if self.sdr_manager:
                self.sdr_manager.disconnect()
            
            # Update UI state
            if self.main_window:
                self.main_window.set_acquisition_state(False)
            
            self.logger.info("SDR acquisition stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping acquisition: {e}")
    
    def on_new_data(self, samples: np.ndarray):
        """Handle new IQ data from acquisition thread."""
        try:
            # Append new samples to buffer
            self.iq_buffer = np.concatenate([self.iq_buffer, samples])
            
            # Process data when we have enough samples
            min_samples = self.settings.dsp.fft_size
            if len(self.iq_buffer) >= min_samples:
                self.process_iq_data()
                
        except Exception as e:
            self.logger.error(f"Error processing new data: {e}")
    
    def process_iq_data(self):
        """Process IQ data to generate spectrum and constellation analysis."""
        try:
            if self.signal_processor is None:
                return
            
            # Extract samples for processing
            fft_size = self.settings.dsp.fft_size
            samples = self.iq_buffer[:fft_size]
            
            self.logger.debug(f"Processing IQ data: {len(samples)} samples")
            
            # Compute spectrum
            spectrum = self.signal_processor.compute_spectrum(samples)
            if spectrum is not None:
                self.spectrum_data = spectrum
                self.logger.debug(f"Spectrum computed: {len(spectrum)} points")
            
            # Update waterfall data
            if len(spectrum) > 0:
                self.waterfall_data.append(spectrum.copy())
                max_waterfall_lines = self.settings.gui.waterfall_height
                if len(self.waterfall_data) > max_waterfall_lines:
                    self.waterfall_data.pop(0)
            
            # Process constellation data (run every few frames for performance)
            self.logger.debug(f"Frame count: {self.frame_count}, modulo: {self.frame_count % 3}")
            if self.frame_count % 1 == 0:  # Every frame for testing
                self.logger.debug("Triggering advanced analysis...")
                try:
                    self.process_advanced_analysis(samples)
                    self.logger.debug("Advanced analysis completed successfully")
                except Exception as e:
                    self.logger.error(f"Error in advanced analysis: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Remove processed samples from buffer
            overlap_samples = int(fft_size * self.settings.dsp.overlap)
            self.iq_buffer = self.iq_buffer[fft_size - overlap_samples:]
            
            self.frame_count += 1
            
        except Exception as e:
            self.logger.error(f"Error in signal processing: {e}")
    
    def process_advanced_analysis(self, iq_samples: np.ndarray):
        """Process IQ data for both constellation and bitstream display."""
        try:
            self.logger.debug("Advanced analysis called")
            
            # Use the complete processing chain from signal processor
            if self.signal_processor is None:
                self.logger.warning("Signal processor not available, skipping advanced analysis")
                return
            
            # Process complete signal chain
            result = self.signal_processor.process_complete_chain(iq_samples)
            
            if not result.get('success', False):
                self.logger.debug(f"Processing chain failed: {result.get('error', 'Unknown error')}")
                # Fallback to demo data for development
                result = self._generate_demo_analysis_result()
            
            self.logger.debug(f"Processing chain result keys: {list(result.keys())}")
            self.logger.debug(f"Processing success: {result.get('success', False)}")
            
            # Update constellation data
            self.update_constellation_data(iq_samples, result)
            
            # Update bitstream data
            self.update_bitstream_data(result)
            
            # Update GUI widgets if available
            if self.main_window:
                self.update_gui_widgets()
                        
        except Exception as e:
            self.logger.debug(f"Error in advanced analysis: {e}")
    
    def _generate_demo_analysis_result(self) -> dict:
        """Generate demo analysis result for fallback."""
        # Create a simplified result for demo mode
        result = {
            'success': True,
            'modulation_analysis': {'type': 'QPSK', 'confidence': 0.8},
            'demodulation': {
                'success': True,
                'data_type': 'digital',
                'symbols': np.array([0, 1, 2, 3] * 8, dtype=int)
            },
            'encoding_analysis': {'type': 'None'},
            'final_data': np.array([])
        }
        
        # Generate bitstream data with realistic pattern
        demo_bits = []
        frame_offset = self.frame_count % 32
        
        # Create pseudo-random but deterministic pattern
        for i in range(32):
            bit_val = (frame_offset + i) % 4
            if bit_val == 0:
                demo_bits.append(0)
            elif bit_val == 1:
                demo_bits.append(1) 
            elif bit_val == 2:
                demo_bits.append(1)
            else:
                demo_bits.append(0)
        
        result['final_data'] = np.array(demo_bits, dtype=np.uint8)
        self.logger.debug(f"Generated demo analysis result with {len(demo_bits)} bits")
        
        return result
    
    def update_constellation_data(self, iq_samples: np.ndarray, processing_result: dict):
        """Update constellation data from processing results."""
        try:
            # Always store raw IQ samples for constellation
            self.constellation_data['iq_samples'] = iq_samples.copy()
            
            # Extract modulation analysis results
            mod_analysis = processing_result.get('modulation_analysis', {})
            self.constellation_data['modulation_info'] = {
                'type': mod_analysis.get('type', 'Unknown'),
                'confidence': mod_analysis.get('confidence', 0.0),
                'snr_estimate': mod_analysis.get('snr_estimate', 0.0)
            }
            
            # Extract demodulation results for symbols
            demod_result = processing_result.get('demodulation', {})
            if demod_result.get('success', False):
                # Extract symbol constellation if available
                if 'constellation_points' in demod_result:
                    self.constellation_data['symbols'] = demod_result['constellation_points']
                elif 'symbols' in demod_result:
                    self.constellation_data['symbols'] = demod_result['symbols']
                
                # Add EVM information if available
                if 'evm' in demod_result:
                    self.constellation_data['modulation_info']['evm'] = demod_result['evm']
                
                # Add SNR information if available
                if 'snr_db' in demod_result:
                    self.constellation_data['modulation_info']['snr_db'] = demod_result['snr_db']
                
                # Generate reference constellation
                mod_type = mod_analysis.get('type', 'Unknown')
                if mod_type not in ['Unknown', 'Analog']:
                    ref_constellation = self.generate_reference_constellation(mod_type)
                    if len(ref_constellation) > 0:
                        self.constellation_data['modulation_info']['reference_constellation'] = ref_constellation
            
        except Exception as e:
            self.logger.debug(f"Error updating constellation data: {e}")
    
    def update_bitstream_data(self, processing_result: dict):
        """Update bitstream data from processing results."""
        try:
            self.logger.debug(f"update_bitstream_data called with success: {processing_result.get('success', False)}")
            
            if not processing_result.get('success', False):
                self.logger.debug("Processing result not successful, skipping bitstream update")
                return
                
            # Extract final data from complete processing chain
            final_data = processing_result.get('final_data', np.array([]))
            self.logger.debug(f"Final data length: {len(final_data)}")
            
            # Check if we have digital data from demodulation
            demod_result = processing_result.get('demodulation', {})
            data_type = demod_result.get('data_type', 'unknown')
            self.logger.debug(f"Demodulation data type: {data_type}")
            
            if len(final_data) > 0:
                self.logger.debug(f"Processing final data: {final_data[:10] if len(final_data) > 10 else final_data}...")
                
                # Convert to binary if needed
                if final_data.dtype == bool:
                    binary_data = final_data.astype(np.uint8)
                elif final_data.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
                    binary_data = np.clip(final_data, 0, 1).astype(np.uint8)
                elif final_data.dtype in [np.float32, np.float64, np.complex64, np.complex128]:
                    # For float/complex data - threshold to binary
                    if np.iscomplexobj(final_data):
                        # Use magnitude for complex data
                        magnitude_data = np.abs(final_data)
                        threshold = np.mean(magnitude_data) if len(magnitude_data) > 0 else 0.5
                        binary_data = (magnitude_data > threshold).astype(np.uint8)
                    else:
                        threshold = np.mean(final_data) if len(final_data) > 0 else 0.5
                        binary_data = (final_data > threshold).astype(np.uint8)
                else:
                    # Fallback: convert to float then threshold
                    try:
                        float_data = np.real(final_data).astype(float)
                        threshold = np.mean(float_data) if len(float_data) > 0 else 0.5
                        binary_data = (float_data > threshold).astype(np.uint8)
                    except:
                        self.logger.warning(f"Could not convert data type {final_data.dtype} to binary")
                        return
                
                # Add to bitstream buffer
                self.bitstream_data = np.concatenate([self.bitstream_data, binary_data])
                
                self.logger.debug(f"Added {len(binary_data)} bits to bitstream, total: {len(self.bitstream_data)}")
                
                # Limit buffer size
                max_bits = 10000  # Keep last 10k bits
                if len(self.bitstream_data) > max_bits:
                    self.bitstream_data = self.bitstream_data[-max_bits:]
            else:
                self.logger.debug(f"No final data to process, length: {len(final_data)}")
                    
        except Exception as e:
            self.logger.debug(f"Error updating bitstream data: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
    
    def update_gui_widgets(self):
        """Update GUI widgets with latest data."""
        try:
            if not self.main_window:
                return
            
            # Update constellation widget
            if hasattr(self.main_window, 'constellation_widget') and self.main_window.constellation_widget:
                constellation_widget = self.main_window.constellation_widget
                if hasattr(constellation_widget, 'update_constellation'):
                    iq_data = self.constellation_data.get('iq_samples', np.array([]))
                    symbols = self.constellation_data.get('symbols', None)
                    mod_info = self.constellation_data.get('modulation_info', {})
                    
                    if len(iq_data) > 0:
                        constellation_widget.update_constellation(iq_data, symbols, mod_info)
                        self.logger.debug(f"Updated constellation with {len(iq_data)} IQ samples")
            
            # Update bitstream widget
            if hasattr(self.main_window, 'bitstream_widget') and self.main_window.bitstream_widget:
                bitstream_widget = self.main_window.bitstream_widget
                if hasattr(bitstream_widget, 'add_bits') and len(self.bitstream_data) > 0:
                    # Send new bits (limit to avoid flooding)
                    new_bits = self.bitstream_data[-100:] if len(self.bitstream_data) > 100 else self.bitstream_data
                    bitstream_widget.add_bits(new_bits)
                    self.logger.debug(f"Updated bitstream with {len(new_bits)} bits")
                    
        except Exception as e:
            self.logger.debug(f"Error updating GUI widgets: {e}")
    
    def generate_reference_constellation(self, modulation_type: str) -> np.ndarray:
        """Generate reference constellation points."""
        try:
            mod_type = modulation_type.upper()
            
            if mod_type == "BPSK":
                return np.array([-1, 1], dtype=complex)
            elif mod_type == "QPSK":
                return np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=complex) / np.sqrt(2)
            elif mod_type == "8PSK":
                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                return np.exp(1j * angles)
            elif mod_type == "16QAM":
                points = []
                for i in [-3, -1, 1, 3]:
                    for q in [-3, -1, 1, 3]:
                        points.append(i + 1j*q)
                return np.array(points) / np.sqrt(10)
            else:
                return np.array([], dtype=complex)
                
        except Exception as e:
            self.logger.debug(f"Error generating reference constellation: {e}")
            return np.array([], dtype=complex)
    
    def update_spectrum_display(self):
        """Update spectrum display in GUI."""
        if self.main_window and len(self.spectrum_data) > 0:
            self.main_window.update_spectrum(self.spectrum_data)
    
    def update_waterfall_display(self):
        """Update waterfall display in GUI."""
        if self.main_window and len(self.waterfall_data) > 0:
            waterfall_array = np.array(self.waterfall_data)
            self.main_window.update_waterfall(waterfall_array)
    
    def update_constellation_display(self):
        """Update constellation display in GUI."""
        if (self.main_window and 
            len(self.constellation_data['iq_samples']) > 0):
            
            self.main_window.update_constellation(
                self.constellation_data['iq_samples'],
                self.constellation_data.get('symbols'),
                self.constellation_data['modulation_info']
            )
    
    def update_bitstream_display(self):
        """Update bitstream display in GUI."""
        if self.main_window and len(self.bitstream_data) > 0:
            # Send recent bits to the display
            recent_bits = self.bitstream_data[-200:] if len(self.bitstream_data) > 200 else self.bitstream_data
            self.main_window.update_bitstream(recent_bits)
    
    def on_acquisition_error(self, error_message: str):
        """Handle acquisition errors."""
        self.logger.error(f"Acquisition error: {error_message}")
        if self.main_window:
            self.main_window.show_error_message(f"SDR Error: {error_message}")
        self.stop_acquisition()
    
    # Device control methods
    def change_device(self, device_type: str):
        """Change SDR device type."""
        try:
            self.logger.info(f"Changing device to: {device_type}")
            self.settings.sdr.device_type = device_type
            
            # Restart acquisition if currently running
            was_running = (self.acquisition_thread and 
                          self.acquisition_thread.isRunning())
            if was_running:
                self.stop_acquisition()
            
            # Reinitialize SDR manager with new device
            self.sdr_manager = SDRBackendManager(self.settings)
            
            if was_running:
                self.start_acquisition()
                
        except Exception as e:
            self.logger.error(f"Error changing device: {e}")
    
    def change_frequency(self, frequency: float):
        """Change center frequency."""
        try:
            self.logger.info(f"Changing frequency to: {frequency} Hz")
            self.settings.sdr.center_frequency = frequency
            if self.sdr_manager and self.sdr_manager.is_connected():
                self.sdr_manager.set_center_frequency(frequency)
        except Exception as e:
            self.logger.error(f"Error changing frequency: {e}")
    
    def change_sample_rate(self, sample_rate: float):
        """Change sample rate."""
        try:
            self.logger.info(f"Changing sample rate to: {sample_rate} Hz")
            self.settings.sdr.sample_rate = sample_rate
            if self.sdr_manager and self.sdr_manager.is_connected():
                self.sdr_manager.set_sample_rate(sample_rate)
            
            # Update signal processor sample rate
            if self.signal_processor:
                self.signal_processor.update_sample_rate(sample_rate)
                
        except Exception as e:
            self.logger.error(f"Error changing sample rate: {e}")
    
    def change_gain(self, gain: float):
        """Change RF gain."""
        try:
            self.logger.info(f"Changing gain to: {gain} dB")
            self.settings.sdr.gain = gain
            if self.sdr_manager and self.sdr_manager.is_connected():
                self.sdr_manager.set_gain(gain)
        except Exception as e:
            self.logger.error(f"Error changing gain: {e}")
    
    def change_fft_size(self, fft_size: int):
        """Change FFT size."""
        try:
            self.logger.info(f"Changing FFT size to: {fft_size}")
            self.settings.dsp.fft_size = fft_size
        except Exception as e:
            self.logger.error(f"Error changing FFT size: {e}")
    
    def change_window_function(self, window: str):
        """Change window function."""
        try:
            self.logger.info(f"Changing window function to: {window}")
            self.settings.dsp.window_function = window
        except Exception as e:
            self.logger.error(f"Error changing window function: {e}")
    
    def change_averaging(self, averaging: int):
        """Change spectral averaging."""
        try:
            self.logger.info(f"Changing averaging to: {averaging}")
            self.settings.dsp.averaging = averaging
        except Exception as e:
            self.logger.error(f"Error changing averaging: {e}")
    
    def shutdown_application(self):
        """Shutdown application and save settings."""
        try:
            self.logger.info("Shutting down application...")
            
            # Stop SDR acquisition
            self.stop_acquisition()
            
            # Save settings
            try:
                if self.main_window:
                    # Save window geometry
                    self.settings.gui.window_width = self.main_window.width()
                    self.settings.gui.window_height = self.main_window.height()
                    self.settings.gui.window_x = self.main_window.x()
                    self.settings.gui.window_y = self.main_window.y()
                
                # Save settings to file
                self.settings.save()
                self.logger.info("Settings saved")
                
            except Exception as e:
                self.logger.error(f"Error saving settings: {e}")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def setup_demo_mode(self):
        """Setup demo mode with simulated data generation."""
        self.logger.info("Setting up demo mode with simulated data")
        
        if self.demo_timer:
            self.demo_timer.timeout.connect(self.generate_demo_data)
            self.demo_timer.start(100)  # Generate data every 100ms
    
    def generate_demo_data(self):
        """Generate simulated IQ data for demonstration."""
        try:
            self.logger.debug(f"Generating demo data, counter: {self.demo_counter}")
            
            # Parameters
            sample_rate = self.settings.sdr.sample_rate
            num_samples = self.settings.dsp.fft_size
            
            # Generate different signal types based on demo counter
            signal_type = (self.demo_counter // 50) % 4
            
            if signal_type == 0:
                # QPSK signal
                symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_samples//4)
                # Upsample and add pulse shaping (simplified)
                signal = np.repeat(symbols, 4)
                signal = signal[:num_samples]
                
            elif signal_type == 1:
                # FSK signal (simplified)
                freq1, freq2 = 1000, 3000  # Hz
                bits = np.random.randint(0, 2, num_samples//100)
                signal = np.array([], dtype=complex)
                for bit in bits:
                    freq = freq1 if bit == 0 else freq2
                    t = np.linspace(0, 0.01, 100)  # 10ms per bit
                    sig_part = np.exp(2j * np.pi * freq * t)
                    signal = np.concatenate([signal, sig_part])
                signal = signal[:num_samples]
                
            elif signal_type == 2:
                # BPSK signal
                bits = np.random.randint(0, 2, num_samples//10)
                symbols = 2*bits - 1  # Convert to ±1
                signal = np.repeat(symbols, 10).astype(complex)
                signal = signal[:num_samples]
                
            else:
                # Noise + weak signal
                noise = (np.random.randn(num_samples) + 1j*np.random.randn(num_samples)) * 0.5
                carrier = 0.1 * np.exp(2j * np.pi * 1000 * np.arange(num_samples) / sample_rate)
                signal = noise + carrier
            
            # Add some noise
            noise_power = 0.1
            noise = (np.random.randn(num_samples) + 1j*np.random.randn(num_samples)) * noise_power
            signal += noise
            
            # Normalize
            signal = signal / np.max(np.abs(signal))
            
            # Send to processing pipeline
            self.on_new_data(signal.astype(np.complex64))
            self.demo_counter += 1
            
        except Exception as e:
            self.logger.debug(f"Error generating demo data: {e}")