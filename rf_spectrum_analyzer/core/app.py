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


class RFSpectrumAnalyzerApp(QMainWindow):
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
        
        # Timers for GUI updates
        self.spectrum_timer = QTimer()
        self.waterfall_timer = QTimer()
        
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
            self.setCentralWidget(self.main_window)
            
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
            
            self.logger.info("Application initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            raise
    
    def setup_window(self):
        """Setup main window properties."""
        self.setWindowTitle("RF Spectrum Analyzer")
        self.resize(
            self.settings.gui.window_width,
            self.settings.gui.window_height
        )
        self.move(
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
            if not self.acquisition_thread.isRunning():
                self.acquisition_thread.start()
            
            # Update GUI state
            if self.main_window:
                self.main_window.set_acquisition_state(True)
            
            self.logger.info("SDR acquisition started successfully")
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
            
            # Disconnect from SDR
            if self.sdr_manager:
                self.sdr_manager.disconnect()
            
            # Update GUI state
            if self.main_window:
                self.main_window.set_acquisition_state(False)
            
            self.logger.info("SDR acquisition stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping acquisition: {e}")
    
    def on_new_data(self, iq_samples: np.ndarray):
        """Handle new IQ data from acquisition thread."""
        try:
            # Add new samples to buffer
            self.iq_buffer = np.concatenate([self.iq_buffer, iq_samples])
            
            # Keep buffer size manageable
            max_buffer_size = self.settings.dsp.fft_size * 10
            if len(self.iq_buffer) > max_buffer_size:
                self.iq_buffer = self.iq_buffer[-max_buffer_size:]
            
            # Process if we have enough samples
            if len(self.iq_buffer) >= self.settings.dsp.fft_size:
                self.process_iq_data()
            
        except Exception as e:
            self.logger.error(f"Error processing new data: {e}")
    
    def process_iq_data(self):
        """Process IQ data to generate spectrum and other analysis."""
        try:
            if self.signal_processor is None:
                return
            
            # Extract samples for processing
            fft_size = self.settings.dsp.fft_size
            samples = self.iq_buffer[:fft_size]
            
            # Compute spectrum
            spectrum = self.signal_processor.compute_spectrum(samples)
            if spectrum is not None:
                self.spectrum_data = spectrum
            
            # Update waterfall data
            if len(spectrum) > 0:
                self.waterfall_data.append(spectrum.copy())
                max_waterfall_lines = self.settings.gui.waterfall_height
                if len(self.waterfall_data) > max_waterfall_lines:
                    self.waterfall_data.pop(0)
            
            # Remove processed samples from buffer
            overlap_samples = int(fft_size * self.settings.dsp.overlap)
            self.iq_buffer = self.iq_buffer[fft_size - overlap_samples:]
            
        except Exception as e:
            self.logger.error(f"Error in signal processing: {e}")
    
    def update_spectrum_display(self):
        """Update spectrum display in GUI."""
        if self.main_window and len(self.spectrum_data) > 0:
            self.main_window.update_spectrum(self.spectrum_data)
    
    def update_waterfall_display(self):
        """Update waterfall display in GUI."""
        if self.main_window and len(self.waterfall_data) > 0:
            waterfall_array = np.array(self.waterfall_data)
            self.main_window.update_waterfall(waterfall_array)
    
    def on_acquisition_error(self, error_message: str):
        """Handle acquisition errors."""
        self.logger.error(f"Acquisition error: {error_message}")
        if self.main_window:
            self.main_window.show_error_message(f"SDR Error: {error_message}")
        self.stop_acquisition()
    
    # Device control methods
    def change_device(self, device_type: str):
        """Change SDR device type."""
        self.settings.sdr.device_type = device_type
        if self.sdr_manager:
            self.sdr_manager.set_device_type(device_type)
    
    def change_frequency(self, frequency: float):
        """Change center frequency."""
        self.settings.sdr.center_frequency = frequency
        if self.sdr_manager and self.sdr_manager.is_connected():
            self.sdr_manager.set_frequency(frequency)
    
    def change_sample_rate(self, sample_rate: float):
        """Change sample rate."""
        self.settings.sdr.sample_rate = sample_rate
        if self.sdr_manager and self.sdr_manager.is_connected():
            self.sdr_manager.set_sample_rate(sample_rate)
    
    def change_gain(self, gain: float):
        """Change RF gain."""
        self.settings.sdr.gain = gain
        if self.sdr_manager and self.sdr_manager.is_connected():
            self.sdr_manager.set_gain(gain)
    
    # Processing control methods
    def change_fft_size(self, fft_size: int):
        """Change FFT size."""
        self.settings.dsp.fft_size = fft_size
        if self.signal_processor:
            self.signal_processor.set_fft_size(fft_size)
    
    def change_window_function(self, window_type: str):
        """Change window function."""
        self.settings.dsp.window_type = window_type
        if self.signal_processor:
            self.signal_processor.set_window_function(window_type)
    
    def change_averaging(self, averaging: int):
        """Change spectrum averaging."""
        self.settings.dsp.averaging = averaging
        if self.signal_processor:
            self.signal_processor.set_averaging(averaging)
    
    def closeEvent(self, event: QCloseEvent):
        """Handle application close event."""
        try:
            self.logger.info("Closing application...")
            
            # Stop acquisition
            self.stop_acquisition()
            
            # Save settings
            if self.main_window:
                # Update settings with current window geometry
                geometry = self.geometry()
                self.settings.gui.window_x = geometry.x()
                self.settings.gui.window_y = geometry.y()
                self.settings.gui.window_width = geometry.width()
                self.settings.gui.window_height = geometry.height()
            
            self.settings.save_to_file()
            
            # Clean up
            if self.acquisition_thread:
                self.acquisition_thread.stop()
            
            event.accept()
            
        except Exception as e:
            self.logger.error(f"Error during application close: {e}")
            event.accept()
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get current device information."""
        if self.sdr_manager:
            return self.sdr_manager.get_device_info()
        return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "buffer_size": len(self.iq_buffer),
            "spectrum_size": len(self.spectrum_data),
            "waterfall_lines": len(self.waterfall_data),
            "connected": self.sdr_manager.is_connected() if self.sdr_manager else False,
        }