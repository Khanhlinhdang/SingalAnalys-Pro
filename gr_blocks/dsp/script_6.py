# Create core/app.py - Main application class
app_content = '''"""
RF Spectrum Analyzer - Main Application Class

This module contains the main application class that coordinates all components
including SDR backends, signal processing, and the GUI interface.

Integrates:
- pyspectrum: Real-time spectrum analysis
- mhostetter/sdr: Digital signal processing
- scikit-dsp-comm: Advanced DSP algorithms
"""

import logging
import threading
import queue
import time
from typing import Optional, Dict, Any
from pathlib import Path

from PySide6.QtWidgets import QMainWindow, QMessageBox, QApplication
from PySide6.QtCore import QTimer, QThread, Signal, QObject
from PySide6.QtGui import QCloseEvent

from config.settings import AppSettings
from core.sdr_backend import SDRBackendManager
from core.signal_processor import SignalProcessor
from gui.main_window import MainWindow
from utils.logger import get_sdr_logger, get_performance_logger, LogContext


class WorkerThread(QThread):
    """Worker thread for background signal processing"""
    
    # Signals for communication with main thread
    samples_ready = Signal(object)  # Raw samples
    spectrum_ready = Signal(object)  # Processed spectrum data
    error_occurred = Signal(str)    # Error messages
    status_update = Signal(str)     # Status updates
    
    def __init__(self, signal_processor: SignalProcessor):
        super().__init__()
        self.signal_processor = signal_processor
        self.running = False
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Main processing loop"""
        self.running = True
        self.logger.info("Worker thread started")
        
        try:
            while self.running:
                # Process signal data
                result = self.signal_processor.process_chunk()
                
                if result:
                    # Emit processed data
                    if 'samples' in result:
                        self.samples_ready.emit(result['samples'])
                    
                    if 'spectrum' in result:
                        self.spectrum_ready.emit(result['spectrum'])
                    
                    if 'status' in result:
                        self.status_update.emit(result['status'])
                
                # Small delay to prevent CPU overload
                time.sleep(0.001)  # 1ms
                
        except Exception as e:
            self.logger.error(f"Worker thread error: {e}", exc_info=True)
            self.error_occurred.emit(str(e))
        
        finally:
            self.logger.info("Worker thread stopped")
    
    def stop(self):
        """Stop the worker thread"""
        self.running = False
        self.wait()


class RFSpectrumApp(QMainWindow):
    """Main RF Spectrum Analyzer application"""
    
    def __init__(self, settings: AppSettings, args=None):
        super().__init__()
        
        # Initialize components
        self.settings = settings
        self.args = args
        self.logger = logging.getLogger(__name__)
        self.sdr_logger = get_sdr_logger()
        self.perf_logger = get_performance_logger()
        
        # Core components
        self.backend_manager = None
        self.signal_processor = None
        self.worker_thread = None
        
        # GUI components
        self.main_window = None
        
        # Status tracking
        self.is_running = False
        self.current_device = None
        self.sample_count = 0
        self.error_count = 0
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        
        # Initialize application
        self.initialize_app()
    
    def initialize_app(self):
        """Initialize all application components"""
        with LogContext(__name__, "Application initialization"):
            try:
                # Update settings from command line arguments
                if self.args:
                    self.settings.update_from_args(self.args)
                
                # Validate configuration
                if not self.settings.validate_config():
                    raise ValueError("Invalid configuration")
                
                # Initialize SDR backend manager
                self.backend_manager = SDRBackendManager(self.settings)
                
                # Initialize signal processor
                self.signal_processor = SignalProcessor(
                    self.settings,
                    self.backend_manager
                )
                
                # Initialize GUI
                self.main_window = MainWindow(self.settings, self)
                self.setCentralWidget(self.main_window)
                
                # Setup window properties
                self.setup_window()
                
                # Connect signals
                self.connect_signals()
                
                self.logger.info("Application initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Application initialization failed: {e}")
                self.show_error("Initialization Error", str(e))
                raise
    
    def setup_window(self):
        """Setup main window properties"""
        # Window title and properties
        self.setWindowTitle("RF Spectrum Analyzer v1.0")
        self.resize(self.settings.gui.window_width, self.settings.gui.window_height)
        
        # Center window on screen
        screen = QApplication.primaryScreen().geometry()
        window_geometry = self.geometry()
        x = (screen.width() - window_geometry.width()) // 2
        y = (screen.height() - window_geometry.height()) // 2
        self.move(x, y)
        
        # Set minimum size
        self.setMinimumSize(800, 600)
    
    def connect_signals(self):
        """Connect signals between components"""
        # Main window signals
        if self.main_window:
            self.main_window.start_requested.connect(self.start_processing)
            self.main_window.stop_requested.connect(self.stop_processing)
            self.main_window.settings_changed.connect(self.apply_settings)
            self.main_window.device_changed.connect(self.change_device)
            self.main_window.frequency_changed.connect(self.change_frequency)
            self.main_window.gain_changed.connect(self.change_gain)
        
        # Backend manager signals
        if self.backend_manager:
            self.backend_manager.device_connected.connect(self.on_device_connected)
            self.backend_manager.device_disconnected.connect(self.on_device_disconnected)
            self.backend_manager.error_occurred.connect(self.on_backend_error)
    
    def start_processing(self):
        """Start RF signal processing"""
        if self.is_running:
            self.logger.warning("Processing already running")
            return
        
        try:
            with LogContext(__name__, "Starting signal processing"):
                # Initialize and start backend
                if not self.backend_manager.initialize():
                    raise RuntimeError("Failed to initialize SDR backend")
                
                # Start signal processor
                self.signal_processor.start()
                
                # Create and start worker thread
                self.worker_thread = WorkerThread(self.signal_processor)
                self.worker_thread.samples_ready.connect(self.on_samples_ready)
                self.worker_thread.spectrum_ready.connect(self.on_spectrum_ready)
                self.worker_thread.error_occurred.connect(self.on_processing_error)
                self.worker_thread.status_update.connect(self.on_status_update)
                self.worker_thread.start()
                
                # Start update timer
                update_interval = 1000 // self.settings.gui.update_rate  # Convert FPS to ms
                self.update_timer.start(update_interval)
                
                self.is_running = True
                self.logger.info("Signal processing started")
                
                # Update GUI state
                if self.main_window:
                    self.main_window.set_processing_state(True)
        
        except Exception as e:
            self.logger.error(f"Failed to start processing: {e}")
            self.show_error("Processing Error", f"Failed to start processing: {e}")
            self.stop_processing()
    
    def stop_processing(self):
        """Stop RF signal processing"""
        if not self.is_running:
            return
        
        try:
            with LogContext(__name__, "Stopping signal processing"):
                self.is_running = False
                
                # Stop update timer
                self.update_timer.stop()
                
                # Stop worker thread
                if self.worker_thread:
                    self.worker_thread.stop()
                    self.worker_thread = None
                
                # Stop signal processor
                if self.signal_processor:
                    self.signal_processor.stop()
                
                # Stop backend
                if self.backend_manager:
                    self.backend_manager.stop()
                
                self.logger.info("Signal processing stopped")
                
                # Update GUI state
                if self.main_window:
                    self.main_window.set_processing_state(False)
        
        except Exception as e:
            self.logger.error(f"Error stopping processing: {e}")
    
    def on_samples_ready(self, samples):
        """Handle new samples from worker thread"""
        self.sample_count += len(samples)
        self.sdr_logger.log_samples_processed(len(samples))
        
        # Pass samples to GUI for IQ display
        if self.main_window:
            self.main_window.update_iq_plot(samples)
    
    def on_spectrum_ready(self, spectrum_data):
        """Handle new spectrum data from worker thread"""
        # Update spectrum display
        if self.main_window:
            self.main_window.update_spectrum(spectrum_data)
    
    def on_processing_error(self, error_msg):
        """Handle processing errors"""
        self.error_count += 1
        self.logger.error(f"Processing error [{self.error_count}]: {error_msg}")
        
        if self.main_window:
            self.main_window.show_status_message(f"Error: {error_msg}", 5000)
    
    def on_status_update(self, status_msg):
        """Handle status updates"""
        self.logger.debug(f"Status: {status_msg}")
        
        if self.main_window:
            self.main_window.show_status_message(status_msg, 2000)
    
    def apply_settings(self, new_settings):
        """Apply new settings"""
        try:
            # Update settings
            self.settings = new_settings
            
            # Apply to signal processor
            if self.signal_processor:
                self.signal_processor.update_settings(new_settings)
            
            # Apply to backend
            if self.backend_manager:
                self.backend_manager.update_settings(new_settings)
            
            self.logger.info("Settings applied successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to apply settings: {e}")
            self.show_error("Settings Error", str(e))
    
    def change_device(self, device_info):
        """Change SDR device"""
        try:
            was_running = self.is_running
            
            # Stop processing if running
            if was_running:
                self.stop_processing()
            
            # Change device
            if self.backend_manager:
                self.backend_manager.set_device(device_info)
                self.current_device = device_info
                
                self.sdr_logger.log_device_info(device_info)
            
            # Restart processing if it was running
            if was_running:
                self.start_processing()
        
        except Exception as e:
            self.logger.error(f"Failed to change device: {e}")
            self.show_error("Device Error", str(e))
    
    def change_frequency(self, frequency):
        """Change center frequency"""
        try:
            old_freq = self.settings.sdr.center_freq
            self.settings.sdr.center_freq = frequency
            
            if self.backend_manager and self.backend_manager.current_backend:
                self.backend_manager.current_backend.set_center_frequency(frequency)
            
            self.sdr_logger.log_frequency_change(old_freq, frequency)
        
        except Exception as e:
            self.logger.error(f"Failed to change frequency: {e}")
            self.show_error("Frequency Error", str(e))
    
    def change_gain(self, gain):
        """Change RF gain"""
        try:
            old_gain = self.settings.sdr.gain
            self.settings.sdr.gain = gain
            
            if self.backend_manager and self.backend_manager.current_backend:
                self.backend_manager.current_backend.set_gain(gain)
            
            self.sdr_logger.log_gain_change(old_gain, gain)
        
        except Exception as e:
            self.logger.error(f"Failed to change gain: {e}")
            self.show_error("Gain Error", str(e))
    
    def on_device_connected(self, device_info):
        """Handle device connection"""
        self.current_device = device_info
        self.logger.info(f"Device connected: {device_info}")
        
        if self.main_window:
            self.main_window.set_device_status(True, str(device_info))
    
    def on_device_disconnected(self):
        """Handle device disconnection"""
        self.logger.warning("Device disconnected")
        self.current_device = None
        
        if self.main_window:
            self.main_window.set_device_status(False, "No device")
        
        # Stop processing
        self.stop_processing()
    
    def on_backend_error(self, error_msg):
        """Handle backend errors"""
        self.logger.error(f"Backend error: {error_msg}")
        
        if self.main_window:
            self.main_window.show_status_message(f"Backend Error: {error_msg}", 5000)
    
    def update_status(self):
        """Update application status (called by timer)"""
        if self.main_window:
            # Update performance metrics
            self.main_window.update_performance_display({
                'sample_count': self.sample_count,
                'error_count': self.error_count,
                'is_running': self.is_running,
                'current_device': str(self.current_device) if self.current_device else "None"
            })
    
    def show_error(self, title: str, message: str):
        """Show error message dialog"""
        QMessageBox.critical(self, title, message)
    
    def show_info(self, title: str, message: str):
        """Show info message dialog"""
        QMessageBox.information(self, title, message)
    
    def closeEvent(self, event: QCloseEvent):
        """Handle application close event"""
        try:
            self.logger.info("Application closing...")
            
            # Stop processing
            self.stop_processing()
            
            # Save settings
            self.settings.save_config()
            
            # Clean up resources
            if self.backend_manager:
                self.backend_manager.cleanup()
            
            event.accept()
            self.logger.info("Application closed successfully")
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            event.accept()  # Close anyway
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get current application status"""
        return {
            'is_running': self.is_running,
            'current_device': self.current_device,
            'sample_count': self.sample_count,
            'error_count': self.error_count,
            'settings': self.settings
        }
'''

with open("rf_spectrum_analyzer/core/app.py", "w") as f:
    f.write(app_content)

print("Created core/app.py")