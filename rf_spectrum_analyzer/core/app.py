"""
Main Application Class for RF Spectrum Analyzer
Coordinates GUI, SDR backend, and signal processing components.
"""

import sys
import logging
import time
import numpy as np
from collections import deque
from threading import Lock
from pathlib import Path
from typing import Optional, Dict, Any, Deque
from datetime import datetime
from uuid import uuid4
try:
    import scipy.signal as sp_signal
except Exception:
    sp_signal = None
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtCore import QTimer, QThread, Signal, QObject
from PySide6.QtGui import QCloseEvent

from rf_spectrum_analyzer.gui.main_window import MainWindow
from rf_spectrum_analyzer.core.sdr_backend import SDRBackendManager
from rf_spectrum_analyzer.core.signal_processor import SignalProcessor
from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer
from rf_spectrum_analyzer.config.settings import Settings
from rf_spectrum_analyzer.utils.logger import get_logger
from rf_spectrum_analyzer.utils.file_io import DataExporter, DataImporter

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
        """Main acquisition loop with robust error handling."""
        try:
            logger.info("Starting data acquisition thread")
            self.running = True
            connection_errors = 0
            max_connection_errors = 5
            
            while self.running:
                if self.sdr_manager.is_connected():
                    # Read IQ samples from SDR
                    samples = self.sdr_manager.read_samples(self.buffer_size)
                    if samples is not None and len(samples) > 0:
                        self.data_ready.emit(samples)
                        connection_errors = 0  # Reset error counter on successful read
                    else:
                        connection_errors += 1
                        if connection_errors >= max_connection_errors:
                            logger.warning(f"No data received for {connection_errors} consecutive attempts. Checking connection health...")
                            # Check if backend has health check method
                            if hasattr(self.sdr_manager.backend, 'is_connection_healthy'):
                                if not self.sdr_manager.backend.is_connection_healthy():
                                    logger.warning("Connection unhealthy, backend will attempt reconnection on next read")
                            connection_errors = 0  # Reset counter
                        self.msleep(10)  # Brief pause if no data available
                else:
                    self.msleep(100)  # Wait longer if not connected
                    
        except Exception as e:
            logger.error(f"Data acquisition error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            logger.info("Data acquisition thread stopped")
    
    def _cancel_backend_read(self):
        """Ask backend to cancel any blocking read operation if supported."""
        try:
            backend = getattr(self.sdr_manager, 'backend', None)
            if backend is None:
                return

            if hasattr(backend, 'cancel_read'):
                backend.cancel_read()
            elif hasattr(backend, 'stop_streaming'):
                backend.stop_streaming()
        except Exception as exc:
            logger.debug(f"Backend cancel hook failed: {exc}")

    def stop(self, wait_timeout_ms: int = 1500) -> bool:
        """Stop the acquisition thread with bounded wait time."""
        self.running = False
        self._cancel_backend_read()
        return self.wait(wait_timeout_ms)


class ProcessingThread(QThread):
    """Background DSP worker with bounded latest-wins queue."""
    processing_result = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.signal_processor = SignalProcessor(settings)
        self.running = False
        self._queue: Deque[np.ndarray] = deque(maxlen=2)
        self._queue_lock = Lock()
        self._frame_count = 0

        self.signal_processor.set_auto_detection(
            getattr(settings.detection, 'auto_detection_enabled', False)
        )
        self.signal_processor.set_advanced_analysis(
            getattr(settings.detection, 'advanced_analysis_enabled', False)
        )

    def submit_samples(self, samples: np.ndarray):
        """Submit a frame for DSP processing with backpressure."""
        if samples is None or len(samples) == 0:
            return
        with self._queue_lock:
            self._queue.append(np.asarray(samples, dtype=np.complex64).copy())

    def _get_next_samples(self) -> Optional[np.ndarray]:
        with self._queue_lock:
            if not self._queue:
                return None
            latest = self._queue[-1]
            self._queue.clear()
            return latest

    def run(self):
        """Run DSP processing loop in background thread."""
        self.running = True
        try:
            while self.running:
                samples = self._get_next_samples()
                if samples is None:
                    self.msleep(5)
                    continue

                spectrum = self.signal_processor.compute_spectrum(samples)

                advanced_result = None
                interval_frames = max(
                    1,
                    int(getattr(self.settings.detection, 'advanced_analysis_interval_frames', 10))
                )
                if (
                    getattr(self.signal_processor, '_advanced_analysis_enabled', False)
                    and self._frame_count % interval_frames == 0
                ):
                    advanced_result = self.signal_processor.process_complete_chain(samples)

                self.processing_result.emit(
                    {
                        'iq_samples': samples,
                        'spectrum': spectrum,
                        'analysis_result': advanced_result,
                        'frame_count': self._frame_count,
                    }
                )
                self._frame_count += 1
        except Exception as exc:
            logger.error(f"Processing thread error: {exc}")
            self.error_occurred.emit(str(exc))

    def stop(self, wait_timeout_ms: int = 1500) -> bool:
        """Stop the processing thread with bounded wait time."""
        self.running = False
        return self.wait(wait_timeout_ms)


class FileAnalysisWorker(QThread):
    """Worker thread that runs the full signal analysis pipeline on a file."""

    analysis_done = Signal(dict)
    error_occurred = Signal(str)

    def __init__(
        self,
        data_importer,
        signal_analyzer_class,
        settings: Settings,
        filename: str,
        center_freq: float,
        bandwidth: float,
        advanced_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self._data_importer = data_importer
        self._signal_analyzer_class = signal_analyzer_class
        self._settings = settings
        self._filename = filename
        self._center_freq = center_freq
        self._bandwidth = bandwidth
        self._advanced_params = advanced_params or {}

    def _apply_advanced(
        self,
        iq_data: np.ndarray,
        sample_rate: float,
    ) -> (np.ndarray, float):
        advanced = self._advanced_params or {}

        sample_rate_override = float(advanced.get('sample_rate_hz', 0.0) or 0.0)
        if sample_rate_override > 1.0:
            sample_rate = sample_rate_override

        iq = np.asarray(iq_data, dtype=np.complex64)

        start_sec = float(advanced.get('start_sec', 0.0) or 0.0)
        duration_sec = float(advanced.get('duration_sec', 0.0) or 0.0)
        if start_sec > 0.0 or duration_sec > 0.0:
            start_idx = int(max(start_sec, 0.0) * sample_rate)
            if duration_sec > 0.0:
                end_idx = start_idx + int(duration_sec * sample_rate)
            else:
                end_idx = len(iq)
            start_idx = min(max(start_idx, 0), len(iq))
            end_idx = min(max(end_idx, start_idx), len(iq))
            iq = iq[start_idx:end_idx]

        freq_offset_hz = float(advanced.get('freq_offset_hz', 0.0) or 0.0)
        if abs(freq_offset_hz) > 0.0 and len(iq) > 0 and sample_rate > 0.0:
            n = np.arange(len(iq), dtype=np.float64)
            rot = np.exp(-1j * 2.0 * np.pi * freq_offset_hz * (n / sample_rate))
            iq = (iq * rot).astype(np.complex64, copy=False)

        return iq, sample_rate

    def run(self):
        try:
            iq_data, metadata = self._data_importer.import_signal_source(self._filename)
            if iq_data is None or len(iq_data) == 0:
                self.error_occurred.emit(f"Failed to import signal source from {self._filename}")
                return

            sample_rate = float(metadata.get('sample_rate') or self._settings.sdr.sample_rate)
            iq_prepared, sample_rate = self._apply_advanced(iq_data, sample_rate)
            if iq_prepared is None or len(iq_prepared) == 0:
                self.error_occurred.emit("Selected segment is empty after applying advanced file settings")
                return

            analyzer = self._signal_analyzer_class(sample_rate)
            result = analyzer.analyze_signal_comprehensive(
                iq_prepared,
                self._center_freq,
                self._bandwidth,
            )

            payload = result.get('payload', result)
            payload['source_metadata'] = metadata
            payload['source_file'] = self._filename
            payload['source_advanced'] = self._advanced_params
            self.analysis_done.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class AnalysisRequestThread(QThread):
    """One-shot async thread for analysis requests from the spectrum widget."""

    completed = Signal(dict)
    error_occurred = Signal(str)

    def __init__(
        self,
        sample_rate: float,
        iq_data: np.ndarray,
        center_freq: float,
        bandwidth: float,
        analysis_context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.iq_data = np.asarray(iq_data, dtype=np.complex64)
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.analysis_context = dict(analysis_context or {})

    def _attach_analysis_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Attach ROI/runtime context into analysis payload for observability."""
        if not isinstance(result, dict):
            return result

        payload = result.get('payload')
        if isinstance(payload, dict):
            payload['analysis_context'] = dict(self.analysis_context)
            return result

        result['analysis_context'] = dict(self.analysis_context)
        return result

    def run(self):
        try:
            analyzer = SignalAnalyzer(self.sample_rate)
            result = analyzer.analyze_signal_comprehensive(
                self.iq_data,
                self.center_freq,
                self.bandwidth,
            )
            self.completed.emit(self._attach_analysis_context(result))
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class RFSpectrumAnalyzerApp(QObject):
    """Main application class that coordinates all components."""
    
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.logger = get_logger(__name__)
        self.headless_mode = getattr(settings, 'headless_mode', False)
        
        # Core components
        self.sdr_manager = None
        self.signal_processor = None
        self.signal_analyzer = None
        self.main_window = None
        self.acquisition_thread = None
        self.processing_thread = None
        self.analysis_request_thread = None
        self.data_exporter = DataExporter()
        self.data_importer = DataImporter()
        self._file_worker = None
        self._resume_acquisition_after_file = False
        self.latest_image_artifact: Optional[Dict[str, Any]] = None
        self.latest_pcm_artifact: Optional[Dict[str, Any]] = None
        self.latest_signal_source: Optional[Dict[str, Any]] = None
        self.analysis_session_records: Deque[Dict[str, Any]] = deque(maxlen=512)
        self.roi_analysis_queue: Deque[Dict[str, Any]] = deque(maxlen=128)
        self._last_roi_request_signature: Optional[str] = None
        self._last_roi_request_mono: float = 0.0
        self._roi_request_debounce_sec: float = 0.25
        self._roi_preset_max_count: int = 32
        
        # Data buffers
        self.iq_snapshot_ring: Deque[np.ndarray] = deque(maxlen=64)
        self.spectrum_data = np.array([])
        self._latest_waterfall_line = None
        self.constellation_data = {
            'iq_samples': np.array([], dtype=np.complex64),
            'symbols': np.array([], dtype=np.complex64),
            'modulation_info': {}
        }
        self.bitstream_data: Deque[int] = deque(maxlen=10000)
        self._bitstream_total_count = 0
        self._bitstream_gui_sent_count = 0
        self.enable_async_analysis_requests = True
        
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
            self.signal_processor.set_auto_detection(
                getattr(self.settings.detection, 'auto_detection_enabled', False)
            )
            self.signal_processor.set_advanced_analysis(
                getattr(self.settings.detection, 'advanced_analysis_enabled', False)
            )
            
            # Initialize signal analyzer
            self.signal_analyzer = SignalAnalyzer(self.settings.sdr.sample_rate)
            
            # Initialize SDR backend manager
            self.sdr_manager = SDRBackendManager(self.settings)
            
            # Check demo mode setting
            self.logger.info(f"Demo mode from settings: {getattr(self.settings, 'demo_mode', False)}")
            self.logger.info(f"Demo mode instance variable: {self.demo_mode}")
            
            # Try to connect to SDR device - if fails, enable demo mode
            self._try_sdr_connection()
            
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
            try:
                self.acquisition_thread = DataAcquisitionThread(
                    self.sdr_manager, self.settings
                )
                self.acquisition_thread.data_ready.connect(self.on_new_data)
                self.acquisition_thread.error_occurred.connect(self.on_acquisition_error)
                self.logger.info("Data acquisition thread initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize acquisition thread: {e}")
                self.acquisition_thread = None

            try:
                self.processing_thread = ProcessingThread(self.settings)
                self.processing_thread.processing_result.connect(self._on_processing_result)
                self.processing_thread.error_occurred.connect(self.on_acquisition_error)
                self.logger.info("Processing thread initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize processing thread: {e}")
                self.processing_thread = None
            
            # Setup demo mode if enabled
            if self.demo_mode and not self.headless_mode:
                self.setup_demo_mode()
            
            self.logger.info("Application initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            raise
    
    def _try_sdr_connection(self):
        """Try to connect to SDR device, enable demo mode if connection fails."""
        try:
            if self.headless_mode:
                self.logger.info("Headless mode enabled - skipping SDR connection")
                self.demo_mode = False
                return

            device_type = self.settings.sdr.device_type
            self.logger.info(f"Attempting to connect to {device_type} device...")
            
            # Check if demo mode was explicitly requested
            demo_requested = getattr(self.settings, 'demo_mode', False)
            
            if demo_requested:
                self.logger.info("Demo mode explicitly requested - skipping SDR connection")
                self._enable_demo_mode()
            elif self.sdr_manager.connect():
                self.logger.info(f"Successfully connected to {device_type}")
                # Only disable demo mode if it wasn't explicitly requested
                if not demo_requested:
                    self.demo_mode = False
                # Auto-start acquisition for real SDR devices
                self.logger.info("Auto-starting data acquisition...")
                self.start_acquisition()
            else:
                self.logger.warning(f"Failed to connect to {device_type}, enabling demo mode")
                self._enable_demo_mode()
                
        except Exception as e:
            self.logger.warning(f"SDR connection error: {e}")
            self.logger.info("Enabling demo mode due to connection failure")
            self._enable_demo_mode()
    
    def _enable_demo_mode(self):
        """Enable demo mode with synthetic data."""
        self.demo_mode = True
        self.settings.demo_mode = True
        
        # Create demo timer if it doesn't exist
        if self.demo_timer is None:
            self.demo_timer = QTimer()
        
        self.logger.info("Demo mode enabled - using synthetic signal data")
    
    def setup_window(self):
        """Setup main window properties."""
        if self.main_window:
            self.main_window.setWindowTitle("RF Spectrum Analyzer")
            self.main_window.resize(
                self.settings.gui.window_width,
                self.settings.gui.window_height
            )

            # Validate saved position against all available screens before restoring.
            # If the saved coordinates land entirely off-screen (e.g. a previously
            # connected external monitor was disconnected) we center instead.
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import QPoint, QRect
            saved_x = self.settings.gui.window_x
            saved_y = self.settings.gui.window_y
            screens = QApplication.screens()
            on_screen = any(
                screen.geometry().contains(QPoint(saved_x + 50, saved_y + 50))
                for screen in screens
            )
            if on_screen:
                self.main_window.move(saved_x, saved_y)
            else:
                # Center on the primary screen
                primary = QApplication.primaryScreen()
                screen_rect: QRect = primary.availableGeometry()
                win_w = self.main_window.width()
                win_h = self.main_window.height()
                cx = screen_rect.x() + (screen_rect.width() - win_w) // 2
                cy = screen_rect.y() + (screen_rect.height() - win_h) // 2
                self.main_window.move(max(screen_rect.x(), cx), max(screen_rect.y(), cy))
                self.logger.info(
                    "Saved window position (%d, %d) is off-screen; centering on primary display.",
                    saved_x, saved_y,
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
            self.main_window.bandwidth_changed.connect(self.change_bandwidth)
            self.main_window.gain_changed.connect(self.change_gain)
            
            # Connect processing signals
            self.main_window.fft_size_changed.connect(self.change_fft_size)
            self.main_window.window_changed.connect(self.change_window_function)
            self.main_window.averaging_changed.connect(self.change_averaging)
            
            # Connect detection signals
            self.main_window.manual_detection_triggered.connect(self.trigger_manual_detection)
            self.main_window.tdma_detection_triggered.connect(self.trigger_tdma_detection)
            self.main_window.auto_detection_toggled.connect(self.toggle_auto_detection)
            self.main_window.advanced_analysis_toggled.connect(self.toggle_advanced_analysis)
            
            # Connect signal analysis signals
            if hasattr(self.main_window.spectrum_widget, 'signal_analysis_requested'):
                self.main_window.spectrum_widget.signal_analysis_requested.connect(self.handle_signal_analysis_request)
            self.main_window.detection_threshold_changed.connect(self.change_detection_threshold)
            self.main_window.detection_interval_changed.connect(self.change_detection_interval)
            
            # Connect frequency analysis signals
            self.main_window.frequency_range_changed.connect(self.change_frequency_range)
            self.main_window.center_frequency_locked.connect(self.toggle_center_frequency_lock)
            self.main_window.analysis_bandwidth_changed.connect(self.change_analysis_bandwidth)
            
            # Connect sequential workflow signals
            self.main_window.demodulate_triggered.connect(self.trigger_sequential_demodulation)
            self.main_window.decode_triggered.connect(self.trigger_sequential_decoding)
            self.main_window.export_image_artifact_requested.connect(self.export_latest_image_artifact)
            self.main_window.export_pcm_artifact_requested.connect(self.export_latest_pcm_artifact)
            self.main_window.export_session_report_requested.connect(self.export_session_decode_report)
            self.main_window.load_session_report_requested.connect(self.load_session_decode_report)
            self.main_window.process_signal_file_requested.connect(self.process_signal_file)
    
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
        
        # Constellation and bitstream updates are driven by incremental data callbacks.
        self.constellation_timer.stop()
        self.bitstream_timer.stop()
    
    def start_acquisition(self):
        """Start SDR data acquisition."""
        try:
            self.logger.info("Starting SDR acquisition...")
            self._bitstream_gui_sent_count = 0
            self._bitstream_total_count = 0
            
            # Connect to SDR device
            if not self.sdr_manager.is_connected() and not self.sdr_manager.connect():
                self.logger.error("Failed to connect to SDR device")
                return False
            
            # Configure device
            device_settings = self.settings.get_device_settings()
            if not self.sdr_manager.configure(device_settings):
                self.logger.error("Failed to configure SDR device")
                return False
            
            # Start acquisition thread
            if self.acquisition_thread is None:
                self.logger.error("Acquisition thread not initialized")
                return False

            if self.processing_thread and not self.processing_thread.isRunning():
                self.processing_thread.start()
            
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
                if not self.acquisition_thread.stop(wait_timeout_ms=1500):
                    self.logger.warning("Acquisition thread did not stop within timeout")

            if self.processing_thread and self.processing_thread.isRunning():
                if not self.processing_thread.stop(wait_timeout_ms=1500):
                    self.logger.warning("Processing thread did not stop within timeout")

            if self.analysis_request_thread and self.analysis_request_thread.isRunning():
                if not self.analysis_request_thread.wait(1000):
                    self.logger.warning("Analysis request thread still running during stop")
            
            # Disconnect SDR
            if self.sdr_manager:
                self.sdr_manager.disconnect()
            
            # Update UI state
            if self.main_window:
                self.main_window.set_acquisition_state(False)

            self._bitstream_gui_sent_count = 0
            self._bitstream_total_count = 0
            self.bitstream_data.clear()
            self.iq_snapshot_ring.clear()
            self._latest_waterfall_line = None
            
            self.logger.info("SDR acquisition stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping acquisition: {e}")
    
    def on_new_data(self, samples: np.ndarray):
        """Handle new IQ data from acquisition thread."""
        try:
            if samples is None or len(samples) == 0:
                return

            self.iq_snapshot_ring.append(np.asarray(samples, dtype=np.complex64).copy())

            # Keep sync processor state fresh for manual/sequential actions.
            if self.signal_processor:
                fft_size = self.settings.dsp.fft_size
                self.signal_processor.update_current_data(samples[:fft_size])

            if self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.submit_samples(samples)
            else:
                self.process_iq_data(samples)
                
        except Exception as e:
            self.logger.error(f"Error processing new data: {e}")
    
    def process_iq_data(self, samples: Optional[np.ndarray] = None):
        """Process IQ data to generate spectrum and constellation analysis."""
        try:
            if self.signal_processor is None:
                return

            if samples is None or len(samples) == 0:
                return
            
            # Extract samples for processing
            fft_size = self.settings.dsp.fft_size
            samples = samples[:fft_size]
            
            # Update signal processor with current data for detection
            self.signal_processor.update_current_data(samples)
            
            # Compute spectrum with adaptive throttling
            spectrum = self.signal_processor.compute_spectrum(samples)
            if spectrum is not None:
                self.spectrum_data = spectrum
                self._latest_waterfall_line = spectrum.copy()
            
            # Process advanced analysis only when feature flag is enabled.
            interval_frames = max(
                1,
                int(getattr(self.settings.detection, 'advanced_analysis_interval_frames', 10))
            )
            if (self.signal_processor and
                getattr(self.signal_processor, '_advanced_analysis_enabled', False) and
                self.frame_count % interval_frames == 0):
                self.logger.debug("Triggering advanced analysis...")
                try:
                    self.process_advanced_analysis(samples)
                    self.logger.debug("Advanced analysis completed successfully")
                except Exception as e:
                    self.logger.error(f"Error in advanced analysis: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
            
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
                # Only fallback to synthetic result in demo mode.
                if self.demo_mode:
                    result = self._generate_demo_analysis_result()
                else:
                    return
            
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

    def _on_processing_result(self, result: Dict[str, Any]):
        """Consume DSP worker outputs on UI thread."""
        try:
            spectrum = result.get('spectrum')
            if spectrum is not None and len(spectrum) > 0:
                self.spectrum_data = spectrum
                self._latest_waterfall_line = spectrum.copy()

            analysis_result = result.get('analysis_result')
            iq_samples = result.get('iq_samples')
            if analysis_result and iq_samples is not None and len(iq_samples) > 0:
                if not analysis_result.get('success', False):
                    if self.demo_mode:
                        analysis_result = self._generate_demo_analysis_result()
                    else:
                        return

                self.update_constellation_data(iq_samples, analysis_result)
                self.update_bitstream_data(analysis_result)
                if self.main_window:
                    self.update_gui_widgets()

            self.frame_count = result.get('frame_count', self.frame_count)
        except Exception as exc:
            self.logger.error(f"Error consuming processing result: {exc}")
    
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
                
                self.bitstream_data.extend(int(bit) for bit in binary_data.tolist())
                self._bitstream_total_count += int(len(binary_data))

                self.logger.debug(
                    f"Added {len(binary_data)} bits to bitstream, buffered: {len(self.bitstream_data)}"
                )
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
                    # Send only incremental bits to avoid duplicate re-appends on timer updates.
                    unsent_count = max(0, self._bitstream_total_count - self._bitstream_gui_sent_count)
                    if unsent_count > 0:
                        send_count = min(unsent_count, 128, len(self.bitstream_data))
                        new_bits = list(self.bitstream_data)[-send_count:] if send_count > 0 else []
                        if new_bits:
                            bitstream_widget.add_bits(new_bits)
                        self._bitstream_gui_sent_count = self._bitstream_total_count
                        self.logger.debug(f"Updated bitstream with {len(new_bits)} new bits")
                    
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
        if self.main_window and self._latest_waterfall_line is not None:
            self.main_window.update_waterfall(self._latest_waterfall_line)
            self._latest_waterfall_line = None
    
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
        # Intentionally unused: incremental path is handled by update_gui_widgets.
        return
    
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

            # "file" is not a live SDR backend — open file dialog instead
            if device_type.lower() == "file":
                if self.main_window and not self.headless_mode:
                    self.main_window.trigger_open_signal_file()
                return

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
        """Change center frequency in real-time without stopping acquisition."""
        try:
            self.logger.info(f"Real-time frequency change to: {frequency/1e6:.3f} MHz")
            
            # Update settings first
            old_frequency = self.settings.sdr.center_frequency
            self.settings.sdr.center_frequency = frequency
            
            # If SDR is connected, apply change immediately
            if self.sdr_manager and self.sdr_manager.is_connected():
                success = self.sdr_manager.set_frequency(frequency)
                if success:
                    self.logger.info(f"Frequency successfully changed from {old_frequency/1e6:.3f} MHz to {frequency/1e6:.3f} MHz")
                    
                    # Update signal processor with new frequency if needed
                    if hasattr(self.signal_processor, 'set_center_frequency'):
                        self.signal_processor.set_center_frequency(frequency)
                    
                    # Notify GUI of successful change
                    if hasattr(self, 'main_window') and self.main_window:
                        self.main_window.update_device_frequency(frequency)
                else:
                    self.logger.error("Failed to set frequency on device")
                    # Revert settings change
                    self.settings.sdr.center_frequency = old_frequency
            else:
                self.logger.warning("SDR not connected - frequency change saved for next connection")
                
        except Exception as e:
            self.logger.error(f"Error changing frequency: {e}")
            # Revert settings on error
            if 'old_frequency' in locals():
                self.settings.sdr.center_frequency = old_frequency
    
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
            if self.processing_thread and self.processing_thread.signal_processor:
                self.processing_thread.signal_processor.update_sample_rate(sample_rate)
                
        except Exception as e:
            self.logger.error(f"Error changing sample rate: {e}")
    
    def change_bandwidth(self, bandwidth: float):
        """Change bandwidth in real-time without stopping acquisition."""
        try:
            self.logger.info(f"Real-time bandwidth change to: {bandwidth/1e6:.3f} MHz")
            
            # Update settings first
            old_bandwidth = self.settings.sdr.bandwidth
            self.settings.sdr.bandwidth = bandwidth
            
            # If SDR is connected, apply change immediately
            if self.sdr_manager and self.sdr_manager.is_connected():
                # Check if backend supports bandwidth setting
                if hasattr(self.sdr_manager, 'set_bandwidth'):
                    success = self.sdr_manager.set_bandwidth(bandwidth)
                    if success:
                        self.logger.info(f"Bandwidth successfully changed from {old_bandwidth/1e6:.3f} MHz to {bandwidth/1e6:.3f} MHz")
                        
                        # For SpyServer, bandwidth change may affect sample rate
                        # Update signal processor if needed
                        if hasattr(self.signal_processor, 'set_sample_rate') and bandwidth != old_bandwidth:
                            # Update signal processor with new effective sample rate
                            self.signal_processor.set_sample_rate(self.settings.sdr.sample_rate)
                        
                        # Notify GUI of successful change
                        if hasattr(self, 'main_window') and self.main_window:
                            self.main_window.update_device_bandwidth(bandwidth)
                    else:
                        self.logger.error("Failed to set bandwidth on device")
                        # Revert settings change
                        self.settings.sdr.bandwidth = old_bandwidth
                else:
                    self.logger.warning("Backend does not support real-time bandwidth setting")
                    # For backends without bandwidth support, log but keep setting
                    self.logger.info("Bandwidth setting saved for future use")
            else:
                self.logger.warning("SDR not connected - bandwidth change saved for next connection")
                
        except Exception as e:
            self.logger.error(f"Error changing bandwidth: {e}")
            # Revert settings on error
            if 'old_bandwidth' in locals():
                self.settings.sdr.bandwidth = old_bandwidth
    
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
                    if hasattr(self.main_window, 'save_dock_layout_to_settings'):
                        self.main_window.save_dock_layout_to_settings()
                
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
        print("DEMO SETUP: Setting up demo mode")
        
        if self.demo_timer:
            print(f"DEMO SETUP: Timer exists, connecting to generate_demo_data")
            self.demo_timer.timeout.connect(self.generate_demo_data)
            self.demo_timer.start(100)  # Generate data every 100ms
            self.logger.info("Demo timer started - generating data every 100ms")
            print(f"DEMO SETUP: Timer started with 100ms interval")
        else:
            print("DEMO SETUP: ERROR - Timer is None!")
            self.logger.error("Demo timer is None!")
    
    def generate_demo_data(self):
        """Generate simulated IQ data for demonstration."""
        # CRITICAL DEBUG: This should appear if timer is working
        print("TIMER TRIGGERED: generate_demo_data called")
        self.logger.debug("TIMER TRIGGERED: generate_demo_data called")
        
        try:
            self.logger.debug(f"Generating demo data, counter: {self.demo_counter}")
            
            # Parameters
            sample_rate = self.settings.sdr.sample_rate
            num_samples = self.settings.dsp.fft_size
            
            self.logger.debug(f"Demo parameters: sample_rate={sample_rate}, num_samples={num_samples}")
            
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
            
            # Normalize safely
            max_amplitude = np.max(np.abs(signal))
            if max_amplitude > 1e-10:  # Avoid division by zero
                signal = signal / max_amplitude
            else:
                # If signal is essentially zero, create a small random signal
                signal = 0.01 * (np.random.randn(num_samples) + 1j*np.random.randn(num_samples))
            
            # Ensure finite values
            signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Send to processing pipeline
            self.on_new_data(signal.astype(np.complex64))
            self.demo_counter += 1
            
            self.logger.debug(f"Demo data stored: {len(signal)} samples, type: {type(signal)}, signal_type: {signal_type}")
            
        except Exception as e:
            self.logger.debug(f"Error generating demo data: {e}")
    
    def trigger_manual_detection(self):
        """Trigger manual signal detection."""
        try:
            if hasattr(self, 'signal_processor') and self.signal_processor:
                results = self.signal_processor.detect_signals_manual()
                if results:
                    # Update GUI with detection results
                    if self.main_window:
                        snr_db = results.get('snr_db', None)
                        confidence = results.get('confidence', None) * 100 if results.get('confidence') else None
                        self.main_window.update_detection_status(True, snr_db, confidence)
                    self.logger.info(f"Manual detection completed: {results}")
                else:
                    # No signals detected
                    if self.main_window:
                        self.main_window.update_detection_status(False)
                    self.logger.info("Manual detection: No signals found")
        except Exception as e:
            self.logger.error(f"Error in manual detection: {e}")
    
    def trigger_tdma_detection(self):
        """Trigger TDMA burst detection and analysis."""
        try:
            if hasattr(self, 'signal_processor') and self.signal_processor:
                results = self.signal_processor.detect_tdma_bursts()
                if results:
                    # Update GUI with TDMA detection results
                    if self.main_window:
                        num_bursts = len(results.get('bursts', []))
                        self.main_window.update_detection_status(True, None, num_bursts * 20)  # Confidence based on burst count
                    self.logger.info(f"TDMA detection completed: {len(results.get('bursts', []))} bursts found")
                else:
                    # No TDMA bursts detected
                    if self.main_window:
                        self.main_window.update_detection_status(False)
                    self.logger.info("TDMA detection: No bursts found")
        except Exception as e:
            self.logger.error(f"Error in TDMA detection: {e}")
    
    def toggle_auto_detection(self, enabled):
        """Toggle automatic signal detection."""
        try:
            if hasattr(self, 'signal_processor') and self.signal_processor:
                self.signal_processor.set_auto_detection(enabled)
            if self.processing_thread and self.processing_thread.signal_processor:
                self.processing_thread.signal_processor.set_auto_detection(enabled)
                self.logger.info(f"Auto detection {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            self.logger.error(f"Error toggling auto detection: {e}")
    
    def toggle_advanced_analysis(self, enabled):
        """Toggle advanced signal analysis mode."""
        try:
            if hasattr(self, 'signal_processor') and self.signal_processor:
                self.signal_processor.set_advanced_analysis(enabled)
            if self.processing_thread and self.processing_thread.signal_processor:
                self.processing_thread.signal_processor.set_advanced_analysis(enabled)
                self.logger.info(f"Advanced analysis {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            self.logger.error(f"Error toggling advanced analysis: {e}")
    
    def handle_signal_analysis_request(self, analysis_request: Dict[str, Any]):
        """Handle signal analysis request from spectrum widget."""
        request_id = None
        try:
            queue_entry = self._enqueue_roi_analysis_request(analysis_request)
            if not queue_entry:
                self.logger.info("Debounced duplicate ROI analysis request")
                return
            request_id = str(queue_entry.get('request_id', ''))
            self._update_roi_queue_status(request_id, 'running', result_note='ROI request accepted')
            self._persist_roi_preset_from_request(analysis_request)

            self.logger.info(f"Processing signal analysis request for {analysis_request['center_freq']/1e6:.3f} MHz")
            
            # Get IQ data for the specified frequency range
            range_result = self._get_iq_data_for_range(
                analysis_request['center_freq'],
                analysis_request['bandwidth'],
                analysis_request.get('freq_range'),
                return_sample_rate=True,
            )

            if isinstance(range_result, tuple) and len(range_result) == 2:
                iq_data, analysis_sample_rate = range_result
            else:
                iq_data = range_result
                analysis_sample_rate = float(self.settings.sdr.sample_rate)
            
            if iq_data is None or len(iq_data) == 0:
                self.logger.warning("No IQ data available for analysis")
                self._update_roi_queue_status(request_id, 'failed', result_note='No IQ data available')
                return

            source_sample_rate = float(getattr(self.settings.sdr, 'sample_rate', analysis_sample_rate))
            req_bandwidth = float(analysis_request.get('bandwidth', 0.0) or 0.0)
            freq_range = analysis_request.get('freq_range')
            roi_start = float(analysis_request.get('center_freq', 0.0) - req_bandwidth * 0.5)
            roi_end = float(analysis_request.get('center_freq', 0.0) + req_bandwidth * 0.5)
            if isinstance(freq_range, (tuple, list)) and len(freq_range) == 2:
                roi_start = float(freq_range[0])
                roi_end = float(freq_range[1])
                if roi_start > roi_end:
                    roi_start, roi_end = roi_end, roi_start

            decimation_factor = 1.0
            if np.isfinite(source_sample_rate) and np.isfinite(float(analysis_sample_rate)) and float(analysis_sample_rate) > 0.0:
                decimation_factor = max(1.0, source_sample_rate / float(analysis_sample_rate))

            stage_status = self._build_stage_status_envelope(
                is_demo_mode=bool(getattr(self, 'demo_mode', False)),
                has_iq_data=bool(iq_data is not None and len(iq_data) > 0),
                roi_bandwidth_hz=max(0.0, roi_end - roi_start),
                decimation_factor=float(decimation_factor),
            )

            analysis_context = {
                'roi_freq_start_hz': roi_start,
                'roi_freq_end_hz': roi_end,
                'roi_bandwidth_hz': max(0.0, roi_end - roi_start),
                'source_sample_rate_hz': source_sample_rate,
                'analysis_sample_rate_hz': float(analysis_sample_rate),
                'decimation_factor': float(decimation_factor),
                'iq_samples_used': int(len(iq_data)),
                'roi_request_id': request_id,
                'stage_status': stage_status,
                'stage_schema_version': '1.0',
            }
            
            if getattr(self, 'enable_async_analysis_requests', False):
                if self.analysis_request_thread and self.analysis_request_thread.isRunning():
                    self.logger.warning("Previous analysis request still running; skipping new request")
                    self._update_roi_queue_status(request_id, 'skipped', result_note='Previous analysis still running')
                    return

                self.analysis_request_thread = AnalysisRequestThread(
                    analysis_sample_rate,
                    iq_data,
                    analysis_request['center_freq'],
                    analysis_request['bandwidth'],
                    analysis_context,
                )
                self.analysis_request_thread.completed.connect(self._on_async_analysis_completed)
                self.analysis_request_thread.error_occurred.connect(
                    lambda msg: self.logger.error(f"Async analysis request error: {msg}")
                )
                self.analysis_request_thread.start()
            else:
                self._analyze_signal_request_sync(
                    iq_data,
                    analysis_request['center_freq'],
                    analysis_request['bandwidth'],
                    analysis_sample_rate,
                    analysis_context,
                )
                
        except Exception as e:
            self.logger.error(f"Error handling signal analysis request: {e}")
            if request_id:
                self._update_roi_queue_status(request_id, 'failed', result_note=str(e))

    def _build_roi_request_signature(self, analysis_request: Dict[str, Any]) -> str:
        """Create a compact signature to debounce repeated ROI requests."""
        if not isinstance(analysis_request, dict):
            return 'invalid'

        center = float(analysis_request.get('center_freq', 0.0) or 0.0)
        bandwidth = float(analysis_request.get('bandwidth', 0.0) or 0.0)
        freq_range = analysis_request.get('freq_range')

        f1 = center - bandwidth * 0.5
        f2 = center + bandwidth * 0.5
        if isinstance(freq_range, (tuple, list)) and len(freq_range) == 2:
            f1 = float(freq_range[0])
            f2 = float(freq_range[1])
            if f1 > f2:
                f1, f2 = f2, f1

        return f"{center:.3f}|{bandwidth:.3f}|{f1:.3f}|{f2:.3f}"

    def _enqueue_roi_analysis_request(self, analysis_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Track ROI requests and debounce immediate duplicates."""
        signature = self._build_roi_request_signature(analysis_request)
        now_mono = time.monotonic()
        if (
            signature == self._last_roi_request_signature
            and (now_mono - self._last_roi_request_mono) < self._roi_request_debounce_sec
        ):
            return None

        self._last_roi_request_signature = signature
        self._last_roi_request_mono = now_mono
        entry = {
            'request_id': str(uuid4()),
            'signature': signature,
            'status': 'queued',
            'queued_at_utc': datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
            'request': dict(analysis_request or {}),
        }
        self.roi_analysis_queue.append(entry)
        self._sync_roi_queue_panel()
        return entry

    def _sync_roi_queue_panel(self):
        """Push latest ROI queue snapshot to main-window ROI panel if available."""
        try:
            if self.main_window and hasattr(self.main_window, 'update_roi_queue_panel'):
                self.main_window.update_roi_queue_panel(list(self.roi_analysis_queue))
        except Exception as exc:
            self.logger.debug(f"ROI panel sync skipped: {exc}")

    def _update_roi_queue_status(
        self,
        request_id: Optional[str],
        status: str,
        result_note: str = '',
        modulation: Optional[str] = None,
        snr_db: Optional[float] = None,
    ):
        """Update tracked ROI request state and mirror it to GUI panel."""
        if not request_id:
            return

        for entry in reversed(self.roi_analysis_queue):
            if str(entry.get('request_id', '')) != str(request_id):
                continue
            entry['status'] = str(status or 'unknown')
            entry['updated_at_utc'] = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
            if result_note:
                entry['result_note'] = str(result_note)
            if modulation:
                entry['modulation'] = str(modulation)
            if snr_db is not None and np.isfinite(float(snr_db)):
                entry['snr_db'] = float(snr_db)
            break

        self._sync_roi_queue_panel()

    def _persist_roi_preset_from_request(self, analysis_request: Dict[str, Any]):
        """Persist ROI presets in settings when a user-triggered analysis is requested."""
        try:
            if not hasattr(self.settings, 'roi'):
                return

            center = float(analysis_request.get('center_freq', 0.0) or 0.0)
            bandwidth = float(analysis_request.get('bandwidth', 0.0) or 0.0)
            freq_range = analysis_request.get('freq_range')

            f1 = center - max(bandwidth, 1.0) * 0.5
            f2 = center + max(bandwidth, 1.0) * 0.5
            if isinstance(freq_range, (tuple, list)) and len(freq_range) == 2:
                f1 = float(freq_range[0])
                f2 = float(freq_range[1])
                if f1 > f2:
                    f1, f2 = f2, f1

            if not (np.isfinite(f1) and np.isfinite(f2) and f2 > f1):
                return

            now_utc = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
            presets = list(getattr(self.settings.roi, 'presets', []) or [])

            matched = None
            for item in presets:
                try:
                    p1 = float(item.get('freq_start_hz', np.nan))
                    p2 = float(item.get('freq_end_hz', np.nan))
                except Exception:
                    continue
                if np.isfinite(p1) and np.isfinite(p2) and abs(p1 - f1) <= 10.0 and abs(p2 - f2) <= 10.0:
                    matched = item
                    break

            if matched is None:
                next_idx = len(presets) + 1
                matched = {
                    'name': f'ROI {next_idx}',
                    'freq_start_hz': float(f1),
                    'freq_end_hz': float(f2),
                    'center_hz': float((f1 + f2) * 0.5),
                    'bandwidth_hz': float(f2 - f1),
                    'usage_count': 0,
                    'last_used_utc': now_utc,
                    'created_utc': now_utc,
                }
                presets.append(matched)
            else:
                matched['freq_start_hz'] = float(f1)
                matched['freq_end_hz'] = float(f2)
                matched['center_hz'] = float((f1 + f2) * 0.5)
                matched['bandwidth_hz'] = float(f2 - f1)
                matched['last_used_utc'] = now_utc

            matched['usage_count'] = int(matched.get('usage_count', 0) or 0) + 1
            presets = sorted(
                presets,
                key=lambda p: (str(p.get('last_used_utc', '')), int(p.get('usage_count', 0) or 0)),
                reverse=True,
            )[: self._roi_preset_max_count]

            self.settings.roi.presets = presets
            self.settings.roi.last_selected_preset = str(matched.get('name', ''))
            self.settings.save()
        except Exception as exc:
            self.logger.debug(f"ROI preset persistence skipped: {exc}")

    def _analyze_signal_request_sync(
        self,
        iq_data: np.ndarray,
        center_freq: float,
        bandwidth: float,
        analysis_sample_rate: Optional[float] = None,
        analysis_context: Optional[Dict[str, Any]] = None,
    ):
        """Synchronous analysis path used in tests or fallback modes."""
        request_id = None
        if isinstance(analysis_context, dict):
            request_id = analysis_context.get('roi_request_id')

        if not self.signal_analyzer and analysis_sample_rate is None:
            self.logger.error("Signal analyzer not initialized")
            self._update_roi_queue_status(request_id, 'failed', result_note='Signal analyzer not initialized')
            return

        requested_sr = float(analysis_sample_rate or self.settings.sdr.sample_rate)
        analyzer = self.signal_analyzer
        analyzer_sr = float(getattr(analyzer, 'sample_rate', requested_sr)) if analyzer else requested_sr
        if analyzer is None or abs(analyzer_sr - requested_sr) > 1e-3:
            analyzer = SignalAnalyzer(requested_sr)

        analysis_results = analyzer.analyze_signal_comprehensive(
            iq_data,
            center_freq,
            bandwidth,
        )
        analysis_results = self._attach_analysis_context(analysis_results, analysis_context)
        payload = analysis_results.get('payload', analysis_results)
        if not analysis_results.get('success', True):
            self.logger.warning(
                f"Signal analysis failed: {analysis_results.get('error', 'Unknown error')}"
            )
            self._update_roi_queue_status(
                request_id,
                'failed',
                result_note=str(analysis_results.get('error', 'Unknown error')),
            )
            return

        self._update_gui_with_analysis_results(payload)
        snr_db = payload.get('demodulation', {}).get('snr') if isinstance(payload, dict) else None
        modulation = payload.get('modulation', {}).get('type') if isinstance(payload, dict) else None
        self._update_roi_queue_status(
            request_id,
            'completed',
            result_note='Analysis completed',
            modulation=modulation,
            snr_db=snr_db,
        )
        self.logger.info(
            f"Signal analysis completed: {payload.get('modulation', {}).get('type', 'Unknown')} detected"
        )

    def _on_async_analysis_completed(self, analysis_results: Dict[str, Any]):
        """Handle completion of async analysis requests."""
        try:
            payload = analysis_results.get('payload', analysis_results)
            analysis_context = payload.get('analysis_context', {}) if isinstance(payload, dict) else {}
            request_id = analysis_context.get('roi_request_id') if isinstance(analysis_context, dict) else None
            if not analysis_results.get('success', True):
                self.logger.warning(
                    f"Signal analysis failed: {analysis_results.get('error', 'Unknown error')}"
                )
                self._update_roi_queue_status(
                    request_id,
                    'failed',
                    result_note=str(analysis_results.get('error', 'Unknown error')),
                )
                return
            self._update_gui_with_analysis_results(payload)
            snr_db = payload.get('demodulation', {}).get('snr') if isinstance(payload, dict) else None
            modulation = payload.get('modulation', {}).get('type') if isinstance(payload, dict) else None
            self._update_roi_queue_status(
                request_id,
                'completed',
                result_note='Analysis completed',
                modulation=modulation,
                snr_db=snr_db,
            )
            self.logger.info(
                f"Signal analysis completed: {payload.get('modulation', {}).get('type', 'Unknown')} detected"
            )
        except Exception as exc:
            self.logger.error(f"Error handling async analysis completion: {exc}")

    def _attach_analysis_context(
        self,
        analysis_results: Dict[str, Any],
        analysis_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Attach analysis context for both legacy and envelope payload formats."""
        if not isinstance(analysis_results, dict):
            return analysis_results
        if not isinstance(analysis_context, dict) or len(analysis_context) == 0:
            return analysis_results

        payload = analysis_results.get('payload')
        if isinstance(payload, dict):
            payload['analysis_context'] = dict(analysis_context)
            return analysis_results

        analysis_results['analysis_context'] = dict(analysis_context)
        return analysis_results

    def _build_stage_status_envelope(
        self,
        is_demo_mode: bool,
        has_iq_data: bool,
        roi_bandwidth_hz: float,
        decimation_factor: float,
    ) -> Dict[str, Any]:
        """Build baseline pipeline stage tags used to track analysis progression."""
        capture_state = 'success' if has_iq_data else 'failed'
        ingest_state = 'success' if has_iq_data else 'failed'
        preprocess_state = 'success' if has_iq_data else 'failed'

        source_note = 'demo_iq' if is_demo_mode else 'live_or_snapshot_iq'
        preprocessing_note = 'roi_extract_and_decimate' if decimation_factor > 1.01 else 'roi_extract_only'

        stages = {
            'capture': {
                'state': capture_state,
                'note': source_note,
            },
            'ingest': {
                'state': ingest_state,
                'note': 'ring_buffer_snapshot',
            },
            'preprocess': {
                'state': preprocess_state,
                'note': preprocessing_note,
            },
            'detect': {
                'state': 'pending' if has_iq_data else 'blocked',
                'note': 'executed_in_signal_analyzer',
            },
            'characterize': {
                'state': 'pending' if has_iq_data else 'blocked',
                'note': 'executed_in_signal_analyzer',
            },
            'demodulate': {
                'state': 'pending' if has_iq_data else 'blocked',
                'note': 'executed_in_signal_analyzer',
            },
            'dechannelize': {
                'state': 'pending' if has_iq_data else 'blocked',
                'note': 'phase_5_target',
            },
            'deinterleave_descramble': {
                'state': 'pending' if has_iq_data else 'blocked',
                'note': 'phase_6_target',
            },
            'fec_decode': {
                'state': 'pending' if has_iq_data else 'blocked',
                'note': 'phase_7_target',
            },
            'payload_parse': {
                'state': 'pending' if has_iq_data else 'blocked',
                'note': 'phase_8_target',
            },
            'output_render': {
                'state': 'pending' if has_iq_data else 'blocked',
                'note': 'phase_8_target',
            },
        }

        return {
            'current_stage': 'detect' if has_iq_data else 'capture',
            'roi_bandwidth_hz': float(roi_bandwidth_hz),
            'decimation_factor': float(decimation_factor),
            'stages': stages,
        }
    
    def _extract_channel_iq(
        self,
        iq_data: np.ndarray,
        source_center_hz: float,
        target_center_hz: float,
        target_bandwidth_hz: float,
        sample_rate_hz: float,
    ) -> np.ndarray:
        """Extract selected frequency channel from wideband IQ.

        This mirrors SDR workflows used in tools like SatDump:
        tune wideband -> select ROI -> shift ROI to baseband -> low-pass filter.
        """
        if iq_data is None or len(iq_data) == 0:
            return np.array([], dtype=np.complex64)

        fs = float(sample_rate_hz)
        if not np.isfinite(fs) or fs <= 0.0:
            return np.asarray(iq_data, dtype=np.complex64)

        bw = float(target_bandwidth_hz)
        if not np.isfinite(bw) or bw <= 0.0:
            bw = fs * 0.1

        # Keep bandwidth inside Nyquist and avoid degenerate filters.
        bw = min(max(bw, fs / 500.0), fs * 0.95)

        iq = np.asarray(iq_data, dtype=np.complex64)

        # Translate selected channel center to DC.
        freq_offset = float(target_center_hz - source_center_hz)
        if abs(freq_offset) > 0.0:
            n = np.arange(len(iq), dtype=np.float64)
            lo = np.exp(-1j * 2.0 * np.pi * freq_offset * (n / fs))
            iq = (iq * lo).astype(np.complex64, copy=False)

        cutoff_hz = min(bw * 0.55, fs * 0.45)
        if cutoff_hz <= 0.0:
            return iq

        if sp_signal is not None:
            # 6th order Butterworth LPF: smooth enough for interactive ROI analysis.
            b, a = sp_signal.butter(6, cutoff_hz / (fs * 0.5), btype='low')
            filtered = sp_signal.filtfilt(b, a, iq)
            return np.asarray(filtered, dtype=np.complex64)

        # Fallback when scipy is unavailable: FFT brick-wall LPF.
        spectrum = np.fft.fft(iq)
        freqs = np.fft.fftfreq(len(iq), d=1.0 / fs)
        mask = np.abs(freqs) <= cutoff_hz
        filtered = np.fft.ifft(spectrum * mask)
        return np.asarray(filtered, dtype=np.complex64)

    def _decimate_roi_if_narrow(
        self,
        iq_data: np.ndarray,
        sample_rate_hz: float,
        roi_bandwidth_hz: float,
    ) -> (np.ndarray, float):
        """Downsample narrow-band ROI to reduce analysis CPU load."""
        iq = np.asarray(iq_data, dtype=np.complex64)
        fs = float(sample_rate_hz)
        bw = float(roi_bandwidth_hz)

        if iq.size == 0 or not np.isfinite(fs) or fs <= 0.0:
            return iq, fs
        if not np.isfinite(bw) or bw <= 0.0:
            return iq, fs

        # Keep at least ~4x BW for robust demod/analysis while cutting compute cost.
        target_fs = max(48_000.0, bw * 4.0)
        if fs <= target_fs * 1.15:
            return iq, fs
        if sp_signal is None:
            return iq, fs

        decim = int(np.floor(fs / target_fs))
        decim = max(1, min(decim, 32))
        if decim <= 1:
            return iq, fs

        if len(iq) // decim < 256:
            return iq, fs

        try:
            decimated = sp_signal.resample_poly(iq, up=1, down=decim)
            out = np.asarray(decimated, dtype=np.complex64)
            out_fs = fs / decim
            return out, out_fs
        except Exception as exc:
            self.logger.debug(f"ROI decimation skipped due to error: {exc}")
            return iq, fs

    def _get_iq_data_for_range(
        self,
        center_freq: float,
        bandwidth: float,
        freq_range: Optional[Any] = None,
        return_sample_rate: bool = False,
    ):
        """Get IQ data for specific frequency range."""
        try:
            if self.demo_mode:
                # Generate synthetic IQ data for demo
                demo_iq = self._generate_demo_iq_data(center_freq, bandwidth)
                if return_sample_rate:
                    return demo_iq, float(self.settings.sdr.sample_rate)
                return demo_iq

            # Snapshot path: avoid synchronous retune/read in UI thread.
            snapshot_ring = getattr(self, 'iq_snapshot_ring', None)
            if not snapshot_ring:
                return None

            required_samples = int(self.settings.sdr.sample_rate * 0.1)
            chunks = []
            collected = 0
            for chunk in reversed(snapshot_ring):
                chunks.append(chunk)
                collected += len(chunk)
                if collected >= required_samples:
                    break

            if not chunks:
                return None

            iq_data = np.concatenate(list(reversed(chunks))).astype(np.complex64)
            if len(iq_data) > required_samples:
                iq_data = iq_data[-required_samples:]

            source_center_hz = float(getattr(self.settings.sdr, 'center_frequency', center_freq))
            target_center_hz = float(center_freq)
            target_bandwidth_hz = float(bandwidth)

            # Prefer explicit ROI limits from spectrum region when available.
            if isinstance(freq_range, (tuple, list)) and len(freq_range) == 2:
                f1 = float(freq_range[0])
                f2 = float(freq_range[1])
                if f1 > f2:
                    f1, f2 = f2, f1
                if np.isfinite(f1) and np.isfinite(f2) and f2 > f1:
                    target_center_hz = 0.5 * (f1 + f2)
                    target_bandwidth_hz = f2 - f1

            extracted = self._extract_channel_iq(
                iq_data=iq_data,
                source_center_hz=source_center_hz,
                target_center_hz=target_center_hz,
                target_bandwidth_hz=target_bandwidth_hz,
                sample_rate_hz=float(self.settings.sdr.sample_rate),
            )
            optimized_iq, optimized_fs = self._decimate_roi_if_narrow(
                extracted,
                float(self.settings.sdr.sample_rate),
                target_bandwidth_hz,
            )

            if return_sample_rate:
                return optimized_iq, float(optimized_fs)
            return optimized_iq
            
        except Exception as e:
            self.logger.error(f"Error getting IQ data for range: {e}")
            if return_sample_rate:
                return None, float(self.settings.sdr.sample_rate)
            return None
    
    def _generate_demo_iq_data(self, center_freq: float, bandwidth: float) -> np.ndarray:
        """Generate synthetic IQ data for demo mode."""
        try:
            duration = 0.1  # 100ms
            sample_rate = self.settings.sdr.sample_rate
            num_samples = int(sample_rate * duration)
            
            # Generate time array
            t = np.linspace(0, duration, num_samples)
            
            # Generate different modulation types based on frequency
            freq_mhz = center_freq / 1e6
            
            if 100 <= freq_mhz <= 200:
                # Generate BPSK signal
                symbol_rate = 1000  # 1 kHz symbol rate
                data_bits = np.random.randint(0, 2, int(duration * symbol_rate))
                symbols = 2 * data_bits - 1  # Convert to +1/-1
                
                # Upsample symbols
                samples_per_symbol = int(sample_rate / symbol_rate)
                upsampled = np.repeat(symbols, samples_per_symbol)[:num_samples]
                
                # Add carrier and noise
                carrier_freq = 1000  # 1 kHz offset
                iq_data = upsampled * np.exp(1j * 2 * np.pi * carrier_freq * t)
                
            elif 200 <= freq_mhz <= 300:
                # Generate QPSK signal
                symbol_rate = 500
                data_bits = np.random.randint(0, 4, int(duration * symbol_rate))
                
                # Map to QPSK constellation
                symbols = np.exp(1j * np.pi * data_bits / 2)
                
                # Upsample
                samples_per_symbol = int(sample_rate / symbol_rate)
                upsampled = np.repeat(symbols, samples_per_symbol)[:num_samples]
                
                # Add carrier
                carrier_freq = 2000
                iq_data = upsampled * np.exp(1j * 2 * np.pi * carrier_freq * t)
                
            elif 400 <= freq_mhz <= 500:
                # Generate FSK signal
                symbol_rate = 1200
                data_bits = np.random.randint(0, 2, int(duration * symbol_rate))
                
                # FSK frequencies
                freq_0 = 1000  # Frequency for bit 0
                freq_1 = 2000  # Frequency for bit 1
                
                samples_per_symbol = int(sample_rate / symbol_rate)
                iq_data = np.zeros(num_samples, dtype=complex)
                
                for i, bit in enumerate(data_bits):
                    start_idx = i * samples_per_symbol
                    end_idx = min(start_idx + samples_per_symbol, num_samples)
                    
                    if end_idx <= start_idx:
                        break
                        
                    freq = freq_1 if bit else freq_0
                    t_symbol = t[start_idx:end_idx]
                    iq_data[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t_symbol)
                    
            else:
                # Generate white noise
                iq_data = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
            
            # Add noise
            noise_power = 0.1
            noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_power / 2)
            iq_data += noise
            
            return iq_data.astype(np.complex64)
            
        except Exception as e:
            self.logger.error(f"Error generating demo IQ data: {e}")
            # Return white noise as fallback
            num_samples = int(self.settings.sdr.sample_rate * 0.1)
            return (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)).astype(np.complex64)
    
    def _update_gui_with_analysis_results(self, analysis_results: Dict[str, Any]):
        """Update GUI components with signal analysis results."""
        try:
            if not self.main_window:
                return

            # Support both envelope format and legacy flat dictionaries.
            results = analysis_results.get('payload', analysis_results)
            if not isinstance(results, dict):
                return
            if not analysis_results.get('success', True):
                self.logger.warning(f"Signal analysis reported error: {analysis_results.get('error', 'Unknown error')}")
                analysis_context = results.get('analysis_context', {})
                if isinstance(analysis_context, dict):
                    self._update_roi_queue_status(
                        analysis_context.get('roi_request_id'),
                        'failed',
                        result_note=str(analysis_results.get('error', 'Unknown error')),
                    )
                return
            
            # Update constellation widget
            if 'constellation_data' in results and results['constellation_data']['points']:
                constellation_points = np.array(results['constellation_data']['points'])
                if len(constellation_points) > 0:
                    # Extract modulation info for the widget
                    modulation_info = {
                        'type': results['modulation']['type'],
                        'confidence': results['modulation']['confidence']
                    }
                    self.main_window.constellation_widget.update_constellation(
                        constellation_points,
                        None,  # symbols (optional)
                        modulation_info
                    )
            
            # Update bitstream widget
            if results.get('demodulation', {}).get('success') and results.get('coding'):
                if results['coding']['decoded_bits'] is not None:
                    decoded_bits = np.array(results['coding']['decoded_bits'])
                    # Convert to list of integers for the widget
                    if decoded_bits.dtype == bool:
                        bits = decoded_bits.astype(int).tolist()
                    else:
                        bits = np.clip(decoded_bits, 0, 1).astype(int).tolist()
                    self.main_window.bitstream_widget.add_bits(bits)

            artifacts = list(results.get('decoded_outputs', []) or [])
            protocol_artifacts = results.get('protocol_outputs', {}).get('artifacts', []) if isinstance(results.get('protocol_outputs', {}), dict) else []
            artifacts.extend(protocol_artifacts or [])

            image_artifact = next((a for a in artifacts if a.get('type') == 'image'), None)
            pcm_artifact = next((a for a in artifacts if a.get('type') == 'pcm'), None)
            image_summary = None
            if image_artifact:
                self.latest_image_artifact = image_artifact
                image_summary = image_artifact.get('payload', {}).get('summary', {})
                if hasattr(self.main_window, 'update_image_artifact_view'):
                    self.main_window.update_image_artifact_view(image_artifact)
            if pcm_artifact:
                self.latest_pcm_artifact = pcm_artifact

            self._record_analysis_snapshot(results, image_artifact)
            
            # Update info display
            info_text = f"Modulation: {results['modulation']['type']} " \
                       f"(Confidence: {results['modulation']['confidence']:.2f})"
            
            if results['demodulation']['snr'] is not None:
                info_text += f", SNR: {results['demodulation']['snr']:.1f} dB"
            
            if 'coding' in results and results['coding']:
                info_text += f", Coding: {results['coding']['coding_type']}"

            if image_summary:
                width = image_summary.get('width')
                height = image_summary.get('height')
                if width is not None and height is not None:
                    info_text += f", NOAA Image: {int(width)}x{int(height)} ready"

            analysis_context = results.get('analysis_context', {}) if isinstance(results, dict) else {}
            if isinstance(analysis_context, dict) and analysis_context:
                analysis_fs = analysis_context.get('analysis_sample_rate_hz')
                decimation = analysis_context.get('decimation_factor')
                if analysis_fs is not None:
                    info_text += f", Fs: {float(analysis_fs)/1e3:.1f} kS/s"
                if decimation is not None:
                    info_text += f", Decim: x{float(decimation):.1f}"
            
            # Display in spectrum widget info label
            self.main_window.spectrum_widget.info_label.setText(info_text)
            
            self.logger.info(f"GUI updated with analysis results: {info_text}")
            
        except Exception as e:
            self.logger.error(f"Error updating GUI with analysis results: {e}")

    def _record_analysis_snapshot(self, results: Dict[str, Any], image_artifact: Optional[Dict[str, Any]] = None):
        """Store compact per-analysis snapshot for session-level reporting."""
        try:
            modulation = results.get('modulation', {}) if isinstance(results, dict) else {}
            demod = results.get('demodulation', {}) if isinstance(results, dict) else {}
            decode_quality = results.get('decode_quality', {}) if isinstance(results, dict) else {}
            protocol_outputs = results.get('protocol_outputs', {}) if isinstance(results, dict) else {}
            decode_depth = results.get('decode_depth', {}) if isinstance(results, dict) else {}

            artifact_refs = []
            for artifact in list(results.get('decoded_outputs', []) or []):
                payload = artifact.get('payload', {}) if isinstance(artifact, dict) else {}
                artifact_refs.append(
                    {
                        'type': artifact.get('type'),
                        'confidence': artifact.get('confidence'),
                        'protocol': payload.get('protocol') if isinstance(payload, dict) else None,
                    }
                )

            if image_artifact:
                payload = image_artifact.get('payload', {})
                summary = payload.get('summary', {}) if isinstance(payload, dict) else {}
                image_ref = {
                    'type': 'image',
                    'protocol': payload.get('protocol') if isinstance(payload, dict) else None,
                    'width': summary.get('width') if isinstance(summary, dict) else None,
                    'height': summary.get('height') if isinstance(summary, dict) else None,
                    'preview_rows': payload.get('preview_rows') if isinstance(payload, dict) else None,
                }
                if image_ref not in artifact_refs:
                    artifact_refs.append(image_ref)

            record = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'modulation_type': modulation.get('type'),
                'modulation_confidence': modulation.get('confidence'),
                'snr': demod.get('snr'),
                'decode_quality': {
                    'bit_count': decode_quality.get('bit_count'),
                    'artifact_count': decode_quality.get('artifact_count'),
                    'frame_count': decode_quality.get('frame_count'),
                    'uncertain_frame_ratio': decode_quality.get('uncertain_frame_ratio'),
                    'ber': decode_quality.get('ber'),
                    'per': decode_quality.get('per'),
                    'crc_ok_rate': decode_quality.get('crc_ok_rate'),
                    'frame_lock_ratio': decode_quality.get('frame_lock_ratio'),
                },
                'stage_metrics': {
                    'decode_depth': decode_depth,
                    'protocol': {
                        'matched_protocol': protocol_outputs.get('matched_protocol') if isinstance(protocol_outputs, dict) else None,
                        'confidence': protocol_outputs.get('confidence') if isinstance(protocol_outputs, dict) else None,
                    },
                },
                'artifact_references': artifact_refs,
            }

            self.analysis_session_records.append(record)
        except Exception as exc:
            self.logger.debug(f"Failed to record analysis snapshot: {exc}")

    def export_latest_image_artifact(self, filename: str):
        """Export the most recent decoded image artifact from analysis results."""
        try:
            if not self.latest_image_artifact:
                if self.main_window:
                    self.main_window.show_error_message("No decoded image artifact available to export.")
                return

            payload = self.latest_image_artifact.get('payload', {})
            summary = payload.get('summary', {}) if isinstance(payload, dict) else {}
            self.data_exporter.set_metadata(
                {
                    "protocol": payload.get('protocol', 'unknown') if isinstance(payload, dict) else 'unknown',
                    "width": summary.get('width'),
                    "height": summary.get('height'),
                }
            )

            success = self.data_exporter.export_artifact_image(self.latest_image_artifact, filename)
            if success:
                self.logger.info(f"Decoded image artifact exported: {filename}")
                if self.main_window:
                    self.main_window.show_info_message(f"Exported decoded image: {filename}")
            else:
                self.logger.error(f"Decoded image artifact export failed: {filename}")
                if self.main_window:
                    self.main_window.show_error_message("Failed to export decoded image artifact.")

        except Exception as exc:
            self.logger.error(f"Error exporting decoded image artifact: {exc}")
            if self.main_window:
                self.main_window.show_error_message(f"Export error: {exc}")

    def export_session_decode_report(self, filename: str):
        """Export session decode report with per-stage trends and artifact references."""
        try:
            records = list(self.analysis_session_records)
            if not records:
                if self.main_window:
                    self.main_window.show_error_message("No session analysis records available for export.")
                return

            success = self.data_exporter.export_decode_session_report(records=records, filename=filename)
            if success:
                self.logger.info(f"Decode session report exported: {filename}")
                if self.main_window:
                    self.main_window.show_info_message(f"Exported decode session report: {filename}")
            else:
                self.logger.error(f"Decode session report export failed: {filename}")
                if self.main_window:
                    self.main_window.show_error_message("Failed to export decode session report.")
        except Exception as exc:
            self.logger.error(f"Error exporting decode session report: {exc}")
            if self.main_window:
                self.main_window.show_error_message(f"Export error: {exc}")

    def export_latest_pcm_artifact(self, filename: str):
        """Export latest PCM artifact as WAV."""
        try:
            if not self.latest_pcm_artifact:
                if self.main_window:
                    self.main_window.show_error_message("No decoded PCM artifact available to export.")
                return

            success = self.data_exporter.export_pcm_wav_from_artifact(self.latest_pcm_artifact, filename)
            if success:
                self.logger.info(f"Decoded PCM artifact exported: {filename}")
                if self.main_window:
                    self.main_window.show_info_message(f"Exported decoded audio: {filename}")
            else:
                self.logger.error(f"Decoded PCM artifact export failed: {filename}")
                if self.main_window:
                    self.main_window.show_error_message("Failed to export decoded audio artifact.")
        except Exception as exc:
            self.logger.error(f"Error exporting decoded PCM artifact: {exc}")
            if self.main_window:
                self.main_window.show_error_message(f"Export error: {exc}")

    def load_session_decode_report(self, filename: str):
        """Load and replay a previously exported decode session report."""
        try:
            payload = self.data_importer.import_decode_session_report(filename)
            records = payload.get('records', []) if isinstance(payload, dict) else []
            if not records:
                if self.main_window:
                    self.main_window.show_error_message("No records found in decode session report.")
                return

            self.analysis_session_records.clear()
            self.analysis_session_records.extend(records)

            if self.main_window and hasattr(self.main_window, 'clear_image_artifact_history'):
                self.main_window.clear_image_artifact_history()

            replayed_images = 0
            for record in records:
                for ref in record.get('artifact_references', []) or []:
                    if ref.get('type') != 'image':
                        continue
                    preview_rows = ref.get('preview_rows')
                    if not preview_rows:
                        continue
                    artifact = {
                        'type': 'image',
                        'confidence': 0.0,
                        'payload': {
                            'protocol': ref.get('protocol', 'unknown'),
                            'summary': {
                                'width': ref.get('width'),
                                'height': ref.get('height'),
                            },
                            'preview_rows': preview_rows,
                        },
                    }
                    if self.main_window and hasattr(self.main_window, 'update_image_artifact_view'):
                        self.main_window.update_image_artifact_view(artifact)
                        replayed_images += 1

            if self.main_window:
                self.main_window.show_info_message(
                    f"Loaded decode session report with {len(records)} records and replayed {replayed_images} image previews."
                )
            self.logger.info(f"Loaded decode session report: {filename}")
        except Exception as exc:
            self.logger.error(f"Error loading decode session report: {exc}")
            if self.main_window:
                self.main_window.show_error_message(f"Load error: {exc}")

    def process_signal_file(
        self,
        filename: str,
        center_freq: float = 0.0,
        bandwidth: float = 0.0,
        advanced_params: Optional[Dict[str, float]] = None,
    ):
        """Load a signal file and run the full analysis pipeline in a background thread."""
        from PySide6.QtWidgets import QProgressDialog
        from PySide6.QtCore import Qt

        advanced_params = advanced_params or {}

        # Persist last-used advanced file-source parameters for next file open.
        try:
            if hasattr(self.settings, 'file_source'):
                self.settings.file_source.sample_rate_hz = float(advanced_params.get('sample_rate_hz', 0.0) or 0.0)
                self.settings.file_source.freq_offset_hz = float(advanced_params.get('freq_offset_hz', 0.0) or 0.0)
                self.settings.file_source.start_sec = float(advanced_params.get('start_sec', 0.0) or 0.0)
                self.settings.file_source.duration_sec = float(advanced_params.get('duration_sec', 0.0) or 0.0)
                self.settings.save()
        except Exception as exc:
            self.logger.warning(f"Could not persist file-source advanced settings: {exc}")

        if center_freq == 0.0:
            center_freq = float(self.settings.sdr.center_frequency)
        if bandwidth == 0.0:
            bandwidth = float(max(self.settings.sdr.sample_rate / 2.0, 1.0))

        # For file analysis, pause live SDR acquisition to avoid UI starvation.
        self._resume_acquisition_after_file = bool(
            self.acquisition_thread and self.acquisition_thread.isRunning()
        )
        if self._resume_acquisition_after_file:
            self.stop_acquisition()

        # Headless mode: run synchronously and return dict for CLI flow.
        if self.headless_mode or not self.main_window:
            try:
                iq_data, metadata = self.data_importer.import_signal_source(filename)
                if iq_data is None or len(iq_data) == 0:
                    return {'success': False, 'error': f'Failed to import signal source from {filename}'}

                sample_rate = float(metadata.get('sample_rate') or self.settings.sdr.sample_rate)
                worker = FileAnalysisWorker(
                    self.data_importer,
                    SignalAnalyzer,
                    self.settings,
                    filename,
                    center_freq,
                    bandwidth,
                    advanced_params,
                )
                iq_prepared, sample_rate = worker._apply_advanced(iq_data, sample_rate)
                if iq_prepared is None or len(iq_prepared) == 0:
                    return {'success': False, 'error': 'Selected segment is empty after applying advanced file settings'}

                analyzer = SignalAnalyzer(sample_rate)
                result = analyzer.analyze_signal_comprehensive(
                    iq_prepared,
                    center_freq,
                    bandwidth,
                )
                payload = result.get('payload', result)
                payload['source_metadata'] = metadata
                payload['source_file'] = filename
                payload['source_advanced'] = advanced_params
                self.latest_signal_source = metadata
                return result
            finally:
                if self._resume_acquisition_after_file:
                    self.start_acquisition()
                    self._resume_acquisition_after_file = False

        # Stop previous file worker if still alive.
        if self._file_worker is not None and self._file_worker.isRunning():
            self._file_worker.quit()
            self._file_worker.wait(1500)

        self._file_worker = FileAnalysisWorker(
            self.data_importer,
            SignalAnalyzer,
            self.settings,
            filename,
            center_freq,
            bandwidth,
            advanced_params,
        )

        progress = QProgressDialog(
            f"Analysing {Path(filename).name}...",
            "",
            0,
            0,
            self.main_window,
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.show()

        def _cleanup_after_file():
            progress.close()
            if self._resume_acquisition_after_file:
                self.start_acquisition()
                self._resume_acquisition_after_file = False

        def _on_done(result):
            _cleanup_after_file()
            payload = result.get('payload', result)
            self.latest_signal_source = payload.get('source_metadata', {})
            self._update_gui_with_analysis_results(payload)
            mod = payload.get('modulation', {}).get('type', 'Unknown')
            self.logger.info(f"Processed signal file {filename}: modulation={mod}")
            self.main_window.show_info_message(
                f"Processed: {Path(filename).name} ({payload.get('analysis_status', 'unknown')})"
            )

        def _on_error(msg):
            _cleanup_after_file()
            self.logger.error(f"Error processing signal file {filename}: {msg}")
            self.main_window.show_error_message(f"File processing error: {msg}")

        self._file_worker.analysis_done.connect(_on_done)
        self._file_worker.error_occurred.connect(_on_error)
        self._file_worker.finished.connect(self._file_worker.deleteLater)
        self._file_worker.start()
        return None
    
    def change_detection_threshold(self, threshold_dbm):
        """Change signal detection threshold."""
        try:
            if hasattr(self, 'signal_processor') and self.signal_processor:
                self.signal_processor.set_detection_threshold(threshold_dbm)
                self.logger.info(f"Detection threshold set to {threshold_dbm} dBm")
        except Exception as e:
            self.logger.error(f"Error changing detection threshold: {e}")
    
    def change_detection_interval(self, interval_ms):
        """Change detection check interval."""
        try:
            if hasattr(self, 'signal_processor') and self.signal_processor:
                self.signal_processor.set_detection_interval(interval_ms)
                self.logger.info(f"Detection interval set to {interval_ms} ms")
        except Exception as e:
            self.logger.error(f"Error changing detection interval: {e}")
    
    # Sequential Workflow Methods
    def trigger_sequential_demodulation(self):
        """Trigger demodulation for selected frequency range."""
        try:
            if not hasattr(self, 'analysis_f1') or not hasattr(self, 'analysis_f2'):
                self.logger.warning("No frequency range selected for demodulation")
                return
                
            self.logger.info(f"Starting demodulation for range {self.analysis_f1/1e6:.3f} - {self.analysis_f2/1e6:.3f} MHz")
            
            if hasattr(self, 'signal_processor') and self.signal_processor:
                # Set frequency range for processing
                self.signal_processor.set_analysis_frequency_range(self.analysis_f1, self.analysis_f2)
                
                # Trigger automatic modulation detection and demodulation
                result = self.signal_processor.detect_and_demodulate()
                
                if result and result.get('success', False):
                    self.logger.info(f"Demodulation successful: {result.get('modulation_type', 'Unknown')}")
                    # Update controls widget with demodulation result
                    if hasattr(self, 'main_window') and self.main_window:
                        self.main_window.controls_widget.update_demodulation_result(
                            True, result.get('modulation_type', 'Unknown')
                        )
                else:
                    self.logger.warning("Demodulation failed or no signal detected")
                    if hasattr(self, 'main_window') and self.main_window:
                        self.main_window.controls_widget.update_demodulation_result(False, "No Signal")
                        
        except Exception as e:
            self.logger.error(f"Error in sequential demodulation: {e}")
            if hasattr(self, 'main_window') and self.main_window:
                self.main_window.controls_widget.update_demodulation_result(False, "Error")
    
    def trigger_sequential_decoding(self):
        """Trigger decoding for demodulated signal."""
        try:
            if not hasattr(self, 'signal_processor') or not self.signal_processor:
                self.logger.warning("Signal processor not available for decoding")
                return
                
            self.logger.info("Starting sequential decoding")
            
            # Check if demodulation was successful first
            if not self.signal_processor.has_demodulated_data():
                self.logger.warning("No demodulated data available for decoding")
                if hasattr(self, 'main_window') and self.main_window:
                    self.main_window.controls_widget.update_decoding_result(False, "No Data")
                return
            
            # Trigger automatic channel coding detection and decoding
            result = self.signal_processor.detect_and_decode()
            
            if result and result.get('success', False):
                self.logger.info(f"Decoding successful: {result.get('coding_type', 'Unknown')}")
                # Update controls widget with decoding result
                if hasattr(self, 'main_window') and self.main_window:
                    self.main_window.controls_widget.update_decoding_result(
                        True, result.get('coding_type', 'Unknown')
                    )
            else:
                self.logger.warning("Decoding failed or no coding detected")
                if hasattr(self, 'main_window') and self.main_window:
                    self.main_window.controls_widget.update_decoding_result(False, "No Coding")
                    
        except Exception as e:
            self.logger.error(f"Error in sequential decoding: {e}")
            if hasattr(self, 'main_window') and self.main_window:
                self.main_window.controls_widget.update_decoding_result(False, "Error")
    
    # Frequency Analysis Methods
    def change_frequency_range(self, f1: float, f2: float):
        """Change frequency range for analysis."""
        try:
            # Store frequency range for analysis
            self.analysis_f1 = f1
            self.analysis_f2 = f2
            
            # Update main window with frequency range
            if hasattr(self, 'main_window') and self.main_window:
                self.main_window.set_frequency_range(f1, f2)
                
        except Exception as e:
            print(f"Error changing frequency range: {e}")  # Use print instead of logger
    
    def toggle_center_frequency_lock(self, locked: bool):
        """Toggle center frequency lock."""
        try:
            self.center_freq_locked = locked
            self.logger.info(f"Center frequency {'locked' if locked else 'unlocked'}")
            
            if locked and hasattr(self, 'analysis_f1') and hasattr(self, 'analysis_f2'):
                # Set center frequency to middle of analysis range
                center_freq = (self.analysis_f1 + self.analysis_f2) / 2
                self.change_frequency(center_freq)
                
        except Exception as e:
            self.logger.error(f"Error toggling center frequency lock: {e}")
    
    def change_analysis_bandwidth(self, bandwidth: float):
        """Change analysis bandwidth."""
        try:
            self.analysis_bandwidth = bandwidth
            self.logger.info(f"Analysis bandwidth set to {bandwidth/1e6:.3f} MHz")
            
            # Update sample rate if needed for bandwidth
            if bandwidth > self.settings.sdr.sample_rate:
                self.logger.warning(f"Analysis bandwidth ({bandwidth/1e6:.3f} MHz) exceeds sample rate ({self.settings.sdr.sample_rate/1e6:.3f} MHz)")
                
        except Exception as e:
            self.logger.error(f"Error changing analysis bandwidth: {e}")
    
    def enable_real_time_spectrum(self, enabled: bool, update_rate: int = 10):
        """Enable real-time spectrum display."""
        try:
            if enabled:
                # Update spectrum update rate
                self.settings.gui.spectrum_update_rate = update_rate
                spectrum_interval = int(1000 / update_rate)
                self.spectrum_timer.setInterval(spectrum_interval)
                
                self.logger.info(f"Real-time spectrum enabled at {update_rate} Hz")
            else:
                self.logger.info("Real-time spectrum disabled")
                
        except Exception as e:
            self.logger.error(f"Error enabling real-time spectrum: {e}")
    
    def get_frequency_range_data(self, f1: float, f2: float) -> Optional[np.ndarray]:
        """Get spectrum data for specific frequency range."""
        try:
            if not hasattr(self, '_latest_spectrum') or self._latest_spectrum is None:
                return None
                
            # Calculate frequency axis
            sample_rate = self.settings.sdr.sample_rate
            center_freq = self.settings.sdr.center_frequency
            fft_size = len(self._latest_spectrum)
            
            freq_axis = np.linspace(
                center_freq - sample_rate/2,
                center_freq + sample_rate/2,
                fft_size
            )
            
            # Find indices for frequency range
            idx1 = np.argmin(np.abs(freq_axis - f1))
            idx2 = np.argmin(np.abs(freq_axis - f2))
            
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
                
            return self._latest_spectrum[idx1:idx2+1]
            
        except Exception as e:
            self.logger.error(f"Error getting frequency range data: {e}")
            return None
    
    def set_performance_mode(self, mode: str):
        """Set spectrum processing performance mode.
        
        Args:
            mode: 'fast', 'balanced', or 'quality'
        """
        self._apply_qos_profile(mode)

        if self.signal_processor:
            self.signal_processor.set_performance_mode(mode)
        if self.processing_thread and self.processing_thread.signal_processor:
            self.processing_thread.signal_processor.set_performance_mode(mode)
            self.logger.info(f"Performance mode set to: {mode}")

    def _apply_qos_profile(self, mode: str):
        """Apply cross-layer QoS knobs for DSP and rendering."""
        profiles = {
            'fast': {
                'spectrum_rate': 12,
                'waterfall_rate': 8,
                'advanced_interval_frames': 20,
                'constellation_max_points': 1200,
                'constellation_autoscale_every': 20,
                'bitstream_redraw_ms': 120,
                'bitstream_max_render_bits': 8000,
            },
            'balanced': {
                'spectrum_rate': 20,
                'waterfall_rate': 12,
                'advanced_interval_frames': 10,
                'constellation_max_points': 2000,
                'constellation_autoscale_every': 10,
                'bitstream_redraw_ms': 100,
                'bitstream_max_render_bits': 12000,
            },
            'quality': {
                'spectrum_rate': 30,
                'waterfall_rate': 18,
                'advanced_interval_frames': 5,
                'constellation_max_points': 3500,
                'constellation_autoscale_every': 5,
                'bitstream_redraw_ms': 60,
                'bitstream_max_render_bits': 20000,
            },
        }

        profile = profiles.get(mode)
        if profile is None:
            self.logger.warning(f"Unknown performance mode for QoS profile: {mode}")
            return

        self.settings.gui.spectrum_update_rate = profile['spectrum_rate']
        self.settings.gui.waterfall_update_rate = profile['waterfall_rate']
        self.settings.detection.advanced_analysis_interval_frames = profile['advanced_interval_frames']

        self.spectrum_timer.setInterval(int(1000 / max(1, profile['spectrum_rate'])))
        self.waterfall_timer.setInterval(int(1000 / max(1, profile['waterfall_rate'])))

        if self.main_window and getattr(self.main_window, 'constellation_widget', None):
            cw = self.main_window.constellation_widget
            if hasattr(cw, 'settings'):
                cw.settings['max_points'] = profile['constellation_max_points']
                cw.settings['auto_scale_every_n_frames'] = profile['constellation_autoscale_every']

        if self.main_window and getattr(self.main_window, 'bitstream_widget', None):
            bw = self.main_window.bitstream_widget
            if hasattr(bw, 'max_render_bits'):
                bw.max_render_bits = profile['bitstream_max_render_bits']
            if hasattr(bw, 'update_timer') and hasattr(bw.update_timer, 'setInterval'):
                bw.update_timer.setInterval(profile['bitstream_redraw_ms'])
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics."""
        if self.signal_processor:
            return self.signal_processor.get_performance_stats()
        return {}