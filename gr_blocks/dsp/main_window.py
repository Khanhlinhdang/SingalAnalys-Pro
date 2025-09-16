"""
Main GUI Window for RF Spectrum Analyzer

This module creates the main application window using PySide6 and PyQtGraph.
Provides real-time spectrum display, waterfall plot, IQ constellation,
and comprehensive control panels.

Integrates all signal processing capabilities from the three main libraries.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List
import time

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QGroupBox, QLabel, QSpinBox, QDoubleSpinBox, 
    QComboBox, QPushButton, QSlider, QCheckBox, QProgressBar,
    QStatusBar, QMenuBar, QToolBar, QSplitter, QTextEdit,
    QFrame, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import (
    Qt, QTimer, Signal, QThread, pyqtSignal
)
from PySide6.QtGui import QAction, QFont, QIcon

import pyqtgraph as pg
from pyqtgraph import PlotWidget, ImageView

from config.settings import AppSettings
from utils.logger import get_performance_logger


class SpectrumWidget(QWidget):
    """Spectrum display widget with waterfall"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create spectrum plot
        self.spectrum_plot = PlotWidget(title="RF Spectrum")
        self.spectrum_plot.setLabel('left', 'Power', units='dB')
        self.spectrum_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.spectrum_plot.showGrid(x=True, y=True)

        # Spectrum plot curve
        self.spectrum_curve = self.spectrum_plot.plot(
            pen=pg.mkPen('y', width=2),
            name='Spectrum'
        )

        # Peak markers
        self.peak_markers = []

        # Create waterfall plot
        self.waterfall_view = ImageView()
        self.waterfall_view.ui.roiBtn.hide()
        self.waterfall_view.ui.menuBtn.hide()

        # Waterfall data buffer
        self.waterfall_data = np.zeros((100, 2048))  # 100 time steps, 2048 freq bins
        self.waterfall_index = 0

        # Add to layout
        layout.addWidget(self.spectrum_plot, 2)  # 2/3 of space
        layout.addWidget(self.waterfall_view, 1)  # 1/3 of space

        # Configuration
        self.frequencies = None
        self.center_freq = 433.92e6
        self.sample_rate = 2e6

    def update_spectrum(self, spectrum_data):
        """Update spectrum display with new data"""
        try:
            if spectrum_data is None:
                return

            self.frequencies = spectrum_data.frequencies
            magnitudes = spectrum_data.magnitudes

            # Update spectrum curve
            self.spectrum_curve.setData(self.frequencies, magnitudes)

            # Update waterfall
            self.waterfall_data[self.waterfall_index] = magnitudes
            self.waterfall_index = (self.waterfall_index + 1) % self.waterfall_data.shape[0]

            # Update waterfall display
            self.waterfall_view.setImage(
                self.waterfall_data,
                axes={'t': 0, 'x': 1},
                scale=[1, (self.frequencies[-1] - self.frequencies[0]) / len(self.frequencies)],
                pos=[0, self.frequencies[0]]
            )

        except Exception as e:
            self.logger.error(f"Error updating spectrum: {e}")

    def add_peak_marker(self, freq: float, power: float):
        """Add peak marker to spectrum"""
        try:
            # Remove old markers if too many
            if len(self.peak_markers) > 10:
                old_marker = self.peak_markers.pop(0)
                self.spectrum_plot.removeItem(old_marker)

            # Create new marker
            marker = pg.InfiniteLine(
                angle=90, 
                pos=freq,
                pen=pg.mkPen('r', width=1, style=Qt.DashLine),
                label=f'{freq/1e6:.3f} MHz\n{power:.1f} dB'
            )

            self.spectrum_plot.addItem(marker)
            self.peak_markers.append(marker)

        except Exception as e:
            self.logger.error(f"Error adding peak marker: {e}")

    def clear_markers(self):
        """Clear all peak markers"""
        for marker in self.peak_markers:
            self.spectrum_plot.removeItem(marker)
        self.peak_markers.clear()

    def set_frequency_range(self, center_freq: float, sample_rate: float):
        """Set frequency range for displays"""
        self.center_freq = center_freq
        self.sample_rate = sample_rate

        # Update plot labels
        self.spectrum_plot.setTitle(f"RF Spectrum - Center: {center_freq/1e6:.3f} MHz")


class IQWidget(QWidget):
    """IQ constellation and time domain display"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        # Create layout
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Create constellation plot
        self.constellation_plot = PlotWidget(title="IQ Constellation")
        self.constellation_plot.setLabel('left', 'Q (Quadrature)')
        self.constellation_plot.setLabel('bottom', 'I (In-phase)')
        self.constellation_plot.showGrid(x=True, y=True)
        self.constellation_plot.setAspectLocked(True)

        # Constellation scatter plot
        self.constellation_scatter = pg.ScatterPlotItem(
            size=3,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 200, 100, 120)
        )
        self.constellation_plot.addItem(self.constellation_scatter)

        # Create time domain plot
        self.time_plot = PlotWidget(title="I/Q Time Domain")
        self.time_plot.setLabel('left', 'Amplitude')
        self.time_plot.setLabel('bottom', 'Time', units='s')
        self.time_plot.showGrid(x=True, y=True)

        # Time domain curves
        self.i_curve = self.time_plot.plot(pen=pg.mkPen('r', width=2), name='I')
        self.q_curve = self.time_plot.plot(pen=pg.mkPen('b', width=2), name='Q')
        self.mag_curve = self.time_plot.plot(pen=pg.mkPen('g', width=2), name='|IQ|')

        # Add legend
        self.time_plot.addLegend()

        # Add to layout
        layout.addWidget(self.constellation_plot)
        layout.addWidget(self.time_plot)

        # Data buffers
        self.max_constellation_points = 5000
        self.max_time_samples = 2048

    def update_iq_data(self, iq_samples):
        """Update IQ displays with new samples"""
        try:
            if iq_samples is None or len(iq_samples) == 0:
                return

            # Limit number of samples for performance
            if len(iq_samples) > self.max_constellation_points:
                # Decimate samples for constellation
                decimation = len(iq_samples) // self.max_constellation_points
                constellation_samples = iq_samples[::decimation]
            else:
                constellation_samples = iq_samples

            # Update constellation plot
            i_data = np.real(constellation_samples)
            q_data = np.imag(constellation_samples)

            positions = np.column_stack([i_data, q_data])
            self.constellation_scatter.setData(positions)

            # Update time domain plot
            if len(iq_samples) > self.max_time_samples:
                time_samples = iq_samples[-self.max_time_samples:]
            else:
                time_samples = iq_samples

            # Generate time axis
            dt = 1.0  # Will be updated with actual sample rate
            t = np.arange(len(time_samples)) * dt

            i_time = np.real(time_samples)
            q_time = np.imag(time_samples)
            mag_time = np.abs(time_samples)

            self.i_curve.setData(t, i_time)
            self.q_curve.setData(t, q_time)
            self.mag_curve.setData(t, mag_time)

        except Exception as e:
            self.logger.error(f"Error updating IQ data: {e}")

    def set_sample_rate(self, sample_rate: float):
        """Set sample rate for time axis"""
        # This will be used to properly scale the time axis
        pass


class ControlPanel(QWidget):
    """Main control panel for SDR parameters"""

    # Signals
    frequency_changed = Signal(float)
    sample_rate_changed = Signal(float)
    gain_changed = Signal(float)
    bandwidth_changed = Signal(float)
    device_changed = Signal(dict)
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        self.setup_ui()
        self.connect_signals()
        self.update_from_settings()

    def setup_ui(self):
        """Setup control panel UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Device selection group
        device_group = QGroupBox("Device")
        device_layout = QGridLayout()
        device_group.setLayout(device_layout)

        device_layout.addWidget(QLabel("Device:"), 0, 0)
        self.device_combo = QComboBox()
        device_layout.addWidget(self.device_combo, 0, 1)

        self.refresh_devices_btn = QPushButton("Refresh")
        device_layout.addWidget(self.refresh_devices_btn, 0, 2)

        device_layout.addWidget(QLabel("Status:"), 1, 0)
        self.device_status_label = QLabel("Not connected")
        device_layout.addWidget(self.device_status_label, 1, 1, 1, 2)

        layout.addWidget(device_group)

        # Frequency control group
        freq_group = QGroupBox("Frequency")
        freq_layout = QGridLayout()
        freq_group.setLayout(freq_layout)

        freq_layout.addWidget(QLabel("Center Freq:"), 0, 0)
        self.freq_spinbox = QDoubleSpinBox()
        self.freq_spinbox.setRange(1.0, 6000.0)  # 1 MHz to 6 GHz
        self.freq_spinbox.setValue(433.92)
        self.freq_spinbox.setDecimals(3)
        self.freq_spinbox.setSuffix(" MHz")
        freq_layout.addWidget(self.freq_spinbox, 0, 1)

        freq_layout.addWidget(QLabel("Sample Rate:"), 1, 0)
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems([
            "250 kSps", "1 MSps", "2 MSps", "2.4 MSps", "8 MSps", "10 MSps", "20 MSps"
        ])
        self.sample_rate_combo.setCurrentText("2 MSps")
        freq_layout.addWidget(self.sample_rate_combo, 1, 1)

        freq_layout.addWidget(QLabel("Bandwidth:"), 2, 0)
        self.bandwidth_spinbox = QDoubleSpinBox()
        self.bandwidth_spinbox.setRange(0.1, 60.0)
        self.bandwidth_spinbox.setValue(20.0)
        self.bandwidth_spinbox.setDecimals(1)
        self.bandwidth_spinbox.setSuffix(" MHz")
        freq_layout.addWidget(self.bandwidth_spinbox, 2, 1)

        layout.addWidget(freq_group)

        # Gain control group
        gain_group = QGroupBox("Gain")
        gain_layout = QGridLayout()
        gain_group.setLayout(gain_layout)

        gain_layout.addWidget(QLabel("RF Gain:"), 0, 0)
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 60)
        self.gain_slider.setValue(30)
        gain_layout.addWidget(self.gain_slider, 0, 1)

        self.gain_label = QLabel("30 dB")
        gain_layout.addWidget(self.gain_label, 0, 2)

        self.auto_gain_cb = QCheckBox("Auto Gain")
        gain_layout.addWidget(self.auto_gain_cb, 1, 0, 1, 3)

        layout.addWidget(gain_group)

        # Processing control group
        processing_group = QGroupBox("Processing")
        processing_layout = QGridLayout()
        processing_group.setLayout(processing_layout)

        processing_layout.addWidget(QLabel("FFT Size:"), 0, 0)
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(["512", "1024", "2048", "4096", "8192"])
        self.fft_size_combo.setCurrentText("2048")
        processing_layout.addWidget(self.fft_size_combo, 0, 1)

        processing_layout.addWidget(QLabel("Window:"), 1, 0)
        self.window_combo = QComboBox()
        self.window_combo.addItems(["hanning", "hamming", "blackman", "rectangular"])
        processing_layout.addWidget(self.window_combo, 1, 1)

        processing_layout.addWidget(QLabel("Averaging:"), 2, 0)
        self.averaging_spinbox = QSpinBox()
        self.averaging_spinbox.setRange(1, 100)
        self.averaging_spinbox.setValue(10)
        processing_layout.addWidget(self.averaging_spinbox, 2, 1)

        layout.addWidget(processing_group)

        # Control buttons
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet("QPushButton { background-color: green; color: white; }")
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("QPushButton { background-color: red; color: white; }")
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        # Add stretch to push everything to top
        layout.addStretch()

    def connect_signals(self):
        """Connect control signals"""
        self.freq_spinbox.valueChanged.connect(
            lambda v: self.frequency_changed.emit(v * 1e6)
        )

        self.sample_rate_combo.currentTextChanged.connect(
            self.on_sample_rate_changed
        )

        self.bandwidth_spinbox.valueChanged.connect(
            lambda v: self.bandwidth_changed.emit(v * 1e6)
        )

        self.gain_slider.valueChanged.connect(self.on_gain_changed)
        self.auto_gain_cb.toggled.connect(self.on_auto_gain_changed)

        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        self.refresh_devices_btn.clicked.connect(self.refresh_devices)

        self.start_btn.clicked.connect(self.start_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

    def on_sample_rate_changed(self, text: str):
        """Handle sample rate change"""
        try:
            # Parse sample rate from text
            if "kSps" in text:
                value = float(text.replace(" kSps", "")) * 1e3
            elif "MSps" in text:
                value = float(text.replace(" MSps", "")) * 1e6
            else:
                value = float(text)

            self.sample_rate_changed.emit(value)
        except:
            pass

    def on_gain_changed(self, value: int):
        """Handle gain slider change"""
        self.gain_label.setText(f"{value} dB")
        if not self.auto_gain_cb.isChecked():
            self.gain_changed.emit(float(value))

    def on_auto_gain_changed(self, enabled: bool):
        """Handle auto gain checkbox"""
        self.gain_slider.setEnabled(not enabled)
        if enabled:
            self.gain_changed.emit(0.0)  # 0 indicates auto gain
        else:
            self.gain_changed.emit(float(self.gain_slider.value()))

    def on_device_changed(self, device_name: str):
        """Handle device selection change"""
        # This would emit device info - simplified for now
        self.device_changed.emit({"name": device_name})

    def refresh_devices(self):
        """Refresh device list"""
        # This would call backend manager to refresh devices
        pass

    def update_from_settings(self):
        """Update controls from settings"""
        self.freq_spinbox.setValue(self.settings.sdr.center_freq / 1e6)
        self.bandwidth_spinbox.setValue((self.settings.sdr.bandwidth or 20e6) / 1e6)
        self.gain_slider.setValue(int(self.settings.sdr.gain))
        self.auto_gain_cb.setChecked(self.settings.sdr.auto_gain)

    def set_processing_state(self, running: bool):
        """Update controls for processing state"""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

        # Disable device controls when running
        self.device_combo.setEnabled(not running)
        self.refresh_devices_btn.setEnabled(not running)


class StatusWidget(QWidget):
    """Status and performance display widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup status UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Performance group
        perf_group = QGroupBox("Performance")
        perf_layout = QGridLayout()
        perf_group.setLayout(perf_layout)

        perf_layout.addWidget(QLabel("Sample Rate:"), 0, 0)
        self.sample_rate_label = QLabel("0 MSps")
        perf_layout.addWidget(self.sample_rate_label, 0, 1)

        perf_layout.addWidget(QLabel("Processed:"), 1, 0)
        self.processed_label = QLabel("0 samples")
        perf_layout.addWidget(self.processed_label, 1, 1)

        perf_layout.addWidget(QLabel("CPU Load:"), 2, 0)
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        perf_layout.addWidget(self.cpu_progress, 2, 1)

        perf_layout.addWidget(QLabel("Buffer:"), 3, 0)
        self.buffer_progress = QProgressBar()
        self.buffer_progress.setRange(0, 100)
        perf_layout.addWidget(self.buffer_progress, 3, 1)

        layout.addWidget(perf_group)

        # Signal info group
        signal_group = QGroupBox("Signal Info")
        signal_layout = QGridLayout()
        signal_group.setLayout(signal_layout)

        signal_layout.addWidget(QLabel("Peak Power:"), 0, 0)
        self.peak_power_label = QLabel("0 dB")
        signal_layout.addWidget(self.peak_power_label, 0, 1)

        signal_layout.addWidget(QLabel("Peak Freq:"), 1, 0)
        self.peak_freq_label = QLabel("0 MHz")
        signal_layout.addWidget(self.peak_freq_label, 1, 1)

        signal_layout.addWidget(QLabel("Bandwidth:"), 2, 0)
        self.bandwidth_label = QLabel("0 kHz")
        signal_layout.addWidget(self.bandwidth_label, 2, 1)

        signal_layout.addWidget(QLabel("SNR:"), 3, 0)
        self.snr_label = QLabel("0 dB")
        signal_layout.addWidget(self.snr_label, 3, 1)

        layout.addWidget(signal_group)

        # Add stretch
        layout.addStretch()

    def update_performance(self, perf_data: Dict[str, Any]):
        """Update performance display"""
        try:
            if 'sample_count' in perf_data:
                self.processed_label.setText(f"{perf_data['sample_count']:,} samples")

            if 'cpu_usage' in perf_data:
                self.cpu_progress.setValue(int(perf_data['cpu_usage']))

            if 'buffer_usage' in perf_data:
                self.buffer_progress.setValue(int(perf_data['buffer_usage']))

        except Exception as e:
            logging.getLogger(__name__).error(f"Error updating performance: {e}")

    def update_signal_info(self, analysis_data):
        """Update signal analysis display"""
        try:
            if analysis_data is None:
                return

            self.peak_power_label.setText(f"{analysis_data.peak_power:.1f} dB")
            self.peak_freq_label.setText(f"{analysis_data.peak_freq/1e6:.3f} MHz")
            self.bandwidth_label.setText(f"{analysis_data.bandwidth/1e3:.1f} kHz")
            self.snr_label.setText(f"{analysis_data.snr_estimate:.1f} dB")

        except Exception as e:
            logging.getLogger(__name__).error(f"Error updating signal info: {e}")


class MainWindow(QWidget):
    """Main application window"""

    # Signals
    start_requested = Signal()
    stop_requested = Signal()
    settings_changed = Signal(object)
    device_changed = Signal(object)
    frequency_changed = Signal(float)
    gain_changed = Signal(float)

    def __init__(self, settings: AppSettings, app_instance, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.app_instance = app_instance
        self.logger = logging.getLogger(__name__)
        self.perf_logger = get_performance_logger()

        self.setup_ui()
        self.setup_menu()
        self.connect_signals()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)

        # Performance tracking
        self.last_update_time = time.time()
        self.frame_count = 0

    def setup_ui(self):
        """Setup main UI layout"""
        # Main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(300)

        # Control panel
        self.control_panel = ControlPanel(self.settings)
        left_layout.addWidget(self.control_panel)

        # Status widget
        self.status_widget = StatusWidget()
        left_layout.addWidget(self.status_widget)

        main_splitter.addWidget(left_panel)

        # Right panel for displays
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Create tab widget for different displays
        self.tab_widget = QTabWidget()
        right_layout.addWidget(self.tab_widget)

        # Spectrum tab
        self.spectrum_widget = SpectrumWidget()
        self.tab_widget.addTab(self.spectrum_widget, "Spectrum")

        # IQ tab
        self.iq_widget = IQWidget()
        self.tab_widget.addTab(self.iq_widget, "I/Q Analysis")

        # Log tab
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumBlockCount(1000)  # Limit log size
        self.tab_widget.addTab(self.log_widget, "Log")

        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([300, 900])

        # Window properties
        self.setWindowTitle("RF Spectrum Analyzer")
        self.resize(1200, 800)

    def setup_menu(self):
        """Setup menu bar (if parent is QMainWindow)"""
        pass  # Would implement if MainWindow inherited from QMainWindow

    def connect_signals(self):
        """Connect internal signals"""
        # Control panel signals
        self.control_panel.start_requested.connect(self.start_requested.emit)
        self.control_panel.stop_requested.connect(self.stop_requested.emit)
        self.control_panel.frequency_changed.connect(self.frequency_changed.emit)
        self.control_panel.gain_changed.connect(self.gain_changed.emit)
        self.control_panel.device_changed.connect(self.device_changed.emit)

    def update_spectrum(self, spectrum_data):
        """Update spectrum display"""
        self.spectrum_widget.update_spectrum(spectrum_data)

        # Update status
        if hasattr(spectrum_data, 'timestamp'):
            self.frame_count += 1

    def update_iq_plot(self, iq_samples):
        """Update IQ display"""
        self.iq_widget.update_iq_data(iq_samples)

    def update_performance_display(self, perf_data: Dict[str, Any]):
        """Update performance metrics display"""
        self.status_widget.update_performance(perf_data)

    def update_displays(self):
        """Update all displays (called by timer)"""
        # Calculate FPS
        now = time.time()
        elapsed = now - self.last_update_time

        if elapsed >= 1.0:  # Update every second
            fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_update_time = now

            # Update performance display
            perf_data = {
                'fps': fps,
                'cpu_usage': 50,  # Placeholder
                'buffer_usage': 30  # Placeholder
            }
            self.update_performance_display(perf_data)

    def set_processing_state(self, running: bool):
        """Update UI for processing state"""
        self.control_panel.set_processing_state(running)

        if running:
            self.update_timer.start(100)  # 10 FPS update
        else:
            self.update_timer.stop()

    def set_device_status(self, connected: bool, status_text: str):
        """Update device status display"""
        # This would update the device status label in control panel
        pass

    def show_status_message(self, message: str, timeout: int = 2000):
        """Show status message"""
        # Add message to log
        self.log_widget.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def add_log_message(self, message: str):
        """Add message to log display"""
        self.log_widget.append(f"[{time.strftime('%H:%M:%S')}] {message}")

        # Auto-scroll to bottom
        self.log_widget.verticalScrollBar().setValue(
            self.log_widget.verticalScrollBar().maximum()
        )

    def closeEvent(self, event):
        """Handle close event"""
        # Stop timers
        self.update_timer.stop()
        event.accept()
