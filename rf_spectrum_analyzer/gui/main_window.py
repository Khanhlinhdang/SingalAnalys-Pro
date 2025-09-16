"""
Main Window for RF Spectrum Analyzer
PySide6-based GUI with PyQtGraph widgets for real-time signal visualization.
"""

import sys
import numpy as np
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QTabWidget, QStatusBar, QMenuBar, 
    QMessageBox, QDockWidget, QFrame
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence

import pyqtgraph as pg

from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
from rf_spectrum_analyzer.gui.waterfall_widget import WaterfallWidget
from rf_spectrum_analyzer.gui.controls_widget import ControlsWidget
from rf_spectrum_analyzer.gui.constellation_widget import ConstellationWidget
from rf_spectrum_analyzer.gui.bitstream_widget import BitstreamWidget
from rf_spectrum_analyzer.config.settings import Settings

# Set PyQtGraph options for better performance
pg.setConfigOptions(antialias=True, useOpenGL=True)


class MainWindow(QMainWindow):
    """Main window widget containing all GUI components."""
    
    # Signals for communication with application
    start_requested = Signal()
    stop_requested = Signal()
    device_changed = Signal(str)
    frequency_changed = Signal(float)
    sample_rate_changed = Signal(float)
    gain_changed = Signal(float)
    fft_size_changed = Signal(int)
    window_changed = Signal(str)
    averaging_changed = Signal(int)
    
    # Detection signals
    manual_detection_triggered = Signal()
    tdma_detection_triggered = Signal()
    auto_detection_toggled = Signal(bool)
    advanced_analysis_toggled = Signal(bool)
    detection_threshold_changed = Signal(float)
    detection_interval_changed = Signal(int)
    
    # Frequency analysis signals
    frequency_range_changed = Signal(float, float)
    frequency_markers_toggled = Signal(bool)
    center_frequency_locked = Signal(bool)
    analysis_bandwidth_changed = Signal(float)
    
    # Sequential workflow signals
    demodulate_triggered = Signal()
    decode_triggered = Signal()
    
    def __init__(self, settings: Settings, app=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.app = app  # Reference to main application
        self.acquisition_active = False
        
        # Performance monitoring
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self._update_fps)
        self.fps_timer.start(1000)  # Update FPS every second
        self.frame_count = 0
        self.last_time = 0
        
        self.setup_ui()
        self.connect_signals()
        self.apply_theme()
    
    def setup_ui(self):
        """Setup the user interface layout."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout for central widget
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Create controls dock (left side)
        self.controls_widget = ControlsWidget(self.settings)
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.StyledPanel)
        controls_frame.setFixedWidth(300)
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.addWidget(self.controls_widget)
        main_splitter.addWidget(controls_frame)
        
        # Create display area (right side)
        display_widget = self.create_display_area()
        main_splitter.addWidget(display_widget)
        
        # Create constellation dock widget
        self.create_constellation_dock()
        
        # Create bitstream dock widget
        self.create_bitstream_dock()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Set splitter proportions
        main_splitter.setSizes([300, 900])
        
        # Set window properties
        self.setWindowTitle("RF Spectrum Analyzer")
        self.resize(self.settings.gui.window_width, self.settings.gui.window_height)
    
    def create_display_area(self) -> QWidget:
        """Create the main display area with spectrum and waterfall."""
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(2)
        
        # Create display splitter (vertical)
        display_splitter = QSplitter(Qt.Vertical)
        display_layout.addWidget(display_splitter)
        
        # Create spectrum widget
        self.spectrum_widget = SpectrumWidget(self.settings)
        spectrum_frame = QFrame()
        spectrum_frame.setFrameStyle(QFrame.StyledPanel)
        spectrum_layout = QVBoxLayout(spectrum_frame)
        spectrum_layout.setContentsMargins(2, 2, 2, 2)
        spectrum_layout.addWidget(self.spectrum_widget)
        display_splitter.addWidget(spectrum_frame)
        
        # Create waterfall widget
        self.waterfall_widget = WaterfallWidget(self.settings)
        waterfall_frame = QFrame()
        waterfall_frame.setFrameStyle(QFrame.StyledPanel)
        waterfall_layout = QVBoxLayout(waterfall_frame)
        waterfall_layout.setContentsMargins(2, 2, 2, 2)
        waterfall_layout.addWidget(self.waterfall_widget)
        display_splitter.addWidget(waterfall_frame)
        
        # Set splitter proportions (spectrum larger than waterfall)
        display_splitter.setSizes([400, 200])
        
        return display_widget
    
    def create_constellation_dock(self):
        """Create constellation dock widget that can be detached."""
        # Create constellation widget
        self.constellation_widget = ConstellationWidget()
        
        # Create dock widget for constellation
        self.constellation_dock = QDockWidget("Constellation Display", self)
        self.constellation_dock.setWidget(self.constellation_widget)
        
        # Set dock widget properties
        self.constellation_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        
        # Set size hints for better layout
        self.constellation_dock.setMinimumSize(400, 300)
        self.constellation_dock.resize(500, 400)
        
        # Add dock widget to main window (initially on the right)
        self.addDockWidget(Qt.RightDockWidgetArea, self.constellation_dock)
        
        # Connect dock widget signals
        self.constellation_dock.visibilityChanged.connect(self.on_constellation_visibility_changed)
        self.constellation_dock.topLevelChanged.connect(self.on_constellation_floating_changed)
        
        # Set default state
        self.constellation_dock.setFloating(False)
        self.constellation_dock.show()
    
    def create_bitstream_dock(self):
        """Create bitstream dock widget that can be detached."""
        # Create bitstream widget
        self.bitstream_widget = BitstreamWidget()
        
        # Create dock widget for bitstream
        self.bitstream_dock = QDockWidget("Bitstream Display", self)
        self.bitstream_dock.setWidget(self.bitstream_widget)
        
        # Set dock widget properties
        self.bitstream_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        
        # Set size hints for better layout
        self.bitstream_dock.setMinimumSize(500, 350)
        self.bitstream_dock.resize(600, 450)
        
        # Add dock widget to main window (initially on the bottom)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bitstream_dock)
        
        # Connect dock widget signals
        self.bitstream_dock.visibilityChanged.connect(self.on_bitstream_visibility_changed)
        self.bitstream_dock.topLevelChanged.connect(self.on_bitstream_floating_changed)
        
        # Set default state
        self.bitstream_dock.setFloating(False)
        self.bitstream_dock.show()
    
    def on_bitstream_visibility_changed(self, visible: bool):
        """Handle bitstream dock visibility change."""
        if hasattr(self, 'view_menu'):
            # Update menu check state if menu exists
            self.bitstream_action.setChecked(visible)
    
    def on_bitstream_floating_changed(self, floating: bool):
        """Handle bitstream dock floating state change."""
        if floating:
            # When floating, set a nice window icon and title
            self.bitstream_dock.setWindowIcon(self.windowIcon())
            self.bitstream_dock.setWindowTitle("Bitstream Display - RF Spectrum Analyzer")
            
            # Set optimal size for floating window
            self.bitstream_dock.resize(700, 550)
    
    def toggle_bitstream_dock(self):
        """Toggle bitstream dock visibility."""
        if self.bitstream_dock.isVisible():
            self.bitstream_dock.hide()
        else:
            self.bitstream_dock.show()
    
    def on_constellation_visibility_changed(self, visible: bool):
        """Handle constellation dock visibility change."""
        if hasattr(self, 'view_menu'):
            # Update menu check state if menu exists
            self.constellation_action.setChecked(visible)
    
    def on_constellation_floating_changed(self, floating: bool):
        """Handle constellation dock floating state change."""
        if floating:
            # When floating, set a nice window icon and title
            self.constellation_dock.setWindowIcon(self.windowIcon())
            self.constellation_dock.setWindowTitle("Constellation Display - RF Spectrum Analyzer")
            
            # Set optimal size for floating window
            self.constellation_dock.resize(600, 500)
    
    def toggle_constellation_dock(self):
        """Toggle constellation dock visibility."""
        if self.constellation_dock.isVisible():
            self.constellation_dock.hide()
        else:
            self.constellation_dock.show()
    
    def reset_dock_layout(self):
        """Reset dock widgets to default layout."""
        self.constellation_dock.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.constellation_dock)
        self.constellation_dock.show()
        
        self.bitstream_dock.setFloating(False)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bitstream_dock)
        self.bitstream_dock.show()
    
    def create_menu_bar(self):
        """Create menu bar with view options."""
        menubar = self.menuBar()
        
        # View menu
        self.view_menu = menubar.addMenu('&View')
        
        # Constellation display toggle
        self.constellation_action = QAction('&Constellation Display', self)
        self.constellation_action.setCheckable(True)
        self.constellation_action.setChecked(True)
        self.constellation_action.setShortcut('Ctrl+D')
        self.constellation_action.triggered.connect(self.toggle_constellation_dock)
        self.view_menu.addAction(self.constellation_action)
        
        # Bitstream display toggle
        self.bitstream_action = QAction('&Bitstream Display', self)
        self.bitstream_action.setCheckable(True)
        self.bitstream_action.setChecked(True)
        self.bitstream_action.setShortcut('Ctrl+B')
        self.bitstream_action.triggered.connect(self.toggle_bitstream_dock)
        self.view_menu.addAction(self.bitstream_action)
        
        self.view_menu.addSeparator()
        
        # Reset layout action
        reset_layout_action = QAction('&Reset Layout', self)
        reset_layout_action.setShortcut('Ctrl+R')
        reset_layout_action.triggered.connect(self.reset_dock_layout)
        self.view_menu.addAction(reset_layout_action)
        
        self.view_menu.addSeparator()
        
        # Fullscreen toggle
        fullscreen_action = QAction('&Fullscreen', self)
        fullscreen_action.setCheckable(True)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        self.view_menu.addAction(fullscreen_action)
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def connect_signals(self):
        """Connect internal signals."""
        # Connect controls widget signals
        self.controls_widget.start_clicked.connect(self.start_requested.emit)
        self.controls_widget.stop_clicked.connect(self.stop_requested.emit)
        self.controls_widget.device_changed.connect(self.device_changed.emit)
        self.controls_widget.frequency_changed.connect(self.frequency_changed.emit)
        self.controls_widget.sample_rate_changed.connect(self.sample_rate_changed.emit)
        self.controls_widget.gain_changed.connect(self.gain_changed.emit)
        self.controls_widget.fft_size_changed.connect(self.fft_size_changed.emit)
        self.controls_widget.window_changed.connect(self.window_changed.emit)
        self.controls_widget.averaging_changed.connect(self.averaging_changed.emit)
        
        # Connect detection signals
        self.controls_widget.manual_detection_triggered.connect(self.manual_detection_triggered.emit)
        self.controls_widget.tdma_detection_triggered.connect(self.tdma_detection_triggered.emit)
        self.controls_widget.auto_detection_toggled.connect(self.auto_detection_toggled.emit)
        self.controls_widget.advanced_analysis_toggled.connect(self.advanced_analysis_toggled.emit)
        self.controls_widget.detection_threshold_changed.connect(self.detection_threshold_changed.emit)
        self.controls_widget.detection_interval_changed.connect(self.detection_interval_changed.emit)
        
        # Connect frequency analysis signals
        self.controls_widget.frequency_range_changed.connect(self.frequency_range_changed.emit)
        self.controls_widget.frequency_markers_toggled.connect(self._on_frequency_markers_toggled)
        self.controls_widget.center_frequency_locked.connect(self.center_frequency_locked.emit)
        self.controls_widget.analysis_bandwidth_changed.connect(self.analysis_bandwidth_changed.emit)
        
        # Connect sequential workflow signals
        self.controls_widget.demodulate_triggered.connect(self.demodulate_triggered.emit)
        self.controls_widget.decode_triggered.connect(self.decode_triggered.emit)
        
        # Connect spectrum widget signals
        self.spectrum_widget.frequency_clicked.connect(self._on_frequency_clicked)
        self.spectrum_widget.frequency_range_selected.connect(self._on_frequency_range_selected)
        
        # Connect settings changes
        self.controls_widget.settings_changed.connect(self._on_settings_changed)
    
    def apply_theme(self):
        """Apply color theme to the interface."""
        if self.settings.gui.theme == "dark":
            self.apply_dark_theme()
        else:
            self.apply_light_theme()
    
    def apply_dark_theme(self):
        """Apply dark theme styling."""
        dark_style = """
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            font-family: Arial, sans-serif;
            font-size: 9pt;
        }
        
        QFrame {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            border-radius: 3px;
        }
        
        QPushButton {
            background-color: #4c4c4c;
            border: 1px solid #666666;
            border-radius: 3px;
            padding: 5px 10px;
            min-width: 80px;
        }
        
        QPushButton:hover {
            background-color: #5c5c5c;
        }
        
        QPushButton:pressed {
            background-color: #3c3c3c;
        }
        
        QPushButton:checked {
            background-color: #0078d4;
            border-color: #106ebe;
        }
        
        QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #4c4c4c;
            border: 1px solid #666666;
            border-radius: 3px;
            padding: 3px;
            min-width: 80px;
        }
        
        QComboBox:drop-down {
            border: none;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #ffffff;
        }
        
        QLabel {
            color: #ffffff;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 1px solid #666666;
            border-radius: 3px;
            margin-top: 10px;
            padding-top: 5px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QTabWidget::pane {
            border: 1px solid #666666;
            background-color: #3c3c3c;
        }
        
        QTabBar::tab {
            background-color: #4c4c4c;
            border: 1px solid #666666;
            padding: 5px 10px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background-color: #0078d4;
        }
        """
        self.setStyleSheet(dark_style)
    
    def apply_light_theme(self):
        """Apply light theme styling."""
        light_style = """
        QWidget {
            background-color: #ffffff;
            color: #000000;
            font-family: Arial, sans-serif;
            font-size: 9pt;
        }
        
        QFrame {
            background-color: #f5f5f5;
            border: 1px solid #cccccc;
            border-radius: 3px;
        }
        
        QPushButton {
            background-color: #e1e1e1;
            border: 1px solid #999999;
            border-radius: 3px;
            padding: 5px 10px;
            min-width: 80px;
        }
        
        QPushButton:hover {
            background-color: #d1d1d1;
        }
        
        QPushButton:pressed {
            background-color: #c1c1c1;
        }
        
        QPushButton:checked {
            background-color: #0078d4;
            color: #ffffff;
            border-color: #106ebe;
        }
        """
        self.setStyleSheet(light_style)
    
    # Data update methods
    def update_spectrum(self, spectrum_data: np.ndarray):
        """Update spectrum display with new data."""
        if self.spectrum_widget:
            self.spectrum_widget.update_data(spectrum_data)
            self.frame_count += 1
    
    def update_waterfall(self, waterfall_data: np.ndarray):
        """Update waterfall display with new data."""
        if self.waterfall_widget:
            self.waterfall_widget.update_data(waterfall_data)
    
    def set_acquisition_state(self, active: bool):
        """Update GUI to reflect acquisition state."""
        self.acquisition_active = active
        self.controls_widget.set_acquisition_state(active)
        
        if active:
            self.controls_widget.update_status("Acquiring...")
            self.controls_widget.update_device_status("Connected", True)
        else:
            self.controls_widget.update_status("Stopped")
            self.controls_widget.update_device_status("Disconnected", False)
    
    def show_error_message(self, message: str):
        """Show error message to user."""
        QMessageBox.critical(self, "RF Spectrum Analyzer Error", message)
    
    def show_info_message(self, message: str):
        """Show information message to user."""
        QMessageBox.information(self, "RF Spectrum Analyzer", message)
    
    # Event handlers
    def _on_frequency_clicked(self, frequency: float):
        """Handle frequency selection from spectrum display."""
        self.controls_widget.set_frequency(frequency)
        self.frequency_changed.emit(frequency)
    
    def _on_settings_changed(self):
        """Handle settings changes."""
        self.apply_theme()
        
        # Update display widgets with new settings
        if hasattr(self, 'spectrum_widget'):
            self.spectrum_widget.update_settings(self.settings)
        if hasattr(self, 'waterfall_widget'):
            self.waterfall_widget.update_settings(self.settings)
    
    def _update_fps(self):
        """Update FPS display."""
        import time
        current_time = time.time()
        
        if self.last_time > 0:
            fps = self.frame_count / (current_time - self.last_time)
            self.controls_widget.update_fps(int(fps))
        
        self.frame_count = 0
        self.last_time = current_time
    
    def update_frequency_display(self, frequency: float):
        """Update frequency display in device controls."""
        self.controls_widget.update_frequency_display(frequency)
    
    def _on_frequency_markers_toggled(self, enabled: bool):
        """Handle frequency markers toggle."""
        self.spectrum_widget.set_frequency_markers_enabled(enabled)
    
    def _on_frequency_range_selected(self, f1: float, f2: float):
        """Handle frequency range selection from spectrum widget."""
        # Update controls widget with selected range
        self.controls_widget.update_frequency_range_from_spectrum(f1, f2)
        
        # Emit signal for application to handle
        self.frequency_range_changed.emit(f1, f2)
    
    def set_frequency_range(self, f1: float, f2: float):
        """Set frequency range from external source."""
        self.spectrum_widget.set_frequency_range(f1, f2)
        self.controls_widget.update_frequency_range_from_spectrum(f1, f2)
    
    def enable_peak_hold(self, enabled: bool):
        """Enable or disable peak hold functionality."""
        self.spectrum_widget.set_peak_hold_enabled(enabled)
    
    def reset_peak_hold(self):
        """Reset peak hold data."""
        self.spectrum_widget.reset_peak_hold()
    
    def update_device_info(self, device_info: Dict[str, Any]):
        """Update device information display."""
        device_type = device_info.get("device_type", "Unknown")
        status = device_info.get("status", "Unknown")
        connected = status.lower() in ["connected", "active", "running"]
        self.controls_widget.update_device_status(device_type, connected)
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current GUI settings."""
        return {
            "window_geometry": self.geometry(),
            "acquisition_active": self.acquisition_active,
        }
    
    def update_constellation(self, iq_data: np.ndarray, symbols: Optional[np.ndarray] = None,
                           modulation_info: Optional[Dict[str, Any]] = None):
        """Update constellation display."""
        if hasattr(self, 'constellation_widget') and self.constellation_widget:
            self.constellation_widget.update_constellation(iq_data, symbols, modulation_info)
    
    def update_bitstream(self, bit_data: np.ndarray):
        """Update bitstream display with new bit data."""
        if hasattr(self, 'bitstream_widget') and self.bitstream_widget:
            # Convert to list of integers if needed
            if isinstance(bit_data, np.ndarray):
                # Ensure we have binary data
                if bit_data.dtype == bool:
                    bits = bit_data.astype(int).tolist()
                elif bit_data.dtype in [np.int8, np.int16, np.int32, np.int64]:
                    # Already integer, just convert to list and ensure binary
                    bits = np.clip(bit_data, 0, 1).astype(int).tolist()
                else:
                    # Float data, threshold to binary
                    threshold = np.mean(bit_data) if len(bit_data) > 0 else 0.5
                    bits = (bit_data > threshold).astype(int).tolist()
            else:
                bits = bit_data
            
            self.bitstream_widget.add_bits(bits)
    
    def update_detection_status(self, detected, snr_db=None, confidence=None):
        """Update detection status in controls widget."""
        if hasattr(self, 'controls_widget') and self.controls_widget:
            self.controls_widget.update_detection_status(detected, snr_db, confidence)
    
    def set_acquisition_state(self, active: bool):
        """Update acquisition state across all widgets."""
        self.acquisition_active = active
        if hasattr(self, 'controls_widget') and self.controls_widget:
            self.controls_widget.set_acquisition_state(active)
    
    def show_error_message(self, message: str):
        """Show error message to user."""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Error", message)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_Space:
            # Space bar toggles start/stop
            if self.acquisition_active:
                self.stop_requested.emit()
            else:
                self.start_requested.emit()
        elif event.key() == Qt.Key_Escape:
            # Escape stops acquisition
            if self.acquisition_active:
                self.stop_requested.emit()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop acquisition before closing
        if self.acquisition_active:
            self.stop_requested.emit()
        
        # Call app shutdown if available
        if self.app and hasattr(self.app, 'shutdown_application'):
            self.app.shutdown_application()
        
        event.accept()