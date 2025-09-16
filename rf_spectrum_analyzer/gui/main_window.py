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
from rf_spectrum_analyzer.config.settings import Settings

# Set PyQtGraph options for better performance
pg.setConfigOptions(antialias=True, useOpenGL=True)


class MainWindow(QWidget):
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
    
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
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
        # Main layout
        main_layout = QHBoxLayout(self)
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
        
        # Set splitter proportions
        main_splitter.setSizes([300, 900])
        
        # Create status bar
        self.create_status_bar()
        
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
    
    def create_status_bar(self):
        """Create status bar with performance indicators."""
        # Note: Since we're inheriting from QWidget, we'll create a custom status area
        status_layout = QHBoxLayout()
        
        # Status labels
        from PySide6.QtWidgets import QLabel
        self.status_label = QLabel("Ready")
        self.fps_label = QLabel("FPS: 0")
        self.device_status_label = QLabel("Device: Disconnected")
        self.frequency_label = QLabel("Freq: 0 MHz")
        
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.fps_label)
        status_layout.addWidget(self.device_status_label)
        status_layout.addWidget(self.frequency_label)
        
        # Add status bar to main layout
        self.layout().addLayout(status_layout)
    
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
        
        # Connect spectrum widget signals
        self.spectrum_widget.frequency_clicked.connect(self._on_frequency_clicked)
        
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
            self.status_label.setText("Acquiring...")
            self.device_status_label.setText("Device: Connected")
        else:
            self.status_label.setText("Stopped")
            self.device_status_label.setText("Device: Disconnected")
    
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
            self.fps_label.setText(f"FPS: {fps:.1f}")
        
        self.frame_count = 0
        self.last_time = current_time
    
    def update_frequency_display(self, frequency: float):
        """Update frequency display in status bar."""
        self.frequency_label.setText(f"Freq: {frequency/1e6:.3f} MHz")
    
    def update_device_info(self, device_info: Dict[str, Any]):
        """Update device information display."""
        device_type = device_info.get("device_type", "Unknown")
        status = device_info.get("status", "Unknown")
        self.device_status_label.setText(f"Device: {device_type} ({status})")
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current GUI settings."""
        return {
            "window_geometry": self.geometry(),
            "acquisition_active": self.acquisition_active,
        }
    
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
        
        event.accept()