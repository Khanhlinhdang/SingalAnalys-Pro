"""
Main Window for RF Spectrum Analyzer
PySide6-based GUI with PyQtGraph widgets for real-time signal visualization.
"""


import os
import re
import sys
import numpy as np
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QStatusBar, QMenuBar,
    QMessageBox, QDockWidget, QFrame, QFileDialog,
    QLabel, QListWidget, QListWidgetItem, QPushButton,
    QDialog, QDialogButtonBox, QFormLayout, QDoubleSpinBox, QLineEdit,
    QComboBox,
    QToolButton,
    QSizePolicy,
    QProgressDialog,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence, QImage, QPixmap
from PySide6.QtCore import QByteArray

import pyqtgraph as pg

from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
from rf_spectrum_analyzer.gui.waterfall_widget import WaterfallWidget
from rf_spectrum_analyzer.utils.logger import get_logger
from rf_spectrum_analyzer.gui.controls_widget import ControlsWidget
from rf_spectrum_analyzer.gui.constellation_widget import ConstellationWidget
from rf_spectrum_analyzer.gui.bitstream_widget import BitstreamWidget
from rf_spectrum_analyzer.config.settings import Settings

# Set PyQtGraph options for better performance; disable OpenGL in offscreen mode.
use_open_gl = os.environ.get('QT_QPA_PLATFORM', '').lower() != 'offscreen'
pg.setConfigOptions(antialias=True, useOpenGL=use_open_gl)


class MainWindow(QMainWindow):
    """Main window widget containing all GUI components."""
    
    # Signals for communication with application
    start_requested = Signal()
    stop_requested = Signal()
    device_changed = Signal(str)
    frequency_changed = Signal(float)
    sample_rate_changed = Signal(float)
    bandwidth_changed = Signal(float)  # New signal for bandwidth
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
    export_image_artifact_requested = Signal(str)
    export_pcm_artifact_requested = Signal(str)
    export_session_report_requested = Signal(str)
    load_session_report_requested = Signal(str)
    # (filename, center_freq_hz, bandwidth_hz, advanced_params)
    process_signal_file_requested = Signal(str, float, float, object)
    
    def __init__(self, settings: Settings, app=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.app = app  # Reference to main application
        self.acquisition_active = False
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
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
        # Use a lightweight central placeholder and place all major panes in docks.
        central_widget = QWidget(self)
        central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        central_widget.setMinimumSize(1, 1)
        self.setCentralWidget(central_widget)

        self.setDockOptions(
            QMainWindow.AllowNestedDocks |
            QMainWindow.AllowTabbedDocks |
            QMainWindow.AnimatedDocks |
            QMainWindow.GroupedDragging
        )

        # Create primary widgets
        self.controls_widget = ControlsWidget(self.settings)
        self.spectrum_widget = SpectrumWidget(self.settings)
        self.waterfall_widget = WaterfallWidget(self.settings)

        # Controls dock
        self.controls_dock = QDockWidget("Controls", self)
        self.controls_dock.setObjectName("controls_dock")
        self.controls_dock.setWidget(self.controls_widget)
        self.controls_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        self.controls_dock.setMinimumWidth(240)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.controls_dock)

        # Spectrum dock
        self.spectrum_dock = QDockWidget("Spectrum", self)
        self.spectrum_dock.setObjectName("spectrum_dock")
        self.spectrum_dock.setWidget(self.spectrum_widget)
        self.spectrum_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, self.spectrum_dock)

        # Waterfall dock
        self.waterfall_dock = QDockWidget("Waterfall", self)
        self.waterfall_dock.setObjectName("waterfall_dock")
        self.waterfall_dock.setWidget(self.waterfall_widget)
        self.waterfall_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, self.waterfall_dock)

        # Compact default arrangement for core panes: controls | spectrum/waterfall.
        self.splitDockWidget(self.controls_dock, self.spectrum_dock, Qt.Horizontal)
        self.splitDockWidget(self.spectrum_dock, self.waterfall_dock, Qt.Vertical)

        # Secondary docks
        self.create_constellation_dock()
        self.create_bitstream_dock()
        self.create_image_artifact_dock()
        self.create_roi_results_dock()

        # Keep right-side visual docks organized
        self.tabifyDockWidget(self.constellation_dock, self.image_artifact_dock)
        self.tabifyDockWidget(self.constellation_dock, self.roi_results_dock)
        self.constellation_dock.raise_()

        # Create menu bar and actions
        self.create_menu_bar()

        # Initial dock sizing hints
        self.resizeDocks([self.controls_dock, self.spectrum_dock], [300, 900], Qt.Horizontal)
        self.resizeDocks([self.spectrum_dock, self.waterfall_dock], [520, 260], Qt.Vertical)

        # Restore previous dock layout if available, else apply stored preset.
        restored = self.restore_dock_layout_from_settings()
        if not restored:
            self.apply_window_preset(getattr(self.settings.gui, 'window_preset', 'monitoring'))
        
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
        self.constellation_dock.setObjectName("constellation_dock")
        self.constellation_dock.setWidget(self.constellation_widget)
        
        # Set dock widget properties
        self.constellation_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        
        # Set size hints for better layout
        self.constellation_dock.setMinimumSize(200, 150)
        self.constellation_dock.resize(400, 300)
        
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
        self.bitstream_dock.setObjectName("bitstream_dock")
        self.bitstream_dock.setWidget(self.bitstream_widget)
        
        # Set dock widget properties
        self.bitstream_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        
        # Set size hints for better layout
        self.bitstream_dock.setMinimumSize(250, 150)
        self.bitstream_dock.resize(500, 350)
        
        # Add dock widget to main window (initially on the bottom)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bitstream_dock)
        
        # Connect dock widget signals
        self.bitstream_dock.visibilityChanged.connect(self.on_bitstream_visibility_changed)
        self.bitstream_dock.topLevelChanged.connect(self.on_bitstream_floating_changed)
        
        # Set default state
        self.bitstream_dock.setFloating(False)
        self.bitstream_dock.show()

    def create_image_artifact_dock(self):
        """Create NOAA image artifact history and preview dock."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.image_preview_label = QLabel("No decoded NOAA image yet")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setMinimumSize(220, 140)
        self.image_preview_label.setFrameStyle(QFrame.Box | QFrame.Plain)

        self.image_history_list = QListWidget()
        self.image_history_list.currentRowChanged.connect(self._on_image_history_selected)

        clear_button = QPushButton("Clear Image History")
        clear_button.clicked.connect(self._clear_image_history)

        layout.addWidget(self.image_preview_label)
        layout.addWidget(self.image_history_list)
        layout.addWidget(clear_button)

        self._image_artifact_history = []

        self.image_artifact_dock = QDockWidget("NOAA Image Artifacts", self)
        self.image_artifact_dock.setObjectName("image_artifact_dock")
        self.image_artifact_dock.setWidget(container)
        self.image_artifact_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        self.image_artifact_dock.setMinimumSize(200, 150)
        self.addDockWidget(Qt.RightDockWidgetArea, self.image_artifact_dock)

        self.image_artifact_dock.visibilityChanged.connect(self.on_image_artifact_visibility_changed)

        self.image_artifact_dock.setFloating(False)
        self.image_artifact_dock.hide()

    def create_roi_results_dock(self):
        """Create ROI queue/result dock used for Phase-2 per-ROI tracking."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.roi_queue_summary_label = QLabel("ROI Queue: 0 pending")
        self.roi_queue_summary_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.roi_results_list = QListWidget()

        clear_button = QPushButton("Clear ROI Results")
        clear_button.clicked.connect(self._clear_roi_results)

        layout.addWidget(self.roi_queue_summary_label)
        layout.addWidget(self.roi_results_list)
        layout.addWidget(clear_button)

        self._roi_result_history = []

        self.roi_results_dock = QDockWidget("ROI Analysis Queue", self)
        self.roi_results_dock.setObjectName("roi_results_dock")
        self.roi_results_dock.setWidget(container)
        self.roi_results_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        self.roi_results_dock.setMinimumSize(220, 160)
        self.addDockWidget(Qt.RightDockWidgetArea, self.roi_results_dock)

        self.roi_results_dock.visibilityChanged.connect(self.on_roi_results_visibility_changed)
        self.roi_results_dock.setFloating(False)
        self.roi_results_dock.show()
    
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

    def on_image_artifact_visibility_changed(self, visible: bool):
        """Handle NOAA artifact dock visibility change."""
        if hasattr(self, 'view_menu'):
            self.image_artifact_action.setChecked(visible)

    def on_roi_results_visibility_changed(self, visible: bool):
        """Handle ROI result dock visibility change."""
        if hasattr(self, 'view_menu'):
            self.roi_results_action.setChecked(visible)
    
    def toggle_constellation_dock(self):
        """Toggle constellation dock visibility."""
        if self.constellation_dock.isVisible():
            self.constellation_dock.hide()
        else:
            self.constellation_dock.show()
    
    def reset_dock_layout(self):
        """Reset dock widgets to default layout."""
        self.controls_dock.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.controls_dock)
        self.controls_dock.show()

        self.spectrum_dock.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.spectrum_dock)
        self.spectrum_dock.show()

        self.waterfall_dock.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.waterfall_dock)
        self.splitDockWidget(self.controls_dock, self.spectrum_dock, Qt.Horizontal)
        self.splitDockWidget(self.spectrum_dock, self.waterfall_dock, Qt.Vertical)
        self.waterfall_dock.show()

        self.constellation_dock.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.constellation_dock)
        self.constellation_dock.show()
        
        self.bitstream_dock.setFloating(False)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bitstream_dock)
        self.bitstream_dock.show()

        self.image_artifact_dock.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.image_artifact_dock)
        self.roi_results_dock.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.roi_results_dock)
        self.tabifyDockWidget(self.constellation_dock, self.image_artifact_dock)
        self.tabifyDockWidget(self.constellation_dock, self.roi_results_dock)
        self.constellation_dock.raise_()

        self.resizeDocks([self.controls_dock, self.spectrum_dock], [300, 900], Qt.Horizontal)
        self.resizeDocks([self.spectrum_dock, self.waterfall_dock], [520, 260], Qt.Vertical)
        self.settings.gui.window_preset = "monitoring"

    def save_dock_layout_to_settings(self):
        """Persist current dock arrangement into settings as base64 state."""
        try:
            state = self.saveState()
            self.settings.gui.dock_layout_state_b64 = bytes(state.toBase64()).decode('ascii')
        except Exception as exc:
            self.logger.warning(f"Failed to save dock layout state: {exc}")

    def restore_dock_layout_from_settings(self) -> bool:
        """Restore dock arrangement from persisted settings state."""
        try:
            encoded = getattr(self.settings.gui, 'dock_layout_state_b64', '') or ''
            if not encoded:
                return False
            state = QByteArray.fromBase64(encoded.encode('ascii'))
            return bool(self.restoreState(state))
        except Exception as exc:
            self.logger.warning(f"Failed to restore dock layout state: {exc}")
            return False

    def apply_window_preset(self, preset: str):
        """Apply one-click dock arrangement presets."""
        preset = (preset or "monitoring").lower()

        # Start from a deterministic baseline.
        self.controls_dock.setFloating(False)
        self.spectrum_dock.setFloating(False)
        self.waterfall_dock.setFloating(False)
        self.constellation_dock.setFloating(False)
        self.bitstream_dock.setFloating(False)
        self.image_artifact_dock.setFloating(False)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.controls_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.spectrum_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.waterfall_dock)
        self.splitDockWidget(self.controls_dock, self.spectrum_dock, Qt.Horizontal)
        self.splitDockWidget(self.spectrum_dock, self.waterfall_dock, Qt.Vertical)
        self.addDockWidget(Qt.RightDockWidgetArea, self.constellation_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bitstream_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.image_artifact_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.roi_results_dock)
        self.tabifyDockWidget(self.constellation_dock, self.image_artifact_dock)
        self.tabifyDockWidget(self.constellation_dock, self.roi_results_dock)

        if preset == "monitoring":
            self.controls_dock.show()
            self.spectrum_dock.show()
            self.waterfall_dock.show()
            self.constellation_dock.hide()
            self.bitstream_dock.hide()
            self.image_artifact_dock.hide()
            self.roi_results_dock.hide()
            self.resizeDocks([self.controls_dock, self.spectrum_dock], [300, 900], Qt.Horizontal)
            self.resizeDocks([self.spectrum_dock, self.waterfall_dock], [520, 260], Qt.Vertical)
        elif preset == "analysis":
            self.controls_dock.show()
            self.spectrum_dock.show()
            self.waterfall_dock.show()
            self.constellation_dock.show()
            self.bitstream_dock.hide()
            self.image_artifact_dock.hide()
            self.roi_results_dock.show()
            self.resizeDocks([self.controls_dock, self.spectrum_dock], [280, 920], Qt.Horizontal)
            self.resizeDocks([self.spectrum_dock, self.waterfall_dock], [460, 260], Qt.Vertical)
        elif preset == "decode":
            self.controls_dock.show()
            self.spectrum_dock.show()
            self.waterfall_dock.hide()
            self.constellation_dock.show()
            self.bitstream_dock.show()
            self.image_artifact_dock.show()
            self.roi_results_dock.show()
            self.constellation_dock.raise_()
            self.resizeDocks([self.controls_dock, self.spectrum_dock], [280, 920], Qt.Horizontal)
            self.resizeDocks([self.spectrum_dock, self.bitstream_dock], [500, 240], Qt.Vertical)
        else:
            return

        self.settings.gui.window_preset = preset
    
    def create_menu_bar(self):
        """Create menu bar with view options."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')

        open_signal_action = QAction('&Open Signal File...', self)
        open_signal_action.setShortcut('Ctrl+Shift+O')
        open_signal_action.triggered.connect(self._on_open_signal_file)
        file_menu.addAction(open_signal_action)

        file_menu.addSeparator()

        export_image_action = QAction('Export Latest &Decoded Image...', self)
        export_image_action.setShortcut('Ctrl+E')
        export_image_action.triggered.connect(self._on_export_image_artifact)
        file_menu.addAction(export_image_action)

        export_pcm_action = QAction('Export Latest Decoded &Audio (WAV)...', self)
        export_pcm_action.setShortcut('Ctrl+Alt+E')
        export_pcm_action.triggered.connect(self._on_export_pcm_artifact)
        file_menu.addAction(export_pcm_action)

        export_report_action = QAction('Export Decode Session &Report...', self)
        export_report_action.setShortcut('Ctrl+Shift+E')
        export_report_action.triggered.connect(self._on_export_session_report)
        file_menu.addAction(export_report_action)

        load_report_action = QAction('&Load Decode Session Report...', self)
        load_report_action.setShortcut('Ctrl+Shift+L')
        load_report_action.triggered.connect(self._on_load_session_report)
        file_menu.addAction(load_report_action)
        
        # View menu
        self.view_menu = menubar.addMenu('&View')

        self.controls_action = QAction('&Controls', self)
        self.controls_action.setCheckable(True)
        self.controls_action.setChecked(True)
        self.controls_action.triggered.connect(lambda _=False: self.controls_dock.setVisible(not self.controls_dock.isVisible()))
        self.controls_dock.visibilityChanged.connect(self.controls_action.setChecked)
        self.view_menu.addAction(self.controls_action)

        self.spectrum_action = QAction('&Spectrum', self)
        self.spectrum_action.setCheckable(True)
        self.spectrum_action.setChecked(True)
        self.spectrum_action.triggered.connect(lambda _=False: self.spectrum_dock.setVisible(not self.spectrum_dock.isVisible()))
        self.spectrum_dock.visibilityChanged.connect(self.spectrum_action.setChecked)
        self.view_menu.addAction(self.spectrum_action)

        self.waterfall_action = QAction('&Waterfall', self)
        self.waterfall_action.setCheckable(True)
        self.waterfall_action.setChecked(True)
        self.waterfall_action.triggered.connect(lambda _=False: self.waterfall_dock.setVisible(not self.waterfall_dock.isVisible()))
        self.waterfall_dock.visibilityChanged.connect(self.waterfall_action.setChecked)
        self.view_menu.addAction(self.waterfall_action)

        self.view_menu.addSeparator()

        presets_menu = self.view_menu.addMenu('Window &Presets')

        monitoring_preset_action = QAction('&Monitoring', self)
        monitoring_preset_action.triggered.connect(lambda _=False: self.apply_window_preset('monitoring'))
        presets_menu.addAction(monitoring_preset_action)

        analysis_preset_action = QAction('&Analysis', self)
        analysis_preset_action.triggered.connect(lambda _=False: self.apply_window_preset('analysis'))
        presets_menu.addAction(analysis_preset_action)

        decode_preset_action = QAction('&Decode', self)
        decode_preset_action.triggered.connect(lambda _=False: self.apply_window_preset('decode'))
        presets_menu.addAction(decode_preset_action)

        self.view_menu.addSeparator()
        
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

        self.image_artifact_action = QAction('NOAA &Image Viewer', self)
        self.image_artifact_action.setCheckable(True)
        self.image_artifact_action.setChecked(False)
        self.image_artifact_action.setShortcut('Ctrl+I')
        self.image_artifact_action.triggered.connect(self.toggle_image_artifact_dock)
        self.view_menu.addAction(self.image_artifact_action)

        self.roi_results_action = QAction('ROI &Queue Results', self)
        self.roi_results_action.setCheckable(True)
        self.roi_results_action.setChecked(True)
        self.roi_results_action.setShortcut('Ctrl+Q')
        self.roi_results_action.triggered.connect(self.toggle_roi_results_dock)
        self.view_menu.addAction(self.roi_results_action)
        
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

    def _on_export_image_artifact(self):
        """Open save dialog and request export of the latest decoded image artifact."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Decoded Image Artifact",
            "decoded_noaa_image.png",
            "Image Files (*.png *.jpg);;NumPy (*.npy *.npz);;JSON (*.json)",
        )
        if filename:
            self.export_image_artifact_requested.emit(filename)

    def _on_export_session_report(self):
        """Open save dialog and request export of decode session report."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Decode Session Report",
            "decode_session_report.json",
            "JSON (*.json)",
        )
        if filename:
            self.export_session_report_requested.emit(filename)

    def _on_load_session_report(self):
        """Open file dialog and request replay load of decode session report."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Decode Session Report",
            "",
            "JSON (*.json)",
        )
        if filename:
            self.load_session_report_requested.emit(filename)

    def _on_open_signal_file(self):
        """Open a local signal file, ask for RF metadata, then request analysis."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Signal File",
            "",
            "Signal Files (*.wav *.npy *.npz);;All Files (*)",
        )
        if not filename:
            return

        defaults = self._infer_file_rf_defaults(filename)
        dlg = _SignalFileParamsDialog(defaults, self)
        if dlg.exec() != QDialog.Accepted:
            return

        self.process_signal_file_requested.emit(
            filename,
            dlg.center_freq(),
            dlg.bandwidth(),
            dlg.advanced_params(),
        )

    def _infer_file_rf_defaults(self, filename: str) -> Dict[str, float]:
        """Infer default RF params from filename; supports range or center frequency naming."""
        file_source = getattr(self.settings, 'file_source', None)
        default_sample_rate = float(getattr(file_source, 'sample_rate_hz', 0.0) or 0.0)
        default_freq_offset = float(getattr(file_source, 'freq_offset_hz', 0.0) or 0.0)
        default_start_sec = float(getattr(file_source, 'start_sec', 0.0) or 0.0)
        default_duration_sec = float(getattr(file_source, 'duration_sec', 0.0) or 0.0)

        defaults = {
            'mode': 'center_bw',
            'center_hz': float(self.settings.sdr.center_frequency),
            'bandwidth_hz': 24000.0,
            'low_hz': 0.0,
            'high_hz': 0.0,
            'sample_rate_hz': default_sample_rate,
            'freq_offset_hz': default_freq_offset,
            'start_sec': default_start_sec,
            'duration_sec': default_duration_sec,
        }

        base = os.path.basename(filename)

        # Pattern example: "1,537.7 MHz - 1,539.685 MHz"
        range_match = re.search(
            r'([0-9][0-9,]*\.?[0-9]*)\s*MHz\s*[\-–—]\s*([0-9][0-9,]*\.?[0-9]*)\s*MHz',
            base,
            re.IGNORECASE,
        )
        if range_match:
            low_hz = float(range_match.group(1).replace(',', '')) * 1e6
            high_hz = float(range_match.group(2).replace(',', '')) * 1e6
            if high_hz > low_hz:
                defaults['mode'] = 'low_high'
                defaults['low_hz'] = low_hz
                defaults['high_hz'] = high_hz
                defaults['center_hz'] = (low_hz + high_hz) / 2.0
                defaults['bandwidth_hz'] = max(high_hz - low_hz, 1.0)
                return defaults

        # Pattern example: *_1537996063Hz_*
        freq_match = re.search(r'_(\d{6,12})Hz', base, re.IGNORECASE)
        if freq_match:
            defaults['center_hz'] = float(freq_match.group(1))

        defaults['low_hz'] = max(defaults['center_hz'] - defaults['bandwidth_hz'] / 2.0, 0.0)
        defaults['high_hz'] = defaults['center_hz'] + defaults['bandwidth_hz'] / 2.0
        return defaults

    def trigger_open_signal_file(self):
        """Public slot to open signal file dialog (e.g. when device combo selects 'file')."""
        self._on_open_signal_file()

    def _on_export_pcm_artifact(self):
        """Open save dialog and request export of latest PCM artifact as WAV."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Decoded Audio Artifact",
            "decoded_audio.wav",
            "WAV (*.wav)",
        )
        if filename:
            self.export_pcm_artifact_requested.emit(filename)

    def toggle_image_artifact_dock(self):
        """Toggle NOAA image artifact dock visibility."""
        if self.image_artifact_dock.isVisible():
            self.image_artifact_dock.hide()
        else:
            self.image_artifact_dock.show()
            self.image_artifact_dock.raise_()

    def toggle_roi_results_dock(self):
        """Toggle ROI queue/result dock visibility."""
        if self.roi_results_dock.isVisible():
            self.roi_results_dock.hide()
        else:
            self.roi_results_dock.show()
            self.roi_results_dock.raise_()

    def update_roi_queue_panel(self, queue_entries: list):
        """Render current ROI queue/result entries in the dedicated dock."""
        entries = list(queue_entries or [])
        self.roi_results_list.clear()
        pending_count = 0

        for entry in reversed(entries):
            request = entry.get('request', {}) if isinstance(entry, dict) else {}
            status = str(entry.get('status', 'queued')).lower()
            if status in ('queued', 'running'):
                pending_count += 1

            center_hz = float(request.get('center_freq', 0.0) or 0.0)
            bandwidth_hz = float(request.get('bandwidth', 0.0) or 0.0)
            modulation = str(entry.get('modulation', '') or '')
            snr = entry.get('snr_db')
            result_note = str(entry.get('result_note', '') or '')

            line = f"[{status.upper()}] {center_hz/1e6:.3f} MHz | BW {bandwidth_hz/1e3:.1f} kHz"
            if modulation:
                line += f" | Mod: {modulation}"
            if snr is not None:
                line += f" | SNR: {float(snr):.1f} dB"
            if result_note:
                line += f" | {result_note}"
            self.roi_results_list.addItem(line)

        self.roi_queue_summary_label.setText(
            f"ROI Queue: {pending_count} pending / {len(entries)} tracked"
        )

    def _clear_roi_results(self):
        """Clear ROI result list in GUI; does not modify app-side queue state."""
        self.roi_results_list.clear()
        self.roi_queue_summary_label.setText("ROI Queue: 0 pending")

    def update_image_artifact_view(self, artifact: Dict[str, Any]):
        """Append NOAA image artifact to history and refresh preview."""
        payload = artifact.get('payload', {}) if isinstance(artifact, dict) else {}
        summary = payload.get('summary', {}) if isinstance(payload, dict) else {}
        width = int(summary.get('width', 0) or 0)
        height = int(summary.get('height', 0) or 0)

        title = f"{len(self._image_artifact_history) + 1}: {width}x{height}"
        item = QListWidgetItem(title)
        self.image_history_list.insertItem(0, item)

        self._image_artifact_history.insert(0, artifact)
        if len(self._image_artifact_history) > 25:
            self._image_artifact_history = self._image_artifact_history[:25]
            while self.image_history_list.count() > 25:
                self.image_history_list.takeItem(self.image_history_list.count() - 1)

        self.image_history_list.setCurrentRow(0)
        self.image_artifact_dock.show()

    def _on_image_history_selected(self, row: int):
        """Render selected artifact from NOAA image history list."""
        if row < 0 or row >= len(self._image_artifact_history):
            return

        payload = self._image_artifact_history[row].get('payload', {})
        matrix = payload.get('image_matrix') or payload.get('preview_rows')
        if matrix is None:
            self.image_preview_label.setText("Artifact missing image matrix")
            self.image_preview_label.setPixmap(QPixmap())
            return

        image = np.asarray(matrix, dtype=np.uint8)
        if image.ndim != 2 or image.size == 0:
            self.image_preview_label.setText("Invalid image matrix")
            self.image_preview_label.setPixmap(QPixmap())
            return

        qimage = QImage(
            image.data,
            int(image.shape[1]),
            int(image.shape[0]),
            int(image.strides[0]),
            QImage.Format_Grayscale8,
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.image_preview_label.width() - 8,
            self.image_preview_label.height() - 8,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_preview_label.setPixmap(pixmap)
        self.image_preview_label.setText("")

    def _clear_image_history(self):
        """Clear NOAA image artifact history and preview."""
        self._image_artifact_history = []
        self.image_history_list.clear()
        self.image_preview_label.setPixmap(QPixmap())
        self.image_preview_label.setText("No decoded NOAA image yet")

    def clear_image_artifact_history(self):
        """Public wrapper to clear NOAA image artifact history."""
        self._clear_image_history()
    
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
        self.controls_widget.bandwidth_changed.connect(self.bandwidth_changed.emit)
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
        self.spectrum_widget.x_range_changed.connect(self.waterfall_widget.set_frequency_view_range)
        self.waterfall_widget.bind_x_axis_to_spectrum(
            self.spectrum_widget.plot_widget.getPlotItem().getViewBox()
        )
        
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

        QDockWidget {
            border: 1px solid #555555;
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }

        QDockWidget::title {
            background: #333333;
            color: #dddddd;
            border-bottom: 1px solid #555555;
            padding: 3px 8px;
            min-height: 14px;
            font-size: 8.5pt;
            text-align: left;
        }

        QMainWindow::separator {
            background: #4a4a4a;
            width: 6px;
            height: 6px;
        }

        QMainWindow::separator:hover {
            background: #0078d4;
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

        QDockWidget {
            border: 1px solid #cccccc;
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }

        QDockWidget::title {
            background: #f2f2f2;
            color: #333333;
            border-bottom: 1px solid #cccccc;
            padding: 3px 8px;
            min-height: 14px;
            font-size: 8.5pt;
            text-align: left;
        }

        QMainWindow::separator {
            background: #d3d3d3;
            width: 6px;
            height: 6px;
        }

        QMainWindow::separator:hover {
            background: #0078d4;
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
    
    def update_device_frequency(self, frequency: float):
        """Update frequency display when device frequency changes."""
        try:
            self.controls_widget.update_frequency_display(frequency)
            self.logger.debug(f"GUI updated with new frequency: {frequency/1e6:.3f} MHz")
        except Exception as e:
            self.logger.error(f"Error updating frequency display: {e}")
    
    def update_device_bandwidth(self, bandwidth: float):
        """Update bandwidth display when device bandwidth changes."""
        try:
            self.controls_widget.update_bandwidth_display(bandwidth)
            self.logger.debug(f"GUI updated with new bandwidth: {bandwidth/1e6:.3f} MHz")
        except Exception as e:
            self.logger.error(f"Error updating bandwidth display: {e}")
    
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

class _SignalFileParamsDialog(QDialog):
    """Dialog for RF metadata of file source (center/bw or low/high)."""

    def __init__(self, defaults: Dict[str, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Signal File Parameters")
        self.setMinimumWidth(420)
        self._syncing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 8)
        layout.setSpacing(10)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Center Frequency + Bandwidth", "center_bw")
        self._mode_combo.addItem("Lower + Upper Frequency", "low_high")
        layout.addWidget(self._mode_combo)

        center_form = QFormLayout()
        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(0.0, 6000.0)
        self._freq_spin.setDecimals(6)
        self._freq_spin.setSuffix(" MHz")
        center_form.addRow("Center Frequency:", self._freq_spin)

        self._bw_spin = QDoubleSpinBox()
        self._bw_spin.setRange(0.1, 56000.0)
        self._bw_spin.setDecimals(3)
        self._bw_spin.setSuffix(" kHz")
        center_form.addRow("Bandwidth:", self._bw_spin)

        self._center_widget = QWidget()
        self._center_widget.setLayout(center_form)
        layout.addWidget(self._center_widget)

        range_form = QFormLayout()
        self._low_spin = QDoubleSpinBox()
        self._low_spin.setRange(0.0, 6000.0)
        self._low_spin.setDecimals(6)
        self._low_spin.setSuffix(" MHz")
        range_form.addRow("Lower Frequency:", self._low_spin)

        self._high_spin = QDoubleSpinBox()
        self._high_spin.setRange(0.0, 6000.0)
        self._high_spin.setDecimals(6)
        self._high_spin.setSuffix(" MHz")
        range_form.addRow("Upper Frequency:", self._high_spin)

        self._range_widget = QWidget()
        self._range_widget.setLayout(range_form)
        layout.addWidget(self._range_widget)

        self._computed_label = QLabel("")
        self._computed_label.setStyleSheet("QLabel { color: #5f6368; }")
        layout.addWidget(self._computed_label)

        self._advanced_toggle = QToolButton()
        self._advanced_toggle.setText("Advanced")
        self._advanced_toggle.setCheckable(True)
        self._advanced_toggle.setChecked(False)
        self._advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._advanced_toggle.setArrowType(Qt.RightArrow)
        layout.addWidget(self._advanced_toggle)

        advanced_form = QFormLayout()

        self._sample_rate_spin = QDoubleSpinBox()
        self._sample_rate_spin.setRange(0.0, 10000000.0)
        self._sample_rate_spin.setDecimals(3)
        self._sample_rate_spin.setSuffix(" kS/s (0 = use file)")
        advanced_form.addRow("Sample Rate Override:", self._sample_rate_spin)

        self._freq_offset_spin = QDoubleSpinBox()
        self._freq_offset_spin.setRange(-5000000.0, 5000000.0)
        self._freq_offset_spin.setDecimals(1)
        self._freq_offset_spin.setSuffix(" Hz")
        advanced_form.addRow("Frequency Offset Correction:", self._freq_offset_spin)

        self._start_sec_spin = QDoubleSpinBox()
        self._start_sec_spin.setRange(0.0, 86400.0)
        self._start_sec_spin.setDecimals(3)
        self._start_sec_spin.setSuffix(" s")
        advanced_form.addRow("Start Time:", self._start_sec_spin)

        self._duration_sec_spin = QDoubleSpinBox()
        self._duration_sec_spin.setRange(0.0, 86400.0)
        self._duration_sec_spin.setDecimals(3)
        self._duration_sec_spin.setSuffix(" s (0 = to end)")
        advanced_form.addRow("Duration:", self._duration_sec_spin)

        self._advanced_widget = QWidget()
        self._advanced_widget.setLayout(advanced_form)
        self._advanced_widget.setVisible(False)
        layout.addWidget(self._advanced_widget)

        mode = defaults.get('mode', 'center_bw')
        self._mode_combo.setCurrentIndex(0 if mode == 'center_bw' else 1)
        self._freq_spin.setValue(float(defaults.get('center_hz', 100e6)) / 1e6)
        self._bw_spin.setValue(max(float(defaults.get('bandwidth_hz', 24000.0)) / 1e3, 0.1))
        self._low_spin.setValue(max(float(defaults.get('low_hz', 0.0)) / 1e6, 0.0))
        self._high_spin.setValue(max(float(defaults.get('high_hz', 0.0)) / 1e6, 0.0))
        self._sample_rate_spin.setValue(max(float(defaults.get('sample_rate_hz', 0.0)) / 1e3, 0.0))
        self._freq_offset_spin.setValue(float(defaults.get('freq_offset_hz', 0.0)))
        self._start_sec_spin.setValue(max(float(defaults.get('start_sec', 0.0)), 0.0))
        self._duration_sec_spin.setValue(max(float(defaults.get('duration_sec', 0.0)), 0.0))

        self._mode_combo.currentIndexChanged.connect(self._update_mode_visibility)
        self._freq_spin.valueChanged.connect(self._sync_from_center)
        self._bw_spin.valueChanged.connect(self._sync_from_center)
        self._low_spin.valueChanged.connect(self._sync_from_range)
        self._high_spin.valueChanged.connect(self._sync_from_range)
        self._advanced_toggle.toggled.connect(self._toggle_advanced)

        self._update_mode_visibility()
        self._sync_from_center()
        self._update_computed_label()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _using_range_mode(self) -> bool:
        return self._mode_combo.currentData() == "low_high"

    def _toggle_advanced(self, checked: bool):
        self._advanced_widget.setVisible(checked)
        self._advanced_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _update_mode_visibility(self):
        using_range = self._using_range_mode()
        self._center_widget.setVisible(not using_range)
        self._range_widget.setVisible(using_range)
        self._update_computed_label()

    def _update_computed_label(self):
        center_mhz = self.center_freq() / 1e6
        bw_hz = self.bandwidth()
        bw_khz = bw_hz / 1e3
        bw_mhz = bw_hz / 1e6
        self._computed_label.setText(
            f"Computed: center={center_mhz:,.6f} MHz | bandwidth={bw_khz:,.3f} kHz ({bw_mhz:,.6f} MHz)"
        )

    def _sync_from_center(self):
        if self._syncing:
            return
        self._syncing = True
        center_mhz = self._freq_spin.value()
        bw_mhz = max(self._bw_spin.value() / 1000.0, 0.0001)
        half = bw_mhz / 2.0
        self._low_spin.setValue(max(center_mhz - half, 0.0))
        self._high_spin.setValue(max(center_mhz + half, 0.0))
        self._syncing = False
        self._update_computed_label()

    def _sync_from_range(self):
        if self._syncing:
            return
        self._syncing = True
        low = self._low_spin.value()
        high = self._high_spin.value()
        if high < low:
            low, high = high, low
            self._low_spin.setValue(low)
            self._high_spin.setValue(high)
        center = (low + high) / 2.0
        bw_khz = max((high - low) * 1000.0, 0.1)
        self._freq_spin.setValue(center)
        self._bw_spin.setValue(bw_khz)
        self._syncing = False
        self._update_computed_label()

    def center_freq(self) -> float:
        if self._using_range_mode():
            low_hz = self._low_spin.value() * 1e6
            high_hz = self._high_spin.value() * 1e6
            return (low_hz + high_hz) / 2.0
        return self._freq_spin.value() * 1e6

    def bandwidth(self) -> float:
        if self._using_range_mode():
            low_hz = self._low_spin.value() * 1e6
            high_hz = self._high_spin.value() * 1e6
            return max(high_hz - low_hz, 1.0)
        return max(self._bw_spin.value() * 1e3, 1.0)

    def advanced_params(self) -> Dict[str, float]:
        return {
            'sample_rate_hz': self._sample_rate_spin.value() * 1e3,
            'freq_offset_hz': self._freq_offset_spin.value(),
            'start_sec': self._start_sec_spin.value(),
            'duration_sec': self._duration_sec_spin.value(),
        }