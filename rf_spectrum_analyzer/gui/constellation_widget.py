"""
Constellation Widget for RF Spectrum Analyzer
Displays IQ constellation diagram for digital modulation analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QCheckBox, QComboBox, QSlider,
                              QGroupBox, QSpinBox, QDoubleSpinBox)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont

import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen, mkBrush, ScatterPlotItem

logger = logging.getLogger(__name__)


class ConstellationWidget(QWidget):
    """
    Widget for displaying constellation diagrams of digital modulations.
    Shows I/Q scatter plot with reference constellation overlay.
    """
    
    # Signals
    settings_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.current_iq_data = np.array([], dtype=complex)
        self.current_symbols = np.array([], dtype=complex)
        self.reference_constellation = np.array([], dtype=complex)
        self.modulation_type = "Unknown"
        
        # Display settings
        self.settings = {
            "display_mode": "IQ",  # IQ, Symbols, Both
            "max_points": 2000,
            "point_size": 3,
            "fade_enabled": True,
            "hold_enabled": False,
            "show_reference": True,
            "color_scheme": "default",
            "grid_enabled": True,
            "auto_scale": True,
            "zoom_factor": 1.0,
            "persistence_alpha": 0.7
        }
        
        # Plot data
        self.iq_scatter = None
        self.symbol_scatter = None
        self.reference_scatter = None
        self.held_data = []
        
        self.setup_ui()
        self.setup_plot()
        self.setup_timer()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        title_label = QLabel("Constellation Diagram")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Controls
        controls_layout = self.create_controls()
        layout.addLayout(controls_layout)
        
        # Plot widget
        self.plot_widget = PlotWidget()
        self.plot_widget.setMinimumHeight(300)
        layout.addWidget(self.plot_widget)
        
        # Info panel
        info_layout = self.create_info_panel()
        layout.addLayout(info_layout)
        
    def create_controls(self) -> QHBoxLayout:
        """Create control panel."""
        layout = QHBoxLayout()
        
        # Display mode
        mode_group = QGroupBox("Display")
        mode_layout = QHBoxLayout(mode_group)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["IQ Data", "Symbols", "Both"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        
        # Reference constellation
        self.show_ref_cb = QCheckBox("Show Reference")
        self.show_ref_cb.setChecked(True)
        self.show_ref_cb.toggled.connect(self.on_reference_toggle)
        mode_layout.addWidget(self.show_ref_cb)
        
        layout.addWidget(mode_group)
        
        # View controls
        view_group = QGroupBox("View")
        view_layout = QHBoxLayout(view_group)
        
        self.hold_cb = QCheckBox("Hold")
        self.hold_cb.toggled.connect(self.on_hold_toggle)
        view_layout.addWidget(self.hold_cb)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_plot)
        view_layout.addWidget(self.clear_btn)
        
        self.auto_scale_cb = QCheckBox("Auto Scale")
        self.auto_scale_cb.setChecked(True)
        self.auto_scale_cb.toggled.connect(self.on_auto_scale_toggle)
        view_layout.addWidget(self.auto_scale_cb)
        
        layout.addWidget(view_group)
        
        # Point settings
        points_group = QGroupBox("Points")
        points_layout = QHBoxLayout(points_group)
        
        points_layout.addWidget(QLabel("Size:"))
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 10)
        self.point_size_spin.setValue(3)
        self.point_size_spin.valueChanged.connect(self.on_point_size_changed)
        points_layout.addWidget(self.point_size_spin)
        
        points_layout.addWidget(QLabel("Max:"))
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(100, 10000)
        self.max_points_spin.setValue(2000)
        self.max_points_spin.setSuffix(" pts")
        self.max_points_spin.valueChanged.connect(self.on_max_points_changed)
        points_layout.addWidget(self.max_points_spin)
        
        layout.addWidget(points_group)
        
        layout.addStretch()
        return layout
        
    def create_info_panel(self) -> QHBoxLayout:
        """Create information panel."""
        layout = QHBoxLayout()
        
        # Modulation info
        self.modulation_label = QLabel("Modulation: Unknown")
        self.modulation_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.modulation_label)
        
        # Points count
        self.points_label = QLabel("Points: 0")
        self.points_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.points_label)
        
        # EVM info
        self.evm_label = QLabel("EVM: --")
        self.evm_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.evm_label)
        
        # SNR info
        self.snr_label = QLabel("SNR: --")
        self.snr_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.snr_label)
        
        layout.addStretch()
        return layout
        
    def setup_plot(self):
        """Setup the constellation plot."""
        # Configure plot
        self.plot_widget.setLabel('left', 'Quadrature (Q)')
        self.plot_widget.setLabel('bottom', 'In-phase (I)')
        self.plot_widget.setTitle('Constellation Diagram')
        
        # Set equal aspect ratio
        self.plot_widget.getViewBox().setAspectLocked(True)
        
        # Add grid
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Add crosshair at origin
        self.plot_widget.addLine(x=0, pen=mkPen('gray', width=1, style=Qt.DashLine))
        self.plot_widget.addLine(y=0, pen=mkPen('gray', width=1, style=Qt.DashLine))
        
        # Initialize scatter plot items
        self.iq_scatter = ScatterPlotItem(
            size=self.settings["point_size"],
            pen=mkPen(None),
            brush=mkBrush(100, 150, 255, 150)  # Blue with transparency
        )
        self.plot_widget.addItem(self.iq_scatter)
        
        self.symbol_scatter = ScatterPlotItem(
            size=self.settings["point_size"] + 2,
            pen=mkPen('red', width=1),
            brush=mkBrush(255, 100, 100, 200)  # Red
        )
        self.plot_widget.addItem(self.symbol_scatter)
        
        self.reference_scatter = ScatterPlotItem(
            size=self.settings["point_size"] + 4,
            pen=mkPen('green', width=2),
            brush=mkBrush(100, 255, 100, 255),  # Green
            symbol='+'
        )
        self.plot_widget.addItem(self.reference_scatter)
        
    def setup_timer(self):
        """Setup update timer."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)  # 20 FPS
        
    def update_constellation(self, iq_data: np.ndarray, symbols: Optional[np.ndarray] = None,
                           modulation_info: Optional[Dict[str, Any]] = None):
        """
        Update constellation data.
        
        Args:
            iq_data: Raw IQ samples
            symbols: Detected/demodulated symbols
            modulation_info: Information about modulation type and parameters
        """
        try:
            if len(iq_data) == 0:
                return
                
            # Store data
            self.current_iq_data = iq_data
            if symbols is not None:
                self.current_symbols = symbols
            
            # Update modulation info
            if modulation_info:
                self.modulation_type = modulation_info.get("type", "Unknown")
                self.modulation_label.setText(f"Modulation: {self.modulation_type}")
                
                # Update reference constellation if available
                if "reference_constellation" in modulation_info:
                    self.reference_constellation = modulation_info["reference_constellation"]
                    
                # Update EVM and SNR if available
                if "evm" in modulation_info:
                    evm_db = 20 * np.log10(modulation_info["evm"]) if modulation_info["evm"] > 0 else -np.inf
                    self.evm_label.setText(f"EVM: {evm_db:.1f} dB")
                    
                if "snr_estimate" in modulation_info:
                    self.snr_label.setText(f"SNR: {modulation_info['snr_estimate']:.1f} dB")
            
            # Limit number of points for performance
            max_points = self.settings["max_points"]
            if len(iq_data) > max_points:
                # Take evenly spaced samples
                indices = np.linspace(0, len(iq_data)-1, max_points, dtype=int)
                self.current_iq_data = iq_data[indices]
                
            if symbols is not None and len(symbols) > max_points:
                indices = np.linspace(0, len(symbols)-1, min(max_points//2, len(symbols)), dtype=int)
                self.current_symbols = symbols[indices]
                
            # Update points count
            self.points_label.setText(f"Points: {len(self.current_iq_data)}")
            
        except Exception as e:
            self.logger.error(f"Error updating constellation: {e}")
            
    def update_display(self):
        """Update the constellation display."""
        try:
            mode = self.mode_combo.currentText()
            
            # Clear all plots first
            self.iq_scatter.clear()
            self.symbol_scatter.clear()
            self.reference_scatter.clear()
            
            # Display IQ data
            if mode in ["IQ Data", "Both"] and len(self.current_iq_data) > 0:
                i_data = np.real(self.current_iq_data)
                q_data = np.imag(self.current_iq_data)
                
                # Handle hold mode
                if self.settings["hold_enabled"]:
                    self.held_data.extend(list(zip(i_data, q_data)))
                    # Limit held data
                    if len(self.held_data) > self.settings["max_points"] * 2:
                        self.held_data = self.held_data[-self.settings["max_points"]:]
                    
                    if self.held_data:
                        held_i, held_q = zip(*self.held_data)
                        self.iq_scatter.setData(held_i, held_q)
                else:
                    self.iq_scatter.setData(i_data, q_data)
            
            # Display symbols
            if mode in ["Symbols", "Both"] and len(self.current_symbols) > 0:
                symbol_i = np.real(self.current_symbols)
                symbol_q = np.imag(self.current_symbols)
                self.symbol_scatter.setData(symbol_i, symbol_q)
            
            # Display reference constellation
            if self.settings["show_reference"] and len(self.reference_constellation) > 0:
                ref_i = np.real(self.reference_constellation)
                ref_q = np.imag(self.reference_constellation)
                self.reference_scatter.setData(ref_i, ref_q)
            
            # Auto scale if enabled
            if self.settings["auto_scale"]:
                self.plot_widget.getViewBox().autoRange()
                
        except Exception as e:
            self.logger.error(f"Error updating display: {e}")
            
    def generate_reference_constellation(self, modulation_type: str) -> np.ndarray:
        """Generate reference constellation points for known modulations."""
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
            elif mod_type == "64QAM":
                points = []
                for i in range(-7, 8, 2):
                    for q in range(-7, 8, 2):
                        points.append(i + 1j*q)
                return np.array(points) / np.sqrt(42)
            else:
                return np.array([], dtype=complex)
                
        except Exception as e:
            self.logger.error(f"Error generating reference constellation: {e}")
            return np.array([], dtype=complex)
            
    def clear_plot(self):
        """Clear all constellation data."""
        self.current_iq_data = np.array([], dtype=complex)
        self.current_symbols = np.array([], dtype=complex)
        self.held_data.clear()
        self.iq_scatter.clear()
        self.symbol_scatter.clear()
        self.points_label.setText("Points: 0")
        
    def on_mode_changed(self, mode: str):
        """Handle display mode change."""
        self.settings["display_mode"] = mode
        self.settings_changed.emit(self.settings.copy())
        
    def on_reference_toggle(self, checked: bool):
        """Handle reference constellation toggle."""
        self.settings["show_reference"] = checked
        if not checked:
            self.reference_scatter.clear()
        self.settings_changed.emit(self.settings.copy())
        
    def on_hold_toggle(self, checked: bool):
        """Handle hold mode toggle."""
        self.settings["hold_enabled"] = checked
        if not checked:
            self.held_data.clear()
        self.settings_changed.emit(self.settings.copy())
        
    def on_auto_scale_toggle(self, checked: bool):
        """Handle auto scale toggle."""
        self.settings["auto_scale"] = checked
        self.settings_changed.emit(self.settings.copy())
        
    def on_point_size_changed(self, size: int):
        """Handle point size change."""
        self.settings["point_size"] = size
        
        # Update scatter plot items
        if self.iq_scatter:
            self.iq_scatter.setSize(size)
        if self.symbol_scatter:
            self.symbol_scatter.setSize(size + 2)
        if self.reference_scatter:
            self.reference_scatter.setSize(size + 4)
            
        self.settings_changed.emit(self.settings.copy())
        
    def on_max_points_changed(self, max_points: int):
        """Handle max points change."""
        self.settings["max_points"] = max_points
        self.settings_changed.emit(self.settings.copy())
        
    def get_settings(self) -> Dict[str, Any]:
        """Get current constellation display settings."""
        return self.settings.copy()
        
    def set_settings(self, settings: Dict[str, Any]):
        """Set constellation display settings."""
        self.settings.update(settings)
        
        # Update UI controls
        if "display_mode" in settings:
            mode_text = {"IQ": "IQ Data", "Symbols": "Symbols", "Both": "Both"}.get(
                settings["display_mode"], "IQ Data")
            self.mode_combo.setCurrentText(mode_text)
            
        if "show_reference" in settings:
            self.show_ref_cb.setChecked(settings["show_reference"])
            
        if "hold_enabled" in settings:
            self.hold_cb.setChecked(settings["hold_enabled"])
            
        if "auto_scale" in settings:
            self.auto_scale_cb.setChecked(settings["auto_scale"])
            
        if "point_size" in settings:
            self.point_size_spin.setValue(settings["point_size"])
            
        if "max_points" in settings:
            self.max_points_spin.setValue(settings["max_points"])
            
    def save_constellation_image(self, filename: str):
        """Save constellation diagram as image."""
        try:
            exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
            exporter.export(filename)
            self.logger.info(f"Constellation diagram saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving constellation image: {e}")