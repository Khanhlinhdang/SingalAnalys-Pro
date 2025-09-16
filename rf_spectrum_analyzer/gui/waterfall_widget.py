"""
Waterfall Widget - Real-time waterfall display using PyQtGraph
High-performance waterfall/spectrogram visualization.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from rf_spectrum_analyzer.config.settings import Settings


class WaterfallWidget(QWidget):
    """Widget for displaying waterfall/spectrogram data."""
    
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        
        # Data storage
        self.waterfall_data = []
        self.frequency_axis = np.array([])
        self.max_lines = settings.gui.waterfall_height
        
        # Display parameters
        self.colormap = settings.gui.waterfall_colormap
        self.min_db = settings.gui.spectrum_min_db
        self.max_db = settings.gui.spectrum_max_db
        
        self.setup_ui()
        self.setup_plot()
    
    def setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Create control bar
        control_layout = QHBoxLayout()
        
        # Title label
        title_label = QLabel("Waterfall")
        title_font = QFont()
        title_font.setBold(True)
        title_label.setFont(title_font)
        control_layout.addWidget(title_label)
        
        control_layout.addStretch()
        
        # Colormap selection
        colormap_label = QLabel("Colormap:")
        control_layout.addWidget(colormap_label)
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot', 'cool'])
        self.colormap_combo.setCurrentText(self.colormap)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        control_layout.addWidget(self.colormap_combo)
        
        # Intensity range controls
        control_layout.addWidget(QLabel("Min:"))
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(-150, 0)
        self.min_slider.setValue(int(self.min_db))
        self.min_slider.valueChanged.connect(self._on_min_changed)
        self.min_slider.setMaximumWidth(100)
        control_layout.addWidget(self.min_slider)
        
        control_layout.addWidget(QLabel("Max:"))
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setRange(-50, 50)
        self.max_slider.setValue(int(self.max_db))
        self.max_slider.valueChanged.connect(self._on_max_changed)
        self.max_slider.setMaximumWidth(100)
        control_layout.addWidget(self.max_slider)
        
        layout.addLayout(control_layout)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('black' if self.settings.gui.theme == 'dark' else 'white')
        layout.addWidget(self.plot_widget)
    
    def setup_plot(self):
        """Setup the waterfall plot."""
        # Configure plot
        plot = self.plot_widget.getPlotItem()
        plot.setLabel('left', 'Time', units='s')
        plot.setLabel('bottom', 'Frequency', units='Hz')
        plot.setTitle('Waterfall')
        
        # Create image item for waterfall
        self.waterfall_image = pg.ImageItem()
        plot.addItem(self.waterfall_image)
        
        # Set up colormap
        self._setup_colormap()
        
        # Configure axes
        plot.invertY(True)  # Time axis goes from top to bottom
        plot.setAspectLocked(False)
        
        # Add colorbar
        self.colorbar = pg.ColorBarItem(
            values=(self.min_db, self.max_db),
            colorMap=self.colormap_obj,
            width=20,
            interactive=False
        )
        self.colorbar.setImageItem(self.waterfall_image)
    
    def _setup_colormap(self):
        """Setup colormap for waterfall display."""
        # Define colormaps
        colormaps = {
            'viridis': pg.colormap.get('viridis'),
            'plasma': pg.colormap.get('plasma'),
            'inferno': pg.colormap.get('inferno'),
            'magma': pg.colormap.get('magma'),
            'jet': self._create_jet_colormap(),
            'hot': self._create_hot_colormap(),
            'cool': self._create_cool_colormap()
        }
        
        # Get or create colormap
        if self.colormap in colormaps:
            self.colormap_obj = colormaps[self.colormap]
        else:
            self.colormap_obj = colormaps['viridis']
        
        # Apply colormap to image
        if hasattr(self, 'waterfall_image'):
            self.waterfall_image.setColorMap(self.colormap_obj)
    
    def _create_jet_colormap(self):
        """Create jet colormap."""
        colors = [
            [0.0, 0.0, 0.5],  # Dark blue
            [0.0, 0.0, 1.0],  # Blue
            [0.0, 0.5, 1.0],  # Light blue
            [0.0, 1.0, 1.0],  # Cyan
            [0.5, 1.0, 0.5],  # Light green
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.5, 0.0],  # Orange
            [1.0, 0.0, 0.0],  # Red
            [0.5, 0.0, 0.0]   # Dark red
        ]
        positions = np.linspace(0, 1, len(colors))
        return pg.ColorMap(positions, colors)
    
    def _create_hot_colormap(self):
        """Create hot colormap."""
        colors = [
            [0.0, 0.0, 0.0],  # Black
            [0.5, 0.0, 0.0],  # Dark red
            [1.0, 0.0, 0.0],  # Red
            [1.0, 0.5, 0.0],  # Orange
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 1.0, 0.5],  # Light yellow
            [1.0, 1.0, 1.0]   # White
        ]
        positions = np.linspace(0, 1, len(colors))
        return pg.ColorMap(positions, colors)
    
    def _create_cool_colormap(self):
        """Create cool colormap."""
        colors = [
            [0.0, 1.0, 1.0],  # Cyan
            [0.5, 0.5, 1.0],  # Light blue
            [1.0, 0.0, 1.0]   # Magenta
        ]
        positions = np.linspace(0, 1, len(colors))
        return pg.ColorMap(positions, colors)
    
    def update_data(self, waterfall_data: np.ndarray):
        """Update waterfall display with new data."""
        if waterfall_data.size == 0:
            return
        
        # Store data as list of 1D arrays
        if len(waterfall_data.shape) == 1:
            # Single spectrum line
            self.waterfall_data.append(waterfall_data.copy())
        else:
            # Multiple spectrum lines
            for line in waterfall_data:
                self.waterfall_data.append(line.copy())
        
        # Limit number of lines
        while len(self.waterfall_data) > self.max_lines:
            self.waterfall_data.pop(0)
        
        # Update frequency axis if needed
        if len(self.waterfall_data) > 0:
            spectrum_length = len(self.waterfall_data[0])
            if len(self.frequency_axis) != spectrum_length:
                self._update_frequency_axis(spectrum_length)
        
        # Update display
        self._update_image()
    
    def _update_frequency_axis(self, num_points: int):
        """Update frequency axis based on current settings."""
        sample_rate = self.settings.sdr.sample_rate
        center_freq = self.settings.sdr.center_frequency
        
        # Create frequency array
        freqs = np.linspace(-sample_rate/2, sample_rate/2, num_points)
        self.frequency_axis = freqs + center_freq
    
    def _update_image(self):
        """Update the waterfall image."""
        if len(self.waterfall_data) == 0:
            return
        
        try:
            # Convert list to 2D array
            image_data = np.array(self.waterfall_data)
            
            if image_data.size == 0:
                return
            
            # Transpose so frequency is X axis, time is Y axis
            image_data = image_data.T
            
            # Set image data
            self.waterfall_image.setImage(
                image_data,
                levels=(self.min_db, self.max_db)
            )
            
            # Set image position and scale
            if len(self.frequency_axis) > 0:
                freq_min = self.frequency_axis[0]
                freq_max = self.frequency_axis[-1]
                freq_range = freq_max - freq_min
                
                # Position: (x, y), Scale: (dx, dy)
                self.waterfall_image.setRect(
                    freq_min, 0,
                    freq_range, len(self.waterfall_data)
                )
            
            # Update plot ranges
            plot = self.plot_widget.getPlotItem()
            if len(self.frequency_axis) > 0:
                plot.setXRange(self.frequency_axis[0], self.frequency_axis[-1])
            # plot.setYRange(0, len(self.waterfall_data))
            
        except Exception as e:
            print(f"Error updating waterfall image: {e}")
    
    def _on_colormap_changed(self, colormap_name: str):
        """Handle colormap change."""
        self.colormap = colormap_name
        self.settings.gui.waterfall_colormap = colormap_name
        self._setup_colormap()
        self._update_image()
    
    def _on_min_changed(self, value: int):
        """Handle minimum level change."""
        self.min_db = float(value)
        self.waterfall_image.setLevels((self.min_db, self.max_db))
        
        # Update colorbar
        if hasattr(self, 'colorbar'):
            self.colorbar.setLevels((self.min_db, self.max_db))
    
    def _on_max_changed(self, value: int):
        """Handle maximum level change."""
        self.max_db = float(value)
        self.waterfall_image.setLevels((self.min_db, self.max_db))
        
        # Update colorbar
        if hasattr(self, 'colorbar'):
            self.colorbar.setLevels((self.min_db, self.max_db))
    
    def set_max_lines(self, max_lines: int):
        """Set maximum number of waterfall lines."""
        self.max_lines = max_lines
        
        # Trim existing data if needed
        while len(self.waterfall_data) > max_lines:
            self.waterfall_data.pop(0)
    
    def clear_data(self):
        """Clear all waterfall data."""
        self.waterfall_data.clear()
        self.waterfall_image.clear()
        self.frequency_axis = np.array([])
    
    def export_data(self) -> dict:
        """Export current waterfall data."""
        return {
            'waterfall_data': [line.tolist() for line in self.waterfall_data],
            'frequency_axis': self.frequency_axis.tolist() if len(self.frequency_axis) > 0 else [],
            'settings': {
                'colormap': self.colormap,
                'min_db': self.min_db,
                'max_db': self.max_db,
                'max_lines': self.max_lines
            }
        }
    
    def update_settings(self, settings: Settings):
        """Update widget with new settings."""
        self.settings = settings
        
        # Update theme
        bg_color = 'black' if settings.gui.theme == 'dark' else 'white'
        self.plot_widget.setBackground(bg_color)
        
        # Update frequency axis if settings changed
        if len(self.waterfall_data) > 0:
            spectrum_length = len(self.waterfall_data[0])
            self._update_frequency_axis(spectrum_length)
            self._update_image()
    
    def save_image(self, filename: str):
        """Save waterfall image to file."""
        try:
            # Export the plot widget as image
            exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
            exporter.export(filename)
            return True
        except Exception as e:
            print(f"Error saving waterfall image: {e}")
            return False