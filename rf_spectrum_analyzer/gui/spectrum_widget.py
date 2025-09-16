"""
Spectrum Widget - Real-time spectrum display using PyQtGraph
High-performance spectrum visualization with interactive features.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from rf_spectrum_analyzer.config.settings import Settings


class SpectrumWidget(QWidget):
    """Widget for displaying real-time spectrum data."""
    
    frequency_clicked = Signal(float)  # Emitted when user clicks on frequency
    frequency_range_selected = Signal(float, float)  # Emitted when user selects frequency range
    signal_analysis_requested = Signal(dict)  # Emitted when user requests signal analysis
    
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        
        # Data storage
        self.spectrum_data = np.array([])
        self.frequency_axis = np.array([])
        
        # Display parameters
        self.min_db = settings.gui.spectrum_min_db
        self.max_db = settings.gui.spectrum_max_db
        self.ref_level = settings.gui.spectrum_ref_level
        
        # Peak detection
        self.peaks_enabled = True
        self.peak_threshold = -60.0  # dB
        self.detected_peaks = []
        
        # Frequency analysis markers - now only using LinearRegionItem
        self.freq_range_region = None
        self.markers_enabled = False
        self.f1_frequency = 0.0
        self.f2_frequency = 0.0
        
        # Peak hold
        self.peak_hold_enabled = False
        self.peak_hold_data = np.array([])
        
        self.setup_ui()
        self.setup_plot()
    
    def setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Create control bar
        control_layout = QHBoxLayout()
        
        # Title label
        title_label = QLabel("Spectrum")
        title_font = QFont()
        title_font.setBold(True)
        title_label.setFont(title_font)
        control_layout.addWidget(title_label)
        
        control_layout.addStretch()
        
        # Center frequency button
        self.center_button = QPushButton("Center")
        self.center_button.setToolTip("Move frequency range to center frequency")
        self.center_button.clicked.connect(self._on_center_button_clicked)
        control_layout.addWidget(self.center_button)
        
        # Signal analysis button
        self.analyze_button = QPushButton("Analyze Signal")
        self.analyze_button.setToolTip("Analyze signal in selected frequency range")
        self.analyze_button.clicked.connect(self._on_analyze_button_clicked)
        control_layout.addWidget(self.analyze_button)
        
        # Peak detection checkbox
        self.peaks_checkbox = QCheckBox("Show Peaks")
        self.peaks_checkbox.setChecked(self.peaks_enabled)
        self.peaks_checkbox.toggled.connect(self._on_peaks_toggled)
        control_layout.addWidget(self.peaks_checkbox)
        
        # Grid checkbox
        self.grid_checkbox = QCheckBox("Grid")
        self.grid_checkbox.setChecked(self.settings.gui.grid_enabled)
        self.grid_checkbox.toggled.connect(self._on_grid_toggled)
        control_layout.addWidget(self.grid_checkbox)
        
        layout.addLayout(control_layout)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('black' if self.settings.gui.theme == 'dark' else 'white')
        layout.addWidget(self.plot_widget)
        
        # Info label for displaying measurements
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("QLabel { color: #00ff00; background-color: rgba(0,0,0,128); padding: 2px; }")
        layout.addWidget(self.info_label)
    
    def setup_plot(self):
        """Setup the spectrum plot."""
        # Configure plot
        plot = self.plot_widget.getPlotItem()
        plot.setLabel('left', 'Power', units='dB')
        plot.setLabel('bottom', 'Frequency', units='Hz')
        plot.setTitle('RF Spectrum')
        
        # Set axis ranges
        # Validate initial Y range values
        if not (np.isfinite(self.min_db) and np.isfinite(self.max_db) and self.min_db < self.max_db):
            print(f"Warning: Invalid initial Y range: min={self.min_db}, max={self.max_db}, using defaults")
            self.min_db = -120.0  # Default minimum
            self.max_db = 0.0     # Default maximum
        
        # Convert to Python float for PyQtGraph compatibility
        plot.setYRange(float(self.min_db), float(self.max_db))
        
        # Enable grid
        if self.settings.gui.grid_enabled:
            plot.showGrid(x=True, y=True, alpha=self.settings.gui.grid_alpha)
        
        # Create spectrum curve
        self.spectrum_curve = plot.plot(
            pen=pg.mkPen(color='cyan', width=1),
            name='Spectrum'
        )
        
        # Create peak markers
        self.peak_scatter = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen(color='red', width=2),
            brush=pg.mkBrush(255, 0, 0, 120),
            symbol='o'
        )
        plot.addItem(self.peak_scatter)
        
        # Create reference level line
        self.ref_line = pg.InfiniteLine(
            angle=0,
            pos=self.ref_level,
            pen=pg.mkPen(color='yellow', style=Qt.DashLine, width=1),
            label='Ref Level',
            labelOpts={'color': 'yellow', 'position': 0.95}
        )
        plot.addItem(self.ref_line)
        
        # Create crosshair cursor
        # self.crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        # self.crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        # plot.addItem(self.crosshair_v, ignoreBounds=True)
        # plot.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Create frequency range region (replaces f1_marker and f2_marker)
        self.freq_range_region = pg.LinearRegionItem(
            values=[1544e6, 1546e6],  # Default range around center
            brush=pg.mkBrush(0, 255, 0, 30),  # Semi-transparent green
            pen=pg.mkPen(color='green', width=2),
            movable=True
        )
        
        # Initially hide frequency region
        self.freq_range_region.setVisible(False)
        
        plot.addItem(self.freq_range_region)
        
        # Connect region movement signal
        self.freq_range_region.sigRegionChanged.connect(self._on_frequency_region_changed)
        
        # Connect mouse events
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        
        # Set initial visibility
        # self.crosshair_v.setVisible(False)
        # self.crosshair_h.setVisible(False)
    
    def update_data(self, spectrum_data: np.ndarray):
        """Update spectrum display with new data."""
        if len(spectrum_data) == 0:
            return
        
        self.spectrum_data = spectrum_data
        
        # Generate frequency axis
        if len(self.frequency_axis) != len(spectrum_data):
            self._update_frequency_axis(len(spectrum_data))
        
        # Handle peak hold
        if self.peak_hold_enabled:
            if len(self.peak_hold_data) != len(spectrum_data):
                self.peak_hold_data = spectrum_data.copy()
            else:
                # Update peak hold data (keep maximum values)
                self.peak_hold_data = np.maximum(self.peak_hold_data, spectrum_data)
            display_data = self.peak_hold_data
        else:
            display_data = spectrum_data
        
        # Update spectrum curve
        self.spectrum_curve.setData(self.frequency_axis, display_data)
        
        # Detect and display peaks
        if self.peaks_enabled:
            self._detect_and_show_peaks(display_data)
        
        # Auto-scale Y axis occasionally
        if hasattr(self, '_auto_scale_counter'):
            self._auto_scale_counter += 1
        else:
            self._auto_scale_counter = 0
        
        if self._auto_scale_counter % 100 == 0:  # Every 100 updates
            self._auto_scale_y()
    
    def _update_frequency_axis(self, num_points: int):
        """Update frequency axis based on current settings."""
        sample_rate = self.settings.sdr.sample_rate
        center_freq = self.settings.sdr.center_frequency
        
        # Create frequency array
        freqs = np.linspace(-sample_rate/2, sample_rate/2, num_points)
        self.frequency_axis = freqs + center_freq
        
        # Update plot X axis
        plot = self.plot_widget.getPlotItem()
        plot.setXRange(self.frequency_axis[0], self.frequency_axis[-1])
    
    def _detect_and_show_peaks(self, spectrum_data=None):
        """Detect peaks in spectrum and display them."""
        data_to_analyze = spectrum_data if spectrum_data is not None else self.spectrum_data
        
        if len(data_to_analyze) < 10:
            return
        
        try:
            from scipy.signal import find_peaks
            
            # Find peaks above threshold
            peaks, properties = find_peaks(
                data_to_analyze,
                height=self.peak_threshold,
                distance=10,  # Minimum distance between peaks
                prominence=5  # Minimum prominence
            )
            
            if len(peaks) > 0:
                # Limit number of peaks displayed
                if len(peaks) > 20:
                    # Keep only the highest peaks
                    peak_heights = data_to_analyze[peaks]
                    sorted_indices = np.argsort(peak_heights)[-20:]
                    peaks = peaks[sorted_indices]
                
                # Update peak scatter plot
                peak_freqs = self.frequency_axis[peaks]
                peak_powers = data_to_analyze[peaks]
                
                # Convert to lists to avoid numpy array issues
                pos_data = [(float(f), float(p)) for f, p in zip(peak_freqs, peak_powers)]
                
                self.peak_scatter.setData(
                    pos=pos_data,
                    brush=[pg.mkBrush(255, 0, 0, 120)] * len(peaks)
                )
                
                self.detected_peaks = list(zip(peak_freqs, peak_powers))
            else:
                # Clear peaks
                self.peak_scatter.clear()
                self.detected_peaks = []
                
        except ImportError:
            # scipy not available, use simple peak detection
            self._simple_peak_detection()
        except Exception as e:
            # Fallback to simple detection on any error
            self._simple_peak_detection()
    
    def _simple_peak_detection(self):
        """Simple peak detection without scipy."""
        if len(self.spectrum_data) < 3:
            return
        
        peaks = []
        threshold = self.peak_threshold
        
        for i in range(1, len(self.spectrum_data) - 1):
            if (self.spectrum_data[i] > threshold and
                self.spectrum_data[i] > self.spectrum_data[i-1] and
                self.spectrum_data[i] > self.spectrum_data[i+1]):
                peaks.append(i)
        
        if len(peaks) > 0:
            # Limit number of peaks
            if len(peaks) > 20:
                peak_heights = self.spectrum_data[peaks]
                sorted_indices = np.argsort(peak_heights)[-20:]
                peaks = [peaks[i] for i in sorted_indices]
            
            peak_freqs = self.frequency_axis[peaks]
            peak_powers = self.spectrum_data[peaks]
            
            # Convert to lists to avoid numpy array issues
            pos_data = [(float(f), float(p)) for f, p in zip(peak_freqs, peak_powers)]
            
            self.peak_scatter.setData(
                pos=pos_data,
                brush=[pg.mkBrush(255, 0, 0, 120)] * len(peaks)
            )
            
            self.detected_peaks = list(zip(peak_freqs, peak_powers))
        else:
            self.peak_scatter.clear()
            self.detected_peaks = []
    
    def _auto_scale_y(self):
        """Auto-scale Y axis based on current data."""
        if len(self.spectrum_data) == 0:
            return
        
        try:
            # Check if data contains valid numeric values
            if not np.isfinite(self.spectrum_data).any():
                return
                
            data_min = np.nanmin(self.spectrum_data)
            data_max = np.nanmax(self.spectrum_data)
            
            # Ensure valid numeric values
            if not (np.isfinite(data_min) and np.isfinite(data_max)):
                return
            
            # Ensure min < max
            if data_min >= data_max:
                # If all values are the same, create a small range
                data_range = max(1.0, abs(data_min))  # Minimum 1 dB range
                data_min = data_min - data_range
                data_max = data_max + data_range
            
            # Add some margin
            margin = max(1.0, (data_max - data_min) * 0.1)  # Minimum 1 dB margin
            y_min = data_min - margin
            y_max = data_max + margin
            
            # Ensure final values are valid and min < max
            if not (np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max):
                return
            
            # Update Y range if significantly different
            plot = self.plot_widget.getPlotItem()
            current_range = plot.getViewBox().viewRange()[1]
            
            if (abs(current_range[0] - y_min) > 10 or 
                abs(current_range[1] - y_max) > 10):
                # Add extra validation before setYRange call
                if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
                    # Convert to Python float to ensure PyQtGraph compatibility
                    y_min_f = float(y_min)
                    y_max_f = float(y_max)
                    plot.setYRange(y_min_f, y_max_f, padding=0)
                
        except Exception as e:
            # Log the error but don't crash - silently handle  
            pass
    
    def _on_mouse_moved(self, pos):
        """Handle mouse movement for crosshair."""
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
            
            # Update crosshair position
            # self.crosshair_v.setPos(mouse_point.x())
            # self.crosshair_h.setPos(mouse_point.y())
            # self.crosshair_v.setVisible(True)
            # self.crosshair_h.setVisible(True)
            
            # Update info label
            freq_mhz = mouse_point.x() / 1e6
            power_db = mouse_point.y()
            self.info_label.setText(f"Freq: {freq_mhz:.3f} MHz, Power: {power_db:.1f} dB")
        else:
            # self.crosshair_v.setVisible(False)
            # self.crosshair_h.setVisible(False)
            self.info_label.setText("")
    
    def _on_mouse_clicked(self, event):
        """Handle mouse click for frequency selection."""
        if event.button() == Qt.LeftButton:
            pos = event.scenePos()
            if self.plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
                frequency = mouse_point.x()
                self.frequency_clicked.emit(frequency)
    
    def _on_peaks_toggled(self, checked: bool):
        """Handle peak detection toggle."""
        self.peaks_enabled = checked
        self.peak_scatter.setVisible(checked)
        if not checked:
            self.peak_scatter.clear()
    
    def _on_grid_toggled(self, checked: bool):
        """Handle grid toggle."""
        plot = self.plot_widget.getPlotItem()
        plot.showGrid(x=checked, y=checked, alpha=self.settings.gui.grid_alpha)
        self.settings.gui.grid_enabled = checked
    
    def _on_center_button_clicked(self):
        """Handle center button click - move frequency range to center frequency."""
        center_freq = self.settings.sdr.center_frequency
        sample_rate = self.settings.sdr.sample_rate
        
        # Create a reasonable range around center frequency (10% of sample rate)
        range_width = sample_rate * 0.1
        f1 = center_freq - range_width / 2
        f2 = center_freq + range_width / 2
        
        # Update the frequency range
        self.set_frequency_range(f1, f2)
        
        # Ensure markers are visible
        if not self.markers_enabled:
            self.set_frequency_markers_enabled(True)
    
    def _on_analyze_button_clicked(self):
        """Handle analyze signal button click."""
        if not self.markers_enabled:
            # Show message that frequency range needs to be selected
            self.info_label.setText("Please enable frequency markers first")
            return
        
        analysis_request = self.request_signal_analysis()
        if analysis_request is None:
            self.info_label.setText("No signal detected in selected range")
        else:
            self.info_label.setText(f"Analyzing signal at {analysis_request['center_freq']/1e6:.3f} MHz...")
    
    def set_reference_level(self, ref_level: float):
        """Set reference level line."""
        self.ref_level = ref_level
        self.ref_line.setPos(ref_level)
    
    def set_y_range(self, min_db: float, max_db: float):
        """Set Y axis range."""
        # Ensure valid numeric values
        if not (np.isfinite(min_db) and np.isfinite(max_db)):
            return
        
        # Ensure min < max
        if min_db >= max_db:
            min_db, max_db = max_db, min_db
        
        # Convert to Python float for PyQtGraph compatibility
        self.min_db = float(min_db)
        self.max_db = float(max_db)
        plot = self.plot_widget.getPlotItem()
        # Set initial Y range with validation
        min_val = float(self.min_db) if np.isfinite(self.min_db) else -100.0
        max_val = float(self.max_db) if np.isfinite(self.max_db) else 0.0
        if min_val >= max_val:
            min_val, max_val = max_val - 10.0, min_val + 10.0
        plot.setYRange(min_val, max_val)
    
    def set_peak_threshold(self, threshold: float):
        """Set peak detection threshold."""
        self.peak_threshold = threshold
    
    def export_data(self) -> dict:
        """Export current spectrum data."""
        return {
            'frequency': self.frequency_axis.tolist() if len(self.frequency_axis) > 0 else [],
            'power': self.spectrum_data.tolist() if len(self.spectrum_data) > 0 else [],
            'peaks': self.detected_peaks,
            'settings': {
                'min_db': self.min_db,
                'max_db': self.max_db,
                'ref_level': self.ref_level,
                'peak_threshold': self.peak_threshold
            }
        }
    
    def update_settings(self, settings: Settings):
        """Update widget with new settings."""
        self.settings = settings
        
        # Update theme
        bg_color = 'black' if settings.gui.theme == 'dark' else 'white'
        self.plot_widget.setBackground(bg_color)
        
        # Update grid
        plot = self.plot_widget.getPlotItem()
        plot.showGrid(x=settings.gui.grid_enabled, y=settings.gui.grid_enabled, 
                     alpha=settings.gui.grid_alpha)
        
        # Update frequency axis if settings changed
        if len(self.spectrum_data) > 0:
            self._update_frequency_axis(len(self.spectrum_data))
    
    def clear_data(self):
        """Clear all displayed data."""
        self.spectrum_curve.clear()
        self.peak_scatter.clear()
        self.spectrum_data = np.array([])
        self.frequency_axis = np.array([])
        self.detected_peaks = []
        self.info_label.setText("")
        
        # Clear peak hold data
        self.peak_hold_data = np.array([])
    
    # Frequency Analysis Methods
    def set_frequency_markers_enabled(self, enabled: bool):
        """Enable or disable frequency range display."""
        self.markers_enabled = enabled
        self.freq_range_region.setVisible(enabled)
    
    def set_frequency_range(self, f1: float, f2: float):
        """Set frequency range region."""
        self.f1_frequency = f1
        self.f2_frequency = f2
        
        self.freq_range_region.setRegion([f1, f2])
    
    def set_peak_hold_enabled(self, enabled: bool):
        """Enable or disable peak hold functionality."""
        self.peak_hold_enabled = enabled
        if not enabled:
            self.peak_hold_data = np.array([])
    
    def reset_peak_hold(self):
        """Reset peak hold data."""
        self.peak_hold_data = np.array([])
    
    def _on_frequency_region_changed(self):
        """Handle frequency region change."""
        f1, f2 = self.freq_range_region.getRegion()
        
        self.f1_frequency = f1
        self.f2_frequency = f2
        
        self.frequency_range_selected.emit(f1, f2)
    
    def get_frequency_range(self):
        """Get current frequency range."""
        return self.f1_frequency, self.f2_frequency
    
    def highlight_frequency_range(self, f1: float, f2: float):
        """Highlight a specific frequency range on the spectrum."""
        self.set_frequency_range(f1, f2)
        if not self.markers_enabled:
            self.set_frequency_markers_enabled(True)
    
    def get_frequency_range_data(self):
        """Get spectrum data within the frequency range region for analysis."""
        if not self.markers_enabled or len(self.spectrum_data) == 0 or len(self.frequency_axis) == 0:
            return None, None, None
        
        f1, f2 = self.freq_range_region.getRegion()
        
        # Ensure f1 < f2
        if f1 > f2:
            f1, f2 = f2, f1
        
        # Find indices within the frequency range
        mask = (self.frequency_axis >= f1) & (self.frequency_axis <= f2)
        
        if not np.any(mask):
            return None, None, None
        
        # Extract data within range
        freq_range = self.frequency_axis[mask]
        power_range = self.spectrum_data[mask]
        
        # Calculate range statistics
        range_stats = {
            'center_freq': (f1 + f2) / 2,
            'bandwidth': f2 - f1,
            'freq_start': f1,
            'freq_end': f2,
            'peak_power': np.max(power_range),
            'avg_power': np.mean(power_range),
            'peak_freq': freq_range[np.argmax(power_range)],
            'num_samples': len(freq_range),
            'snr_estimate': self._estimate_snr(power_range),
            'signal_present': self._detect_signal_presence(power_range)
        }
        
        return freq_range, power_range, range_stats
    
    def _estimate_snr(self, power_data):
        """Estimate Signal-to-Noise Ratio from power data."""
        if len(power_data) == 0:
            return None
        
        try:
            # Simple SNR estimation: (peak - median) as signal, std as noise
            peak_power = np.max(power_data)
            noise_floor = np.median(power_data)
            noise_std = np.std(power_data)
            
            # SNR = (Signal - Noise) / Noise_std
            snr_db = (peak_power - noise_floor) / max(noise_std, 0.1)
            return float(snr_db)
        except:
            return None
    
    def _detect_signal_presence(self, power_data):
        """Detect if there's a significant signal in the power data."""
        if len(power_data) == 0:
            return False
        
        try:
            # Check if peak power is significantly above noise floor
            peak_power = np.max(power_data)
            noise_floor = np.median(power_data)
            threshold = 10.0  # 10 dB above noise floor
            
            return (peak_power - noise_floor) > threshold
        except:
            return False
    
    def request_signal_analysis(self):
        """Request detailed signal analysis for the selected frequency range."""
        freq_data, power_data, stats = self.get_frequency_range_data()
        
        if freq_data is None or not stats['signal_present']:
            return None
        
        # Emit signal to request IQ data analysis
        analysis_request = {
            'center_freq': stats['center_freq'],
            'bandwidth': stats['bandwidth'],
            'freq_range': (stats['freq_start'], stats['freq_end']),
            'power_stats': stats,
            'analysis_type': 'full',  # modulation, coding, demodulation
            'timestamp': np.datetime64('now')
        }
        
        # This will be connected to the main app for full signal processing
        if hasattr(self, 'signal_analysis_requested'):
            self.signal_analysis_requested.emit(analysis_request)
        
        return analysis_request