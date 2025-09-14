# Create GUI integration for the complete SDR application
gui_integration_code = '''
"""
GUI Integration for Burst Detection

Tích hợp burst detection vào GUI của Complete SDR Application
"""

import numpy as np
import time
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from typing import Dict, List, Optional

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

# Import burst detection components
try:
    from burst_detection_integration import BurstDetectionPipeline, BurstDetectionWidget
    BURST_DETECTION_AVAILABLE = True
except ImportError:  
    BURST_DETECTION_AVAILABLE = False

class BurstDetectionControlPanel(QWidget):
    """Control panel for burst detection settings"""
    
    # Signals
    detection_started = Signal()
    detection_stopped = Signal()
    settings_changed = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # Current settings
        self.current_settings = {
            'threshold_db': 18,
            'fft_size': 1024,
            'history_size': 100,
            'lookahead': 5,
            'burst_pre_len': 10,
            'burst_post_len': 10
        }
    
    def setup_ui(self):
        """Setup burst detection control UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🎯 Burst Detection & Demodulation")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2E8B57;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Detection control
        control_group = QGroupBox("Detection Control")
        control_layout = QGridLayout(control_group)
        
        self.enable_detection_cb = QCheckBox("Enable Burst Detection")
        self.enable_detection_cb.setChecked(True)
        self.enable_detection_cb.toggled.connect(self.on_detection_toggled)
        control_layout.addWidget(self.enable_detection_cb, 0, 0, 1, 2)
        
        self.start_detection_btn = QPushButton("▶️ Start Detection")
        self.start_detection_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_detection_btn, 1, 0)
        
        self.stop_detection_btn = QPushButton("⏹️ Stop Detection")
        self.stop_detection_btn.clicked.connect(self.stop_detection)
        self.stop_detection_btn.setEnabled(False)
        control_layout.addWidget(self.stop_detection_btn, 1, 1)
        
        layout.addWidget(control_group)
        
        # Detection parameters
        params_group = QGroupBox("Detection Parameters")
        params_layout = QFormLayout(params_group)
        
        # Threshold
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(10.0, 30.0)
        self.threshold_spinbox.setValue(18.0)
        self.threshold_spinbox.setSuffix(" dB")
        self.threshold_spinbox.valueChanged.connect(self.on_params_changed)
        params_layout.addRow("Detection Threshold:", self.threshold_spinbox)
        
        # FFT Size
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(["512", "1024", "2048", "4096"])
        self.fft_size_combo.setCurrentText("1024")
        self.fft_size_combo.currentTextChanged.connect(self.on_params_changed)
        params_layout.addRow("FFT Size:", self.fft_size_combo)
        
        # History size
        self.history_size_spinbox = QSpinBox()
        self.history_size_spinbox.setRange(50, 500)
        self.history_size_spinbox.setValue(100)
        self.history_size_spinbox.valueChanged.connect(self.on_params_changed)
        params_layout.addRow("History Size:", self.history_size_spinbox)
        
        # Lookahead
        self.lookahead_spinbox = QSpinBox()
        self.lookahead_spinbox.setRange(2, 20)
        self.lookahead_spinbox.setValue(5)
        self.lookahead_spinbox.valueChanged.connect(self.on_params_changed)
        params_layout.addRow("Lookahead FFTs:", self.lookahead_spinbox)
        
        layout.addWidget(params_group)
        
        # Frame type filter
        filter_group = QGroupBox("Frame Type Filter")
        filter_layout = QVBoxLayout(filter_group)
        
        self.show_all_cb = QCheckBox("Show All Frame Types")
        self.show_all_cb.setChecked(True)
        filter_layout.addWidget(self.show_all_cb)
        
        self.frame_type_group = QButtonGroup()
        frame_types = ["IRA", "IBC", "TLC", "DATA", "UNKNOWN"]
        
        for frame_type in frame_types:
            cb = QCheckBox(frame_type)
            cb.setChecked(True)
            filter_layout.addWidget(cb)
            self.frame_type_group.addButton(cb)
        
        layout.addWidget(filter_group)
        
        # Statistics display
        stats_group = QGroupBox("Detection Statistics")
        stats_layout = QGridLayout(stats_group)
        
        # Create labels for statistics
        self.stats_labels = {}
        stats_items = [
            ("Samples Processed", "samples_processed"),
            ("Bursts Detected", "bursts_detected"), 
            ("Frames Decoded", "frames_decoded"),
            ("Success Rate", "decode_success_rate"),
            ("Processing Rate", "processing_rate")
        ]
        
        for i, (display_name, key) in enumerate(stats_items):
            label = QLabel(f"{display_name}:")
            value_label = QLabel("0")
            value_label.setStyleSheet("font-weight: bold; color: #2E8B57;")
            
            stats_layout.addWidget(label, i, 0)
            stats_layout.addWidget(value_label, i, 1)
            
            self.stats_labels[key] = value_label
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
    
    def on_detection_toggled(self, checked):
        """Handle detection enable/disable"""
        self.start_detection_btn.setEnabled(checked)
        if not checked:
            self.stop_detection()
    
    def start_detection(self):
        """Start burst detection"""
        self.start_detection_btn.setEnabled(False)
        self.stop_detection_btn.setEnabled(True)
        self.detection_started.emit()
    
    def stop_detection(self):
        """Stop burst detection"""
        self.start_detection_btn.setEnabled(True)
        self.stop_detection_btn.setEnabled(False)
        self.detection_stopped.emit()
    
    def on_params_changed(self):
        """Handle parameter changes"""
        self.current_settings.update({
            'threshold_db': self.threshold_spinbox.value(),
            'fft_size': int(self.fft_size_combo.currentText()),
            'history_size': self.history_size_spinbox.value(),
            'lookahead': self.lookahead_spinbox.value()
        })
        
        self.settings_changed.emit(self.current_settings)
    
    def update_statistics(self, stats):
        """Update statistics display"""
        for key, label in self.stats_labels.items():
            if key in stats:
                value = stats[key]
                
                if key == 'decode_success_rate':
                    label.setText(f"{value:.1%}")
                elif key == 'processing_rate':
                    if value > 1e6:
                        label.setText(f"{value/1e6:.2f} MS/s")
                    elif value > 1e3:
                        label.setText(f"{value/1e3:.2f} kS/s")
                    else:
                        label.setText(f"{value:.1f} S/s")
                elif isinstance(value, (int, float)):
                    if value > 1e6:
                        label.setText(f"{value/1e6:.2f}M")
                    elif value > 1e3:
                        label.setText(f"{value/1e3:.2f}k")
                    else:
                        label.setText(f"{value:,.0f}")
                else:
                    label.setText(str(value))

class BurstDetectionResultsPanel(QWidget):
    """Panel displaying burst detection results"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # Data storage
        self.detected_frames = []
        self.max_frames = 100
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # Update every second
    
    def setup_ui(self):
        """Setup results display UI"""
        layout = QVBoxLayout(self)
        
        # Title and controls
        header_layout = QHBoxLayout()
        
        title = QLabel("📡 Detected Frames")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.clear_frames)
        header_layout.addWidget(self.clear_btn)
        
        self.export_btn = QPushButton("💾 Export")
        self.export_btn.clicked.connect(self.export_frames)
        header_layout.addWidget(self.export_btn)
        
        layout.addLayout(header_layout)
        
        # Frames table
        self.frames_table = QTableWidget()
        self.frames_table.setColumnCount(7)
        self.frames_table.setHorizontalHeaderLabels([
            "Time", "Frame Type", "Confidence", "Bits", "Signal Level", "Burst ID", "Status"
        ])
        
        # Set column widths
        header = self.frames_table.horizontalHeader()
        header.setResizeMode(0, QHeaderView.ResizeToContents)
        header.setResizeMode(1, QHeaderView.ResizeToContents)
        header.setResizeMode(2, QHeaderView.ResizeToContents)
        header.setResizeMode(3, QHeaderView.ResizeToContents)
        header.setResizeMode(4, QHeaderView.ResizeToContents)
        header.setResizeMode(5, QHeaderView.Stretch)
        header.setResizeMode(6, QHeaderView.ResizeToContents)
        
        # Enable sorting
        self.frames_table.setSortingEnabled(True)
        
        # Enable selection
        self.frames_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.frames_table.itemSelectionChanged.connect(self.on_frame_selected)
        
        layout.addWidget(self.frames_table)
        
        # Frame details
        details_group = QGroupBox("Frame Details")
        details_layout = QVBoxLayout(details_group)
        
        self.frame_details_text = QTextEdit()
        self.frame_details_text.setMaximumHeight(150)
        self.frame_details_text.setReadOnly(True)
        self.frame_details_text.setStyleSheet("font-family: Consolas; font-size: 9pt;")
        details_layout.addWidget(self.frame_details_text)
        
        layout.addWidget(details_group)
    
    def add_detected_frame(self, frame_info):
        """Add newly detected frame to display"""
        # Add timestamp
        frame_info['display_time'] = time.strftime("%H:%M:%S", time.localtime())
        
        # Add to list
        self.detected_frames.append(frame_info)
        
        # Limit list size
        if len(self.detected_frames) > self.max_frames:
            self.detected_frames = self.detected_frames[-self.max_frames:]
        
        # Update display immediately for new frames
        self.update_display()
    
    def update_display(self):
        """Update the frames table display"""
        # Clear and repopulate table
        self.frames_table.setRowCount(len(self.detected_frames))
        
        for row, frame in enumerate(self.detected_frames):
            # Time
            time_item = QTableWidgetItem(frame.get('display_time', ''))
            self.frames_table.setItem(row, 0, time_item)
            
            # Frame type
            frame_type = frame.get('frame_type', 'UNKNOWN')
            type_item = QTableWidgetItem(frame_type)
            
            # Color code by frame type
            if frame_type == 'IRA':
                type_item.setBackground(QColor(144, 238, 144))  # Light green
            elif frame_type == 'TLC':
                type_item.setBackground(QColor(173, 216, 230))  # Light blue
            elif frame_type == 'DATA':
                type_item.setBackground(QColor(255, 218, 185))  # Peach
            
            self.frames_table.setItem(row, 1, type_item)
            
            # Confidence
            confidence = frame.get('confidence', 0.0)
            conf_item = QTableWidgetItem(f"{confidence:.1%}")
            
            # Color code by confidence
            if confidence > 0.8:
                conf_item.setForeground(QColor(0, 128, 0))  # Green
            elif confidence > 0.5:
                conf_item.setForeground(QColor(255, 140, 0))  # Orange
            else:
                conf_item.setForeground(QColor(255, 0, 0))  # Red
            
            self.frames_table.setItem(row, 2, conf_item)
            
            # Bit count
            bit_count = frame.get('bit_count', 0)
            bits_item = QTableWidgetItem(str(bit_count))
            self.frames_table.setItem(row, 3, bits_item)
            
            # Signal level
            signal_level = frame.get('signal_level', 0.0)
            level_item = QTableWidgetItem(f"{signal_level:.3f}")
            self.frames_table.setItem(row, 4, level_item)
            
            # Burst ID
            burst_id = frame.get('burst_id', 'N/A')
            id_item = QTableWidgetItem(str(burst_id))
            self.frames_table.setItem(row, 5, id_item)
            
            # Status
            status = "✅ OK" if confidence > 0.5 else "⚠️ Low Conf"
            status_item = QTableWidgetItem(status)
            self.frames_table.setItem(row, 6, status_item)
        
        # Auto-scroll to bottom for new entries
        if self.detected_frames:
            self.frames_table.scrollToBottom()
    
    def on_frame_selected(self):
        """Handle frame selection"""
        selected_rows = self.frames_table.selectionModel().selectedRows()
        
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self.detected_frames):
                frame = self.detected_frames[row]
                self.display_frame_details(frame)
    
    def display_frame_details(self, frame):
        """Display detailed frame information"""
        details = "=== FRAME DETAILS ===\\n\\n"
        
        # Basic info
        details += f"Frame Type: {frame.get('frame_type', 'UNKNOWN')}\\n"
        details += f"Confidence: {frame.get('confidence', 0):.1%}\\n"
        details += f"Signal Level: {frame.get('signal_level', 0):.6f}\\n"
        details += f"Bit Count: {frame.get('bit_count', 0)}\\n"
        details += f"Burst ID: {frame.get('burst_id', 'N/A')}\\n"
        
        # Metadata
        if 'metadata' in frame:
            metadata = frame['metadata']
            details += f"\\n=== BURST METADATA ===\\n"
            details += f"Center Frequency: {metadata.get('center_freq', 0):.0f} Hz\\n"
            details += f"Bandwidth: {metadata.get('bandwidth_est', 0):.0f} Hz\\n" 
            details += f"Magnitude: {metadata.get('magnitude', 0):.1f} dB\\n"
            details += f"Noise Floor: {metadata.get('noise_floor', 0):.1f} dB\\n"
        
        # Raw bits (first 100)
        if 'raw_bits' in frame and len(frame['raw_bits']) > 0:
            bits = frame['raw_bits']
            details += f"\\n=== RAW BITS (first 100) ===\\n"
            bit_str = ''.join(map(str, bits[:100]))
            # Format in groups of 8
            formatted_bits = ' '.join([bit_str[i:i+8] for i in range(0, len(bit_str), 8)])
            details += formatted_bits
            
            if len(bits) > 100:
                details += f"\\n... and {len(bits) - 100} more bits"
        
        # Parsed data
        if 'parsed_data' in frame:
            parsed = frame['parsed_data']
            details += f"\\n=== PARSED DATA ===\\n"
            details += f"Description: {parsed.get('description', 'N/A')}\\n"
            
            if 'header_bits' in parsed:
                header_str = ''.join(map(str, parsed['header_bits'][:16]))
                details += f"Header: {header_str}\\n"
        
        self.frame_details_text.setPlainText(details)
    
    def clear_frames(self):
        """Clear all detected frames"""
        self.detected_frames = []
        self.frames_table.setRowCount(0)
        self.frame_details_text.clear()
    
    def export_frames(self):
        """Export detected frames to file"""
        if not self.detected_frames:
            QMessageBox.information(self, "Export", "No frames to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Detected Frames", 
            f"burst_frames_{int(time.time())}.txt",
            "Text Files (*.txt);;CSV Files (*.csv)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("# Burst Detection Results\\n")
                    f.write(f"# Export Time: {time.ctime()}\\n")
                    f.write(f"# Total Frames: {len(self.detected_frames)}\\n\\n")
                    
                    for i, frame in enumerate(self.detected_frames):
                        f.write(f"=== FRAME {i+1} ===\\n")
                        f.write(f"Time: {frame.get('display_time', '')}\\n") 
                        f.write(f"Type: {frame.get('frame_type', 'UNKNOWN')}\\n")
                        f.write(f"Confidence: {frame.get('confidence', 0):.3f}\\n")
                        f.write(f"Bits: {frame.get('bit_count', 0)}\\n")
                        f.write(f"Signal: {frame.get('signal_level', 0):.6f}\\n")
                        f.write(f"Burst ID: {frame.get('burst_id', 'N/A')}\\n")
                        
                        if 'raw_bits' in frame:
                            bit_str = ''.join(map(str, frame['raw_bits']))
                            f.write(f"Raw Bits: {bit_str}\\n")
                        
                        f.write("\\n")
                
                QMessageBox.information(self, "Export", f"Exported {len(self.detected_frames)} frames to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

class BurstDetectionVisualizationPanel(QWidget):
    """Visualization panel for burst detection"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # Visualization data
        self.burst_history = []
        self.max_history = 1000
        
    def setup_ui(self):
        """Setup visualization UI"""
        if not PYQTGRAPH_AVAILABLE:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("PyQtGraph not available for visualization"))
            return
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📊 Burst Detection Visualization")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(title)
        
        # Plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        
        # Time domain plot
        self.time_plot = self.plot_widget.addPlot(title="Burst Detection Timeline")
        self.time_plot.setLabel('left', 'Frame Count')
        self.time_plot.setLabel('bottom', 'Time (s)')
        self.time_plot.showGrid(True, True)
        
        self.plot_widget.nextRow()
        
        # Frequency domain plot
        self.freq_plot = self.plot_widget.addPlot(title="Burst Frequency Distribution")
        self.freq_plot.setLabel('left', 'Count')
        self.freq_plot.setLabel('bottom', 'Frequency (Hz)')
        self.freq_plot.showGrid(True, True)
        
        self.plot_widget.nextRow()
        
        # Confidence histogram
        self.conf_plot = self.plot_widget.addPlot(title="Confidence Distribution")
        self.conf_plot.setLabel('left', 'Count')
        self.conf_plot.setLabel('bottom', 'Confidence')
        self.conf_plot.showGrid(True, True)
        
        layout.addWidget(self.plot_widget)
    
    def add_burst_data(self, frame_info):
        """Add burst data for visualization"""
        if not PYQTGRAPH_AVAILABLE:
            return
        
        # Add to history
        timestamp = time.time()
        burst_data = {
            'timestamp': timestamp,
            'confidence': frame_info.get('confidence', 0),
            'frequency': frame_info.get('metadata', {}).get('center_freq', 0),
            'frame_type': frame_info.get('frame_type', 'UNKNOWN')
        }
        
        self.burst_history.append(burst_data)
        
        # Limit history
        if len(self.burst_history) > self.max_history:
            self.burst_history = self.burst_history[-self.max_history:]
        
        # Update plots periodically
        if len(self.burst_history) % 10 == 0:  # Update every 10 bursts
            self.update_plots()
    
    def update_plots(self):
        """Update visualization plots"""
        if not PYQTGRAPH_AVAILABLE or not self.burst_history:
            return
        
        try:
            # Time domain plot - frame count over time
            time_data = [b['timestamp'] for b in self.burst_history]
            if time_data:
                min_time = min(time_data)
                relative_times = [(t - min_time) for t in time_data]
                
                # Create histogram of frames per second
                time_bins = np.arange(0, max(relative_times) + 1, 1)
                counts, _ = np.histogram(relative_times, bins=time_bins)
                
                self.time_plot.clear()
                self.time_plot.plot(time_bins[:-1], counts, pen='b', symbol='o')
            
            # Frequency distribution
            frequencies = [b['frequency'] for b in self.burst_history if b['frequency'] != 0]
            if frequencies:
                freq_counts, freq_bins = np.histogram(frequencies, bins=20)
                self.freq_plot.clear()
                self.freq_plot.plot(freq_bins[:-1], freq_counts, stepMode=True, fillLevel=0, brush=(0,0,255,150))
            
            # Confidence distribution
            confidences = [b['confidence'] for b in self.burst_history]
            if confidences:
                conf_counts, conf_bins = np.histogram(confidences, bins=20, range=(0, 1))
                self.conf_plot.clear()
                self.conf_plot.plot(conf_bins[:-1], conf_counts, stepMode=True, fillLevel=0, brush=(0,255,0,150))
        
        except Exception as e:
            print(f"Plot update error: {e}")

# Integration test
def test_gui_integration():
    """Test GUI integration components"""
    print("🧪 Testing GUI Integration Components")
    
    app = QApplication([])
    
    # Test control panel
    control_panel = BurstDetectionControlPanel()
    control_panel.show()
    
    # Test results panel
    results_panel = BurstDetectionResultsPanel()
    results_panel.show()
    
    # Add some test frames
    test_frames = [
        {
            'frame_type': 'IRA',
            'confidence': 0.95,
            'bit_count': 180,
            'signal_level': 0.245,
            'burst_id': '123456789',
            'raw_bits': np.random.randint(0, 2, 180).tolist()
        },
        {
            'frame_type': 'TLC', 
            'confidence': 0.78,
            'bit_count': 240,
            'signal_level': 0.189,
            'burst_id': '123456790',
            'raw_bits': np.random.randint(0, 2, 240).tolist()
        }
    ]
    
    for frame in test_frames:
        results_panel.add_detected_frame(frame)
    
    # Test visualization panel
    if PYQTGRAPH_AVAILABLE:
        viz_panel = BurstDetectionVisualizationPanel()
        viz_panel.show()
        
        for frame in test_frames:
            viz_panel.add_burst_data(frame)
    
    print("✅ GUI components displayed")
    print("Close windows to complete test")
    
    return app.exec()

if __name__ == "__main__":
    test_gui_integration()
'''

# Write the GUI integration module
with open('burst_detection_gui.py', 'w', encoding='utf-8') as f:
    f.write(gui_integration_code)

print("✅ Created burst_detection_gui.py")
print("📁 File size:", len(gui_integration_code), "characters")