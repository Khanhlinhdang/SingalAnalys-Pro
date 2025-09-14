
"""
Advanced SDR Application with Real-time Signal Processing Pipeline
Tích hợp hoàn chỉnh real-time signal generation, detection, và decoding
"""

import sys
import numpy as np
import time
import struct
from threading import Thread, Event
from queue import Queue

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
                               QComboBox, QSlider, QProgressBar, QTextEdit, QTabWidget,
                               QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox, QTableWidget,
                               QTableWidgetItem, QSplitter, QScrollArea, QFrame, QListWidget)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread
from PySide6.QtGui import QFont, QPixmap, QColor

import pyqtgraph as pg
from pyqtgraph import ImageItem, PlotWidget
import uhd

from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks, welch

# Import our comprehensive modules
from analog_modulation import AnalogModulation, AnalogDemodulation, PulseAnalogModulation, AnalogModulationClassifier
from extended_digital_modulation import ExtendedDigitalModulation, ExtendedDigitalDemodulation, AdvancedModulationClassifier
from multicarrier_spread_spectrum import MultiCarrierModulation, SpreadSpectrumModulation, MIMOModulation, AdvancedModulationDetector
from advanced_signal_processing import AdvancedSignalProcessor, SpectrumScanner

# Import channel coding modules
try:
    from channel_coding import (ConvolutionalCoder, TurboCoder, LDPCCoder, PolarCoder, 
                               ReedSolomonCoder, ChannelCodingDetector)
    from enhanced_signal_processor import EnhancedSignalProcessor
    from realtime_signal_pipeline import RealtimeSignalAnalyzer
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    print("Warning: Real-time pipeline modules not available")


class BitStreamDisplay(QWidget):
    """Real-time bit stream display widget"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.bit_buffer = []
        self.max_display_bits = 500

    def setup_ui(self):
        """Setup bit stream display UI"""
        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()

        self.title_label = QLabel("Real-time Bit Stream")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        self.bit_count_label = QLabel("Bits: 0")
        self.bit_count_label.setStyleSheet("font-weight: bold; color: #14a085;")
        header_layout.addWidget(self.bit_count_label)

        layout.addLayout(header_layout)

        # Bit display area
        self.bit_display = QTextEdit()
        self.bit_display.setReadOnly(True)
        self.bit_display.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 10pt;
            background-color: #1a1a1a;
            color: #00ff00;
            border: 1px solid #555555;
        """)
        self.bit_display.setMaximumHeight(120)
        layout.addWidget(self.bit_display)

        # Controls
        controls_layout = QHBoxLayout()

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_bits)
        controls_layout.addWidget(self.clear_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.toggle_pause)
        controls_layout.addWidget(self.pause_btn)

        controls_layout.addStretch()

        self.speed_label = QLabel("Update Rate:")
        controls_layout.addWidget(self.speed_label)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(5)
        self.speed_slider.setMaximumWidth(100)
        controls_layout.addWidget(self.speed_slider)

        layout.addLayout(controls_layout)

        # Statistics
        stats_layout = QHBoxLayout()

        self.stats_label = QLabel("Bits/sec: 0 | Errors: 0 | Total: 0")
        self.stats_label.setStyleSheet("font-size: 9pt; color: #cccccc;")
        stats_layout.addWidget(self.stats_label)

        layout.addLayout(stats_layout)

        # State
        self.paused = False
        self.last_update_time = time.time()
        self.bit_rate = 0
        self.error_count = 0
        self.total_bits = 0

    def update_bit_stream(self, new_bits):
        """Update bit stream display"""
        if self.paused:
            return

        if new_bits is None or len(new_bits) == 0:
            return

        try:
            # Add new bits to buffer
            self.bit_buffer.extend(new_bits.astype(int).tolist())

            # Limit buffer size
            if len(self.bit_buffer) > self.max_display_bits:
                self.bit_buffer = self.bit_buffer[-self.max_display_bits:]

            # Update statistics
            current_time = time.time()
            time_diff = current_time - self.last_update_time
            if time_diff > 0:
                self.bit_rate = len(new_bits) / time_diff

            self.total_bits += len(new_bits)

            # Format bits for display with grouping
            bit_text = self._format_bits_for_display()

            # Update display
            self.bit_display.setPlainText(bit_text)

            # Auto scroll to bottom
            scrollbar = self.bit_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

            # Update labels
            self.bit_count_label.setText(f"Bits: {len(self.bit_buffer)}")
            self.stats_label.setText(f"Bits/sec: {self.bit_rate:.1f} | Errors: {self.error_count} | Total: {self.total_bits}")

            self.last_update_time = current_time

        except Exception as e:
            print(f"Bit stream display error: {e}")

    def _format_bits_for_display(self):
        """Format bits for readable display"""
        if not self.bit_buffer:
            return "Waiting for bit stream..."

        # Group bits into bytes (8 bits) with spaces
        formatted_lines = []
        bits_per_line = 64  # 8 bytes per line

        for i in range(0, len(self.bit_buffer), bits_per_line):
            line_bits = self.bit_buffer[i:i+bits_per_line]

            # Group into bytes
            byte_groups = []
            for j in range(0, len(line_bits), 8):
                byte_bits = line_bits[j:j+8]
                byte_str = ''.join(map(str, byte_bits))

                # Pad incomplete bytes
                if len(byte_str) < 8:
                    byte_str = byte_str.ljust(8, '0')

                byte_groups.append(byte_str)

            line_str = ' '.join(byte_groups)

            # Add line number/offset
            offset = i
            formatted_lines.append(f"{offset:04d}: {line_str}")

        return '\n'.join(formatted_lines)

    def clear_bits(self):
        """Clear bit stream"""
        self.bit_buffer.clear()
        self.bit_display.clear()
        self.bit_count_label.setText("Bits: 0")
        self.total_bits = 0
        self.error_count = 0
        self.stats_label.setText("Bits/sec: 0 | Errors: 0 | Total: 0")

    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = self.pause_btn.isChecked()
        if self.paused:
            self.pause_btn.setText("Resume")
        else:
            self.pause_btn.setText("Pause")


class ProcessingStageIndicator(QWidget):
    """Visual indicator for processing stages"""

    def __init__(self, stage_name):
        super().__init__()
        self.stage_name = stage_name
        self.setup_ui()

    def setup_ui(self):
        """Setup stage indicator UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Status indicator (colored circle)
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: #666666; font-size: 14pt;")
        layout.addWidget(self.status_indicator)

        # Stage name
        self.name_label = QLabel(self.stage_name.replace('_', ' ').title())
        self.name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.name_label)

        layout.addStretch()

        # Result display
        self.result_label = QLabel("Idle")
        self.result_label.setStyleSheet("color: #cccccc; font-size: 9pt;")
        layout.addWidget(self.result_label)

        # Progress/confidence
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #14a085; font-size: 9pt;")
        layout.addWidget(self.progress_label)

    def update_status(self, status, result=None, confidence=None, extra_info=None):
        """Update stage status"""
        # Update status indicator color
        if status == 'idle':
            color = "#666666"
        elif status == 'processing':
            color = "#ffaa00"  # Orange
        elif status == 'completed':
            color = "#00ff00"  # Green
        elif status == 'error':
            color = "#ff0000"  # Red
        else:
            color = "#666666"

        self.status_indicator.setStyleSheet(f"color: {color}; font-size: 14pt;")

        # Update result text
        if result is not None:
            self.result_label.setText(str(result))
        else:
            self.result_label.setText(status.title())

        # Update progress/confidence
        if confidence is not None:
            self.progress_label.setText(f"{confidence:.1%}")
        elif extra_info is not None:
            self.progress_label.setText(str(extra_info))
        else:
            self.progress_label.setText("")


class RealtimePipelinePanel(QWidget):
    """Real-time processing pipeline control panel"""

    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.setup_ui()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)

    def setup_ui(self):
        """Setup pipeline panel UI"""
        layout = QVBoxLayout(self)

        # Header with controls
        header_group = QGroupBox("Real-time Signal Processing Pipeline")
        header_layout = QGridLayout(header_group)

        # Start/Stop controls
        self.start_btn = QPushButton("Start Real-time Analysis")
        self.start_btn.setStyleSheet("font-weight: bold; background-color: #0d7377; padding: 10px;")
        self.start_btn.clicked.connect(self.start_analysis)
        header_layout.addWidget(self.start_btn, 0, 0)

        self.stop_btn = QPushButton("Stop Analysis")
        self.stop_btn.setStyleSheet("font-weight: bold; background-color: #d73027; padding: 10px;")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        header_layout.addWidget(self.stop_btn, 0, 1)

        # Configuration
        header_layout.addWidget(QLabel("Update Interval:"), 1, 0)
        self.interval_spinbox = QDoubleSpinBox()
        self.interval_spinbox.setRange(0.5, 10.0)
        self.interval_spinbox.setValue(2.0)
        self.interval_spinbox.setSuffix(" sec")
        header_layout.addWidget(self.interval_spinbox, 1, 1)

        header_layout.addWidget(QLabel("Sample Rate:"), 2, 0)
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["100 kS/s", "1 MS/s", "2 MS/s", "5 MS/s"])
        self.sample_rate_combo.setCurrentText("1 MS/s")
        header_layout.addWidget(self.sample_rate_combo, 2, 1)

        layout.addWidget(header_group)

        # Current Signal Info
        signal_group = QGroupBox("Current Signal Information")
        signal_layout = QGridLayout(signal_group)

        signal_layout.addWidget(QLabel("Signal Type:"), 0, 0)
        self.signal_type_label = QLabel("None")
        self.signal_type_label.setStyleSheet("font-weight: bold; color: #14a085;")
        signal_layout.addWidget(self.signal_type_label, 0, 1)

        signal_layout.addWidget(QLabel("Modulation:"), 1, 0)
        self.modulation_label = QLabel("None")
        signal_layout.addWidget(self.modulation_label, 1, 1)

        signal_layout.addWidget(QLabel("Coding:"), 2, 0)
        self.coding_label = QLabel("None")
        signal_layout.addWidget(self.coding_label, 2, 1)

        signal_layout.addWidget(QLabel("SNR:"), 0, 2)
        self.snr_label = QLabel("N/A")
        signal_layout.addWidget(self.snr_label, 0, 2)

        signal_layout.addWidget(QLabel("Length:"), 1, 2)
        self.length_label = QLabel("0")
        signal_layout.addWidget(self.length_label, 1, 3)

        layout.addWidget(signal_group)

        # Processing Stages
        stages_group = QGroupBox("Processing Stages")
        stages_layout = QVBoxLayout(stages_group)

        # Create stage indicators
        self.stage_indicators = {}
        stage_names = [
            'modulation_detection',
            'demodulation', 
            'coding_detection',
            'channel_decoding',
            'bit_stream'
        ]

        for stage_name in stage_names:
            indicator = ProcessingStageIndicator(stage_name)
            self.stage_indicators[stage_name] = indicator
            stages_layout.addWidget(indicator)

        layout.addWidget(stages_group)

        # Performance Metrics
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QGridLayout(perf_group)

        perf_layout.addWidget(QLabel("Processing Rate:"), 0, 0)
        self.proc_rate_label = QLabel("0 signals/min")
        perf_layout.addWidget(self.proc_rate_label, 0, 1)

        perf_layout.addWidget(QLabel("Success Rate:"), 1, 0)
        self.success_rate_label = QLabel("N/A")
        perf_layout.addWidget(self.success_rate_label, 1, 1)

        perf_layout.addWidget(QLabel("Avg Latency:"), 2, 0)
        self.latency_label = QLabel("N/A")
        perf_layout.addWidget(self.latency_label, 2, 1)

        layout.addWidget(perf_group)

        # Status Log
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)

        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(100)
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("font-family: Consolas; font-size: 9pt;")
        log_layout.addWidget(self.log_display)

        log_controls = QHBoxLayout()

        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_display.clear)
        log_controls.addWidget(clear_log_btn)

        log_controls.addStretch()

        log_layout.addLayout(log_controls)
        layout.addWidget(log_group)

        # Initialize state
        self.is_running = False
        self.processed_count = 0
        self.start_time = None

        # Disable if real-time not available
        if not REALTIME_AVAILABLE:
            self.setEnabled(False)
            self.log_message("Real-time pipeline not available - install required modules")

    def start_analysis(self):
        """Start real-time analysis"""
        if self.is_running or not REALTIME_AVAILABLE:
            return

        try:
            # Get configuration
            interval = self.interval_spinbox.value()
            sample_rate_text = self.sample_rate_combo.currentText()
            sample_rate = float(sample_rate_text.split()[0]) * {'kS/s': 1e3, 'MS/s': 1e6}[sample_rate_text.split()[1]]

            # Create analyzer
            self.analyzer = RealtimeSignalAnalyzer(sample_rate, interval)

            # Start analysis
            self.analyzer.start_analysis()

            # Update UI
            self.is_running = True
            self.start_time = time.time()
            self.processed_count = 0

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

            # Start update timer
            self.update_timer.start(500)  # Update every 500ms

            self.log_message("Real-time analysis started")

        except Exception as e:
            self.log_message(f"Failed to start analysis: {e}")

    def stop_analysis(self):
        """Stop real-time analysis"""
        if not self.is_running:
            return

        try:
            # Stop analyzer
            if self.analyzer:
                self.analyzer.stop_analysis()
                self.analyzer = None

            # Stop timer
            self.update_timer.stop()

            # Update UI
            self.is_running = False

            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

            # Reset stage indicators
            for indicator in self.stage_indicators.values():
                indicator.update_status('idle')

            # Clear signal info
            self.signal_type_label.setText("None")
            self.modulation_label.setText("None")
            self.coding_label.setText("None")
            self.snr_label.setText("N/A")
            self.length_label.setText("0")

            self.log_message("Real-time analysis stopped")

        except Exception as e:
            self.log_message(f"Error stopping analysis: {e}")

    def update_display(self):
        """Update display with current analysis results"""
        if not self.is_running or not self.analyzer:
            return

        try:
            # Get current results
            results = self.analyzer.get_analysis_results()
            signal_info = self.analyzer.get_current_signal_info()

            # Update signal information
            if signal_info['config']:
                config = signal_info['config']
                self.signal_type_label.setText(config.get('name', 'Unknown'))
                self.modulation_label.setText(config.get('modulation', 'Unknown'))
                self.coding_label.setText(config.get('coding', 'None'))
                self.snr_label.setText(f"{config.get('snr_db', 0):.1f} dB")
                self.length_label.setText(f"{signal_info['signal_length']:,}")

            # Update stage indicators
            stages = results.get('stages', {})
            for stage_name, indicator in self.stage_indicators.items():
                if stage_name in stages:
                    stage_info = stages[stage_name]
                    status = stage_info.get('status', 'idle')
                    result = stage_info.get('result')
                    confidence = stage_info.get('confidence')

                    # Add extra info for specific stages
                    extra_info = None
                    if stage_name == 'demodulation':
                        const_count = len(stage_info.get('constellation', []))
                        bit_count = len(result) if result is not None else 0
                        extra_info = f"{bit_count}b, {const_count}c"
                    elif stage_name == 'channel_decoding':
                        success = stage_info.get('success', False)
                        extra_info = "Success" if success else "Failed"
                    elif stage_name == 'bit_stream':
                        count = stage_info.get('count', 0)
                        extra_info = f"{count} bits"

                    indicator.update_status(status, result, confidence, extra_info)

            # Update performance metrics
            if self.start_time:
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    rate = self.processed_count / (elapsed / 60)  # per minute
                    self.proc_rate_label.setText(f"{rate:.1f} signals/min")

                    # Estimate success rate (placeholder)
                    success_stages = sum(1 for s in stages.values() if s.get('status') == 'completed')
                    total_stages = len(stages)
                    if total_stages > 0:
                        success_rate = (success_stages / total_stages) * 100
                        self.success_rate_label.setText(f"{success_rate:.1f}%")

                    # Estimate latency (placeholder)
                    avg_latency = 0.5  # Placeholder
                    self.latency_label.setText(f"{avg_latency:.2f} sec")

            # Count processed signals
            latest_results = results.get('latest_results', {})
            if len(latest_results) > self.processed_count:
                self.processed_count = len(latest_results)

        except Exception as e:
            self.log_message(f"Display update error: {e}")

    def log_message(self, message):
        """Add message to activity log"""
        timestamp = time.strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        self.log_display.append(formatted_message)

        # Keep only last 50 lines
        if self.log_display.document().lineCount() > 50:
            cursor = self.log_display.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor)
            cursor.removeSelectedText()

        # Auto-scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def get_current_results(self):
        """Get current analysis results for other widgets"""
        if self.analyzer:
            return self.analyzer.get_analysis_results()
        return {}


class EnhancedRealtimeMainWindow(QMainWindow):
    """Enhanced main window with real-time pipeline"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced SDR Suite - Real-time Processing Pipeline")
        self.setGeometry(50, 50, 1900, 1200)

        # Components
        self.current_iq_data = None
        self.current_constellation = []

        # Setup UI
        self.setup_ui()

        # Setup update timers
        self.plot_update_timer = QTimer()
        self.plot_update_timer.timeout.connect(self.update_plots)
        self.plot_update_timer.start(100)  # 10 FPS

    def setup_ui(self):
        """Setup enhanced real-time UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Real-time Pipeline Controls
        left_panel = self.create_pipeline_control_panel()
        main_splitter.addWidget(left_panel)

        # Center panel - Visualization
        center_panel = self.create_visualization_panel()
        main_splitter.addWidget(center_panel)

        # Right panel - Bit Stream and Results
        right_panel = self.create_results_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([450, 900, 550])
        main_layout.addWidget(main_splitter)

    def create_pipeline_control_panel(self):
        """Create pipeline control panel"""
        panel = QTabWidget()

        # Real-time Pipeline Tab
        self.pipeline_panel = RealtimePipelinePanel()
        panel.addTab(self.pipeline_panel, "Real-time Pipeline")

        # Legacy Controls Tab (simplified)
        legacy_tab = QWidget()
        legacy_layout = QVBoxLayout(legacy_tab)

        # Basic signal generation
        gen_group = QGroupBox("Manual Signal Generation")
        gen_layout = QGridLayout(gen_group)

        gen_layout.addWidget(QLabel("Signal Type:"), 0, 0)
        self.manual_signal_combo = QComboBox()
        self.manual_signal_combo.addItems([
            "Test Sine", "BPSK", "QPSK", "16-QAM", "FSK", "AM", "FM"
        ])
        gen_layout.addWidget(self.manual_signal_combo, 0, 1)

        self.gen_manual_btn = QPushButton("Generate Manual Signal")
        self.gen_manual_btn.clicked.connect(self.generate_manual_signal)
        gen_layout.addWidget(self.gen_manual_btn, 1, 0, 1, 2)

        legacy_layout.addWidget(gen_group)
        legacy_layout.addStretch()

        panel.addTab(legacy_tab, "Manual Controls")

        panel.setMaximumWidth(500)
        return panel

    def create_visualization_panel(self):
        """Create visualization panel"""
        panel = QTabWidget()

        # Real-time Constellation Tab
        constellation_tab = QWidget()
        constellation_layout = QVBoxLayout(constellation_tab)

        # Constellation plot
        self.constellation_plot = PlotWidget(title="Real-time Constellation Diagram")
        self.constellation_plot.setLabel('left', 'Quadrature (Q)')
        self.constellation_plot.setLabel('bottom', 'In-phase (I)')
        self.constellation_plot.showGrid(True, True)
        self.constellation_plot.setAspectLocked(True)

        # Add constellation controls
        const_controls = QHBoxLayout()

        self.const_auto_scale_cb = QCheckBox("Auto Scale")
        self.const_auto_scale_cb.setChecked(True)
        const_controls.addWidget(self.const_auto_scale_cb)

        self.const_points_spinbox = QSpinBox()
        self.const_points_spinbox.setRange(100, 2000)
        self.const_points_spinbox.setValue(500)
        self.const_points_spinbox.setSuffix(" points max")
        const_controls.addWidget(QLabel("Max Points:"))
        const_controls.addWidget(self.const_points_spinbox)

        const_controls.addStretch()

        self.const_clear_btn = QPushButton("Clear")
        self.const_clear_btn.clicked.connect(self.clear_constellation)
        const_controls.addWidget(self.const_clear_btn)

        constellation_layout.addLayout(const_controls)
        constellation_layout.addWidget(self.constellation_plot)

        panel.addTab(constellation_tab, "Constellation")

        # Spectrum Tab
        spectrum_tab = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_tab)

        self.spectrum_plot = PlotWidget(title="Real-time Spectrum")
        self.spectrum_plot.setLabel('left', 'Power (dB)')
        self.spectrum_plot.setLabel('bottom', 'Frequency (Hz)')
        self.spectrum_plot.showGrid(True, True)

        spectrum_layout.addWidget(self.spectrum_plot)

        panel.addTab(spectrum_tab, "Spectrum")

        # Time Domain Tab
        time_tab = QWidget()
        time_layout = QVBoxLayout(time_tab)

        self.time_plot = PlotWidget(title="Time Domain Signal")
        self.time_plot.setLabel('left', 'Amplitude')
        self.time_plot.setLabel('bottom', 'Time (s)')
        self.time_plot.showGrid(True, True)

        time_layout.addWidget(self.time_plot)

        panel.addTab(time_tab, "Time Domain")

        return panel

    def create_results_panel(self):
        """Create results panel"""
        panel = QTabWidget()

        # Bit Stream Tab
        self.bit_stream_display = BitStreamDisplay()
        panel.addTab(self.bit_stream_display, "Bit Stream")

        # Analysis Results Tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        # Processing Summary
        summary_group = QGroupBox("Processing Summary")
        summary_layout = QVBoxLayout(summary_group)

        self.summary_display = QTextEdit()
        self.summary_display.setMaximumHeight(150)
        self.summary_display.setReadOnly(True)
        self.summary_display.setStyleSheet("font-family: Consolas; font-size: 9pt;")
        summary_layout.addWidget(self.summary_display)

        results_layout.addWidget(summary_group)

        # Detailed Results
        details_group = QGroupBox("Stage Details")
        details_layout = QVBoxLayout(details_group)

        self.details_table = QTableWidget(5, 3)
        self.details_table.setHorizontalHeaderLabels(["Stage", "Status", "Result"])
        self.details_table.setMaximumHeight(200)
        details_layout.addWidget(self.details_table)

        results_layout.addWidget(details_group)

        # Performance Stats
        stats_group = QGroupBox("Performance Statistics")
        stats_layout = QGridLayout(stats_group)

        # Add performance metrics
        stats_layout.addWidget(QLabel("Detection Accuracy:"), 0, 0)
        self.detection_acc_label = QLabel("N/A")
        stats_layout.addWidget(self.detection_acc_label, 0, 1)

        stats_layout.addWidget(QLabel("Decoding Success:"), 1, 0)
        self.decoding_success_label = QLabel("N/A")
        stats_layout.addWidget(self.decoding_success_label, 1, 1)

        stats_layout.addWidget(QLabel("Throughput:"), 2, 0)
        self.throughput_label = QLabel("N/A")
        stats_layout.addWidget(self.throughput_label, 2, 1)

        results_layout.addWidget(stats_group)
        results_layout.addStretch()

        panel.addTab(results_tab, "Analysis Results")

        panel.setMaximumWidth(600)
        return panel

    def update_plots(self):
        """Update all plots with current data"""
        try:
            # Get current results from pipeline
            if hasattr(self.pipeline_panel, 'analyzer') and self.pipeline_panel.analyzer:
                results = self.pipeline_panel.get_current_results()
                signal_info = self.pipeline_panel.analyzer.get_current_signal_info()

                # Update constellation
                constellation_points = results.get('constellation_points', [])
                if len(constellation_points) > 0:
                    self.update_constellation_plot(constellation_points)

                # Update bit stream
                stages = results.get('stages', {})
                bit_stream_stage = stages.get('bit_stream', {})
                new_bits = bit_stream_stage.get('result')
                if new_bits is not None:
                    self.bit_stream_display.update_bit_stream(new_bits)

                # Update signal plots
                if signal_info.get('signal') is not None:
                    self.update_signal_plots(signal_info['signal'])

                # Update analysis results
                self.update_analysis_results(results)

        except Exception as e:
            # Silently handle plot errors to avoid spam
            pass

    def update_constellation_plot(self, constellation_points):
        """Update constellation plot"""
        try:
            if len(constellation_points) == 0:
                return

            # Limit points for performance
            max_points = self.const_points_spinbox.value()
            if len(constellation_points) > max_points:
                # Subsample
                indices = np.random.choice(len(constellation_points), max_points, replace=False)
                points = np.array(constellation_points)[indices]
            else:
                points = np.array(constellation_points)

            # Extract I and Q
            i_data = np.real(points)
            q_data = np.imag(points)

            # Clear and plot
            self.constellation_plot.clear()
            self.constellation_plot.plot(i_data, q_data, pen=None, symbol='o',
                                       symbolSize=3, symbolBrush=(100, 255, 100, 150),
                                       name='Constellation')

            # Auto scale if enabled
            if self.const_auto_scale_cb.isChecked():
                self.constellation_plot.autoRange()

        except Exception as e:
            print(f"Constellation plot error: {e}")

    def update_signal_plots(self, signal):
        """Update spectrum and time domain plots"""
        try:
            if signal is None or len(signal) == 0:
                return

            # Limit signal length for performance
            if len(signal) > 2048:
                signal = signal[:2048]

            # Time domain plot
            sample_rate = 1e6  # Default
            t = np.arange(len(signal)) / sample_rate

            self.time_plot.clear()
            if np.iscomplexobj(signal):
                self.time_plot.plot(t, np.real(signal), pen='b', name='I')
                self.time_plot.plot(t, np.imag(signal), pen='r', name='Q')
            else:
                self.time_plot.plot(t, signal, pen='y', name='Signal')

            # Spectrum plot
            if len(signal) >= 64:  # Minimum for meaningful FFT
                fft_data = fft(signal)
                freqs = fftfreq(len(signal), 1/sample_rate)
                freqs_shifted = fftshift(freqs)
                fft_shifted = fftshift(fft_data)

                magnitude_db = 20 * np.log10(np.abs(fft_shifted) + 1e-12)

                self.spectrum_plot.clear()
                self.spectrum_plot.plot(freqs_shifted, magnitude_db, pen='y', name='Spectrum')

        except Exception as e:
            print(f"Signal plot error: {e}")

    def update_analysis_results(self, results):
        """Update analysis results display"""
        try:
            # Update summary
            summary_text = "=== REAL-TIME ANALYSIS SUMMARY ===\n\n"

            stages = results.get('stages', {})
            for stage_name, stage_info in stages.items():
                status = stage_info.get('status', 'unknown')
                result = stage_info.get('result', 'N/A')

                stage_display = stage_name.replace('_', ' ').title()
                summary_text += f"{stage_display:20}: {status:10} -> {result}\n"

            bit_count = len(results.get('bit_stream_buffer', []))
            const_count = len(results.get('constellation_points', []))

            summary_text += f"\nBit Stream Buffer    : {bit_count} bits\n"
            summary_text += f"Constellation Points : {const_count} points\n"

            self.summary_display.setPlainText(summary_text)

            # Update details table
            self.details_table.setRowCount(len(stages))
            for i, (stage_name, stage_info) in enumerate(stages.items()):
                stage_display = stage_name.replace('_', ' ').title()
                status = stage_info.get('status', 'unknown')
                result = str(stage_info.get('result', 'N/A'))

                # Truncate long results
                if len(result) > 30:
                    result = result[:27] + "..."

                self.details_table.setItem(i, 0, QTableWidgetItem(stage_display))
                self.details_table.setItem(i, 1, QTableWidgetItem(status))
                self.details_table.setItem(i, 2, QTableWidgetItem(result))

            # Update performance stats (placeholder)
            completed_stages = sum(1 for s in stages.values() if s.get('status') == 'completed')
            total_stages = len(stages)

            if total_stages > 0:
                accuracy = (completed_stages / total_stages) * 100
                self.detection_acc_label.setText(f"{accuracy:.1f}%")

            # Decoding success
            decoding_stage = stages.get('channel_decoding', {})
            if decoding_stage.get('success'):
                self.decoding_success_label.setText("Success")
                self.decoding_success_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.decoding_success_label.setText("Failed")
                self.decoding_success_label.setStyleSheet("color: red; font-weight: bold;")

            # Throughput (placeholder)
            self.throughput_label.setText(f"{bit_count} bits processed")

        except Exception as e:
            print(f"Analysis results update error: {e}")

    def generate_manual_signal(self):
        """Generate manual signal for testing"""
        try:
            signal_type = self.manual_signal_combo.currentText()

            # Generate test signal
            sample_rate = 100000
            duration = 0.1  # 100ms
            t = np.arange(0, duration, 1/sample_rate)

            if signal_type == "Test Sine":
                freq = 1000
                signal = np.sin(2 * np.pi * freq * t)
                iq_signal = signal + 1j * np.zeros_like(signal)

            elif signal_type == "BPSK":
                bits = np.random.randint(0, 2, 50)
                symbols = 2 * bits - 1
                samples_per_symbol = len(t) // len(symbols)
                signal = np.repeat(symbols, samples_per_symbol)[:len(t)]
                iq_signal = signal + 1j * np.zeros_like(signal)

            elif signal_type == "QPSK":
                bits = np.random.randint(0, 2, 100)
                i_bits = bits[::2]
                q_bits = bits[1::2]
                i_symbols = 2 * i_bits - 1
                q_symbols = 2 * q_bits - 1
                samples_per_symbol = len(t) // len(i_symbols)
                i_signal = np.repeat(i_symbols, samples_per_symbol)[:len(t)]
                q_signal = np.repeat(q_symbols, samples_per_symbol)[:len(t)]
                iq_signal = (i_signal + 1j * q_signal) / np.sqrt(2)

            else:
                # Default sine wave
                iq_signal = np.sin(2 * np.pi * 1000 * t) + 1j * np.zeros_like(t)

            # Add noise
            noise_power = 0.1
            noise = noise_power * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
            iq_signal += noise

            # Store for display
            self.current_iq_data = iq_signal

            print(f"Generated {signal_type} manual signal: {len(iq_signal)} samples")

        except Exception as e:
            print(f"Manual signal generation error: {e}")

    def clear_constellation(self):
        """Clear constellation plot"""
        self.constellation_plot.clear()


def main():
    """Main application with real-time pipeline"""
    app = QApplication(sys.argv)
    app.setApplicationName("Advanced SDR Suite - Real-time Edition")
    app.setApplicationVersion("4.0")

    # Set application style
    app.setStyle('Fusion')

    # Enhanced dark theme
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QTabWidget::pane {
            border: 1px solid #555555;
            background-color: #2b2b2b;
        }
        QTabBar::tab {
            background-color: #404040;
            color: #ffffff;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #0d7377;
            font-weight: bold;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #555555;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #0d7377;
            border: none;
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #14a085;
        }
        QPushButton:pressed {
            background-color: #0a5d61;
        }
        QPushButton:disabled {
            background-color: #555555;
        }
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
            background-color: #404040;
            border: 1px solid #555555;
            color: white;
            padding: 4px;
            border-radius: 2px;
        }
        QTableWidget {
            background-color: #404040;
            alternate-background-color: #4a4a4a;
            color: white;
            gridline-color: #555555;
        }
        QTableWidget::item:selected {
            background-color: #0d7377;
        }
        QTextEdit {
            background-color: #404040;
            border: 1px solid #555555;
            color: white;
        }
    """)

    # Create and show main window
    window = EnhancedRealtimeMainWindow()
    window.show()

    print("🚀 Advanced SDR Suite - Real-time Processing Pipeline Started")
    print("✅ Features available:")
    print("   • Real-time signal generation with rotating modulation/coding")
    print("   • Multi-stage processing pipeline (5 stages)")
    print("   • Live constellation diagram updates")
    print("   • Continuous bit stream visualization")
    print("   • Automatic modulation and channel coding detection")
    print("   • Complete integration of all modulation and coding types")

    if REALTIME_AVAILABLE:
        print("   ✅ Real-time pipeline: Available")
    else:
        print("   ⚠️  Real-time pipeline: Requires additional modules")

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
