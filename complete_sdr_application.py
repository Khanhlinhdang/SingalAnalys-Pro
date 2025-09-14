
"""
Complete SDR Application - Final Version
Tích hợp hoàn chỉnh: USRP, signal generation, auto-detection, visual bitstream
"""

import sys
import os
import numpy as np
import time
import threading
from queue import Queue, Empty

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
                               QComboBox, QSlider, QProgressBar, QTextEdit, QTabWidget,
                               QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox, QTableWidget,
                               QTableWidgetItem, QSplitter, QScrollArea, QFrame, QListWidget,
                               QButtonGroup, QRadioButton, QFormLayout, QMessageBox, QFileDialog)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread
from PySide6.QtGui import QFont, QPixmap, QColor, QPalette

# Import pyqtgraph for plotting
try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("Warning: pyqtgraph not available. Install with: pip install pyqtgraph")

from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq, fftshift

# Import our comprehensive modules
try:
    from usrp_interface import create_usrp_interface, USRP_AVAILABLE
    from enhanced_signal_generator import EnhancedSignalGenerator
    from visual_bitstream import VisualBitstreamWidget, BitstreamAnalyzer
    from enhanced_processing_pipeline import EnhancedProcessingPipeline, ParameterTable
    from channel_coding import ChannelCodingDetector
    MODULES_COMPLETE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODULES_COMPLETE = False


class ParameterControlPanel(QWidget):
    """Parameter control panel for modulation and coding parameters"""

    # Signal emitted when parameters change
    parameters_changed = Signal(str, dict)  # (type, parameters)

    def __init__(self):
        super().__init__()
        self.setup_ui()

        # Parameter storage
        self.current_mod_type = 'bpsk'
        self.current_coding_type = 'none'
        self.param_table = ParameterTable() if MODULES_COMPLETE else None

    def setup_ui(self):
        """Setup parameter control UI"""
        layout = QVBoxLayout(self)

        # Mode selection
        mode_group = QGroupBox("Detection Mode")
        mode_layout = QGridLayout(mode_group)

        # Auto/Manual mode selection
        self.auto_detection_group = QButtonGroup()

        self.auto_mod_rb = QRadioButton("Auto Detect Modulation")
        self.auto_mod_rb.setChecked(True)
        self.auto_detection_group.addButton(self.auto_mod_rb, 0)
        mode_layout.addWidget(self.auto_mod_rb, 0, 0)

        self.manual_mod_rb = QRadioButton("Manual Select")
        self.auto_detection_group.addButton(self.manual_mod_rb, 1)
        mode_layout.addWidget(self.manual_mod_rb, 0, 1)

        self.auto_coding_rb = QRadioButton("Auto Detect Coding")
        self.auto_coding_rb.setChecked(True)
        mode_layout.addWidget(self.auto_coding_rb, 1, 0)

        self.manual_coding_rb = QRadioButton("Manual Select")
        mode_layout.addWidget(self.manual_coding_rb, 1, 1)

        layout.addWidget(mode_group)

        # Modulation parameters
        mod_group = QGroupBox("Modulation Parameters")
        mod_layout = QFormLayout(mod_group)

        # Modulation type selection
        mod_layout.addRow("Type:", QLabel(""))
        self.mod_type_combo = QComboBox()
        self.mod_type_combo.addItems([
            'bpsk', 'qpsk', '8psk', '16qam', '64qam', '256qam',
            'fsk', 'gfsk', 'msk', 'gmsk', 'am_dsb_lc', 'fm_wb',
            'ofdm_qpsk', 'dsss_bpsk'
        ])
        self.mod_type_combo.currentTextChanged.connect(self.on_mod_type_changed)
        mod_layout.addRow("Modulation:", self.mod_type_combo)

        # Dynamic modulation parameters
        self.mod_param_widgets = {}

        # Symbol rate
        self.symbol_rate_spinbox = QSpinBox()
        self.symbol_rate_spinbox.setRange(1000, 100000)
        self.symbol_rate_spinbox.setValue(10000)
        self.symbol_rate_spinbox.setSuffix(" Hz")
        self.symbol_rate_spinbox.valueChanged.connect(self.on_mod_params_changed)
        mod_layout.addRow("Symbol Rate:", self.symbol_rate_spinbox)
        self.mod_param_widgets['symbol_rate'] = self.symbol_rate_spinbox

        # Carrier frequency
        self.carrier_freq_spinbox = QSpinBox()
        self.carrier_freq_spinbox.setRange(0, 50000)
        self.carrier_freq_spinbox.setValue(0)
        self.carrier_freq_spinbox.setSuffix(" Hz")
        self.carrier_freq_spinbox.valueChanged.connect(self.on_mod_params_changed)
        mod_layout.addRow("Carrier Freq:", self.carrier_freq_spinbox)
        self.mod_param_widgets['carrier_freq'] = self.carrier_freq_spinbox

        # Additional parameters (will be shown/hidden based on modulation type)
        self.freq_deviation_spinbox = QSpinBox()
        self.freq_deviation_spinbox.setRange(500, 10000)
        self.freq_deviation_spinbox.setValue(2000)
        self.freq_deviation_spinbox.setSuffix(" Hz")
        self.freq_deviation_spinbox.valueChanged.connect(self.on_mod_params_changed)
        mod_layout.addRow("Freq Deviation:", self.freq_deviation_spinbox)
        self.mod_param_widgets['freq_deviation'] = self.freq_deviation_spinbox

        self.bt_product_spinbox = QDoubleSpinBox()
        self.bt_product_spinbox.setRange(0.1, 1.0)
        self.bt_product_spinbox.setValue(0.3)
        self.bt_product_spinbox.setSingleStep(0.1)
        self.bt_product_spinbox.valueChanged.connect(self.on_mod_params_changed)
        mod_layout.addRow("BT Product:", self.bt_product_spinbox)
        self.mod_param_widgets['bt_product'] = self.bt_product_spinbox

        self.modulation_index_spinbox = QDoubleSpinBox()
        self.modulation_index_spinbox.setRange(0.1, 0.95)
        self.modulation_index_spinbox.setValue(0.8)
        self.modulation_index_spinbox.setSingleStep(0.05)
        self.modulation_index_spinbox.valueChanged.connect(self.on_mod_params_changed)
        mod_layout.addRow("Mod Index:", self.modulation_index_spinbox)
        self.mod_param_widgets['modulation_index'] = self.modulation_index_spinbox

        layout.addWidget(mod_group)

        # Channel coding parameters
        coding_group = QGroupBox("Channel Coding Parameters")
        coding_layout = QFormLayout(coding_group)

        # Coding type selection
        self.coding_type_combo = QComboBox()
        self.coding_type_combo.addItems([
            'none', 'convolutional', 'turbo', 'ldpc', 'polar', 'reed_solomon'
        ])
        self.coding_type_combo.currentTextChanged.connect(self.on_coding_type_changed)
        coding_layout.addRow("Coding:", self.coding_type_combo)

        # Dynamic coding parameters
        self.coding_param_widgets = {}

        # Constraint length
        self.constraint_length_spinbox = QSpinBox()
        self.constraint_length_spinbox.setRange(3, 15)
        self.constraint_length_spinbox.setValue(7)
        self.constraint_length_spinbox.valueChanged.connect(self.on_coding_params_changed)
        coding_layout.addRow("Constraint Length:", self.constraint_length_spinbox)
        self.coding_param_widgets['constraint_length'] = self.constraint_length_spinbox

        # Code rate
        self.code_rate_combo = QComboBox()
        self.code_rate_combo.addItems(['1/3', '1/2', '2/3', '3/4'])
        self.code_rate_combo.setCurrentText('1/2')
        self.code_rate_combo.currentTextChanged.connect(self.on_coding_params_changed)
        coding_layout.addRow("Code Rate:", self.code_rate_combo)
        self.coding_param_widgets['code_rate'] = self.code_rate_combo

        # Interleaver size
        self.interleaver_size_spinbox = QSpinBox()
        self.interleaver_size_spinbox.setRange(64, 8192)
        self.interleaver_size_spinbox.setValue(1024)
        self.interleaver_size_spinbox.valueChanged.connect(self.on_coding_params_changed)
        coding_layout.addRow("Interleaver Size:", self.interleaver_size_spinbox)
        self.coding_param_widgets['interleaver_size'] = self.interleaver_size_spinbox

        # Max iterations
        self.max_iterations_spinbox = QSpinBox()
        self.max_iterations_spinbox.setRange(1, 100)
        self.max_iterations_spinbox.setValue(50)
        self.max_iterations_spinbox.valueChanged.connect(self.on_coding_params_changed)
        coding_layout.addRow("Max Iterations:", self.max_iterations_spinbox)
        self.coding_param_widgets['max_iterations'] = self.max_iterations_spinbox

        layout.addWidget(coding_group)

        # Signal parameters
        signal_group = QGroupBox("Signal Parameters")
        signal_layout = QFormLayout(signal_group)

        self.signal_power_spinbox = QDoubleSpinBox()
        self.signal_power_spinbox.setRange(-30, 30)
        self.signal_power_spinbox.setValue(0)
        self.signal_power_spinbox.setSuffix(" dB")
        signal_layout.addRow("Signal Power:", self.signal_power_spinbox)

        self.noise_power_spinbox = QDoubleSpinBox()
        self.noise_power_spinbox.setRange(-50, 10)
        self.noise_power_spinbox.setValue(-20)
        self.noise_power_spinbox.setSuffix(" dB")
        signal_layout.addRow("Noise Power:", self.noise_power_spinbox)

        self.snr_estimate_spinbox = QDoubleSpinBox()
        self.snr_estimate_spinbox.setRange(-10, 30)
        self.snr_estimate_spinbox.setValue(10)
        self.snr_estimate_spinbox.setSuffix(" dB")
        signal_layout.addRow("SNR Estimate:", self.snr_estimate_spinbox)

        layout.addWidget(signal_group)

        # Initialize parameter visibility
        self.on_mod_type_changed('bpsk')
        self.on_coding_type_changed('none')

    def on_mod_type_changed(self, mod_type):
        """Handle modulation type change"""
        self.current_mod_type = mod_type

        # Show/hide relevant parameters
        self.freq_deviation_spinbox.setVisible(mod_type in ['fsk', 'gfsk'])
        self.bt_product_spinbox.setVisible(mod_type in ['gfsk', 'gmsk'])
        self.modulation_index_spinbox.setVisible(mod_type in ['am_dsb_lc'])

        # Update parameter ranges based on type
        if mod_type in ['16qam', '64qam', '256qam']:
            self.symbol_rate_spinbox.setMaximum(25000)
        else:
            self.symbol_rate_spinbox.setMaximum(100000)

        # Emit parameters changed signal
        self.emit_mod_parameters()

    def on_coding_type_changed(self, coding_type):
        """Handle coding type change"""
        self.current_coding_type = coding_type

        # Show/hide relevant parameters
        conv_visible = coding_type == 'convolutional'
        self.constraint_length_spinbox.setVisible(conv_visible)
        self.code_rate_combo.setVisible(conv_visible)

        turbo_visible = coding_type == 'turbo'
        self.interleaver_size_spinbox.setVisible(turbo_visible)

        ldpc_visible = coding_type == 'ldpc'
        self.max_iterations_spinbox.setVisible(ldpc_visible)

        # Emit parameters changed signal
        self.emit_coding_parameters()

    def on_mod_params_changed(self):
        """Handle modulation parameter change"""
        self.emit_mod_parameters()

    def on_coding_params_changed(self):
        """Handle coding parameter change"""
        self.emit_coding_parameters()

    def emit_mod_parameters(self):
        """Emit modulation parameters"""
        params = {
            'symbol_rate': self.symbol_rate_spinbox.value(),
            'carrier_freq': self.carrier_freq_spinbox.value(),
        }

        # Add type-specific parameters
        if self.current_mod_type in ['fsk', 'gfsk']:
            params['freq_deviation'] = self.freq_deviation_spinbox.value()

        if self.current_mod_type in ['gfsk', 'gmsk']:
            params['bt_product'] = self.bt_product_spinbox.value()

        if self.current_mod_type == 'am_dsb_lc':
            params['modulation_index'] = self.modulation_index_spinbox.value()

        self.parameters_changed.emit('modulation', params)

    def emit_coding_parameters(self):
        """Emit coding parameters"""
        params = {}

        if self.current_coding_type == 'convolutional':
            params['constraint_length'] = self.constraint_length_spinbox.value()
            # Convert code rate string to float
            rate_str = self.code_rate_combo.currentText()
            params['code_rate'] = eval(rate_str)  # Simple evaluation of fractions

        elif self.current_coding_type == 'turbo':
            params['interleaver_size'] = self.interleaver_size_spinbox.value()

        elif self.current_coding_type == 'ldpc':
            params['max_iterations'] = self.max_iterations_spinbox.value()

        self.parameters_changed.emit('coding', params)

    def get_signal_parameters(self):
        """Get signal parameters"""
        return {
            'signal_power': self.signal_power_spinbox.value(),
            'noise_power': self.noise_power_spinbox.value(),
            'snr_estimate': self.snr_estimate_spinbox.value()
        }

    def is_auto_modulation(self):
        """Check if auto modulation detection is enabled"""
        return self.auto_mod_rb.isChecked()

    def is_auto_coding(self):
        """Check if auto coding detection is enabled"""
        return self.auto_coding_rb.isChecked()

    def get_manual_selections(self):
        """Get manual type selections"""
        return {
            'modulation': self.current_mod_type if not self.is_auto_modulation() else None,
            'coding': self.current_coding_type if not self.is_auto_coding() else None
        }


class USRPControlPanel(QWidget):
    """USRP control panel"""

    # Signals
    usrp_connected = Signal(bool)
    streaming_started = Signal()
    streaming_stopped = Signal()

    def __init__(self):
        super().__init__()
        self.setup_ui()

        # USRP interface
        self.usrp = None
        self.streaming = False

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(1000)  # Update every second

    def setup_ui(self):
        """Setup USRP control UI"""
        layout = QVBoxLayout(self)

        # Connection controls
        conn_group = QGroupBox("USRP Connection")
        conn_layout = QGridLayout(conn_group)

        conn_layout.addWidget(QLabel("Device:"), 0, 0)
        self.device_combo = QComboBox()
        self.refresh_devices()
        conn_layout.addWidget(self.device_combo, 0, 1)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_devices)
        conn_layout.addWidget(self.refresh_btn, 0, 2)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        conn_layout.addWidget(self.connect_btn, 1, 0, 1, 3)

        layout.addWidget(conn_group)

        # USRP parameters
        params_group = QGroupBox("USRP Parameters")
        params_layout = QFormLayout(params_group)

        # Sample rate
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems([
            "100 kS/s", "200 kS/s", "500 kS/s", "1 MS/s", 
            "2 MS/s", "5 MS/s", "10 MS/s", "20 MS/s"
        ])
        self.sample_rate_combo.setCurrentText("1 MS/s")
        params_layout.addRow("Sample Rate:", self.sample_rate_combo)

        # Center frequency
        self.center_freq_spinbox = QDoubleSpinBox()
        self.center_freq_spinbox.setRange(50, 6000)
        self.center_freq_spinbox.setValue(100)
        self.center_freq_spinbox.setSuffix(" MHz")
        self.center_freq_spinbox.setDecimals(3)
        params_layout.addRow("Center Freq:", self.center_freq_spinbox)

        # RX Gain
        self.rx_gain_spinbox = QSpinBox()
        self.rx_gain_spinbox.setRange(0, 70)
        self.rx_gain_spinbox.setValue(30)
        self.rx_gain_spinbox.setSuffix(" dB")
        params_layout.addRow("RX Gain:", self.rx_gain_spinbox)

        # Bandwidth
        self.bandwidth_spinbox = QDoubleSpinBox()
        self.bandwidth_spinbox.setRange(0, 40)
        self.bandwidth_spinbox.setValue(0)  # Auto
        self.bandwidth_spinbox.setSuffix(" MHz (0=Auto)")
        self.bandwidth_spinbox.setDecimals(1)
        params_layout.addRow("Bandwidth:", self.bandwidth_spinbox)

        # Apply button
        self.apply_params_btn = QPushButton("Apply Parameters")
        self.apply_params_btn.clicked.connect(self.apply_parameters)
        self.apply_params_btn.setEnabled(False)
        params_layout.addWidget(self.apply_params_btn)

        layout.addWidget(params_group)

        # Streaming controls
        stream_group = QGroupBox("Streaming Control")
        stream_layout = QGridLayout(stream_group)

        self.start_stream_btn = QPushButton("Start Streaming")
        self.start_stream_btn.clicked.connect(self.toggle_streaming)
        self.start_stream_btn.setEnabled(False)
        stream_layout.addWidget(self.start_stream_btn, 0, 0)

        self.stop_stream_btn = QPushButton("Stop Streaming")
        self.stop_stream_btn.clicked.connect(self.stop_streaming)
        self.stop_stream_btn.setEnabled(False)
        stream_layout.addWidget(self.stop_stream_btn, 0, 1)

        layout.addWidget(stream_group)

        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)

        self.status_label = QLabel("Not connected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        status_layout.addWidget(self.status_label)

        self.stats_label = QLabel("No streaming statistics")
        self.stats_label.setStyleSheet("font-family: Consolas; font-size: 9pt;")
        status_layout.addWidget(self.stats_label)

        layout.addWidget(status_group)

        layout.addStretch()

    def refresh_devices(self):
        """Refresh USRP device list"""
        self.device_combo.clear()

        try:
            if MODULES_COMPLETE:
                # Create temporary interface to get device list
                temp_usrp = create_usrp_interface(use_simulator=False)
                devices = temp_usrp.get_device_list()

                if not devices:  # No hardware devices, add simulator
                    temp_usrp = create_usrp_interface(use_simulator=True)
                    devices = temp_usrp.get_device_list()

                for device in devices:
                    display_name = f"{device['type']} ({device['serial']})"
                    self.device_combo.addItem(display_name, device)

            if self.device_combo.count() == 0:
                self.device_combo.addItem("No devices found", None)

        except Exception as e:
            self.device_combo.addItem(f"Error: {str(e)}", None)

    def toggle_connection(self):
        """Toggle USRP connection"""
        if self.usrp is None:
            self.connect_usrp()
        else:
            self.disconnect_usrp()

    def connect_usrp(self):
        """Connect to USRP"""
        if not MODULES_COMPLETE:
            QMessageBox.warning(self, "Error", "USRP modules not available")
            return

        try:
            device_data = self.device_combo.currentData()
            if device_data is None:
                QMessageBox.warning(self, "Error", "No valid device selected")
                return

            # Parse sample rate
            sample_rate_text = self.sample_rate_combo.currentText()
            sample_rate_value = float(sample_rate_text.split()[0])
            if 'kS/s' in sample_rate_text:
                sample_rate = sample_rate_value * 1e3
            else:  # MS/s
                sample_rate = sample_rate_value * 1e6

            # Parse center frequency
            center_freq = self.center_freq_spinbox.value() * 1e6  # Convert to Hz

            # Create USRP interface
            use_simulator = device_data['type'] == 'Simulated USRP'
            self.usrp = create_usrp_interface(use_simulator=use_simulator)

            # Connect
            if use_simulator:
                device_args = ""
            else:
                device_args = f"addr={device_data.get('address', '')}"

            if self.usrp.connect(device_args, sample_rate, center_freq):
                # Apply additional parameters
                self.apply_parameters()

                # Update UI
                self.connect_btn.setText("Disconnect")
                self.apply_params_btn.setEnabled(True)
                self.start_stream_btn.setEnabled(True)
                self.status_label.setText("Connected")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")

                self.usrp_connected.emit(True)
            else:
                self.usrp = None
                QMessageBox.critical(self, "Error", "Failed to connect to USRP")

        except Exception as e:
            self.usrp = None
            QMessageBox.critical(self, "Error", f"Connection error: {str(e)}")

    def disconnect_usrp(self):
        """Disconnect from USRP"""
        try:
            if self.streaming:
                self.stop_streaming()

            if self.usrp:
                self.usrp.disconnect()
                self.usrp = None

            # Update UI
            self.connect_btn.setText("Connect")
            self.apply_params_btn.setEnabled(False)
            self.start_stream_btn.setEnabled(False)
            self.stop_stream_btn.setEnabled(False)
            self.status_label.setText("Not connected")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")

            self.usrp_connected.emit(False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Disconnect error: {str(e)}")

    def apply_parameters(self):
        """Apply USRP parameters"""
        if not self.usrp:
            return

        try:
            # Parse parameters
            center_freq = self.center_freq_spinbox.value() * 1e6
            gain = self.rx_gain_spinbox.value()
            bandwidth = self.bandwidth_spinbox.value() * 1e6 if self.bandwidth_spinbox.value() > 0 else 0

            # Apply parameters
            success = self.usrp.set_rx_parameters(
                center_freq=center_freq,
                gain=gain,
                bandwidth=bandwidth
            )

            if success:
                self.status_label.setText("Parameters applied")
            else:
                self.status_label.setText("Parameter application failed")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Parameter error: {str(e)}")

    def toggle_streaming(self):
        """Toggle streaming"""
        if self.streaming:
            self.stop_streaming()
        else:
            self.start_streaming()

    def start_streaming(self):
        """Start USRP streaming"""
        if not self.usrp:
            return

        try:
            if self.usrp.start_streaming():
                self.streaming = True
                self.start_stream_btn.setText("Streaming...")
                self.start_stream_btn.setEnabled(False)
                self.stop_stream_btn.setEnabled(True)
                self.status_label.setText("Streaming")
                self.status_label.setStyleSheet("color: blue; font-weight: bold;")

                self.streaming_started.emit()
            else:
                QMessageBox.critical(self, "Error", "Failed to start streaming")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Streaming error: {str(e)}")

    def stop_streaming(self):
        """Stop USRP streaming"""
        if not self.usrp:
            return

        try:
            self.usrp.stop_streaming()
            self.streaming = False

            self.start_stream_btn.setText("Start Streaming")
            self.start_stream_btn.setEnabled(True)
            self.stop_stream_btn.setEnabled(False)
            self.status_label.setText("Connected (not streaming)")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")

            self.streaming_stopped.emit()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Stop streaming error: {str(e)}")

    def update_status(self):
        """Update USRP status display"""
        if self.usrp and self.streaming:
            try:
                stats = self.usrp.get_streaming_stats()
                stats_text = (
                    f"Samples: {stats['samples_received']:,}\n"
                    f"Rate: {stats['sample_rate']/1e6:.2f} MS/s\n"
                    f"Overruns: {stats['overruns']}\n"
                    f"Queue: {stats['queue_size']}"
                )
                self.stats_label.setText(stats_text)
            except:
                self.stats_label.setText("Statistics unavailable")
        else:
            self.stats_label.setText("No streaming statistics")

    def get_samples(self, timeout=0.1):
        """Get samples from USRP"""
        if self.usrp and self.streaming:
            return self.usrp.get_samples(timeout)
        return None


class CompleteSdrMainWindow(QMainWindow):
    """Complete SDR main window with all features"""
    
    # Signal for thread-safe communication from background threads
    signal_generated = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Complete SDR Suite - Professional Edition")
        self.setGeometry(50, 50, 1920, 1200)

        # Core components
        self.signal_generator = None
        self.processing_pipeline = None
        self.current_signal = None

        # Initialize components
        if MODULES_COMPLETE:
            self.init_components()

        # Setup UI
        self.setup_ui()

        # Timers
        self.signal_update_timer = QTimer()
        self.signal_update_timer.timeout.connect(self.update_signal_processing)

        self.plot_update_timer = QTimer()
        self.plot_update_timer.timeout.connect(self.update_plots)
        self.plot_update_timer.start(100)  # 10 FPS

        # Connect signal for thread-safe GUI updates
        self.signal_generated.connect(self.on_signal_generated_safe)

        # Processing state
        self.processing_active = False
        self.signal_source = 'generator'  # 'generator' or 'usrp'

    def init_components(self):
        """Initialize core components"""
        try:
            self.signal_generator = EnhancedSignalGenerator(sample_rate=1e6)
            self.processing_pipeline = EnhancedProcessingPipeline(sample_rate=1e6)
            print("✅ Core components initialized")
        except Exception as e:
            print(f"❌ Component initialization error: {e}")

    def setup_ui(self):
        """Setup complete UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Controls
        left_panel = self.create_control_panel()
        left_panel.setMaximumWidth(450)
        main_splitter.addWidget(left_panel)

        # Center panel - Visualization
        center_panel = self.create_visualization_panel()
        main_splitter.addWidget(center_panel)

        # Right panel - Results
        right_panel = self.create_results_panel()
        right_panel.setMaximumWidth(600)
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([450, 1000, 600])
        main_layout.addWidget(main_splitter)

    def create_control_panel(self):
        """Create control panel"""
        panel = QTabWidget()

        # Signal Source Tab
        source_tab = QWidget()
        source_layout = QVBoxLayout(source_tab)

        # Signal source selection
        source_group = QGroupBox("Signal Source")
        source_inner_layout = QVBoxLayout(source_group)

        self.source_button_group = QButtonGroup()

        self.generator_rb = QRadioButton("Signal Generator")
        self.generator_rb.setChecked(True)
        self.generator_rb.toggled.connect(self.on_source_changed)
        self.source_button_group.addButton(self.generator_rb)
        source_inner_layout.addWidget(self.generator_rb)

        self.usrp_rb = QRadioButton("USRP Hardware")
        self.usrp_rb.toggled.connect(self.on_source_changed)
        self.source_button_group.addButton(self.usrp_rb)
        source_inner_layout.addWidget(self.usrp_rb)

        source_layout.addWidget(source_group)

        # Signal generator controls
        gen_group = QGroupBox("Signal Generator")
        gen_layout = QVBoxLayout(gen_group)

        # Generation mode
        mode_layout = QHBoxLayout()
        self.single_gen_rb = QRadioButton("Single Generation")
        self.single_gen_rb.setChecked(True)
        mode_layout.addWidget(self.single_gen_rb)

        self.continuous_gen_rb = QRadioButton("Continuous")
        mode_layout.addWidget(self.continuous_gen_rb)
        gen_layout.addLayout(mode_layout)

        # Generation controls
        gen_controls = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Signal")
        self.generate_btn.clicked.connect(self.generate_signal)
        gen_controls.addWidget(self.generate_btn)

        self.start_continuous_btn = QPushButton("Start Continuous")
        self.start_continuous_btn.clicked.connect(self.toggle_continuous_generation)
        gen_controls.addWidget(self.start_continuous_btn)
        gen_layout.addLayout(gen_controls)

        # Data source
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("Data Source:"))
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(['random', 'sequence', 'text', 'prbs'])
        data_layout.addWidget(self.data_source_combo)
        gen_layout.addLayout(data_layout)

        # Modulation type selection for generator
        mod_layout = QHBoxLayout()
        mod_layout.addWidget(QLabel("Modulation:"))
        self.gen_modulation_combo = QComboBox()
        self.gen_modulation_combo.addItems([
            'bpsk', 'qpsk', '8psk', '16qam', '64qam', '256qam',
            'fsk', 'gfsk', 'msk', 'gmsk', 'am_dsb_lc', 'fm_wb',
            'ofdm_qpsk', 'dsss_bpsk'
        ])
        self.gen_modulation_combo.setCurrentText('bpsk')
        self.gen_modulation_combo.currentTextChanged.connect(self.on_gen_modulation_changed)
        mod_layout.addWidget(self.gen_modulation_combo)
        gen_layout.addLayout(mod_layout)

        # Coding type selection for generator
        coding_layout = QHBoxLayout()
        coding_layout.addWidget(QLabel("Coding:"))
        self.gen_coding_combo = QComboBox()
        self.gen_coding_combo.addItems([
            'none', 'convolutional', 'turbo', 'ldpc', 'polar', 'reed_solomon'
        ])
        self.gen_coding_combo.setCurrentText('none')
        self.gen_coding_combo.currentTextChanged.connect(self.on_gen_coding_changed)
        coding_layout.addWidget(self.gen_coding_combo)
        gen_layout.addLayout(coding_layout)

        # Signal parameters for generator
        signal_params_group = QGroupBox("Signal Parameters")
        signal_params_layout = QFormLayout(signal_params_group)

        # Symbol rate
        self.gen_symbol_rate_spinbox = QSpinBox()
        self.gen_symbol_rate_spinbox.setRange(1000, 100000)
        self.gen_symbol_rate_spinbox.setValue(10000)
        self.gen_symbol_rate_spinbox.setSuffix(" Hz")
        signal_params_layout.addRow("Symbol Rate:", self.gen_symbol_rate_spinbox)

        # SNR
        self.gen_snr_spinbox = QDoubleSpinBox()
        self.gen_snr_spinbox.setRange(-10, 30)
        self.gen_snr_spinbox.setValue(15)
        self.gen_snr_spinbox.setSuffix(" dB")
        signal_params_layout.addRow("SNR:", self.gen_snr_spinbox)

        gen_layout.addWidget(signal_params_group)

        # Generated signal info display
        info_group = QGroupBox("Generated Signal Info")
        info_layout = QVBoxLayout(info_group)
        
        self.gen_signal_info_label = QLabel("No signal generated yet")
        self.gen_signal_info_label.setStyleSheet(
            "font-family: Consolas; font-size: 9pt; "
            "background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;"
        )
        self.gen_signal_info_label.setWordWrap(True)
        info_layout.addWidget(self.gen_signal_info_label)
        
        gen_layout.addWidget(info_group)

        source_layout.addWidget(gen_group)
        source_layout.addStretch()

        panel.addTab(source_tab, "Signal Source")

        # USRP Tab
        if MODULES_COMPLETE:
            self.usrp_panel = USRPControlPanel()
            self.usrp_panel.usrp_connected.connect(self.on_usrp_connection_changed)
            self.usrp_panel.streaming_started.connect(self.on_usrp_streaming_started)
            self.usrp_panel.streaming_stopped.connect(self.on_usrp_streaming_stopped)
            panel.addTab(self.usrp_panel, "USRP Control")

        # Parameter Tab
        if MODULES_COMPLETE:
            self.param_panel = ParameterControlPanel()
            self.param_panel.parameters_changed.connect(self.on_parameters_changed)
            panel.addTab(self.param_panel, "Parameters")

        # Processing Tab
        proc_tab = QWidget()
        proc_layout = QVBoxLayout(proc_tab)

        # Processing controls
        proc_controls_group = QGroupBox("Processing Control")
        proc_controls_layout = QGridLayout(proc_controls_group)

        self.start_processing_btn = QPushButton("Start Processing")
        self.start_processing_btn.clicked.connect(self.toggle_processing)
        proc_controls_layout.addWidget(self.start_processing_btn, 0, 0)

        self.clear_results_btn = QPushButton("Clear Results")
        self.clear_results_btn.clicked.connect(self.clear_results)
        proc_controls_layout.addWidget(self.clear_results_btn, 0, 1)

        proc_layout.addWidget(proc_controls_group)

        # Processing status
        status_group = QGroupBox("Processing Status")
        status_layout = QVBoxLayout(status_group)

        self.processing_status_label = QLabel("Idle")
        self.processing_status_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        status_layout.addWidget(self.processing_status_label)

        self.processing_progress = QProgressBar()
        status_layout.addWidget(self.processing_progress)

        proc_layout.addWidget(status_group)
        proc_layout.addStretch()

        panel.addTab(proc_tab, "Processing")

        return panel

    def create_visualization_panel(self):
        """Create visualization panel"""
        panel = QTabWidget()

        # Constellation Tab
        const_tab = QWidget()
        const_layout = QVBoxLayout(const_tab)

        # Constellation controls
        const_controls = QHBoxLayout()

        self.const_clear_btn = QPushButton("Clear")
        self.const_clear_btn.clicked.connect(self.clear_constellation)
        const_controls.addWidget(self.const_clear_btn)

        const_controls.addStretch()

        const_controls.addWidget(QLabel("Max Points:"))
        self.const_points_spinbox = QSpinBox()
        self.const_points_spinbox.setRange(100, 5000)
        self.const_points_spinbox.setValue(1000)
        const_controls.addWidget(self.const_points_spinbox)

        const_layout.addLayout(const_controls)

        # Constellation plot
        if PYQTGRAPH_AVAILABLE:
            self.constellation_plot = pg.PlotWidget(title="Signal Constellation")
            self.constellation_plot.setLabel('left', 'Quadrature (Q)')
            self.constellation_plot.setLabel('bottom', 'In-phase (I)')
            self.constellation_plot.showGrid(True, True)
            self.constellation_plot.setAspectLocked(True)
            const_layout.addWidget(self.constellation_plot)
        else:
            const_layout.addWidget(QLabel("PyQtGraph not available for plots"))

        panel.addTab(const_tab, "Constellation")

        # Spectrum Tab
        spectrum_tab = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_tab)

        if PYQTGRAPH_AVAILABLE:
            self.spectrum_plot = pg.PlotWidget(title="Signal Spectrum")
            self.spectrum_plot.setLabel('left', 'Power (dB)')
            self.spectrum_plot.setLabel('bottom', 'Frequency (Hz)')
            self.spectrum_plot.showGrid(True, True)
            spectrum_layout.addWidget(self.spectrum_plot)
        else:
            spectrum_layout.addWidget(QLabel("PyQtGraph not available for plots"))

        panel.addTab(spectrum_tab, "Spectrum")

        # Time Domain Tab
        time_tab = QWidget()
        time_layout = QVBoxLayout(time_tab)

        if PYQTGRAPH_AVAILABLE:
            self.time_plot = pg.PlotWidget(title="Time Domain")
            self.time_plot.setLabel('left', 'Amplitude')
            self.time_plot.setLabel('bottom', 'Time (s)')
            self.time_plot.showGrid(True, True)
            time_layout.addWidget(self.time_plot)
        else:
            time_layout.addWidget(QLabel("PyQtGraph not available for plots"))

        panel.addTab(time_tab, "Time Domain")

        return panel

    def create_results_panel(self):
        """Create results panel"""
        panel = QTabWidget()

        # Bitstream Tab
        if MODULES_COMPLETE:
            try:
                self.bitstream_widget = VisualBitstreamWidget()
                # Ensure bit_buffer is initialized
                if not hasattr(self.bitstream_widget, 'bit_buffer'):
                    self.bitstream_widget.bit_buffer = []
                panel.addTab(self.bitstream_widget, "Visual Bitstream")

                # Bitstream analyzer
                self.bitstream_analyzer = BitstreamAnalyzer()
            except Exception as e:
                print(f"Error initializing bitstream widget: {e}")
                # Create fallback simple widget
                fallback_widget = QLabel("Bitstream display unavailable")
                panel.addTab(fallback_widget, "Visual Bitstream")
            panel.addTab(self.bitstream_analyzer, "Analysis")

        # Processing Results Tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        # Stage results
        stages_group = QGroupBox("Processing Stages")
        stages_layout = QVBoxLayout(stages_group)

        self.stages_table = QTableWidget(5, 4)
        self.stages_table.setHorizontalHeaderLabels(["Stage", "Status", "Result", "Confidence"])
        self.stages_table.setMaximumHeight(200)
        stages_layout.addWidget(self.stages_table)

        results_layout.addWidget(stages_group)

        # Detailed results
        details_group = QGroupBox("Detailed Results")
        details_layout = QVBoxLayout(details_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("font-family: Consolas; font-size: 9pt;")
        details_layout.addWidget(self.results_text)

        results_layout.addWidget(details_group)

        panel.addTab(results_tab, "Processing Results")

        # Statistics Tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        # Performance metrics
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QGridLayout(perf_group)

        perf_layout.addWidget(QLabel("Processing Time:"), 0, 0)
        self.proc_time_label = QLabel("N/A")
        perf_layout.addWidget(self.proc_time_label, 0, 1)

        perf_layout.addWidget(QLabel("Detection Accuracy:"), 1, 0)
        self.detection_acc_label = QLabel("N/A")
        perf_layout.addWidget(self.detection_acc_label, 1, 1)

        perf_layout.addWidget(QLabel("Decoding Success:"), 2, 0)
        self.decoding_success_label = QLabel("N/A")
        perf_layout.addWidget(self.decoding_success_label, 2, 1)

        perf_layout.addWidget(QLabel("Bit Error Rate:"), 3, 0)
        self.ber_label = QLabel("N/A")
        perf_layout.addWidget(self.ber_label, 3, 1)

        stats_layout.addWidget(perf_group)
        stats_layout.addStretch()

        panel.addTab(stats_tab, "Statistics")

        # Detection Comparison Tab
        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)

        # Comparison table
        comp_group = QGroupBox("Detection Accuracy Test")
        comp_layout = QVBoxLayout(comp_group)

        self.comparison_table = QTableWidget(3, 4)
        self.comparison_table.setHorizontalHeaderLabels(["Parameter", "Generated", "Detected", "Match"])
        self.comparison_table.setMaximumHeight(150)
        
        # Set row labels
        self.comparison_table.setVerticalHeaderLabels(["Modulation", "Coding", "Overall"])
        
        comp_layout.addWidget(self.comparison_table)
        comparison_layout.addWidget(comp_group)

        # Test results summary
        summary_group = QGroupBox("Test Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.test_summary_label = QLabel("No test results available yet")
        self.test_summary_label.setStyleSheet(
            "font-family: Consolas; font-size: 10pt; "
            "background-color: #f5f5f5; padding: 10px; border: 1px solid #ddd;"
        )
        self.test_summary_label.setWordWrap(True)
        summary_layout.addWidget(self.test_summary_label)
        
        comparison_layout.addWidget(summary_group)
        comparison_layout.addStretch()

        panel.addTab(comparison_tab, "Detection Test")

        return panel

    def on_source_changed(self):
        """Handle signal source change"""
        self.signal_source = 'generator' if self.generator_rb.isChecked() else 'usrp'
        print(f"Signal source changed to: {self.signal_source}")

    def on_gen_modulation_changed(self, modulation_type):
        """Handle generator modulation type change"""
        print(f"Generator modulation changed to: {modulation_type}")
        # Update signal generator configuration
        if MODULES_COMPLETE and self.signal_generator:
            try:
                self.signal_generator.set_modulation_type(modulation_type)
                
                # Update modulation parameters based on type
                params = {'symbol_rate': self.gen_symbol_rate_spinbox.value()}
                
                # Add type-specific parameters
                if modulation_type in ['fsk', 'gfsk']:
                    params['freq_deviation'] = 2000  # Default frequency deviation
                
                if modulation_type in ['gfsk', 'gmsk']:
                    params['bt_product'] = 0.3  # Default BT product
                
                if modulation_type == 'am_dsb_lc':
                    params['modulation_index'] = 0.8  # Default modulation index
                
                self.signal_generator.set_modulation_parameters(params)
                print(f"✅ Updated modulation parameters: {params}")
                
            except Exception as e:
                print(f"❌ Error updating modulation: {e}")

    def on_gen_coding_changed(self, coding_type):
        """Handle generator coding type change"""
        print(f"Generator coding changed to: {coding_type}")
        # Update signal generator configuration
        if MODULES_COMPLETE and self.signal_generator:
            try:
                self.signal_generator.set_coding_type(coding_type)
                
                # Update coding parameters based on type
                params = {}
                
                if coding_type == 'convolutional':
                    params = {
                        'constraint_length': 7,
                        'code_rate': 0.5
                    }
                elif coding_type == 'turbo':
                    params = {
                        'interleaver_size': 1024
                    }
                elif coding_type == 'ldpc':
                    params = {
                        'max_iterations': 50
                    }
                
                if params:
                    self.signal_generator.set_coding_parameters(params)
                    print(f"✅ Updated coding parameters: {params}")
                
            except Exception as e:
                print(f"❌ Error updating coding: {e}")

    def on_usrp_connection_changed(self, connected):
        """Handle USRP connection change"""
        if connected:
            print("USRP connected")
        else:
            print("USRP disconnected")

    def on_usrp_streaming_started(self):
        """Handle USRP streaming start"""
        print("USRP streaming started")
        if self.processing_active:
            self.signal_update_timer.start(100)  # Check for new samples every 100ms

    def on_usrp_streaming_stopped(self):
        """Handle USRP streaming stop"""
        print("USRP streaming stopped")
        self.signal_update_timer.stop()

    def on_parameters_changed(self, param_type, params):
        """Handle parameter changes"""
        if not MODULES_COMPLETE:
            return

        try:
            if param_type == 'modulation' and self.signal_generator:
                self.signal_generator.set_modulation_parameters(params)
            elif param_type == 'coding' and self.signal_generator:
                self.signal_generator.set_coding_parameters(params)
        except Exception as e:
            print(f"Parameter update error: {e}")

    def generate_signal(self):
        """Generate single signal"""
        if not MODULES_COMPLETE or not self.signal_generator:
            QMessageBox.warning(self, "Error", "Signal generator not available")
            return

        try:
            # Update generator configuration
            self.update_generator_config()

            # Generate signal
            result = self.signal_generator.generate_signal(duration=0.1)  # 100ms signal

            if result:
                self.current_signal = result['signal']
                
                # Update signal info display
                info_text = f"✅ Signal Generated Successfully:\n"
                info_text += f"📡 Modulation: {self.gen_modulation_combo.currentText().upper()}\n"
                info_text += f"🔐 Coding: {self.gen_coding_combo.currentText().upper()}\n"
                info_text += f"⚡ Symbol Rate: {self.gen_symbol_rate_spinbox.value():,} Hz\n"
                info_text += f"📊 SNR: {self.gen_snr_spinbox.value()} dB\n"
                info_text += f"📈 Samples: {len(self.current_signal):,}\n"
                info_text += f"📄 Data Source: {self.data_source_combo.currentText()}\n"
                info_text += f"⏱️ Duration: 100ms"
                
                self.gen_signal_info_label.setText(info_text)
                self.gen_signal_info_label.setStyleSheet(
                    "font-family: Consolas; font-size: 9pt; "
                    "background-color: #e8f5e8; padding: 5px; border: 1px solid #4CAF50; "
                    "color: #2e7d32;"
                )
                
                print(f"✅ Generated {len(self.current_signal)} samples")
                print(f"📊 Modulation: {self.gen_modulation_combo.currentText()}")
                print(f"🔐 Coding: {self.gen_coding_combo.currentText()}")

                # Process signal if processing is active
                if self.processing_active:
                    self.process_current_signal()
            else:
                self.gen_signal_info_label.setText("❌ Signal generation failed")
                self.gen_signal_info_label.setStyleSheet(
                    "font-family: Consolas; font-size: 9pt; "
                    "background-color: #ffebee; padding: 5px; border: 1px solid #f44336; "
                    "color: #c62828;"
                )
                QMessageBox.warning(self, "Error", "Signal generation failed")

        except Exception as e:
            error_text = f"❌ Signal Generation Error:\n{str(e)}"
            self.gen_signal_info_label.setText(error_text)
            self.gen_signal_info_label.setStyleSheet(
                "font-family: Consolas; font-size: 9pt; "
                "background-color: #ffebee; padding: 5px; border: 1px solid #f44336; "
                "color: #c62828;"
            )
            QMessageBox.critical(self, "Error", f"Signal generation error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Signal generation error: {str(e)}")

    def toggle_continuous_generation(self):
        """Toggle continuous generation"""
        if not MODULES_COMPLETE or not self.signal_generator:
            return

        if self.signal_generator.generating:
            self.signal_generator.stop_continuous_generation()
            self.start_continuous_btn.setText("Start Continuous")
        else:
            self.update_generator_config()
            self.signal_generator.start_continuous_generation(callback=self.on_signal_generated)
            self.start_continuous_btn.setText("Stop Continuous")

    def on_signal_generated(self, result):
        """Handle continuously generated signal (called from background thread)"""
        # Emit signal to safely handle in main thread
        self.signal_generated.emit(result)

    def on_signal_generated_safe(self, result):
        """Handle continuously generated signal (thread-safe, runs in main thread)"""
        self.current_signal = result['signal']

        if self.processing_active:
            self.process_current_signal()

    def update_generator_config(self):
        """Update signal generator configuration"""
        if not MODULES_COMPLETE or not self.signal_generator:
            return

        try:
            # Use generator control settings (priority over parameter panel)
            modulation_type = self.gen_modulation_combo.currentText()
            coding_type = self.gen_coding_combo.currentText()
            
            # Set modulation and coding types from generator controls
            self.signal_generator.set_modulation_type(modulation_type)
            self.signal_generator.set_coding_type(coding_type)
            
            # Update modulation parameters
            mod_params = {
                'symbol_rate': self.gen_symbol_rate_spinbox.value()
            }
            
            # Add type-specific modulation parameters
            if modulation_type in ['fsk', 'gfsk']:
                mod_params['freq_deviation'] = 2000
            
            if modulation_type in ['gfsk', 'gmsk']:
                mod_params['bt_product'] = 0.3
            
            if modulation_type == 'am_dsb_lc':
                mod_params['modulation_index'] = 0.8
            
            self.signal_generator.set_modulation_parameters(mod_params)
            
            # Update coding parameters
            coding_params = {}
            if coding_type == 'convolutional':
                coding_params = {
                    'constraint_length': 7,
                    'code_rate': 0.5
                }
            elif coding_type == 'turbo':
                coding_params = {
                    'interleaver_size': 1024
                }
            elif coding_type == 'ldpc':
                coding_params = {
                    'max_iterations': 50
                }
            
            if coding_params:
                self.signal_generator.set_coding_parameters(coding_params)

            # Calculate SNR and power parameters
            snr_db = self.gen_snr_spinbox.value()
            signal_power = 0  # Reference power
            noise_power = signal_power - snr_db
            
            # Update signal configuration
            self.signal_generator.current_config.update({
                'signal_power': signal_power,
                'noise_power': noise_power,
                'data_source': self.data_source_combo.currentText(),
                'duration': 0.1  # 100ms default duration
            })
            
            print(f"🔧 Generator config updated:")
            print(f"   Modulation: {modulation_type}")
            print(f"   Coding: {coding_type}")
            print(f"   Symbol Rate: {self.gen_symbol_rate_spinbox.value()} Hz")
            print(f"   SNR: {snr_db} dB")
            print(f"   Data Source: {self.data_source_combo.currentText()}")

        except Exception as e:
            print(f"❌ Generator config update error: {e}")

    def toggle_processing(self):
        """Toggle signal processing"""
        if self.processing_active:
            self.stop_processing()
        else:
            self.start_processing()

    def start_processing(self):
        """Start signal processing"""
        if not MODULES_COMPLETE or not self.processing_pipeline:
            QMessageBox.warning(self, "Error", "Processing pipeline not available")
            return

        self.processing_active = True
        self.start_processing_btn.setText("Stop Processing")
        self.processing_status_label.setText("Active")
        self.processing_status_label.setStyleSheet("color: green; font-weight: bold;")

        # Update pipeline configuration
        if self.param_panel:
            selections = self.param_panel.get_manual_selections()
            signal_params = self.param_panel.get_signal_parameters()

            self.processing_pipeline.set_user_parameters(
                modulation_type_override=selections['modulation'],
                coding_type_override=selections['coding'],
                snr_estimate=signal_params['snr_estimate']
            )

        # Start timer for USRP processing if streaming
        if (self.signal_source == 'usrp' and hasattr(self, 'usrp_panel') and 
            self.usrp_panel.streaming):
            self.signal_update_timer.start(100)

    def stop_processing(self):
        """Stop signal processing"""
        self.processing_active = False
        self.start_processing_btn.setText("Start Processing")
        self.processing_status_label.setText("Idle")
        self.processing_status_label.setStyleSheet("color: red; font-weight: bold;")

        self.signal_update_timer.stop()

    def update_signal_processing(self):
        """Update signal processing (for USRP mode)"""
        if (self.signal_source == 'usrp' and hasattr(self, 'usrp_panel') and 
            self.usrp_panel.streaming):

            # Get samples from USRP
            samples = self.usrp_panel.get_samples()
            if samples is not None and len(samples) > 0:
                self.current_signal = samples
                self.process_current_signal()

    def process_current_signal(self):
        """Process current signal through pipeline"""
        if (not MODULES_COMPLETE or not self.processing_pipeline or 
            self.current_signal is None):
            return

        try:
            start_time = time.time()

            # Process signal
            results = self.processing_pipeline.process_signal(self.current_signal)

            end_time = time.time()
            processing_time = end_time - start_time

            # Update UI with results
            self.update_processing_results(results, processing_time)
            
            # Store results for constellation plotting
            self.last_processing_results = results

            # Extract bit stream for visual display
            bit_extraction = results.get('stage_5_bit_extraction', {})
            if (bit_extraction.get('status') == 'completed' and 
                bit_extraction.get('result') is not None):

                bits = bit_extraction['result']
                if len(bits) > 0:
                    self.bitstream_widget.add_bits(bits)
                    # Safely access bit_buffer with error handling
                    if hasattr(self.bitstream_widget, 'bit_buffer'):
                        self.bitstream_analyzer.update_bits(self.bitstream_widget.bit_buffer)
                    else:
                        print("Warning: VisualBitstreamWidget missing bit_buffer attribute")

        except Exception as e:
            print(f"Signal processing error: {e}")

    def update_processing_results(self, results, processing_time):
        """Update processing results with enhanced validation metrics"""
        try:
            # Update stages table
            self.stages_table.clearContents()
            self.stages_table.setRowCount(len(results))

            stage_names = [
                "Modulation Detection",
                "Demodulation", 
                "Coding Detection",
                "Channel Decoding",
                "Bit Extraction"
            ]

            for i, (stage_key, stage_result) in enumerate(results.items()):
                # Stage name
                self.stages_table.setItem(i, 0, QTableWidgetItem(stage_names[i]))

                # Status
                status = stage_result.get('status', 'unknown')
                status_item = QTableWidgetItem(status.capitalize())
                if status == 'completed':
                    status_item.setForeground(QColor(0, 255, 0))  # Green
                elif status == 'error' or status == 'failed':
                    status_item.setForeground(QColor(255, 0, 0))  # Red
                else:
                    status_item.setForeground(QColor(255, 165, 0))  # Orange
                self.stages_table.setItem(i, 1, status_item)

                # Result
                result = stage_result.get('result', 'N/A')
                if isinstance(result, np.ndarray):
                    result_text = f"Array[{len(result)}]"
                elif isinstance(result, list):
                    result_text = f"List[{len(result)}]"
                else:
                    result_text = str(result)
                self.stages_table.setItem(i, 2, QTableWidgetItem(result_text))

                # Confidence with validation metrics
                confidence = stage_result.get('confidence', 0)
                confidence_text = f"{confidence:.1%}"
                
                # Add validation info for demodulation stage
                if 'validation_metrics' in stage_result:
                    val_metrics = stage_result['validation_metrics']
                    if 'constellation_accuracy' in val_metrics:
                        accuracy = val_metrics['constellation_accuracy']
                        confidence_text += f" (Acc: {accuracy:.1f}%)"
                
                self.stages_table.setItem(i, 3, QTableWidgetItem(confidence_text))

            # Update detailed results with validation information
            results_text = "=== SIGNAL PROCESSING RESULTS ===\n\n"

            for stage_name, stage_result in results.items():
                stage_display = stage_name.replace('stage_', '').replace('_', ' ').title()
                results_text += f"{stage_display}:\n"
                results_text += f"  Status: {stage_result.get('status', 'unknown')}\n"

                if 'result' in stage_result:
                    result = stage_result['result']
                    if isinstance(result, np.ndarray):
                        results_text += f"  Result: {len(result)} elements\n"
                    else:
                        results_text += f"  Result: {result}\n"

                if 'confidence' in stage_result:
                    results_text += f"  Confidence: {stage_result['confidence']:.1%}\n"

                # Add enhanced validation metrics for demodulation
                if 'constellation_analysis' in stage_result:
                    analysis = stage_result['constellation_analysis']
                    results_text += f"  === CONSTELLATION ANALYSIS ===\n"
                    
                    if 'evm_percent' in analysis:
                        results_text += f"    EVM: {analysis['evm_percent']:.2f}%\n"
                    
                    if 'validation_metrics' in analysis:
                        val_metrics = analysis['validation_metrics']
                        results_text += f"    Constellation Accuracy: {val_metrics.get('constellation_accuracy', 0):.1f}%\n"
                        results_text += f"    Demodulator Performance: {val_metrics.get('demodulator_performance', 'unknown')}\n"
                        results_text += f"    Estimated SNR: {val_metrics.get('snr_estimate_db', 0):.1f} dB\n"
                        results_text += f"    Quality Assessment: {val_metrics.get('constellation_quality', 'unknown')}\n"
                    
                    if 'cluster_separation' in analysis:
                        cluster_info = analysis['cluster_separation']
                        if isinstance(cluster_info, dict) and 'separation_ratio' in cluster_info:
                            results_text += f"    Cluster Separation: {cluster_info['separation_ratio']:.2f}\n"
                    
                    if 'data_pattern_analysis' in analysis:
                        pattern_info = analysis['data_pattern_analysis']
                        if isinstance(pattern_info, dict) and 'pattern_detected' in pattern_info:
                            results_text += f"    Data Pattern: {pattern_info['pattern_detected']}\n"

                if 'params' in stage_result and stage_result['params']:
                    results_text += f"  Parameters: {stage_result['params']}\n"

                results_text += "\n"

            self.results_text.setPlainText(results_text)

            # Update statistics with enhanced metrics
            self.proc_time_label.setText(f"{processing_time:.3f} sec")

            # Calculate detection accuracy
            completed_stages = sum(1 for s in results.values() if s.get('status') == 'completed')
            total_stages = len(results)
            accuracy = (completed_stages / total_stages) * 100 if total_stages > 0 else 0
            self.detection_acc_label.setText(f"{accuracy:.1f}%")

            # Enhanced decoding success with validation
            decoding_result = results.get('stage_4_channel_decoding', {})
            decoding_success = decoding_result.get('success', False)
            
            # Add validation quality indicator
            demod_result = results.get('stage_2_demodulation', {})
            if 'validation_metrics' in demod_result:
                val_metrics = demod_result['validation_metrics']
                quality = val_metrics.get('constellation_quality', 'unknown')
                success_text = f"{'Success' if decoding_success else 'Failed'} ({quality})"
            else:
                success_text = "Success" if decoding_success else "Failed"
            
            self.decoding_success_label.setText(success_text)
            self.decoding_success_label.setStyleSheet(
                "color: green; font-weight: bold;" if decoding_success else "color: red; font-weight: bold;"
            )

            # Update detection comparison if signal was generated
            if self.signal_source == 'generator':
                self.update_detection_comparison(results)

        except Exception as e:
            print(f"Results update error: {e}")

    def update_detection_comparison(self, results):
        """Update detection comparison between generated and detected parameters"""
        try:
            # Get generated parameters
            generated_mod = self.gen_modulation_combo.currentText().upper()
            generated_coding = self.gen_coding_combo.currentText().upper()
            
            # Get detected parameters
            mod_detection = results.get('stage_1_modulation_detection', {})
            detected_mod = mod_detection.get('result', 'unknown').upper() if mod_detection.get('result') else 'UNKNOWN'
            
            coding_detection = results.get('stage_3_coding_detection', {})
            detected_coding = coding_detection.get('result', 'unknown').upper() if coding_detection.get('result') else 'UNKNOWN'
            
            # Update comparison table
            self.comparison_table.setRowCount(3)
            
            # Modulation row
            self.comparison_table.setItem(0, 0, QTableWidgetItem("Modulation"))
            self.comparison_table.setItem(0, 1, QTableWidgetItem(generated_mod))
            self.comparison_table.setItem(0, 2, QTableWidgetItem(detected_mod))
            
            mod_match = generated_mod == detected_mod
            mod_match_item = QTableWidgetItem("✅ MATCH" if mod_match else "❌ NO MATCH")
            mod_match_item.setForeground(QColor(0, 128, 0) if mod_match else QColor(255, 0, 0))
            self.comparison_table.setItem(0, 3, mod_match_item)
            
            # Coding row
            self.comparison_table.setItem(1, 0, QTableWidgetItem("Coding"))
            self.comparison_table.setItem(1, 1, QTableWidgetItem(generated_coding))
            self.comparison_table.setItem(1, 2, QTableWidgetItem(detected_coding))
            
            coding_match = generated_coding == detected_coding
            coding_match_item = QTableWidgetItem("✅ MATCH" if coding_match else "❌ NO MATCH")
            coding_match_item.setForeground(QColor(0, 128, 0) if coding_match else QColor(255, 0, 0))
            self.comparison_table.setItem(1, 3, coding_match_item)
            
            # Overall row
            overall_match = mod_match and coding_match
            self.comparison_table.setItem(2, 0, QTableWidgetItem("Overall"))
            self.comparison_table.setItem(2, 1, QTableWidgetItem(f"{generated_mod} + {generated_coding}"))
            self.comparison_table.setItem(2, 2, QTableWidgetItem(f"{detected_mod} + {detected_coding}"))
            
            overall_match_item = QTableWidgetItem("✅ PERFECT MATCH" if overall_match else "❌ MISMATCH")
            overall_match_item.setForeground(QColor(0, 128, 0) if overall_match else QColor(255, 0, 0))
            self.comparison_table.setItem(2, 3, overall_match_item)
            
            # Update test summary
            mod_confidence = mod_detection.get('confidence', 0) * 100
            coding_confidence = coding_detection.get('confidence', 0) * 100
            
            summary_text = "🧪 DETECTION ACCURACY TEST RESULTS\n\n"
            summary_text += f"📡 Modulation Detection:\n"
            summary_text += f"   Generated: {generated_mod}\n"
            summary_text += f"   Detected: {detected_mod}\n"
            summary_text += f"   Confidence: {mod_confidence:.1f}%\n"
            summary_text += f"   Result: {'✅ CORRECT' if mod_match else '❌ INCORRECT'}\n\n"
            
            summary_text += f"🔐 Coding Detection:\n"
            summary_text += f"   Generated: {generated_coding}\n"
            summary_text += f"   Detected: {detected_coding}\n"
            summary_text += f"   Confidence: {coding_confidence:.1f}%\n"
            summary_text += f"   Result: {'✅ CORRECT' if coding_match else '❌ INCORRECT'}\n\n"
            
            summary_text += f"🎯 Overall Test Result:\n"
            if overall_match:
                summary_text += f"   ✅ SUCCESS - All parameters detected correctly!\n"
                summary_text += f"   🎉 Detection accuracy: 100%\n"
                bg_color = "#e8f5e8"
                border_color = "#4CAF50"
                text_color = "#2e7d32"
            else:
                accuracy = (int(mod_match) + int(coding_match)) * 50  # 50% per parameter
                summary_text += f"   ❌ PARTIAL SUCCESS - Some parameters incorrect\n"
                summary_text += f"   📊 Detection accuracy: {accuracy}%\n"
                if accuracy >= 50:
                    bg_color = "#fff3e0"
                    border_color = "#ff9800"
                    text_color = "#ef6c00"
                else:
                    bg_color = "#ffebee"
                    border_color = "#f44336"
                    text_color = "#c62828"
            
            summary_text += f"\n💡 Tips for Testing:\n"
            summary_text += f"   • Try different modulation types to test robustness\n"
            summary_text += f"   • Adjust SNR to test noise tolerance\n"
            summary_text += f"   • Use different coding schemes to test versatility"
            
            self.test_summary_label.setText(summary_text)
            self.test_summary_label.setStyleSheet(
                f"font-family: Consolas; font-size: 10pt; "
                f"background-color: {bg_color}; padding: 10px; border: 1px solid {border_color}; "
                f"color: {text_color};"
            )
            
            # Resize columns to fit content
            self.comparison_table.resizeColumnsToContents()
            
        except Exception as e:
            print(f"Detection comparison update error: {e}")

    def clear_results(self):
        """Clear all results"""
        try:
            # Clear stages table
            self.stages_table.clearContents()

            # Clear results text
            self.results_text.clear()

            # Clear comparison table and summary
            self.comparison_table.clearContents()
            self.test_summary_label.setText("No test results available yet")
            self.test_summary_label.setStyleSheet(
                "font-family: Consolas; font-size: 10pt; "
                "background-color: #f5f5f5; padding: 10px; border: 1px solid #ddd;"
            )

            # Clear bitstream
            if MODULES_COMPLETE:
                self.bitstream_widget.clear_display()

            # Clear plots
            self.clear_constellation()
            if PYQTGRAPH_AVAILABLE and hasattr(self, 'spectrum_plot'):
                self.spectrum_plot.clear()
            if PYQTGRAPH_AVAILABLE and hasattr(self, 'time_plot'):
                self.time_plot.clear()

            # Reset statistics
            self.proc_time_label.setText("N/A")
            self.detection_acc_label.setText("N/A")
            self.decoding_success_label.setText("N/A")
            self.ber_label.setText("N/A")

            # Clear signal info
            self.gen_signal_info_label.setText("No signal generated yet")
            self.gen_signal_info_label.setStyleSheet(
                "font-family: Consolas; font-size: 9pt; "
                "background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;"
            )

        except Exception as e:
            print(f"Clear results error: {e}")

    def clear_constellation(self):
        """Clear constellation plot"""
        if PYQTGRAPH_AVAILABLE and hasattr(self, 'constellation_plot'):
            self.constellation_plot.clear()

    def update_plots(self):
        """Update visualization plots"""
        if self.current_signal is None or not PYQTGRAPH_AVAILABLE:
            return

        try:
            signal = self.current_signal

            # Limit signal length for performance
            if len(signal) > 2048:
                signal = signal[:2048]

            # Update constellation plot
            if MODULES_COMPLETE and hasattr(self, 'constellation_plot'):
                constellation_data = None
                
                # First try to get constellation from processing results
                if hasattr(self, 'last_processing_results'):
                    results = self.last_processing_results
                    demod_result = results.get('stage_2_demodulation', {})
                    constellation_data = demod_result.get('constellation', None)
                
                # If no constellation from pipeline, generate from current signal
                if constellation_data is None or len(constellation_data) == 0:
                    if np.iscomplexobj(signal):
                        # Use signal directly as constellation points
                        constellation_data = signal
                    else:
                        # Convert real signal to complex
                        # Simple approach: use consecutive pairs as I,Q
                        if len(signal) >= 2:
                            i_samples = signal[::2]
                            q_samples = signal[1::2] if len(signal) > 1 else np.zeros_like(i_samples)
                            min_len = min(len(i_samples), len(q_samples))
                            constellation_data = i_samples[:min_len] + 1j * q_samples[:min_len]

                if constellation_data is not None and len(constellation_data) > 0:
                    # print(f"📊 Updating constellation plot with {len(constellation_data)} points")
                    max_points = self.const_points_spinbox.value()
                    if len(constellation_data) > max_points:
                        indices = np.random.choice(len(constellation_data), max_points, replace=False)
                        plot_constellation = np.array(constellation_data)[indices]
                    else:
                        plot_constellation = np.array(constellation_data)

                    if len(plot_constellation) > 0:
                        i_data = np.real(plot_constellation)
                        q_data = np.imag(plot_constellation)

                        self.constellation_plot.clear()
                        self.constellation_plot.plot(i_data, q_data, pen=None, symbol='o',
                                                   symbolSize=2, symbolBrush=(100, 255, 100, 150))
                        # print(f"✅ Constellation plot updated: I range [{i_data.min():.3f}, {i_data.max():.3f}], Q range [{q_data.min():.3f}, {q_data.max():.3f}]")
                else:
                    print("❌ No constellation data available for plotting")

            # Update spectrum plot
            if len(signal) >= 64 and hasattr(self, 'spectrum_plot'):
                fft_data = fft(signal)
                freqs = fftfreq(len(signal), 1/1e6)  # Assume 1 MS/s
                freqs_shifted = fftshift(freqs)
                fft_shifted = fftshift(fft_data)

                magnitude_db = 20 * np.log10(np.abs(fft_shifted) + 1e-12)

                self.spectrum_plot.clear()
                self.spectrum_plot.plot(freqs_shifted, magnitude_db, pen='y')

            # Update time domain plot
            if hasattr(self, 'time_plot'):
                sample_rate = 1e6  # Assume 1 MS/s
                t = np.arange(len(signal)) / sample_rate

                self.time_plot.clear()
                if np.iscomplexobj(signal):
                    self.time_plot.plot(t, np.real(signal), pen='b', name='I')
                    self.time_plot.plot(t, np.imag(signal), pen='r', name='Q')
                else:
                    self.time_plot.plot(t, signal, pen='y', name='Signal')

        except Exception as e:
            # Silently handle plot errors to avoid spam
            pass


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Complete SDR Suite")
    app.setApplicationVersion("5.0 Professional")

    # Set application style
    app.setStyle('Fusion')

    # Dark theme
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(dark_palette)

    # Additional stylesheet
    app.setStyleSheet("""
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
        QTabWidget::pane {
            border: 1px solid #555555;
            background-color: #353535;
        }
        QTabBar::tab {
            background-color: #404040;
            color: white;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #0d7377;
            font-weight: bold;
        }
    """)

    # Create and show main window
    try:
        window = CompleteSdrMainWindow()
        window.show()

        print("🚀 Complete SDR Suite - Professional Edition")
        print("=" * 60)
        print("✅ Features Available:")
        print("   • USRP Hardware Integration")
        print("   • Enhanced Signal Generation (User-selectable)")
        print("   • Auto-detection with Manual Override")
        print("   • Visual Bitstream Display (1=Green, 0=Black)")
        print("   • Comprehensive Parameter Control")
        print("   • 5-Stage Processing Pipeline")
        print("   • Real-time Constellation Display")
        print("   • Professional GUI with Dark Theme")

        if MODULES_COMPLETE:
            print("   ✅ All modules: Available")
        else:
            print("   ⚠️  Some modules: Limited functionality")

        if USRP_AVAILABLE:
            print("   ✅ USRP support: Available")
        else:
            print("   ⚠️  USRP support: Simulator only")

        print("\n🎯 Usage:")
        print("   1. Select signal source (Generator or USRP)")
        print("   2. Configure parameters (Auto or Manual)")
        print("   3. Start processing")
        print("   4. View results in real-time")

        sys.exit(app.exec())

    except Exception as e:
        print(f"❌ Application startup error: {e}")
        QMessageBox.critical(None, "Startup Error", f"Failed to start application:\n{str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
