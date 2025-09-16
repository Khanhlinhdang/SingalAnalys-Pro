"""
Controls Widget - Control panel for RF Spectrum Analyzer
Provides controls for device configuration, signal processing, and display settings.
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, 
    QSlider, QCheckBox, QTabWidget, QFormLayout, QLineEdit
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QFont, QValidator

from rf_spectrum_analyzer.config.settings import Settings


class FrequencyValidator(QValidator):
    """Custom validator for frequency input."""
    
    def validate(self, input_str: str, pos: int):
        try:
            value = float(input_str)
            if 0 <= value <= 6000:  # 0 to 6 GHz
                return (QValidator.Acceptable, input_str, pos)
            else:
                return (QValidator.Invalid, input_str, pos)
        except ValueError:
            if input_str == "" or input_str.replace(".", "").replace("e", "").replace("-", "").isdigit():
                return (QValidator.Intermediate, input_str, pos)
            return (QValidator.Invalid, input_str, pos)


class ControlsWidget(QWidget):
    """Widget containing all control elements."""
    
    # Control signals
    start_clicked = Signal()
    stop_clicked = Signal()
    device_changed = Signal(str)
    frequency_changed = Signal(float)
    sample_rate_changed = Signal(float)
    bandwidth_changed = Signal(float)  # New signal for bandwidth
    gain_changed = Signal(float)
    fft_size_changed = Signal(int)
    window_changed = Signal(str)
    averaging_changed = Signal(int)
    settings_changed = Signal()
    
    # New signals for modulation/demodulation and encoding/decoding
    modulation_detected = Signal(str)
    modulation_changed = Signal(str)
    symbol_rate_changed = Signal(float)
    demodulation_toggled = Signal(bool)
    encoding_detected = Signal(str)
    encoding_changed = Signal(str)
    code_rate_changed = Signal(str)
    decoding_toggled = Signal(bool)
    auto_detect_modulation_toggled = Signal(bool)
    auto_detect_coding_toggled = Signal(bool)
    
    # New signals for detection features
    manual_detection_triggered = Signal()
    tdma_detection_triggered = Signal()
    auto_detection_toggled = Signal(bool)
    advanced_analysis_toggled = Signal(bool)
    detection_threshold_changed = Signal(float)
    detection_interval_changed = Signal(int)
    
    # SpyServer specific signals
    spyserver_config_changed = Signal(dict)
    
    # Frequency analysis signals
    frequency_range_changed = Signal(float, float)  # f1, f2
    frequency_markers_toggled = Signal(bool)
    center_frequency_locked = Signal(bool)
    analysis_bandwidth_changed = Signal(float)
    
    # New sequential workflow signals
    demodulate_triggered = Signal()  # Step 2: Demodulate button clicked
    decode_triggered = Signal()     # Step 3: Decode button clicked
    capture_ready = Signal(bool)    # Capture status changed
    
    # Frequency analysis signals
    frequency_range_changed = Signal(float, float)  # f1, f2
    frequency_markers_toggled = Signal(bool)
    center_frequency_locked = Signal(bool)
    analysis_bandwidth_changed = Signal(float)
    
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.acquisition_active = False
        
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        """Setup the control panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        title_label = QLabel("RF Spectrum Analyzer Controls")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create tab widget for organized controls
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create control tabs
        self.create_device_tab()
        self.create_processing_tab()
        self.create_detection_tab()
        self.create_frequency_analysis_tab()  # New tab for frequency analysis
        self.create_display_tab()
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        # Control buttons at bottom
        self.create_control_buttons(layout)
    
    def create_device_tab(self):
        """Create device configuration tab."""
        device_widget = QWidget()
        layout = QVBoxLayout(device_widget)
        
        # Device selection group
        device_group = QGroupBox("Device Selection")
        device_layout = QFormLayout(device_group)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(['rtlsdr', 'hackrf', 'pluto', 'usrp', 'USRP N2xx/X3xx Series', 'soapy', 'spyserver', 'file'])
        self.device_combo.currentTextChanged.connect(self.device_changed.emit)
        device_layout.addRow("Device Type:", self.device_combo)
        
        # Status indicators section
        status_layout = QHBoxLayout()
        
        # Ready status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { font-weight: bold; color: green; }")
        status_layout.addWidget(QLabel("Status:"))
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        # FPS indicator
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("QLabel { font-weight: bold; }")
        status_layout.addWidget(self.fps_label)
        
        device_layout.addRow("", status_layout)
        
        # Device status and frequency info
        info_layout = QHBoxLayout()
        
        self.device_status_label = QLabel("Device: Disconnected")
        self.device_status_label.setStyleSheet("QLabel { color: red; }")
        info_layout.addWidget(self.device_status_label)
        info_layout.addStretch()
        
        self.frequency_label = QLabel("Freq: 0 MHz")
        self.frequency_label.setStyleSheet("QLabel { font-weight: bold; }")
        info_layout.addWidget(self.frequency_label)
        
        device_layout.addRow("", info_layout)
        
        layout.addWidget(device_group)
        
        # Frequency control group
        freq_group = QGroupBox("Frequency Control")
        freq_layout = QFormLayout(freq_group)
        
        # Center frequency
        self.frequency_input = QLineEdit()
        self.frequency_input.setValidator(FrequencyValidator())
        self.frequency_input.editingFinished.connect(self._on_frequency_changed)
        self.frequency_input.textChanged.connect(self._on_frequency_typing)  # Real-time while typing
        freq_layout.addRow("Center Freq (MHz):", self.frequency_input)
        
        # Frequency presets
        self.freq_preset_combo = QComboBox()
        freq_presets = [
            "Custom", "FM Radio (100 MHz)", "Air Traffic (125 MHz)",
            "Marine (156 MHz)", "PMR446 (446 MHz)", "ISM 2.4G (2440 MHz)"
        ]
        self.freq_preset_combo.addItems(freq_presets)
        self.freq_preset_combo.currentTextChanged.connect(self._on_freq_preset_changed)
        freq_layout.addRow("Presets:", self.freq_preset_combo)
        
        layout.addWidget(freq_group)
        
        # Sample rate and gain group
        radio_group = QGroupBox("Radio Parameters")
        radio_layout = QFormLayout(radio_group)
        
        # Sample rate
        self.sample_rate_combo = QComboBox()
        sample_rates = ["0.25", "0.5", "1.0", "2.0", "2.4", "3.2", "5.0", "8.0", "10.0", "20.0"]
        self.sample_rate_combo.addItems([f"{rate} MHz" for rate in sample_rates])
        self.sample_rate_combo.currentTextChanged.connect(self._on_sample_rate_changed)
        radio_layout.addRow("Sample Rate:", self.sample_rate_combo)
        
        # Bandwidth control
        self.bandwidth_combo = QComboBox()
        bandwidths = ["0.25", "0.5", "1.0", "2.0", "2.4", "3.2", "5.0", "8.0", "10.0", "20.0"]
        self.bandwidth_combo.addItems([f"{bw} MHz" for bw in bandwidths])
        self.bandwidth_combo.currentTextChanged.connect(self._on_bandwidth_changed)
        radio_layout.addRow("Bandwidth:", self.bandwidth_combo)
        
        # RF Gain
        self.gain_spinbox = QDoubleSpinBox()
        self.gain_spinbox.setRange(0.0, 100.0)
        self.gain_spinbox.setSuffix(" dB")
        self.gain_spinbox.valueChanged.connect(self.gain_changed.emit)
        radio_layout.addRow("RF Gain:", self.gain_spinbox)
        
        # AGC checkbox
        self.agc_checkbox = QCheckBox("Automatic Gain Control")
        self.agc_checkbox.toggled.connect(self._on_agc_toggled)
        radio_layout.addRow("", self.agc_checkbox)
        
        layout.addWidget(radio_group)
        
        # SpyServer specific controls group
        self.spyserver_group = QGroupBox("SpyServer Configuration")
        spyserver_layout = QFormLayout(self.spyserver_group)
        
        # Server info label
        server_info_label = QLabel("Default: RomanPort Airspy Mini (24 MHz - 1.8 GHz)")
        server_info_label.setStyleSheet("QLabel { color: #2E8B57; font-weight: bold; font-size: 10px; }")
        spyserver_layout.addRow("", server_info_label)
        
        # Host and port
        connection_layout = QHBoxLayout()
        self.spyserver_host_input = QLineEdit("64.31.248.40")
        self.spyserver_port_input = QSpinBox()
        self.spyserver_port_input.setRange(1, 65535)
        self.spyserver_port_input.setValue(63863)
        connection_layout.addWidget(self.spyserver_host_input)
        connection_layout.addWidget(QLabel(":"))
        connection_layout.addWidget(self.spyserver_port_input)
        spyserver_layout.addRow("Host:Port:", connection_layout)
        
        # Timeout
        self.spyserver_timeout_spinbox = QDoubleSpinBox()
        self.spyserver_timeout_spinbox.setRange(1.0, 60.0)
        self.spyserver_timeout_spinbox.setValue(15.0)
        self.spyserver_timeout_spinbox.setSuffix(" s")
        spyserver_layout.addRow("Timeout:", self.spyserver_timeout_spinbox)
        
        # Device capabilities label
        capabilities_label = QLabel("• Max Sample Rate: 3.0 MSPS\n• Resolution: 12-bit\n• Gain Stages: 22 levels")
        capabilities_label.setStyleSheet("QLabel { color: #4169E1; font-size: 9px; }")
        spyserver_layout.addRow("Capabilities:", capabilities_label)
        
        # Connection test button
        self.spyserver_test_button = QPushButton("Test Connection")
        self.spyserver_test_button.clicked.connect(self._on_spyserver_test_connection)
        spyserver_layout.addRow("", self.spyserver_test_button)
        
        layout.addWidget(self.spyserver_group)
        
        # Hide SpyServer controls initially
        self.spyserver_group.setVisible(False)
        
        # Connect device type change to show/hide SpyServer controls
        self.device_combo.currentTextChanged.connect(self._on_device_type_changed)
        
        self.tab_widget.addTab(device_widget, "Device")
    
    def create_processing_tab(self):
        """Create signal processing tab."""
        processing_widget = QWidget()
        layout = QVBoxLayout(processing_widget)
        
        # FFT settings group
        fft_group = QGroupBox("FFT Settings")
        fft_layout = QFormLayout(fft_group)
        
        # FFT size
        self.fft_size_combo = QComboBox()
        fft_sizes = ["256", "512", "1024", "2048", "4096", "8192"]
        self.fft_size_combo.addItems(fft_sizes)
        self.fft_size_combo.currentTextChanged.connect(self._on_fft_size_changed)
        fft_layout.addRow("FFT Size:", self.fft_size_combo)
        
        # Window function
        self.window_combo = QComboBox()
        windows = ["hann", "hamming", "blackman", "bartlett", "kaiser"]
        self.window_combo.addItems(windows)
        self.window_combo.currentTextChanged.connect(self.window_changed.emit)
        fft_layout.addRow("Window:", self.window_combo)
        
        # Overlap
        self.overlap_spinbox = QDoubleSpinBox()
        self.overlap_spinbox.setRange(0.0, 0.9)
        self.overlap_spinbox.setSingleStep(0.1)
        self.overlap_spinbox.setSuffix(" %")
        self.overlap_spinbox.valueChanged.connect(self._on_overlap_changed)
        fft_layout.addRow("Overlap:", self.overlap_spinbox)
        
        layout.addWidget(fft_group)
        
        # Averaging and smoothing group
        avg_group = QGroupBox("Averaging & Smoothing")
        avg_layout = QFormLayout(avg_group)
        
        # Averaging
        self.averaging_spinbox = QSpinBox()
        self.averaging_spinbox.setRange(1, 100)
        self.averaging_spinbox.valueChanged.connect(self.averaging_changed.emit)
        avg_layout.addRow("Averaging:", self.averaging_spinbox)
        
        layout.addWidget(avg_group)
        
        # Filter settings group
        filter_group = QGroupBox("Digital Filters")
        filter_layout = QFormLayout(filter_group)
        
        # Filter enable
        self.filter_enable_checkbox = QCheckBox("Enable Filtering")
        filter_layout.addRow("", self.filter_enable_checkbox)
        
        # Filter type
        self.filter_type_combo = QComboBox()
        filter_types = ["lowpass", "highpass", "bandpass", "bandstop"]
        self.filter_type_combo.addItems(filter_types)
        filter_layout.addRow("Filter Type:", self.filter_type_combo)
        
        # Filter cutoffs
        self.filter_low_spinbox = QDoubleSpinBox()
        self.filter_low_spinbox.setRange(0.01, 0.99)
        self.filter_low_spinbox.setSingleStep(0.01)
        filter_layout.addRow("Low Cutoff:", self.filter_low_spinbox)
        
        self.filter_high_spinbox = QDoubleSpinBox()
        self.filter_high_spinbox.setRange(0.01, 0.99)
        self.filter_high_spinbox.setSingleStep(0.01)
        filter_layout.addRow("High Cutoff:", self.filter_high_spinbox)
        
        layout.addWidget(filter_group)
        
        # Modulation Analysis group
        mod_group = QGroupBox("Modulation Analysis")
        mod_layout = QFormLayout(mod_group)
        
        # Auto-detect modulation
        self.auto_detect_mod_checkbox = QCheckBox("Auto-detect Modulation")
        self.auto_detect_mod_checkbox.toggled.connect(self._on_auto_detect_toggled)
        mod_layout.addRow("", self.auto_detect_mod_checkbox)
        
        # Modulation type selection
        self.modulation_combo = QComboBox()
        modulation_types = ["Unknown", "PSK", "QPSK", "8PSK", "QAM16", "QAM64", "QAM256", "FSK", "GFSK", "MSK", "OFDM", "AM", "FM"]
        self.modulation_combo.addItems(modulation_types)
        self.modulation_combo.currentTextChanged.connect(self._on_modulation_changed)
        mod_layout.addRow("Detected/Selected:", self.modulation_combo)
        
        # Symbol rate
        self.symbol_rate_spinbox = QDoubleSpinBox()
        self.symbol_rate_spinbox.setRange(100, 10000000)  # 100 Hz to 10 MHz
        self.symbol_rate_spinbox.setSuffix(" Hz")
        self.symbol_rate_spinbox.valueChanged.connect(self._on_symbol_rate_changed)
        mod_layout.addRow("Symbol Rate:", self.symbol_rate_spinbox)
        
        # Demodulation controls
        self.demod_enable_checkbox = QCheckBox("Enable Demodulation")
        self.demod_enable_checkbox.toggled.connect(self._on_demod_toggled)
        mod_layout.addRow("", self.demod_enable_checkbox)
        
        layout.addWidget(mod_group)
        
        # Encoding/Decoding group
        coding_group = QGroupBox("Coding Analysis")
        coding_layout = QFormLayout(coding_group)
        
        # Auto-detect encoding
        self.auto_detect_coding_checkbox = QCheckBox("Auto-detect Encoding")
        self.auto_detect_coding_checkbox.toggled.connect(self._on_auto_detect_coding_toggled)
        coding_layout.addRow("", self.auto_detect_coding_checkbox)
        
        # Encoding type selection
        self.encoding_combo = QComboBox()
        encoding_types = ["None", "Hamming", "BCH", "Reed-Solomon", "Convolutional", "Turbo", "LDPC", "Polar"]
        self.encoding_combo.addItems(encoding_types)
        self.encoding_combo.currentTextChanged.connect(self._on_encoding_changed)
        coding_layout.addRow("Detected/Selected:", self.encoding_combo)
        
        # Code rate
        self.code_rate_combo = QComboBox()
        code_rates = ["1/2", "2/3", "3/4", "5/6", "7/8"]
        self.code_rate_combo.addItems(code_rates)
        self.code_rate_combo.currentTextChanged.connect(self._on_code_rate_changed)
        coding_layout.addRow("Code Rate:", self.code_rate_combo)
        
        # Decoding controls
        self.decode_enable_checkbox = QCheckBox("Enable Decoding")
        self.decode_enable_checkbox.toggled.connect(self._on_decode_toggled)
        coding_layout.addRow("", self.decode_enable_checkbox)
        
        layout.addWidget(coding_group)

        self.tab_widget.addTab(processing_widget, "Processing")
    
    def create_detection_tab(self):
        """Create signal detection tab using sdr._detection module."""
        detection_widget = QWidget()
        layout = QVBoxLayout(detection_widget)
        
        # Auto Detection group
        auto_group = QGroupBox("Automatic Detection")
        auto_layout = QFormLayout(auto_group)
        
        # Auto detection enable
        self.auto_detection_checkbox = QCheckBox("Enable Auto Detection")
        self.auto_detection_checkbox.setToolTip("Automatically detect signals when power > threshold")
        self.auto_detection_checkbox.toggled.connect(self._on_auto_detection_toggled)
        auto_layout.addRow("", self.auto_detection_checkbox)
        
        # Energy threshold
        self.energy_threshold_spinbox = QDoubleSpinBox()
        self.energy_threshold_spinbox.setRange(-120.0, 0.0)
        self.energy_threshold_spinbox.setSuffix(" dBm")
        self.energy_threshold_spinbox.setSingleStep(1.0)
        self.energy_threshold_spinbox.setToolTip("Signal power threshold for auto detection")
        self.energy_threshold_spinbox.valueChanged.connect(self._on_detection_threshold_changed)
        auto_layout.addRow("Power Threshold:", self.energy_threshold_spinbox)
        
        # Detection interval
        self.detection_interval_spinbox = QSpinBox()
        self.detection_interval_spinbox.setRange(10, 10000)
        self.detection_interval_spinbox.setSuffix(" ms")
        self.detection_interval_spinbox.setSingleStep(10)
        self.detection_interval_spinbox.setToolTip("Periodic detection interval")
        self.detection_interval_spinbox.valueChanged.connect(self._on_detection_interval_changed)
        auto_layout.addRow("Check Interval:", self.detection_interval_spinbox)
        
        layout.addWidget(auto_group)
        
        # Manual Detection group
        manual_group = QGroupBox("Manual Detection")
        manual_layout = QVBoxLayout(manual_group)
        
        # Manual detection button
        self.manual_detect_button = QPushButton("🔍 Detect Signals")
        self.manual_detect_button.setToolTip("Manually trigger signal detection")
        self.manual_detect_button.clicked.connect(self._on_manual_detection_triggered)
        manual_layout.addWidget(self.manual_detect_button)
        
        # TDMA detection button
        self.tdma_detect_button = QPushButton("📡 TDMA Analysis")
        self.tdma_detect_button.setToolTip("Detect and analyze TDMA bursts")
        self.tdma_detect_button.clicked.connect(self._on_tdma_detection_triggered)
        manual_layout.addWidget(self.tdma_detect_button)
        
        layout.addWidget(manual_group)
        
        # Advanced Detection group
        advanced_group = QGroupBox("Advanced Features")
        advanced_layout = QFormLayout(advanced_group)
        
        # Advanced analysis mode
        self.advanced_analysis_checkbox = QCheckBox("Advanced Analysis Mode")
        self.advanced_analysis_checkbox.setToolTip("Enable advanced signal analysis features")
        self.advanced_analysis_checkbox.toggled.connect(self._on_advanced_analysis_toggled)
        advanced_layout.addRow("", self.advanced_analysis_checkbox)
        
        # Spectrum sensing
        self.spectrum_sensing_checkbox = QCheckBox("Spectrum Sensing")
        self.spectrum_sensing_checkbox.setToolTip("Enable multi-band spectrum sensing")
        advanced_layout.addRow("", self.spectrum_sensing_checkbox)
        
        # Cognitive radio mode
        self.cognitive_radio_checkbox = QCheckBox("Cognitive Radio Mode")
        self.cognitive_radio_checkbox.setToolTip("Enable cognitive radio capabilities")
        advanced_layout.addRow("", self.cognitive_radio_checkbox)
        
        layout.addWidget(advanced_group)
        
        # Detection Status group
        status_group = QGroupBox("Detection Status")
        status_layout = QFormLayout(status_group)
        
        # Detection status indicators
        self.detection_status_label = QLabel("🔴 No Signal")
        self.detection_status_label.setStyleSheet("color: red; font-weight: bold;")
        status_layout.addRow("Status:", self.detection_status_label)
        
        # SNR display
        self.snr_label = QLabel("-- dB")
        status_layout.addRow("SNR:", self.snr_label)
        
        # Confidence display
        self.confidence_label = QLabel("--.-%")
        status_layout.addRow("Confidence:", self.confidence_label)
        
        layout.addWidget(status_group)
        
        self.tab_widget.addTab(detection_widget, "Detection")
    
    def create_display_tab(self):
        """Create display settings tab."""
        display_widget = QWidget()
        layout = QVBoxLayout(display_widget)
        
        # Spectrum display group
        spectrum_group = QGroupBox("Spectrum Display")
        spectrum_layout = QFormLayout(spectrum_group)
        
        # Y-axis range
        self.y_min_spinbox = QDoubleSpinBox()
        self.y_min_spinbox.setRange(-200.0, 0.0)
        self.y_min_spinbox.setSuffix(" dB")
        self.y_min_spinbox.valueChanged.connect(self._on_y_range_changed)
        spectrum_layout.addRow("Y Min:", self.y_min_spinbox)
        
        self.y_max_spinbox = QDoubleSpinBox()
        self.y_max_spinbox.setRange(-50.0, 50.0)
        self.y_max_spinbox.setSuffix(" dB")
        self.y_max_spinbox.valueChanged.connect(self._on_y_range_changed)
        spectrum_layout.addRow("Y Max:", self.y_max_spinbox)
        
        # Reference level
        self.ref_level_spinbox = QDoubleSpinBox()
        self.ref_level_spinbox.setRange(-100.0, 50.0)
        self.ref_level_spinbox.setSuffix(" dB")
        spectrum_layout.addRow("Ref Level:", self.ref_level_spinbox)
        
        layout.addWidget(spectrum_group)
        
        # Update rates group
        update_group = QGroupBox("Update Rates")
        update_layout = QFormLayout(update_group)
        
        self.spectrum_rate_spinbox = QDoubleSpinBox()
        self.spectrum_rate_spinbox.setRange(1.0, 100.0)
        self.spectrum_rate_spinbox.setSuffix(" Hz")
        update_layout.addRow("Spectrum Rate:", self.spectrum_rate_spinbox)
        
        self.waterfall_rate_spinbox = QDoubleSpinBox()
        self.waterfall_rate_spinbox.setRange(1.0, 50.0)
        self.waterfall_rate_spinbox.setSuffix(" Hz")
        update_layout.addRow("Waterfall Rate:", self.waterfall_rate_spinbox)
        
        layout.addWidget(update_group)
        
        # Theme group
        theme_group = QGroupBox("Appearance")
        theme_layout = QFormLayout(theme_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        theme_layout.addRow("Theme:", self.theme_combo)
        
        layout.addWidget(theme_group)
        
        self.tab_widget.addTab(display_widget, "Display")
    
    def create_frequency_analysis_tab(self):
        """Create frequency analysis tab with new sequential workflow."""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)
        
        # Frequency Range Selection Group
        range_group = QGroupBox("2MHz Bandwidth Capture (f1-f2)")
        range_layout = QFormLayout(range_group)
        
        # Enable frequency range selection
        self.freq_range_enabled = QCheckBox("Enable 2MHz Bandwidth Capture")
        self.freq_range_enabled.stateChanged.connect(self._on_frequency_range_toggle)
        range_layout.addRow(self.freq_range_enabled)
        
        # Start frequency (f1)
        self.freq_start_spinbox = QDoubleSpinBox()
        self.freq_start_spinbox.setDecimals(3)
        self.freq_start_spinbox.setRange(0.001, 6000.0)  # 1 kHz to 6 GHz
        self.freq_start_spinbox.setValue(432.0)  # 432 MHz default
        self.freq_start_spinbox.setSuffix(" MHz")
        self.freq_start_spinbox.setSingleStep(1.0)
        self.freq_start_spinbox.valueChanged.connect(self._on_frequency_range_changed)
        range_layout.addRow("Start Frequency (f1):", self.freq_start_spinbox)
        
        # End frequency (f2) - automatically set to f1 + 2MHz
        self.freq_end_spinbox = QDoubleSpinBox()
        self.freq_end_spinbox.setDecimals(3)
        self.freq_end_spinbox.setRange(0.001, 6000.0)  # 1 kHz to 6 GHz
        self.freq_end_spinbox.setValue(434.0)  # f1 + 2MHz default
        self.freq_end_spinbox.setSuffix(" MHz")
        self.freq_end_spinbox.setSingleStep(1.0)
        self.freq_end_spinbox.valueChanged.connect(self._on_frequency_range_changed)
        range_layout.addRow("End Frequency (f2):", self.freq_end_spinbox)
        
        # Fixed 2MHz bandwidth display
        self.bandwidth_label = QLabel("2.000 MHz")
        self.bandwidth_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        range_layout.addRow("Capture Bandwidth:", self.bandwidth_label)
        
        layout.addWidget(range_group)
        
        # Spectrum Markers Group
        markers_group = QGroupBox("Interactive Frequency Markers")
        markers_layout = QFormLayout(markers_group)
        
        # Show frequency markers
        self.show_freq_markers = QCheckBox("Show f1/f2 Horizontal Lines")
        self.show_freq_markers.setChecked(True)
        self.show_freq_markers.stateChanged.connect(self._on_frequency_markers_toggle)
        markers_layout.addRow(self.show_freq_markers)
        
        # Lock center frequency
        self.lock_center_freq = QCheckBox("Lock Center Frequency")
        self.lock_center_freq.stateChanged.connect(self._on_center_frequency_lock)
        markers_layout.addRow(self.lock_center_freq)
        
        # Current center frequency display
        self.current_center_label = QLabel("433.000 MHz")
        self.current_center_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        markers_layout.addRow("Current Center:", self.current_center_label)
        
        # Current span display
        self.current_span_label = QLabel("2.000 MHz")
        self.current_span_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        markers_layout.addRow("Current Span:", self.current_span_label)
        
        layout.addWidget(markers_group)
        
        # Sequential Processing Workflow
        workflow_group = QGroupBox("Sequential Processing Workflow")
        workflow_layout = QVBoxLayout(workflow_group)
        
        # Step 1: Capture
        capture_layout = QHBoxLayout()
        self.capture_status_label = QLabel("Step 1: Ready to capture 2MHz bandwidth")
        self.capture_status_label.setStyleSheet("font-weight: bold; color: #757575;")
        capture_layout.addWidget(self.capture_status_label)
        capture_layout.addStretch()
        workflow_layout.addLayout(capture_layout)
        
        # Step 2: Demodulation
        demod_layout = QHBoxLayout()
        self.demod_button = QPushButton("Step 2: DEMODULATE")
        self.demod_button.setEnabled(False)
        self.demod_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
            QPushButton:hover:enabled {
                background-color: #1976D2;
            }
        """)
        self.demod_button.clicked.connect(self._on_demodulate_clicked)
        demod_layout.addWidget(self.demod_button)
        
        self.demod_status_label = QLabel("Detect and demodulate signal")
        self.demod_status_label.setStyleSheet("color: #757575;")
        demod_layout.addWidget(self.demod_status_label)
        demod_layout.addStretch()
        
        self.detected_modulation_label = QLabel("")
        self.detected_modulation_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        demod_layout.addWidget(self.detected_modulation_label)
        
        workflow_layout.addLayout(demod_layout)
        
        # Step 3: Decoding
        decode_layout = QHBoxLayout()
        self.decode_button = QPushButton("Step 3: DECODE")
        self.decode_button.setEnabled(False)
        self.decode_button.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
            QPushButton:hover:enabled {
                background-color: #E64A19;
            }
        """)
        self.decode_button.clicked.connect(self._on_decode_clicked)
        decode_layout.addWidget(self.decode_button)
        
        self.decode_status_label = QLabel("Detect encoding and decode data")
        self.decode_status_label.setStyleSheet("color: #757575;")
        decode_layout.addWidget(self.decode_status_label)
        decode_layout.addStretch()
        
        self.detected_encoding_label = QLabel("")
        self.detected_encoding_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        decode_layout.addWidget(self.detected_encoding_label)
        
        workflow_layout.addLayout(decode_layout)
        
        layout.addWidget(workflow_group)
        
        # Initially disable frequency range controls
        self._update_frequency_range_controls()
        
        self.tab_widget.addTab(analysis_widget, "Sequential Analysis")
    
    def create_control_buttons(self, layout):
        """Create start/stop control buttons."""
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; font-weight: bold; }")
        self.start_button.clicked.connect(self.start_clicked.emit)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; font-weight: bold; }")
        self.stop_button.clicked.connect(self.stop_clicked.emit)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
    
    def load_settings(self):
        """Load settings into controls."""
        # Device settings
        self.device_combo.setCurrentText(self.settings.sdr.device_type)
        self.frequency_input.setText(f"{self.settings.sdr.center_frequency/1e6:.3f}")
        self.gain_spinbox.setValue(self.settings.sdr.gain)
        
        # Sample rate
        sample_rate_mhz = self.settings.sdr.sample_rate / 1e6
        for i in range(self.sample_rate_combo.count()):
            if f"{sample_rate_mhz:.1f}" in self.sample_rate_combo.itemText(i):
                self.sample_rate_combo.setCurrentIndex(i)
                break
        
        # Bandwidth
        bandwidth_mhz = self.settings.sdr.bandwidth / 1e6
        for i in range(self.bandwidth_combo.count()):
            if f"{bandwidth_mhz:.1f}" in self.bandwidth_combo.itemText(i):
                self.bandwidth_combo.setCurrentIndex(i)
                break
        
        # Processing settings
        self.fft_size_combo.setCurrentText(str(self.settings.dsp.fft_size))
        self.window_combo.setCurrentText(self.settings.dsp.window_type)
        self.overlap_spinbox.setValue(self.settings.dsp.overlap * 100)
        self.averaging_spinbox.setValue(self.settings.dsp.averaging)
        
        # Filter settings
        self.filter_type_combo.setCurrentText(self.settings.dsp.filter_type)
        self.filter_low_spinbox.setValue(self.settings.dsp.filter_cutoff_low)
        self.filter_high_spinbox.setValue(self.settings.dsp.filter_cutoff_high)
        
        # Display settings
        self.y_min_spinbox.setValue(self.settings.gui.spectrum_min_db)
        self.y_max_spinbox.setValue(self.settings.gui.spectrum_max_db)
        self.ref_level_spinbox.setValue(self.settings.gui.spectrum_ref_level)
        self.spectrum_rate_spinbox.setValue(self.settings.gui.spectrum_update_rate)
        self.waterfall_rate_spinbox.setValue(self.settings.gui.waterfall_update_rate)
        self.theme_combo.setCurrentText(self.settings.gui.theme)
        
        # Initialize status indicators
        self.update_status("Ready")
        self.update_fps(0)
        
        # Initialize frequency analysis settings
        if hasattr(self, 'freq_range_enabled'):
            self.freq_range_enabled.setChecked(False)
            self.show_freq_markers.setChecked(True)
            self.lock_center_freq.setChecked(False)
            # self.realtime_spectrum.setChecked(True)  # Widget removed in sequential workflow
            # self.peak_hold_enabled.setChecked(False)  # Widget removed in sequential workflow
            self._update_frequency_range_controls()
        self.update_device_status(self.settings.sdr.device_type, False)
        self.update_frequency_display(self.settings.sdr.center_frequency)
        
        # Load SpyServer settings
        self.spyserver_host_input.setText(self.settings.sdr.spyserver_host)
        self.spyserver_port_input.setValue(self.settings.sdr.spyserver_port)
        self.spyserver_timeout_spinbox.setValue(self.settings.sdr.spyserver_timeout)
        
        # Update device type visibility  
        # Force trigger to ensure SpyServer controls are shown correctly
        self._on_device_type_changed(self.settings.sdr.device_type)
        
        # Load detection settings if available
        if hasattr(self.settings, 'detection'):
            self.auto_detection_checkbox.setChecked(self.settings.detection.auto_detection_enabled)
            self.energy_threshold_spinbox.setValue(self.settings.detection.energy_threshold_dbm)
            self.detection_interval_spinbox.setValue(self.settings.detection.detection_interval_ms)
        else:
            # Set default values
            self.auto_detection_checkbox.setChecked(False)
            self.energy_threshold_spinbox.setValue(-80.0)
            self.detection_interval_spinbox.setValue(100)
    
    def set_acquisition_state(self, active: bool):
        """Update controls based on acquisition state."""
        self.acquisition_active = active
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(active)
        
        # Disable device settings during acquisition
        self.device_combo.setEnabled(not active)
        self.sample_rate_combo.setEnabled(not active)
    
    def set_frequency(self, frequency: float):
        """Set frequency value from external source."""
        self.frequency_input.setText(f"{frequency/1e6:.3f}")
    
    # Event handlers
    def _on_frequency_typing(self, text: str):
        """Handle real-time frequency input while typing."""
        try:
            # Only emit if we have a valid number
            if text and text.replace('.', '').replace('-', '').isdigit():
                freq_mhz = float(text)
                frequency = freq_mhz * 1e6
                
                # Show typing feedback with light blue background
                self.frequency_input.setStyleSheet("QLineEdit { background-color: #F0F8FF; }")
                
                # Use timer to debounce rapid typing
                if hasattr(self, '_freq_timer'):
                    self._freq_timer.stop()
                    
                self._freq_timer = QTimer()
                self._freq_timer.timeout.connect(lambda: self._emit_frequency_change(frequency))
                self._freq_timer.setSingleShot(True)
                self._freq_timer.start(300)  # 300ms delay
                
        except ValueError:
            # Show invalid input with light red background
            self.frequency_input.setStyleSheet("QLineEdit { background-color: #FFF8F8; }")
    
    def _emit_frequency_change(self, frequency: float):
        """Emit frequency change after debounce delay."""
        self.frequency_changed.emit(frequency)
        self.settings.sdr.center_frequency = frequency
        # Reset to normal background
        self.frequency_input.setStyleSheet("")
        # Update status
        self.status_label.setText(f"Frequency: {frequency/1e6:.3f} MHz")
        self.status_label.setStyleSheet("QLabel { font-weight: bold; color: blue; }")
        QTimer.singleShot(2000, lambda: self._reset_status_label())
    
    def _on_frequency_changed(self):
        """Handle frequency input change with real-time feedback."""
        try:
            freq_mhz = float(self.frequency_input.text())
            frequency = freq_mhz * 1e6
            
            # Provide immediate visual feedback
            self.frequency_input.setStyleSheet("QLineEdit { background-color: #E8F5E8; }")  # Green background
            
            # Emit signal for real-time change
            self.frequency_changed.emit(frequency)
            self.settings.sdr.center_frequency = frequency
            
            # Reset background color after short delay
            QTimer.singleShot(500, lambda: self.frequency_input.setStyleSheet(""))
            
        except ValueError:
            # Show error feedback
            self.frequency_input.setStyleSheet("QLineEdit { background-color: #FFE8E8; }")  # Red background
            QTimer.singleShot(1000, lambda: self.frequency_input.setStyleSheet(""))
    
    def _on_sample_rate_changed(self, rate_text: str):
        """Handle sample rate change."""
        try:
            rate_mhz = float(rate_text.split()[0])
            sample_rate = rate_mhz * 1e6
            self.sample_rate_changed.emit(sample_rate)
            self.settings.sdr.sample_rate = sample_rate
        except (ValueError, IndexError):
            pass
    
    def _on_bandwidth_changed(self, bw_text: str):
        """Handle bandwidth change with real-time feedback."""
        try:
            bw_mhz = float(bw_text.split()[0])
            bandwidth = bw_mhz * 1e6
            
            # Provide immediate visual feedback
            self.bandwidth_combo.setStyleSheet("QComboBox { background-color: #E8F5E8; }")  # Green background
            
            # Emit signal for real-time change
            self.bandwidth_changed.emit(bandwidth)
            
            # Add bandwidth to settings if not exists
            if not hasattr(self.settings.sdr, 'bandwidth'):
                self.settings.sdr.bandwidth = bandwidth
            else:
                self.settings.sdr.bandwidth = bandwidth
            
            # Update status
            self.status_label.setText(f"Bandwidth: {bw_mhz:.1f} MHz")
            self.status_label.setStyleSheet("QLabel { font-weight: bold; color: blue; }")
            
            # Reset background color after short delay
            QTimer.singleShot(500, lambda: self.bandwidth_combo.setStyleSheet(""))
            QTimer.singleShot(2000, lambda: self._reset_status_label())
            
        except (ValueError, IndexError):
            # Show error feedback
            self.bandwidth_combo.setStyleSheet("QComboBox { background-color: #FFE8E8; }")  # Red background
            QTimer.singleShot(1000, lambda: self.bandwidth_combo.setStyleSheet(""))
    
    def _on_fft_size_changed(self, size_text: str):
        """Handle FFT size change."""
        try:
            fft_size = int(size_text)
            self.fft_size_changed.emit(fft_size)
            self.settings.dsp.fft_size = fft_size
        except ValueError:
            pass
    
    def _on_overlap_changed(self, value: float):
        """Handle overlap change."""
        overlap = value / 100.0
        self.settings.dsp.overlap = overlap
    
    def _on_agc_toggled(self, checked: bool):
        """Handle AGC toggle."""
        self.gain_spinbox.setEnabled(not checked)
        self.settings.sdr.agc = checked
    
    def _on_y_range_changed(self):
        """Handle Y-axis range change."""
        self.settings.gui.spectrum_min_db = self.y_min_spinbox.value()
        self.settings.gui.spectrum_max_db = self.y_max_spinbox.value()
    
    def _on_theme_changed(self, theme: str):
        """Handle theme change."""
        self.settings.gui.theme = theme
        self.settings_changed.emit()
    
    def _on_freq_preset_changed(self, preset: str):
        """Handle frequency preset change."""
        presets = {
            "FM Radio (100 MHz)": 100.0,
            "Air Traffic (125 MHz)": 125.0,
            "Marine (156 MHz)": 156.0,
            "PMR446 (446 MHz)": 446.0,
            "ISM 2.4G (2440 MHz)": 2440.0
        }
        
        if preset in presets:
            freq_mhz = presets[preset]
            self.frequency_input.setText(f"{freq_mhz:.1f}")
            self._on_frequency_changed()
    
    # New callback methods for modulation/demodulation and encoding/decoding
    def _on_auto_detect_toggled(self, checked: bool):
        """Handle auto-detect modulation toggle."""
        self.modulation_combo.setEnabled(not checked)
        self.auto_detect_modulation_toggled.emit(checked)
        self.settings.dsp.auto_detect_modulation = checked
    
    def _on_modulation_changed(self, modulation: str):
        """Handle modulation type change."""
        self.modulation_changed.emit(modulation)
        self.settings.dsp.modulation_type = modulation
        
        # Update symbol rate defaults based on modulation type
        default_rates = {
            "PSK": 1000, "QPSK": 2000, "8PSK": 3000,
            "QAM16": 4000, "QAM64": 6000, "QAM256": 8000,
            "FSK": 1200, "GFSK": 9600, "MSK": 1200,
            "OFDM": 2000, "AM": 1000, "FM": 15000
        }
        if modulation in default_rates:
            self.symbol_rate_spinbox.setValue(default_rates[modulation])
    
    def _on_symbol_rate_changed(self, rate: float):
        """Handle symbol rate change."""
        self.symbol_rate_changed.emit(rate)
        self.settings.dsp.symbol_rate = rate
    
    def _on_demod_toggled(self, checked: bool):
        """Handle demodulation enable toggle."""
        self.demodulation_toggled.emit(checked)
        self.settings.dsp.demodulation_enabled = checked
    
    def _on_auto_detection_toggled(self, enabled):
        """Handle auto detection toggle."""
        self.auto_detection_toggled.emit(enabled)
        if hasattr(self.settings, 'detection'):
            self.settings.detection.auto_detection_enabled = enabled
    
    def _on_detection_threshold_changed(self, threshold):
        """Handle detection threshold change."""
        self.detection_threshold_changed.emit(threshold)
        if hasattr(self.settings, 'detection'):
            self.settings.detection.energy_threshold_dbm = threshold
    
    def _on_detection_interval_changed(self, interval):
        """Handle detection interval change."""
        self.detection_interval_changed.emit(interval)
        if hasattr(self.settings, 'detection'):
            self.settings.detection.detection_interval_ms = interval
    
    def _on_manual_detection_triggered(self):
        """Handle manual detection trigger."""
        self.manual_detection_triggered.emit()
    
    def _on_tdma_detection_triggered(self):
        """Handle TDMA detection trigger."""
        self.tdma_detection_triggered.emit()
    
    def _on_advanced_analysis_toggled(self, enabled):
        """Handle advanced analysis toggle."""
        self.advanced_analysis_toggled.emit(enabled)
    
    def update_detection_status(self, detected, snr_db=None, confidence=None):
        """Update detection status indicators."""
        if detected:
            self.detection_status_label.setText("🟢 Signal Detected")
            self.detection_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.detection_status_label.setText("🔴 No Signal")
            self.detection_status_label.setStyleSheet("color: red; font-weight: bold;")
        
        if snr_db is not None:
            self.snr_label.setText(f"{snr_db:.1f} dB")
        else:
            self.snr_label.setText("-- dB")
        
        if confidence is not None:
            self.confidence_label.setText(f"{confidence:.1f}%")
        else:
            self.confidence_label.setText("--.-%")
    
    def _on_auto_detect_coding_toggled(self, checked: bool):
        """Handle auto-detect coding toggle."""
        self.encoding_combo.setEnabled(not checked)
        self.auto_detect_coding_toggled.emit(checked)
        self.settings.dsp.auto_detect_coding = checked
    
    def _on_encoding_changed(self, encoding: str):
        """Handle encoding type change."""
        self.encoding_changed.emit(encoding)
        self.settings.dsp.encoding_type = encoding
    
    def _on_code_rate_changed(self, rate: str):
        """Handle code rate change."""
        self.code_rate_changed.emit(rate)
        self.settings.dsp.code_rate = rate
    
    def _on_decode_toggled(self, checked: bool):
        """Handle decoding enable toggle."""
        self.decoding_toggled.emit(checked)
        self.settings.dsp.decoding_enabled = checked
    
    def update_detected_modulation(self, modulation: str):
        """Update the detected modulation display."""
        if self.auto_detect_mod_checkbox.isChecked():
            self.modulation_combo.setCurrentText(modulation)
    
    def _on_device_type_changed(self, device_type: str):
        """Handle device type change to show/hide specific controls."""
        # Show/hide SpyServer controls
        is_spyserver = device_type == "spyserver"
        self.spyserver_group.setVisible(is_spyserver)
        
        # Emit device changed signal
        self.device_changed.emit(device_type)
    
    def _on_spyserver_test_connection(self):
        """Test SpyServer connection."""
        try:
            # Import here to avoid circular imports
            from sdrconnect import SpyServerClient, SDRConfig as SDRConnectConfig
            
            host = self.spyserver_host_input.text()
            port = self.spyserver_port_input.value()
            timeout = self.spyserver_timeout_spinbox.value()
            
            config = SDRConnectConfig(host=host, port=port, timeout=timeout)
            client = SpyServerClient(config)
            
            self.spyserver_test_button.setText("Testing...")
            self.spyserver_test_button.setEnabled(False)
            
            # Try to connect
            client.connect()
            device_info = client.get_device_info()
            client.disconnect()
            
            # Success
            self.spyserver_test_button.setText("✓ Connected")
            self.spyserver_test_button.setStyleSheet("QPushButton { color: green; font-weight: bold; }")
            
            # Update settings
            if not hasattr(self.settings.sdr, 'spyserver_host'):
                self.settings.sdr.spyserver_host = host
            else:
                self.settings.sdr.spyserver_host = host
            if not hasattr(self.settings.sdr, 'spyserver_port'):
                self.settings.sdr.spyserver_port = port
            else:
                self.settings.sdr.spyserver_port = port
            if not hasattr(self.settings.sdr, 'spyserver_timeout'):
                self.settings.sdr.spyserver_timeout = timeout
            else:
                self.settings.sdr.spyserver_timeout = timeout
            
        except ImportError:
            self.spyserver_test_button.setText("✗ sdrconnect not installed")
            self.spyserver_test_button.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
        except Exception as e:
            self.spyserver_test_button.setText(f"✗ Failed: {str(e)[:20]}...")
            self.spyserver_test_button.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
        finally:
            # Reset button after 3 seconds
            from PySide6.QtCore import QTimer
            QTimer.singleShot(3000, self._reset_spyserver_test_button)
    
    def _reset_spyserver_test_button(self):
        """Reset SpyServer test button to default state."""
        self.spyserver_test_button.setText("Test Connection")
        self.spyserver_test_button.setStyleSheet("")
        self.spyserver_test_button.setEnabled(True)
    
    def get_spyserver_config(self) -> dict:
        """Get SpyServer configuration from controls."""
        return {
            'host': self.spyserver_host_input.text(),
            'port': self.spyserver_port_input.value(),
            'timeout': self.spyserver_timeout_spinbox.value()
        }
    
    def update_detected_encoding(self, encoding: str):
        """Update the detected encoding display."""
        if self.auto_detect_coding_checkbox.isChecked():
            self.encoding_combo.setCurrentText(encoding)
            self.encoding_detected.emit(encoding)
    
    def update_status(self, status: str):
        """Update the ready status indicator."""
        self.status_label.setText(status)
        if "Ready" in status:
            self.status_label.setStyleSheet("QLabel { font-weight: bold; color: green; }")
        elif "Running" in status or "Active" in status:
            self.status_label.setStyleSheet("QLabel { font-weight: bold; color: blue; }")
        else:
            self.status_label.setStyleSheet("QLabel { font-weight: bold; color: orange; }")
    
    def update_fps(self, fps: int):
        """Update the FPS indicator."""
        self.fps_label.setText(f"FPS: {fps}")
    
    def update_device_status(self, device: str, connected: bool):
        """Update the device status indicator."""
        if connected:
            self.device_status_label.setText(f"Device: {device} (Connected)")
            self.device_status_label.setStyleSheet("QLabel { color: green; }")
        else:
            self.device_status_label.setText(f"Device: {device} (Disconnected)")
            self.device_status_label.setStyleSheet("QLabel { color: red; }")
    
    def update_frequency_display(self, frequency: float):
        """Update the frequency display."""
        self.frequency_label.setText(f"Freq: {frequency/1e6:.3f} MHz")
        
        # Update current center frequency display in frequency analysis tab
        if hasattr(self, 'current_center_label'):
            self.current_center_label.setText(f"{frequency/1e6:.3f} MHz")
    
    # Frequency Analysis Tab Callbacks
    def _on_frequency_range_toggle(self, state):
        """Handle frequency range analysis toggle."""
        enabled = state == Qt.Checked
        self._update_frequency_range_controls()
        
        if enabled:
            # Emit initial frequency range
            f1 = self.freq_start_spinbox.value() * 1e6  # Convert MHz to Hz
            f2 = self.freq_end_spinbox.value() * 1e6
            self.frequency_range_changed.emit(f1, f2)
    
    def _update_frequency_range_controls(self):
        """Update the enabled state of frequency range controls."""
        enabled = self.freq_range_enabled.isChecked()
        self.freq_start_spinbox.setEnabled(enabled)
        self.freq_end_spinbox.setEnabled(enabled)
        self.lock_center_freq.setEnabled(enabled)
    
    def _on_frequency_range_changed(self):
        """Handle frequency range change - automatically maintain 2MHz bandwidth."""
        if self.freq_range_enabled.isChecked():
            # Get current f1 value
            f1 = self.freq_start_spinbox.value()  # MHz
            
            # Automatically set f2 to f1 + 2MHz
            f2 = f1 + 2.0
            
            # Temporarily disconnect signals to avoid recursion
            self.freq_end_spinbox.valueChanged.disconnect(self._on_frequency_range_changed)
            self.freq_end_spinbox.setValue(f2)
            self.freq_end_spinbox.valueChanged.connect(self._on_frequency_range_changed)
            
            # Convert to Hz and emit signal
            f1_hz = f1 * 1e6
            f2_hz = f2 * 1e6
            self.frequency_range_changed.emit(f1_hz, f2_hz)
            
            # Update bandwidth display
            self.bandwidth_label.setText("2.000 MHz")
            
            # Update center frequency if locked
            if self.lock_center_freq.isChecked():
                center_freq = (f1 + f2) / 2
                # Update the frequency input field
                if hasattr(self, 'frequency_input'):
                    self.frequency_input.setText(f"{center_freq:.3f}")
                    self.frequency_changed.emit(center_freq * 1e6)  # Convert to Hz
            
            # Update current center display
            if hasattr(self, 'current_center_label'):
                center_freq = (f1 + f2) / 2
                self.current_center_label.setText(f"{center_freq:.3f} MHz")
    
    def _on_frequency_markers_toggle(self, state):
        """Handle frequency markers toggle."""
        enabled = state == Qt.Checked
        self.frequency_markers_toggled.emit(enabled)
    
    def _on_center_frequency_lock(self, state):
        """Handle center frequency lock toggle."""
        locked = state == Qt.Checked
        self.center_frequency_locked.emit(locked)
        
        if locked:
            # Automatically set center frequency to middle of range
            f1 = self.freq_start_spinbox.value() * 1e6
            f2 = self.freq_end_spinbox.value() * 1e6
            center_freq = (f1 + f2) / 2
            self.frequency_spinbox.setValue(center_freq / 1e6)
    
    def update_frequency_range_from_spectrum(self, f1: float, f2: float):
        """Update frequency range controls from spectrum widget interaction."""
        # Temporarily disconnect signals to avoid recursion
        self.freq_start_spinbox.valueChanged.disconnect(self._on_frequency_range_changed)
        self.freq_end_spinbox.valueChanged.disconnect(self._on_frequency_range_changed)
        
        self.freq_start_spinbox.setValue(f1 / 1e6)  # Convert Hz to MHz
        self.freq_end_spinbox.setValue(f2 / 1e6)
        
        # Automatically adjust f2 to maintain 2MHz bandwidth without emitting signals
        f1_val = self.freq_start_spinbox.value()
        f2_val = f1_val + 2.0  # Always 2MHz bandwidth
        self.freq_end_spinbox.setValue(f2_val)
        
        # Update bandwidth display
        self.bandwidth_label.setText("2.000 MHz")
        
        # Reconnect signals
        self.freq_start_spinbox.valueChanged.connect(self._on_frequency_range_changed)
        self.freq_end_spinbox.valueChanged.connect(self._on_frequency_range_changed)
    
    def _enforce_2mhz_bandwidth(self):
        """Enforce 2MHz bandwidth between f1 and f2."""
        f1 = self.freq_start_spinbox.value()
        f2 = f1 + 2.0  # Always 2MHz bandwidth
        self.freq_end_spinbox.setValue(f2)
        
        # Update bandwidth display
        self.bandwidth_label.setText("2.000 MHz")
        
        # Update span display
        if hasattr(self, 'current_span_label'):
            self.current_span_label.setText("2.000 MHz")
    
    # Sequential Workflow Callbacks
    def _on_demodulate_clicked(self):
        """Handle demodulate button click."""
        self.demod_button.setEnabled(False)
        self.demod_status_label.setText("Processing...")
        self.demod_status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        
        # Emit signal to start demodulation
        self.demodulate_triggered.emit()
    
    def _on_decode_clicked(self):
        """Handle decode button click."""
        self.decode_button.setEnabled(False)
        self.decode_status_label.setText("Processing...")
        self.decode_status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        
        # Emit signal to start decoding
        self.decode_triggered.emit()
    
    def update_capture_status(self, ready: bool):
        """Update capture status and enable/disable demodulate button."""
        if ready:
            self.capture_status_label.setText("Step 1: ✓ 2MHz bandwidth captured")
            self.capture_status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
            self.demod_button.setEnabled(True)
            self.capture_ready.emit(True)
        else:
            self.capture_status_label.setText("Step 1: Ready to capture 2MHz bandwidth")
            self.capture_status_label.setStyleSheet("font-weight: bold; color: #757575;")
            self.demod_button.setEnabled(False)
            self.decode_button.setEnabled(False)
            self.capture_ready.emit(False)
            
            # Reset workflow status
            self.reset_workflow_status()
    
    def update_demodulation_result(self, success: bool, modulation_type: str = ""):
        """Update demodulation result and enable/disable decode button."""
        if success and modulation_type:
            self.demod_status_label.setText("✓ Completed")
            self.demod_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.detected_modulation_label.setText(f"Found: {modulation_type}")
            self.decode_button.setEnabled(True)
        else:
            self.demod_status_label.setText("✗ Failed or no signal detected")
            self.demod_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            self.detected_modulation_label.setText("")
            self.decode_button.setEnabled(False)
        
        # Re-enable demodulate button for retry
        self.demod_button.setEnabled(True)
    
    def update_decoding_result(self, success: bool, encoding_type: str = ""):
        """Update decoding result."""
        if success and encoding_type:
            self.decode_status_label.setText("✓ Completed")
            self.decode_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.detected_encoding_label.setText(f"Found: {encoding_type}")
        else:
            self.decode_status_label.setText("✗ Failed or no encoding detected")
            self.decode_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            self.detected_encoding_label.setText("")
        
        # Re-enable decode button for retry
        self.decode_button.setEnabled(True)
    
    def reset_workflow_status(self):
        """Reset all workflow status to initial state."""
        # Reset demodulation status
        self.demod_status_label.setText("Detect and demodulate signal")
        self.demod_status_label.setStyleSheet("color: #757575;")
        self.detected_modulation_label.setText("")
        
        # Reset decoding status
        self.decode_status_label.setText("Detect encoding and decode data")
        self.decode_status_label.setStyleSheet("color: #757575;")
        self.detected_encoding_label.setText("")
        
        # Re-enable buttons
        self.demod_button.setEnabled(True)
        self.decode_button.setEnabled(True)
    
    def update_frequency_display(self, frequency: float):
        """Update frequency display when device frequency changes."""
        try:
            # Temporarily disconnect signal to avoid loops
            if hasattr(self, 'frequency_input'):
                self.frequency_input.textChanged.disconnect()  # Use textChanged for QLineEdit
                self.frequency_input.setText(f"{frequency / 1e6:.3f}")  # Use setText for QLineEdit
                self.frequency_input.textChanged.connect(self._on_frequency_typing)
                
        except Exception as e:
            print(f"Error updating frequency display: {e}")
    
    def update_bandwidth_display(self, bandwidth: float):
        """Update bandwidth display when device bandwidth changes."""
        try:
            # Update bandwidth combo box to reflect current setting
            if hasattr(self, 'bandwidth_combo'):
                # Find matching value in combo box
                bandwidth_mhz = bandwidth / 1e6
                combo_index = -1
                for i in range(self.bandwidth_combo.count()):
                    combo_text = self.bandwidth_combo.itemText(i)
                    if str(bandwidth_mhz) in combo_text:
                        combo_index = i
                        break
                
                if combo_index >= 0:
                    # Temporarily disconnect signal to avoid loops
                    self.bandwidth_combo.currentTextChanged.disconnect()
                    self.bandwidth_combo.setCurrentIndex(combo_index)
                    self.bandwidth_combo.currentTextChanged.connect(self._on_bandwidth_changed)
                
        except Exception as e:
            print(f"Error updating bandwidth display: {e}")
    
    def _reset_status_label(self):
        """Reset status label to default."""
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("QLabel { font-weight: bold; color: green; }")