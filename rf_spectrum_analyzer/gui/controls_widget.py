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
from PySide6.QtCore import Signal, Qt
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
        self.device_combo.addItems(['rtlsdr', 'hackrf', 'pluto', 'usrp', 'USRP N2xx/X3xx Series', 'soapy', 'file'])
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
        self.update_device_status(self.settings.sdr.device_type, False)
        self.update_frequency_display(self.settings.sdr.center_frequency)
    
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
    def _on_frequency_changed(self):
        """Handle frequency input change."""
        try:
            freq_mhz = float(self.frequency_input.text())
            frequency = freq_mhz * 1e6
            self.frequency_changed.emit(frequency)
            self.settings.sdr.center_frequency = frequency
        except ValueError:
            pass
    
    def _on_sample_rate_changed(self, rate_text: str):
        """Handle sample rate change."""
        try:
            rate_mhz = float(rate_text.split()[0])
            sample_rate = rate_mhz * 1e6
            self.sample_rate_changed.emit(sample_rate)
            self.settings.sdr.sample_rate = sample_rate
        except (ValueError, IndexError):
            pass
    
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
            self.modulation_detected.emit(modulation)
    
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