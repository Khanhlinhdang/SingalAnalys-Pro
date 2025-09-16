"""
Settings Dialog for RF Spectrum Analyzer
Provides comprehensive configuration interface for all application settings
"""

import sys
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QLineEdit, QPushButton, QGroupBox, QSlider, QLabel,
    QDialogButtonBox, QColorDialog, QFontDialog, QMessageBox,
    QScrollArea, QSplitter, QTextEdit, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QPalette

from rf_spectrum_analyzer.config.settings import (
    AppSettings, DeviceSettings, ProcessingSettings,
    DisplaySettings, AdvancedSettings
)
from rf_spectrum_analyzer.utils.logger import get_logger

logger = get_logger('settings_dialog')

class SettingsDialog(QDialog):
    """Main settings dialog with tabbed interface"""
    
    settings_changed = Signal(object)  # Emits AppSettings object
    
    def __init__(self, current_settings: AppSettings, parent=None):
        super().__init__(parent)
        self.current_settings = current_settings
        self.temp_settings = None
        
        self.setWindowTitle("RF Spectrum Analyzer - Settings")
        self.setModal(True)
        self.resize(800, 600)
        
        self.setup_ui()
        self.load_current_settings()
        
        # Auto-apply timer for real-time preview
        self.auto_apply_timer = QTimer()
        self.auto_apply_timer.timeout.connect(self.preview_changes)
        self.auto_apply_timer.setSingleShot(True)
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.device_tab = DeviceSettingsTab(self.current_settings.device)
        self.processing_tab = ProcessingSettingsTab(self.current_settings.processing)
        self.display_tab = DisplaySettingsTab(self.current_settings.display)
        self.advanced_tab = AdvancedSettingsTab(self.current_settings.advanced)
        
        self.tab_widget.addTab(self.device_tab, "Device")
        self.tab_widget.addTab(self.processing_tab, "Processing")
        self.tab_widget.addTab(self.display_tab, "Display")
        self.tab_widget.addTab(self.advanced_tab, "Advanced")
        
        # Connect change signals for real-time preview
        self.device_tab.settings_changed.connect(self.on_settings_changed)
        self.processing_tab.settings_changed.connect(self.on_settings_changed)
        self.display_tab.settings_changed.connect(self.on_settings_changed)
        self.advanced_tab.settings_changed.connect(self.on_settings_changed)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | 
            QDialogButtonBox.Apply | QDialogButtonBox.RestoreDefaults
        )
        
        button_box.accepted.connect(self.accept_settings)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_settings)
        button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)
        
        layout.addWidget(button_box)
        
        self.button_box = button_box
    
    def load_current_settings(self):
        """Load current settings into all tabs"""
        self.device_tab.load_settings(self.current_settings.device)
        self.processing_tab.load_settings(self.current_settings.processing)
        self.display_tab.load_settings(self.current_settings.display)
        self.advanced_tab.load_settings(self.current_settings.advanced)
    
    def on_settings_changed(self):
        """Handle settings change for real-time preview"""
        self.auto_apply_timer.stop()
        self.auto_apply_timer.start(500)  # 500ms delay
    
    def preview_changes(self):
        """Preview changes without permanently applying them"""
        try:
            new_settings = self.get_current_settings()
            if new_settings:
                self.settings_changed.emit(new_settings)
        except Exception as e:
            logger.error(f"Preview failed: {str(e)}")
    
    def get_current_settings(self) -> Optional[AppSettings]:
        """Get current settings from all tabs"""
        try:
            device_settings = self.device_tab.get_settings()
            processing_settings = self.processing_tab.get_settings()
            display_settings = self.display_tab.get_settings()
            advanced_settings = self.advanced_tab.get_settings()
            
            return AppSettings(
                device=device_settings,
                processing=processing_settings,
                display=display_settings,
                advanced=advanced_settings
            )
        except Exception as e:
            logger.error(f"Failed to get current settings: {str(e)}")
            return None
    
    def apply_settings(self):
        """Apply current settings"""
        new_settings = self.get_current_settings()
        if new_settings:
            self.current_settings = new_settings
            self.settings_changed.emit(new_settings)
            logger.info("Settings applied")
    
    def accept_settings(self):
        """Accept and apply settings, then close dialog"""
        self.apply_settings()
        self.accept()
    
    def restore_defaults(self):
        """Restore default settings"""
        reply = QMessageBox.question(
            self,
            "Restore Defaults",
            "Are you sure you want to restore all settings to their default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            default_settings = AppSettings()
            self.current_settings = default_settings
            self.load_current_settings()
            self.settings_changed.emit(default_settings)
            logger.info("Settings restored to defaults")

class DeviceSettingsTab(QWidget):
    """Device configuration tab"""
    
    settings_changed = Signal()
    
    def __init__(self, settings: DeviceSettings):
        super().__init__()
        self.settings = settings
        self.setup_ui()
    
    def setup_ui(self):
        """Setup device settings UI"""
        layout = QVBoxLayout(self)
        
        # Device selection group
        device_group = QGroupBox("Device Selection")
        device_layout = QFormLayout(device_group)
        
        self.device_type_combo = QComboBox()
        self.device_type_combo.addItems(['rtlsdr', 'hackrf', 'pluto', 'soapy'])
        self.device_type_combo.currentTextChanged.connect(self.settings_changed.emit)
        device_layout.addRow("Device Type:", self.device_type_combo)
        
        self.device_id_edit = QLineEdit()
        self.device_id_edit.textChanged.connect(self.settings_changed.emit)
        device_layout.addRow("Device ID:", self.device_id_edit)
        
        layout.addWidget(device_group)
        
        # Frequency settings group
        freq_group = QGroupBox("Frequency Settings")
        freq_layout = QFormLayout(freq_group)
        
        self.center_freq_spin = QDoubleSpinBox()
        self.center_freq_spin.setRange(0, 6000)
        self.center_freq_spin.setSuffix(" MHz")
        self.center_freq_spin.setDecimals(3)
        self.center_freq_spin.valueChanged.connect(self.settings_changed.emit)
        freq_layout.addRow("Center Frequency:", self.center_freq_spin)
        
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems([
            '240 kHz', '960 kHz', '1.92 MHz', '2.4 MHz', '3.2 MHz'
        ])
        self.sample_rate_combo.currentTextChanged.connect(self.settings_changed.emit)
        freq_layout.addRow("Sample Rate:", self.sample_rate_combo)
        
        layout.addWidget(freq_group)
        
        # Gain settings group
        gain_group = QGroupBox("Gain Settings")
        gain_layout = QFormLayout(gain_group)
        
        self.auto_gain_check = QCheckBox("Automatic Gain Control")
        self.auto_gain_check.toggled.connect(self.settings_changed.emit)
        gain_layout.addRow(self.auto_gain_check)
        
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0, 60)
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.setDecimals(1)
        self.gain_spin.valueChanged.connect(self.settings_changed.emit)
        gain_layout.addRow("Manual Gain:", self.gain_spin)
        
        layout.addWidget(gain_group)
        
        # Advanced device settings
        advanced_group = QGroupBox("Advanced Device Settings")
        advanced_layout = QFormLayout(advanced_group)
        
        self.bandwidth_spin = QDoubleSpinBox()
        self.bandwidth_spin.setRange(0, 100)
        self.bandwidth_spin.setSuffix(" MHz")
        self.bandwidth_spin.setDecimals(3)
        self.bandwidth_spin.valueChanged.connect(self.settings_changed.emit)
        advanced_layout.addRow("Bandwidth:", self.bandwidth_spin)
        
        self.ppm_correction_spin = QSpinBox()
        self.ppm_correction_spin.setRange(-100, 100)
        self.ppm_correction_spin.setSuffix(" ppm")
        self.ppm_correction_spin.valueChanged.connect(self.settings_changed.emit)
        advanced_layout.addRow("PPM Correction:", self.ppm_correction_spin)
        
        layout.addWidget(advanced_group)
        
        layout.addStretch()
    
    def load_settings(self, settings: DeviceSettings):
        """Load settings into UI"""
        self.device_type_combo.setCurrentText(settings.device_type)
        self.device_id_edit.setText(settings.device_id)
        self.center_freq_spin.setValue(settings.center_frequency / 1e6)
        
        # Convert sample rate to combo text
        sample_rate_mhz = settings.sample_rate / 1e6
        if sample_rate_mhz < 1:
            sample_rate_text = f"{settings.sample_rate / 1e3:.0f} kHz"
        else:
            sample_rate_text = f"{sample_rate_mhz:.2f} MHz"
        
        index = self.sample_rate_combo.findText(sample_rate_text, Qt.MatchContains)
        if index >= 0:
            self.sample_rate_combo.setCurrentIndex(index)
        
        self.auto_gain_check.setChecked(settings.auto_gain)
        self.gain_spin.setValue(settings.gain)
        self.bandwidth_spin.setValue(settings.bandwidth / 1e6)
        self.ppm_correction_spin.setValue(settings.ppm_correction)
    
    def get_settings(self) -> DeviceSettings:
        """Get current settings from UI"""
        # Parse sample rate from combo text
        sample_rate_text = self.sample_rate_combo.currentText()
        if 'kHz' in sample_rate_text:
            sample_rate = float(sample_rate_text.split()[0]) * 1e3
        else:
            sample_rate = float(sample_rate_text.split()[0]) * 1e6
        
        return DeviceSettings(
            device_type=self.device_type_combo.currentText(),
            device_id=self.device_id_edit.text(),
            center_frequency=int(self.center_freq_spin.value() * 1e6),
            sample_rate=int(sample_rate),
            auto_gain=self.auto_gain_check.isChecked(),
            gain=self.gain_spin.value(),
            bandwidth=int(self.bandwidth_spin.value() * 1e6),
            ppm_correction=self.ppm_correction_spin.value()
        )

class ProcessingSettingsTab(QWidget):
    """Processing configuration tab"""
    
    settings_changed = Signal()
    
    def __init__(self, settings: ProcessingSettings):
        super().__init__()
        self.settings = settings
        self.setup_ui()
    
    def setup_ui(self):
        """Setup processing settings UI"""
        layout = QVBoxLayout(self)
        
        # FFT settings group
        fft_group = QGroupBox("FFT Settings")
        fft_layout = QFormLayout(fft_group)
        
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(['512', '1024', '2048', '4096', '8192', '16384'])
        self.fft_size_combo.currentTextChanged.connect(self.settings_changed.emit)
        fft_layout.addRow("FFT Size:", self.fft_size_combo)
        
        self.window_type_combo = QComboBox()
        self.window_type_combo.addItems(['hann', 'blackman', 'hamming', 'bartlett', 'rectangular'])
        self.window_type_combo.currentTextChanged.connect(self.settings_changed.emit)
        fft_layout.addRow("Window Type:", self.window_type_combo)
        
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 90)
        self.overlap_spin.setSuffix(" %")
        self.overlap_spin.valueChanged.connect(self.settings_changed.emit)
        fft_layout.addRow("Overlap:", self.overlap_spin)
        
        layout.addWidget(fft_group)
        
        # Averaging settings group
        avg_group = QGroupBox("Averaging Settings")
        avg_layout = QFormLayout(avg_group)
        
        self.averaging_check = QCheckBox("Enable Averaging")
        self.averaging_check.toggled.connect(self.settings_changed.emit)
        avg_layout.addRow(self.averaging_check)
        
        self.avg_count_spin = QSpinBox()
        self.avg_count_spin.setRange(2, 1000)
        self.avg_count_spin.valueChanged.connect(self.settings_changed.emit)
        avg_layout.addRow("Average Count:", self.avg_count_spin)
        
        self.avg_type_combo = QComboBox()
        self.avg_type_combo.addItems(['linear', 'exponential', 'peak_hold'])
        self.avg_type_combo.currentTextChanged.connect(self.settings_changed.emit)
        avg_layout.addRow("Average Type:", self.avg_type_combo)
        
        layout.addWidget(avg_group)
        
        # Processing settings group
        proc_group = QGroupBox("Processing Settings")
        proc_layout = QFormLayout(proc_group)
        
        self.update_rate_spin = QSpinBox()
        self.update_rate_spin.setRange(1, 60)
        self.update_rate_spin.setSuffix(" Hz")
        self.update_rate_spin.valueChanged.connect(self.settings_changed.emit)
        proc_layout.addRow("Update Rate:", self.update_rate_spin)
        
        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setRange(1, 100)
        self.buffer_size_spin.setSuffix(" MB")
        self.buffer_size_spin.valueChanged.connect(self.settings_changed.emit)
        proc_layout.addRow("Buffer Size:", self.buffer_size_spin)
        
        layout.addWidget(proc_group)
        
        layout.addStretch()
    
    def load_settings(self, settings: ProcessingSettings):
        """Load settings into UI"""
        self.fft_size_combo.setCurrentText(str(settings.fft_size))
        self.window_type_combo.setCurrentText(settings.window_type)
        self.overlap_spin.setValue(int(settings.overlap * 100))
        self.averaging_check.setChecked(settings.averaging_enabled)
        self.avg_count_spin.setValue(settings.averaging_count)
        self.avg_type_combo.setCurrentText(settings.averaging_type)
        self.update_rate_spin.setValue(settings.update_rate)
        self.buffer_size_spin.setValue(settings.buffer_size_mb)
    
    def get_settings(self) -> ProcessingSettings:
        """Get current settings from UI"""
        return ProcessingSettings(
            fft_size=int(self.fft_size_combo.currentText()),
            window_type=self.window_type_combo.currentText(),
            overlap=self.overlap_spin.value() / 100.0,
            averaging_enabled=self.averaging_check.isChecked(),
            averaging_count=self.avg_count_spin.value(),
            averaging_type=self.avg_type_combo.currentText(),
            update_rate=self.update_rate_spin.value(),
            buffer_size_mb=self.buffer_size_spin.value()
        )

class DisplaySettingsTab(QWidget):
    """Display configuration tab"""
    
    settings_changed = Signal()
    
    def __init__(self, settings: DisplaySettings):
        super().__init__()
        self.settings = settings
        self.setup_ui()
    
    def setup_ui(self):
        """Setup display settings UI"""
        layout = QVBoxLayout(self)
        
        # Spectrum display group
        spectrum_group = QGroupBox("Spectrum Display")
        spectrum_layout = QFormLayout(spectrum_group)
        
        self.ref_level_spin = QDoubleSpinBox()
        self.ref_level_spin.setRange(-200, 100)
        self.ref_level_spin.setSuffix(" dB")
        self.ref_level_spin.setDecimals(1)
        self.ref_level_spin.valueChanged.connect(self.settings_changed.emit)
        spectrum_layout.addRow("Reference Level:", self.ref_level_spin)
        
        self.dynamic_range_spin = QSpinBox()
        self.dynamic_range_spin.setRange(20, 200)
        self.dynamic_range_spin.setSuffix(" dB")
        self.dynamic_range_spin.valueChanged.connect(self.settings_changed.emit)
        spectrum_layout.addRow("Dynamic Range:", self.dynamic_range_spin)
        
        self.peak_detection_check = QCheckBox("Enable Peak Detection")
        self.peak_detection_check.toggled.connect(self.settings_changed.emit)
        spectrum_layout.addRow(self.peak_detection_check)
        
        layout.addWidget(spectrum_group)
        
        # Waterfall display group
        waterfall_group = QGroupBox("Waterfall Display")
        waterfall_layout = QFormLayout(waterfall_group)
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot'])
        self.colormap_combo.currentTextChanged.connect(self.settings_changed.emit)
        waterfall_layout.addRow("Colormap:", self.colormap_combo)
        
        self.waterfall_history_spin = QSpinBox()
        self.waterfall_history_spin.setRange(100, 10000)
        self.waterfall_history_spin.setSuffix(" lines")
        self.waterfall_history_spin.valueChanged.connect(self.settings_changed.emit)
        waterfall_layout.addRow("History Size:", self.waterfall_history_spin)
        
        layout.addWidget(waterfall_group)
        
        # Theme and appearance group
        theme_group = QGroupBox("Theme and Appearance")
        theme_layout = QFormLayout(theme_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(['dark', 'light', 'auto'])
        self.theme_combo.currentTextChanged.connect(self.settings_changed.emit)
        theme_layout.addRow("Theme:", self.theme_combo)
        
        # Font selection
        font_layout = QHBoxLayout()
        self.font_label = QLabel("Default Font")
        self.font_button = QPushButton("Change Font...")
        self.font_button.clicked.connect(self.select_font)
        font_layout.addWidget(self.font_label)
        font_layout.addWidget(self.font_button)
        theme_layout.addRow("Font:", font_layout)
        
        layout.addWidget(theme_group)
        
        layout.addStretch()
    
    def select_font(self):
        """Open font selection dialog"""
        current_font = QFont()
        font, ok = QFontDialog.getFont(current_font, self)
        if ok:
            self.font_label.setText(f"{font.family()}, {font.pointSize()}pt")
            self.settings_changed.emit()
    
    def load_settings(self, settings: DisplaySettings):
        """Load settings into UI"""
        self.ref_level_spin.setValue(settings.reference_level)
        self.dynamic_range_spin.setValue(settings.dynamic_range)
        self.peak_detection_check.setChecked(settings.peak_detection_enabled)
        self.colormap_combo.setCurrentText(settings.colormap)
        self.waterfall_history_spin.setValue(settings.waterfall_history_size)
        self.theme_combo.setCurrentText(settings.theme)
    
    def get_settings(self) -> DisplaySettings:
        """Get current settings from UI"""
        return DisplaySettings(
            reference_level=self.ref_level_spin.value(),
            dynamic_range=self.dynamic_range_spin.value(),
            peak_detection_enabled=self.peak_detection_check.isChecked(),
            colormap=self.colormap_combo.currentText(),
            waterfall_history_size=self.waterfall_history_spin.value(),
            theme=self.theme_combo.currentText()
        )

class AdvancedSettingsTab(QWidget):
    """Advanced configuration tab"""
    
    settings_changed = Signal()
    
    def __init__(self, settings: AdvancedSettings):
        super().__init__()
        self.settings = settings
        self.setup_ui()
    
    def setup_ui(self):
        """Setup advanced settings UI"""
        layout = QVBoxLayout(self)
        
        # Performance group
        perf_group = QGroupBox("Performance Settings")
        perf_layout = QFormLayout(perf_group)
        
        self.multithreading_check = QCheckBox("Enable Multithreading")
        self.multithreading_check.toggled.connect(self.settings_changed.emit)
        perf_layout.addRow(self.multithreading_check)
        
        self.num_threads_spin = QSpinBox()
        self.num_threads_spin.setRange(1, 16)
        self.num_threads_spin.valueChanged.connect(self.settings_changed.emit)
        perf_layout.addRow("Number of Threads:", self.num_threads_spin)
        
        self.gpu_acceleration_check = QCheckBox("Enable GPU Acceleration")
        self.gpu_acceleration_check.toggled.connect(self.settings_changed.emit)
        perf_layout.addRow(self.gpu_acceleration_check)
        
        layout.addWidget(perf_group)
        
        # Logging group
        logging_group = QGroupBox("Logging Settings")
        logging_layout = QFormLayout(logging_group)
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        self.log_level_combo.currentTextChanged.connect(self.settings_changed.emit)
        logging_layout.addRow("Log Level:", self.log_level_combo)
        
        self.log_to_file_check = QCheckBox("Log to File")
        self.log_to_file_check.toggled.connect(self.settings_changed.emit)
        logging_layout.addRow(self.log_to_file_check)
        
        layout.addWidget(logging_group)
        
        # Export/Import group
        export_group = QGroupBox("Data Export/Import")
        export_layout = QFormLayout(export_group)
        
        self.default_export_format_combo = QComboBox()
        self.default_export_format_combo.addItems(['csv', 'json', 'mat', 'h5'])
        self.default_export_format_combo.currentTextChanged.connect(self.settings_changed.emit)
        export_layout.addRow("Default Export Format:", self.default_export_format_combo)
        
        self.auto_save_check = QCheckBox("Auto-save Data")
        self.auto_save_check.toggled.connect(self.settings_changed.emit)
        export_layout.addRow(self.auto_save_check)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
    
    def load_settings(self, settings: AdvancedSettings):
        """Load settings into UI"""
        self.multithreading_check.setChecked(settings.enable_multithreading)
        self.num_threads_spin.setValue(settings.num_threads)
        self.gpu_acceleration_check.setChecked(settings.enable_gpu_acceleration)
        self.log_level_combo.setCurrentText(settings.log_level)
        self.log_to_file_check.setChecked(settings.log_to_file)
        self.default_export_format_combo.setCurrentText(settings.default_export_format)
        self.auto_save_check.setChecked(settings.auto_save_data)
    
    def get_settings(self) -> AdvancedSettings:
        """Get current settings from UI"""
        return AdvancedSettings(
            enable_multithreading=self.multithreading_check.isChecked(),
            num_threads=self.num_threads_spin.value(),
            enable_gpu_acceleration=self.gpu_acceleration_check.isChecked(),
            log_level=self.log_level_combo.currentText(),
            log_to_file=self.log_to_file_check.isChecked(),
            default_export_format=self.default_export_format_combo.currentText(),
            auto_save_data=self.auto_save_check.isChecked()
        )