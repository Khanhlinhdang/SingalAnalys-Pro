#!/usr/bin/env python3
"""
Demo application showing SDRConnect integration features
Simple RF Spectrum Analyzer demo with SpyServer and enhanced analysis
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QLabel, QPushButton, QGroupBox, QTextEdit,
    QComboBox, QSpinBox, QLineEdit, QFormLayout
)
from PySide6.QtCore import QTimer, Signal
import pyqtgraph as pg

from rf_spectrum_analyzer.config.settings import Settings
from rf_spectrum_analyzer.core.signal_processor import SignalProcessor
from rf_spectrum_analyzer.backends.spyserver_backend import SpyServerBackend


class SDRConnectDemo(QMainWindow):
    """Demo window for SDRConnect integration."""
    
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.signal_processor = SignalProcessor(self.settings)
        self.spyserver_backend = None
        
        self.init_ui()
        self.setup_demo_data()
        
        # Timer for demo updates
        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(self.update_demo)
        self.demo_timer.start(1000)  # Update every second
    
    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("SDRConnect Integration Demo")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        self.create_controls_panel(layout)
        
        # Right panel - Analysis results
        self.create_analysis_panel(layout)
    
    def create_controls_panel(self, parent_layout):
        """Create controls panel."""
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(300)
        controls_layout = QVBoxLayout(controls_widget)
        
        # Title
        title = QLabel("SDRConnect Demo")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        controls_layout.addWidget(title)
        
        # SpyServer group
        spyserver_group = QGroupBox("SpyServer Configuration")
        spyserver_layout = QFormLayout(spyserver_group)
        
        self.host_input = QLineEdit("localhost")
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(5555)
        
        spyserver_layout.addRow("Host:", self.host_input)
        spyserver_layout.addRow("Port:", self.port_input)
        
        self.test_connection_btn = QPushButton("Test Connection")
        self.test_connection_btn.clicked.connect(self.test_spyserver_connection)
        spyserver_layout.addRow(self.test_connection_btn)
        
        controls_layout.addWidget(spyserver_group)
        
        # Analysis group
        analysis_group = QGroupBox("Enhanced Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.enhanced_analysis_btn = QPushButton("Run Enhanced Analysis")
        self.enhanced_analysis_btn.clicked.connect(self.run_enhanced_analysis)
        analysis_layout.addWidget(self.enhanced_analysis_btn)
        
        self.basic_analysis_btn = QPushButton("Run Basic Analysis")
        self.basic_analysis_btn.clicked.connect(self.run_basic_analysis)
        analysis_layout.addWidget(self.basic_analysis_btn)
        
        controls_layout.addWidget(analysis_group)
        
        # Demo signal group
        demo_group = QGroupBox("Demo Signal")
        demo_layout = QFormLayout(demo_group)
        
        self.signal_type_combo = QComboBox()
        self.signal_type_combo.addItems(["Sine Wave", "QPSK", "Noise", "Complex Chirp"])
        self.signal_type_combo.currentTextChanged.connect(self.setup_demo_data)
        
        self.freq_input = QSpinBox()
        self.freq_input.setRange(100, 10000)
        self.freq_input.setValue(1000)
        self.freq_input.setSuffix(" Hz")
        self.freq_input.valueChanged.connect(self.setup_demo_data)
        
        demo_layout.addRow("Signal Type:", self.signal_type_combo)
        demo_layout.addRow("Frequency:", self.freq_input)
        
        controls_layout.addWidget(demo_group)
        
        controls_layout.addStretch()
        parent_layout.addWidget(controls_widget)
    
    def create_analysis_panel(self, parent_layout):
        """Create analysis results panel."""
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold; color: green; margin: 5px;")
        analysis_layout.addWidget(self.status_label)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget(title="Signal Analysis")
        self.plot_widget.setLabel('left', 'Magnitude (dB)')
        self.plot_widget.setLabel('bottom', 'Frequency (Hz)')
        analysis_layout.addWidget(self.plot_widget)
        
        # Results text
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setPlainText("Analysis results will appear here...")
        analysis_layout.addWidget(self.results_text)
        
        parent_layout.addWidget(analysis_widget)
    
    def setup_demo_data(self):
        """Setup demo signal data."""
        sample_rate = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        signal_type = self.signal_type_combo.currentText()
        frequency = self.freq_input.value()
        
        if signal_type == "Sine Wave":
            self.demo_signal = np.exp(1j * 2 * np.pi * frequency * t)
        elif signal_type == "QPSK":
            # Simple QPSK signal
            symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], len(t)//100)
            symbols_upsampled = np.repeat(symbols, 100)[:len(t)]
            self.demo_signal = symbols_upsampled * np.exp(1j * 2 * np.pi * frequency * t)
        elif signal_type == "Noise":
            self.demo_signal = 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        elif signal_type == "Complex Chirp":
            # Frequency sweep
            f_start, f_end = frequency, frequency * 2
            instantaneous_phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration))
            self.demo_signal = np.exp(1j * instantaneous_phase)
        
        # Add some noise
        noise_level = 0.05
        noise = noise_level * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        self.demo_signal += noise
        
        self.status_label.setText(f"Demo signal: {signal_type} @ {frequency} Hz")
    
    def test_spyserver_connection(self):
        """Test SpyServer connection."""
        try:
            self.test_connection_btn.setText("Testing...")
            self.test_connection_btn.setEnabled(False)
            
            # Update settings
            self.settings.sdr.spyserver_host = self.host_input.text()
            self.settings.sdr.spyserver_port = self.port_input.value()
            
            # Create backend
            self.spyserver_backend = SpyServerBackend(self.settings)
            
            # Test connection (will fail if no server, but that's OK for demo)
            device_info = self.spyserver_backend.get_device_info()
            
            self.status_label.setText(f"SpyServer config: {device_info['host']}:{device_info['port']}")
            self.test_connection_btn.setText("✓ Config OK")
            self.test_connection_btn.setStyleSheet("color: green; font-weight: bold;")
            
        except Exception as e:
            self.status_label.setText(f"SpyServer test failed: {str(e)[:50]}...")
            self.test_connection_btn.setText("✗ Failed")
            self.test_connection_btn.setStyleSheet("color: red; font-weight: bold;")
        finally:
            # Reset button after 2 seconds
            QTimer.singleShot(2000, self.reset_test_button)
    
    def reset_test_button(self):
        """Reset test button to default state."""
        self.test_connection_btn.setText("Test Connection")
        self.test_connection_btn.setStyleSheet("")
        self.test_connection_btn.setEnabled(True)
    
    def run_enhanced_analysis(self):
        """Run enhanced analysis on demo signal."""
        try:
            self.status_label.setText("Running enhanced analysis...")
            
            result = self.signal_processor.enhanced_analysis(self.demo_signal)
            
            if result['success']:
                # Update plot
                frequencies = np.array(result['frequency_axis'])
                power_spectrum = np.array(result['power_spectrum'])
                
                self.plot_widget.clear()
                self.plot_widget.plot(frequencies, power_spectrum, pen='b', name='Enhanced Analysis')
                
                # Update results text
                text = f"Enhanced Analysis Results:\n"
                text += f"Method: {result['analysis_method']}\n"
                text += f"Peak Frequency: {result['peak_frequency']:.1f} Hz\n"
                text += f"SNR Estimate: {result['snr_estimate']:.1f} dB\n"
                text += f"Bandwidth: {result['bandwidth']:.1f} Hz\n"
                text += f"Enhanced features available: {result['has_enhanced_data']}\n"
                
                if result['has_enhanced_data'] and 'enhanced_metrics' in result:
                    metrics = result['enhanced_metrics']
                    text += f"\nEnhanced Metrics:\n"
                    text += f"RMS Power: {metrics.get('rms_power', 'N/A')}\n"
                    text += f"Crest Factor: {metrics.get('crest_factor', 'N/A')}\n"
                    text += f"DC Offset I/Q: {metrics.get('dc_offset_i', 'N/A'):.4f}/{metrics.get('dc_offset_q', 'N/A'):.4f}\n"
                    text += f"Occupied BW: {metrics.get('occupied_bandwidth', 'N/A')} kHz\n"
                    text += f"Noise Floor: {metrics.get('noise_floor', 'N/A')} dB\n"
                
                self.results_text.setPlainText(text)
                self.status_label.setText("Enhanced analysis completed ✓")
            else:
                self.status_label.setText(f"Analysis failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.status_label.setText(f"Analysis error: {str(e)[:50]}...")
    
    def run_basic_analysis(self):
        """Run basic FFT analysis."""
        try:
            self.status_label.setText("Running basic analysis...")
            
            # Simple FFT analysis
            fft_data = np.fft.fftshift(np.fft.fft(self.demo_signal[:1024]))
            power_spectrum = 20 * np.log10(np.abs(fft_data) + 1e-12)
            frequencies = np.linspace(-24000, 24000, len(power_spectrum))
            
            # Update plot
            self.plot_widget.clear()
            self.plot_widget.plot(frequencies, power_spectrum, pen='r', name='Basic FFT')
            
            # Find peak
            peak_idx = np.argmax(power_spectrum)
            peak_freq = frequencies[peak_idx]
            peak_power = power_spectrum[peak_idx]
            
            # Update results
            text = f"Basic FFT Analysis Results:\n"
            text += f"Peak Frequency: {peak_freq:.1f} Hz\n"
            text += f"Peak Power: {peak_power:.1f} dB\n"
            text += f"FFT Size: {len(fft_data)}\n"
            text += f"Frequency Resolution: {frequencies[1] - frequencies[0]:.1f} Hz\n"
            
            self.results_text.setPlainText(text)
            self.status_label.setText("Basic analysis completed ✓")
            
        except Exception as e:
            self.status_label.setText(f"Basic analysis error: {str(e)[:50]}...")
    
    def update_demo(self):
        """Update demo display periodically."""
        # This could update real-time data if connected to actual SDR
        pass


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("SDRConnect Demo")
    app.setApplicationVersion("1.0")
    
    # Create and show demo window
    demo = SDRConnectDemo()
    demo.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()