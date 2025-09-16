"""
GUI Demo Script - Test Signal Analysis Integration
This script demonstrates the GUI integration of signal analysis features.
"""

import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QHBoxLayout
from PySide6.QtCore import QTimer
import pyqtgraph as pg

# Add project root to path
sys.path.append('e:/SingalAnalys-Pro')

from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer

class SignalAnalysisDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Signal Analysis Demo")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Status label
        self.status_label = QLabel("Ready to test signal analysis...")
        layout.addWidget(self.status_label)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.bpsk_btn = QPushButton("Generate BPSK Signal")
        self.bpsk_btn.clicked.connect(self.generate_bpsk)
        control_layout.addWidget(self.bpsk_btn)
        
        self.qpsk_btn = QPushButton("Generate QPSK Signal")
        self.qpsk_btn.clicked.connect(self.generate_qpsk)
        control_layout.addWidget(self.qpsk_btn)
        
        self.fsk_btn = QPushButton("Generate FSK Signal")
        self.fsk_btn.clicked.connect(self.generate_fsk)
        control_layout.addWidget(self.fsk_btn)
        
        self.noise_btn = QPushButton("Generate Noise")
        self.noise_btn.clicked.connect(self.generate_noise)
        control_layout.addWidget(self.noise_btn)
        
        layout.addLayout(control_layout)
        
        # Create spectrum widget
        self.spectrum_widget = SpectrumWidget()
        layout.addWidget(self.spectrum_widget)
        
        # Connect signal analysis result
        self.spectrum_widget.signal_analysis_requested.connect(self.handle_analysis_request)
        
        # Initialize with some data
        self.sample_rate = 1e6
        self.center_freq = 100e6
        self.spectrum_widget.set_frequency_range(self.center_freq - self.sample_rate/2, 
                                                self.center_freq + self.sample_rate/2)
        
        # Generate initial noise signal
        self.generate_noise()
        
    def generate_bpsk(self):
        """Generate BPSK signal for testing."""
        self.status_label.setText("Generating BPSK signal...")
        
        duration = 0.1
        symbol_rate = 1000
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        # Generate random bits
        num_symbols = int(duration * symbol_rate)
        data_bits = np.random.randint(0, 2, num_symbols)
        symbols = 2 * data_bits - 1  # Convert to +1/-1
        
        # Upsample symbols
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        upsampled = np.repeat(symbols, samples_per_symbol)[:num_samples]
        
        # Add carrier
        carrier_freq = 5000
        iq_data = upsampled * np.exp(1j * 2 * np.pi * carrier_freq * t)
        
        # Add noise
        noise_power = 0.1
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_power / 2)
        iq_data += noise
        
        # Generate spectrum
        fft_data = np.fft.fftshift(np.fft.fft(iq_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(iq_data), 1/self.sample_rate)) + self.center_freq
        spectrum = 20 * np.log10(np.abs(fft_data) + 1e-12)
        
        self.spectrum_widget.update_spectrum(freqs, spectrum)
        self.current_iq_data = iq_data
        self.status_label.setText("BPSK signal generated. Select frequency range and click 'Analyze Signal'")
        
    def generate_qpsk(self):
        """Generate QPSK signal for testing."""
        self.status_label.setText("Generating QPSK signal...")
        
        duration = 0.1
        symbol_rate = 500
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        # Generate random symbols (0-3)
        num_symbols = int(duration * symbol_rate)
        data_symbols = np.random.randint(0, 4, num_symbols)
        
        # Map to QPSK constellation
        constellation_map = {
            0: 1 + 1j,
            1: -1 + 1j, 
            2: -1 - 1j,
            3: 1 - 1j
        }
        
        symbols = np.array([constellation_map[s] for s in data_symbols])
        
        # Upsample symbols
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        upsampled = np.repeat(symbols, samples_per_symbol)[:num_samples]
        
        # Add carrier
        carrier_freq = 10000
        iq_data = upsampled * np.exp(1j * 2 * np.pi * carrier_freq * t)
        
        # Add noise
        noise_power = 0.1
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_power / 2)
        iq_data += noise
        
        # Generate spectrum
        fft_data = np.fft.fftshift(np.fft.fft(iq_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(iq_data), 1/self.sample_rate)) + self.center_freq
        spectrum = 20 * np.log10(np.abs(fft_data) + 1e-12)
        
        self.spectrum_widget.update_spectrum(freqs, spectrum)
        self.current_iq_data = iq_data
        self.status_label.setText("QPSK signal generated. Select frequency range and click 'Analyze Signal'")
        
    def generate_fsk(self):
        """Generate FSK signal for testing."""
        self.status_label.setText("Generating FSK signal...")
        
        duration = 0.1
        symbol_rate = 1200
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        # Generate random bits
        num_symbols = int(duration * symbol_rate)
        data_bits = np.random.randint(0, 2, num_symbols)
        
        # FSK frequencies
        freq_0 = 3000  # Frequency for bit 0
        freq_1 = 7000  # Frequency for bit 1
        
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        iq_data = np.zeros(num_samples, dtype=complex)
        
        for i, bit in enumerate(data_bits):
            start_idx = i * samples_per_symbol
            end_idx = min(start_idx + samples_per_symbol, num_samples)
            
            if end_idx <= start_idx:
                break
                
            freq = freq_1 if bit else freq_0
            t_symbol = t[start_idx:end_idx]
            iq_data[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t_symbol)
        
        # Add noise
        noise_power = 0.1
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_power / 2)
        iq_data += noise
        
        # Generate spectrum
        fft_data = np.fft.fftshift(np.fft.fft(iq_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(iq_data), 1/self.sample_rate)) + self.center_freq
        spectrum = 20 * np.log10(np.abs(fft_data) + 1e-12)
        
        self.spectrum_widget.update_spectrum(freqs, spectrum)
        self.current_iq_data = iq_data
        self.status_label.setText("FSK signal generated. Select frequency range and click 'Analyze Signal'")
        
    def generate_noise(self):
        """Generate noise signal for testing."""
        self.status_label.setText("Generating noise signal...")
        
        num_samples = 100000
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.5
        
        # Generate spectrum
        fft_data = np.fft.fftshift(np.fft.fft(noise))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(noise), 1/self.sample_rate)) + self.center_freq
        spectrum = 20 * np.log10(np.abs(fft_data) + 1e-12)
        
        self.spectrum_widget.update_spectrum(freqs, spectrum)
        self.current_iq_data = noise
        self.status_label.setText("Noise signal generated. Select frequency range and click 'Analyze Signal'")
        
    def handle_analysis_request(self, start_freq, end_freq):
        """Handle signal analysis request from spectrum widget."""
        try:
            self.status_label.setText(f"Analyzing signal from {start_freq/1e6:.2f} MHz to {end_freq/1e6:.2f} MHz...")
            
            # Create analyzer
            analyzer = SignalAnalyzer(self.sample_rate)
            
            # Use current IQ data for analysis
            if hasattr(self, 'current_iq_data'):
                results = analyzer.analyze_signal_comprehensive(self.current_iq_data, 
                                                              (start_freq + end_freq) / 2, 
                                                              end_freq - start_freq)
                
                # Display results
                mod_type = results['modulation']['type']
                confidence = results['modulation']['confidence']
                demod_success = results['demodulation']['success']
                
                status_text = f"Analysis Complete: {mod_type} (Conf: {confidence:.2f})"
                if demod_success:
                    status_text += f" | Demod: SUCCESS"
                    if results['demodulation']['snr']:
                        status_text += f" | SNR: {results['demodulation']['snr']:.1f} dB"
                else:
                    status_text += f" | Demod: FAILED"
                
                if results.get('coding') and results['coding']['coding_type'] != 'Unknown':
                    coding_type = results['coding']['coding_type']
                    coding_conf = results['coding']['confidence']
                    status_text += f" | Coding: {coding_type} ({coding_conf:.2f})"
                
                self.status_label.setText(status_text)
                
            else:
                self.status_label.setText("No signal data available for analysis")
                
        except Exception as e:
            self.status_label.setText(f"Analysis failed: {str(e)}")
            print(f"Analysis error: {e}")

def main():
    app = QApplication(sys.argv)
    
    # Set up PyQtGraph
    pg.setConfigOptions(antialias=True)
    
    demo = SignalAnalysisDemo()
    demo.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()