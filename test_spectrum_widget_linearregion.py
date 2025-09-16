#!/usr/bin/env python3
"""
Test script for SpectrumWidget with LinearRegionItem functionality
Tests the new frequency range selection and center button feature.
"""

import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import QTimer

# Add the project directory to path for imports
sys.path.append('.')

from rf_spectrum_analyzer.config.settings import Settings
from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget


class TestMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectrumWidget LinearRegion Test")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create settings
        self.settings = Settings()
        self.settings.sdr.center_frequency = 1545e6  # 1545 MHz
        self.settings.sdr.sample_rate = 10e6  # 10 MHz
        
        # Create info label
        self.info_label = QLabel("Testing SpectrumWidget with LinearRegionItem")
        layout.addWidget(self.info_label)
        
        # Create spectrum widget
        self.spectrum_widget = SpectrumWidget(self.settings)
        layout.addWidget(self.spectrum_widget)
        
        # Connect frequency range signal
        self.spectrum_widget.frequency_range_selected.connect(self.on_frequency_range_changed)
        
        # Create test data timer
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.update_test_data)
        self.data_timer.start(100)  # Update every 100ms
        
        # Enable frequency markers initially
        self.spectrum_widget.set_frequency_markers_enabled(True)
        
        print("Test Instructions:")
        print("1. The spectrum should show with a green LinearRegionItem")
        print("2. You can drag the region boundaries to change frequency range")
        print("3. Click the 'Center' button to move region to center frequency")
        print("4. Watch the frequency range updates in the status")
    
    def update_test_data(self):
        """Generate synthetic spectrum data for testing."""
        # Generate frequency axis
        sample_rate = self.settings.sdr.sample_rate
        center_freq = self.settings.sdr.center_frequency
        num_points = 1024
        
        # Create synthetic spectrum with some peaks
        freqs = np.linspace(-sample_rate/2, sample_rate/2, num_points)
        freqs_abs = freqs + center_freq
        
        # Generate base noise
        spectrum = -80 + 10 * np.random.randn(num_points)
        
        # Add some synthetic signals
        for i, peak_freq in enumerate([1543e6, 1545e6, 1547e6]):
            if freqs_abs[0] <= peak_freq <= freqs_abs[-1]:
                # Find closest frequency bin
                freq_idx = np.argmin(np.abs(freqs_abs - peak_freq))
                # Add signal with some width
                width = 20
                start_idx = max(0, freq_idx - width)
                end_idx = min(num_points, freq_idx + width)
                
                # Create signal envelope
                signal_indices = np.arange(start_idx, end_idx)
                signal_envelope = np.exp(-0.1 * (signal_indices - freq_idx)**2)
                spectrum[start_idx:end_idx] += (-40 + 10*i) * signal_envelope
        
        # Update spectrum widget
        self.spectrum_widget.update_data(spectrum)
    
    def on_frequency_range_changed(self, f1, f2):
        """Handle frequency range changes."""
        f1_mhz = f1 / 1e6
        f2_mhz = f2 / 1e6
        bandwidth_mhz = (f2 - f1) / 1e6
        
        info_text = f"Frequency Range: {f1_mhz:.3f} - {f2_mhz:.3f} MHz (BW: {bandwidth_mhz:.3f} MHz)"
        self.info_label.setText(info_text)
        print(info_text)


def main():
    """Main test function."""
    app = QApplication(sys.argv)
    
    # Create and show test window
    window = TestMainWindow()
    window.show()
    
    print("Starting SpectrumWidget LinearRegion Test...")
    print("Close the window to exit.")
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()