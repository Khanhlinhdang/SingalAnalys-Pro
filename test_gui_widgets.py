#!/usr/bin/env python3
"""
Test GUI Widgets - Constellation and Bitstream
Kiểm tra hiển thị constellation và bitstream với dữ liệu thực từ signal processor.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import QTimer

from rf_spectrum_analyzer.gui.constellation_widget import ConstellationWidget
from rf_spectrum_analyzer.gui.bitstream_widget import BitstreamWidget
from rf_spectrum_analyzer.core.signal_processor import SignalProcessor
from rf_spectrum_analyzer.config.settings import Settings

# SDR and scikit-dsp-comm imports
try:
    import sdr
    SDR_AVAILABLE = True
except ImportError:
    SDR_AVAILABLE = False

try:
    import sk_dsp_comm.digitalcom as dc
    SCIKIT_DSP_AVAILABLE = True
except ImportError:
    SCIKIT_DSP_AVAILABLE = False


class TestWindow(QMainWindow):
    """Test window for GUI widgets."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Spectrum Analyzer - GUI Widget Test")
        self.setGeometry(100, 100, 1400, 800)
        
        # Setup signal processor
        self.settings = Settings()
        self.signal_processor = SignalProcessor(self.settings)
        
        # Setup UI
        self.setup_ui()
        
        # Test data generator
        self.test_data_index = 0
        self.test_modulations = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
        
        # Timer for simulating data updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.generate_test_data)
        self.update_timer.start(2000)  # Update every 2 seconds
        
        print("🚀 GUI Widget Test Started")
        print("Testing constellation and bitstream display with real signal processing...")
        
    def setup_ui(self):
        """Setup the test UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("RF Spectrum Analyzer - Widget Test")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #2E8B57; margin: 10px;")
        layout.addWidget(title_label)
        
        # Status
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("font-size: 10pt; color: #666666; margin: 5px;")
        layout.addWidget(self.status_label)
        
        # Main content
        content_layout = QHBoxLayout()
        
        # Constellation widget
        self.constellation_widget = ConstellationWidget()
        content_layout.addWidget(self.constellation_widget)
        
        # Bitstream widget
        self.bitstream_widget = BitstreamWidget()
        content_layout.addWidget(self.bitstream_widget)
        
        layout.addLayout(content_layout)
        
    def generate_test_data(self):
        """Generate test data and process through signal processor."""
        try:
            # Cycle through modulations
            modulation_type = self.test_modulations[self.test_data_index % len(self.test_modulations)]
            self.test_data_index += 1
            
            print(f"\n🧪 Testing {modulation_type} modulation...")
            self.status_label.setText(f"Testing {modulation_type} modulation...")
            
            # Generate test signal
            test_signal, original_bits = self.generate_modulated_signal(modulation_type)
            
            if test_signal is not None and len(test_signal) > 0:
                # Process through complete signal chain
                result = self.signal_processor.process_complete_chain(test_signal)
                
                # Update constellation widget
                self.update_constellation_display(test_signal, result, modulation_type)
                
                # Update bitstream widget
                self.update_bitstream_display(result, original_bits)
                
                # Print results
                self.print_processing_results(result, modulation_type, len(original_bits))
                
            else:
                print(f"❌ Failed to generate {modulation_type} signal")
                
        except Exception as e:
            print(f"❌ Error generating test data: {e}")
            self.status_label.setText(f"Error: {e}")
    
    def generate_modulated_signal(self, modulation_type: str) -> tuple:
        """Generate modulated test signal."""
        try:
            # Generate random data
            data_bits = np.random.randint(0, 2, 512)
            
            # Sample rate and parameters
            sample_rate = self.settings.sdr.sample_rate
            symbol_rate = 100000  # 100 kHz symbol rate
            samples_per_symbol = int(sample_rate / symbol_rate)
            
            if modulation_type == "BPSK" and SDR_AVAILABLE:
                # BPSK modulation
                modulator = sdr.PSK(2)
                symbols = modulator.modulate(data_bits)
                signal_samples = sdr.upsample(symbols, samples_per_symbol)
                
            elif modulation_type == "QPSK" and SDR_AVAILABLE:
                # QPSK modulation
                modulator = sdr.PSK(4)
                # Group bits into symbols (2 bits per symbol for QPSK)
                symbol_data = []
                for i in range(0, len(data_bits), 2):
                    if i+1 < len(data_bits):
                        symbol_value = data_bits[i] * 2 + data_bits[i+1]
                        symbol_data.append(symbol_value)
                symbols = modulator.modulate(np.array(symbol_data))
                signal_samples = sdr.upsample(symbols, samples_per_symbol)
                
            elif modulation_type == "8PSK" and SDR_AVAILABLE:
                # 8PSK modulation
                modulator = sdr.PSK(8)
                # Group bits into symbols (3 bits per symbol)
                symbol_data = []
                for i in range(0, len(data_bits), 3):
                    if i+2 < len(data_bits):
                        symbol_value = data_bits[i] * 4 + data_bits[i+1] * 2 + data_bits[i+2]
                        symbol_data.append(symbol_value)
                symbols = modulator.modulate(np.array(symbol_data))
                signal_samples = sdr.upsample(symbols, samples_per_symbol)
                
            elif modulation_type == "16QAM":
                # 16QAM using manual constellation
                constellation_points = np.array([
                    -3-3j, -3-1j, -3+1j, -3+3j,
                    -1-3j, -1-1j, -1+1j, -1+3j,
                     1-3j,  1-1j,  1+1j,  1+3j,
                     3-3j,  3-1j,  3+1j,  3+3j
                ]) / np.sqrt(10)  # Normalize
                
                # Group bits into symbols (4 bits per symbol)
                symbols = []
                for i in range(0, len(data_bits), 4):
                    if i+3 < len(data_bits):
                        symbol_index = (data_bits[i] * 8 + data_bits[i+1] * 4 + 
                                      data_bits[i+2] * 2 + data_bits[i+3])
                        symbols.append(constellation_points[symbol_index])
                
                symbols = np.array(symbols)
                signal_samples = np.repeat(symbols, samples_per_symbol)
                
            elif modulation_type == "64QAM":
                # 64QAM using manual constellation
                constellation_points = []
                for i in range(-7, 8, 2):
                    for j in range(-7, 8, 2):
                        constellation_points.append(i + 1j * j)
                constellation_points = np.array(constellation_points) / np.sqrt(42)
                
                # Group bits into symbols (6 bits per symbol)
                symbols = []
                for i in range(0, len(data_bits), 6):
                    if i+5 < len(data_bits):
                        symbol_index = (data_bits[i] * 32 + data_bits[i+1] * 16 + 
                                      data_bits[i+2] * 8 + data_bits[i+3] * 4 +
                                      data_bits[i+4] * 2 + data_bits[i+5])
                        if symbol_index < len(constellation_points):
                            symbols.append(constellation_points[symbol_index])
                
                symbols = np.array(symbols)
                signal_samples = np.repeat(symbols, samples_per_symbol)
                
            else:
                # Fallback: simple BPSK
                symbols = 2 * data_bits - 1  # Map 0,1 to -1,1
                signal_samples = np.repeat(symbols, samples_per_symbol)
            
            # Add noise for realistic testing
            if SDR_AVAILABLE:
                snr_db = 20  # 20 dB SNR
                signal_samples = sdr.awgn(signal_samples, snr=snr_db, seed=42)
            else:
                noise_power = 0.1
                noise = (np.random.normal(0, np.sqrt(noise_power/2), len(signal_samples)) + 
                        1j * np.random.normal(0, np.sqrt(noise_power/2), len(signal_samples)))
                signal_samples = signal_samples + noise
                
            return signal_samples.astype(np.complex64), data_bits
            
        except Exception as e:
            print(f"❌ Error generating {modulation_type} signal: {e}")
            return None, np.array([])
    
    def update_constellation_display(self, iq_data, processing_result, modulation_type):
        """Update constellation widget with new data."""
        try:
            # Extract symbols if available
            symbols = None
            if processing_result.get("success", False):
                demod_result = processing_result.get("demodulation", {})
                symbols = demod_result.get("symbols", None)
                constellation_points = demod_result.get("constellation_points", None)
                if symbols is None:
                    symbols = constellation_points
                    
            # Create modulation info for display
            modulation_info = {
                "type": modulation_type,
                "evm": processing_result.get("demodulation", {}).get("evm", 0),
                "snr_estimate": processing_result.get("demodulation", {}).get("snr_db", 0)
            }
            
            # Update constellation
            self.constellation_widget.update_constellation(iq_data, symbols, modulation_info)
            
        except Exception as e:
            print(f"❌ Error updating constellation: {e}")
            
    def update_bitstream_display(self, processing_result, original_bits):
        """Update bitstream widget with new data."""
        try:
            # Get final processed bits
            final_bits = processing_result.get("final_data", np.array([]))
            
            if len(final_bits) > 0:
                # Convert to binary if needed
                if final_bits.dtype != bool and final_bits.dtype != int:
                    binary_bits = (final_bits > np.mean(final_bits)).astype(int)
                else:
                    binary_bits = final_bits.astype(int)
                
                # Add bits to display
                self.bitstream_widget.add_bits(binary_bits)
                
                print(f"   📊 Bitstream: {len(binary_bits)} bits added to display")
            else:
                print("   ⚠️  No bits to display")
                
        except Exception as e:
            print(f"❌ Error updating bitstream: {e}")
            
    def print_processing_results(self, result, modulation_type, original_bits_count):
        """Print processing results."""
        if result.get("success", False):
            mod_analysis = result.get("modulation_analysis", {})
            demod_result = result.get("demodulation", {})
            encoding_analysis = result.get("encoding_analysis", {})
            final_bits = result.get("final_data", np.array([]))
            
            detected_mod = mod_analysis.get("type", "Unknown")
            confidence = mod_analysis.get("confidence", 0)
            demod_bits_count = len(demod_result.get("demodulated_data", []))
            final_bits_count = len(final_bits)
            evm = demod_result.get("evm", 0)
            
            print(f"   ✅ Detected: {detected_mod} (conf: {confidence:.2f})")
            print(f"   📈 Original: {original_bits_count} bits → Demod: {demod_bits_count} bits → Final: {final_bits_count} bits")
            print(f"   📊 EVM: {evm:.1f}%")
            
            if encoding_analysis.get("type", "None") != "None":
                print(f"   🔐 Encoding: {encoding_analysis['type']} (conf: {encoding_analysis['confidence']:.2f})")
        else:
            print(f"   ❌ Processing failed: {result.get('error', 'Unknown error')}")


def main():
    """Main test function."""
    print("🔬 RF SPECTRUM ANALYZER - GUI WIDGET TEST")
    print("=" * 50)
    
    # Check libraries
    print(f"✓ SDR library available: {SDR_AVAILABLE}")
    print(f"✓ Scikit-DSP-Comm available: {SCIKIT_DSP_AVAILABLE}")
    
    app = QApplication(sys.argv)
    
    # Create and show test window
    window = TestWindow()
    window.show()
    
    print("\n🎯 Starting GUI test...")
    print("   - Constellation widget will show I/Q data and symbols")
    print("   - Bitstream widget will show decoded bits")
    print("   - Testing will cycle through different modulations")
    
    # Run application
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")


if __name__ == "__main__":
    main()