#!/usr/bin/env python3
"""
Bitstream Demo for RF Spectrum Analyzer
Demonstrates the new bitstream dock functionality with simulated data.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PySide6.QtCore import QTimer, Qt

from rf_spectrum_analyzer.gui.bitstream_widget import BitstreamWidget


class BitstreamDemo(QMainWindow):
    """Demo application for bitstream widget functionality."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitstream Widget Demo - RF Spectrum Analyzer")
        self.setGeometry(100, 100, 900, 700)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add title
        title = QLabel("🔬 RF Spectrum Analyzer - Bitstream Display Demo")
        title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #14a085; padding: 10px;")
        layout.addWidget(title)
        
        # Add description
        desc = QLabel("""
        This demo shows the new bitstream visualization feature:
        • Real-time bit display with colored pixels (green=1, black=0)
        • Configurable layout (bits per row, pixel size)
        • Statistical analysis (entropy, bit distribution)
        • Export capabilities and buffer management
        • Pause/resume and auto-scroll functionality
        """)
        desc.setStyleSheet("color: #cccccc; padding: 5px;")
        layout.addWidget(desc)
        
        # Create bitstream widget
        self.bitstream_widget = BitstreamWidget()
        layout.addWidget(self.bitstream_widget)
        
        # Control buttons
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Demo data buttons
        self.start_demo_btn = QPushButton("Start Demo Data Stream")
        self.start_demo_btn.clicked.connect(self.start_demo)
        controls_layout.addWidget(self.start_demo_btn)
        
        self.stop_demo_btn = QPushButton("Stop Demo Data")
        self.stop_demo_btn.clicked.connect(self.stop_demo)
        self.stop_demo_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_demo_btn)
        
        layout.addWidget(controls_widget)
        
        # Demo timer
        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(self.generate_demo_data)
        self.demo_counter = 0
        
        # Apply dark theme
        self.apply_dark_theme()
        
    def apply_dark_theme(self):
        """Apply dark theme to the demo."""
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #2b2b2b; 
                color: #ffffff; 
            }
            QPushButton { 
                background-color: #4c4c4c; 
                border: 1px solid #666; 
                padding: 8px; 
                border-radius: 4px; 
                color: #ffffff;
            }
            QPushButton:hover { 
                background-color: #5c5c5c; 
            }
            QPushButton:disabled {
                background-color: #3a3a3a;
                color: #888;
            }
        """)
        
    def start_demo(self):
        """Start demo data generation."""
        print("🚀 Starting bitstream demo...")
        self.demo_timer.start(50)  # Generate data every 50ms
        self.start_demo_btn.setEnabled(False)
        self.stop_demo_btn.setEnabled(True)
        self.demo_counter = 0
        
    def stop_demo(self):
        """Stop demo data generation."""
        print("🛑 Stopping bitstream demo...")
        self.demo_timer.stop()
        self.start_demo_btn.setEnabled(True)
        self.stop_demo_btn.setEnabled(False)
        
    def generate_demo_data(self):
        """Generate interesting demo bitstream patterns."""
        # Cycle through different patterns
        pattern_type = (self.demo_counter // 100) % 8
        
        if pattern_type == 0:
            # Alternating pattern
            bits = [self.demo_counter % 2] * 8
            pattern_name = "Alternating"
        elif pattern_type == 1:
            # Block pattern
            bits = [1] * 6 + [0] * 6
            pattern_name = "Block"
        elif pattern_type == 2:
            # Manchester encoding simulation
            data_bit = (self.demo_counter // 10) % 2
            bits = [0, 1] if data_bit else [1, 0]
            bits = bits * 4  # Repeat pattern
            pattern_name = "Manchester"
        elif pattern_type == 3:
            # PRBS pattern (Pseudo-Random Binary Sequence)
            np.random.seed(self.demo_counter // 20)
            bits = np.random.randint(0, 2, 12).tolist()
            pattern_name = "PRBS"
        elif pattern_type == 4:
            # Sync pattern + data
            sync = [1, 0, 1, 0, 1, 1, 0, 0]  # Sync word
            data = np.random.randint(0, 2, 8).tolist()
            bits = sync + data
            pattern_name = "Sync+Data"
        elif pattern_type == 5:
            # High entropy random
            bits = np.random.randint(0, 2, 10).tolist()
            pattern_name = "Random"
        elif pattern_type == 6:
            # Low entropy (mostly zeros)
            bits = [0] * 9 + [1]
            pattern_name = "Low Entropy"
        else:
            # Binary counter
            counter_val = (self.demo_counter // 5) % 256
            bits = [(counter_val >> i) & 1 for i in range(8)]
            pattern_name = "Counter"
        
        # Add to display
        self.bitstream_widget.add_bits(bits)
        
        # Print pattern info every 100 iterations
        if self.demo_counter % 100 == 0:
            print(f"📊 Pattern {pattern_type + 1}/8: {pattern_name}")
            print(f"   Bits: {bits[:8]}{'...' if len(bits) > 8 else ''}")
            
            # Show current statistics
            info = self.bitstream_widget.get_display_info()
            print(f"   Total bits: {info['total_bits']}")
            print(f"   Entropy: {info['entropy']:.3f}")
            print(f"   Buffer usage: {info['buffer_usage']*100:.1f}%")
        
        self.demo_counter += 1


def main():
    """Main demo application."""
    app = QApplication(sys.argv)
    
    print("🔬 RF Spectrum Analyzer - Bitstream Widget Demo")
    print("=" * 60)
    print("Features demonstrated:")
    print("  ✅ Real-time bitstream visualization")
    print("  ✅ Configurable display parameters")
    print("  ✅ Statistical analysis and entropy calculation")
    print("  ✅ Multiple bit patterns and encodings")
    print("  ✅ Interactive controls and data management")
    print("  ✅ Export and analysis capabilities")
    print("=" * 60)
    
    # Create demo window
    demo = BitstreamDemo()
    demo.show()
    
    print("🎮 Demo started - Use buttons to control data stream")
    print("   • Start Demo Data: Begin pattern demonstration")
    print("   • Stop Demo Data: Pause data generation")
    print("   • Widget controls: Adjust display settings")
    
    # Show initial test pattern
    test_pattern = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    demo.bitstream_widget.add_bits(test_pattern)
    print(f"   • Initial pattern: {test_pattern}")
    
    # Run application
    exit_code = app.exec()
    
    print("\n👋 Demo completed")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())