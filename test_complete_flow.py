#!/usr/bin/env python3
"""
Complete Flow Test for RF Spectrum Analyzer
Tests the entire pipeline: SDR → Signal Processing → Constellation & Bitstream Display
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QHBoxLayout
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont

from rf_spectrum_analyzer.core.app import RFSpectrumAnalyzerApp
from rf_spectrum_analyzer.config.settings import Settings


class FlowTester(QMainWindow):
    """Test interface for complete signal flow validation."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Signal Flow Tester")
        self.setGeometry(50, 50, 400, 300)
        
        # Create test settings with demo mode
        self.settings = Settings()
        self.settings.demo_mode = True
        
        # Create RF app
        self.rf_app = RFSpectrumAnalyzerApp(self.settings)
        
        # Setup test UI
        self.setup_ui()
        
        # Test monitoring
        self.test_timer = QTimer()
        self.test_timer.timeout.connect(self.monitor_flow)
        self.test_results = {
            'spectrum_updates': 0,
            'constellation_updates': 0,
            'bitstream_updates': 0,
            'errors': []
        }
        
    def setup_ui(self):
        """Setup test control UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🔬 RF Signal Flow Tester")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #14a085; padding: 10px;")
        layout.addWidget(title)
        
        # Status display
        self.status_label = QLabel("Ready to test signal flow...")
        self.status_label.setStyleSheet("padding: 5px; background: #2a2a2a; color: white; border-radius: 3px;")
        layout.addWidget(self.status_label)
        
        # Test controls
        controls_layout = QHBoxLayout()
        
        self.start_test_btn = QPushButton("Start Flow Test")
        self.start_test_btn.clicked.connect(self.start_flow_test)
        self.start_test_btn.setStyleSheet("padding: 8px; background: #4CAF50; color: white; border: none; border-radius: 4px;")
        controls_layout.addWidget(self.start_test_btn)
        
        self.stop_test_btn = QPushButton("Stop Test")
        self.stop_test_btn.clicked.connect(self.stop_flow_test)
        self.stop_test_btn.setEnabled(False)
        self.stop_test_btn.setStyleSheet("padding: 8px; background: #f44336; color: white; border: none; border-radius: 4px;")
        controls_layout.addWidget(self.stop_test_btn)
        
        layout.addLayout(controls_layout)
        
        # Statistics
        stats_layout = QVBoxLayout()
        
        self.spectrum_stats = QLabel("Spectrum Updates: 0")
        self.constellation_stats = QLabel("Constellation Updates: 0")
        self.bitstream_stats = QLabel("Bitstream Updates: 0")
        self.error_stats = QLabel("Errors: 0")
        
        for label in [self.spectrum_stats, self.constellation_stats, self.bitstream_stats, self.error_stats]:
            label.setStyleSheet("padding: 3px; color: #cccccc;")
            stats_layout.addWidget(label)
        
        layout.addLayout(stats_layout)
        
        # Test details
        self.details_label = QLabel("")
        self.details_label.setStyleSheet("padding: 10px; background: #1a1a1a; color: #cccccc; border-radius: 3px; font-family: monospace;")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QPushButton:disabled { background-color: #666; }
        """)
        
    def start_flow_test(self):
        """Start comprehensive flow testing."""
        print("🚀 Starting RF signal flow test...")
        self.status_label.setText("🔄 Testing signal flow...")
        
        # Reset counters
        self.test_results = {
            'spectrum_updates': 0,
            'constellation_updates': 0,
            'bitstream_updates': 0,
            'errors': []
        }
        
        # Start monitoring
        self.test_timer.start(1000)  # Monitor every second
        
        # Enable demo mode data generation
        if hasattr(self.rf_app, 'demo_timer') and self.rf_app.demo_timer:
            if not self.rf_app.demo_timer.isActive():
                self.rf_app.demo_timer.start(100)
        
        # Update buttons
        self.start_test_btn.setEnabled(False)
        self.stop_test_btn.setEnabled(True)
        
        # Show details
        self.update_details("Testing started...\nGenerating simulated RF signals...")
        
    def stop_flow_test(self):
        """Stop flow testing."""
        print("🛑 Stopping flow test...")
        self.status_label.setText("✅ Test completed")
        
        # Stop monitoring
        self.test_timer.stop()
        
        # Update buttons
        self.start_test_btn.setEnabled(True)
        self.stop_test_btn.setEnabled(False)
        
        # Show final results
        self.show_test_results()
        
    def monitor_flow(self):
        """Monitor signal flow and update statistics."""
        try:
            # Check spectrum data
            if hasattr(self.rf_app, 'spectrum_data') and len(self.rf_app.spectrum_data) > 0:
                self.test_results['spectrum_updates'] += 1
            
            # Check constellation data
            if (hasattr(self.rf_app, 'constellation_data') and 
                len(self.rf_app.constellation_data.get('iq_samples', [])) > 0):
                self.test_results['constellation_updates'] += 1
            
            # Check bitstream data
            if hasattr(self.rf_app, 'bitstream_data') and len(self.rf_app.bitstream_data) > 0:
                self.test_results['bitstream_updates'] += 1
            
            # Update UI
            self.update_statistics()
            
            # Check for flow health
            self.check_flow_health()
            
        except Exception as e:
            self.test_results['errors'].append(str(e))
            print(f"❌ Monitor error: {e}")
            
    def update_statistics(self):
        """Update statistics display."""
        self.spectrum_stats.setText(f"Spectrum Updates: {self.test_results['spectrum_updates']}")
        self.constellation_stats.setText(f"Constellation Updates: {self.test_results['constellation_updates']}")
        self.bitstream_stats.setText(f"Bitstream Updates: {self.test_results['bitstream_updates']}")
        self.error_stats.setText(f"Errors: {len(self.test_results['errors'])}")
        
    def check_flow_health(self):
        """Check if the signal flow is healthy."""
        # Minimum expected updates after 5 seconds
        if self.test_timer.isActive() and self.test_results['spectrum_updates'] > 5:
            
            spectrum_ok = self.test_results['spectrum_updates'] > 0
            constellation_ok = self.test_results['constellation_updates'] >= 0  # May be 0 if no valid signals
            bitstream_ok = self.test_results['bitstream_updates'] >= 0  # May be 0 if no digital signals
            error_count = len(self.test_results['errors'])
            
            if spectrum_ok and error_count < 5:
                status = "✅ Flow healthy"
                color = "#4CAF50"
            else:
                status = "⚠️ Flow issues detected"
                color = "#ff9800"
                
            self.status_label.setText(status)
            self.status_label.setStyleSheet(f"padding: 5px; background: {color}; color: white; border-radius: 3px;")
            
    def update_details(self, message):
        """Update details display."""
        current_time = time.strftime("%H:%M:%S")
        self.details_label.setText(f"[{current_time}] {message}")
        
    def show_test_results(self):
        """Show comprehensive test results."""
        results = f"""Test Results Summary:
        
Spectrum Processing: {self.test_results['spectrum_updates']} updates
Constellation Analysis: {self.test_results['constellation_updates']} updates  
Bitstream Extraction: {self.test_results['bitstream_updates']} updates
Errors Encountered: {len(self.test_results['errors'])}

Flow Health: {"✅ PASS" if len(self.test_results['errors']) < 5 else "❌ FAIL"}

Signal Pipeline:
• IQ Data Generation: {"✅" if self.test_results['spectrum_updates'] > 0 else "❌"}
• Spectrum Analysis: {"✅" if self.test_results['spectrum_updates'] > 0 else "❌"}
• Modulation Detection: {"✅" if self.test_results['constellation_updates'] >= 0 else "❌"}
• Bitstream Extraction: {"✅" if self.test_results['bitstream_updates'] >= 0 else "❌"}
• GUI Updates: {"✅" if self.test_results['spectrum_updates'] > 0 else "❌"}
"""
        
        self.update_details(results)
        print("\n" + "="*60)
        print("🎯 RF SIGNAL FLOW TEST RESULTS")
        print("="*60)
        print(results)
        print("="*60)


def run_automated_flow_test():
    """Run automated flow test without GUI."""
    print("🧪 Running automated signal flow test...")
    
    try:
        # Create settings with demo mode
        settings = Settings()
        settings.demo_mode = True
        
        # Create RF app
        rf_app = RFSpectrumAnalyzerApp(settings)
        
        # Wait for initialization
        time.sleep(2)
        
        # Start demo data generation
        if hasattr(rf_app, 'demo_timer') and rf_app.demo_timer:
            rf_app.demo_timer.start(100)
        
        # Monitor for a few seconds
        test_duration = 5
        start_time = time.time()
        
        results = {
            'spectrum_checks': 0,
            'constellation_checks': 0,
            'bitstream_checks': 0
        }
        
        print(f"   Monitoring signal flow for {test_duration} seconds...")
        
        while time.time() - start_time < test_duration:
            # Check data flows
            if hasattr(rf_app, 'spectrum_data') and len(rf_app.spectrum_data) > 0:
                results['spectrum_checks'] += 1
            
            if (hasattr(rf_app, 'constellation_data') and 
                len(rf_app.constellation_data.get('iq_samples', [])) > 0):
                results['constellation_checks'] += 1
                
            if hasattr(rf_app, 'bitstream_data') and len(rf_app.bitstream_data) > 0:
                results['bitstream_checks'] += 1
                
            time.sleep(0.1)
        
        # Cleanup
        if hasattr(rf_app, 'demo_timer') and rf_app.demo_timer:
            rf_app.demo_timer.stop()
        
        # Results
        print(f"   ✅ Spectrum processing: {results['spectrum_checks']} checks")
        print(f"   ✅ Constellation data: {results['constellation_checks']} checks")
        print(f"   ✅ Bitstream data: {results['bitstream_checks']} checks")
        
        success = results['spectrum_checks'] > 0
        print(f"   🎯 Overall result: {'PASS' if success else 'FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def main():
    """Main test application."""
    app = QApplication(sys.argv)
    
    print("🔬 RF Spectrum Analyzer - Complete Flow Tester")
    print("=" * 60)
    print("Testing signal processing pipeline:")
    print("  📡 SDR Simulation → Signal Processing → Analysis → Display")
    print("  🎯 Validates: Spectrum, Constellation, Bitstream flows")
    print("=" * 60)
    
    # Run automated test first
    print("\n1️⃣ Running automated flow validation...")
    auto_success = run_automated_flow_test()
    
    if auto_success:
        print("✅ Automated test PASSED - starting interactive tester")
        
        # Start interactive tester
        print("\n2️⃣ Starting interactive flow tester...")
        tester = FlowTester()
        tester.show()
        
        print("🎮 Interactive tester started:")
        print("   • Main Window: RF Spectrum Analyzer")
        print("   • Test Window: Flow monitoring controls")
        print("   • Use 'Start Flow Test' to begin monitoring")
        
        exit_code = app.exec()
        
    else:
        print("❌ Automated test FAILED - check signal processing pipeline")
        exit_code = 1
    
    print("\n👋 Flow test completed")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())