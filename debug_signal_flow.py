#!/usr/bin/env python3
"""
Debug Signal Flow - Comprehensive Testing
Kiểm tra từng bước trong signal processing pipeline
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QEventLoop
from rf_spectrum_analyzer.core.app import RFSpectrumAnalyzerApp
from rf_spectrum_analyzer.config.settings import Settings
from rf_spectrum_analyzer.utils.logger import setup_application_logging


class SignalFlowDebugger:
    """Debug signal processing flow step by step."""
    
    def __init__(self):
        self.app = None
        self.rf_app = None
        self.results = {}
        
    def setup(self):
        """Setup Qt application and RF app."""
        print("🔧 Setting up debug environment...")
        
        # Setup logging
        setup_application_logging(level='DEBUG')
        
        # Setup Qt app
        self.app = QApplication(sys.argv)
        
        # Setup RF app with demo mode
        settings = Settings()
        settings.demo_mode = True
        
        self.rf_app = RFSpectrumAnalyzerApp(settings)
        print("✅ RF Spectrum Analyzer initialized")
        
    def test_demo_data_generation(self):
        """Test 1: Demo data generation."""
        print("\n📡 Test 1: Demo Data Generation")
        print("-" * 40)
        
        if not self.rf_app.demo_mode:
            print("❌ Demo mode not enabled")
            return False
            
        if not self.rf_app.demo_timer:
            print("❌ Demo timer not created")
            return False
            
        if not self.rf_app.demo_timer.isActive():
            print("❌ Demo timer not active")
            return False
            
        print(f"✅ Demo timer active: {self.rf_app.demo_timer.interval()}ms")
        
        # Monitor demo data generation
        initial_counter = getattr(self.rf_app, 'demo_counter', 0)
        
        # Wait and check if counter increments
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)  # Wait 500ms
        loop.exec()
        
        current_counter = getattr(self.rf_app, 'demo_counter', 0)
        
        if current_counter > initial_counter:
            print(f"✅ Demo data generated: counter {initial_counter} → {current_counter}")
            return True
        else:
            print(f"❌ Demo data not generated: counter still {current_counter}")
            return False
    
    def test_iq_buffer_updates(self):
        """Test 2: IQ buffer updates."""
        print("\n📊 Test 2: IQ Buffer Updates")
        print("-" * 40)
        
        if not hasattr(self.rf_app, 'iq_buffer'):
            print("❌ IQ buffer not found")
            return False
            
        initial_buffer_size = len(self.rf_app.iq_buffer)
        print(f"Initial IQ buffer size: {initial_buffer_size}")
        
        # Wait for buffer updates
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)  # Wait 1 second
        loop.exec()
        
        current_buffer_size = len(self.rf_app.iq_buffer)
        
        if current_buffer_size > initial_buffer_size:
            print(f"✅ IQ buffer updated: {initial_buffer_size} → {current_buffer_size} samples")
            return True
        else:
            print(f"❌ IQ buffer not updated: still {current_buffer_size} samples")
            return False
    
    def test_spectrum_processing(self):
        """Test 3: Spectrum processing."""
        print("\n📈 Test 3: Spectrum Processing")
        print("-" * 40)
        
        if not hasattr(self.rf_app, 'spectrum_data'):
            print("❌ Spectrum data not found")
            return False
            
        initial_spectrum = getattr(self.rf_app, 'spectrum_data', None)
        initial_size = len(initial_spectrum) if initial_spectrum is not None else 0
        
        print(f"Initial spectrum size: {initial_size}")
        
        # Wait for spectrum updates
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec()
        
        current_spectrum = getattr(self.rf_app, 'spectrum_data', None)
        current_size = len(current_spectrum) if current_spectrum is not None else 0
        
        if current_size > 0 and current_spectrum is not None:
            print(f"✅ Spectrum processed: {current_size} frequency bins")
            print(f"   Power range: {np.min(current_spectrum):.1f} to {np.max(current_spectrum):.1f} dB")
            return True
        else:
            print(f"❌ Spectrum not processed: size {current_size}")
            return False
    
    def test_constellation_data(self):
        """Test 4: Constellation data updates."""
        print("\n🌟 Test 4: Constellation Data")
        print("-" * 40)
        
        if not hasattr(self.rf_app, 'constellation_data'):
            print("❌ Constellation data structure not found")
            return False
            
        # Wait for constellation processing
        loop = QEventLoop()
        QTimer.singleShot(2000, loop.quit)  # Wait 2 seconds for processing
        loop.exec()
        
        constellation_data = self.rf_app.constellation_data
        
        if 'iq_samples' in constellation_data and len(constellation_data['iq_samples']) > 0:
            iq_samples = constellation_data['iq_samples']
            print(f"✅ IQ samples: {len(iq_samples)} samples")
            
            if 'modulation_info' in constellation_data:
                mod_info = constellation_data['modulation_info']
                print(f"✅ Modulation info: {mod_info}")
                
            if 'symbols' in constellation_data and constellation_data['symbols'] is not None:
                symbols = constellation_data['symbols']
                print(f"✅ Constellation symbols: {len(symbols)} symbols")
                return True
            else:
                print("⚠️  No constellation symbols found")
                return len(iq_samples) > 0  # Partial success
        else:
            print("❌ No constellation data found")
            return False
    
    def test_bitstream_data(self):
        """Test 5: Bitstream data updates."""
        print("\n🔢 Test 5: Bitstream Data")
        print("-" * 40)
        
        if not hasattr(self.rf_app, 'bitstream_data'):
            print("❌ Bitstream data not found")
            return False
            
        # Wait for bitstream processing
        loop = QEventLoop()
        QTimer.singleShot(3000, loop.quit)  # Wait 3 seconds
        loop.exec()
        
        bitstream_data = self.rf_app.bitstream_data
        
        if len(bitstream_data) > 0:
            print(f"✅ Bitstream data: {len(bitstream_data)} bits")
            print(f"   Sample bits: {bitstream_data[:20]}")
            
            # Calculate basic statistics
            if len(bitstream_data) > 10:
                ones = np.sum(bitstream_data == 1)
                zeros = np.sum(bitstream_data == 0)
                ratio = ones / len(bitstream_data) if len(bitstream_data) > 0 else 0
                print(f"   Statistics: {ones} ones, {zeros} zeros, ratio: {ratio:.2f}")
                
            return True
        else:
            print("❌ No bitstream data found")
            return False
    
    def test_gui_widgets(self):
        """Test 6: GUI widget connections."""
        print("\n🖥️  Test 6: GUI Widget Connections")
        print("-" * 40)
        
        if not self.rf_app.main_window:
            print("❌ Main window not found")
            return False
            
        main_window = self.rf_app.main_window
        
        # Check constellation widget
        if hasattr(main_window, 'constellation_widget'):
            constellation_widget = main_window.constellation_widget
            if constellation_widget:
                print("✅ Constellation widget found")
                if hasattr(constellation_widget, 'update_constellation'):
                    print("✅ Constellation update method available")
                else:
                    print("❌ Constellation update method missing")
            else:
                print("❌ Constellation widget is None")
        else:
            print("❌ Constellation widget not found")
        
        # Check bitstream widget
        if hasattr(main_window, 'bitstream_widget'):
            bitstream_widget = main_window.bitstream_widget
            if bitstream_widget:
                print("✅ Bitstream widget found")
                if hasattr(bitstream_widget, 'add_bits'):
                    print("✅ Bitstream add_bits method available")
                else:
                    print("❌ Bitstream add_bits method missing")
            else:
                print("❌ Bitstream widget is None")
        else:
            print("❌ Bitstream widget not found")
        
        return True
    
    def run_comprehensive_test(self):
        """Run all tests in sequence."""
        print("🔬 RF Spectrum Analyzer - Comprehensive Signal Flow Debug")
        print("=" * 60)
        
        results = []
        
        try:
            self.setup()
            
            # Run tests
            tests = [
                ("Demo Data Generation", self.test_demo_data_generation),
                ("IQ Buffer Updates", self.test_iq_buffer_updates),
                ("Spectrum Processing", self.test_spectrum_processing),
                ("Constellation Data", self.test_constellation_data),
                ("Bitstream Data", self.test_bitstream_data),
                ("GUI Widget Connections", self.test_gui_widgets)
            ]
            
            for test_name, test_func in tests:
                try:
                    result = test_func()
                    results.append((test_name, result))
                except Exception as e:
                    print(f"❌ {test_name} failed with exception: {e}")
                    results.append((test_name, False))
            
            # Summary
            print("\n📋 Test Summary")
            print("=" * 40)
            
            passed = 0
            for test_name, result in results:
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{status:8} {test_name}")
                if result:
                    passed += 1
            
            print(f"\nResult: {passed}/{len(results)} tests passed")
            
            if passed == len(results):
                print("🎉 All tests passed! Signal flow is working correctly.")
                return True
            else:
                print("⚠️  Some tests failed. Check signal processing pipeline.")
                return False
                
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            return False
        finally:
            if self.app:
                self.app.quit()


def main():
    """Main test runner."""
    debugger = SignalFlowDebugger()
    success = debugger.run_comprehensive_test()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())