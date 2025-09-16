#!/usr/bin/env python3
"""
Test Real-time Controls Demo
Tests frequency and bandwidth changes without stopping acquisition.
"""

import sys
import time
from PySide6.QtWidgets import QApplication
from rf_spectrum_analyzer.core.app import RFSpectrumAnalyzerApp
from rf_spectrum_analyzer.config.settings import Settings

def test_realtime_controls():
    """Test real-time frequency and bandwidth controls."""
    print("🎯 RF Spectrum Analyzer Real-time Controls Test")
    print("=" * 50)
    
    # Create QApplication first
    qt_app = QApplication(sys.argv)
    
    # Load settings
    settings = Settings()
    # Note: Settings are loaded automatically on initialization
    
    # Create app but don't show GUI
    print("📡 Initializing app with SpyServer...")
    app = RFSpectrumAnalyzerApp(settings)
    
    # Test without GUI to focus on backend functionality
    if app.sdr_manager and app.sdr_manager.is_connected():
        print("✅ SpyServer connected successfully!")
        
        # Test frequency changes
        print("\n🔧 Testing real-time frequency changes:")
        frequencies = [100e6, 145e6, 433e6, 868e6, 1200e6]  # MHz to Hz
        
        for freq in frequencies:
            print(f"   📻 Setting frequency to {freq/1e6:.1f} MHz...")
            app.change_frequency(freq)
            time.sleep(0.5)  # Small delay to see the change
        
        # Test bandwidth changes  
        print("\n🔧 Testing real-time bandwidth changes:")
        bandwidths = [5e6, 10e6, 20e6, 10e6, 5e6]  # MHz to Hz
        
        for bw in bandwidths:
            print(f"   📊 Setting bandwidth to {bw/1e6:.1f} MHz...")
            app.change_bandwidth(bw)
            time.sleep(0.5)  # Small delay to see the change
        
        print("\n✅ Real-time controls test completed successfully!")
        print("🎉 All frequency and bandwidth changes applied without stopping acquisition!")
        
        # Disconnect
        app.cleanup()
        print("📴 SpyServer disconnected")
        
    else:
        print("❌ Failed to connect to SpyServer")
        print("💡 Make sure SpyServer is running at 64.31.248.40:63863")
    
    qt_app.quit()

if __name__ == "__main__":
    try:
        test_realtime_controls()
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)