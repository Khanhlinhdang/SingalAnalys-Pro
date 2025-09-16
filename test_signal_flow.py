#!/usr/bin/env python3
"""
Test script để kiểm tra signal analysis flow từ spectrum widget
"""

import sys
import os
import numpy as np

# Add the project directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_spectrum_widget_analysis():
    """Test signal analysis flow trong spectrum widget"""
    print("=== Testing Spectrum Widget Signal Analysis Flow ===")
    
    try:
        from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
        from rf_spectrum_analyzer.config.settings import Settings
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer
        import pyqtgraph as pg
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create settings
        settings = Settings()
        settings.sdr.center_frequency = 100e6
        settings.sdr.sample_rate = 1e6
        
        # Create spectrum widget
        spectrum_widget = SpectrumWidget(settings)
        
        print("✓ Spectrum widget created successfully")
        
        # Test 1: Check if frequency range region exists
        print("\n--- Test 1: Frequency Range Region ---")
        assert hasattr(spectrum_widget, 'freq_range_region'), "freq_range_region not found"
        assert spectrum_widget.freq_range_region is not None, "freq_range_region is None"
        print("✓ freq_range_region exists")
        
        # Test 2: Check signal connections
        print("\n--- Test 2: Signal Connections ---")
        
        # Check if signal_analysis_requested signal exists
        assert hasattr(spectrum_widget, 'signal_analysis_requested'), "signal_analysis_requested signal not found"
        print("✓ signal_analysis_requested signal exists")
        
        # Check if frequency_range_selected signal exists
        assert hasattr(spectrum_widget, 'frequency_range_selected'), "frequency_range_selected signal not found"
        print("✓ frequency_range_selected signal exists")
        
        # Test 3: Test frequency range region functionality
        print("\n--- Test 3: Frequency Range Region Functionality ---")
        
        # Generate test spectrum data
        freq_center = 100e6
        sample_rate = 1e6
        num_points = 1024
        
        # Create test signal with a peak
        freqs = np.linspace(freq_center - sample_rate/2, freq_center + sample_rate/2, num_points)
        signal_freq = freq_center + 50e3  # 50 kHz offset
        test_spectrum = -80 * np.ones(num_points)  # Noise floor
        
        # Add a signal peak
        signal_idx = np.argmin(np.abs(freqs - signal_freq))
        test_spectrum[signal_idx-5:signal_idx+5] = -30  # Signal peak
        
        # Update spectrum widget with test data
        spectrum_widget.update_data(test_spectrum)
        print("✓ Test spectrum data loaded")
        
        # Test 4: Enable frequency markers and set range
        print("\n--- Test 4: Frequency Markers ---")
        spectrum_widget.set_frequency_markers_enabled(True)
        print("✓ Frequency markers enabled")
        
        # Set frequency range around the signal
        f1 = signal_freq - 25e3
        f2 = signal_freq + 25e3
        spectrum_widget.set_frequency_range(f1, f2)
        print(f"✓ Frequency range set: {f1/1e6:.3f} - {f2/1e6:.3f} MHz")
        
        # Test 5: Test get_frequency_range_data
        print("\n--- Test 5: Get Frequency Range Data ---")
        freq_data, power_data, stats = spectrum_widget.get_frequency_range_data()
        
        if freq_data is not None:
            print(f"✓ Frequency range data extracted:")
            print(f"  - Frequency points: {len(freq_data)}")
            print(f"  - Power points: {len(power_data)}")
            print(f"  - Center frequency: {stats['center_freq']/1e6:.3f} MHz")
            print(f"  - Bandwidth: {stats['bandwidth']/1e3:.1f} kHz")
            print(f"  - Peak power: {stats['peak_power']:.1f} dB")
            print(f"  - Signal present: {stats['signal_present']}")
        else:
            print("✗ Failed to extract frequency range data")
            
        # Test 6: Test signal analysis request
        print("\n--- Test 6: Signal Analysis Request ---")
        
        # Create a signal capture to test if signal is emitted
        signal_emitted = {'count': 0, 'data': None}
        
        def on_signal_analysis_requested(data):
            signal_emitted['count'] += 1
            signal_emitted['data'] = data
            print(f"✓ Signal analysis requested signal emitted!")
            print(f"  Data: {data}")
        
        # Connect the signal
        spectrum_widget.signal_analysis_requested.connect(on_signal_analysis_requested)
        
        # Test request_signal_analysis method
        analysis_request = spectrum_widget.request_signal_analysis()
        
        if analysis_request is not None:
            print("✓ Signal analysis request created:")
            print(f"  - Center freq: {analysis_request['center_freq']/1e6:.3f} MHz")
            print(f"  - Bandwidth: {analysis_request['bandwidth']/1e3:.1f} kHz")
        else:
            print("✗ Signal analysis request failed")
        
        # Test 7: Test analyze button functionality
        print("\n--- Test 7: Analyze Button ---")
        
        # Simulate analyze button click
        spectrum_widget._on_analyze_button_clicked()
        
        if signal_emitted['count'] > 0:
            print(f"✓ Analyze button triggered signal emission ({signal_emitted['count']} times)")
        else:
            print("✗ Analyze button did not trigger signal emission")
        
        app.quit()
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_connections():
    """Test xem signal connections có đúng không"""
    print("\n=== Testing Signal Connections trong Main App ===")
    
    try:
        # Import app components
        from rf_spectrum_analyzer.core.app import RFSpectrumAnalyzerApp
        from rf_spectrum_analyzer.config.settings import Settings
        from PySide6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        
        settings = Settings()
        settings.demo_mode = True  # Use demo mode
        
        # Create main app
        rf_app = RFSpectrumAnalyzerApp(settings)
        
        print("✓ Main app created")
        
        # Check if main window has spectrum widget
        main_window = rf_app.main_window
        assert hasattr(main_window, 'spectrum'), "Main window missing spectrum widget"
        
        spectrum_widget = main_window.spectrum
        print("✓ Spectrum widget found in main window")
        
        # Check if constellation widget exists
        assert hasattr(main_window, 'constellation'), "Main window missing constellation widget"
        constellation_widget = main_window.constellation
        print("✓ Constellation widget found")
        
        # Check if bitstream widget exists  
        assert hasattr(main_window, 'bitstream'), "Main window missing bitstream widget"
        bitstream_widget = main_window.bitstream
        print("✓ Bitstream widget found")
        
        # Check signal connections
        print("\n--- Checking Signal Connections ---")
        
        # Check if spectrum widget's signal_analysis_requested is connected
        spectrum_signal = spectrum_widget.signal_analysis_requested
        print(f"spectrum signal_analysis_requested receivers: {spectrum_signal.receivers}")
        
        if len(spectrum_signal.receivers) > 0:
            print("✓ signal_analysis_requested is connected")
        else:
            print("✗ signal_analysis_requested is NOT connected")
        
        app.quit()
        
        return True
        
    except Exception as e:
        print(f"✗ Signal connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Signal Analysis Flow...")
    
    # Test 1: Spectrum widget functionality
    test1_success = test_spectrum_widget_analysis()
    
    # Test 2: Signal connections in main app
    test2_success = test_signal_connections()
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        
    print("\nNext: Check main_window.py để xem signal connections...")