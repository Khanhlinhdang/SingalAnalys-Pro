#!/usr/bin/env python3
"""
Unit test for SpectrumWidget LinearRegionItem functionality
Verifies the new features without GUI dependency.
"""

import sys
import numpy as np

# Add the project directory to path for imports
sys.path.append('.')

from rf_spectrum_analyzer.config.settings import Settings


def test_spectrum_widget_imports():
    """Test that SpectrumWidget can be imported successfully."""
    try:
        from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
        print("✓ SpectrumWidget imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import SpectrumWidget: {e}")
        return False


def test_spectrum_widget_initialization():
    """Test SpectrumWidget initialization without showing GUI."""
    try:
        from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
        
        # Create settings
        settings = Settings()
        settings.sdr.center_frequency = 1545e6  # 1545 MHz
        settings.sdr.sample_rate = 10e6  # 10 MHz
        
        # Create widget (without parent to avoid GUI display)
        widget = SpectrumWidget(settings, parent=None)
        
        # Check that LinearRegionItem replaced the old markers
        assert hasattr(widget, 'freq_range_region'), "freq_range_region should exist"
        assert not hasattr(widget, 'f1_marker') or widget.f1_marker is None, "f1_marker should not exist"
        assert not hasattr(widget, 'f2_marker') or widget.f2_marker is None, "f2_marker should not exist"
        
        print("✓ SpectrumWidget initialized successfully with LinearRegionItem")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize SpectrumWidget: {e}")
        return False


def test_frequency_range_methods():
    """Test frequency range methods."""
    try:
        from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
        
        # Create settings
        settings = Settings()
        settings.sdr.center_frequency = 1545e6
        settings.sdr.sample_rate = 10e6
        
        # Create widget
        widget = SpectrumWidget(settings, parent=None)
        
        # Test setting frequency range
        f1, f2 = 1540e6, 1550e6
        widget.set_frequency_range(f1, f2)
        
        # Verify internal state
        assert widget.f1_frequency == f1, f"Expected f1={f1}, got {widget.f1_frequency}"
        assert widget.f2_frequency == f2, f"Expected f2={f2}, got {widget.f2_frequency}"
        
        # Test getting frequency range
        retrieved_f1, retrieved_f2 = widget.get_frequency_range()
        assert retrieved_f1 == f1, f"Retrieved f1 mismatch: {retrieved_f1} != {f1}"
        assert retrieved_f2 == f2, f"Retrieved f2 mismatch: {retrieved_f2} != {f2}"
        
        print("✓ Frequency range methods work correctly")
        return True
    except Exception as e:
        print(f"✗ Frequency range methods failed: {e}")
        return False


def test_center_button_functionality():
    """Test center button functionality (method only, no GUI)."""
    try:
        from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
        
        # Create settings
        settings = Settings()
        settings.sdr.center_frequency = 1545e6
        settings.sdr.sample_rate = 10e6
        
        # Create widget
        widget = SpectrumWidget(settings, parent=None)
        
        # Test _on_center_button_clicked method
        widget._on_center_button_clicked()
        
        # Verify that frequency range is set around center frequency
        center_freq = settings.sdr.center_frequency
        sample_rate = settings.sdr.sample_rate
        expected_range_width = sample_rate * 0.1
        expected_f1 = center_freq - expected_range_width / 2
        expected_f2 = center_freq + expected_range_width / 2
        
        # Allow some tolerance for floating point comparisons
        tolerance = 1e3  # 1 kHz
        assert abs(widget.f1_frequency - expected_f1) < tolerance, f"f1 not centered correctly: {widget.f1_frequency} != {expected_f1}"
        assert abs(widget.f2_frequency - expected_f2) < tolerance, f"f2 not centered correctly: {widget.f2_frequency} != {expected_f2}"
        
        print("✓ Center button functionality works correctly")
        return True
    except Exception as e:
        print(f"✗ Center button functionality failed: {e}")
        return False


def test_markers_enabled_toggle():
    """Test markers enabled/disabled functionality."""
    try:
        from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
        
        # Create settings
        settings = Settings()
        
        # Create widget
        widget = SpectrumWidget(settings, parent=None)
        
        # Test enabling markers
        widget.set_frequency_markers_enabled(True)
        assert widget.markers_enabled == True, "Markers should be enabled"
        
        # Test disabling markers
        widget.set_frequency_markers_enabled(False)
        assert widget.markers_enabled == False, "Markers should be disabled"
        
        print("✓ Markers enable/disable functionality works correctly")
        return True
    except Exception as e:
        print(f"✗ Markers enable/disable functionality failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing SpectrumWidget LinearRegionItem functionality...")
    print("=" * 60)
    
    tests = [
        test_spectrum_widget_imports,
        test_spectrum_widget_initialization,
        test_frequency_range_methods,
        test_center_button_functionality,
        test_markers_enabled_toggle,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LinearRegionItem integration successful.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())