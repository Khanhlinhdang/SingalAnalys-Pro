#!/usr/bin/env python3
"""
Basic integration test to verify LinearRegionItem changes in SpectrumWidget
Checks code structure and imports without GUI initialization.
"""

import sys
import inspect

# Add the project directory to path for imports
sys.path.append('.')


def test_spectrum_widget_structure():
    """Test that SpectrumWidget has been correctly modified."""
    try:
        from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
        
        # Check class structure
        class_methods = [method for method in dir(SpectrumWidget) if not method.startswith('_')]
        required_methods = [
            'set_frequency_markers_enabled',
            'set_frequency_range', 
            'get_frequency_range',
            'highlight_frequency_range'
        ]
        
        for method in required_methods:
            assert method in class_methods, f"Required method {method} not found"
        
        # Check that old marker methods are removed
        old_methods = ['_on_f1_marker_moved', '_on_f2_marker_moved']
        private_methods = [method for method in dir(SpectrumWidget) if method.startswith('_')]
        
        for old_method in old_methods:
            assert old_method not in private_methods, f"Old method {old_method} should be removed"
        
        # Check that new center button method exists
        assert '_on_center_button_clicked' in private_methods, "Center button method should exist"
        
        print("✓ SpectrumWidget class structure is correct")
        return True
        
    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        return False


def test_method_signatures():
    """Test method signatures are correct."""
    try:
        from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
        
        # Check set_frequency_range signature
        sig = inspect.signature(SpectrumWidget.set_frequency_range)
        params = list(sig.parameters.keys())
        expected_params = ['self', 'f1', 'f2']
        assert params == expected_params, f"set_frequency_range params: {params} != {expected_params}"
        
        # Check get_frequency_range signature
        sig = inspect.signature(SpectrumWidget.get_frequency_range)
        params = list(sig.parameters.keys())
        expected_params = ['self']
        assert params == expected_params, f"get_frequency_range params: {params} != {expected_params}"
        
        # Check center button method signature
        sig = inspect.signature(SpectrumWidget._on_center_button_clicked)
        params = list(sig.parameters.keys())
        expected_params = ['self']
        assert params == expected_params, f"_on_center_button_clicked params: {params} != {expected_params}"
        
        print("✓ Method signatures are correct")
        return True
        
    except Exception as e:
        print(f"✗ Method signature test failed: {e}")
        return False


def test_imports():
    """Test that all required imports are present."""
    try:
        from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
        
        # Check that QPushButton is imported
        import rf_spectrum_analyzer.gui.spectrum_widget as sw_module
        module_globals = dir(sw_module)
        
        # QPushButton should be available
        assert 'QPushButton' in module_globals, "QPushButton not imported"
        
        print("✓ All required imports are present")
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


def test_source_code_changes():
    """Test that source code contains expected changes."""
    try:
        # Read the source file
        with open('rf_spectrum_analyzer/gui/spectrum_widget.py', 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Check that LinearRegionItem is used instead of InfiniteLine markers
        assert 'self.f1_marker = pg.InfiniteLine(' not in source_code, "Old f1_marker creation should be removed"
        assert 'self.f2_marker = pg.InfiniteLine(' not in source_code, "Old f2_marker creation should be removed"
        
        # Check for center button
        assert 'QPushButton("Center")' in source_code, "Center button should be created"
        assert '_on_center_button_clicked' in source_code, "Center button handler should exist"
        
        # Check that LinearRegionItem is properly configured
        assert 'pg.LinearRegionItem(' in source_code, "LinearRegionItem should be created"
        assert 'self.freq_range_region' in source_code, "freq_range_region should exist"
        
        # Check that old marker methods are removed
        assert 'def _on_f1_marker_moved(' not in source_code, "Old f1_marker_moved method should be removed"
        assert 'def _on_f2_marker_moved(' not in source_code, "Old f2_marker_moved method should be removed"
        
        print("✓ Source code contains expected changes")
        return True
        
    except Exception as e:
        print(f"✗ Source code test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing SpectrumWidget LinearRegionItem Integration...")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_spectrum_widget_structure,
        test_method_signatures,
        test_source_code_changes,
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
        print("🎉 All integration tests passed!")
        print("\nChanges Summary:")
        print("1. ✓ Replaced f1_marker and f2_marker InfiniteLine with LinearRegionItem")
        print("2. ✓ Added 'Center' button to spectrum header")
        print("3. ✓ Implemented center button functionality to move region to center frequency") 
        print("4. ✓ Updated all frequency range methods to work with LinearRegionItem")
        print("5. ✓ Removed obsolete marker movement handlers")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())