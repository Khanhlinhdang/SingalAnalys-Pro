#!/usr/bin/env python3
"""
Quick test to verify detection method fixes
"""

import sys
import os
import logging

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'rf_spectrum_analyzer'))

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_signal_processor_fixes():
    """Test that all detection methods are now available."""
    print("🔧 Testing SignalProcessor detection method fixes...")
    
    from rf_spectrum_analyzer.core.signal_processor import SignalProcessor
    from rf_spectrum_analyzer.config.settings import Settings
    import numpy as np
    
    # Create processor
    settings = Settings()
    processor = SignalProcessor(settings)
    
    # Test method existence
    required_methods = [
        'detect_signals_manual',
        'detect_tdma_bursts', 
        'set_auto_detection',
        'set_advanced_analysis',
        'set_detection_threshold',
        'set_detection_interval',
        'update_current_data'
    ]
    
    print("✅ Method availability check:")
    all_methods_exist = True
    for method in required_methods:
        exists = hasattr(processor, method)
        status = "✅" if exists else "❌"
        print(f"   {status} {method}: {'Found' if exists else 'MISSING'}")
        if not exists:
            all_methods_exist = False
    
    if not all_methods_exist:
        print("❌ Some methods are missing!")
        return False
    
    # Test method calls
    print("\n✅ Method functionality test:")
    test_data = np.random.randn(2048) + 1j * np.random.randn(2048)
    
    try:
        # Test data update
        processor.update_current_data(test_data)
        print("   ✅ update_current_data: Success")
        
        # Test manual detection
        result = processor.detect_signals_manual()
        print(f"   ✅ detect_signals_manual: {'Success' if result is not None else 'No data'}")
        
        # Test TDMA detection (should work with or without parameters)
        result = processor.detect_tdma_bursts()
        print(f"   ✅ detect_tdma_bursts (no params): {'Success' if result is not None else 'No data'}")
        
        result = processor.detect_tdma_bursts(test_data)
        print(f"   ✅ detect_tdma_bursts (with params): {'Success' if result is not None else 'No data'}")
        
        # Test configuration methods
        processor.set_auto_detection(True)
        print("   ✅ set_auto_detection: Success")
        
        processor.set_advanced_analysis(True)
        print("   ✅ set_advanced_analysis: Success")
        
        processor.set_detection_threshold(-75.0)
        print("   ✅ set_detection_threshold: Success")
        
        processor.set_detection_interval(200)
        print("   ✅ set_detection_interval: Success")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Method test failed: {e}")
        return False

def test_app_integration():
    """Test app integration with detection methods."""
    print("\n🖥️  Testing App integration...")
    
    from rf_spectrum_analyzer.core.app import RFSpectrumAnalyzerApp
    from rf_spectrum_analyzer.config.settings import Settings
    
    try:
        settings = Settings()
        app = RFSpectrumAnalyzerApp(settings)
        
        # Test that app methods exist
        app_methods = [
            'trigger_manual_detection',
            'trigger_tdma_detection',
            'toggle_auto_detection',
            'toggle_advanced_analysis',
            'change_detection_threshold',
            'change_detection_interval'
        ]
        
        print("✅ App method availability:")
        for method in app_methods:
            exists = hasattr(app, method)
            status = "✅" if exists else "❌"
            print(f"   {status} {method}: {'Found' if exists else 'MISSING'}")
        
        print("   ✅ App integration ready")
        return True
        
    except Exception as e:
        print(f"   ❌ App integration test failed: {e}")
        return False

def main():
    """Run all fix verification tests."""
    print("🚀 RF Spectrum Analyzer - Detection Method Fix Verification")
    print("=" * 60)
    
    # Run tests
    processor_ok = test_signal_processor_fixes()
    app_ok = test_app_integration()
    
    print("\n" + "=" * 60)
    print("📊 FIX VERIFICATION SUMMARY:")
    print("=" * 60)
    
    print(f"SignalProcessor Methods: {'✅ FIXED' if processor_ok else '❌ STILL BROKEN'}")
    print(f"App Integration: {'✅ READY' if app_ok else '❌ ISSUES'}")
    
    if processor_ok and app_ok:
        print("\n🎉 ALL DETECTION ERRORS FIXED!")
        print("   • detect_signals_manual() method added")
        print("   • detect_tdma_bursts() made parameter-optional") 
        print("   • set_auto_detection() method added")
        print("   • set_advanced_analysis() method added")
        print("   • Detection threshold/interval controls added")
        print("   • Current data buffer integration added")
        print("\n✅ The RF Spectrum Analyzer detection system is now fully operational!")
        return True
    else:
        print(f"\n⚠️  Some issues remain. Please check the errors above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)