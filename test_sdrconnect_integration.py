#!/usr/bin/env python3
"""
Test script for SDRConnect integration into RF Spectrum Analyzer
Tests all integrated features: SpyServer backend, enhanced analysis, configuration
"""

import sys
import numpy as np
import logging
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sdrconnect_availability():
    """Test if sdrconnect is available."""
    try:
        import sdrconnect
        logger.info(f"✓ sdrconnect available, version: {sdrconnect.__version__}")
        return True
    except ImportError as e:
        logger.error(f"✗ sdrconnect not available: {e}")
        return False

def test_spyserver_backend():
    """Test SpyServer backend creation."""
    try:
        from rf_spectrum_analyzer.config.settings import Settings
        from rf_spectrum_analyzer.backends.spyserver_backend import SpyServerBackend
        
        settings = Settings()
        backend = SpyServerBackend(settings)
        
        logger.info("✓ SpyServer backend created successfully")
        
        # Test device info without connection
        device_info = backend.get_device_info()
        logger.info(f"✓ Device info: {device_info}")
        
        # Test detect devices
        devices = SpyServerBackend.detect_devices()
        logger.info(f"✓ Detected devices: {devices}")
        
        return True
    except Exception as e:
        logger.error(f"✗ SpyServer backend test failed: {e}")
        return False

def test_enhanced_analysis():
    """Test enhanced signal analysis."""
    try:
        from rf_spectrum_analyzer.dsp.enhanced_analysis import EnhancedSignalAnalysis
        
        # Create analyzer
        analyzer = EnhancedSignalAnalysis(sample_rate=2e6, fft_size=1024)
        
        logger.info(f"✓ Enhanced analyzer created")
        logger.info(f"  Capabilities: {analyzer.get_analysis_info()}")
        
        # Test with synthetic signal
        t = np.linspace(0, 1, 2048)
        freq = 1000  # 1 kHz
        iq_signal = np.exp(1j * 2 * np.pi * freq * t) + 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        
        # Perform analysis
        result = analyzer.analyze_iq_data(iq_signal)
        
        logger.info(f"✓ Analysis completed using {result.analysis_method} method")
        logger.info(f"  Peak frequency: {result.peak_frequency:.1f} Hz")
        logger.info(f"  SNR estimate: {result.snr_estimate:.1f} dB")
        logger.info(f"  Bandwidth: {result.bandwidth:.1f} Hz")
        
        if result.sdrconnect_available and result.analysis_method == "enhanced":
            logger.info(f"  Enhanced metrics available:")
            logger.info(f"    RMS power: {result.rms_power:.2f}")
            logger.info(f"    Crest factor: {result.crest_factor:.2f}")
            logger.info(f"    DC offset I/Q: {result.dc_offset_i:.4f}/{result.dc_offset_q:.4f}")
            logger.info(f"    Occupied BW: {result.occupied_bandwidth:.1f} kHz")
        
        return True
    except Exception as e:
        logger.error(f"✗ Enhanced analysis test failed: {e}")
        return False

def test_signal_processor_integration():
    """Test enhanced analysis integration in signal processor."""
    try:
        from rf_spectrum_analyzer.config.settings import Settings
        from rf_spectrum_analyzer.core.signal_processor import SignalProcessor
        
        settings = Settings()
        processor = SignalProcessor(settings)
        
        logger.info("✓ Signal processor with enhanced analysis created")
        
        # Test capabilities
        capabilities = processor.get_analysis_capabilities()
        logger.info(f"✓ Analysis capabilities: {capabilities}")
        
        # Test enhanced analysis method
        t = np.linspace(0, 1, 2048)
        iq_signal = np.exp(1j * 2 * np.pi * 1000 * t) + 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        
        result = processor.enhanced_analysis(iq_signal)
        
        if result['success']:
            logger.info(f"✓ Enhanced analysis via signal processor successful")
            logger.info(f"  Method: {result['analysis_method']}")
            logger.info(f"  Peak frequency: {result['peak_frequency']:.1f} Hz")
            logger.info(f"  Enhanced data available: {result['has_enhanced_data']}")
        else:
            logger.error(f"✗ Enhanced analysis failed: {result.get('error', 'Unknown error')}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"✗ Signal processor integration test failed: {e}")
        return False

def test_sdr_backend_manager():
    """Test SpyServer integration in SDR backend manager."""
    try:
        from rf_spectrum_analyzer.config.settings import Settings
        from rf_spectrum_analyzer.core.sdr_backend import SDRBackendManager, SDRDeviceType
        
        settings = Settings()
        manager = SDRBackendManager(settings)
        
        logger.info("✓ SDR backend manager created")
        
        # Test available devices
        available_devices = manager.get_available_devices()
        logger.info(f"✓ Available device types: {available_devices}")
        
        # Check if SpyServer is available
        if 'spyserver' in available_devices:
            logger.info("✓ SpyServer device type available")
            
            # Test setting SpyServer device type
            if manager.set_device_type('spyserver'):
                logger.info("✓ SpyServer device type set successfully")
                
                # Get device info without connecting
                device_info = manager.get_device_info()
                logger.info(f"✓ SpyServer device info: {device_info}")
            else:
                logger.error("✗ Failed to set SpyServer device type")
                return False
        else:
            logger.error("✗ SpyServer device type not available")
            return False
        
        return True
    except Exception as e:
        logger.error(f"✗ SDR backend manager test failed: {e}")
        return False

def test_settings_sdrconfig_integration():
    """Test SDRConfig integration in settings."""
    try:
        from rf_spectrum_analyzer.config.settings import Settings
        
        settings = Settings()
        
        # Test SpyServer settings
        settings.sdr.device_type = "spyserver"
        settings.sdr.spyserver_host = "test.example.com"
        settings.sdr.spyserver_port = 5556
        settings.sdr.spyserver_timeout = 15.0
        
        logger.info("✓ SpyServer settings configured")
        
        # Test to_sdrconfig conversion
        sdrconfig = settings.sdr.to_sdrconfig()
        if sdrconfig is not None:
            logger.info("✓ SDRConfig conversion successful")
            logger.info(f"  Host: {sdrconfig.host}")
            logger.info(f"  Port: {sdrconfig.port}")
            logger.info(f"  Timeout: {sdrconfig.timeout}")
        else:
            logger.warning("⚠ SDRConfig conversion returned None (sdrconnect not available)")
        
        # Test device-specific settings
        device_settings = settings.get_device_settings()
        logger.info(f"✓ Device settings: {device_settings}")
        
        if 'host' in device_settings and 'port' in device_settings:
            logger.info("✓ SpyServer-specific settings present")
        else:
            logger.error("✗ SpyServer-specific settings missing")
            return False
        
        return True
    except Exception as e:
        logger.error(f"✗ Settings SDRConfig integration test failed: {e}")
        return False

def test_gui_controls():
    """Test GUI controls for SpyServer."""
    try:
        from rf_spectrum_analyzer.config.settings import Settings
        from rf_spectrum_analyzer.gui.controls_widget import ControlsWidget
        from PySide6.QtWidgets import QApplication
        
        # Create minimal Qt application
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        settings = Settings()
        controls = ControlsWidget(settings)
        
        logger.info("✓ Controls widget created")
        
        # Test SpyServer device selection
        controls.device_combo.setCurrentText("spyserver")
        
        # Check if SpyServer controls are visible
        if hasattr(controls, 'spyserver_group'):
            logger.info("✓ SpyServer controls group exists")
            
            # Test SpyServer configuration
            spyserver_config = controls.get_spyserver_config()
            logger.info(f"✓ SpyServer config: {spyserver_config}")
        else:
            logger.error("✗ SpyServer controls group not found")
            return False
        
        return True
    except Exception as e:
        logger.error(f"✗ GUI controls test failed: {e}")
        return False

def run_all_tests():
    """Run all integration tests."""
    logger.info("Starting SDRConnect integration tests...")
    logger.info("=" * 60)
    
    tests = [
        ("SDRConnect Availability", test_sdrconnect_availability),
        ("SpyServer Backend", test_spyserver_backend),
        ("Enhanced Analysis", test_enhanced_analysis),
        ("Signal Processor Integration", test_signal_processor_integration),
        ("SDR Backend Manager", test_sdr_backend_manager),
        ("Settings SDRConfig Integration", test_settings_sdrconfig_integration),
        ("GUI Controls", test_gui_controls),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        logger.info(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! SDRConnect integration successful.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Check logs for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)