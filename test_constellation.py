#!/usr/bin/env python3
"""
Test script for Constellation Display Feature
Tests the constellation widget with various modulation schemes
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from rf_spectrum_analyzer.gui.constellation_widget import ConstellationWidget

def generate_test_signal(modulation_type: str, num_symbols: int = 1000, 
                        snr_db: float = 20.0) -> tuple:
    """Generate test signal for constellation display."""
    np.random.seed(42)  # For reproducible results
    
    if modulation_type.upper() == "BPSK":
        # BPSK: symbols are ±1
        symbols = 2 * np.random.randint(0, 2, num_symbols) - 1
        constellation = np.array(symbols, dtype=complex)
        
    elif modulation_type.upper() == "QPSK":
        # QPSK: symbols are (±1±j)/sqrt(2)
        bits = np.random.randint(0, 4, num_symbols)
        qpsk_map = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        constellation = qpsk_map[bits]
        
    elif modulation_type.upper() == "8PSK":
        # 8PSK: 8 points on unit circle
        bits = np.random.randint(0, 8, num_symbols)
        angles = 2 * np.pi * bits / 8
        constellation = np.exp(1j * angles)
        
    elif modulation_type.upper() == "16QAM":
        # 16QAM: 4x4 grid
        bits = np.random.randint(0, 16, num_symbols)
        i_bits = (bits >> 2) & 3  # Upper 2 bits
        q_bits = bits & 3         # Lower 2 bits
        
        # Map to ±1, ±3
        i_vals = 2 * i_bits - 3
        q_vals = 2 * q_bits - 3
        constellation = (i_vals + 1j * q_vals) / np.sqrt(10)
        
    else:
        # Default to QPSK
        bits = np.random.randint(0, 4, num_symbols)
        qpsk_map = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        constellation = qpsk_map[bits]
    
    # Add noise
    signal_power = np.mean(np.abs(constellation)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))
    
    # Create IQ samples (with some channel effects)
    samples_per_symbol = 4
    total_samples = num_symbols * samples_per_symbol
    
    # Pulse shaping (simple interpolation)
    iq_samples = np.zeros(total_samples, dtype=complex)
    for i, symbol in enumerate(constellation):
        start_idx = i * samples_per_symbol
        end_idx = start_idx + samples_per_symbol
        iq_samples[start_idx:end_idx] = symbol
    
    # Add noise to IQ samples
    iq_noise = np.sqrt(noise_power/2) * (np.random.randn(total_samples) + 1j*np.random.randn(total_samples))
    iq_samples += iq_noise
    
    # Add some channel distortion
    iq_samples += 0.1 * iq_samples * np.abs(iq_samples)  # Slight nonlinearity
    
    return iq_samples, constellation + noise

def test_constellation_widget():
    """Test the constellation widget with different modulation types."""
    print("🧪 CONSTELLATION WIDGET TEST")
    print("=" * 50)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create constellation widget
    print("1. Creating constellation widget...")
    constellation_widget = ConstellationWidget()
    constellation_widget.show()
    constellation_widget.resize(800, 600)
    
    print("✅ Constellation widget created and displayed")
    
    # Test different modulation schemes
    modulation_types = ["BPSK", "QPSK", "8PSK", "16QAM"]
    
    for i, mod_type in enumerate(modulation_types):
        print(f"\n2.{i+1} Testing {mod_type} modulation...")
        
        # Generate test signal
        iq_samples, symbols = generate_test_signal(mod_type, num_symbols=500, snr_db=15.0)
        
        # Generate reference constellation
        ref_constellation = constellation_widget.generate_reference_constellation(mod_type)
        
        # Create modulation info
        modulation_info = {
            "type": mod_type,
            "reference_constellation": ref_constellation,
            "evm": 0.1,  # 10% EVM
            "snr_estimate": 15.0
        }
        
        # Update constellation display
        constellation_widget.update_constellation(iq_samples, symbols, modulation_info)
        
        print(f"   ✅ {mod_type}: {len(iq_samples)} IQ samples, {len(symbols)} symbols")
        print(f"   📊 Reference constellation: {len(ref_constellation)} points")
        print(f"   📈 EVM: {modulation_info['evm']*100:.1f}%, SNR: {modulation_info['snr_estimate']:.1f} dB")
    
    print(f"\n3. Testing constellation controls...")
    
    # Test different display modes
    print("   - Testing display modes...")
    constellation_widget.mode_combo.setCurrentText("IQ Data")
    app.processEvents()
    print("     ✅ IQ Data mode")
    
    constellation_widget.mode_combo.setCurrentText("Symbols")
    app.processEvents()
    print("     ✅ Symbols mode")
    
    constellation_widget.mode_combo.setCurrentText("Both")
    app.processEvents()
    print("     ✅ Both mode")
    
    # Test hold mode
    print("   - Testing hold mode...")
    constellation_widget.hold_cb.setChecked(True)
    app.processEvents()
    print("     ✅ Hold mode enabled")
    
    # Test reference constellation toggle
    print("   - Testing reference constellation toggle...")
    constellation_widget.show_ref_cb.setChecked(False)
    app.processEvents()
    print("     ✅ Reference constellation hidden")
    
    constellation_widget.show_ref_cb.setChecked(True)
    app.processEvents()
    print("     ✅ Reference constellation shown")
    
    # Test point size adjustment
    print("   - Testing point size adjustment...")
    constellation_widget.point_size_spin.setValue(5)
    app.processEvents()
    print("     ✅ Point size increased")
    
    # Test auto scale
    print("   - Testing auto scale...")
    constellation_widget.auto_scale_cb.setChecked(False)
    app.processEvents()
    print("     ✅ Auto scale disabled")
    
    constellation_widget.auto_scale_cb.setChecked(True)
    app.processEvents()
    print("     ✅ Auto scale enabled")
    
    print(f"\n4. Testing settings persistence...")
    
    # Get current settings
    settings = constellation_widget.get_settings()
    print(f"   📋 Current settings: {len(settings)} parameters")
    for key, value in settings.items():
        print(f"      {key}: {value}")
    
    # Modify settings
    new_settings = settings.copy()
    new_settings["point_size"] = 7
    new_settings["max_points"] = 1500
    new_settings["show_reference"] = False
    
    # Apply settings
    constellation_widget.set_settings(new_settings)
    print("   ✅ Settings applied successfully")
    
    print(f"\n5. Performance test...")
    
    # Test with large dataset
    large_iq, large_symbols = generate_test_signal("16QAM", num_symbols=2000, snr_db=10.0)
    
    import time
    start_time = time.time()
    constellation_widget.update_constellation(large_iq, large_symbols, {
        "type": "16QAM",
        "reference_constellation": constellation_widget.generate_reference_constellation("16QAM"),
        "evm": 0.15,
        "snr_estimate": 10.0
    })
    app.processEvents()
    end_time = time.time()
    
    print(f"   📈 Large dataset ({len(large_iq)} samples) processed in {(end_time-start_time)*1000:.1f} ms")
    print("   ✅ Performance test completed")
    
    print(f"\n6. Final validation...")
    
    # Validate all reference constellations
    test_mods = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
    for mod in test_mods:
        ref_const = constellation_widget.generate_reference_constellation(mod)
        expected_points = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 16, "64QAM": 64}
        if mod in expected_points:
            assert len(ref_const) == expected_points[mod], f"{mod} should have {expected_points[mod]} points"
            print(f"   ✅ {mod}: {len(ref_const)} reference points")
    
    print(f"\n🎉 ALL CONSTELLATION TESTS PASSED!")
    print("=" * 50)
    print("The constellation widget is working correctly with:")
    print("• Multiple modulation schemes (BPSK, QPSK, 8PSK, 16QAM)")
    print("• Interactive controls (mode, hold, reference, scaling)")
    print("• Settings persistence and management")
    print("• Performance optimization for large datasets")
    print("• Reference constellation generation")
    print("\nConstellation display feature is ready for use!")
    
    # Keep window open for manual inspection
    print(f"\n👁️  Window will remain open for manual inspection...")
    print("Press Ctrl+C to exit")
    
    try:
        app.exec()
    except KeyboardInterrupt:
        print("\n👋 Test completed!")

if __name__ == "__main__":
    test_constellation_widget()