"""
DSP Modulation Module Tests
Comprehensive testing of all modulation and demodulation classes
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import warnings

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

try:
    from rf_spectrum_analyzer.dsp.modulation import (
        ModulationType, ModulationConfig, BaseModulator,
        PSKModulator, QAMModulator, FSKModulator, OFDMModulator,
        BaseDemodulator, PSKDemodulator,
        create_psk_modulator, create_qam_modulator,
        create_fsk_modulator, create_ofdm_modulator,
        plot_constellation, calculate_evm, estimate_snr
    )
    DSP_MODULATION_AVAILABLE = True
except ImportError as e:
    DSP_MODULATION_AVAILABLE = False
    print(f"DSP modulation not available: {e}")


class TestModulationType(unittest.TestCase):
    """Test ModulationType enum"""
    
    @unittest.skipUnless(DSP_MODULATION_AVAILABLE, "DSP modulation not available")
    def test_modulation_types(self):
        """Test all modulation type enums"""
        self.assertEqual(ModulationType.PSK.value, "PSK")
        self.assertEqual(ModulationType.QPSK.value, "QPSK")
        self.assertEqual(ModulationType.BPSK.value, "BPSK")
        self.assertEqual(ModulationType.QAM16.value, "16QAM")
        self.assertEqual(ModulationType.FSK.value, "FSK")
        self.assertEqual(ModulationType.OQPSK.value, "OQPSK")


class TestModulationConfig(unittest.TestCase):
    """Test ModulationConfig dataclass"""
    
    @unittest.skipUnless(DSP_MODULATION_AVAILABLE, "DSP modulation not available")
    def test_default_config(self):
        """Test default modulation configuration"""
        config = ModulationConfig()
        self.assertEqual(config.modulation_type, "bpsk")
        self.assertEqual(config.symbol_rate, 1e6)
        self.assertEqual(config.sample_rate, 10e6)
        self.assertEqual(config.constellation_size, 4)
        self.assertEqual(config.pulse_shape, "rrc")
        self.assertEqual(config.alpha, 0.35)
    
    @unittest.skipUnless(DSP_MODULATION_AVAILABLE, "DSP modulation not available")
    def test_custom_config(self):
        """Test custom modulation configuration"""
        config = ModulationConfig(
            modulation_type="qpsk",
            symbol_rate=2e6,
            sample_rate=20e6,
            constellation_size=16,
            pulse_shape="rc",
            alpha=0.5
        )
        self.assertEqual(config.modulation_type, "qpsk")
        self.assertEqual(config.symbol_rate, 2e6)
        self.assertEqual(config.sample_rate, 20e6)
        self.assertEqual(config.constellation_size, 16)
        self.assertEqual(config.pulse_shape, "rc")
        self.assertEqual(config.alpha, 0.5)


class TestPSKModulator(unittest.TestCase):
    """Test PSK Modulator implementation"""
    
    @unittest.skipUnless(DSP_MODULATION_AVAILABLE, "DSP modulation not available")
    def setUp(self):
        """Set up test environment"""
        self.sample_rate = 1e6
        self.symbol_rate = 100e3
    
    def test_bpsk_modulator_creation(self):
        """Test BPSK modulator creation"""
        bpsk_mod = PSKModulator(
            m=2,
            symbol_rate=self.symbol_rate,
            sample_rate=self.sample_rate
        )
        
        self.assertEqual(bpsk_mod.m, 2)
        self.assertEqual(len(bpsk_mod.constellation), 2)
        self.assertIsNotNone(bpsk_mod.pulse_filter)
    
    def test_qpsk_modulator_creation(self):
        """Test QPSK modulator creation"""
        qpsk_mod = PSKModulator(
            m=4,
            symbol_rate=self.symbol_rate,
            sample_rate=self.sample_rate
        )
        
        self.assertEqual(qpsk_mod.m, 4)
        self.assertEqual(len(qpsk_mod.constellation), 4)
    
    def test_8psk_modulator_creation(self):
        """Test 8-PSK modulator creation"""
        psk8_mod = PSKModulator(
            m=8,
            symbol_rate=self.symbol_rate,
            sample_rate=self.sample_rate
        )
        
        self.assertEqual(psk8_mod.m, 8)
        self.assertEqual(len(psk8_mod.constellation), 8)
    
    def test_psk_constellation_generation(self):
        """Test PSK constellation generation"""
        qpsk_mod = PSKModulator(m=4)
        constellation = qpsk_mod.constellation
        
        # Check constellation properties
        self.assertEqual(len(constellation), 4)
        # All points should have unit magnitude
        magnitudes = np.abs(constellation)
        np.testing.assert_array_almost_equal(magnitudes, np.ones(4))
    
    def test_psk_symbol_generation(self):
        """Test PSK symbol generation from bits"""
        bpsk_mod = PSKModulator(m=2)
        
        # Test with known bit sequence
        test_bits = np.array([0, 1, 0, 1, 1, 0])
        symbols = bpsk_mod.generate_symbols(test_bits)
        
        self.assertEqual(len(symbols), len(test_bits))
        
        # QPSK test
        qpsk_mod = PSKModulator(m=4)
        test_bits_qpsk = np.array([0, 0, 0, 1, 1, 0, 1, 1])  # 4 symbols
        symbols_qpsk = qpsk_mod.generate_symbols(test_bits_qpsk)
        
        self.assertEqual(len(symbols_qpsk), 4)
    
    def test_psk_modulation(self):
        """Test PSK modulation process"""
        qpsk_mod = PSKModulator(
            m=4,
            symbol_rate=self.symbol_rate,
            sample_rate=self.sample_rate
        )
        
        # Generate test symbols
        test_bits = np.random.randint(0, 2, 100)
        symbols = qpsk_mod.generate_symbols(test_bits)
        
        # Modulate
        modulated_signal = qpsk_mod.modulate(symbols)
        
        # Check output properties
        expected_length = len(symbols) * qpsk_mod.samples_per_symbol
        self.assertAlmostEqual(len(modulated_signal), expected_length, delta=100)
        self.assertTrue(np.iscomplexobj(modulated_signal))


class TestQAMModulator(unittest.TestCase):
    """Test QAM Modulator implementation"""
    
    @unittest.skipUnless(DSP_MODULATION_AVAILABLE, "DSP modulation not available")
    def test_qam16_modulator_creation(self):
        """Test 16-QAM modulator creation"""
        qam16_mod = QAMModulator(m=16)
        
        self.assertEqual(qam16_mod.m, 16)
        self.assertEqual(len(qam16_mod.constellation), 16)
    
    def test_qam64_modulator_creation(self):
        """Test 64-QAM modulator creation"""
        qam64_mod = QAMModulator(m=64)
        
        self.assertEqual(qam64_mod.m, 64)
        self.assertEqual(len(qam64_mod.constellation), 64)
    
    def test_qam_constellation_properties(self):
        """Test QAM constellation properties"""
        qam16_mod = QAMModulator(m=16)
        constellation = qam16_mod.constellation
        
        # Check constellation size
        self.assertEqual(len(constellation), 16)
        
        # Check power normalization (average power should be close to 1)
        avg_power = np.mean(np.abs(constellation)**2)
        self.assertAlmostEqual(avg_power, 1.0, places=2)
    
    def test_qam_symbol_generation(self):
        """Test QAM symbol generation"""
        qam16_mod = QAMModulator(m=16)
        
        # 16-QAM uses 4 bits per symbol
        test_bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])  # 2 symbols
        symbols = qam16_mod.generate_symbols(test_bits)
        
        self.assertEqual(len(symbols), 2)
    
    def test_invalid_qam_size(self):
        """Test invalid QAM constellation size"""
        # QAM size must be perfect square
        with self.assertRaises(ValueError):
            QAMModulator(m=15)  # Not a perfect square


class TestFSKModulator(unittest.TestCase):
    """Test FSK Modulator implementation"""
    
    @unittest.skipUnless(DSP_MODULATION_AVAILABLE, "DSP modulation not available")
    def test_binary_fsk_creation(self):
        """Test binary FSK modulator creation"""
        bfsk_mod = FSKModulator(m=2)
        
        self.assertEqual(bfsk_mod.m, 2)
        self.assertEqual(len(bfsk_mod.frequencies), 2)
    
    def test_4fsk_creation(self):
        """Test 4-FSK modulator creation"""
        fsk4_mod = FSKModulator(m=4, frequency_separation=50e3)
        
        self.assertEqual(fsk4_mod.m, 4)
        self.assertEqual(len(fsk4_mod.frequencies), 4)
        self.assertEqual(fsk4_mod.frequency_separation, 50e3)
    
    def test_fsk_frequency_generation(self):
        """Test FSK frequency generation"""
        fsk_mod = FSKModulator(m=4, frequency_separation=1000)
        frequencies = fsk_mod.frequencies
        
        # Check frequency spacing
        freq_diff = np.diff(frequencies)
        np.testing.assert_array_almost_equal(freq_diff, np.ones(3) * 1000)
    
    def test_fsk_symbol_generation(self):
        """Test FSK symbol generation"""
        fsk_mod = FSKModulator(m=4)
        
        test_bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])  # 4 symbols
        symbols = fsk_mod.generate_symbols(test_bits)
        
        self.assertEqual(len(symbols), 4)
        # Symbols should be frequency indices
        self.assertTrue(all(0 <= s < 4 for s in symbols))
    
    def test_fsk_modulation(self):
        """Test FSK modulation process"""
        fsk_mod = FSKModulator(
            m=2,
            symbol_rate=1e3,
            sample_rate=10e3
        )
        
        test_bits = np.array([0, 1, 0, 1])
        symbols = fsk_mod.generate_symbols(test_bits)
        modulated = fsk_mod.modulate(symbols)
        
        self.assertGreater(len(modulated), 0)
        self.assertTrue(np.iscomplexobj(modulated))


class TestOFDMModulator(unittest.TestCase):
    """Test OFDM Modulator implementation"""
    
    @unittest.skipUnless(DSP_MODULATION_AVAILABLE, "DSP modulation not available")
    def test_ofdm_creation(self):
        """Test OFDM modulator creation"""
        ofdm_mod = OFDMModulator(
            n_subcarriers=64,
            cp_length=16,
            subcarrier_modulation="qpsk"
        )
        
        self.assertEqual(ofdm_mod.n_subcarriers, 64)
        self.assertEqual(ofdm_mod.cp_length, 16)
        self.assertEqual(ofdm_mod.subcarrier_modulation, "qpsk")
    
    def test_ofdm_with_qam16(self):
        """Test OFDM with 16-QAM subcarriers"""
        ofdm_mod = OFDMModulator(
            n_subcarriers=32,
            cp_length=8,
            subcarrier_modulation="qam16"
        )
        
        self.assertIsNotNone(ofdm_mod.subcarrier_mod)
    
    def test_ofdm_modulation(self):
        """Test OFDM modulation process"""
        ofdm_mod = OFDMModulator(
            n_subcarriers=16,
            cp_length=4,
            subcarrier_modulation="qpsk"
        )
        
        # Generate enough data for one OFDM symbol
        # 16 subcarriers * 2 bits per QPSK symbol = 32 bits
        test_data = np.random.randint(0, 2, 32)
        
        modulated = ofdm_mod.modulate(test_data)
        
        # Output should include cyclic prefix
        expected_length = ofdm_mod.n_subcarriers + ofdm_mod.cp_length
        self.assertEqual(len(modulated), expected_length)


class TestPSKDemodulator(unittest.TestCase):
    """Test PSK Demodulator implementation"""
    
    @unittest.skipUnless(DSP_MODULATION_AVAILABLE, "DSP modulation not available")
    def test_psk_demodulator_creation(self):
        """Test PSK demodulator creation"""
        psk_demod = PSKDemodulator(m=4)
        
        self.assertEqual(psk_demod.m, 4)
        self.assertEqual(len(psk_demod.constellation), 4)
    
    def test_symbol_detection(self):
        """Test PSK symbol detection"""
        psk_demod = PSKDemodulator(m=4)
        
        # Create test symbols (noisy constellation points)
        clean_symbols = psk_demod.constellation
        noise = 0.1 * (np.random.randn(4) + 1j * np.random.randn(4))
        noisy_symbols = clean_symbols + noise
        
        detected = psk_demod.detect_symbols(noisy_symbols)
        
        # Should detect correct symbols
        expected = np.arange(4)
        np.testing.assert_array_equal(detected, expected)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience modulator creation functions"""
    
    @unittest.skipUnless(DSP_MODULATION_AVAILABLE, "DSP modulation not available")
    def test_create_psk_modulator(self):
        """Test PSK modulator creation function"""
        psk_mod = create_psk_modulator(m=4, symbol_rate=1e6, sample_rate=10e6)
        
        self.assertIsInstance(psk_mod, PSKModulator)
        self.assertEqual(psk_mod.m, 4)
    
    def test_create_qam_modulator(self):
        """Test QAM modulator creation function"""
        qam_mod = create_qam_modulator(m=16, symbol_rate=2e6, sample_rate=20e6)
        
        self.assertIsInstance(qam_mod, QAMModulator)
        self.assertEqual(qam_mod.m, 16)
    
    def test_create_fsk_modulator(self):
        """Test FSK modulator creation function"""
        fsk_mod = create_fsk_modulator(m=2, symbol_rate=1e6, sample_rate=10e6)
        
        self.assertIsInstance(fsk_mod, FSKModulator)
        self.assertEqual(fsk_mod.m, 2)
    
    def test_create_ofdm_modulator(self):
        """Test OFDM modulator creation function"""
        ofdm_mod = create_ofdm_modulator(n_subcarriers=64, cp_length=16)
        
        self.assertIsInstance(ofdm_mod, OFDMModulator)
        self.assertEqual(ofdm_mod.n_subcarriers, 64)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for modulation analysis"""
    
    @unittest.skipUnless(DSP_MODULATION_AVAILABLE, "DSP modulation not available")
    def test_calculate_evm(self):
        """Test EVM calculation"""
        # Create reference symbols
        reference = np.array([1+0j, 0+1j, -1+0j, 0-1j])
        
        # Add small error
        measured = reference + 0.1 * (np.random.randn(4) + 1j * np.random.randn(4))
        
        evm = calculate_evm(reference, measured)
        
        # EVM should be reasonable (< 50% for this noise level)
        self.assertLess(evm, 50.0)
        self.assertGreater(evm, 0.0)
    
    def test_estimate_snr(self):
        """Test SNR estimation"""
        # Create constellation
        constellation = np.array([1+0j, -1+0j, 0+1j, 0-1j])
        
        # Create noisy symbols
        clean_symbols = np.repeat(constellation, 10)
        noise = 0.2 * (np.random.randn(40) + 1j * np.random.randn(40))
        noisy_symbols = clean_symbols + noise
        
        snr = estimate_snr(noisy_symbols, constellation)
        
        # SNR should be reasonable
        self.assertGreater(snr, 0)
        self.assertLess(snr, 50)  # Not infinite SNR


class TestPulseShaping(unittest.TestCase):
    """Test pulse shaping filters"""
    
    @unittest.skipUnless(DSP_MODULATION_AVAILABLE, "DSP modulation not available")
    def test_rrc_filter_design(self):
        """Test RRC filter design"""
        psk_mod = PSKModulator(m=2, pulse_shape="rrc", alpha=0.35, span=6)
        
        # Check filter properties
        self.assertIsNotNone(psk_mod.pulse_filter)
        self.assertGreater(len(psk_mod.pulse_filter), 1)
    
    def test_rc_filter_design(self):
        """Test RC filter design"""
        psk_mod = PSKModulator(m=2, pulse_shape="rc", alpha=0.5, span=8)
        
        self.assertIsNotNone(psk_mod.pulse_filter)
    
    def test_rectangular_pulse(self):
        """Test rectangular pulse"""
        psk_mod = PSKModulator(m=2, pulse_shape="rect")
        
        # Rectangular pulse should have length equal to samples per symbol
        expected_length = psk_mod.samples_per_symbol
        self.assertEqual(len(psk_mod.pulse_filter), expected_length)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    @unittest.skipUnless(DSP_MODULATION_AVAILABLE, "DSP modulation not available")
    def test_empty_data_modulation(self):
        """Test modulation with empty data"""
        psk_mod = PSKModulator(m=2)
        
        empty_data = np.array([])
        symbols = psk_mod.generate_symbols(empty_data)
        
        self.assertEqual(len(symbols), 0)
    
    def test_mismatched_signal_lengths_evm(self):
        """Test EVM calculation with mismatched lengths"""
        ref = np.array([1, 0, -1])
        measured = np.array([1.1, 0.1])  # Shorter
        
        # Should handle length mismatch gracefully
        evm = calculate_evm(ref, measured)
        self.assertIsInstance(evm, float)
    
    def test_zero_power_signals(self):
        """Test handling of zero power signals"""
        ref = np.zeros(10)
        measured = np.ones(10) * 0.1
        
        # Should handle zero reference power
        evm = calculate_evm(ref, measured)
        # EVM should be infinite or very large
        self.assertTrue(np.isinf(evm) or evm > 1000)


if __name__ == '__main__':
    unittest.main(verbosity=2)