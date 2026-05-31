"""Core runtime tests aligned with current app contracts."""

import unittest
from pathlib import Path
import sys

import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rf_spectrum_analyzer.config.settings import Settings
from rf_spectrum_analyzer.core.signal_processor import SignalProcessor
from rf_spectrum_analyzer.core.sdr_backend import SDRBackendManager, SDRDeviceType


class TestSignalProcessorRuntime(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.settings.dsp.fft_size = 1024
        self.processor = SignalProcessor(self.settings)

    def test_compute_spectrum_shape_and_finite(self):
        t = np.arange(self.settings.dsp.fft_size) / self.settings.sdr.sample_rate
        iq = np.exp(2j * np.pi * 20_000 * t).astype(np.complex64)

        spectrum = self.processor.compute_spectrum(iq)

        self.assertIsNotNone(spectrum)
        self.assertEqual(len(spectrum), self.settings.dsp.fft_size)
        self.assertTrue(np.isfinite(spectrum).all())

    def test_process_complete_chain_empty_contract(self):
        result = self.processor.process_complete_chain(np.array([]))

        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("error", result)
        self.assertIn("payload", result)
        self.assertIn("meta", result)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Empty signal")

    def test_detection_flags_can_be_toggled(self):
        self.processor.set_auto_detection(True)
        self.processor.set_advanced_analysis(True)

        self.assertTrue(self.processor._auto_detection_enabled)
        self.assertTrue(self.processor._advanced_analysis_enabled)


class TestSDRBackendManagerRuntime(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.manager = SDRBackendManager(self.settings)

    def test_available_device_types_include_runtime_backends(self):
        available = set(self.manager.get_available_devices())
        expected = {
            SDRDeviceType.RTLSDR.value,
            SDRDeviceType.HACKRF.value,
            SDRDeviceType.PLUTO.value,
            SDRDeviceType.SOAPY.value,
            SDRDeviceType.SPYSERVER.value,
            SDRDeviceType.FILE.value,
        }
        self.assertTrue(expected.issubset(available))

    def test_backend_property_tracks_current_backend(self):
        ok = self.manager.set_device_type("spyserver")

        self.assertTrue(ok)
        self.assertIsNotNone(self.manager.backend)
        self.assertEqual(self.manager.backend, self.manager.current_backend)


if __name__ == "__main__":
    unittest.main(verbosity=2)
