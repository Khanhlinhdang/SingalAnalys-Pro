"""Contract tests for versioned dict-based API schema."""

import unittest
from pathlib import Path
import sys

import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rf_spectrum_analyzer.config.settings import Settings
from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer
from rf_spectrum_analyzer.core.signal_processor import SignalProcessor
from rf_spectrum_analyzer.backends.rtlsdr_backend import RTLSDRBackend
from rf_spectrum_analyzer.backends.hackrf_backend import HackRFBackend
from rf_spectrum_analyzer.backends.pluto_backend import PlutoSDRBackend
from rf_spectrum_analyzer.backends.soapy_backend import SoapySDRBackend
from rf_spectrum_analyzer.backends.spyserver_backend import SpyServerBackend
from rf_spectrum_analyzer.utils.schema import API_SCHEMA_VERSION


class ContractMixin:
    def assert_contract(self, result: dict):
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("error", result)
        self.assertIn("payload", result)
        self.assertIn("meta", result)
        self.assertIsInstance(result["payload"], dict)
        self.assertIsInstance(result["meta"], dict)
        self.assertEqual(result["meta"].get("schema_version"), API_SCHEMA_VERSION)


class TestSignalAnalyzerContract(unittest.TestCase, ContractMixin):
    def setUp(self):
        self.analyzer = SignalAnalyzer(sample_rate=1e6)

    def test_analyze_signal_comprehensive_success_contract(self):
        t = np.arange(0, 0.002, 1 / 1e6)
        iq_data = np.exp(1j * 2 * np.pi * 15e3 * t)

        result = self.analyzer.analyze_signal_comprehensive(iq_data, center_freq=100e6, bandwidth=1e6)

        self.assert_contract(result)
        self.assertTrue(result["success"])
        self.assertEqual(result["analysis_status"], "success")
        self.assertIn("signal_info", result["payload"])
        self.assertIn("modulation", result["payload"])
        self.assertEqual(result["meta"].get("api"), "SignalAnalyzer.analyze_signal_comprehensive")

    def test_analyze_signal_comprehensive_error_contract(self):
        result = self.analyzer.analyze_signal_comprehensive(np.array([]), center_freq=100e6, bandwidth=1e6)

        self.assert_contract(result)
        self.assertFalse(result["success"])
        self.assertEqual(result["analysis_status"], "failed")
        self.assertEqual(result["error"], "No IQ data provided")


class TestBackendDeviceInfoContract(unittest.TestCase, ContractMixin):
    def setUp(self):
        self.settings = Settings()

    def test_backend_get_device_info_contract_disconnected(self):
        backends = [
            RTLSDRBackend(self.settings),
            HackRFBackend(self.settings),
            PlutoSDRBackend(self.settings),
            SoapySDRBackend(self.settings),
            SpyServerBackend(self.settings),
        ]

        for backend in backends:
            info = backend.get_device_info()
            self.assert_contract(info)
            self.assertIn("get_device_info", info["meta"].get("api", ""))


class TestSignalProcessorContract(unittest.TestCase, ContractMixin):
    def setUp(self):
        self.settings = Settings()
        self.processor = SignalProcessor(self.settings)

    def test_process_complete_chain_error_contract(self):
        result = self.processor.process_complete_chain(np.array([]))

        self.assert_contract(result)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Empty signal")
        self.assertIn("modulation_analysis", result["payload"])


if __name__ == "__main__":
    unittest.main()
