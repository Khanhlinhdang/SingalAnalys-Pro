"""Backend runtime tests aligned with the current adapter contracts."""

import unittest
from pathlib import Path
import sys

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rf_spectrum_analyzer.config.settings import Settings
from rf_spectrum_analyzer.core.sdr_backend import SDRBackendManager
from rf_spectrum_analyzer.backends.rtlsdr_backend import RTLSDRBackend
from rf_spectrum_analyzer.backends.hackrf_backend import HackRFBackend
from rf_spectrum_analyzer.backends.pluto_backend import PlutoSDRBackend
from rf_spectrum_analyzer.backends.soapy_backend import SoapySDRBackend
from rf_spectrum_analyzer.backends.spyserver_backend import SpyServerBackend


class ContractMixin:
    def assert_result_envelope(self, result):
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("error", result)
        self.assertIn("payload", result)
        self.assertIn("meta", result)
        self.assertIsInstance(result["payload"], dict)
        self.assertIsInstance(result["meta"], dict)


class TestBackendDeviceInfoContracts(unittest.TestCase, ContractMixin):
    def setUp(self):
        self.settings = Settings()

    def test_device_info_disconnected_contract(self):
        backends = [
            RTLSDRBackend(self.settings),
            HackRFBackend(self.settings),
            PlutoSDRBackend(self.settings),
            SoapySDRBackend(self.settings),
            SpyServerBackend(self.settings),
        ]

        for backend in backends:
            info = backend.get_device_info()
            self.assert_result_envelope(info)
            self.assertIn("get_device_info", info["meta"].get("api", ""))


class TestBackendManagerRouting(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.manager = SDRBackendManager(self.settings)

    def test_set_device_type_selects_backend_instance(self):
        self.assertTrue(self.manager.set_device_type("spyserver"))
        self.assertIsNotNone(self.manager.current_backend)
        self.assertEqual(self.settings.sdr.device_type, "spyserver")

    def test_get_device_info_without_connection_returns_dict(self):
        self.manager.set_device_type("spyserver")
        info = self.manager.get_device_info()
        self.assertIsInstance(info, dict)
        self.assertIn("payload", info)

    def test_set_device_type_usrp_graceful_failure_when_backend_is_not_concrete(self):
        # Current USRP backend path is expected to fail gracefully if adapter is abstract-incomplete.
        result = self.manager.set_device_type("usrp")

        self.assertFalse(result)
        self.assertIsNone(self.manager.current_backend)
        self.assertEqual(self.settings.sdr.device_type, "usrp")

        # Manager should remain usable after failure by switching to a supported backend.
        self.assertTrue(self.manager.set_device_type("spyserver"))
        self.assertIsNotNone(self.manager.current_backend)


if __name__ == "__main__":
    unittest.main(verbosity=2)
