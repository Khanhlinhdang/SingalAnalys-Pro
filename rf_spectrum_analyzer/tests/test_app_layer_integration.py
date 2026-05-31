"""App-layer integration tests for analysis request to GUI update flow."""

import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rf_spectrum_analyzer.core.app import RFSpectrumAnalyzerApp


class TestAppLayerAnalysisFlow(unittest.TestCase):
    def _build_app_stub(self):
        app = RFSpectrumAnalyzerApp.__new__(RFSpectrumAnalyzerApp)
        app.logger = MagicMock()

        app.main_window = SimpleNamespace(
            constellation_widget=SimpleNamespace(update_constellation=MagicMock()),
            bitstream_widget=SimpleNamespace(add_bits=MagicMock()),
            spectrum_widget=SimpleNamespace(info_label=SimpleNamespace(setText=MagicMock())),
        )
        return app

    def test_handle_signal_analysis_request_updates_gui_with_payload(self):
        app = self._build_app_stub()

        iq_data = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        app._get_iq_data_for_range = MagicMock(return_value=iq_data)

        app.signal_analyzer = SimpleNamespace(
            analyze_signal_comprehensive=MagicMock(
                return_value={
                    "success": True,
                    "error": None,
                    "payload": {
                        "modulation": {"type": "QPSK", "confidence": 0.97},
                        "demodulation": {"success": True, "snr": 14.2},
                        "coding": {"coding_type": "None", "decoded_bits": [1, 0, 1, 1]},
                        "constellation_data": {"points": [[1.0, 1.0], [-1.0, -1.0]]},
                    },
                    "meta": {"schema_version": "1.0"},
                }
            )
        )

        request = {
            "center_freq": 100e6,
            "bandwidth": 100e3,
            "analysis_type": "full",
        }

        app.handle_signal_analysis_request(request)

        app.main_window.constellation_widget.update_constellation.assert_called_once()
        app.main_window.bitstream_widget.add_bits.assert_called_once_with([1, 0, 1, 1])
        app.main_window.spectrum_widget.info_label.setText.assert_called_once()

    def test_handle_signal_analysis_request_skips_gui_update_on_error_envelope(self):
        app = self._build_app_stub()

        iq_data = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        app._get_iq_data_for_range = MagicMock(return_value=iq_data)

        app.signal_analyzer = SimpleNamespace(
            analyze_signal_comprehensive=MagicMock(
                return_value={
                    "success": False,
                    "error": "analysis failed",
                    "payload": {},
                    "meta": {"schema_version": "1.0"},
                }
            )
        )

        request = {
            "center_freq": 100e6,
            "bandwidth": 100e3,
            "analysis_type": "full",
        }

        app.handle_signal_analysis_request(request)

        app.main_window.constellation_widget.update_constellation.assert_not_called()
        app.main_window.bitstream_widget.add_bits.assert_not_called()
        app.main_window.spectrum_widget.info_label.setText.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
