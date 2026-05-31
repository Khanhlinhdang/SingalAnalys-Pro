"""App-layer integration tests for analysis request to GUI update flow."""

import unittest
from pathlib import Path
import sys
import tempfile
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
            update_image_artifact_view=MagicMock(),
            show_info_message=MagicMock(),
            show_error_message=MagicMock(),
        )
        app.data_exporter = MagicMock()
        app.data_importer = MagicMock()
        app.latest_image_artifact = None
        app.latest_pcm_artifact = None
        app.analysis_session_records = []
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

    def test_update_gui_captures_noaa_image_artifact(self):
        app = self._build_app_stub()

        app._update_gui_with_analysis_results(
            {
                "modulation": {"type": "FM", "confidence": 0.88},
                "demodulation": {"success": True, "snr": 12.0},
                "coding": {"coding_type": "None", "decoded_bits": [1, 0, 1, 0]},
                "constellation_data": {"points": [[1.0, 0.0], [0.0, 1.0]]},
                "decoded_outputs": [
                    {
                        "type": "image",
                        "confidence": 0.9,
                        "payload": {
                            "protocol": "NOAA APT Baseline",
                            "summary": {"width": 64, "height": 8},
                            "image_matrix": [[0, 1], [2, 3]],
                        },
                    }
                ],
            }
        )

        self.assertIsNotNone(app.latest_image_artifact)
        app.main_window.spectrum_widget.info_label.setText.assert_called_once()
        app.main_window.update_image_artifact_view.assert_called_once()

    def test_export_latest_image_artifact_calls_exporter(self):
        app = self._build_app_stub()
        app.latest_image_artifact = {
            "type": "image",
            "confidence": 0.9,
            "payload": {
                "protocol": "NOAA APT Baseline",
                "summary": {"width": 64, "height": 8},
                "image_matrix": [[0, 1], [2, 3]],
            },
        }
        app.data_exporter.export_artifact_image.return_value = True

        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "noaa.png")
            app.export_latest_image_artifact(output)

        app.data_exporter.set_metadata.assert_called_once()
        app.data_exporter.export_artifact_image.assert_called_once()
        app.main_window.show_info_message.assert_called_once()

    def test_export_session_decode_report_calls_exporter(self):
        app = self._build_app_stub()
        app.analysis_session_records = [
            {
                "timestamp": "2026-05-31T12:00:00Z",
                "snr": 11.2,
                "decode_quality": {"artifact_count": 2, "ber": 0.12},
                "artifact_references": [{"type": "image", "protocol": "NOAA APT Baseline"}],
            }
        ]
        app.data_exporter.export_decode_session_report.return_value = True

        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "decode_report.json")
            app.export_session_decode_report(output)

        app.data_exporter.export_decode_session_report.assert_called_once()
        app.main_window.show_info_message.assert_called_once()

    def test_export_latest_pcm_artifact_calls_exporter(self):
        app = self._build_app_stub()
        app.latest_pcm_artifact = {
            "type": "pcm",
            "payload": {"sample_rate": 16000, "samples": [0, 100, -100]},
        }
        app.data_exporter.export_pcm_wav_from_artifact.return_value = True

        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "audio.wav")
            app.export_latest_pcm_artifact(output)

        app.data_exporter.export_pcm_wav_from_artifact.assert_called_once()
        app.main_window.show_info_message.assert_called_once()

    def test_load_session_decode_report_replays_image_preview(self):
        app = self._build_app_stub()
        app.main_window.clear_image_artifact_history = MagicMock()
        app.data_importer.import_decode_session_report.return_value = {
            "records": [
                {
                    "timestamp": "2026-05-31T12:00:00Z",
                    "artifact_references": [
                        {
                            "type": "image",
                            "protocol": "NOAA APT Baseline",
                            "width": 4,
                            "height": 2,
                            "preview_rows": [[1, 2, 3, 4], [5, 6, 7, 8]],
                        }
                    ],
                }
            ]
        }

        app.load_session_decode_report("session.json")

        app.data_importer.import_decode_session_report.assert_called_once()
        app.main_window.clear_image_artifact_history.assert_called_once()
        app.main_window.update_image_artifact_view.assert_called_once()
        app.main_window.show_info_message.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
