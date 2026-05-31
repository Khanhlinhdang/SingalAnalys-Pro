"""Tests for output adapter extraction pipeline."""

import unittest
from pathlib import Path
import sys
import tempfile
import json

import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rf_spectrum_analyzer.dsp.output_adapters import extract_all_artifacts, normalize_artifact_contracts
from rf_spectrum_analyzer.utils.file_io import DataExporter, DataImporter


class TestOutputAdapters(unittest.TestCase):
    def test_extract_text_artifact_from_bits(self):
        text = "HELLO RF WORLD"
        bit_stream = np.unpackbits(np.frombuffer(text.encode("utf-8"), dtype=np.uint8))

        artifacts = extract_all_artifacts(
            decoded_bits=bit_stream,
            demodulated_audio=None,
            sample_rate=48_000,
        )

        types = {a["type"] for a in artifacts}
        self.assertIn("text", types)

    def test_extract_mail_artifact_from_bits(self):
        msg = (
            "From: sat@example.org\n"
            "To: ground@example.org\n"
            "Subject: TEST LINK\n\n"
            "payload"
        )
        bit_stream = np.unpackbits(np.frombuffer(msg.encode("utf-8"), dtype=np.uint8))

        artifacts = extract_all_artifacts(
            decoded_bits=bit_stream,
            demodulated_audio=None,
            sample_rate=48_000,
        )

        types = {a["type"] for a in artifacts}
        self.assertIn("text", types)
        self.assertIn("mail", types)

    def test_extract_audio_artifact_from_array(self):
        t = np.linspace(0, 0.2, 9600, endpoint=False)
        audio = 0.4 * np.sin(2 * np.pi * 1000 * t)

        artifacts = extract_all_artifacts(
            decoded_bits=None,
            demodulated_audio=audio,
            sample_rate=48_000,
        )

        types = {a["type"] for a in artifacts}
        self.assertIn("audio", types)
        self.assertIn("pcm", types)

        pcm = next(a for a in artifacts if a["type"] == "pcm")
        self.assertEqual(pcm["payload"].get("encoding"), "pcm_s16le")
        self.assertIn("samples", pcm["payload"])

    def test_normalize_image_video_contracts(self):
        raw = [
            {"type": "image", "confidence": 0.7, "payload": {"summary": {"width": 16, "height": 8}}},
            {"type": "video", "confidence": 0.5, "payload": {"frame_count": 12}},
        ]

        normalized = normalize_artifact_contracts(raw)
        image = next(a for a in normalized if a["type"] == "image")
        video = next(a for a in normalized if a["type"] == "video")

        self.assertEqual(image["payload"].get("artifact_kind"), "image")
        self.assertEqual(image["payload"].get("media_class"), "image")
        self.assertEqual(image["payload"].get("width"), 16)
        self.assertEqual(video["payload"].get("artifact_kind"), "video")
        self.assertEqual(video["payload"].get("media_class"), "video")

    def test_export_image_artifact_json_and_npy(self):
        exporter = DataExporter()
        artifact = {
            "type": "image",
            "confidence": 0.9,
            "payload": {
                "protocol": "NOAA APT Baseline",
                "summary": {"width": 4, "height": 3},
                "image_matrix": [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]],
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "image_artifact.json"
            out_npy = Path(tmp) / "image_artifact.npy"

            self.assertTrue(exporter.export_artifact_image(artifact, str(out_json), format="json"))
            self.assertTrue(exporter.export_artifact_image(artifact, str(out_npy), format="npy"))

            self.assertTrue(out_json.exists())
            self.assertTrue(out_npy.exists())

            with open(out_json, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            self.assertIn("artifact", parsed)

            arr = np.load(out_npy)
            self.assertEqual(arr.shape, (3, 4))

    def test_export_decode_session_report_json(self):
        exporter = DataExporter()
        records = [
            {
                "timestamp": "2026-05-31T12:00:00Z",
                "snr": 12.0,
                "decode_quality": {
                    "artifact_count": 2,
                    "frame_count": 3,
                    "ber": 0.05,
                    "per": 0.1,
                    "crc_ok_rate": 0.8,
                    "frame_lock_ratio": 0.66,
                },
                "artifact_references": [
                    {"type": "image", "protocol": "NOAA APT Baseline", "width": 128, "height": 64}
                ],
            },
            {
                "timestamp": "2026-05-31T12:00:05Z",
                "snr": 10.5,
                "decode_quality": {
                    "artifact_count": 1,
                    "frame_count": 2,
                    "ber": 0.09,
                    "per": 0.2,
                    "crc_ok_rate": 0.5,
                    "frame_lock_ratio": 0.5,
                },
                "artifact_references": [
                    {"type": "protocol_packet_log", "protocol": "Meteor LRPT Baseline"}
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "decode_session_report.json"
            self.assertTrue(exporter.export_decode_session_report(records, str(out_json)))
            self.assertTrue(out_json.exists())

            with open(out_json, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(payload.get("record_count"), 2)
            self.assertIn("trends", payload)
            self.assertIn("records", payload)
            self.assertEqual(len(payload.get("trends", {}).get("snr", [])), 2)

    def test_export_pcm_wav_and_import_decode_report(self):
        exporter = DataExporter()
        importer = DataImporter()

        pcm_artifact = {
            "type": "pcm",
            "payload": {
                "sample_rate": 16000,
                "channels": 1,
                "samples": [0, 1000, -1000, 32767, -32768, 10, -10],
            },
        }
        report_payload = {
            "record_count": 1,
            "records": [{"timestamp": "2026-05-31T12:00:00Z", "artifact_references": []}],
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_wav = Path(tmp) / "decoded_audio.wav"
            out_report = Path(tmp) / "decode_session_report.json"

            self.assertTrue(exporter.export_pcm_wav_from_artifact(pcm_artifact, str(out_wav)))
            self.assertTrue(out_wav.exists())

            with open(out_report, "w", encoding="utf-8") as f:
                json.dump(report_payload, f)
            loaded = importer.import_decode_session_report(str(out_report))
            self.assertEqual(loaded.get("record_count"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
