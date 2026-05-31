"""Unit tests for protocol plugin registry and Inmarsat parser adapter."""

import unittest
from pathlib import Path
import sys

import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rf_spectrum_analyzer.dsp.protocol_plugins import (
    InmarsatProtocolPlugin,
    IridiumBurstProtocolPlugin,
    MeteorLrptProtocolPlugin,
    NoaaAptProtocolPlugin,
    create_default_protocol_registry,
)


class TestProtocolPlugins(unittest.TestCase):
    def _build_noaa_apt_like_bits(self) -> np.ndarray:
        plugin = NoaaAptProtocolPlugin()
        lines = []
        for line_idx in range(8):
            payload_bytes = ((np.arange((plugin.LINE_BITS - len(plugin.SYNC_PATTERN)) // 8) * 7) + line_idx * 11) % 256
            payload_bits = np.unpackbits(payload_bytes.astype(np.uint8))
            line_bits = np.concatenate([plugin.SYNC_PATTERN, payload_bits])
            lines.append(line_bits.astype(np.uint8))
        return np.concatenate(lines)

    def _build_noaa_apt_like_bits_with_sync_errors(self) -> np.ndarray:
        plugin = NoaaAptProtocolPlugin()
        bits = self._build_noaa_apt_like_bits().copy()
        # Add small sync corruption: one flipped bit per line, still within tolerance.
        for line_idx in range(6):
            start = line_idx * plugin.LINE_BITS
            flip = start + (line_idx % len(plugin.SYNC_PATTERN))
            bits[flip] ^= np.uint8(1)
        return bits

    def _build_meteor_like_bits(self) -> np.ndarray:
        plugin = MeteorLrptProtocolPlugin()
        rng = np.random.default_rng(777)
        frame1 = np.concatenate([plugin.SYNC_WORD, rng.integers(0, 2, plugin.FRAME_BITS, dtype=np.uint8)])
        gap = np.zeros(20, dtype=np.uint8)
        frame2 = np.concatenate([plugin.SYNC_WORD, rng.integers(0, 2, plugin.FRAME_BITS, dtype=np.uint8)])
        return np.concatenate([frame1, gap, frame2])

    def _build_meteor_burst_loss_bits(self) -> np.ndarray:
        plugin = MeteorLrptProtocolPlugin()
        rng = np.random.default_rng(919)
        frame1_payload = rng.integers(0, 2, plugin.FRAME_BITS, dtype=np.uint8)
        frame2_payload = rng.integers(0, 2, plugin.FRAME_BITS, dtype=np.uint8)

        # Simulate burst loss by zeroing a contiguous payload segment in frame 1.
        frame1_payload[280:360] = 0
        frame1 = np.concatenate([plugin.SYNC_WORD, frame1_payload])

        # Simulate truncated second frame (dropped tail during capture).
        frame2_partial = np.concatenate([plugin.SYNC_WORD, frame2_payload[: plugin.FRAME_BITS // 2]])
        gap = np.zeros(36, dtype=np.uint8)
        return np.concatenate([frame1, gap, frame2_partial])

    def _build_iridium_like_bits(self) -> np.ndarray:
        rng = np.random.default_rng(123)
        burst1 = rng.integers(0, 2, 180, dtype=np.uint8)
        gap = np.zeros(32, dtype=np.uint8)
        burst2 = rng.integers(0, 2, 320, dtype=np.uint8)
        return np.concatenate([burst1, gap, burst2])

    def _build_inmarsat_like_bits(self, message: str) -> np.ndarray:
        plugin = InmarsatProtocolPlugin()
        descriptor_bits = np.unpackbits(np.array([0x14], dtype=np.uint8))
        message_bytes = message.encode("ascii")

        payload_len = plugin.FRAME_LENGTH - len(descriptor_bits) - plugin.CRC_BITS
        payload = np.zeros(payload_len, dtype=np.uint8)
        message_bits = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))
        copy_len = min(payload_len, len(message_bits))
        payload[:copy_len] = message_bits[:copy_len]

        frame_without_crc = np.concatenate([descriptor_bits, payload])
        frame_without_crc_bytes = np.packbits(frame_without_crc).tobytes()
        crc_value = plugin._crc16_ccitt_false(frame_without_crc_bytes)
        crc_bits = np.unpackbits(np.frombuffer(crc_value.to_bytes(2, byteorder="big"), dtype=np.uint8))

        frame = np.concatenate([descriptor_bits, payload, crc_bits])
        scrambled_frame = plugin._descramble(frame)
        return np.concatenate([plugin.SYNC_WORD, scrambled_frame])

    def test_registry_selects_inmarsat_plugin(self):
        bits = self._build_inmarsat_like_bits("INMARSAT TEST MESSAGE")
        registry = create_default_protocol_registry()

        result = registry.decode(
            bits=bits,
            modulation_type="BPSK",
            sample_rate=48000,
            center_freq=1542e6,
        )

        self.assertEqual(result.get("matched_protocol"), "inmarsat_c")
        self.assertGreater(result.get("confidence", 0.0), 0.25)
        self.assertGreater(len(result.get("results", [])), 0)
        self.assertIsNotNone(result.get("ber"))
        self.assertIsNotNone(result.get("per"))
        self.assertIsNotNone(result.get("frame_lock_ratio"))
        self.assertEqual(result.get("counter_source"), "plugin_internal")
        self.assertTrue(result.get("crc_available"))
        self.assertEqual(result.get("crc_ok_rate"), 1.0)

    def test_inmarsat_plugin_emits_protocol_text_artifact(self):
        bits = self._build_inmarsat_like_bits("HELLO SATCOM")
        plugin = InmarsatProtocolPlugin()

        result = plugin.decode(
            type("Request", (), {
                "bits": bits,
                "modulation_type": "BPSK",
                "sample_rate": 48000,
                "center_freq": 1542e6,
            })()
        )

        artifacts = result.get("artifacts", [])
        self.assertTrue(any(a.get("type") == "protocol_text" for a in artifacts))
        self.assertTrue(result.get("crc_available"))
        self.assertEqual(result.get("crc_ok_rate"), 1.0)

    def test_registry_selects_iridium_plugin_for_lband_burst_stream(self):
        bits = self._build_iridium_like_bits()
        registry = create_default_protocol_registry()

        result = registry.decode(
            bits=bits,
            modulation_type="QPSK",
            sample_rate=250000,
            center_freq=1621.0e6,
        )

        self.assertEqual(result.get("matched_protocol"), "iridium_burst")
        self.assertGreater(len(result.get("results", [])), 0)
        self.assertIn("ber", result)
        self.assertIn("per", result)
        self.assertIn("crc_ok_rate", result)
        self.assertIn("frame_lock_ratio", result)
        self.assertEqual(result.get("counter_source"), "plugin_internal")
        self.assertFalse(result.get("crc_available"))
        self.assertIsNone(result.get("crc_ok_rate"))

    def test_iridium_plugin_emits_burst_log_artifact(self):
        bits = self._build_iridium_like_bits()
        plugin = IridiumBurstProtocolPlugin()

        result = plugin.decode(
            type("Request", (), {
                "bits": bits,
                "modulation_type": "QPSK",
                "sample_rate": 250000,
                "center_freq": 1621.0e6,
            })()
        )

        artifacts = result.get("artifacts", [])
        self.assertTrue(any(a.get("type") == "protocol_burst_log" for a in artifacts))

    def test_registry_selects_meteor_plugin_for_vhf_lrpt_stream(self):
        bits = self._build_meteor_like_bits()
        registry = create_default_protocol_registry()

        result = registry.decode(
            bits=bits,
            modulation_type="OQPSK",
            sample_rate=72000,
            center_freq=137.1e6,
        )

        self.assertEqual(result.get("matched_protocol"), "meteor_lrpt")
        self.assertGreater(len(result.get("results", [])), 0)
        self.assertIn("ber", result)
        self.assertIn("per", result)
        self.assertIn("crc_ok_rate", result)
        self.assertIn("frame_lock_ratio", result)
        self.assertEqual(result.get("counter_source"), "plugin_internal")
        self.assertIn("deinterleave_depth", result)
        self.assertIn("viterbi_metric_avg", result)
        self.assertIn("viterbi_soft_metric_avg", result)
        self.assertIn("viterbi_soft_confidence_avg", result)
        self.assertIn("viterbi_soft_path_rate", result)
        self.assertIn("rs_corrected_symbols_avg", result)
        self.assertIn("rs_decode_success_rate", result)
        self.assertEqual(result.get("decode_chain_depth"), "frame_sync->deinterleave->viterbi->reed_solomon")

    def test_meteor_plugin_emits_packet_log_artifact(self):
        bits = self._build_meteor_like_bits()
        plugin = MeteorLrptProtocolPlugin()

        result = plugin.decode(
            type("Request", (), {
                "bits": bits,
                "modulation_type": "QPSK",
                "sample_rate": 72000,
                "center_freq": 137.9e6,
            })()
        )

        artifacts = result.get("artifacts", [])
        self.assertTrue(any(a.get("type") == "protocol_packet_log" for a in artifacts))
        self.assertTrue(all("decode_chain" in item for item in result.get("results", [])))

    def test_registry_selects_noaa_apt_for_fm_vhf_stream(self):
        bits = self._build_noaa_apt_like_bits()
        registry = create_default_protocol_registry()

        result = registry.decode(
            bits=bits,
            modulation_type="FM",
            sample_rate=41600,
            center_freq=137.62e6,
        )

        self.assertEqual(result.get("matched_protocol"), "noaa_apt")
        self.assertGreater(len(result.get("results", [])), 0)
        self.assertEqual(result.get("counter_source"), "plugin_internal")
        self.assertIsNotNone(result.get("frame_lock_ratio"))

    def test_noaa_sync_tolerates_small_pattern_errors(self):
        bits = self._build_noaa_apt_like_bits_with_sync_errors()
        plugin = NoaaAptProtocolPlugin()

        result = plugin.decode(
            type("Request", (), {
                "bits": bits,
                "modulation_type": "FM",
                "sample_rate": 41600,
                "center_freq": 137.62e6,
            })()
        )

        self.assertGreater(len(result.get("results", [])), 0)
        self.assertIn("line_sync_stage", result)
        self.assertEqual(result.get("line_sync_stage", {}).get("max_sync_errors"), plugin.MAX_SYNC_ERRORS)
        image_artifact = next((a for a in result.get("artifacts", []) if a.get("type") == "image"), {})
        self.assertIn("line_sync_errors", image_artifact.get("payload", {}))

    def test_meteor_burst_loss_pattern_preserves_contract(self):
        bits = self._build_meteor_burst_loss_bits()
        plugin = MeteorLrptProtocolPlugin()

        result = plugin.decode(
            type("Request", (), {
                "bits": bits,
                "modulation_type": "QPSK",
                "sample_rate": 72000,
                "center_freq": 137.4e6,
            })()
        )

        self.assertIn("ber", result)
        self.assertIn("per", result)
        self.assertIn("crc_ok_rate", result)
        self.assertIn("frame_lock_ratio", result)
        self.assertIn("counter_source", result)
        self.assertIn("viterbi_soft_metric_avg", result)
        self.assertIn("viterbi_soft_confidence_avg", result)
        self.assertIn("viterbi_soft_path_rate", result)

    def test_noaa_apt_plugin_emits_image_artifact(self):
        bits = self._build_noaa_apt_like_bits()
        plugin = NoaaAptProtocolPlugin()

        result = plugin.decode(
            type("Request", (), {
                "bits": bits,
                "modulation_type": "WFM",
                "sample_rate": 41600,
                "center_freq": 137.5e6,
            })()
        )

        artifacts = result.get("artifacts", [])
        artifact_types = {a.get("type") for a in artifacts}
        self.assertIn("image", artifact_types)
        self.assertIn("audio", artifact_types)
        self.assertIn("fm_demod_stage", result)
        self.assertIn("line_sync_stage", result)

        image_artifact = next((a for a in artifacts if a.get("type") == "image"), {})
        payload = image_artifact.get("payload", {})
        self.assertIn("image_matrix", payload)
        self.assertGreater(len(payload.get("image_matrix", [])), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
