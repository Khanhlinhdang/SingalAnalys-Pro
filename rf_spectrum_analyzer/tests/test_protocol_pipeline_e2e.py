"""E2E regression for IQ -> demod -> decode -> artifacts chain."""

import unittest
from pathlib import Path
import sys

import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rf_spectrum_analyzer.dsp.signal_analysis import (
    SignalAnalyzer,
    ModulationAnalysisResult,
)
from rf_spectrum_analyzer.dsp.protocol_plugins import InmarsatProtocolPlugin
from rf_spectrum_analyzer.dsp.output_adapters import extract_all_artifacts


class TestProtocolPipelineE2E(unittest.TestCase):
    def _build_tx_bits(self, message: str) -> np.ndarray:
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

        frame_bits = np.concatenate([descriptor_bits, payload, crc_bits])
        scrambled_frame_bits = plugin._descramble(frame_bits)
        tx_bits = np.concatenate([plugin.SYNC_WORD, scrambled_frame_bits]).astype(np.uint8)
        return tx_bits

    def _bpsk_bits_to_iq(self, bits: np.ndarray) -> np.ndarray:
        symbols = 2.0 * bits.astype(np.float32) - 1.0
        noise = 0.01 * np.random.default_rng(1234).standard_normal(len(symbols)).astype(np.float32)
        return (symbols + noise).astype(np.complex64)

    def test_iq_to_protocol_and_output_artifacts(self):
        tx_bits = self._build_tx_bits(
            "From: sat@example.org\nTo: ground@example.org\nSubject: LINK\n\nHELLO"
        )
        iq = self._bpsk_bits_to_iq(tx_bits)

        analyzer = SignalAnalyzer(sample_rate=48000)
        mod = ModulationAnalysisResult(
            modulation_type="BPSK",
            confidence=0.99,
            parameters={},
        )

        demod = analyzer.demodulate_signal(iq, mod)
        self.assertTrue(demod.success)
        self.assertIsNotNone(demod.bits)

        coding = analyzer.analyze_coding(demod.bits)
        decoded_bits = analyzer._select_best_decoded_bits(demod, coding)
        self.assertIsNotNone(decoded_bits)

        decode_depth = analyzer._run_decode_depth_stages(
            decoded_bits=decoded_bits,
            modulation_type="BPSK",
            protocol_hint="inmarsat_c",
        )
        decoded_bits_for_output = decode_depth.get("output_bits")

        artifacts = extract_all_artifacts(
            decoded_bits=decoded_bits_for_output,
            demodulated_audio=None,
            sample_rate=48000,
        )
        artifact_types = {a["type"] for a in artifacts}
        self.assertIn("text", artifact_types)

        protocol_out = analyzer._run_protocol_decode(
            decoded_bits=decoded_bits,
            modulation_type="BPSK",
            center_freq=1542e6,
        )
        self.assertEqual(protocol_out.get("matched_protocol"), "inmarsat_c")
        self.assertGreater(len(protocol_out.get("results", [])), 0)
        self.assertEqual(protocol_out.get("counter_source"), "plugin_internal")
        self.assertIsNotNone(protocol_out.get("ber"))
        self.assertIsNotNone(protocol_out.get("per"))
        self.assertIsNotNone(protocol_out.get("frame_lock_ratio"))
        self.assertTrue(protocol_out.get("crc_available"))
        self.assertEqual(protocol_out.get("crc_ok_rate"), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
