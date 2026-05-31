"""Golden fixture tests for Meteor LRPT staged decode chain."""

import unittest
from pathlib import Path
import sys

import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rf_spectrum_analyzer.dsp.meteor_lrpt_chain import MeteorLrptDecodeChain


class TestMeteorLrptDecodeChain(unittest.TestCase):
    def setUp(self):
        self.chain = MeteorLrptDecodeChain(interleave_depth=4, rs_nsym=16)

    def test_deinterleave_roundtrip_golden(self):
        bits = np.random.default_rng(11).integers(0, 2, 256, dtype=np.uint8)
        interleaved = self.chain.interleave(bits, 4)
        restored = self.chain.deinterleave(interleaved, 4)

        self.assertTrue(np.array_equal(bits, restored[: bits.size]))

    def test_viterbi_decodes_rate_half_reference(self):
        bits = np.random.default_rng(21).integers(0, 2, 96, dtype=np.uint8)
        encoded = self.chain.conv_encode(bits)
        decoded = self.chain.viterbi_decode_hard(encoded)

        self.assertTrue(np.array_equal(bits, decoded[: bits.size]))

    def test_viterbi_soft_metrics_path(self):
        bits = np.random.default_rng(18).integers(0, 2, 96, dtype=np.uint8)
        encoded = self.chain.conv_encode(bits)
        soft = np.where(encoded > 0, 0.95, -0.95).astype(np.float32)

        decoded = self.chain.viterbi_decode_soft(soft)

        self.assertIn("decoded_bits", decoded)
        self.assertIn("path_metric", decoded)
        self.assertIn("soft_confidence", decoded)
        self.assertGreaterEqual(decoded.get("soft_confidence", 0.0), 0.0)
        self.assertLessEqual(decoded.get("soft_confidence", 1.0), 1.0)
        self.assertTrue(np.array_equal(bits, decoded.get("decoded_bits")[: bits.size]))

    def test_reed_solomon_single_symbol_correction(self):
        message = np.arange(40, dtype=np.uint8)
        codeword = self.chain.rs_encode(message, nsym=16)
        corrupted = codeword.copy()
        corrupted[5] ^= np.uint8(0x37)

        decoded = self.chain.rs_decode(corrupted, nsym=16)

        self.assertTrue(decoded.get("decode_success"))
        self.assertEqual(decoded.get("corrected_symbols"), 1)
        self.assertEqual(decoded.get("algorithm"), "rs_bm_forney")
        self.assertTrue(np.array_equal(decoded.get("message"), message))

    def test_reed_solomon_multi_symbol_correction(self):
        message = np.array([(7 * i + 3) % 256 for i in range(52)], dtype=np.uint8)
        codeword = self.chain.rs_encode(message, nsym=16)
        corrupted = codeword.copy()

        error_positions = [3, 10, 21, 45]
        error_masks = [0x41, 0x93, 0x2D, 0xE7]
        for idx, mask in zip(error_positions, error_masks):
            corrupted[idx] ^= np.uint8(mask)

        decoded = self.chain.rs_decode(corrupted, nsym=16)

        self.assertTrue(decoded.get("decode_success"))
        self.assertEqual(decoded.get("corrected_symbols"), len(error_positions))
        self.assertEqual(decoded.get("algorithm"), "rs_bm_forney")
        self.assertTrue(np.array_equal(decoded.get("message"), message))

    def test_full_chain_golden_fixture(self):
        payload = np.array([(13 * i) % 256 for i in range(48)], dtype=np.uint8)
        encoded_bits = self.chain.encode_for_test(payload)
        decoded = self.chain.decode_frame(encoded_bits)

        decoded_bytes = decoded.get("decoded_bytes")
        self.assertIsNotNone(decoded_bytes)
        self.assertTrue(np.array_equal(decoded_bytes[: payload.size], payload))

        stages = decoded.get("stages", {})
        self.assertTrue(stages.get("deinterleave", {}).get("applied"))
        self.assertTrue(stages.get("viterbi", {}).get("applied"))
        self.assertTrue(stages.get("reed_solomon", {}).get("applied"))
        self.assertIn("soft_metric", stages.get("viterbi", {}))
        self.assertIn("soft_confidence", stages.get("viterbi", {}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
