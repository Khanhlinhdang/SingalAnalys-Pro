"""Tests for decode depth stages and quality metrics."""

import unittest
from pathlib import Path
import sys

import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rf_spectrum_analyzer.dsp.decode_stages import create_default_decode_depth_pipeline
from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer, DemodulationResult


class TestDecodeDepthStages(unittest.TestCase):
    def test_decode_depth_pipeline_bpsk_keeps_shape(self):
        pipeline = create_default_decode_depth_pipeline()
        bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)

        result = pipeline.process(bits=bits, modulation_type="BPSK", protocol_hint=None)

        self.assertEqual(len(result.output_bits), len(bits))
        self.assertFalse(result.deinterleave_applied)
        self.assertFalse(result.descramble_applied)
        self.assertIn("normalize_bits", result.operations_applied)

    def test_decode_depth_pipeline_inmarsat_descramble_applies(self):
        pipeline = create_default_decode_depth_pipeline()
        bits = np.random.default_rng(7).integers(0, 2, size=128, dtype=np.uint8)

        result = pipeline.process(bits=bits, modulation_type="BPSK", protocol_hint="inmarsat_c")

        self.assertTrue(result.descramble_applied)
        self.assertEqual(len(result.output_bits), len(bits))

    def test_decode_quality_metrics_payload(self):
        analyzer = SignalAnalyzer(sample_rate=48000)
        demod = DemodulationResult(success=True, bits=np.array([1, 0, 1], dtype=np.uint8), snr=12.5)

        quality = analyzer._calculate_decode_quality_metrics(
            demod_result=demod,
            decoded_bits=np.array([1, 0, 1, 1], dtype=np.uint8),
            output_artifacts=[{"type": "text", "confidence": 0.7, "payload": {}}],
            protocol_outputs={
                "matched_protocol": "inmarsat_c",
                "confidence": 0.8,
                "results": [
                    {"is_uncertain": False},
                    {"is_uncertain": True},
                ],
            },
        )

        self.assertEqual(quality["bit_count"], 4)
        self.assertEqual(quality["artifact_count"], 1)
        self.assertEqual(quality["frame_count"], 2)
        self.assertEqual(quality["uncertain_frame_count"], 1)
        self.assertAlmostEqual(quality["uncertain_frame_ratio"], 0.5)
        self.assertTrue(quality["protocol_matched"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
