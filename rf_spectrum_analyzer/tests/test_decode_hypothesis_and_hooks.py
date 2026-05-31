"""Tests for multi-hypothesis scoring history and dechannelization hooks."""

import unittest
from pathlib import Path
import sys

import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer


class TestDecodeHypothesisAndHooks(unittest.TestCase):
    def test_modulation_hypotheses_and_history_are_recorded(self):
        analyzer = SignalAnalyzer(sample_rate=48000)

        rng = np.random.default_rng(123)
        bits = rng.integers(0, 2, 4000, dtype=np.uint8)
        symbols = (2 * bits - 1).astype(np.float32)
        iq = (symbols + 1j * 0.02 * rng.standard_normal(len(symbols))).astype(np.complex64)

        result = analyzer._analyze_modulation_advanced(iq)
        params = result.parameters

        self.assertIn("modulation_hypotheses", params)
        self.assertGreater(len(params["modulation_hypotheses"]), 0)
        self.assertIn("modulation_score_history", params)
        self.assertGreater(len(params["modulation_score_history"]), 0)

    def test_coding_hypotheses_and_history_are_recorded(self):
        analyzer = SignalAnalyzer(sample_rate=48000)
        bits = np.random.default_rng(42).integers(0, 2, 2048, dtype=np.uint8)

        result = analyzer._analyze_coding_advanced(bits)
        self.assertIsNotNone(result)
        self.assertIn("coding_hypotheses", result.parameters)
        self.assertGreater(len(result.parameters["coding_hypotheses"]), 0)
        self.assertIn("coding_score_history", result.parameters)
        self.assertGreater(len(result.parameters["coding_score_history"]), 0)

    def test_dechannelization_hooks_emit_strategy_metrics(self):
        analyzer = SignalAnalyzer(sample_rate=48000)
        rng = np.random.default_rng(7)

        iq = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
        bits = rng.integers(0, 2, 2048, dtype=np.uint8)

        hook_result = analyzer._run_dechannelization_hooks(
            iq_data=iq,
            decoded_bits=bits,
            modulation_type="QPSK",
        )
        metrics = hook_result["metrics"]

        self.assertTrue(metrics.get("hook_executed"))
        self.assertIn(metrics.get("selected_strategy"), {"none", "tdma", "fdma"})
        self.assertIn("strategy_scores", metrics)

    def test_decode_quality_reads_protocol_counters(self):
        analyzer = SignalAnalyzer(sample_rate=48000)
        quality = analyzer._calculate_decode_quality_metrics(
            demod_result=None,
            decoded_bits=np.array([1, 0, 1, 1], dtype=np.uint8),
            output_artifacts=[{"type": "text", "confidence": 0.8, "payload": {}}],
            protocol_outputs={
                "matched_protocol": "inmarsat_c",
                "confidence": 0.9,
                "ber": 0.02,
                "per": 0.1,
                "crc_ok_rate": 0.85,
                "frame_lock_ratio": 0.95,
                "results": [{"is_uncertain": False}],
            },
        )

        self.assertAlmostEqual(quality["ber"], 0.02)
        self.assertAlmostEqual(quality["per"], 0.1)
        self.assertAlmostEqual(quality["crc_ok_rate"], 0.85)
        self.assertAlmostEqual(quality["frame_lock_ratio"], 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=2)
