"""
Pipeline end-to-end integration test.
Validates backend mock -> SignalProcessor -> GUI mock data flow.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from rf_spectrum_analyzer.config.settings import Settings
from rf_spectrum_analyzer.core.signal_processor import SignalProcessor


class MockBackend:
    """Minimal backend mock that returns deterministic IQ data."""

    def __init__(self, sample_rate: float, center_freq: float):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.connected = False

    def connect(self):
        self.connected = True
        return True

    def read_samples(self, n: int):
        if not self.connected:
            return None
        t = np.arange(n) / self.sample_rate
        tone = np.exp(2j * np.pi * 50_000 * t)
        noise = 0.03 * (np.random.randn(n) + 1j * np.random.randn(n))
        return (tone + noise).astype(np.complex64)


class MockGUI:
    """Minimal GUI sink for spectrum updates."""

    def __init__(self):
        self.last_spectrum = None
        self.update_count = 0

    def update_spectrum(self, spectrum):
        self.last_spectrum = spectrum
        self.update_count += 1


class TestPipelineE2E(unittest.TestCase):
    """End-to-end flow from IQ source to GUI-ready spectrum."""

    def setUp(self):
        self.settings = Settings()
        self.settings.dsp.fft_size = 1024
        self.settings.dsp.averaging = 4

        self.backend = MockBackend(
            sample_rate=self.settings.sdr.sample_rate,
            center_freq=self.settings.sdr.center_frequency,
        )
        self.processor = SignalProcessor(self.settings)
        self.gui = MockGUI()

    def test_backend_processor_gui_flow(self):
        connected = self.backend.connect()
        self.assertTrue(connected)

        iq = self.backend.read_samples(self.settings.dsp.fft_size * 2)
        self.assertIsNotNone(iq)
        self.assertGreater(len(iq), self.settings.dsp.fft_size)

        frame = iq[: self.settings.dsp.fft_size]
        self.processor.update_current_data(frame)

        spectrum = self.processor.compute_spectrum(frame)
        self.assertIsNotNone(spectrum)
        self.assertEqual(len(spectrum), self.settings.dsp.fft_size)
        self.assertTrue(np.isfinite(spectrum).all())

        self.gui.update_spectrum(spectrum)
        self.assertEqual(self.gui.update_count, 1)
        self.assertIsNotNone(self.gui.last_spectrum)


if __name__ == '__main__':
    unittest.main(verbosity=2)
