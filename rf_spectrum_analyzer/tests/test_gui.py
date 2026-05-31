"""GUI runtime tests aligned with current widget APIs."""

import unittest
from pathlib import Path
import sys

import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from PySide6.QtWidgets import QApplication

from rf_spectrum_analyzer.config.settings import Settings
from rf_spectrum_analyzer.gui.controls_widget import ControlsWidget
from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
from rf_spectrum_analyzer.gui.waterfall_widget import WaterfallWidget


class QApplicationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])


class TestControlsWidgetRuntime(QApplicationTestCase):
    def setUp(self):
        self.settings = Settings()
        self.widget = ControlsWidget(self.settings)

    def test_controls_widget_creation(self):
        self.assertIsNotNone(self.widget)
        self.assertIsNotNone(self.widget.start_button)
        self.assertIsNotNone(self.widget.stop_button)

    def test_acquisition_state_toggles_buttons(self):
        self.widget.set_acquisition_state(True)
        self.assertFalse(self.widget.start_button.isEnabled())
        self.assertTrue(self.widget.stop_button.isEnabled())

        self.widget.set_acquisition_state(False)
        self.assertTrue(self.widget.start_button.isEnabled())
        self.assertFalse(self.widget.stop_button.isEnabled())


class TestSpectrumWidgetRuntime(QApplicationTestCase):
    def setUp(self):
        self.settings = Settings()
        self.settings.sdr.sample_rate = 2.4e6
        self.settings.sdr.center_frequency = 100e6
        self.widget = SpectrumWidget(self.settings)

    def test_spectrum_widget_update_data(self):
        spectrum = -90 + 10 * np.random.randn(1024)
        self.widget.update_data(spectrum)

        self.assertEqual(len(self.widget.spectrum_data), 1024)
        self.assertEqual(len(self.widget.frequency_axis), 1024)

    def test_signal_analysis_request_when_range_contains_signal(self):
        spectrum = -100 * np.ones(1024)
        spectrum[500:510] = -30
        self.widget.update_data(spectrum)

        self.widget.set_frequency_markers_enabled(True)
        center = self.settings.sdr.center_frequency
        self.widget.set_frequency_range(center - 50_000, center + 50_000)

        request = self.widget.request_signal_analysis()
        self.assertIsNotNone(request)
        self.assertIn("center_freq", request)
        self.assertIn("bandwidth", request)


class TestWaterfallWidgetRuntime(QApplicationTestCase):
    def setUp(self):
        self.settings = Settings()
        self.widget = WaterfallWidget(self.settings)

    def test_waterfall_update_data(self):
        line = -95 + 5 * np.random.randn(1024)
        self.widget.update_data(line)

        self.assertGreater(len(self.widget.waterfall_data), 0)

    def test_waterfall_clear_data(self):
        line = -95 + 5 * np.random.randn(1024)
        self.widget.update_data(line)
        self.widget.clear_data()

        self.assertEqual(len(self.widget.waterfall_data), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
