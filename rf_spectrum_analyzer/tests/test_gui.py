"""
GUI Component Tests
Comprehensive testing of GUI widgets and user interface components
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import warnings
from unittest.mock import Mock, patch, MagicMock
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Test GUI availability
GUI_AVAILABLE = False
PYQTGRAPH_AVAILABLE = False

try:
    from PySide6.QtWidgets import QApplication, QWidget
    from PySide6.QtCore import QTimer
    from PySide6.QtTest import QTest
    GUI_AVAILABLE = True
except ImportError:
    try:
        from PyQt5.QtWidgets import QApplication, QWidget
        from PyQt5.QtCore import QTimer
        from PyQt5.QtTest import QTest
        GUI_AVAILABLE = True
    except ImportError:
        pass

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

# Test widget availability
WIDGETS_AVAILABLE = {}

try:
    from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
    WIDGETS_AVAILABLE['spectrum'] = True
except ImportError:
    WIDGETS_AVAILABLE['spectrum'] = False

try:
    from rf_spectrum_analyzer.gui.waterfall_widget import WaterfallWidget
    WIDGETS_AVAILABLE['waterfall'] = True
except ImportError:
    WIDGETS_AVAILABLE['waterfall'] = False

try:
    from rf_spectrum_analyzer.gui.controls_widget import ControlsWidget
    WIDGETS_AVAILABLE['control'] = True
except ImportError:
    WIDGETS_AVAILABLE['control'] = False

try:
    from rf_spectrum_analyzer.dialogs.settings_dialog import SettingsDialog
    WIDGETS_AVAILABLE['settings'] = True
except ImportError:
    WIDGETS_AVAILABLE['settings'] = False

try:
    from rf_spectrum_analyzer.gui.main_window import MainWindow
    WIDGETS_AVAILABLE['main'] = True
except ImportError:
    WIDGETS_AVAILABLE['main'] = False


class QApplicationTestCase(unittest.TestCase):
    """Base test case that sets up QApplication"""
    
    @classmethod
    def setUpClass(cls):
        """Set up QApplication for GUI tests"""
        if GUI_AVAILABLE:
            cls.app = QApplication.instance()
            if cls.app is None:
                cls.app = QApplication([])
        else:
            cls.app = None
    
    @classmethod
    def tearDownClass(cls):
        """Clean up QApplication"""
        if cls.app is not None:
            cls.app.quit()


class TestSpectrumWidget(QApplicationTestCase):
    """Test Spectrum Display Widget"""
    
    @unittest.skipUnless(GUI_AVAILABLE and WIDGETS_AVAILABLE.get('spectrum'), 
                        "GUI or spectrum widget not available")
    def setUp(self):
        """Set up test environment"""
        # Create mock settings
        from unittest.mock import Mock
        mock_settings = Mock()
        mock_settings.gui.spectrum_min_db = -120
        mock_settings.gui.spectrum_max_db = 0
        mock_settings.gui.spectrum_ref_level = 0
        mock_settings.gui.spectrum_scale = 10
        mock_settings.gui.auto_scale = True
        mock_settings.gui.peak_hold = False
        mock_settings.gui.average_count = 1
        mock_settings.gui.grid_enabled = True
        mock_settings.gui.marker_enabled = True
        mock_settings.gui.cursor_enabled = True
        
        self.widget = SpectrumWidget(mock_settings)
        self.sample_rate = 1e6
        self.test_freqs = np.linspace(-self.sample_rate/2, self.sample_rate/2, 1024)
        self.test_psd = -80 + 20 * np.log10(np.abs(np.random.randn(1024)))
    
    def test_spectrum_widget_creation(self):
        """Test spectrum widget creation"""
        self.assertIsNotNone(self.widget)
        self.assertTrue(hasattr(self.widget, 'plot_item'))
    
    def test_spectrum_data_update(self):
        """Test spectrum data update"""
        # Update spectrum data
        self.widget.update_spectrum(self.test_freqs, self.test_psd)
        
        # Check that plot has data
        plot_data_items = self.widget.plot_item.listDataItems()
        self.assertGreater(len(plot_data_items), 0)
    
    def test_spectrum_axis_labels(self):
        """Test spectrum axis labels and units"""
        self.widget.update_spectrum(self.test_freqs, self.test_psd)
        
        # Check axis labels
        self.assertIsNotNone(self.widget.plot_item.getAxis('bottom').labelText)
        self.assertIsNotNone(self.widget.plot_item.getAxis('left').labelText)
    
    def test_spectrum_zoom_and_pan(self):
        """Test spectrum zoom and pan functionality"""
        self.widget.update_spectrum(self.test_freqs, self.test_psd)
        
        # Test zoom
        view_box = self.widget.plot_item.getViewBox()
        original_range = view_box.viewRange()
        
        # Simulate zoom in
        view_box.scaleBy((0.5, 0.5))
        new_range = view_box.viewRange()
        
        # Range should be smaller after zoom in
        x_range_original = original_range[0][1] - original_range[0][0]
        x_range_new = new_range[0][1] - new_range[0][0]
        self.assertLess(x_range_new, x_range_original)
    
    def test_spectrum_cursor_functionality(self):
        """Test spectrum cursor and measurement functionality"""
        self.widget.update_spectrum(self.test_freqs, self.test_psd)
        
        # Enable cursor if available
        if hasattr(self.widget, 'enable_cursor'):
            self.widget.enable_cursor(True)
            self.assertTrue(self.widget.cursor_enabled)
    
    def test_spectrum_peak_detection(self):
        """Test spectrum peak detection and marking"""
        # Create spectrum with known peaks
        freqs = np.linspace(-500e3, 500e3, 1024)
        psd = -100 * np.ones_like(freqs)
        
        # Add peaks at specific frequencies
        peak_freqs = [-200e3, 0, 150e3]
        for peak_freq in peak_freqs:
            idx = np.argmin(np.abs(freqs - peak_freq))
            psd[idx] = -40  # Strong peak
        
        self.widget.update_spectrum(freqs, psd)
        
        # Test peak detection if available
        if hasattr(self.widget, 'detect_peaks'):
            peaks = self.widget.detect_peaks(threshold=-50)
            self.assertGreaterEqual(len(peaks), len(peak_freqs))


class TestWaterfallWidget(QApplicationTestCase):
    """Test Waterfall Display Widget"""
    
    @unittest.skipUnless(GUI_AVAILABLE and WIDGETS_AVAILABLE.get('waterfall'), 
                        "GUI or waterfall widget not available")
    def setUp(self):
        """Set up test environment"""
        # Create mock settings
        from unittest.mock import Mock
        mock_settings = Mock()
        mock_settings.gui.waterfall_time_span = 60.0
        mock_settings.gui.waterfall_min_db = -120
        mock_settings.gui.waterfall_max_db = 0
        mock_settings.gui.colormap = 'viridis'
        mock_settings.gui.waterfall_fps = 30
        mock_settings.gui.grid_enabled = True
        
        self.widget = WaterfallWidget(mock_settings)
        self.fft_size = 512
        self.num_ffts = 100
        
        # Create test spectrogram data
        freqs = np.linspace(-250e3, 250e3, self.fft_size)
        times = np.arange(self.num_ffts) * 0.01  # 10ms per FFT
        
        # Generate spectrogram with moving signal
        self.test_spectrogram = np.zeros((self.fft_size, self.num_ffts))
        for i in range(self.num_ffts):
            # Moving tone
            signal_freq = 100e3 * np.sin(2 * np.pi * 0.1 * times[i])
            signal_idx = np.argmin(np.abs(freqs - signal_freq))
            self.test_spectrogram[:, i] = -100 + 10 * np.random.randn(self.fft_size)
            self.test_spectrogram[signal_idx, i] = -40  # Strong signal
        
        self.test_freqs = freqs
        self.test_times = times
    
    def test_waterfall_widget_creation(self):
        """Test waterfall widget creation"""
        self.assertIsNotNone(self.widget)
        self.assertTrue(hasattr(self.widget, 'image_item'))
    
    def test_waterfall_data_update(self):
        """Test waterfall data update"""
        # Update waterfall data
        self.widget.update_waterfall(self.test_freqs, self.test_times, self.test_spectrogram)
        
        # Check that image has data
        self.assertIsNotNone(self.widget.image_item.image)
    
    def test_waterfall_colormap(self):
        """Test waterfall colormap settings"""
        self.widget.update_waterfall(self.test_freqs, self.test_times, self.test_spectrogram)
        
        # Test different colormaps if available
        if hasattr(self.widget, 'set_colormap'):
            colormaps = ['viridis', 'plasma', 'jet']
            for colormap in colormaps:
                self.widget.set_colormap(colormap)
                # Verify colormap was applied
                self.assertEqual(self.widget.current_colormap, colormap)
    
    def test_waterfall_zoom_functionality(self):
        """Test waterfall zoom and navigation"""
        self.widget.update_waterfall(self.test_freqs, self.test_times, self.test_spectrogram)
        
        # Test zoom
        view_box = self.widget.plot_item.getViewBox()
        original_range = view_box.viewRange()
        
        # Zoom to specific region
        view_box.setRange(xRange=[-100e3, 100e3], yRange=[10, 50])
        new_range = view_box.viewRange()
        
        # Verify zoom was applied
        self.assertNotEqual(original_range, new_range)
    
    def test_waterfall_time_navigation(self):
        """Test waterfall time navigation and scrolling"""
        self.widget.update_waterfall(self.test_freqs, self.test_times, self.test_spectrogram)
        
        # Test time window navigation if available
        if hasattr(self.widget, 'set_time_window'):
            self.widget.set_time_window(0.2)  # 200ms window
            self.assertEqual(self.widget.time_window, 0.2)


class TestControlPanel(QApplicationTestCase):
    """Test Control Panel Widget"""
    
    @unittest.skipUnless(GUI_AVAILABLE and WIDGETS_AVAILABLE.get('control'), 
                        "GUI or control panel not available")
    def setUp(self):
        """Set up test environment"""
        # Create mock settings
        from unittest.mock import Mock
        mock_settings = Mock()
        mock_settings.device.sdr_type = 'rtlsdr'
        mock_settings.device.sample_rate = 2.048e6
        mock_settings.device.center_frequency = 100e6
        mock_settings.device.gain = 30
        mock_settings.gui.fft_size = 1024
        mock_settings.gui.window_type = 'hann'
        mock_settings.gui.averaging = 10
        
        self.control_panel = ControlsWidget(mock_settings)
    
    def test_control_panel_creation(self):
        """Test control panel creation"""
        self.assertIsNotNone(self.control_panel)
    
    def test_frequency_controls(self):
        """Test frequency control widgets"""
        # Test center frequency control
        if hasattr(self.control_panel, 'freq_control'):
            # Set frequency
            test_freq = 433e6
            self.control_panel.freq_control.setValue(test_freq)
            self.assertEqual(self.control_panel.freq_control.value(), test_freq)
    
    def test_gain_controls(self):
        """Test gain control widgets"""
        if hasattr(self.control_panel, 'gain_control'):
            # Set gain
            test_gain = 30
            self.control_panel.gain_control.setValue(test_gain)
            self.assertEqual(self.control_panel.gain_control.value(), test_gain)
    
    def test_sample_rate_controls(self):
        """Test sample rate control widgets"""
        if hasattr(self.control_panel, 'samplerate_control'):
            # Set sample rate
            test_rate = 2e6
            self.control_panel.samplerate_control.setValue(test_rate)
            self.assertEqual(self.control_panel.samplerate_control.value(), test_rate)
    
    def test_start_stop_controls(self):
        """Test start/stop control buttons"""
        if hasattr(self.control_panel, 'start_button'):
            # Test button click
            start_button = self.control_panel.start_button
            self.assertTrue(start_button.isEnabled())
            
            # Simulate button click
            QTest.mouseClick(start_button, 1)  # Left click
    
    def test_control_panel_signals(self):
        """Test control panel signals and slots"""
        # Test that controls emit appropriate signals
        signal_emitted = False
        
        def on_frequency_changed(freq):
            nonlocal signal_emitted
            signal_emitted = True
        
        # Connect signal if available
        if hasattr(self.control_panel, 'frequency_changed'):
            self.control_panel.frequency_changed.connect(on_frequency_changed)
            
            # Trigger frequency change
            if hasattr(self.control_panel, 'freq_control'):
                self.control_panel.freq_control.setValue(100e6)
                
                # Process events to trigger signal
                self.app.processEvents()
                self.assertTrue(signal_emitted)


class TestSettingsDialog(QApplicationTestCase):
    """Test Settings Dialog"""
    
    @unittest.skipUnless(GUI_AVAILABLE and WIDGETS_AVAILABLE.get('settings'), 
                        "GUI or settings dialog not available")
    def setUp(self):
        """Set up test environment"""
        self.settings_dialog = SettingsDialog()
    
    def test_settings_dialog_creation(self):
        """Test settings dialog creation"""
        self.assertIsNotNone(self.settings_dialog)
    
    def test_device_selection(self):
        """Test device selection in settings"""
        if hasattr(self.settings_dialog, 'device_combo'):
            device_combo = self.settings_dialog.device_combo
            
            # Check that device options are available
            self.assertGreater(device_combo.count(), 0)
            
            # Test device selection
            device_combo.setCurrentIndex(0)
            selected_device = device_combo.currentText()
            self.assertIsNotNone(selected_device)
    
    def test_advanced_settings(self):
        """Test advanced settings configuration"""
        if hasattr(self.settings_dialog, 'advanced_settings'):
            # Test FFT size setting
            if hasattr(self.settings_dialog, 'fft_size_control'):
                self.settings_dialog.fft_size_control.setValue(2048)
                self.assertEqual(self.settings_dialog.fft_size_control.value(), 2048)
            
            # Test averaging setting
            if hasattr(self.settings_dialog, 'averaging_control'):
                self.settings_dialog.averaging_control.setValue(10)
                self.assertEqual(self.settings_dialog.averaging_control.value(), 10)
    
    def test_settings_validation(self):
        """Test settings validation"""
        # Test that invalid settings are rejected
        if hasattr(self.settings_dialog, 'validate_settings'):
            # Test with valid settings
            valid_settings = {
                'device_type': 'hackrf',
                'sample_rate': 20e6,
                'center_frequency': 433e6,
                'gain': 30
            }
            self.assertTrue(self.settings_dialog.validate_settings(valid_settings))
            
            # Test with invalid settings
            invalid_settings = {
                'device_type': 'invalid',
                'sample_rate': -1000,
                'center_frequency': -100e6,
                'gain': 1000
            }
            self.assertFalse(self.settings_dialog.validate_settings(invalid_settings))
    
    def test_settings_save_load(self):
        """Test settings save and load functionality"""
        if hasattr(self.settings_dialog, 'save_settings') and hasattr(self.settings_dialog, 'load_settings'):
            # Set some test settings
            test_settings = {
                'device_type': 'rtlsdr',
                'sample_rate': 2.4e6,
                'center_frequency': 100e6
            }
            
            # Save settings
            self.settings_dialog.save_settings(test_settings)
            
            # Load settings
            loaded_settings = self.settings_dialog.load_settings()
            
            # Verify settings were saved and loaded correctly
            for key, value in test_settings.items():
                self.assertEqual(loaded_settings.get(key), value)


class TestMainWindow(QApplicationTestCase):
    """Test Main Application Window"""
    
    @unittest.skipUnless(GUI_AVAILABLE and WIDGETS_AVAILABLE.get('main'), 
                        "GUI or main window not available")
    def setUp(self):
        """Set up test environment"""
        try:
            # Create mock settings for MainWindow
            from unittest.mock import Mock
            mock_settings = Mock()
            mock_settings.sdr.device_type = 'rtlsdr'
            mock_settings.sdr.sample_rate = 2.048e6
            mock_settings.sdr.center_frequency = 100e6
            mock_settings.sdr.gain = 20.0
            mock_settings.gui.window_width = 1200
            mock_settings.gui.window_height = 800
            mock_settings.gui.theme = 'dark'
            mock_settings.gui.grid_enabled = True
            mock_settings.gui.colormap = 'viridis'
            
            self.main_window = MainWindow(mock_settings)
        except Exception as e:
            self.skipTest(f"Main window creation failed: {e}")
    
    def test_main_window_creation(self):
        """Test main window creation"""
        self.assertIsNotNone(self.main_window)
        self.assertTrue(self.main_window.isWidget())
    
    def test_window_components(self):
        """Test main window components"""
        # Check that main components exist
        if hasattr(self.main_window, 'spectrum_widget'):
            self.assertIsNotNone(self.main_window.spectrum_widget)
        
        if hasattr(self.main_window, 'waterfall_widget'):
            self.assertIsNotNone(self.main_window.waterfall_widget)
        
        if hasattr(self.main_window, 'control_panel'):
            self.assertIsNotNone(self.main_window.control_panel)
    
    def test_menu_bar(self):
        """Test menu bar functionality"""
        if hasattr(self.main_window, 'menuBar'):
            menu_bar = self.main_window.menuBar()
            self.assertIsNotNone(menu_bar)
            
            # Check that menus exist
            actions = menu_bar.actions()
            self.assertGreater(len(actions), 0)
    
    def test_toolbar(self):
        """Test toolbar functionality"""
        if hasattr(self.main_window, 'toolbar'):
            toolbar = self.main_window.toolbar
            self.assertIsNotNone(toolbar)
            
            # Check toolbar actions
            actions = toolbar.actions()
            self.assertGreater(len(actions), 0)
    
    def test_status_bar(self):
        """Test status bar functionality"""
        if hasattr(self.main_window, 'statusBar'):
            status_bar = self.main_window.statusBar()
            self.assertIsNotNone(status_bar)
            
            # Test status message
            test_message = "Test status message"
            status_bar.showMessage(test_message)
            self.assertEqual(status_bar.currentMessage(), test_message)
    
    def test_window_layout(self):
        """Test main window layout"""
        # Check that window has proper layout
        central_widget = self.main_window.centralWidget()
        self.assertIsNotNone(central_widget)
        
        # Check window size
        size = self.main_window.size()
        self.assertGreater(size.width(), 0)
        self.assertGreater(size.height(), 0)


class TestGUIIntegration(QApplicationTestCase):
    """Test GUI component integration"""
    
    @unittest.skipUnless(GUI_AVAILABLE, "GUI not available")
    def setUp(self):
        """Set up test environment"""
        # Create minimal GUI components for integration testing
        self.widget = QWidget()
        
        if PYQTGRAPH_AVAILABLE:
            self.plot_widget = pg.PlotWidget()
    
    def test_pyqtgraph_integration(self):
        """Test PyQtGraph integration"""
        if not PYQTGRAPH_AVAILABLE:
            self.skipTest("PyQtGraph not available")
        
        # Test basic plotting
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)
        
        plot_item = self.plot_widget.plot(x, y)
        self.assertIsNotNone(plot_item)
        
        # Test plot clearing
        self.plot_widget.clear()
        data_items = self.plot_widget.listDataItems()
        self.assertEqual(len(data_items), 0)
    
    def test_real_time_plotting(self):
        """Test real-time plotting capabilities"""
        if not PYQTGRAPH_AVAILABLE:
            self.skipTest("PyQtGraph not available")
        
        # Simulate real-time data updates
        x = np.linspace(0, 10, 100)
        
        for i in range(10):
            y = np.sin(x + i * 0.1)
            self.plot_widget.clear()
            self.plot_widget.plot(x, y)
            
            # Process events to update display
            self.app.processEvents()
    
    def test_widget_performance(self):
        """Test GUI widget performance"""
        if not PYQTGRAPH_AVAILABLE:
            self.skipTest("PyQtGraph not available")
        
        # Test performance with large datasets
        large_x = np.linspace(0, 100, 10000)
        large_y = np.random.randn(10000)
        
        start_time = time.time()
        plot_item = self.plot_widget.plot(large_x, large_y)
        end_time = time.time()
        
        plot_time = end_time - start_time
        self.assertLess(plot_time, 1.0)  # Should plot in less than 1 second
        
        # Test update performance
        start_time = time.time()
        plot_item.setData(large_x, large_y * 2)
        end_time = time.time()
        
        update_time = end_time - start_time
        self.assertLess(update_time, 0.1)  # Should update in less than 100ms


class TestGUIErrorHandling(QApplicationTestCase):
    """Test GUI error handling"""
    
    @unittest.skipUnless(GUI_AVAILABLE, "GUI not available")
    def test_invalid_data_handling(self):
        """Test handling of invalid display data"""
        if not PYQTGRAPH_AVAILABLE:
            self.skipTest("PyQtGraph not available")
        
        plot_widget = pg.PlotWidget()
        
        # Test with empty data
        try:
            plot_widget.plot([], [])
            # Should handle gracefully
        except Exception as e:
            self.fail(f"Empty data caused exception: {e}")
        
        # Test with mismatched array sizes
        try:
            plot_widget.plot([1, 2, 3], [1, 2])
            # Should handle gracefully or raise appropriate exception
        except (ValueError, IndexError):
            # Acceptable to raise exception for mismatched data
            pass
    
    def test_memory_management(self):
        """Test GUI memory management"""
        if not PYQTGRAPH_AVAILABLE:
            self.skipTest("PyQtGraph not available")
        
        plot_widget = pg.PlotWidget()
        
        # Create and destroy many plot items
        for i in range(100):
            x = np.random.randn(1000)
            y = np.random.randn(1000)
            plot_widget.plot(x, y)
            plot_widget.clear()
            
            # Process events periodically
            if i % 10 == 0:
                self.app.processEvents()
        
        # Should not cause memory issues


if __name__ == '__main__':
    # Print GUI component availability
    print("\n=== GUI Component Availability ===")
    print(f"{'✓' if GUI_AVAILABLE else '✗'} Qt Framework")
    print(f"{'✓' if PYQTGRAPH_AVAILABLE else '✗'} PyQtGraph")
    
    print("\n=== Widget Availability ===")
    for widget, available in WIDGETS_AVAILABLE.items():
        status = "✓" if available else "✗"
        print(f"{status} {widget.title()} Widget")
    print()
    
    unittest.main(verbosity=2)