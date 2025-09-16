"""
Import Validation Tests
Comprehensive testing of all imports and dependencies
"""

import unittest
import sys
import warnings
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestImports(unittest.TestCase):
    """Test all module imports and dependencies"""
    
    def setUp(self):
        """Set up test environment"""
        self.import_results = {}
        
    def test_core_python_libraries(self):
        """Test core Python library imports"""
        core_libs = [
            'numpy', 'scipy', 'matplotlib', 'logging', 'threading',
            'queue', 'dataclasses', 'typing', 'pathlib', 'json',
            'configparser', 'argparse', 'time', 'datetime'
        ]
        
        for lib in core_libs:
            with self.subTest(library=lib):
                try:
                    __import__(lib)
                    self.import_results[lib] = True
                    print(f"✓ {lib} imported successfully")
                except ImportError as e:
                    self.import_results[lib] = False
                    self.fail(f"Failed to import {lib}: {e}")
    
    def test_pyside6_imports(self):
        """Test PySide6/Qt imports"""
        pyside6_modules = [
            'PySide6.QtCore',
            'PySide6.QtWidgets', 
            'PySide6.QtGui',
            'PySide6.QtOpenGL'
        ]
        
        for module in pyside6_modules:
            with self.subTest(module=module):
                try:
                    __import__(module)
                    self.import_results[module] = True
                    print(f"✓ {module} imported successfully")
                except ImportError as e:
                    self.import_results[module] = False
                    print(f"⚠ {module} not available: {e}")
                    # Don't fail for Qt - it's optional for some tests
    
    def test_pyqtgraph_imports(self):
        """Test PyQtGraph imports"""
        try:
            import pyqtgraph as pg
            import pyqtgraph.opengl as gl
            self.import_results['pyqtgraph'] = True
            print("✓ PyQtGraph imported successfully")
        except ImportError as e:
            self.import_results['pyqtgraph'] = False
            print(f"⚠ PyQtGraph not available: {e}")
    
    def test_sdr_libraries(self):
        """Test SDR library imports"""
        sdr_libs = {
            'sdr': 'mhostetter/sdr library',
            'sk_dsp_comm': 'scikit-dsp-comm library'
        }
        
        for lib, description in sdr_libs.items():
            with self.subTest(library=lib):
                try:
                    __import__(lib)
                    self.import_results[lib] = True
                    print(f"✓ {description} imported successfully")
                except ImportError as e:
                    self.import_results[lib] = False
                    print(f"⚠ {description} not available: {e}")
                    # Don't fail - these are optional
    
    def test_sdr_hardware_libraries(self):
        """Test SDR hardware library imports"""
        hardware_libs = {
            'hackrf': 'HackRF library',
            'rtlsdr': 'RTL-SDR library', 
            'adi': 'Analog Devices (PlutoSDR) library',
            'SoapySDR': 'SoapySDR library',
            'uhd': 'USRP library'
        }
        
        for lib, description in hardware_libs.items():
            with self.subTest(library=lib):
                try:
                    __import__(lib)
                    self.import_results[lib] = True
                    print(f"✓ {description} imported successfully")
                except ImportError as e:
                    self.import_results[lib] = False
                    print(f"⚠ {description} not available: {e}")
                    # Don't fail - hardware libraries are optional
    
    def test_project_modules(self):
        """Test project module imports"""
        project_modules = [
            'core',
            'core.app',
            'core.sdr_backend',
            'core.signal_processor',
            'dsp',
            'dsp.filters',
            'dsp.modulation',
            'dsp.analysis',
            'dsp.utils',
            'backends',
            'backends.hackrf_backend',
            'backends.rtlsdr_backend',
            'backends.pluto_backend',
            'backends.soapy_backend',
            'backends.usrp_backend',
            'gui',
            'gui.main_window',
            'gui.spectrum_widget',
            'gui.waterfall_widget',
            'gui.controls_widget',
            'config.settings',
            'utils.logger',
            'utils.helpers',
            'utils.file_io',
            'resources.icons',
            'resources.themes'
        ]
        
        for module in project_modules:
            with self.subTest(module=module):
                try:
                    __import__(module)
                    self.import_results[module] = True
                    print(f"✓ {module} imported successfully")
                except ImportError as e:
                    self.import_results[module] = False
                    print(f"❌ Failed to import {module}: {e}")
                    # Fail for project modules - these should work
                    self.fail(f"Project module {module} import failed: {e}")
    
    def test_dsp_module_components(self):
        """Test DSP module component imports"""
        try:
            from rf_spectrum_analyzer.dsp.filters import (
                FIRFilter, IIRFilter, PolyphaseFilter, AdaptiveFilter,
                ButterworthFilter, ChebyshevFilter, EllipticFilter
            )
            print("✓ DSP filters imported successfully")
            
            from rf_spectrum_analyzer.dsp.modulation import (
                PSKModulator, QAMModulator, FSKModulator, OFDMModulator,
                PSKDemodulator
            )
            print("✓ DSP modulation imported successfully")
            
            from rf_spectrum_analyzer.dsp.analysis import (
                SpectrumAnalyzer, SignalDetector, ParameterEstimator,
                InterferenceAnalyzer
            )
            print("✓ DSP analysis imported successfully")
            
            from rf_spectrum_analyzer.dsp.utils import (
                create_window, generate_awgn, generate_tone, resample_signal
            )
            print("✓ DSP utils imported successfully")
            
        except ImportError as e:
            self.fail(f"DSP component import failed: {e}")
    
    def test_backend_components(self):
        """Test backend component imports"""
        backend_classes = [
            ('backends.hackrf_backend', 'HackRFBackend'),
            ('backends.rtlsdr_backend', 'RTLSDRBackend'),
            ('backends.pluto_backend', 'PlutoBackend'),
            ('backends.soapy_backend', 'SoapyBackend'),
            ('backends.usrp_backend', 'USRPBackend')
        ]
        
        for module_name, class_name in backend_classes:
            with self.subTest(backend=class_name):
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    backend_class = getattr(module, class_name)
                    print(f"✓ {class_name} imported successfully")
                except (ImportError, AttributeError) as e:
                    print(f"⚠ {class_name} not available: {e}")
    
    def test_gui_components(self):
        """Test GUI component imports"""
        try:
            from rf_spectrum_analyzer.gui.main_window import MainWindow
            from rf_spectrum_analyzer.gui.spectrum_widget import SpectrumWidget
            from rf_spectrum_analyzer.gui.waterfall_widget import WaterfallWidget
            from rf_spectrum_analyzer.gui.controls_widget import ControlsWidget
            print("✓ GUI components imported successfully")
        except ImportError as e:
            print(f"⚠ GUI components not available: {e}")
    
    def test_version_compatibility(self):
        """Test library version compatibility"""
        version_checks = []
        
        try:
            import numpy as np
            numpy_version = np.__version__
            print(f"NumPy version: {numpy_version}")
            version_checks.append(('numpy', numpy_version))
        except ImportError:
            pass
        
        try:
            import scipy
            scipy_version = scipy.__version__
            print(f"SciPy version: {scipy_version}")
            version_checks.append(('scipy', scipy_version))
        except ImportError:
            pass
        
        try:
            from PySide6 import __version__ as pyside_version
            print(f"PySide6 version: {pyside_version}")
            version_checks.append(('PySide6', pyside_version))
        except ImportError:
            pass
        
        try:
            import pyqtgraph as pg
            pyqtgraph_version = pg.__version__
            print(f"PyQtGraph version: {pyqtgraph_version}")
            version_checks.append(('pyqtgraph', pyqtgraph_version))
        except ImportError:
            pass
        
        self.assertGreater(len(version_checks), 0, "No library versions detected")
    
    def tearDown(self):
        """Print import summary"""
        total_imports = len(self.import_results)
        successful_imports = sum(1 for success in self.import_results.values() if success)
        
        print(f"\nImport Summary: {successful_imports}/{total_imports} successful")
        
        if successful_imports < total_imports:
            print("Failed imports:")
            for module, success in self.import_results.items():
                if not success:
                    print(f"  - {module}")


class TestDependencyVersions(unittest.TestCase):
    """Test dependency version requirements"""
    
    def test_minimum_versions(self):
        """Check minimum required versions"""
        try:
            import numpy as np
            version_parts = np.__version__.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            self.assertGreaterEqual((major, minor), (1, 20), 
                                   f"NumPy version {np.__version__} too old")
        except ImportError:
            self.skipTest("NumPy not available")
        
        try:
            import scipy
            version_parts = scipy.__version__.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            self.assertGreaterEqual((major, minor), (1, 7),
                                   f"SciPy version {scipy.__version__} too old")
        except ImportError:
            self.skipTest("SciPy not available")


if __name__ == '__main__':
    unittest.main(verbosity=2)