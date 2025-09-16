"""
RF Spectrum Analyzer Package
============================

A comprehensive RF spectrum analyzer application with GUI and SDR backend support.

Modules:
    core: Core application components (app, SDR backend, signal processor)
    dsp: Digital signal processing components (filters, modulation, analysis)
    gui: User interface components (main window, widgets, controls)
    backends: SDR hardware backend implementations
    config: Configuration and settings
    utils: Utility functions and helpers
    resources: Application resources (icons, themes)
"""

__version__ = "1.0.0"
__author__ = "RF Spectrum Analyzer Development Team"
__license__ = "MIT"

# Package level imports for convenience
from rf_spectrum_analyzer.core.app import RFSpectrumAnalyzerApp
from rf_spectrum_analyzer.config.settings import Settings

__all__ = [
    'RFSpectrumAnalyzerApp',
    'Settings',
    '__version__',
    '__author__',
    '__license__'
]
