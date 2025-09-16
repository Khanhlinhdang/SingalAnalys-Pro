#!/usr/bin/env python3
"""
RF Spectrum Analyzer - Main Entry Point
Integrates pyspectrum, mhostetter/sdr, and scikit-dsp-comm libraries
with PySide6/PyQtGraph GUI for RF signal analysis and processing.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QIcon

from rf_spectrum_analyzer.core.app import RFSpectrumAnalyzerApp
from rf_spectrum_analyzer.utils.logger import setup_application_logging
from rf_spectrum_analyzer.config.settings import Settings


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RF Spectrum Analyzer')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging')
    parser.add_argument('--config', type=str, 
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, 
                       help='SDR device to use (hackrf, rtlsdr, pluto, soapy)')
    parser.add_argument('--sample-rate', type=float, 
                       help='Sample rate in Hz')
    parser.add_argument('--frequency', type=float, 
                       help='Center frequency in Hz')
    parser.add_argument('--gain', type=float, 
                       help='RF gain in dB')
    return parser.parse_args()


def setup_application():
    """Setup Qt application with proper attributes."""
    # Set high DPI attributes before creating QApplication
    # Note: These attributes are handled automatically in Qt 6.0+
    # but we keep them for compatibility
    try:
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        # These attributes might not exist in newer Qt versions
        pass
    
    app = QApplication(sys.argv)
    app.setApplicationName("RF Spectrum Analyzer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("RF Signal Processing")
    
    # Set application icon
    icon_path = project_root / "resources" / "icons" / "app_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    return app


def main():
    """Main application entry point."""
    logger = None  # Initialize logger variable for exception handling
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Setup logging
        log_level = 'DEBUG' if args.debug else 'INFO'
        setup_application_logging(level=log_level)
        
        logger = logging.getLogger(__name__)
        logger.info("Starting RF Spectrum Analyzer...")
        
        # Load configuration
        settings = Settings()
        if args.config:
            settings.load_from_file(args.config)
        
        # Override settings with command line arguments
        if args.device:
            settings.sdr.device_type = args.device
        if args.sample_rate:
            settings.sdr.sample_rate = args.sample_rate
        if args.frequency:
            settings.sdr.center_frequency = args.frequency
        if args.gain:
            settings.sdr.gain = args.gain
        
        # Setup Qt application
        qt_app = setup_application()
        
        # Create and run main application
        rf_app = RFSpectrumAnalyzerApp(settings)
        # Main window is shown from within the app initialization
        
        logger.info("RF Spectrum Analyzer started successfully")
        
        # Run Qt event loop
        exit_code = qt_app.exec()
        
        logger.info("RF Spectrum Analyzer shutting down...")
        return exit_code
        
    except KeyboardInterrupt:
        if logger:
            logger.info("Application interrupted by user")
        else:
            print("Application interrupted by user")
        return 0
    except Exception as e:
        if logger:
            logger.error(f"Fatal error: {e}", exc_info=True)
        else:
            print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())