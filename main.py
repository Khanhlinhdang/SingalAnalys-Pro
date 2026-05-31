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

import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QIcon

from rf_spectrum_analyzer.core.app import RFSpectrumAnalyzerApp
from rf_spectrum_analyzer.dsp.signal_analysis import SignalAnalyzer
from rf_spectrum_analyzer.utils.file_io import DataImporter
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
                       help='SDR device to use (hackrf, rtlsdr, pluto, soapy, spyserver)')
    parser.add_argument('--sample-rate', type=float, 
                       help='Sample rate in Hz')
    parser.add_argument('--frequency', type=float, 
                       help='Center frequency in Hz')
    parser.add_argument('--gain', type=float, 
                       help='RF gain in dB')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode with synthetic data')
    parser.add_argument('--input-file', type=str,
                       help='Process a local signal source file before or instead of live SDR capture')
    parser.add_argument('--input-frequency', type=float,
                       help='Center frequency in Hz for an input file')
    parser.add_argument('--input-bandwidth', type=float,
                       help='Analysis bandwidth in Hz for an input file')
    parser.add_argument('--headless', action='store_true',
                       help='Process the input file and exit without entering the Qt event loop')
    return parser.parse_args()


def setup_application():
    """Setup Qt application with proper attributes."""
    # Set high DPI attributes before creating QApplication
    # Note: These attributes are handled automatically in Qt 6.0+
    # but we keep them for compatibility
    try:
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception as e:
        print(e)
    
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
        if args.demo:
            settings.demo_mode = True
            logger.info("Demo mode enabled via command line")
        if args.headless:
            logger.info("Headless mode enabled - live SDR startup will be skipped")
            settings.headless_mode = True

        if args.headless and args.input_file:
            logger.info("Running headless file-processing path without Qt UI")
            importer = DataImporter()
            iq_data, metadata = importer.import_signal_source(args.input_file)
            if iq_data is None or len(iq_data) == 0:
                logger.error(f"Failed to import signal source from {args.input_file}")
                return 1

            sample_rate = float(metadata.get('sample_rate') or settings.sdr.sample_rate)
            center_frequency = float(args.input_frequency if args.input_frequency is not None else settings.sdr.center_frequency)
            analysis_bandwidth = float(args.input_bandwidth if args.input_bandwidth is not None else max(sample_rate / 2.0, 1.0))
            analyzer = SignalAnalyzer(sample_rate)
            result = analyzer.analyze_signal_comprehensive(
                np.asarray(iq_data, dtype=np.complex64),
                center_frequency,
                analysis_bandwidth,
            )
            payload = result.get('payload', result)
            protocol_outputs = payload.get('protocol_outputs', {}) or {}
            protocol_results = protocol_outputs.get('results', []) or []
            locked_results = sum(1 for item in protocol_results if item.get('frame_locked'))
            stage_errors = payload.get('stage_errors', []) or []
            summary = (
                "HEADLESS_SUMMARY "
                f"modulation={payload.get('modulation', {}).get('type', 'Unknown')} "
                f"confidence={float(payload.get('modulation', {}).get('confidence', 0.0)):.2f} "
                f"protocol={protocol_outputs.get('matched_protocol', 'None')} "
                f"protocol_confidence={float(protocol_outputs.get('confidence', 0.0)):.2f} "
                f"results={len(protocol_results)} "
                f"locked={locked_results} "
                f"artifacts={len(protocol_outputs.get('artifacts', []) or [])} "
                f"stage_errors={len(stage_errors)}"
            )
            logger.info(
                "Headless analysis completed: modulation=%s confidence=%.2f artifacts=%d stage_errors=%d",
                payload.get('modulation', {}).get('type', 'Unknown'),
                float(payload.get('modulation', {}).get('confidence', 0.0)),
                len(protocol_outputs.get('artifacts', []) or []),
                len(stage_errors),
            )
            print(summary)
            return 0
        
        # Setup Qt application
        qt_app = setup_application()
        
        # Create and run main application
        rf_app = RFSpectrumAnalyzerApp(settings)
        # Main window is shown from within the app initialization

        if args.input_file:
            rf_app.process_signal_file(
                args.input_file,
                center_freq=args.input_frequency,
                bandwidth=args.input_bandwidth,
            )

            if args.headless:
                logger.info("Headless file processing completed")
                return 0
        
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