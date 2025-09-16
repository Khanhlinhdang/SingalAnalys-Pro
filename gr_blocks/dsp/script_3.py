# Create main.py - the main entry point for the RF spectrum analyzer
main_content = '''#!/usr/bin/env python3
"""
RF Spectrum Analyzer - Main Entry Point

Advanced RF signal acquisition, analysis, and processing software that integrates:
- pyspectrum: Real-time FFT spectrum analysis and SDR input
- mhostetter/sdr: Digital signal processing, filtering, modulation/demodulation
- scikit-dsp-comm: Advanced DSP, synchronization, FEC coding

Author: RF Spectrum Team
License: MIT
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QIcon

from core.app import RFSpectrumApp
from config.settings import AppSettings
from utils.logger import setup_logging


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RF Spectrum Analyzer - Advanced SDR Signal Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --source pluto --freq 433.92e6 --samplerate 2e6
  python main.py --source rtlsdr --freq 915e6 --gain 30
  python main.py --source file --input test_signal.wav
  python main.py --debug --verbose --log-level DEBUG
        """
    )
    
    # SDR source options
    parser.add_argument(
        "--source", "-s",
        choices=["auto", "pluto", "rtlsdr", "hackrf", "soapy", "audio", "file"],
        default="auto",
        help="SDR source type (default: auto-detect)"
    )
    
    # Frequency and sampling parameters
    parser.add_argument(
        "--freq", "-f",
        type=float,
        default=433.92e6,
        help="Center frequency in Hz (default: 433.92 MHz)"
    )
    
    parser.add_argument(
        "--samplerate", "-r",
        type=float,
        default=2e6,
        help="Sample rate in Hz (default: 2 MSps)"
    )
    
    parser.add_argument(
        "--gain", "-g",
        type=float,
        default=30,
        help="RF gain in dB (default: 30 dB)"
    )
    
    parser.add_argument(
        "--bandwidth", "-b",
        type=float,
        help="Bandwidth in Hz (default: auto)"
    )
    
    # File input options
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input file path (for file source)"
    )
    
    # Processing options
    parser.add_argument(
        "--fft-size",
        type=int,
        default=2048,
        help="FFT size (default: 2048)"
    )
    
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="FFT overlap ratio (default: 0.5)"
    )
    
    # Debug and logging options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv, -vvv)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: logs/rf_spectrum.log)"
    )
    
    # GUI options
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (for processing only)"
    )
    
    parser.add_argument(
        "--theme",
        choices=["light", "dark", "auto"],
        default="auto",
        help="GUI theme (default: auto)"
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available SDR devices and exit"
    )
    
    return parser.parse_args()


def setup_application():
    """Setup Qt application with proper attributes"""
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    
    app = QApplication(sys.argv)
    app.setApplicationName("RF Spectrum Analyzer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("RF Spectrum Team")
    app.setOrganizationDomain("rfspectrum.dev")
    
    # Set application icon
    icon_path = project_root / "resources" / "icons" / "app_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    return app


def list_devices():
    """List available SDR devices"""
    print("\\n=== Available SDR Devices ===\\n")
    
    try:
        from backends.soapy_backend import SoapyBackend
        devices = SoapyBackend.enumerate_devices()
        if devices:
            print("SoapySDR Devices:")
            for i, device in enumerate(devices):
                print(f"  [{i}] {device}")
        else:
            print("No SoapySDR devices found")
    except Exception as e:
        print(f"SoapySDR error: {e}")
    
    try:
        from backends.rtlsdr_backend import RTLSDRBackend
        devices = RTLSDRBackend.enumerate_devices()
        if devices:
            print("\\nRTL-SDR Devices:")
            for i, device in enumerate(devices):
                print(f"  [{i}] {device}")
        else:
            print("No RTL-SDR devices found")
    except Exception as e:
        print(f"RTL-SDR error: {e}")
    
    try:
        from backends.pluto_backend import PlutoBackend
        devices = PlutoBackend.enumerate_devices()
        if devices:
            print("\\nPluto SDR Devices:")
            for i, device in enumerate(devices):
                print(f"  [{i}] {device}")
        else:
            print("No Pluto SDR devices found")
    except Exception as e:
        print(f"Pluto SDR error: {e}")
    
    print("\\n=== End of Device List ===\\n")


def main():
    """Main application entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    if args.verbose >= 3:
        log_level = logging.DEBUG
    elif args.verbose >= 2:
        log_level = logging.INFO
    elif args.verbose >= 1:
        log_level = logging.WARNING
    
    log_file = args.log_file or (project_root / "logs" / "rf_spectrum.log")
    setup_logging(log_level, log_file, args.debug)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting RF Spectrum Analyzer")
    
    # List devices if requested
    if args.list_devices:
        list_devices()
        return 0
    
    try:
        if args.headless:
            # Headless mode - command line processing only
            logger.info("Running in headless mode")
            from core.signal_processor import SignalProcessor
            
            # Initialize signal processor
            processor = SignalProcessor()
            
            # Configure parameters from command line
            config = {
                'source': args.source,
                'freq': args.freq,
                'samplerate': args.samplerate,
                'gain': args.gain,
                'fft_size': args.fft_size,
                'overlap': args.overlap
            }
            
            if args.input:
                config['input_file'] = args.input
            if args.bandwidth:
                config['bandwidth'] = args.bandwidth
            
            # Run processing
            processor.run_headless(config)
            
        else:
            # GUI mode
            logger.info("Starting GUI application")
            
            # Setup Qt application
            app = setup_application()
            
            # Load application settings
            settings = AppSettings(args.config)
            
            # Apply theme
            if args.theme != "auto":
                settings.set_theme(args.theme)
            
            # Create main application
            rf_app = RFSpectrumApp(settings, args)
            
            # Show main window
            rf_app.show()
            
            logger.info("Application started successfully")
            
            # Run application
            return app.exec()
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''

with open("rf_spectrum_analyzer/main.py", "w") as f:
    f.write(main_content)

print("Created main.py")