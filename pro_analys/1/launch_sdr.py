#!/usr/bin/env python3
"""
SDR Application Launcher
Script khởi động với kiểm tra dependencies
"""

import sys
import os
import subprocess
import importlib


def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher required")
        print(f"Current version: {sys.version}")
        return False
    return True


def check_dependencies():
    """Check required Python packages"""
    required_packages = [
        'PySide6',
        'pyqtgraph', 
        'numpy',
        'scipy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (missing)")

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False

    return True


def check_uhd():
    """Check UHD availability"""
    try:
        import uhd
        print(f"✓ UHD version: {uhd.get_version_string()}")
        return True
    except ImportError:
        print("✗ UHD Python bindings not found")
        print("Install UHD and enable Python API")
        return False


def check_usrp_connection():
    """Check USRP device availability (optional)"""
    try:
        import uhd
        usrp = uhd.usrp.MultiUSRP()
        device_info = usrp.get_pp_string()
        print(f"✓ USRP connected: {device_info.split()[0]}")
        return True
    except Exception as e:
        print(f"⚠ USRP not detected: {str(e)}")
        print("You can still run in simulation mode")
        return False


def main():
    """Main launcher"""
    print("SDR Application Launcher")
    print("=" * 40)

    # Check system requirements
    print("\nChecking system requirements...")
    if not check_python_version():
        sys.exit(1)

    print("\nChecking Python dependencies...")
    if not check_dependencies():
        sys.exit(1)

    print("\nChecking UHD...")
    uhd_available = check_uhd()

    print("\nChecking USRP connection...")
    usrp_available = check_usrp_connection()

    if not uhd_available:
        print("\nWarning: UHD not available. Some features will be disabled.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    print("\n" + "=" * 40)
    print("Starting SDR Application...")

    # Launch main application
    try:
        from sdr_application import main
        main()
    except ImportError:
        print("Error: sdr_application.py not found")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"\nApplication error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
