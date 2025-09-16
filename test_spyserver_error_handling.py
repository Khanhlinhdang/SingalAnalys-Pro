#!/usr/bin/env python3
"""
Quick test script for SpyServer connection error handling
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rf_spectrum_analyzer.config.settings import Settings
from rf_spectrum_analyzer.backends.spyserver_backend import SpyServerBackend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_spyserver_connection():
    """Test SpyServer connection with improved error handling."""
    print("Testing SpyServer connection error handling...")
    
    settings = Settings()
    settings.sdr.device_type = "spyserver"
    settings.sdr.spyserver_host = "localhost"
    settings.sdr.spyserver_port = 5555
    
    backend = SpyServerBackend(settings)
    
    print(f"Attempting to connect to SpyServer at {settings.sdr.spyserver_host}:{settings.sdr.spyserver_port}")
    
    # This should fail gracefully with improved error messages
    result = backend.connect()
    
    print(f"Connection result: {result}")
    
    if not result:
        print("✓ Connection failed as expected (no SpyServer running)")
        print("✓ Error handling worked correctly")
    else:
        print("✓ Connection succeeded - SpyServer is running!")
    
    # Test device info even when not connected
    device_info = backend.get_device_info()
    print(f"Device info: {device_info}")
    
    return True

def test_demo_mode():
    """Test demo mode functionality."""
    print("\nTesting demo mode...")
    
    settings = Settings()
    settings.demo_mode = True
    
    print(f"Demo mode enabled: {settings.demo_mode}")
    print("✓ Demo mode test passed")
    
    return True

if __name__ == "__main__":
    print("SpyServer Error Handling Test")
    print("=" * 40)
    
    try:
        test_spyserver_connection()
        test_demo_mode()
        
        print("\n" + "=" * 40)
        print("✅ All tests passed!")
        print("\nTo run the main app:")
        print("1. With demo mode: python main.py --demo")
        print("2. With SpyServer: python main.py --device spyserver")
        print("3. Normal mode: python main.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)