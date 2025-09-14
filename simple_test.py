#!/usr/bin/env python3
"""
Simple test script để debug signal generation
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_signal_generator import EnhancedSignalGenerator

def simple_test():
    """Simple test of signal generation"""
    print("🧪 SIMPLE SIGNAL GENERATION TEST")
    print("=" * 50)
    
    # Initialize generator
    print("1. Initializing signal generator...")
    generator = EnhancedSignalGenerator()
    
    print("2. Current config:")
    for key, value in generator.current_config.items():
        print(f"   {key}: {value}")
    
    print("\n3. Testing basic signal generation...")
    try:
        # Test without any config change
        result = generator.generate_signal(num_bits=100)
        
        if result is None:
            print("   ❌ Generation failed")
            return
            
        print(f"   ✅ Generated signal with {len(result['signal'])} samples")
        print(f"   📊 Data bits: {len(result['data_bits'])}")
        print(f"   📊 Coded bits: {len(result['coded_bits'])}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n4. Testing with config change...")
    try:
        # Test with simple config
        config = {
            'modulation_type': 'qpsk',
            'channel_coding': 'none',
            'snr_db': 20
        }
        
        result = generator.generate_signal(config=config, num_bits=100)
        
        if result is None:
            print("   ❌ Generation failed")
            return
            
        print(f"   ✅ Generated QPSK signal with {len(result['signal'])} samples")
        print(f"   📊 Data bits: {len(result['data_bits'])}")
        print(f"   📊 Coded bits: {len(result['coded_bits'])}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()
