#!/usr/bin/env python3
"""
Test script cho tính năng constellation validation
Kiểm tra các chức năng:
1. Enhanced signal generation với validation mode
2. Constellation analysis và EVM calculation  
3. Demodulator performance assessment
4. Quality metrics và validation reports
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_signal_generator import EnhancedSignalGenerator
from enhanced_processing_pipeline import EnhancedProcessingPipeline

def test_constellation_validation():
    """Test constellation validation features"""
    print("🧪 TESTING CONSTELLATION VALIDATION FEATURES")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing components...")
    generator = EnhancedSignalGenerator()
    pipeline = EnhancedProcessingPipeline()
    
    # Test configuration
    test_configs = [
        {
            'modulation': 'qpsk',
            'coding': 'convolutional',
            'data_length': 1000,
            'snr_db': 15,
            'description': 'QPSK with Convolutional Coding (Good SNR)'
        },
        {
            'modulation': '16qam', 
            'coding': 'turbo',
            'data_length': 1000,
            'snr_db': 10,
            'description': '16-QAM with Turbo Coding (Medium SNR)'
        },
        {
            'modulation': 'bpsk',
            'coding': 'none',
            'data_length': 500,
            'snr_db': 5,
            'description': 'BPSK without Coding (Low SNR)'
        }
    ]
    
    # Test each configuration
    for i, config in enumerate(test_configs, 1):
        print(f"\n{i}. Testing: {config['description']}")
        print("-" * 50)
        
        # Generate signal with validation enabled
        print(f"   📡 Generating {config['modulation']} signal...")
        signal_config = {
            'modulation_type': config['modulation'],
            'channel_coding': config['coding'],
            'data_length': config['data_length'],
            'snr_db': config['snr_db'],
            'validation_mode': True,  # Enable validation
            'reference_data_tracking': True
        }
        
        try:
            # Generate signal
            signal_result = generator.generate_signal(config=signal_config, num_bits=config['data_length'])
            if signal_result is None:
                print(f"   ❌ Signal generation failed")
                continue
                
            signal_data = signal_result['signal']
            signal_info = signal_result
            print(f"   ✅ Signal generated: {len(signal_data)} samples")
            
            # Extract validation info
            if 'validation_info' in signal_info:
                val_info = signal_info['validation_info']
                print(f"   📊 Reference constellation: {len(val_info.get('reference_constellation', []))} points")
                print(f"   🔢 Original data: {len(val_info.get('original_data', []))} bits")
            
            # Process signal through pipeline
            print(f"   🔄 Processing through 5-stage pipeline...")
            results = pipeline.process_signal(signal_data)
            
            # Check demodulation stage for constellation analysis
            demod_result = results.get('stage_2_demodulation', {})
            if 'constellation_analysis' in demod_result:
                analysis = demod_result['constellation_analysis']
                print(f"   📈 Constellation Analysis Results:")
                
                # EVM metrics
                if 'evm_percent' in analysis:
                    evm = analysis['evm_percent']
                    print(f"      • EVM: {evm:.2f}%")
                    
                    # Quality assessment
                    if evm < 5:
                        quality = "Excellent"
                    elif evm < 10:
                        quality = "Good" 
                    elif evm < 20:
                        quality = "Fair"
                    else:
                        quality = "Poor"
                    print(f"      • Quality: {quality}")
                
                # Validation metrics
                if 'validation_metrics' in analysis:
                    val_metrics = analysis['validation_metrics']
                    
                    accuracy = val_metrics.get('constellation_accuracy', 0)
                    print(f"      • Constellation Accuracy: {accuracy:.1f}%")
                    
                    snr_est = val_metrics.get('snr_estimate_db', 0)
                    print(f"      • Estimated SNR: {snr_est:.1f} dB")
                    
                    performance = val_metrics.get('demodulator_performance', 'unknown')
                    print(f"      • Demodulator Performance: {performance}")
                    
                    qual_assess = val_metrics.get('constellation_quality', 'unknown')
                    print(f"      • Overall Quality: {qual_assess}")
                
                # Cluster separation
                if 'cluster_separation' in analysis:
                    cluster_info = analysis['cluster_separation']
                    if isinstance(cluster_info, dict) and 'separation_ratio' in cluster_info:
                        sep_ratio = cluster_info['separation_ratio']
                        print(f"      • Cluster Separation: {sep_ratio:.2f}")
                
                # Data pattern analysis
                if 'data_pattern_analysis' in analysis:
                    pattern_info = analysis['data_pattern_analysis']
                    if isinstance(pattern_info, dict):
                        pattern = pattern_info.get('pattern_detected', 'unknown')
                        print(f"      • Data Pattern: {pattern}")
                        
                        match_rate = pattern_info.get('match_rate', 0)
                        print(f"      • Pattern Match Rate: {match_rate:.1%}")
            
            # Overall pipeline success
            completed_stages = sum(1 for s in results.values() if s.get('status') == 'completed')
            total_stages = len(results)
            success_rate = (completed_stages / total_stages) * 100
            print(f"   🎯 Pipeline Success Rate: {success_rate:.1f}% ({completed_stages}/{total_stages})")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✅ CONSTELLATION VALIDATION TEST COMPLETED")
    print("\n🔍 Summary:")
    print("   • Enhanced signal generation with validation tracking")
    print("   • Constellation analysis with EVM calculation") 
    print("   • Demodulator performance assessment")
    print("   • Quality metrics and validation reports")
    print("   • Data pattern analysis and match rate calculation")

if __name__ == "__main__":
    test_constellation_validation()
