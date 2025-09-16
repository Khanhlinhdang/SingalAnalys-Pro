"""
Comprehensive test suite for signal detection capabilities.
Tests sdr._detection integration and TDMA burst detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Add rf_spectrum_analyzer to path
rf_analyzer_path = os.path.join(project_root, 'rf_spectrum_analyzer')
if os.path.exists(rf_analyzer_path):
    sys.path.insert(0, rf_analyzer_path)

try:
    from rf_spectrum_analyzer.core.signal_processor import SignalProcessor
    from rf_spectrum_analyzer.config.settings import Settings
    from rf_spectrum_analyzer.dsp.signal_detection import create_signal_detector
    from rf_spectrum_analyzer.dsp.tdma_detector import TDMABurstDetector
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Try alternative imports
    try:
        sys.path.append('.')
        from core.signal_processor import SignalProcessor
        from config.settings import Settings
        from dsp.signal_detection import create_signal_detector
        from dsp.tdma_detector import TDMABurstDetector
    except ImportError as e2:
        logger.error(f"Alternative import also failed: {e2}")
        sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_test_signals(sample_rate: float = 1e6) -> dict:
    """Generate various test signals for detection testing."""
    duration = 0.01  # 10ms signals
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    
    signals = {}
    
    # 1. Pure noise
    noise_power = 0.1
    signals['noise'] = np.sqrt(noise_power) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    
    # 2. BPSK signal in noise
    bpsk_power = 1.0
    bits = np.random.randint(0, 2, size=100)
    bpsk_symbols = 2 * bits - 1  # Map to ±1
    samples_per_symbol = n_samples // len(bits)
    bpsk_signal = np.repeat(bpsk_symbols, samples_per_symbol)[:n_samples]
    bpsk_noise = np.sqrt(noise_power) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    signals['bpsk_in_noise'] = np.sqrt(bpsk_power) * bpsk_signal + bpsk_noise
    
    # 3. QPSK signal
    qpsk_power = 1.5
    qpsk_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=100) / np.sqrt(2)
    qpsk_signal = np.repeat(qpsk_symbols, samples_per_symbol)[:n_samples]
    qpsk_noise = np.sqrt(noise_power) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    signals['qpsk_in_noise'] = np.sqrt(qpsk_power) * qpsk_signal + qpsk_noise
    
    # 4. Sinusoidal signal (narrowband)
    sine_freq = 100e3  # 100 kHz
    sine_power = 2.0
    sine_signal = np.sqrt(sine_power) * np.exp(1j * 2 * np.pi * sine_freq * t)
    sine_noise = np.sqrt(noise_power) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    signals['sine_in_noise'] = sine_signal + sine_noise
    
    # 5. TDMA-like burst signal
    burst_duration = 0.002  # 2ms bursts
    burst_samples = int(burst_duration * sample_rate)
    burst_power = 3.0
    
    # Create sync pattern (Barker sequence)
    sync_pattern = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1], dtype=complex)
    
    # Create burst signal with 3 bursts
    tdma_signal = np.zeros(n_samples, dtype=complex)
    burst_starts = [1000, 4000, 7000]
    
    for start_idx in burst_starts:
        if start_idx + burst_samples < n_samples:
            # Add sync pattern
            sync_end = min(start_idx + len(sync_pattern), n_samples)
            tdma_signal[start_idx:sync_end] = sync_pattern[:sync_end-start_idx]
            
            # Add random data
            data_start = start_idx + len(sync_pattern)
            data_end = min(start_idx + burst_samples, n_samples)
            if data_start < data_end:
                data_bits = np.random.choice([-1, 1], size=data_end-data_start)
                tdma_signal[data_start:data_end] = data_bits
    
    tdma_noise = np.sqrt(noise_power) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    signals['tdma_bursts'] = np.sqrt(burst_power) * tdma_signal + tdma_noise
    
    # Store sync pattern for TDMA testing
    signals['sync_pattern'] = sync_pattern
    
    return signals


def test_energy_detection():
    """Test energy detection capabilities."""
    logger.info("=== Testing Energy Detection ===")
    
    sample_rate = 1e6
    detector = create_signal_detector(sample_rate)
    signals = generate_test_signals(sample_rate)
    
    # Calibrate with noise
    noise_samples = signals['noise']
    detector.calibrate_noise_floor(noise_samples)
    
    # Test detection on different signals
    test_cases = [
        ('noise', 1e-6, False),
        ('bpsk_in_noise', 1e-6, True),
        ('qpsk_in_noise', 1e-6, True),
        ('sine_in_noise', 1e-6, True),
        ('tdma_bursts', 1e-6, True)
    ]
    
    results = []
    for signal_name, p_fa, expected_detection in test_cases:
        signal = signals[signal_name]
        result = detector.energy_detection(signal, p_fa)
        
        status = "✓" if result.signal_detected == expected_detection else "✗"
        logger.info(f"{status} {signal_name}: Detected={result.signal_detected}, "
                   f"SNR={result.snr_estimate:.1f}dB, Confidence={result.confidence:.3f}")
        
        results.append({
            'signal': signal_name,
            'expected': expected_detection,
            'detected': result.signal_detected,
            'snr': result.snr_estimate,
            'confidence': result.confidence,
            'success': result.signal_detected == expected_detection
        })
    
    success_rate = sum(r['success'] for r in results) / len(results)
    logger.info(f"Energy Detection Success Rate: {success_rate:.1%}")
    
    return results


def test_correlation_detection():
    """Test correlation detection capabilities."""
    logger.info("=== Testing Correlation Detection ===")
    
    sample_rate = 1e6
    detector = create_signal_detector(sample_rate)
    signals = generate_test_signals(sample_rate)
    
    # Add sync pattern as template
    sync_pattern = signals['sync_pattern']
    detector.add_signal_template('sync', sync_pattern)
    
    # Test correlation detection
    test_cases = [
        ('noise', False),
        ('tdma_bursts', True),  # Should detect sync pattern
        ('bpsk_in_noise', False),  # No sync pattern present
    ]
    
    results = []
    for signal_name, expected_detection in test_cases:
        signal = signals[signal_name]
        result = detector.correlation_detection(signal, 'sync')
        
        status = "✓" if result.signal_detected == expected_detection else "✗"
        logger.info(f"{status} {signal_name}: Detected={result.signal_detected}, "
                   f"Confidence={result.confidence:.3f}")
        
        results.append({
            'signal': signal_name,
            'expected': expected_detection,
            'detected': result.signal_detected,
            'confidence': result.confidence,
            'success': result.signal_detected == expected_detection
        })
    
    success_rate = sum(r['success'] for r in results) / len(results)
    logger.info(f"Correlation Detection Success Rate: {success_rate:.1%}")
    
    return results


def test_tdma_burst_detection():
    """Test TDMA burst detection."""
    logger.info("=== Testing TDMA Burst Detection ===")
    
    sample_rate = 1e6
    tdma_detector = TDMABurstDetector(sample_rate)
    signals = generate_test_signals(sample_rate)
    
    # Set sync pattern
    sync_pattern = signals['sync_pattern']
    tdma_detector.set_sync_pattern(sync_pattern)
    
    # Test burst detection
    signal = signals['tdma_bursts']
    bursts = tdma_detector.detect_bursts(signal)
    
    logger.info(f"Detected {len(bursts)} bursts")
    
    # Analyze timing if bursts found
    if bursts:
        timing_analysis = tdma_detector.analyze_timing(bursts)
        logger.info(f"Timing Analysis:")
        logger.info(f"  Average burst interval: {timing_analysis['avg_interval']:.1f} samples")
        logger.info(f"  Interval std dev: {timing_analysis['interval_std']:.1f} samples")
        logger.info(f"  Frame period estimate: {timing_analysis['frame_period']:.1f} samples")
        
        # Extract burst data
        for i, burst in enumerate(bursts[:3]):  # Show first 3 bursts
            burst_data = tdma_detector.extract_burst_data(signal, burst)
            if burst_data is not None:
                logger.info(f"  Burst {i+1}: {len(burst_data)} samples extracted")
    
    # Expected: 3 bursts (as generated)
    expected_bursts = 3
    success = abs(len(bursts) - expected_bursts) <= 1  # Allow ±1 burst tolerance
    
    status = "✓" if success else "✗"
    logger.info(f"{status} TDMA Detection: Expected ~{expected_bursts}, Found {len(bursts)}")
    
    return {
        'expected_bursts': expected_bursts,
        'detected_bursts': len(bursts),
        'success': success,
        'timing_analysis': timing_analysis if bursts else None
    }


def test_spectrum_sensing():
    """Test spectrum sensing across frequency bands."""
    logger.info("=== Testing Spectrum Sensing ===")
    
    sample_rate = 1e6
    detector = create_signal_detector(sample_rate)
    signals = generate_test_signals(sample_rate)
    
    # Define frequency bands for sensing
    frequency_bands = {
        'band_1': (0, 50e3),        # Lower band
        'band_2': (50e3, 150e3),    # Contains sine signal at 100kHz
        'band_3': (150e3, 250e3),   # Upper band
        'band_4': (250e3, 350e3),   # Highest band
    }
    
    # Test with sine signal (should detect in band_2)
    signal = signals['sine_in_noise']
    sensing_results = detector.spectrum_sensing(signal, frequency_bands)
    
    logger.info("Spectrum Sensing Results:")
    for band_name, result in sensing_results.items():
        logger.info(f"  {band_name}: Signal={'Yes' if result.signal_detected else 'No'}, "
                   f"SNR={result.snr_db:.1f}dB, Confidence={result.detection_confidence:.3f}")
    
    # Expected: detection mainly in band_2 (contains 100kHz sine)
    band_2_detected = sensing_results['band_2'].signal_detected
    other_bands_quiet = not any(sensing_results[band].signal_detected 
                               for band in ['band_1', 'band_3', 'band_4'])
    
    success = band_2_detected  # At minimum, should detect in band_2
    status = "✓" if success else "✗"
    logger.info(f"{status} Spectrum Sensing: Band 2 detected={band_2_detected}")
    
    return {
        'band_results': sensing_results,
        'target_band_detected': band_2_detected,
        'success': success
    }


def test_signal_processor_integration():
    """Test signal processor integration with detection capabilities."""
    logger.info("=== Testing Signal Processor Integration ===")
    
    # Create settings
    settings = Settings()
    settings.sdr.sample_rate = 1e6
    
    # Create signal processor
    processor = SignalProcessor(settings)
    signals = generate_test_signals(settings.sdr.sample_rate)
    
    # Test 1: Calibrate detector
    noise_samples = signals['noise']
    calib_result = processor.calibrate_detector(noise_samples)
    logger.info(f"Calibration: Success={calib_result['success']}, "
               f"Noise variance={calib_result.get('noise_variance', 'N/A')}")
    
    # Test 2: Signal detection
    signal = signals['bpsk_in_noise']
    detection_result = processor.detect_signal(signal, method="energy")
    logger.info(f"Signal Detection: Detected={detection_result.get('signal_detected', False)}, "
               f"SNR={detection_result.get('snr_estimate', 'N/A'):.1f}dB")
    
    # Test 3: Add template and test correlation
    sync_pattern = signals['sync_pattern']
    processor.add_signal_template('test_sync', sync_pattern)
    
    corr_result = processor.detect_signal(signals['tdma_bursts'], method="correlation")
    logger.info(f"Correlation Detection: Detected={corr_result.get('signal_detected', False)}, "
               f"Confidence={corr_result.get('confidence', 'N/A'):.3f}")
    
    # Test 4: TDMA burst detection
    tdma_result = processor.detect_tdma_bursts(signals['tdma_bursts'], sync_pattern)
    logger.info(f"TDMA Detection: {tdma_result.get('burst_count', 0)} bursts found")
    
    # Test 5: Spectrum sensing
    frequency_bands = {
        'low_band': (0, 50e3),
        'mid_band': (50e3, 150e3),
        'high_band': (150e3, 250e3)
    }
    sensing_result = processor.spectrum_sensing(signals['sine_in_noise'], frequency_bands)
    occupied_bands = sensing_result.get('occupied_bands', 0)
    logger.info(f"Spectrum Sensing: {occupied_bands}/{len(frequency_bands)} bands occupied")
    
    # Test 6: Detection statistics
    stats = processor.get_detection_statistics()
    logger.info(f"Detection Statistics: {stats.get('total_detections', 0)} total detections")
    
    # Overall success
    tests_passed = sum([
        calib_result.get('success', False),
        detection_result.get('success', False),
        corr_result.get('success', False),
        tdma_result.get('success', False),
        sensing_result.get('success', False)
    ])
    
    success_rate = tests_passed / 5
    status = "✓" if success_rate >= 0.8 else "✗"
    logger.info(f"{status} Integration Tests: {tests_passed}/5 passed ({success_rate:.1%})")
    
    return {
        'calibration': calib_result,
        'detection': detection_result,
        'correlation': corr_result,
        'tdma': tdma_result,
        'sensing': sensing_result,
        'statistics': stats,
        'success_rate': success_rate
    }


def create_detection_plots(test_results: dict):
    """Create visualization plots for detection results."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Signal Detection Test Results', fontsize=16)
        
        # Plot 1: Energy Detection Results
        ax = axes[0, 0]
        energy_results = test_results.get('energy_detection', [])
        if energy_results:
            signals = [r['signal'] for r in energy_results]
            snrs = [r['snr'] for r in energy_results]
            confidences = [r['confidence'] for r in energy_results]
            
            x = np.arange(len(signals))
            ax.bar(x - 0.2, snrs, 0.4, label='SNR (dB)', alpha=0.7)
            ax.bar(x + 0.2, [c*20 for c in confidences], 0.4, label='Confidence (×20)', alpha=0.7)
            
            ax.set_xlabel('Signal Type')
            ax.set_ylabel('Value')
            ax.set_title('Energy Detection Results')
            ax.set_xticks(x)
            ax.set_xticklabels(signals, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Correlation Detection Results
        ax = axes[0, 1]
        corr_results = test_results.get('correlation_detection', [])
        if corr_results:
            signals = [r['signal'] for r in corr_results]
            confidences = [r['confidence'] for r in corr_results]
            detected = [r['detected'] for r in corr_results]
            
            colors = ['green' if d else 'red' for d in detected]
            ax.bar(signals, confidences, color=colors, alpha=0.7)
            ax.set_xlabel('Signal Type')
            ax.set_ylabel('Confidence')
            ax.set_title('Correlation Detection Results')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: TDMA Burst Detection
        ax = axes[1, 0]
        tdma_result = test_results.get('tdma_detection', {})
        if tdma_result:
            expected = tdma_result.get('expected_bursts', 0)
            detected = tdma_result.get('detected_bursts', 0)
            
            ax.bar(['Expected', 'Detected'], [expected, detected], 
                  color=['blue', 'orange'], alpha=0.7)
            ax.set_ylabel('Number of Bursts')
            ax.set_title('TDMA Burst Detection')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Spectrum Sensing
        ax = axes[1, 1]
        sensing_result = test_results.get('spectrum_sensing', {})
        if sensing_result and 'band_results' in sensing_result:
            bands = list(sensing_result['band_results'].keys())
            detected = [sensing_result['band_results'][band].signal_detected for band in bands]
            snrs = [sensing_result['band_results'][band].snr_db for band in bands]
            
            colors = ['green' if d else 'gray' for d in detected]
            ax.bar(bands, snrs, color=colors, alpha=0.7)
            ax.set_xlabel('Frequency Band')
            ax.set_ylabel('SNR (dB)')
            ax.set_title('Spectrum Sensing Results')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = "detection_test_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Detection test plots saved to {output_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error creating plots: {e}")


def main():
    """Run comprehensive signal detection tests."""
    logger.info("Starting Comprehensive Signal Detection Tests")
    logger.info("=" * 60)
    
    try:
        # Run all tests
        test_results = {}
        
        # Individual component tests
        test_results['energy_detection'] = test_energy_detection()
        test_results['correlation_detection'] = test_correlation_detection()
        test_results['tdma_detection'] = test_tdma_burst_detection()
        test_results['spectrum_sensing'] = test_spectrum_sensing()
        
        # Integration test
        test_results['integration'] = test_signal_processor_integration()
        
        # Summary
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        # Energy detection summary
        energy_success = sum(r['success'] for r in test_results['energy_detection']) / len(test_results['energy_detection'])
        logger.info(f"Energy Detection: {energy_success:.1%} success rate")
        
        # Correlation detection summary
        corr_success = sum(r['success'] for r in test_results['correlation_detection']) / len(test_results['correlation_detection'])
        logger.info(f"Correlation Detection: {corr_success:.1%} success rate")
        
        # TDMA detection summary
        tdma_success = test_results['tdma_detection']['success']
        logger.info(f"TDMA Detection: {'✓' if tdma_success else '✗'} {test_results['tdma_detection']['detected_bursts']} bursts found")
        
        # Spectrum sensing summary
        sensing_success = test_results['spectrum_sensing']['success']
        logger.info(f"Spectrum Sensing: {'✓' if sensing_success else '✗'} Target band detected")
        
        # Integration summary
        integration_success = test_results['integration']['success_rate']
        logger.info(f"Integration Tests: {integration_success:.1%} success rate")
        
        # Overall assessment
        overall_success = (energy_success + corr_success + integration_success) / 3
        if tdma_success:
            overall_success = (overall_success * 3 + 1) / 4
        if sensing_success:
            overall_success = (overall_success * 4 + 1) / 5
        
        logger.info("=" * 60)
        logger.info(f"OVERALL SUCCESS RATE: {overall_success:.1%}")
        
        if overall_success >= 0.8:
            logger.info("✅ SIGNAL DETECTION MODULE: EXCELLENT PERFORMANCE")
        elif overall_success >= 0.6:
            logger.info("⚠️  SIGNAL DETECTION MODULE: GOOD PERFORMANCE")
        else:
            logger.info("❌ SIGNAL DETECTION MODULE: NEEDS IMPROVEMENT")
        
        # Create visualization
        create_detection_plots(test_results)
        
        return test_results
        
    except Exception as e:
        logger.error(f"Test execution error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()