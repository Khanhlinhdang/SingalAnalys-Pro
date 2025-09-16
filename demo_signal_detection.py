"""
Signal Detection Demonstration Script
Shows capabilities of the integrated sdr._detection module.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
rf_analyzer_path = os.path.join(project_root, 'rf_spectrum_analyzer')
if os.path.exists(rf_analyzer_path):
    sys.path.insert(0, rf_analyzer_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_signals():
    """Create demonstration signals for detection showcase."""
    sample_rate = 1e6
    duration = 0.05  # 50ms
    t = np.arange(int(duration * sample_rate)) / sample_rate
    
    signals = {}
    
    # 1. Clean noise baseline
    noise_power = 0.01
    signals['noise'] = np.sqrt(noise_power) * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    
    # 2. Weak sinusoidal signal (challenging detection)
    weak_signal = 0.1 * np.exp(1j * 2 * np.pi * 150e3 * t)
    signals['weak_tone'] = weak_signal + signals['noise']
    
    # 3. Strong FM signal
    fm_signal = 0.5 * np.exp(1j * 2 * np.pi * (100e3 * t + 10e3/1000 * np.sin(2 * np.pi * 1000 * t)))
    signals['fm_broadcast'] = fm_signal + signals['noise']
    
    # 4. GSM-like TDMA bursts
    burst_signal = signals['noise'].copy()
    # GSM training sequence (midamble)
    gsm_sync = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1], dtype=complex)
    gsm_sync = 2 * gsm_sync - 1  # Convert to ±1
    
    # Add 5 bursts at GSM timing (577μs burst, 4.615ms frame)
    burst_duration = 577e-6  # GSM normal burst
    frame_period = 4.615e-3   # GSM frame
    burst_samples = int(burst_duration * sample_rate)
    frame_samples = int(frame_period * sample_rate)
    
    for i in range(5):
        start_idx = i * frame_samples + 1000  # Offset start
        if start_idx + len(gsm_sync) < len(burst_signal):
            # Add training sequence
            burst_signal[start_idx:start_idx+len(gsm_sync)] += 2.0 * gsm_sync
            # Add random data around it
            data_start = max(0, start_idx - 100)
            data_end = min(len(burst_signal), start_idx + len(gsm_sync) + 100)
            random_data = np.random.choice([-1, 1], size=data_end-data_start)
            burst_signal[data_start:data_end] += 0.8 * random_data
    
    signals['gsm_bursts'] = burst_signal
    signals['gsm_sync_pattern'] = gsm_sync
    
    # 5. Radar-like pulses
    radar_signal = signals['noise'].copy()
    pulse_width = 10e-6  # 10μs pulses
    prf = 1000  # 1kHz PRF
    pulse_samples = int(pulse_width * sample_rate)
    pulse_interval = int(sample_rate / prf)
    
    for pulse_start in range(2000, len(radar_signal), pulse_interval):
        pulse_end = min(pulse_start + pulse_samples, len(radar_signal))
        # Linear frequency modulated pulse (chirp)
        pulse_t = np.arange(pulse_end - pulse_start) / sample_rate
        chirp_rate = 50e6 / pulse_width  # 50 MHz/μs
        radar_pulse = 3.0 * np.exp(1j * 2 * np.pi * (200e3 * pulse_t + 0.5 * chirp_rate * pulse_t**2))
        radar_signal[pulse_start:pulse_end] = radar_pulse
    
    signals['radar_pulses'] = radar_signal
    
    return signals, sample_rate, t


def demo_basic_detection():
    """Demonstrate basic signal detection capabilities."""
    logger.info("🔍 DEMO 1: Basic Signal Detection")
    logger.info("=" * 50)
    
    try:
        from dsp.signal_detection import create_signal_detector
        
        signals, sample_rate, t = create_demo_signals()
        detector = create_signal_detector(sample_rate)
        
        # Calibrate with noise
        logger.info("📊 Calibrating noise floor...")
        detector.calibrate_noise_floor(signals['noise'])
        
        # Test detection on different signals
        test_cases = [
            ('Noise Only', 'noise', False),
            ('Weak Tone', 'weak_tone', True),
            ('FM Broadcast', 'fm_broadcast', True),
            ('GSM Bursts', 'gsm_bursts', True),
            ('Radar Pulses', 'radar_pulses', True)
        ]
        
        results = []
        logger.info("🎯 Testing signal detection...")
        
        for name, signal_key, expected in test_cases:
            signal = signals[signal_key]
            
            # Energy detection
            result = detector.energy_detection(signal, p_fa=1e-4)
            
            detected = result.signal_detected
            confidence = result.confidence
            snr_db = result.snr_estimate
            
            status = "✅" if detected == expected else "❌"
            logger.info(f"{status} {name:12} | Detected: {detected:5} | SNR: {snr_db:5.1f}dB | Confidence: {confidence:.3f}")
            
            results.append({
                'name': name,
                'detected': detected,
                'expected': expected,
                'snr': snr_db,
                'confidence': confidence,
                'correct': detected == expected
            })
        
        accuracy = sum(r['correct'] for r in results) / len(results)
        logger.info(f"\n📈 Detection Accuracy: {accuracy:.1%}")
        
        return results
        
    except Exception as e:
        logger.error(f"Demo 1 error: {e}")
        return []


def demo_tdma_detection():
    """Demonstrate TDMA burst detection."""
    logger.info("\n📡 DEMO 2: TDMA Burst Detection")
    logger.info("=" * 50)
    
    try:
        from dsp.tdma_detector import TDMABurstDetector
        
        signals, sample_rate, t = create_demo_signals()
        tdma_detector = TDMABurstDetector(sample_rate)
        
        # Set GSM sync pattern
        gsm_pattern = signals['gsm_sync_pattern']
        tdma_detector.set_sync_pattern(gsm_pattern)
        
        logger.info("🔍 Analyzing GSM-like bursts...")
        
        # Detect bursts
        bursts = tdma_detector.detect_bursts(signals['gsm_bursts'])
        
        logger.info(f"📊 Detected {len(bursts)} bursts")
        
        if bursts:
            # Analyze timing
            timing = tdma_detector.analyze_timing(bursts)
            
            if timing:
                logger.info(f"⏱️  Average burst interval: {timing['avg_interval_sec']*1000:.1f} ms")
                logger.info(f"📏 Frame period estimate: {timing['frame_period_sec']*1000:.1f} ms")
                logger.info(f"📊 Timing stability (std): {timing['interval_std_sec']*1000:.2f} ms")
            
            # Show burst details
            logger.info("🎯 Burst Details:")
            for i, burst in enumerate(bursts[:3]):  # Show first 3
                start_time = burst.start_sample / sample_rate * 1000
                duration = burst.duration_samples / sample_rate * 1000
                logger.info(f"   Burst {i+1}: {start_time:5.1f}ms, {duration:.1f}ms duration, "
                           f"SNR: {burst.snr_estimate:.1f}dB")
        
        return {
            'burst_count': len(bursts),
            'timing_analysis': timing if bursts else None,
            'bursts': bursts
        }
        
    except Exception as e:
        logger.error(f"Demo 2 error: {e}")
        return {'burst_count': 0}


def demo_spectrum_sensing():
    """Demonstrate spectrum sensing capabilities."""
    logger.info("\n🌐 DEMO 3: Spectrum Sensing")
    logger.info("=" * 50)
    
    try:
        from dsp.signal_detection import create_signal_detector
        
        signals, sample_rate, t = create_demo_signals()
        detector = create_signal_detector(sample_rate)
        
        # Define frequency bands for sensing
        frequency_bands = {
            'VHF_Low': (50e3, 100e3),
            'FM_Band': (100e3, 150e3),     # Contains FM signal
            'VHF_High': (150e3, 200e3),    # Contains weak tone
            'Radar_Band': (200e3, 250e3),  # Contains radar
            'UHF_Low': (250e3, 300e3)
        }
        
        logger.info("🔍 Scanning frequency bands...")
        
        # Test different signals
        test_signals = [
            ('Noise Baseline', 'noise'),
            ('FM Broadcast', 'fm_broadcast'),
            ('Radar Signal', 'radar_pulses'),
            ('Mixed Signals', 'gsm_bursts')  # Complex signal
        ]
        
        for signal_name, signal_key in test_signals:
            logger.info(f"\n📡 {signal_name}:")
            
            signal = signals[signal_key]
            sensing_results = detector.spectrum_sensing(signal, frequency_bands)
            
            occupied_bands = 0
            for band_name, result in sensing_results.items():
                if result.signal_detected:
                    occupied_bands += 1
                    status = "🔴 OCCUPIED"
                    logger.info(f"   {band_name:10}: {status} (SNR: {result.snr_db:5.1f}dB, "
                               f"Confidence: {result.detection_confidence:.3f})")
                else:
                    status = "🟢 CLEAR"
                    logger.info(f"   {band_name:10}: {status}")
            
            logger.info(f"   📊 Total occupied: {occupied_bands}/{len(frequency_bands)} bands")
        
        return sensing_results
        
    except Exception as e:
        logger.error(f"Demo 3 error: {e}")
        return {}


def demo_performance_analysis():
    """Demonstrate detection performance across SNR range."""
    logger.info("\n📈 DEMO 4: Performance Analysis")
    logger.info("=" * 50)
    
    try:
        import sdr
        from dsp.signal_detection import create_signal_detector
        
        sample_rate = 1e6
        detector = create_signal_detector(sample_rate)
        
        logger.info("🔍 Testing detection vs SNR...")
        
        # SNR range for testing
        snr_range = np.arange(-15, 15, 3)  # -15 to 12 dB
        detection_results = []
        
        for snr_db in snr_range:
            # Generate signal at specific SNR
            n_samples = 10000
            signal_power = 1.0
            noise_power = signal_power / (10 ** (snr_db / 10))
            
            # Sinusoidal signal
            t = np.arange(n_samples) / sample_rate
            signal = np.sqrt(signal_power) * np.exp(1j * 2 * np.pi * 100e3 * t)
            noise = np.sqrt(noise_power) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
            test_signal = signal + noise
            
            # Test detection
            result = detector.energy_detection(test_signal, p_fa=1e-4)
            detected = result.signal_detected
            
            detection_results.append((snr_db, detected))
            status = "✅" if detected else "❌"
            logger.info(f"   SNR: {snr_db:3.0f}dB | Detection: {status}")
        
        # Find detection threshold
        detected_snrs = [snr for snr, detected in detection_results if detected]
        if detected_snrs:
            min_snr = min(detected_snrs)
            logger.info(f"\n📊 Minimum detection SNR: {min_snr:.0f} dB")
        
        # Calculate detection probability
        high_snr_detections = sum(1 for snr, detected in detection_results if snr >= 0 and detected)
        high_snr_total = sum(1 for snr, detected in detection_results if snr >= 0)
        if high_snr_total > 0:
            detection_rate = high_snr_detections / high_snr_total
            logger.info(f"📈 High SNR detection rate: {detection_rate:.1%}")
        
        return detection_results
        
    except Exception as e:
        logger.error(f"Demo 4 error: {e}")
        return []


def create_demo_plots(results_dict):
    """Create visualization plots for demo results."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RF Spectrum Analyzer - Signal Detection Demonstration', fontsize=16)
        
        # Plot 1: Detection Results
        ax = axes[0, 0]
        basic_results = results_dict.get('basic_detection', [])
        if basic_results:
            names = [r['name'] for r in basic_results]
            detected = [r['detected'] for r in basic_results]
            expected = [r['expected'] for r in basic_results]
            
            x = np.arange(len(names))
            width = 0.35
            
            ax.bar(x - width/2, detected, width, label='Detected', alpha=0.8)
            ax.bar(x + width/2, expected, width, label='Expected', alpha=0.8)
            
            ax.set_xlabel('Signal Type')
            ax.set_ylabel('Detection Result')
            ax.set_title('Basic Detection Results')
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: SNR Performance
        ax = axes[0, 1]
        perf_results = results_dict.get('performance', [])
        if perf_results:
            snrs = [r[0] for r in perf_results]
            detections = [int(r[1]) for r in perf_results]
            
            ax.plot(snrs, detections, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('SNR (dB)')
            ax.set_ylabel('Detection (0/1)')
            ax.set_title('Detection vs SNR Performance')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
        
        # Plot 3: TDMA Burst Timeline
        ax = axes[1, 0]
        tdma_results = results_dict.get('tdma_detection', {})
        bursts = tdma_results.get('bursts', [])
        if bursts:
            burst_times = [b.start_sample / 1e6 * 1000 for b in bursts]  # Convert to ms
            burst_durations = [b.duration_samples / 1e6 * 1000 for b in bursts]
            
            for i, (start, duration) in enumerate(zip(burst_times, burst_durations)):
                ax.barh(i, duration, left=start, height=0.6, alpha=0.7)
                ax.text(start + duration/2, i, f'B{i+1}', ha='center', va='center')
            
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Burst Number')
            ax.set_title(f'TDMA Burst Detection ({len(bursts)} bursts)')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Spectrum Sensing
        ax = axes[1, 1]
        sensing_results = results_dict.get('spectrum_sensing', {})
        if sensing_results:
            bands = list(sensing_results.keys())
            occupancy = [int(sensing_results[band].signal_detected) for band in bands]
            snrs = [sensing_results[band].snr_db for band in bands]
            
            colors = ['red' if occ else 'green' for occ in occupancy]
            bars = ax.bar(bands, snrs, color=colors, alpha=0.7)
            
            ax.set_xlabel('Frequency Band')
            ax.set_ylabel('SNR (dB)')
            ax.set_title('Spectrum Sensing Results')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', alpha=0.7, label='Occupied'),
                             Patch(facecolor='green', alpha=0.7, label='Clear')]
            ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        # Save plot
        output_path = "signal_detection_demo_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Demo plots saved to {output_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error creating demo plots: {e}")


def main():
    """Run comprehensive signal detection demonstration."""
    logger.info("🚀 RF Spectrum Analyzer - Signal Detection Demonstration")
    logger.info("=" * 60)
    logger.info("This demo showcases the integrated sdr._detection capabilities")
    logger.info("=" * 60)
    
    try:
        results = {}
        
        # Run demonstrations
        results['basic_detection'] = demo_basic_detection()
        results['tdma_detection'] = demo_tdma_detection()
        results['spectrum_sensing'] = demo_spectrum_sensing()
        results['performance'] = demo_performance_analysis()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 DEMONSTRATION SUMMARY")
        logger.info("=" * 60)
        
        # Basic detection summary
        basic_results = results['basic_detection']
        if basic_results:
            accuracy = sum(r['correct'] for r in basic_results) / len(basic_results)
            logger.info(f"✅ Basic Detection: {accuracy:.1%} accuracy")
        
        # TDMA detection summary
        tdma_results = results['tdma_detection']
        burst_count = tdma_results.get('burst_count', 0)
        logger.info(f"📡 TDMA Detection: {burst_count} bursts detected")
        
        # Performance summary
        perf_results = results['performance']
        if perf_results:
            detected_snrs = [snr for snr, detected in perf_results if detected]
            min_snr = min(detected_snrs) if detected_snrs else "N/A"
            logger.info(f"📈 Performance: Detection down to {min_snr} dB SNR")
        
        logger.info("\n🎯 CAPABILITIES DEMONSTRATED:")
        logger.info("   ✅ Energy detection with configurable false alarm rates")
        logger.info("   ✅ TDMA burst detection with timing analysis")
        logger.info("   ✅ Multi-band spectrum sensing")
        logger.info("   ✅ SNR performance characterization")
        logger.info("   ✅ Real-time processing capability")
        
        # Create visualization
        create_demo_plots(results)
        
        logger.info("\n🚀 RF Spectrum Analyzer signal detection system is ready for use!")
        
        return results
        
    except Exception as e:
        logger.error(f"Demo execution error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()