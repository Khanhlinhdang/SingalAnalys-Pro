"""
Simple signal detection test without complex imports.
"""

import numpy as np
import logging
import sys
import os

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sdr_detection_availability():
    """Test if sdr._detection module is available."""
    logger.info("=== Testing SDR Detection Module Availability ===")
    
    try:
        import sdr
        logger.info("✓ sdr library imported successfully")
        
        # Test EnergyDetector
        if hasattr(sdr, 'EnergyDetector'):
            logger.info("✓ EnergyDetector available")
            
            # Test threshold calculation
            try:
                threshold = sdr.EnergyDetector.threshold(
                    N=1000,
                    p_fa=1e-6,
                    sigma2=1.0,
                    complex=True
                )
                logger.info(f"✓ EnergyDetector.threshold(): {threshold:.3f}")
            except Exception as e:
                logger.error(f"✗ EnergyDetector.threshold() failed: {e}")
            
            # Test p_d calculation
            try:
                p_d = sdr.EnergyDetector.p_d(
                    snr=10.0,
                    N_nc=1000,
                    p_fa=1e-6,
                    complex=True
                )
                logger.info(f"✓ EnergyDetector.p_d(): {p_d:.3f}")
            except Exception as e:
                logger.error(f"✗ EnergyDetector.p_d() failed: {e}")
        else:
            logger.error("✗ EnergyDetector not available")
        
        # Test ReplicaCorrelator
        if hasattr(sdr, 'ReplicaCorrelator'):
            logger.info("✓ ReplicaCorrelator available")
            
            # Test threshold calculation
            try:
                threshold = sdr.ReplicaCorrelator.threshold(
                    p_fa=1e-6,
                    energy=1.0,
                    sigma2=1.0,
                    complex=True
                )
                logger.info(f"✓ ReplicaCorrelator.threshold(): {threshold:.3f}")
            except Exception as e:
                logger.error(f"✗ ReplicaCorrelator.threshold() failed: {e}")
            
            # Test p_d calculation
            try:
                p_d = sdr.ReplicaCorrelator.p_d(
                    enr=10.0,
                    p_fa=1e-6,
                    complex=True
                )
                logger.info(f"✓ ReplicaCorrelator.p_d(): {p_d:.3f}")
            except Exception as e:
                logger.error(f"✗ ReplicaCorrelator.p_d() failed: {e}")
        else:
            logger.error("✗ ReplicaCorrelator not available")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ sdr library not available: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return False


def test_basic_signal_detection():
    """Test basic signal detection without custom classes."""
    logger.info("=== Testing Basic Signal Detection ===")
    
    try:
        import sdr
        import scipy.stats
        
        # Generate test signals
        np.random.seed(42)  # For reproducible results
        n_samples = 10000
        sample_rate = 1e6
        
        # Pure noise
        noise_power = 1.0
        noise = np.sqrt(noise_power) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
        
        # Signal in noise
        signal_power = 3.0
        signal = np.sqrt(signal_power) * np.ones(n_samples, dtype=complex)
        signal_in_noise = signal + noise
        
        logger.info(f"Generated {n_samples} samples")
        logger.info(f"Noise power: {noise_power}, Signal power: {signal_power}")
        logger.info(f"Expected SNR: {10*np.log10(signal_power/noise_power):.1f} dB")
        
        # Energy detection parameters
        integration_length = 1000
        p_fa = 1e-6
        
        # Test on noise only
        noise_energy = np.sum(np.abs(noise[:integration_length]) ** 2)
        noise_threshold = sdr.EnergyDetector.threshold(
            N=integration_length,
            p_fa=p_fa,
            sigma2=noise_power,
            complex=True
        )
        noise_detected = noise_energy > noise_threshold
        
        logger.info(f"Noise test: Energy={noise_energy:.1f}, Threshold={noise_threshold:.1f}, "
                   f"Detected={noise_detected}")
        
        # Test on signal in noise
        signal_energy = np.sum(np.abs(signal_in_noise[:integration_length]) ** 2)
        signal_threshold = sdr.EnergyDetector.threshold(
            N=integration_length,
            p_fa=p_fa,
            sigma2=noise_power,
            complex=True
        )
        signal_detected = signal_energy > signal_threshold
        
        # Calculate actual SNR
        signal_power_measured = signal_energy / integration_length
        snr_measured = 10 * np.log10(signal_power_measured / noise_power)
        
        logger.info(f"Signal test: Energy={signal_energy:.1f}, Threshold={signal_threshold:.1f}, "
                   f"Detected={signal_detected}, SNR={snr_measured:.1f} dB")
        
        # Test correlation detection with known pattern
        pattern = np.array([1, 1, -1, 1, -1, -1, 1], dtype=complex)
        
        # Create signal with pattern
        pattern_signal = np.zeros(n_samples, dtype=complex)
        pattern_signal[100:100+len(pattern)] = pattern * np.sqrt(signal_power)
        pattern_with_noise = pattern_signal + noise
        
        # Cross-correlation
        correlation = np.correlate(pattern_with_noise, pattern, mode='full')
        max_corr = np.max(np.abs(correlation))
        
        # Simple threshold
        pattern_energy = np.sum(np.abs(pattern) ** 2)
        corr_threshold = np.sqrt(pattern_energy * noise_power) * 3  # Simple threshold
        pattern_detected = max_corr > corr_threshold
        
        logger.info(f"Pattern test: Max correlation={max_corr:.1f}, Threshold={corr_threshold:.1f}, "
                   f"Detected={pattern_detected}")
        
        # Summary
        tests_passed = sum([
            not noise_detected,  # Should not detect noise
            signal_detected,     # Should detect signal
            pattern_detected     # Should detect pattern
        ])
        
        logger.info(f"Basic detection tests: {tests_passed}/3 passed")
        
        return tests_passed >= 2
        
    except Exception as e:
        logger.error(f"Basic detection test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spectrum_analysis():
    """Test basic spectrum analysis capabilities."""
    logger.info("=== Testing Spectrum Analysis ===")
    
    try:
        # Generate test signal with multiple frequency components
        sample_rate = 1e6
        duration = 0.01
        t = np.arange(int(duration * sample_rate)) / sample_rate
        
        # Multi-tone signal
        f1, f2, f3 = 100e3, 200e3, 350e3
        signal = (np.exp(1j * 2 * np.pi * f1 * t) + 
                 0.5 * np.exp(1j * 2 * np.pi * f2 * t) + 
                 0.3 * np.exp(1j * 2 * np.pi * f3 * t))
        
        # Add noise
        noise_power = 0.1
        noise = np.sqrt(noise_power) * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        signal_with_noise = signal + noise
        
        # FFT analysis
        fft_result = np.fft.fft(signal_with_noise)
        freqs = np.fft.fftfreq(len(signal_with_noise), 1/sample_rate)
        power_spectrum = np.abs(fft_result) ** 2
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(power_spectrum[:len(power_spectrum)//2], height=np.max(power_spectrum)/10)
        peak_freqs = freqs[peaks]
        
        logger.info(f"Expected frequencies: {f1/1e3:.1f}, {f2/1e3:.1f}, {f3/1e3:.1f} kHz")
        logger.info(f"Detected peaks at: {[f/1e3 for f in peak_freqs]:.1f} kHz")
        
        # Check if we detected the expected frequencies (within tolerance)
        tolerance = 5e3  # 5 kHz tolerance
        detected_f1 = any(abs(f - f1) < tolerance for f in peak_freqs)
        detected_f2 = any(abs(f - f2) < tolerance for f in peak_freqs)
        detected_f3 = any(abs(f - f3) < tolerance for f in peak_freqs)
        
        detections = sum([detected_f1, detected_f2, detected_f3])
        logger.info(f"Spectrum analysis: {detections}/3 frequencies detected")
        
        return detections >= 2
        
    except Exception as e:
        logger.error(f"Spectrum analysis test error: {e}")
        return False


def test_modulation_recognition():
    """Test basic modulation recognition."""
    logger.info("=== Testing Modulation Recognition ===")
    
    try:
        sample_rate = 1e6
        symbol_rate = 10e3
        n_symbols = 1000
        samples_per_symbol = int(sample_rate / symbol_rate)
        
        # Generate BPSK signal
        bits = np.random.randint(0, 2, n_symbols)
        bpsk_symbols = 2 * bits - 1  # Map to ±1
        bpsk_signal = np.repeat(bpsk_symbols, samples_per_symbol)
        
        # Generate QPSK signal
        qpsk_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], n_symbols) / np.sqrt(2)
        qpsk_signal = np.repeat(qpsk_symbols, samples_per_symbol)
        
        # Add noise
        noise_power = 0.1
        def add_noise(sig):
            noise = np.sqrt(noise_power) * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
            return sig + noise
        
        bpsk_noisy = add_noise(bpsk_signal)
        qpsk_noisy = add_noise(qpsk_signal)
        
        # Simple modulation analysis based on constellation properties
        def analyze_constellation(signal, samples_per_symbol):
            # Downsample to symbol rate
            symbols = signal[::samples_per_symbol]
            
            # Calculate constellation properties
            magnitude_std = np.std(np.abs(symbols))
            phase_std = np.std(np.angle(symbols))
            
            # Count distinct levels
            magnitudes = np.abs(symbols)
            mag_threshold = np.std(magnitudes) / 2
            unique_mags = len(np.unique(np.round(magnitudes / mag_threshold)))
            
            phases = np.angle(symbols)
            phase_threshold = np.pi / 8
            unique_phases = len(np.unique(np.round(phases / phase_threshold)))
            
            return {
                'magnitude_std': magnitude_std,
                'phase_std': phase_std,
                'unique_magnitudes': unique_mags,
                'unique_phases': unique_phases
            }
        
        # Analyze both signals
        bpsk_analysis = analyze_constellation(bpsk_noisy, samples_per_symbol)
        qpsk_analysis = analyze_constellation(qpsk_noisy, samples_per_symbol)
        
        logger.info(f"BPSK analysis: mag_std={bpsk_analysis['magnitude_std']:.3f}, "
                   f"phase_std={bpsk_analysis['phase_std']:.3f}, "
                   f"unique_phases={bpsk_analysis['unique_phases']}")
        
        logger.info(f"QPSK analysis: mag_std={qpsk_analysis['magnitude_std']:.3f}, "
                   f"phase_std={qpsk_analysis['phase_std']:.3f}, "
                   f"unique_phases={qpsk_analysis['unique_phases']}")
        
        # Simple classification rules
        bpsk_likely = bpsk_analysis['unique_phases'] <= 4  # BPSK has 2 phases (± noise)
        qpsk_likely = qpsk_analysis['unique_phases'] >= 6  # QPSK has 4 phases (± noise)
        
        correct_classifications = sum([bpsk_likely, qpsk_likely])
        logger.info(f"Modulation recognition: {correct_classifications}/2 correct")
        
        return correct_classifications >= 1
        
    except Exception as e:
        logger.error(f"Modulation recognition test error: {e}")
        return False


def main():
    """Run simple signal detection tests."""
    logger.info("Starting Simple Signal Detection Tests")
    logger.info("=" * 50)
    
    test_results = []
    
    # Test 1: SDR detection module availability
    sdr_available = test_sdr_detection_availability()
    test_results.append(("SDR Detection Module", sdr_available))
    
    # Test 2: Basic signal detection
    basic_detection = test_basic_signal_detection()
    test_results.append(("Basic Signal Detection", basic_detection))
    
    # Test 3: Spectrum analysis
    spectrum_analysis = test_spectrum_analysis()
    test_results.append(("Spectrum Analysis", spectrum_analysis))
    
    # Test 4: Modulation recognition
    modulation_recognition = test_modulation_recognition()
    test_results.append(("Modulation Recognition", modulation_recognition))
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
        if result:
            passed_tests += 1
    
    success_rate = passed_tests / len(test_results)
    logger.info("=" * 50)
    logger.info(f"OVERALL RESULT: {passed_tests}/{len(test_results)} tests passed ({success_rate:.1%})")
    
    if success_rate >= 0.75:
        logger.info("🎉 EXCELLENT: Signal detection capabilities are working well!")
    elif success_rate >= 0.5:
        logger.info("⚠️  GOOD: Signal detection capabilities are mostly working")
    else:
        logger.info("❌ NEEDS WORK: Signal detection capabilities need improvement")
    
    return test_results


if __name__ == "__main__":
    main()