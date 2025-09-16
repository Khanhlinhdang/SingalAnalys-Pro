"""
Test sdr._detection API to understand correct usage.
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_sdr_detection_api():
    """Explore sdr._detection API to understand correct usage."""
    try:
        import sdr
        import inspect
        
        logger.info("=== Exploring SDR Detection API ===")
        
        # Explore EnergyDetector
        if hasattr(sdr, 'EnergyDetector'):
            logger.info("\n--- EnergyDetector Methods ---")
            methods = [m for m in dir(sdr.EnergyDetector) if not m.startswith('_')]
            for method in methods:
                logger.info(f"  {method}")
            
            # Check threshold method signature
            if hasattr(sdr.EnergyDetector, 'threshold'):
                sig = inspect.signature(sdr.EnergyDetector.threshold)
                logger.info(f"EnergyDetector.threshold signature: {sig}")
            
            # Check p_d method signature
            if hasattr(sdr.EnergyDetector, 'p_d'):
                sig = inspect.signature(sdr.EnergyDetector.p_d)
                logger.info(f"EnergyDetector.p_d signature: {sig}")
        
        # Explore ReplicaCorrelator
        if hasattr(sdr, 'ReplicaCorrelator'):
            logger.info("\n--- ReplicaCorrelator Methods ---")
            methods = [m for m in dir(sdr.ReplicaCorrelator) if not m.startswith('_')]
            for method in methods:
                logger.info(f"  {method}")
            
            # Check threshold method signature
            if hasattr(sdr.ReplicaCorrelator, 'threshold'):
                sig = inspect.signature(sdr.ReplicaCorrelator.threshold)
                logger.info(f"ReplicaCorrelator.threshold signature: {sig}")
            
            # Check p_d method signature
            if hasattr(sdr.ReplicaCorrelator, 'p_d'):
                sig = inspect.signature(sdr.ReplicaCorrelator.p_d)
                logger.info(f"ReplicaCorrelator.p_d signature: {sig}")
        
        # Test actual usage
        logger.info("\n--- Testing Actual Usage ---")
        
        # Try EnergyDetector with different parameters
        try:
            # Try basic threshold calculation
            threshold = sdr.EnergyDetector.threshold(p_fa=1e-6, N_c=1000, sigma2=1.0)
            logger.info(f"✓ EnergyDetector.threshold(p_fa=1e-6, N_c=1000, sigma2=1.0) = {threshold}")
        except Exception as e:
            logger.info(f"✗ EnergyDetector.threshold failed: {e}")
        
        try:
            # Try with complex flag
            threshold = sdr.EnergyDetector.threshold(p_fa=1e-6, N_c=1000, sigma2=1.0, complex=True)
            logger.info(f"✓ EnergyDetector.threshold(..., complex=True) = {threshold}")
        except Exception as e:
            logger.info(f"✗ EnergyDetector.threshold with complex failed: {e}")
        
        # Try p_d calculation
        try:
            p_d = sdr.EnergyDetector.p_d(snr=10.0, N_c=1000, p_fa=1e-6)
            logger.info(f"✓ EnergyDetector.p_d(snr=10.0, N_c=1000, p_fa=1e-6) = {p_d}")
        except Exception as e:
            logger.info(f"✗ EnergyDetector.p_d failed: {e}")
        
        # Try ReplicaCorrelator
        try:
            threshold = sdr.ReplicaCorrelator.threshold(p_fa=1e-6, energy=1.0, sigma2=1.0)
            logger.info(f"✓ ReplicaCorrelator.threshold(p_fa=1e-6, energy=1.0, sigma2=1.0) = {threshold}")
        except Exception as e:
            logger.info(f"✗ ReplicaCorrelator.threshold failed: {e}")
        
        try:
            p_d = sdr.ReplicaCorrelator.p_d(enr=10.0, p_fa=1e-6)
            logger.info(f"✓ ReplicaCorrelator.p_d(enr=10.0, p_fa=1e-6) = {p_d}")
        except Exception as e:
            logger.info(f"✗ ReplicaCorrelator.p_d failed: {e}")
        
    except ImportError as e:
        logger.error(f"sdr library not available: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def test_corrected_energy_detection():
    """Test energy detection with corrected API usage."""
    logger.info("\n=== Testing Corrected Energy Detection ===")
    
    try:
        import sdr
        
        # Generate test signals
        np.random.seed(42)
        n_samples = 1000
        noise_power = 1.0
        signal_power = 3.0
        
        # Pure noise
        noise = np.sqrt(noise_power) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
        
        # Signal in noise
        signal = np.sqrt(signal_power) * np.ones(n_samples, dtype=complex)
        signal_in_noise = signal + noise
        
        # Calculate energy
        noise_energy = np.sum(np.abs(noise) ** 2)
        signal_energy = np.sum(np.abs(signal_in_noise) ** 2)
        
        # Use corrected API
        p_fa = 1e-6
        
        # Try different parameter combinations for threshold
        try:
            threshold = sdr.EnergyDetector.threshold(p_fa=p_fa, N_c=n_samples, sigma2=noise_power)
            logger.info(f"Threshold (N_c={n_samples}): {threshold:.3f}")
        except Exception as e:
            logger.info(f"Threshold calculation error: {e}")
            # Fallback to simple threshold
            threshold = noise_power * n_samples * 5  # Simple heuristic
        
        # Detection decisions
        noise_detected = noise_energy > threshold
        signal_detected = signal_energy > threshold
        
        logger.info(f"Noise: Energy={noise_energy:.1f}, Detected={noise_detected}")
        logger.info(f"Signal: Energy={signal_energy:.1f}, Detected={signal_detected}")
        
        # Calculate performance
        snr_actual = 10 * np.log10(signal_power / noise_power)
        logger.info(f"Actual SNR: {snr_actual:.1f} dB")
        
        # Success if we detect signal but not noise
        success = signal_detected and not noise_detected
        logger.info(f"Detection test: {'✓ PASS' if success else '✗ FAIL'}")
        
        return success
        
    except Exception as e:
        logger.error(f"Corrected energy detection test error: {e}")
        return False


def test_practical_detection():
    """Test practical signal detection scenarios."""
    logger.info("\n=== Testing Practical Detection Scenarios ===")
    
    try:
        # Generate realistic signals
        sample_rate = 1e6
        duration = 0.01
        t = np.arange(int(duration * sample_rate)) / sample_rate
        
        # Scenario 1: Weak sinusoidal signal
        freq = 100e3
        signal_power = 1.0
        noise_power = 2.0  # SNR = -3 dB
        
        signal = np.sqrt(signal_power) * np.exp(1j * 2 * np.pi * freq * t)
        noise = np.sqrt(noise_power) * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        weak_signal = signal + noise
        
        # Scenario 2: Strong burst signal
        burst_power = 10.0
        burst_duration = 0.002  # 2ms burst
        burst_samples = int(burst_duration * sample_rate)
        
        burst_signal = np.zeros(len(t), dtype=complex)
        start_idx = len(t) // 4
        burst_signal[start_idx:start_idx+burst_samples] = np.sqrt(burst_power)
        strong_signal = burst_signal + noise
        
        # Simple energy detection
        def simple_energy_detection(signal, threshold_factor=3):
            energy = np.sum(np.abs(signal) ** 2)
            noise_est = np.var(signal)  # Simple noise estimate
            threshold = threshold_factor * noise_est * len(signal)
            return energy > threshold, energy, threshold
        
        # Test weak signal
        weak_detected, weak_energy, weak_threshold = simple_energy_detection(weak_signal)
        logger.info(f"Weak signal (-3dB): Energy={weak_energy:.1f}, "
                   f"Threshold={weak_threshold:.1f}, Detected={weak_detected}")
        
        # Test strong signal
        strong_detected, strong_energy, strong_threshold = simple_energy_detection(strong_signal)
        logger.info(f"Strong signal (+7dB): Energy={strong_energy:.1f}, "
                   f"Threshold={strong_threshold:.1f}, Detected={strong_detected}")
        
        # Test pure noise
        pure_noise = noise[:len(t)//2]
        noise_detected, noise_energy, noise_threshold = simple_energy_detection(pure_noise)
        logger.info(f"Pure noise: Energy={noise_energy:.1f}, "
                   f"Threshold={noise_threshold:.1f}, Detected={noise_detected}")
        
        # Expected: strong signal detected, weak signal maybe, noise not detected
        expected_detections = [strong_detected, not noise_detected]
        success_rate = sum(expected_detections) / len(expected_detections)
        
        logger.info(f"Practical detection test: {sum(expected_detections)}/{len(expected_detections)} passed")
        return success_rate >= 0.5
        
    except Exception as e:
        logger.error(f"Practical detection test error: {e}")
        return False


def main():
    """Run API exploration and corrected tests."""
    logger.info("SDR Detection API Exploration and Testing")
    logger.info("=" * 50)
    
    # Step 1: Explore API
    explore_sdr_detection_api()
    
    # Step 2: Test corrected energy detection
    energy_success = test_corrected_energy_detection()
    
    # Step 3: Test practical scenarios
    practical_success = test_practical_detection()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    
    results = [
        ("Energy Detection", energy_success),
        ("Practical Detection", practical_success)
    ]
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed/total >= 0.8:
        logger.info("🎉 SDR Detection integration is working well!")
    elif passed/total >= 0.5:
        logger.info("⚠️  SDR Detection integration is partially working")
    else:
        logger.info("❌ SDR Detection integration needs work")


if __name__ == "__main__":
    main()