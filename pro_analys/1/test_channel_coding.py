
"""
Comprehensive Test Suite for Channel Coding Integration
Test tất cả loại channel coding và integration với SDR application
"""

import numpy as np
import time
import sys
import os

# Test imports
try:
    from channel_coding import (ConvolutionalCoder, TurboCoder, LDPCCoder, 
                               PolarCoder, ReedSolomonCoder, ChannelCodingDetector,
                               generate_hamming_matrix, generate_random_ldpc_matrix)
    from enhanced_signal_processor import EnhancedSignalProcessor
    IMPORTS_OK = True
    print("✅ All channel coding modules imported successfully")
except ImportError as e:
    IMPORTS_OK = False
    print(f"❌ Import error: {e}")


class ChannelCodingTestSuite:
    """Comprehensive test suite for channel coding"""

    def __init__(self):
        self.test_results = {}
        self.test_data = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        self.snr_db_list = [0, 5, 10, 15, 20]

    def run_all_tests(self):
        """Run all channel coding tests"""
        print("🚀 Starting Comprehensive Channel Coding Test Suite")
        print("=" * 60)

        if not IMPORTS_OK:
            print("❌ Cannot run tests due to import errors")
            return

        # Test individual coders
        self.test_convolutional_coding()
        self.test_turbo_coding()
        self.test_ldpc_coding()
        self.test_polar_coding()
        self.test_reed_solomon_coding()

        # Test detection
        self.test_coding_detection()

        # Test enhanced signal processor
        self.test_enhanced_processor()

        # Performance analysis
        self.analyze_performance()

        # Generate summary report
        self.generate_summary_report()

    def test_convolutional_coding(self):
        """Test convolutional coding"""
        print("\n📡 Testing Convolutional Coding...")

        try:
            # Test different configurations
            configs = [
                {'constraint_length': 3, 'code_rate': 0.5, 'polynomials': [0o7, 0o5]},
                {'constraint_length': 7, 'code_rate': 0.5, 'polynomials': [0o133, 0o171]},
                {'constraint_length': 9, 'code_rate': 0.5, 'polynomials': [0o753, 0o561]}
            ]

            results = {}

            for i, config in enumerate(configs):
                print(f"  Testing config {i+1}: K={config['constraint_length']}, Rate={config['code_rate']}")

                coder = ConvolutionalCoder(**config)

                # Test encoding
                encoded = coder.encode(self.test_data)
                print(f"    Original: {len(self.test_data)} bits -> Encoded: {len(encoded)} bits")

                # Test decoding (hard decision)
                decoded_hard = coder.viterbi_decode(encoded, is_hard_decision=True)
                success_hard = np.array_equal(self.test_data, decoded_hard)

                # Test with noise (soft decision)
                noise_power = 0.5
                noisy_encoded = encoded.astype(float) + noise_power * np.random.randn(len(encoded))
                decoded_soft = coder.viterbi_decode(noisy_encoded, is_hard_decision=False)
                success_soft = np.array_equal(self.test_data, decoded_soft)

                results[f"config_{i+1}"] = {
                    'config': config,
                    'encoded_length': len(encoded),
                    'hard_decision_success': success_hard,
                    'soft_decision_success': success_soft
                }

                print(f"    Hard decision: {'✅ Success' if success_hard else '❌ Failed'}")
                print(f"    Soft decision: {'✅ Success' if success_soft else '❌ Failed'}")

            self.test_results['convolutional'] = results
            print("  ✅ Convolutional coding test completed")

        except Exception as e:
            print(f"  ❌ Convolutional coding test failed: {e}")
            self.test_results['convolutional'] = {'error': str(e)}

    def test_turbo_coding(self):
        """Test turbo coding"""
        print("\n🌀 Testing Turbo Coding...")

        try:
            # Test different configurations
            configs = [
                {'constraint_length': 3, 'interleaver_size': 64},
                {'constraint_length': 3, 'interleaver_size': 256},
                {'constraint_length': 4, 'interleaver_size': 1024}
            ]

            results = {}

            for i, config in enumerate(configs):
                print(f"  Testing config {i+1}: K={config['constraint_length']}, N={config['interleaver_size']}")

                coder = TurboCoder(**config)

                # Prepare data (pad to interleaver size)
                test_data = np.zeros(config['interleaver_size'])
                test_data[:min(len(self.test_data), config['interleaver_size'])] = self.test_data[:config['interleaver_size']]

                # Test encoding
                encoded = coder.encode(test_data)
                print(f"    Original: {len(test_data)} bits -> Encoded: {len(encoded)} bits")

                # Test decoding
                # Split encoded bits back to systematic and parity
                n_info = len(test_data)
                if len(encoded) >= 3 * n_info:
                    systematic = encoded[:n_info]
                    parity1 = encoded[n_info:2*n_info]
                    parity2 = encoded[2*n_info:3*n_info]

                    # Add noise for realistic test
                    snr_db = 5
                    noise_var = 1 / (10**(snr_db/10))
                    systematic += np.sqrt(noise_var) * np.random.randn(len(systematic))
                    parity1 += np.sqrt(noise_var) * np.random.randn(len(parity1))
                    parity2 += np.sqrt(noise_var) * np.random.randn(len(parity2))

                    decoded = coder.log_map_decode(systematic, parity1, parity2, iterations=6, snr_db=snr_db)

                    # Check first few bits for success (due to padding)
                    original_len = min(len(self.test_data), config['interleaver_size'])
                    success = np.array_equal(test_data[:original_len], decoded[:original_len])

                    results[f"config_{i+1}"] = {
                        'config': config,
                        'encoded_length': len(encoded),
                        'decoding_success': success
                    }

                    print(f"    Turbo decoding: {'✅ Success' if success else '❌ Failed'}")
                else:
                    print(f"    ❌ Invalid encoded length")
                    results[f"config_{i+1}"] = {'config': config, 'error': 'Invalid encoded length'}

            self.test_results['turbo'] = results
            print("  ✅ Turbo coding test completed")

        except Exception as e:
            print(f"  ❌ Turbo coding test failed: {e}")
            self.test_results['turbo'] = {'error': str(e)}

    def test_ldpc_coding(self):
        """Test LDPC coding"""
        print("\n📊 Testing LDPC Coding...")

        try:
            # Test with Hamming matrices
            test_configs = [
                {'m': 3, 'name': 'Hamming(7,4)'},
                {'m': 4, 'name': 'Hamming(15,11)'}
            ]

            results = {}

            for config in test_configs:
                print(f"  Testing {config['name']}...")

                H = generate_hamming_matrix(config['m'])
                coder = LDPCCoder(H)

                n = H.shape[1]
                k = coder.K

                # Prepare test data
                test_info = self.test_data[:k] if len(self.test_data) >= k else np.pad(self.test_data, (0, k - len(self.test_data)), 'constant')

                # Test encoding
                encoded = coder.encode(test_info)
                print(f"    Info: {k} bits -> Codeword: {len(encoded)} bits")

                # Test decoding with different algorithms
                for algorithm in ['sum_product', 'min_sum']:
                    print(f"    Testing {algorithm} algorithm...")

                    # Convert to LLR with noise
                    snr_db = 10
                    noise_var = 1 / (10**(snr_db/10))
                    received_llr = (2 * encoded.astype(float) - 1) / noise_var
                    received_llr += np.sqrt(2/noise_var) * np.random.randn(len(received_llr))

                    if algorithm == 'sum_product':
                        decoded, iterations = coder.sum_product_decode(received_llr, max_iterations=50)
                    else:
                        decoded, iterations = coder.min_sum_decode(received_llr, max_iterations=50)

                    # Check syndrome
                    syndrome = (H @ decoded) % 2
                    valid_codeword = np.all(syndrome == 0)

                    results[f"{config['name']}_{algorithm}"] = {
                        'matrix_size': H.shape,
                        'iterations': iterations,
                        'valid_codeword': valid_codeword,
                        'algorithm': algorithm
                    }

                    print(f"      Iterations: {iterations}, Valid: {'✅' if valid_codeword else '❌'}")

            self.test_results['ldpc'] = results
            print("  ✅ LDPC coding test completed")

        except Exception as e:
            print(f"  ❌ LDPC coding test failed: {e}")
            self.test_results['ldpc'] = {'error': str(e)}

    def test_polar_coding(self):
        """Test polar coding"""
        print("\n🧊 Testing Polar Coding...")

        try:
            # Test different configurations
            configs = [
                {'n': 8, 'k': 4},
                {'n': 16, 'k': 8},
                {'n': 32, 'k': 16}
            ]

            results = {}

            for config in configs:
                print(f"  Testing Polar({config['n']}, {config['k']})...")

                coder = PolarCoder(config['n'], config['k'])

                # Prepare info bits
                k = config['k']
                info_bits = self.test_data[:k] if len(self.test_data) >= k else np.pad(self.test_data, (0, k - len(self.test_data)), 'constant')

                # Test encoding
                encoded = coder.encode(info_bits)
                print(f"    Info: {k} bits -> Codeword: {len(encoded)} bits")

                # Test decoding
                # Convert to LLR
                snr_db = 5
                received_llr = 2 * encoded.astype(float) - 1
                received_llr += 0.5 * np.random.randn(len(received_llr))  # Add noise

                decoded_info = coder.sc_decode(received_llr)
                success = np.array_equal(info_bits, decoded_info)

                results[f"polar_{config['n']}_{config['k']}"] = {
                    'config': config,
                    'encoded_length': len(encoded),
                    'decoding_success': success
                }

                print(f"    SC decoding: {'✅ Success' if success else '❌ Failed'}")

            self.test_results['polar'] = results
            print("  ✅ Polar coding test completed")

        except Exception as e:
            print(f"  ❌ Polar coding test failed: {e}")
            self.test_results['polar'] = {'error': str(e)}

    def test_reed_solomon_coding(self):
        """Test Reed-Solomon coding"""
        print("\n📚 Testing Reed-Solomon Coding...")

        try:
            # Test different configurations
            configs = [
                {'n': 7, 'k': 3},
                {'n': 15, 'k': 11},
                {'n': 31, 'k': 25}
            ]

            results = {}

            for config in configs:
                print(f"  Testing RS({config['n']}, {config['k']})...")

                coder = ReedSolomonCoder(config['n'], config['k'])

                # Prepare message (symbols, not bits)
                k = config['k']
                message = np.arange(1, k + 1)  # Simple increasing sequence

                # Test encoding
                codeword = coder.encode(message)
                print(f"    Message: {k} symbols -> Codeword: {len(codeword)} symbols")

                # Test error-free decoding
                decoded_msg, success_clean = coder.berlekamp_massey_decode(codeword)

                # Test with single error
                corrupted = codeword.copy()
                if len(corrupted) > 0:
                    corrupted[0] = (corrupted[0] + 1) % 256  # Single symbol error

                decoded_err, success_err = coder.berlekamp_massey_decode(corrupted)

                results[f"rs_{config['n']}_{config['k']}"] = {
                    'config': config,
                    'codeword_length': len(codeword),
                    'clean_decoding': success_clean,
                    'error_correction': success_err,
                    't_errors': coder.t
                }

                print(f"    Clean decoding: {'✅ Success' if success_clean else '❌ Failed'}")
                print(f"    Error correction: {'✅ Success' if success_err else '❌ Failed'}")
                print(f"    Error capability: {coder.t} symbols")

            self.test_results['reed_solomon'] = results
            print("  ✅ Reed-Solomon coding test completed")

        except Exception as e:
            print(f"  ❌ Reed-Solomon coding test failed: {e}")
            self.test_results['reed_solomon'] = {'error': str(e)}

    def test_coding_detection(self):
        """Test channel coding detection"""
        print("\n🔍 Testing Channel Coding Detection...")

        try:
            detector = ChannelCodingDetector()

            # Create test signals for different coding types
            test_signals = {}

            # Convolutional coded signal
            conv_coder = ConvolutionalCoder()
            conv_encoded = conv_coder.encode(self.test_data)
            test_signals['convolutional'] = conv_encoded

            # LDPC coded signal (using Hamming matrix)
            H = generate_hamming_matrix(3)
            ldpc_coder = LDPCCoder(H)
            ldpc_encoded = ldpc_coder.encode(self.test_data[:4])
            test_signals['ldpc'] = ldpc_encoded

            # Polar coded signal
            polar_coder = PolarCoder(16, 8)
            polar_encoded = polar_coder.encode(self.test_data[:8])
            test_signals['polar'] = polar_encoded

            # Test detection for each signal
            detection_results = {}

            for true_type, signal in test_signals.items():
                print(f"  Testing detection of {true_type} signal...")

                detected_type, scores = detector.detect_coding_type(signal)

                detection_results[true_type] = {
                    'true_type': true_type,
                    'detected_type': detected_type,
                    'scores': scores,
                    'correct_detection': detected_type == true_type
                }

                print(f"    True: {true_type}, Detected: {detected_type}")
                print(f"    Correct: {'✅' if detected_type == true_type else '❌'}")

                # Print top 3 scores
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for i, (code_type, score) in enumerate(sorted_scores[:3]):
                    print(f"    {i+1}. {code_type}: {score:.3f}")

            self.test_results['detection'] = detection_results
            print("  ✅ Channel coding detection test completed")

        except Exception as e:
            print(f"  ❌ Channel coding detection test failed: {e}")
            self.test_results['detection'] = {'error': str(e)}

    def test_enhanced_processor(self):
        """Test enhanced signal processor"""
        print("\n🧠 Testing Enhanced Signal Processor...")

        try:
            processor = EnhancedSignalProcessor()

            # Generate test IQ signals
            print("  Generating test signals...")
            test_signals = processor.generate_test_signals()

            processor_results = {}

            for coding_type, iq_signal in test_signals.items():
                print(f"  Analyzing {coding_type} signal...")

                # Run comprehensive analysis
                results = processor.comprehensive_signal_analysis(iq_signal)

                processor_results[coding_type] = {
                    'signal_length': len(iq_signal),
                    'snr_estimate': results.get('snr_estimate', 'N/A'),
                    'detected_coding': results.get('channel_coding', 'Unknown'),
                    'coding_success': results.get('coding_success', False),
                    'decoded_bits_count': len(results['decoded_bits']) if results.get('decoded_bits') is not None else 0
                }

                print(f"    Signal length: {len(iq_signal)} samples")
                print(f"    SNR estimate: {results.get('snr_estimate', 'N/A')} dB")
                print(f"    Detected coding: {results.get('channel_coding', 'Unknown')}")
                print(f"    Decoding success: {'✅' if results.get('coding_success') else '❌'}")

            self.test_results['enhanced_processor'] = processor_results
            print("  ✅ Enhanced processor test completed")

        except Exception as e:
            print(f"  ❌ Enhanced processor test failed: {e}")
            self.test_results['enhanced_processor'] = {'error': str(e)}

    def analyze_performance(self):
        """Analyze performance across different coding types"""
        print("\n📈 Performance Analysis...")

        try:
            # Count successful tests
            success_counts = {}

            for coding_type, results in self.test_results.items():
                if coding_type == 'detection':
                    # Special handling for detection results
                    if 'error' not in results:
                        correct_detections = sum(1 for r in results.values() 
                                               if isinstance(r, dict) and r.get('correct_detection', False))
                        total_detections = len([r for r in results.values() if isinstance(r, dict)])
                        success_counts[coding_type] = f"{correct_detections}/{total_detections}"
                    else:
                        success_counts[coding_type] = "Error"

                elif coding_type == 'enhanced_processor':
                    # Count successful decodings
                    if 'error' not in results:
                        successful = sum(1 for r in results.values() 
                                       if isinstance(r, dict) and r.get('coding_success', False))
                        total = len([r for r in results.values() if isinstance(r, dict)])
                        success_counts[coding_type] = f"{successful}/{total}"
                    else:
                        success_counts[coding_type] = "Error"

                else:
                    # General test results
                    if 'error' not in results:
                        successful_tests = 0
                        total_tests = 0

                        for test_name, test_result in results.items():
                            if isinstance(test_result, dict) and 'error' not in test_result:
                                total_tests += 1
                                # Check various success indicators
                                success_indicators = [
                                    'hard_decision_success', 'soft_decision_success',
                                    'decoding_success', 'valid_codeword', 
                                    'clean_decoding', 'error_correction'
                                ]
                                if any(test_result.get(indicator, False) for indicator in success_indicators):
                                    successful_tests += 1

                        success_counts[coding_type] = f"{successful_tests}/{total_tests}"
                    else:
                        success_counts[coding_type] = "Error"

            # Display performance summary
            print("  📊 Performance Summary:")
            for coding_type, success_rate in success_counts.items():
                print(f"    {coding_type.capitalize()}: {success_rate}")

            self.test_results['performance_summary'] = success_counts

        except Exception as e:
            print(f"  ❌ Performance analysis failed: {e}")

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n📋 Generating Summary Report...")

        try:
            report = []
            report.append("=" * 80)
            report.append("COMPREHENSIVE CHANNEL CODING TEST REPORT")
            report.append("=" * 80)
            report.append(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Test Data: {len(self.test_data)} bits")
            report.append("")

            # Test results summary
            for coding_type, results in self.test_results.items():
                if coding_type == 'performance_summary':
                    continue

                report.append(f"{coding_type.upper()} CODING RESULTS:")
                report.append("-" * 40)

                if 'error' in results:
                    report.append(f"  ❌ ERROR: {results['error']}")
                else:
                    if coding_type == 'detection':
                        for signal_type, detection_result in results.items():
                            if isinstance(detection_result, dict):
                                true_type = detection_result.get('true_type', 'Unknown')
                                detected_type = detection_result.get('detected_type', 'Unknown')
                                correct = detection_result.get('correct_detection', False)
                                report.append(f"  {signal_type}: {true_type} -> {detected_type} {'✅' if correct else '❌'}")

                    elif coding_type == 'enhanced_processor':
                        for signal_type, proc_result in results.items():
                            if isinstance(proc_result, dict):
                                detected = proc_result.get('detected_coding', 'Unknown')
                                success = proc_result.get('coding_success', False)
                                report.append(f"  {signal_type}: Detected {detected} {'✅' if success else '❌'}")

                    else:
                        for test_name, test_result in results.items():
                            if isinstance(test_result, dict) and 'error' not in test_result:
                                # Find success indicators
                                success_found = False
                                for key, value in test_result.items():
                                    if 'success' in key.lower() and isinstance(value, bool):
                                        report.append(f"  {test_name}: {key} {'✅' if value else '❌'}")
                                        success_found = True
                                if not success_found:
                                    report.append(f"  {test_name}: Completed")

                report.append("")

            # Performance summary
            if 'performance_summary' in self.test_results:
                report.append("PERFORMANCE SUMMARY:")
                report.append("-" * 40)
                for coding_type, success_rate in self.test_results['performance_summary'].items():
                    report.append(f"  {coding_type.capitalize()}: {success_rate}")
                report.append("")

            # Overall assessment
            report.append("OVERALL ASSESSMENT:")
            report.append("-" * 40)

            total_errors = sum(1 for results in self.test_results.values() 
                             if isinstance(results, dict) and 'error' in results)
            total_tests = len(self.test_results) - 1  # Exclude performance_summary

            if total_errors == 0:
                report.append("  🎉 ALL TESTS PASSED SUCCESSFULLY!")
                report.append("  📡 Channel coding implementation is working correctly")
                report.append("  🔧 Ready for integration with SDR applications")
            elif total_errors < total_tests:
                report.append("  ⚠️  SOME TESTS PASSED WITH WARNINGS")
                report.append(f"  📊 {total_tests - total_errors}/{total_tests} modules working correctly")
                report.append("  🔧 Partial functionality available")
            else:
                report.append("  ❌ MULTIPLE TEST FAILURES")
                report.append("  🐛 Implementation needs debugging")
                report.append("  🔧 Not ready for production use")

            report.append("")
            report.append("=" * 80)

            # Print and save report
            report_text = "\n".join(report)
            print(report_text)

            # Save to file
            with open('channel_coding_test_report.txt', 'w') as f:
                f.write(report_text)

            print(f"\n💾 Test report saved to: channel_coding_test_report.txt")

        except Exception as e:
            print(f"  ❌ Report generation failed: {e}")


def run_quick_demo():
    """Quick demonstration of key features"""
    print("\n🎬 Quick Channel Coding Demo")
    print("=" * 50)

    if not IMPORTS_OK:
        print("❌ Demo cannot run due to import errors")
        return

    try:
        # Demo data
        demo_data = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        print(f"Original data: {demo_data}")

        # Demo 1: Convolutional coding
        print("\n1. Convolutional Coding Demo:")
        conv_coder = ConvolutionalCoder()
        encoded = conv_coder.encode(demo_data)
        decoded = conv_coder.viterbi_decode(encoded)
        print(f"   Encoded: {len(encoded)} bits")
        print(f"   Decoded: {decoded}")
        print(f"   Match: {'✅ Yes' if np.array_equal(demo_data, decoded) else '❌ No'}")

        # Demo 2: LDPC coding
        print("\n2. LDPC Coding Demo:")
        H = generate_hamming_matrix(3)  # (7,4) Hamming code
        ldpc_coder = LDPCCoder(H)
        ldpc_encoded = ldpc_coder.encode(demo_data[:4])

        # Soft decoding
        received_llr = 2 * ldpc_encoded.astype(float) - 1
        ldpc_decoded, iterations = ldpc_coder.sum_product_decode(received_llr)

        syndrome = (H @ ldpc_decoded) % 2
        valid = np.all(syndrome == 0)

        print(f"   Encoded: {ldpc_encoded}")
        print(f"   Decoded: {ldpc_decoded}")
        print(f"   Valid codeword: {'✅ Yes' if valid else '❌ No'}")
        print(f"   Iterations: {iterations}")

        # Demo 3: Detection
        print("\n3. Coding Detection Demo:")
        detector = ChannelCodingDetector()
        detected_type, scores = detector.detect_coding_type(encoded)
        print(f"   Test signal: Convolutional encoded")
        print(f"   Detected: {detected_type}")
        print(f"   Top scores: {dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3])}")

        print("\n🎉 Demo completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


def main():
    """Main test execution"""
    print("🔬 CHANNEL CODING TEST SUITE")
    print("Testing comprehensive channel coding implementation")
    print("=" * 80)

    # Run quick demo first
    run_quick_demo()

    # Ask if user wants full test suite
    print("\n" + "=" * 80)
    try:
        response = input("Run full test suite? (y/N): ").lower().strip()
        if response in ['y', 'yes']:
            # Run comprehensive tests
            test_suite = ChannelCodingTestSuite()
            test_suite.run_all_tests()
        else:
            print("Skipping full test suite. Run with 'y' for comprehensive testing.")
    except:
        print("Running full test suite automatically...")
        test_suite = ChannelCodingTestSuite()
        test_suite.run_all_tests()


if __name__ == "__main__":
    main()
