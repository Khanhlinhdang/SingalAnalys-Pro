
"""
Advanced SDR Suite - Final Launcher
Khởi động phần mềm SDR với Real-time Processing Pipeline
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check for required dependencies"""
    print("🔍 Checking dependencies...")

    required_modules = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('PySide6', 'PySide6'),
        ('pyqtgraph', 'pyqtgraph'),
    ]

    optional_modules = [
        ('uhd', 'uhd (for USRP hardware)'),
    ]

    missing_required = []
    missing_optional = []

    for module, display_name in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name}")
            missing_required.append(display_name)

    for module, display_name in optional_modules:
        try:
            __import__(module)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ⚠️  {display_name} (optional)")
            missing_optional.append(display_name)

    return missing_required, missing_optional

def check_project_files():
    """Check for project files"""
    print("\n📁 Checking project files...")

    required_files = [
        'analog_modulation.py',
        'extended_digital_modulation.py', 
        'multicarrier_spread_spectrum.py',
        'channel_coding.py',
        'enhanced_signal_processor.py',
        'realtime_signal_pipeline.py',
        'sdr_application_realtime.py'
    ]

    missing_files = []

    for filename in required_files:
        if os.path.exists(filename):
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename}")
            missing_files.append(filename)

    return missing_files

def run_tests():
    """Run basic functionality tests"""
    print("\n🧪 Running basic tests...")

    try:
        # Test core modules
        print("  Testing analog modulation...")
        from analog_modulation import AnalogModulation
        analog_mod = AnalogModulation()
        print("    ✅ Analog modulation OK")

        print("  Testing digital modulation...")
        from extended_digital_modulation import ExtendedDigitalModulation
        digital_mod = ExtendedDigitalModulation()
        print("    ✅ Digital modulation OK")

        print("  Testing channel coding...")
        from channel_coding import ConvolutionalCoder
        conv_coder = ConvolutionalCoder()
        print("    ✅ Channel coding OK")

        print("  Testing real-time pipeline...")
        from realtime_signal_pipeline import RealtimeSignalAnalyzer
        analyzer = RealtimeSignalAnalyzer(sample_rate=100000, update_interval=1.0)
        print("    ✅ Real-time pipeline OK")

        print("  ✅ All tests passed!")
        return True

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def launch_application(mode='realtime'):
    """Launch SDR application"""
    print(f"\n🚀 Launching SDR application in {mode} mode...")

    try:
        if mode == 'realtime':
            app_file = 'sdr_application_realtime.py'
            print("  Starting Real-time Processing Pipeline...")
        else:
            app_file = 'sdr_application_complete.py'
            print("  Starting Standard Application...")

        if not os.path.exists(app_file):
            print(f"  ❌ Application file not found: {app_file}")
            return False

        # Launch application
        process = subprocess.Popen([sys.executable, app_file])
        print(f"  ✅ Application launched (PID: {process.pid})")

        return True

    except Exception as e:
        print(f"  ❌ Launch failed: {e}")
        return False

def show_feature_summary():
    """Show comprehensive feature summary"""
    print("\n" + "="*80)
    print("🎯 ADVANCED SDR SUITE - COMPREHENSIVE FEATURE SUMMARY")
    print("="*80)

    print("\n📡 SUPPORTED MODULATION TYPES (50+ variants):")
    print("├── Analog Modulation")
    print("│   ├── AM: DSB-LC, DSB-SC, SSB (USB/LSB), VSB")
    print("│   ├── FM: Narrow-band FM, Wide-band FM")
    print("│   ├── PM: Phase Modulation")
    print("│   └── Pulse: PAM, PWM, PPM")
    print("├── Digital Single Carrier")
    print("│   ├── Amplitude: OOK, ASK (2/4/8-ASK)")
    print("│   ├── Frequency: FSK, GFSK, MSK, GMSK, CPFSK")
    print("│   ├── Phase: BPSK, QPSK, 8PSK, OQPSK, π/4-QPSK")
    print("│   ├── Differential: DPSK, DBPSK, DQPSK")
    print("│   ├── QAM: 16/64/256/1024-QAM")
    print("│   └── APSK: 16/32-APSK (DVB-S2)")
    print("├── Multi-Carrier")
    print("│   ├── OFDM: Standard OFDM với pilots")
    print("│   ├── SC-FDMA: LTE uplink")
    print("│   ├── FBMC: Filter Bank Multi-Carrier")
    print("│   └── f-OFDM: Filtered OFDM")
    print("├── Spread Spectrum")
    print("│   ├── DSSS: Direct Sequence với PN/Gold codes")
    print("│   ├── FHSS: Frequency Hopping")
    print("│   └── CSS: Chirp Spread Spectrum (LoRa)")
    print("└── MIMO/Spatial")
    print("    ├── STBC: Alamouti Space-Time Block Code")
    print("    ├── V-BLAST: Spatial Multiplexing")
    print("    └── Beamforming: ZF/MMSE detection")

    print("\n🔐 CHANNEL CODING SUPPORT (Forward Error Correction):")
    print("├── Convolutional Codes")
    print("│   ├── Viterbi Decoder: Hard/soft decision")
    print("│   ├── Constraint Length: K=3 to K=15")
    print("│   └── Code Rates: 1/2, 1/3, 2/3, 3/4")
    print("├── Turbo Codes")
    print("│   ├── Log-MAP/BCJR Algorithm")
    print("│   ├── Iterative Decoding: 1-20 iterations")
    print("│   └── RSC Components: Configurable interleavers")
    print("├── LDPC Codes")
    print("│   ├── Sum-Product Algorithm (Belief Propagation)")
    print("│   ├── Min-Sum Algorithm (Hardware efficient)")
    print("│   └── Flexible Matrices: Custom parity check support")
    print("├── Polar Codes")
    print("│   ├── Successive Cancellation Decoder")
    print("│   ├── Construction Methods: Bhattacharyya parameters")
    print("│   └── 5G NR Standard: Control channel support")
    print("└── Reed-Solomon Codes")
    print("    ├── Berlekamp-Massey Algorithm")
    print("    ├── Galois Field GF(2^8) arithmetic")
    print("    └── Error/Erasure Correction: Configurable (n,k)")

    print("\n🔄 REAL-TIME PROCESSING PIPELINE:")
    print("├── Stage 1: Automatic Modulation Detection")
    print("├── Stage 2: Signal Demodulation + Constellation Display")
    print("├── Stage 3: Channel Coding Detection")
    print("├── Stage 4: Channel Decoding (FEC)")
    print("└── Stage 5: Bit Stream Extraction + Live Display")

    print("\n🎨 ENHANCED GUI FEATURES:")
    print("├── Professional Dark Theme")
    print("├── Real-time Constellation Diagram")
    print("├── Live Bit Stream Window")
    print("├── Processing Stage Indicators")
    print("├── Performance Monitoring")
    print("├── Spectrum Analyzer")
    print("├── Waterfall Display")
    print("└── Comprehensive Analysis Results")

    print("\n📊 STANDARDS COMPLIANCE:")
    print("├── IEEE 802.11 (WiFi): OFDM, Convolutional codes")
    print("├── 3GPP LTE/5G: OFDMA, SC-FDMA, Turbo, LDPC, Polar")
    print("├── DVB-S2/T2: COFDM, LDPC, APSK")
    print("├── Bluetooth: GFSK, FHSS")
    print("├── LoRaWAN: CSS modulation")
    print("├── GPS: DSSS với Gold codes")
    print("└── GSM: GMSK modulation")

    print("\n🎯 APPLICATION MODES:")
    print("├── Real-time Mode: Automatic signal generation và processing")
    print("├── USRP Mode: Hardware integration với Ettus USRP")
    print("├── Analysis Mode: Manual signal analysis và testing")
    print("└── Research Mode: Algorithm development và validation")

def interactive_menu():
    """Interactive launcher menu"""
    while True:
        print("\n" + "="*60)
        print("🎛️  ADVANCED SDR SUITE LAUNCHER")
        print("="*60)
        print("1. 🔍 Check Dependencies & Project Files")
        print("2. 🧪 Run Basic Tests")
        print("3. 🚀 Launch Real-time Application")
        print("4. 🚀 Launch Standard Application")
        print("5. 📊 Show Feature Summary")
        print("6. ❓ Help & Documentation")
        print("0. ❌ Exit")
        print("-" * 60)

        try:
            choice = input("Select option (0-6): ").strip()

            if choice == '0':
                print("\n👋 Goodbye!")
                break
            elif choice == '1':
                missing_req, missing_opt = check_dependencies()
                missing_files = check_project_files()

                if missing_req:
                    print("\n❌ Missing required dependencies. Install with:")
                    print("   pip install numpy scipy PySide6 pyqtgraph")

                if missing_files:
                    print("\n❌ Missing project files. Please ensure all modules are present.")
                else:
                    print("\n✅ All project files found!")

            elif choice == '2':
                success = run_tests()
                if success:
                    print("\n🎉 All tests passed! Ready to launch application.")
                else:
                    print("\n❌ Some tests failed. Check error messages above.")

            elif choice == '3':
                print("\n🔄 Launching Real-time Processing Application...")
                launch_application('realtime')

            elif choice == '4':
                print("\n📊 Launching Standard SDR Application...")
                launch_application('standard')

            elif choice == '5':
                show_feature_summary()

            elif choice == '6':
                print("\n📖 HELP & DOCUMENTATION")
                print("-" * 40)
                print("📁 Project Files:")
                print("  • sdr_application_realtime.py - Main real-time application")
                print("  • channel_coding.py - FEC algorithms")
                print("  • realtime_signal_pipeline.py - Processing pipeline")
                print("  • All modulation modules - Complete signal processing")
                print("\n📚 Documentation:")
                print("  • README_COMPREHENSIVE.md - Complete project overview")
                print("  • CHANNEL_CODING_GUIDE.md - FEC technical reference")
                print("  • MODULATION_REFERENCE.md - Modulation algorithms")
                print("\n🎯 Quick Start:")
                print("  1. Run option 1 to check dependencies")
                print("  2. Run option 2 to test functionality")
                print("  3. Run option 3 for real-time application")
                print("\n💡 Tips:")
                print("  • Real-time mode: Automatic signal generation")
                print("  • Use USRP for hardware signals")
                print("  • Check constellation for signal quality")
                print("  • Monitor bit stream for decoded data")

            else:
                print("\n⚠️  Invalid option. Please select 0-6.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main launcher function"""
    print("🚀 ADVANCED SDR SUITE - COMPREHENSIVE LAUNCHER")
    print("=" * 60)
    print("Complete Modulation Analysis & Channel Coding Platform")
    print("Real-time Processing Pipeline with 50+ Modulation Types")
    print("=" * 60)

    # Check if running directly
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'realtime':
            launch_application('realtime')
        elif mode == 'standard':  
            launch_application('standard')
        elif mode == 'test':
            run_tests()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python launch_advanced_sdr.py [realtime|standard|test]")
    else:
        # Interactive mode
        interactive_menu()

if __name__ == '__main__':
    main()
