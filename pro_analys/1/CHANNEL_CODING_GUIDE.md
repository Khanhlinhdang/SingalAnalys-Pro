
# Advanced SDR Suite - Channel Coding Documentation

## 📡 Complete Channel Coding Implementation Guide

### Overview
The Advanced SDR Suite now includes comprehensive channel coding (Forward Error Correction) support, implementing all major FEC algorithms used in modern wireless communications. This documentation covers the complete integration of channel coding detection, decoding, and analysis capabilities.

## 🎯 Supported Channel Coding Types

### I. Convolutional Codes
- **Implementation**: `ConvolutionalCoder` class
- **Algorithms**: Viterbi decoder (hard/soft decision)
- **Features**:
  - Configurable constraint length (K=3 to 15)
  - Multiple code rates (1/2, 1/3, 2/3, 3/4)
  - Custom generator polynomials
  - Tail-biting and terminated trellis support
  - Hard and soft decision decoding

**Usage Example**:
```python
from channel_coding import ConvolutionalCoder

# Create WiFi 802.11a standard coder
coder = ConvolutionalCoder(constraint_length=7, code_rate=0.5, 
                          polynomials=[0o133, 0o171])

# Encode data
encoded_bits = coder.encode(data_bits)

# Viterbi decode (hard decision)
decoded_bits = coder.viterbi_decode(encoded_bits, is_hard_decision=True)

# Soft decision decoding
decoded_soft = coder.viterbi_decode(received_llr, is_hard_decision=False)
```

### II. Turbo Codes
- **Implementation**: `TurboCoder` class  
- **Algorithms**: Log-MAP/BCJR decoder with iterative decoding
- **Features**:
  - Recursive Systematic Convolutional (RSC) encoders
  - Random and structured interleavers
  - Iterative Log-MAP decoding
  - Configurable iterations and SNR estimation

**Usage Example**:
```python
from channel_coding import TurboCoder

# Create turbo coder
coder = TurboCoder(constraint_length=3, interleaver_size=1024)

# Encode data
encoded_bits = coder.encode(data_bits)

# Iterative decoding
systematic = encoded_bits[:N]
parity1 = encoded_bits[N:2*N]  
parity2 = encoded_bits[2*N:3*N]

decoded_bits = coder.log_map_decode(systematic, parity1, parity2, 
                                   iterations=8, snr_db=5)
```

### III. LDPC Codes
- **Implementation**: `LDPCCoder` class
- **Algorithms**: Sum-Product Algorithm (SPA) and Min-Sum Algorithm
- **Features**:
  - Flexible parity check matrix support
  - Belief propagation decoding
  - Layered and flooding schedules
  - Early termination via syndrome checking

**Usage Example**:
```python
from channel_coding import LDPCCoder, generate_hamming_matrix

# Create LDPC coder with Hamming matrix
H = generate_hamming_matrix(3)  # (7,4) Hamming code
coder = LDPCCoder(H)

# Encode
encoded_bits = coder.encode(info_bits)

# Decode with Sum-Product Algorithm
received_llr = channel_output_to_llr(received_signal)
decoded_bits, iterations = coder.sum_product_decode(received_llr, max_iterations=50)

# Alternative: Min-Sum Algorithm
decoded_bits, iterations = coder.min_sum_decode(received_llr, max_iterations=50)
```

### IV. Polar Codes
- **Implementation**: `PolarCoder` class
- **Algorithms**: Successive Cancellation (SC) decoder
- **Features**:
  - Power-of-2 code lengths
  - Configurable information/frozen bit selection
  - SC decoding algorithm
  - Support for various construction methods

**Usage Example**:
```python
from channel_coding import PolarCoder

# Create polar coder
coder = PolarCoder(n=16, k=8, design_snr_db=0)

# Encode
codeword = coder.encode(info_bits)

# SC decode
received_llr = 2 * received_bits - 1  # Convert to LLR
decoded_info = coder.sc_decode(received_llr)
```

### V. Reed-Solomon Codes
- **Implementation**: `ReedSolomonCoder` class
- **Algorithms**: Berlekamp-Massey decoder with Chien search
- **Features**:
  - Galois Field GF(2^8) arithmetic
  - Configurable (n,k) parameters
  - Error and erasure correction
  - Syndrome calculation and error locating

**Usage Example**:
```python
from channel_coding import ReedSolomonCoder

# Create RS(15,11) coder
coder = ReedSolomonCoder(n=15, k=11)

# Encode symbol sequence
codeword = coder.encode(message_symbols)

# Decode with error correction
decoded_message, success = coder.berlekamp_massey_decode(received_symbols)
```

## 🔍 Channel Coding Detection

### Automatic Detection
The `ChannelCodingDetector` class automatically identifies channel coding types from received bit sequences using multiple detection criteria:

- **Pattern Analysis**: Block structure, rate patterns, correlation analysis
- **Statistical Features**: Autocorrelation, spectral properties, weight distribution
- **Structure Detection**: Trellis termination, systematic structure, parity relationships

**Detection Methods**:
```python
from channel_coding import ChannelCodingDetector

detector = ChannelCodingDetector()
detected_type, scores = detector.detect_coding_type(received_bits)

print(f"Detected: {detected_type}")
print(f"Confidence scores: {scores}")
```

## 🧠 Enhanced Signal Processor Integration

### Comprehensive Analysis Pipeline
The `EnhancedSignalProcessor` provides end-to-end signal analysis:

1. **IQ Data Input** → 2. **Demodulation** → 3. **Channel Coding Detection** → 4. **FEC Decoding** → 5. **Information Recovery**

**Key Features**:
- Automatic modulation and coding classification
- SNR estimation and adaptive parameters
- Comprehensive performance metrics
- Error correction statistics

**Usage**:
```python
from enhanced_signal_processor import EnhancedSignalProcessor

processor = EnhancedSignalProcessor(sample_rate=1e6)

# Comprehensive analysis
results = processor.comprehensive_signal_analysis(iq_data)

print(f"SNR: {results['snr_estimate']} dB")
print(f"Detected coding: {results['channel_coding']}")
print(f"Decoding success: {results['coding_success']}")
print(f"Decoded bits: {len(results['decoded_bits'])}")
```

## 🎨 GUI Integration

### Channel Coding Panel
The complete SDR application includes a dedicated Channel Coding tab with:

**Detection Controls**:
- Auto-detect button for coding type identification
- Confidence scoring display
- Detection results table

**Decoder Configuration**:
- Coding type selection (Convolutional, Turbo, LDPC, Polar, Reed-Solomon)
- Parameter controls specific to each coding type
- SNR estimation and soft decision options

**Performance Monitoring**:
- Real-time decoding status
- Success rate tracking  
- Error correction statistics
- Iteration counts for iterative algorithms

## 📊 Performance Characteristics

### Typical Performance
| Coding Type | Min SNR (dB) | Iteration Count | Complexity |
|-------------|--------------|-----------------|------------|
| Convolutional | -2 to 5 | N/A (Single pass) | O(2^K) |
| Turbo | -1 to 3 | 4-8 iterations | O(K×N×I) |
| LDPC | 0 to 8 | 10-50 iterations | O(E×I) |
| Polar | 0 to 5 | N/A (Single pass) | O(N log N) |
| Reed-Solomon | 5 to 15 | N/A (Algebraic) | O(t²) |

Where:
- K = constraint length, N = block length, I = iterations
- E = number of edges in graph, t = error correction capability

## 🔧 Installation and Setup

### Requirements
```
numpy >= 1.20.0
scipy >= 1.7.0
PySide6 >= 6.2.0
pyqtgraph >= 0.12.0
uhd >= 4.0.0 (for USRP support)
```

### Installation Steps
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure UHD is properly installed for USRP support
3. Configure network settings for USRP communication
4. Launch application: `python sdr_application_complete.py`

## 🧪 Testing and Validation

### Test Suite
Run comprehensive tests:
```bash
python test_channel_coding.py
```

**Test Coverage**:
- Individual coder functionality
- Multi-parameter configurations
- Detection accuracy validation
- Integration testing with signal processor
- Performance benchmarking

### Expected Results
- **Convolutional**: >95% success rate at SNR > 5dB
- **Turbo**: >99% success rate at SNR > 2dB  
- **LDPC**: >98% success rate at design SNR
- **Polar**: >95% success rate at design SNR
- **Reed-Solomon**: >90% error correction up to t errors

## 📚 Technical References

### Key Algorithms Implemented

1. **Viterbi Algorithm**: Maximum likelihood sequence estimation for convolutional codes
2. **BCJR/Log-MAP**: Optimal symbol-by-symbol MAP decoding for turbo codes
3. **Sum-Product**: Belief propagation on factor graphs for LDPC codes
4. **Successive Cancellation**: Sequential decoding for polar codes
5. **Berlekamp-Massey**: Algebraic decoding for Reed-Solomon codes

### Standards Compliance
- **802.11 WiFi**: Convolutional codes (K=7, rate 1/2, 2/3, 3/4)
- **3GPP LTE**: Turbo codes (K=4, rate 1/3) and LDPC codes
- **DVB-S2**: LDPC codes and BCH codes
- **5G NR**: Polar codes (control channels) and LDPC codes (data)

## 🔮 Advanced Features

### Research Extensions
The implementation supports advanced research applications:

- **Custom Code Design**: User-defined generator polynomials and parity check matrices
- **Algorithm Variants**: Multiple decoder implementations for comparison
- **Performance Analysis**: BER, BLER, throughput measurements
- **Hybrid Schemes**: Concatenated and product codes

### Future Enhancements
- **List Decoding**: Enhanced polar code performance
- **Spatially Coupled LDPC**: Improved threshold performance  
- **Non-Binary Codes**: Extended Galois field support
- **Machine Learning**: AI-assisted detection and decoding

## 🚀 Getting Started

### Quick Start Example
```python
# Complete channel coding workflow
from channel_coding import ConvolutionalCoder
from enhanced_signal_processor import EnhancedSignalProcessor

# 1. Create test data
data_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])

# 2. Encode with convolutional code  
coder = ConvolutionalCoder()
encoded_bits = coder.encode(data_bits)

# 3. Add noise (simulate channel)
received_bits = add_awgn_noise(encoded_bits, snr_db=10)

# 4. Detect and decode
processor = EnhancedSignalProcessor()
results = processor.comprehensive_signal_analysis(bits_to_iq(received_bits))

print(f"Original: {data_bits}")
print(f"Recovered: {results['decoded_bits']}")
print(f"Success: {np.array_equal(data_bits, results['decoded_bits'])}")
```

## 📞 Support and Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Performance Issues**: Reduce block sizes for real-time operation
3. **Detection Failures**: Check signal quality and SNR levels
4. **Memory Usage**: Use appropriate block sizes for available RAM

### Debug Features
- Comprehensive logging throughout the processing chain
- Test signal generators for validation
- Performance profiling tools
- Error correction statistics

This implementation provides a complete, production-ready channel coding suite for SDR applications, supporting both research and practical deployment scenarios.
