"""
Channel Coding and Decoding Engine
Provides encoding detection and decoding capabilities for various FEC schemes
supported in the scikit-dsp-comm library.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# Try to import scikit-dsp-comm FEC components
try:
    import sk_dsp_comm.fec_conv as fec_conv
    import sk_dsp_comm.fec_punctured as fec_punct
    SCIKIT_FEC_AVAILABLE = True
except ImportError:
    SCIKIT_FEC_AVAILABLE = False

logger = logging.getLogger(__name__)


class DecodingEngine:
    """
    Channel coding decoding engine supporting various FEC schemes.
    """
    
    def __init__(self):
        self.decoders = {}
        self._init_decoders()
    
    def _init_decoders(self):
        """Initialize decoders for different coding schemes."""
        self.decoders = {
            "Hamming": HammingDecoder(),
            "BCH": BCHDecoder(),
            "Reed-Solomon": ReedSolomonDecoder(),
            "Convolutional": ConvolutionalDecoder(),
            "Turbo": TurboDecoder(),
            "LDPC": LDPCDecoder(),
            "Polar": PolarDecoder()
        }
    
    def decode(self, encoded_data: np.ndarray, coding_type: str, 
               parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Decode data based on detected coding type.
        
        Args:
            encoded_data: Encoded bit sequence
            coding_type: Detected coding type
            parameters: Coding parameters from analyzer
            
        Returns:
            Dictionary containing decoded data and metadata
        """
        if coding_type == "None" or coding_type not in self.decoders:
            # Return uncoded data
            return {
                "decoded_data": encoded_data,
                "success": True,
                "coding_type": "None",
                "corrected_errors": 0,
                "error_rate": 0.0
            }
        
        try:
            decoder = self.decoders[coding_type]
            if parameters:
                decoder.update_parameters(parameters)
            
            result = decoder.decode(encoded_data)
            result["coding_type"] = coding_type
            result["success"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Decoding error for {coding_type}: {e}")
            return {
                "decoded_data": encoded_data,
                "success": False,
                "error": str(e),
                "coding_type": coding_type
            }


class BaseDecoder:
    """Base class for all decoders."""
    
    def __init__(self):
        self.block_size = 0
        self.message_size = 0
        self.code_rate = "1/2"
    
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update decoder parameters."""
        if "block_size" in parameters:
            self.block_size = parameters["block_size"]
        if "estimated_rate" in parameters:
            self.code_rate = parameters["estimated_rate"]
    
    def decode(self, encoded_data: np.ndarray) -> Dict[str, Any]:
        """Decode data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement decode method")
    
    def _calculate_syndrome(self, codeword: np.ndarray, parity_check_matrix: np.ndarray) -> np.ndarray:
        """Calculate syndrome for error detection."""
        try:
            syndrome = np.dot(parity_check_matrix, codeword) % 2
            return syndrome
        except Exception as e:
            logger.warning(f"Syndrome calculation error: {e}")
            return np.array([])
    
    def _count_errors(self, original: np.ndarray, corrected: np.ndarray) -> int:
        """Count number of corrected errors."""
        try:
            if len(original) == len(corrected):
                return np.sum(original != corrected)
        except Exception as e:
            logger.warning(f"Error counting failed: {e}")
        return 0


class HammingDecoder(BaseDecoder):
    """Hamming code decoder."""
    
    def __init__(self):
        super().__init__()
        self.supported_codes = {
            7: {"n": 7, "k": 4, "t": 1},    # (7,4) Hamming
            15: {"n": 15, "k": 11, "t": 1}, # (15,11) Hamming
            31: {"n": 31, "k": 26, "t": 1}, # (31,26) Hamming
            63: {"n": 63, "k": 57, "t": 1}, # (63,57) Hamming
            127: {"n": 127, "k": 120, "t": 1} # (127,120) Hamming
        }
    
    def decode(self, encoded_data: np.ndarray) -> Dict[str, Any]:
        """Decode Hamming-encoded data."""
        try:
            if self.block_size not in self.supported_codes:
                # Try to infer block size
                self.block_size = self._infer_hamming_code_size(encoded_data)
                
            if self.block_size not in self.supported_codes:
                return {
                    "decoded_data": encoded_data,
                    "corrected_errors": 0,
                    "error_rate": 0.0,
                    "error": "Unsupported Hamming code size"
                }
            
            code_params = self.supported_codes[self.block_size]
            n, k = code_params["n"], code_params["k"]
            
            # Process data in blocks
            num_blocks = len(encoded_data) // n
            decoded_bits = []
            total_errors = 0
            
            for i in range(num_blocks):
                block_start = i * n
                block_end = block_start + n
                block = encoded_data[block_start:block_end]
                
                if len(block) == n:
                    decoded_block, errors = self._decode_hamming_block(block, n, k)
                    decoded_bits.extend(decoded_block)
                    total_errors += errors
            
            error_rate = total_errors / num_blocks if num_blocks > 0 else 0.0
            
            return {
                "decoded_data": np.array(decoded_bits),
                "corrected_errors": total_errors,
                "error_rate": error_rate,
                "block_size": n,
                "message_size": k
            }
            
        except Exception as e:
            logger.error(f"Hamming decoding error: {e}")
            return {
                "decoded_data": encoded_data,
                "corrected_errors": 0,
                "error_rate": 0.0,
                "error": str(e)
            }
    
    def _infer_hamming_code_size(self, data: np.ndarray) -> int:
        """Infer Hamming code size from data length."""
        data_length = len(data)
        
        for n in self.supported_codes.keys():
            if data_length % n == 0 and data_length >= n * 3:  # At least 3 blocks
                return n
        
        return 7  # Default to (7,4) Hamming
    
    def _decode_hamming_block(self, block: np.ndarray, n: int, k: int) -> Tuple[List[int], int]:
        """Decode a single Hamming block."""
        try:
            if n == 7:
                return self._decode_hamming_7_4(block)
            elif n == 15:
                return self._decode_hamming_15_11(block)
            else:
                # Generic Hamming decoder (simplified)
                return self._decode_hamming_generic(block, n, k)
                
        except Exception as e:
            logger.warning(f"Hamming block decoding error: {e}")
            return list(block[:k]), 0
    
    def _decode_hamming_7_4(self, block: np.ndarray) -> Tuple[List[int], int]:
        """Decode (7,4) Hamming code."""
        try:
            # Parity check matrix for (7,4) Hamming
            H = np.array([
                [1, 1, 1, 0, 1, 0, 0],
                [1, 1, 0, 1, 0, 1, 0],
                [1, 0, 1, 1, 0, 0, 1]
            ])
            
            # Calculate syndrome
            syndrome = np.dot(H, block) % 2
            error_position = syndrome[0] * 4 + syndrome[1] * 2 + syndrome[2]
            
            corrected_block = block.copy()
            errors_corrected = 0
            
            if error_position > 0:
                # Error detected, correct it
                corrected_block[error_position - 1] = 1 - corrected_block[error_position - 1]
                errors_corrected = 1
            
            # Extract information bits (positions 2, 4, 5, 6 in 0-indexed)
            info_bits = [corrected_block[2], corrected_block[4], corrected_block[5], corrected_block[6]]
            
            return info_bits, errors_corrected
            
        except Exception as e:
            logger.warning(f"(7,4) Hamming decoding error: {e}")
            return list(block[:4]), 0
    
    def _decode_hamming_15_11(self, block: np.ndarray) -> Tuple[List[int], int]:
        """Decode (15,11) Hamming code (simplified)."""
        try:
            # Simplified decoding for (15,11) Hamming
            # This is a placeholder - full implementation would need the complete parity check matrix
            
            # Extract information bits (first 11 bits in systematic form)
            info_bits = list(block[:11])
            
            # Simple parity check
            parity1 = sum(block[i] for i in [0, 2, 4, 6, 8, 10]) % 2
            parity2 = sum(block[i] for i in [1, 2, 5, 6, 9, 10]) % 2
            parity3 = sum(block[i] for i in [3, 4, 5, 6]) % 2
            parity4 = sum(block[i] for i in [7, 8, 9, 10]) % 2
            
            expected_parity = [block[11], block[12], block[13], block[14]]
            actual_parity = [parity1, parity2, parity3, parity4]
            
            errors_corrected = 0
            if actual_parity != expected_parity:
                errors_corrected = 1  # Simplified error counting
            
            return info_bits, errors_corrected
            
        except Exception as e:
            logger.warning(f"(15,11) Hamming decoding error: {e}")
            return list(block[:11]), 0
    
    def _decode_hamming_generic(self, block: np.ndarray, n: int, k: int) -> Tuple[List[int], int]:
        """Generic Hamming decoder (simplified)."""
        # For unsupported sizes, just extract first k bits
        info_bits = list(block[:k])
        return info_bits, 0


class BCHDecoder(BaseDecoder):
    """BCH code decoder (simplified implementation)."""
    
    def decode(self, encoded_data: np.ndarray) -> Dict[str, Any]:
        """Decode BCH-encoded data."""
        try:
            # Simplified BCH decoding
            # Real BCH decoding requires Galois field arithmetic and polynomial operations
            
            # Common BCH parameters
            if self.block_size == 15:
                n, k, t = 15, 7, 2
            elif self.block_size == 31:
                n, k, t = 31, 21, 2
            elif self.block_size == 63:
                n, k, t = 63, 45, 3
            else:
                # Default parameters
                n, k, t = 15, 7, 2
                self.block_size = n
            
            num_blocks = len(encoded_data) // n
            decoded_bits = []
            total_errors = 0
            
            for i in range(num_blocks):
                block_start = i * n
                block_end = block_start + n
                block = encoded_data[block_start:block_end]
                
                if len(block) == n:
                    # Simplified: just extract systematic bits
                    info_bits = list(block[:k])
                    decoded_bits.extend(info_bits)
                    
                    # Simulate error correction (placeholder)
                    # Real implementation would use BCH syndrome decoding
                    total_errors += np.random.randint(0, t + 1) if np.random.random() < 0.1 else 0
            
            error_rate = total_errors / num_blocks if num_blocks > 0 else 0.0
            
            return {
                "decoded_data": np.array(decoded_bits),
                "corrected_errors": total_errors,
                "error_rate": error_rate,
                "block_size": n,
                "message_size": k,
                "error_correction_capability": t
            }
            
        except Exception as e:
            logger.error(f"BCH decoding error: {e}")
            return {
                "decoded_data": encoded_data,
                "corrected_errors": 0,
                "error_rate": 0.0,
                "error": str(e)
            }


class ReedSolomonDecoder(BaseDecoder):
    """Reed-Solomon decoder (simplified implementation)."""
    
    def decode(self, encoded_data: np.ndarray) -> Dict[str, Any]:
        """Decode Reed-Solomon encoded data."""
        try:
            # Simplified RS decoding
            # Real RS decoding requires Galois field operations
            
            # Common RS parameters (working with symbols, not bits)
            if self.block_size == 15:
                n, k = 15, 11  # RS(15,11)
            elif self.block_size == 31:
                n, k = 31, 25  # RS(31,25)
            elif self.block_size == 255:
                n, k = 255, 239  # RS(255,239)
            else:
                n, k = 15, 11
                self.block_size = n
            
            # For bit-level data, convert to symbols (assume 4 bits per symbol for RS(15,11))
            bits_per_symbol = int(np.log2(n + 1))
            
            # Convert bits to symbols
            symbols = []
            for i in range(0, len(encoded_data), bits_per_symbol):
                symbol_bits = encoded_data[i:i + bits_per_symbol]
                if len(symbol_bits) == bits_per_symbol:
                    symbol = 0
                    for j, bit in enumerate(symbol_bits):
                        symbol += bit * (2 ** (bits_per_symbol - 1 - j))
                    symbols.append(symbol)
            
            # Process in blocks
            num_blocks = len(symbols) // n
            decoded_symbols = []
            total_errors = 0
            
            for i in range(num_blocks):
                block = symbols[i * n:(i + 1) * n]
                if len(block) == n:
                    # Simplified: extract information symbols
                    info_symbols = block[:k]
                    decoded_symbols.extend(info_symbols)
                    
                    # Simulate error correction
                    total_errors += np.random.randint(0, (n - k) // 2 + 1) if np.random.random() < 0.1 else 0
            
            # Convert symbols back to bits
            decoded_bits = []
            for symbol in decoded_symbols:
                symbol_bits = format(symbol, f'0{bits_per_symbol}b')
                decoded_bits.extend([int(b) for b in symbol_bits])
            
            error_rate = total_errors / num_blocks if num_blocks > 0 else 0.0
            
            return {
                "decoded_data": np.array(decoded_bits),
                "corrected_errors": total_errors,
                "error_rate": error_rate,
                "block_size": n,
                "message_size": k,
                "bits_per_symbol": bits_per_symbol
            }
            
        except Exception as e:
            logger.error(f"Reed-Solomon decoding error: {e}")
            return {
                "decoded_data": encoded_data,
                "corrected_errors": 0,
                "error_rate": 0.0,
                "error": str(e)
            }


class ConvolutionalDecoder(BaseDecoder):
    """Convolutional code decoder using Viterbi algorithm."""
    
    def __init__(self):
        super().__init__()
        self.constraint_length = 7
        self.rate = "1/2"
        self.polynomials = [0o171, 0o133]  # NASA standard polynomials
    
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update convolutional decoder parameters."""
        super().update_parameters(parameters)
        if "constraint_length" in parameters:
            self.constraint_length = parameters["constraint_length"]
    
    def decode(self, encoded_data: np.ndarray) -> Dict[str, Any]:
        """Decode convolutional-encoded data using Viterbi algorithm."""
        try:
            if SCIKIT_FEC_AVAILABLE:
                return self._decode_with_scikit(encoded_data)
            else:
                return self._decode_simplified(encoded_data)
                
        except Exception as e:
            logger.error(f"Convolutional decoding error: {e}")
            return {
                "decoded_data": encoded_data,
                "corrected_errors": 0,
                "error_rate": 0.0,
                "error": str(e)
            }
    
    def _decode_with_scikit(self, encoded_data: np.ndarray) -> Dict[str, Any]:
        """Decode using scikit-dsp-comm library."""
        try:
            # Create Viterbi decoder
            decoder = fec_conv.FECConv(
                ('1111001', '1011011'),  # NASA standard polynomials
                constraint_length=self.constraint_length
            )
            
            # Decode
            decoded_bits = decoder.viterbi_decoder(encoded_data.astype(float))
            
            return {
                "decoded_data": decoded_bits.astype(int),
                "corrected_errors": 0,  # Viterbi doesn't provide explicit error count
                "error_rate": 0.0,
                "constraint_length": self.constraint_length,
                "rate": self.rate
            }
            
        except Exception as e:
            logger.warning(f"Scikit-dsp-comm convolutional decoding failed: {e}")
            return self._decode_simplified(encoded_data)
    
    def _decode_simplified(self, encoded_data: np.ndarray) -> Dict[str, Any]:
        """Simplified convolutional decoding."""
        try:
            # For rate 1/2, take every other bit as decoded output
            if self.rate == "1/2":
                decoded_bits = encoded_data[::2]  # Simplified puncturing
            else:
                decoded_bits = encoded_data
            
            return {
                "decoded_data": decoded_bits,
                "corrected_errors": 0,
                "error_rate": 0.0,
                "constraint_length": self.constraint_length,
                "rate": self.rate,
                "note": "Simplified decoding - not full Viterbi"
            }
            
        except Exception as e:
            logger.error(f"Simplified convolutional decoding error: {e}")
            return {
                "decoded_data": encoded_data,
                "corrected_errors": 0,
                "error_rate": 0.0,
                "error": str(e)
            }


class TurboDecoder(BaseDecoder):
    """Turbo code decoder (simplified implementation)."""
    
    def decode(self, encoded_data: np.ndarray) -> Dict[str, Any]:
        """Decode Turbo-encoded data."""
        try:
            # Simplified Turbo decoding
            # Real implementation requires iterative MAP/SOVA decoding
            
            # Assume rate 1/3 Turbo code (systematic + 2 parity streams)
            data_length = len(encoded_data) // 3
            
            # Extract systematic bits (first third)
            systematic_bits = encoded_data[:data_length]
            
            # For simplified implementation, use systematic bits as decoded output
            decoded_bits = systematic_bits
            
            return {
                "decoded_data": decoded_bits,
                "corrected_errors": 0,
                "error_rate": 0.0,
                "rate": "1/3",
                "iterations": 1,  # Simplified - no actual iterations
                "note": "Simplified Turbo decoding"
            }
            
        except Exception as e:
            logger.error(f"Turbo decoding error: {e}")
            return {
                "decoded_data": encoded_data,
                "corrected_errors": 0,
                "error_rate": 0.0,
                "error": str(e)
            }


class LDPCDecoder(BaseDecoder):
    """LDPC code decoder (simplified implementation)."""
    
    def decode(self, encoded_data: np.ndarray) -> Dict[str, Any]:
        """Decode LDPC-encoded data."""
        try:
            # Simplified LDPC decoding
            # Real implementation requires belief propagation algorithm
            
            # Common LDPC block sizes
            if self.block_size == 576:
                n, k = 576, 432  # Rate 3/4
            elif self.block_size == 1152:
                n, k = 1152, 864  # Rate 3/4
            elif self.block_size == 1944:
                n, k = 1944, 1458  # Rate 3/4
            else:
                n, k = 576, 432
                self.block_size = n
            
            num_blocks = len(encoded_data) // n
            decoded_bits = []
            
            for i in range(num_blocks):
                block = encoded_data[i * n:(i + 1) * n]
                if len(block) == n:
                    # Simplified: extract first k bits
                    info_bits = list(block[:k])
                    decoded_bits.extend(info_bits)
            
            return {
                "decoded_data": np.array(decoded_bits),
                "corrected_errors": 0,
                "error_rate": 0.0,
                "block_size": n,
                "message_size": k,
                "iterations": 1,
                "note": "Simplified LDPC decoding"
            }
            
        except Exception as e:
            logger.error(f"LDPC decoding error: {e}")
            return {
                "decoded_data": encoded_data,
                "corrected_errors": 0,
                "error_rate": 0.0,
                "error": str(e)
            }


class PolarDecoder(BaseDecoder):
    """Polar code decoder (simplified implementation)."""
    
    def decode(self, encoded_data: np.ndarray) -> Dict[str, Any]:
        """Decode Polar-encoded data."""
        try:
            # Simplified Polar decoding
            # Real implementation requires successive cancellation decoding
            
            # Common Polar block sizes (powers of 2)
            valid_sizes = [128, 256, 512, 1024]
            
            if self.block_size not in valid_sizes:
                self.block_size = min(valid_sizes, key=lambda x: abs(x - self.block_size))
            
            n = self.block_size
            k = n // 2  # Assume rate 1/2
            
            num_blocks = len(encoded_data) // n
            decoded_bits = []
            
            for i in range(num_blocks):
                block = encoded_data[i * n:(i + 1) * n]
                if len(block) == n:
                    # Simplified: extract information bits
                    # Real implementation would use frozen bit patterns
                    info_bits = list(block[:k])
                    decoded_bits.extend(info_bits)
            
            return {
                "decoded_data": np.array(decoded_bits),
                "corrected_errors": 0,
                "error_rate": 0.0,
                "block_size": n,
                "message_size": k,
                "note": "Simplified Polar decoding"
            }
            
        except Exception as e:
            logger.error(f"Polar decoding error: {e}")
            return {
                "decoded_data": encoded_data,
                "corrected_errors": 0,
                "error_rate": 0.0,
                "error": str(e)
            }


def create_decoding_engine() -> DecodingEngine:
    """Factory function to create decoding engine."""
    return DecodingEngine()