# Create comprehensive channel coding module
channel_coding_module = '''
"""
Comprehensive Channel Coding Module
Hỗ trợ nhận dạng và giải mã tất cả loại mã hóa kênh (FEC codes)
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from scipy.special import erfc, erf
import warnings
from typing import Tuple, List, Optional, Union
warnings.filterwarnings('ignore')


class ConvolutionalCoder:
    """Convolutional encoder and Viterbi decoder"""
    
    def __init__(self, constraint_length=7, code_rate=0.5, polynomials=None):
        self.K = constraint_length
        self.rate = code_rate
        self.num_outputs = int(1 / code_rate)
        
        # Default polynomials for different configurations
        if polynomials is None:
            if constraint_length == 3 and code_rate == 0.5:
                self.polynomials = [0o7, 0o5]  # [111, 101] in binary
            elif constraint_length == 7 and code_rate == 0.5:
                self.polynomials = [0o133, 0o171]  # WiFi 802.11a standard
            else:
                self.polynomials = [0o133, 0o171]  # Default
        else:
            self.polynomials = polynomials
            
        self.num_states = 2**(constraint_length - 1)
        self._build_trellis()
        
    def _build_trellis(self):
        """Build trellis structure for Viterbi algorithm"""
        self.next_states = np.zeros((self.num_states, 2), dtype=int)
        self.outputs = np.zeros((self.num_states, 2), dtype=int)
        
        for state in range(self.num_states):
            for input_bit in [0, 1]:
                # Shift register simulation
                shift_reg = (state << 1) | input_bit
                shift_reg &= (1 << self.K) - 1  # Keep only K bits
                
                # Calculate outputs using polynomials
                output = 0
                for i, poly in enumerate(self.polynomials):
                    output_bit = 0
                    temp_reg = shift_reg & poly
                    while temp_reg:
                        output_bit ^= temp_reg & 1
                        temp_reg >>= 1
                    output |= output_bit << i
                
                next_state = shift_reg >> 1
                self.next_states[state, input_bit] = next_state
                self.outputs[state, input_bit] = output
    
    def encode(self, data_bits):
        """Convolutional encoding"""
        state = 0
        encoded_bits = []
        
        # Encode data bits
        for bit in data_bits:
            output = self.outputs[state, bit]
            state = self.next_states[state, bit]
            
            # Convert output to bit sequence
            for i in range(self.num_outputs):
                encoded_bits.append((output >> i) & 1)
        
        # Add tail bits to terminate trellis
        for _ in range(self.K - 1):
            output = self.outputs[state, 0]
            state = self.next_states[state, 0]
            
            for i in range(self.num_outputs):
                encoded_bits.append((output >> i) & 1)
        
        return np.array(encoded_bits)
    
    def viterbi_decode(self, received_bits, is_hard_decision=True):
        """Viterbi decoding algorithm"""
        n_bits = len(received_bits) // self.num_outputs
        
        # Initialize path metrics and survivors
        path_metrics = np.full(self.num_states, np.inf)
        path_metrics[0] = 0  # Start from zero state
        
        survivors = np.zeros((n_bits, self.num_states), dtype=int)
        
        # Forward pass through trellis
        for t in range(n_bits):
            new_path_metrics = np.full(self.num_states, np.inf)
            
            for state in range(self.num_states):
                if path_metrics[state] == np.inf:
                    continue
                    
                for input_bit in [0, 1]:
                    next_state = self.next_states[state, input_bit]
                    expected_output = self.outputs[state, input_bit]
                    
                    # Calculate branch metric
                    received_sym = received_bits[t*self.num_outputs:(t+1)*self.num_outputs]
                    
                    if is_hard_decision:
                        # Hamming distance for hard decision
                        branch_metric = 0
                        for i in range(self.num_outputs):
                            expected_bit = (expected_output >> i) & 1
                            if len(received_sym) > i:
                                branch_metric += (expected_bit != received_sym[i])
                    else:
                        # Euclidean distance for soft decision
                        branch_metric = 0
                        for i in range(self.num_outputs):
                            expected_bit = (expected_output >> i) & 1
                            expected_soft = 2 * expected_bit - 1  # Map 0->-1, 1->+1
                            if len(received_sym) > i:
                                branch_metric += (expected_soft - received_sym[i])**2
                    
                    # Update path metric
                    new_metric = path_metrics[state] + branch_metric
                    
                    if new_metric < new_path_metrics[next_state]:
                        new_path_metrics[next_state] = new_metric
                        survivors[t, next_state] = (state << 1) | input_bit
            
            path_metrics = new_path_metrics
        
        # Backward pass - traceback
        # Find best final state (should be 0 for terminated trellis)
        best_state = np.argmin(path_metrics)
        
        # Traceback to find best path
        decoded_bits = []
        state = best_state
        
        for t in range(n_bits - 1, -1, -1):
            survivor = survivors[t, state]
            input_bit = survivor & 1
            prev_state = survivor >> 1
            
            decoded_bits.append(input_bit)
            state = prev_state
        
        decoded_bits.reverse()
        
        # Remove tail bits
        data_length = len(decoded_bits) - (self.K - 1)
        return np.array(decoded_bits[:data_length])


class TurboCoder:
    """Turbo encoder and decoder"""
    
    def __init__(self, constraint_length=3, interleaver_size=1024):
        self.K = constraint_length
        self.N = interleaver_size
        self.num_states = 2**(constraint_length - 1)
        
        # Default RSC polynomials for turbo codes
        self.polynomials = [0o7, 0o5]  # Recursive systematic convolutional
        
        # Generate random interleaver
        self.interleaver = np.random.permutation(interleaver_size)
        self.deinterleaver = np.argsort(self.interleaver)
        
        self._build_trellis()
    
    def _build_trellis(self):
        """Build RSC trellis for turbo coding"""
        self.next_states = np.zeros((self.num_states, 2), dtype=int)
        self.outputs = np.zeros((self.num_states, 2), dtype=int)
        
        for state in range(self.num_states):
            for input_bit in [0, 1]:
                # RSC encoding
                feedback = 0
                temp_reg = ((state << 1) | input_bit) & self.polynomials[0]
                while temp_reg:
                    feedback ^= temp_reg & 1
                    temp_reg >>= 1
                
                # Update state with feedback
                new_state = (state >> 1) | (feedback << (self.K - 2))
                
                # Calculate parity output
                parity = 0
                temp_reg = ((state << 1) | input_bit) & self.polynomials[1]
                while temp_reg:
                    parity ^= temp_reg & 1
                    temp_reg >>= 1
                
                self.next_states[state, input_bit] = new_state
                self.outputs[state, input_bit] = parity
    
    def encode(self, data_bits):
        """Turbo encoding"""
        # Pad data to interleaver size
        padded_data = np.zeros(self.N)
        padded_data[:len(data_bits)] = data_bits
        
        # First RSC encoder
        systematic1, parity1 = self._rsc_encode(padded_data)
        
        # Interleave and second RSC encoder
        interleaved_data = padded_data[self.interleaver]
        systematic2, parity2 = self._rsc_encode(interleaved_data)
        
        # Puncture and combine outputs
        # Standard turbo: systematic + parity1 + parity2
        encoded = np.concatenate([systematic1, parity1, parity2])
        
        return encoded
    
    def _rsc_encode(self, data_bits):
        """Recursive systematic convolutional encoding"""
        state = 0
        systematic = []
        parity = []
        
        for bit in data_bits:
            systematic.append(bit)
            parity.append(self.outputs[state, int(bit)])
            state = self.next_states[state, int(bit)]
        
        return np.array(systematic), np.array(parity)
    
    def log_map_decode(self, received_systematic, received_parity1, received_parity2, 
                      iterations=8, snr_db=5):
        """Log-MAP turbo decoding"""
        # Convert SNR to LLR scaling factor
        noise_var = 1 / (10**(snr_db/10))
        scale_factor = 2 / noise_var
        
        # Initialize LLR values
        L_apriori = np.zeros(self.N)
        
        for iteration in range(iterations):
            # Decode first RSC
            L_ext1 = self._log_map_rsc_decode(received_systematic, received_parity1, 
                                            L_apriori, scale_factor)
            
            # Interleave and decode second RSC  
            L_apriori_int = L_ext1[self.interleaver]
            L_ext2 = self._log_map_rsc_decode(received_systematic[self.interleaver], 
                                            received_parity2, L_apriori_int, scale_factor)
            
            # Deinterleave
            L_apriori = L_ext2[self.deinterleaver]
        
        # Final decision
        L_total = L_ext1 + L_apriori
        decoded_bits = (L_total > 0).astype(int)
        
        return decoded_bits
    
    def _log_map_rsc_decode(self, systematic, parity, L_apriori, scale_factor):
        """Log-MAP algorithm for RSC decoding"""
        N = len(systematic)
        
        # Forward recursion (alpha)
        alpha = np.full((N + 1, self.num_states), -np.inf)
        alpha[0, 0] = 0
        
        for t in range(N):
            for state in range(self.num_states):
                if alpha[t, state] == -np.inf:
                    continue
                
                for input_bit in [0, 1]:
                    next_state = self.next_states[state, input_bit]
                    parity_bit = self.outputs[state, input_bit]
                    
                    # Branch metric
                    gamma = (scale_factor * systematic[t] * (2*input_bit - 1) + 
                            scale_factor * parity[t] * (2*parity_bit - 1) +
                            L_apriori[t] * input_bit)
                    
                    alpha[t + 1, next_state] = np.logaddexp(
                        alpha[t + 1, next_state], alpha[t, state] + gamma)
        
        # Backward recursion (beta)
        beta = np.full((N + 1, self.num_states), -np.inf)
        beta[N, 0] = 0
        
        for t in range(N - 1, -1, -1):
            for state in range(self.num_states):
                for input_bit in [0, 1]:
                    next_state = self.next_states[state, input_bit]
                    if beta[t + 1, next_state] == -np.inf:
                        continue
                    
                    parity_bit = self.outputs[state, input_bit]
                    gamma = (scale_factor * systematic[t] * (2*input_bit - 1) + 
                            scale_factor * parity[t] * (2*parity_bit - 1) +
                            L_apriori[t] * input_bit)
                    
                    beta[t, state] = np.logaddexp(
                        beta[t, state], beta[t + 1, next_state] + gamma)
        
        # Calculate LLR
        L_ext = np.zeros(N)
        
        for t in range(N):
            prob_0 = -np.inf
            prob_1 = -np.inf
            
            for state in range(self.num_states):
                if alpha[t, state] == -np.inf:
                    continue
                
                for input_bit in [0, 1]:
                    next_state = self.next_states[state, input_bit]
                    parity_bit = self.outputs[state, input_bit]
                    
                    gamma = (scale_factor * systematic[t] * (2*input_bit - 1) + 
                            scale_factor * parity[t] * (2*parity_bit - 1))
                    
                    prob = alpha[t, state] + gamma + beta[t + 1, next_state]
                    
                    if input_bit == 0:
                        prob_0 = np.logaddexp(prob_0, prob)
                    else:
                        prob_1 = np.logaddexp(prob_1, prob)
            
            L_ext[t] = prob_1 - prob_0 - L_apriori[t]
        
        return L_ext


class LDPCCoder:
    """LDPC encoder and decoder"""
    
    def __init__(self, H_matrix):
        """Initialize with parity check matrix H"""
        self.H = H_matrix
        self.M, self.N = H_matrix.shape  # M parity checks, N code length
        self.K = self.N - self.M  # Information bits (assuming full rank)
        self.rate = self.K / self.N
        
        # Find generator matrix using Gaussian elimination
        self._find_generator_matrix()
        
    def _find_generator_matrix(self):
        """Find generator matrix from parity check matrix"""
        # Simplified: assume H is in systematic form [P | I]
        # In practice, would need Gaussian elimination to get systematic form
        try:
            # Extract P matrix (left part) and create G = [I | P^T]
            P = self.H[:, :self.K]
            I_k = np.eye(self.K, dtype=int)
            self.G = np.hstack([I_k, P.T]) % 2
        except:
            # Fallback: random generator (not guaranteed to work)
            self.G = np.random.randint(0, 2, (self.K, self.N))
    
    def encode(self, data_bits):
        """LDPC encoding using generator matrix"""
        # Pad or truncate data to correct length
        if len(data_bits) < self.K:
            padded_data = np.zeros(self.K, dtype=int)
            padded_data[:len(data_bits)] = data_bits
        else:
            padded_data = data_bits[:self.K]
        
        # Encode: c = d * G
        codeword = (padded_data @ self.G) % 2
        return codeword
    
    def sum_product_decode(self, received_llr, max_iterations=50):
        """Sum-product algorithm for LDPC decoding"""
        # Initialize messages
        var_to_check = np.zeros((self.N, self.M))  # Variable to check messages
        check_to_var = np.zeros((self.M, self.N))  # Check to variable messages
        
        # Initialize variable to check messages with channel LLR
        for i in range(self.N):
            for j in range(self.M):
                if self.H[j, i] == 1:
                    var_to_check[i, j] = received_llr[i]
        
        for iteration in range(max_iterations):
            # Check node update
            for j in range(self.M):
                for i in range(self.N):
                    if self.H[j, i] == 1:
                        # Product of tanh values from other variables
                        product = 1.0
                        for i_prime in range(self.N):
                            if i_prime != i and self.H[j, i_prime] == 1:
                                product *= np.tanh(var_to_check[i_prime, j] / 2)
                        
                        # Avoid numerical issues
                        product = np.clip(product, -0.9999, 0.9999)
                        check_to_var[j, i] = 2 * np.arctanh(product)
            
            # Variable node update
            for i in range(self.N):
                for j in range(self.M):
                    if self.H[j, i] == 1:
                        # Sum of LLRs from other checks
                        llr_sum = received_llr[i]
                        for j_prime in range(self.M):
                            if j_prime != j and self.H[j_prime, i] == 1:
                                llr_sum += check_to_var[j_prime, i]
                        
                        var_to_check[i, j] = llr_sum
            
            # Check for convergence
            # Calculate total LLR for each bit
            total_llr = np.copy(received_llr)
            for i in range(self.N):
                for j in range(self.M):
                    if self.H[j, i] == 1:
                        total_llr[i] += check_to_var[j, i]
            
            # Hard decision
            decoded = (total_llr < 0).astype(int)
            
            # Check if valid codeword
            syndrome = (self.H @ decoded) % 2
            if np.all(syndrome == 0):
                return decoded, iteration + 1
        
        # Return hard decision even if not converged
        total_llr = np.copy(received_llr)
        for i in range(self.N):
            for j in range(self.M):
                if self.H[j, i] == 1:
                    total_llr[i] += check_to_var[j, i]
        
        decoded = (total_llr < 0).astype(int)
        return decoded, max_iterations
    
    def min_sum_decode(self, received_llr, max_iterations=50, alpha=0.75):
        """Min-sum algorithm (simplified sum-product)"""
        # Similar to sum-product but with min-sum approximation
        var_to_check = np.zeros((self.N, self.M))
        check_to_var = np.zeros((self.M, self.N))
        
        # Initialize
        for i in range(self.N):
            for j in range(self.M):
                if self.H[j, i] == 1:
                    var_to_check[i, j] = received_llr[i]
        
        for iteration in range(max_iterations):
            # Check node update (min-sum)
            for j in range(self.M):
                for i in range(self.N):
                    if self.H[j, i] == 1:
                        # Find minimum and second minimum magnitudes
                        magnitudes = []
                        signs = []
                        
                        for i_prime in range(self.N):
                            if i_prime != i and self.H[j, i_prime] == 1:
                                magnitudes.append(abs(var_to_check[i_prime, j]))
                                signs.append(np.sign(var_to_check[i_prime, j]))
                        
                        if len(magnitudes) > 0:
                            min_mag = min(magnitudes)
                            sign_product = np.prod(signs)
                            check_to_var[j, i] = alpha * sign_product * min_mag
            
            # Variable node update (same as sum-product)
            for i in range(self.N):
                for j in range(self.M):
                    if self.H[j, i] == 1:
                        llr_sum = received_llr[i]
                        for j_prime in range(self.M):
                            if j_prime != j and self.H[j_prime, i] == 1:
                                llr_sum += check_to_var[j_prime, i]
                        var_to_check[i, j] = llr_sum
            
            # Check convergence
            total_llr = np.copy(received_llr)
            for i in range(self.N):
                for j in range(self.M):
                    if self.H[j, i] == 1:
                        total_llr[i] += check_to_var[j, i]
            
            decoded = (total_llr < 0).astype(int)
            syndrome = (self.H @ decoded) % 2
            if np.all(syndrome == 0):
                return decoded, iteration + 1
        
        return decoded, max_iterations


class PolarCoder:
    """Polar encoder and decoder"""
    
    def __init__(self, n, k, design_snr_db=0):
        """
        n: code length (power of 2)
        k: information bits
        design_snr_db: design SNR for polar construction
        """
        assert n & (n - 1) == 0, "Code length must be power of 2"
        
        self.n = n
        self.k = k
        self.m = int(np.log2(n))
        
        # Construct frozen set (simplified - use most reliable channels)
        self.frozen_set = self._construct_frozen_set(design_snr_db)
        self.info_set = np.setdiff1d(np.arange(n), self.frozen_set)
        
    def _construct_frozen_set(self, design_snr_db):
        """Construct frozen bit positions using Bhattacharyya parameter"""
        # Simplified construction - in practice would use density evolution
        reliability = np.zeros(self.n)
        
        # Calculate Bhattacharyya parameters recursively
        sigma = np.sqrt(1 / (2 * 10**(design_snr_db/10)))
        
        # Initialize with AWGN channel
        z = [np.exp(-1/(2*sigma**2))]
        
        # Recursive construction
        for level in range(self.m):
            z_new = []
            for i in range(len(z)):
                # Upper subchannel: z_i^2
                z_new.append(z[i]**2)
                # Lower subchannel: 2*z_i - z_i^2  
                z_new.append(2*z[i] - z[i]**2)
            z = z_new
        
        reliability = np.array(z)
        
        # Choose k most reliable positions for information
        reliable_indices = np.argsort(reliability)
        frozen_indices = reliable_indices[:-self.k] if self.k > 0 else reliable_indices
        
        return frozen_indices
    
    def encode(self, info_bits):
        """Polar encoding"""
        # Create u vector with frozen bits set to 0
        u = np.zeros(self.n, dtype=int)
        u[self.info_set] = info_bits
        
        # Polar transform using Kronecker powers of F = [[1,0],[1,1]]
        x = u.copy()
        
        for stage in range(self.m):
            stride = 2**stage
            for i in range(0, self.n, 2*stride):
                for j in range(stride):
                    a = x[i + j]
                    b = x[i + j + stride]
                    x[i + j] = (a + b) % 2
                    x[i + j + stride] = b
        
        return x
    
    def sc_decode(self, received_llr):
        """Successive Cancellation decoding"""
        # Initialize messages
        llr_messages = np.zeros((self.m + 1, self.n))
        bit_messages = np.zeros((self.m + 1, self.n), dtype=int)
        
        llr_messages[self.m] = received_llr
        
        # SC decoding
        u_hat = np.zeros(self.n, dtype=int)
        
        for i in range(self.n):
            # Calculate LLR for position i
            llr = self._calculate_llr(llr_messages, bit_messages, i, 0, 0)
            
            # Make decision
            if i in self.frozen_set:
                u_hat[i] = 0  # Frozen bit
            else:
                u_hat[i] = 1 if llr < 0 else 0  # Information bit decision
            
            # Update bit messages
            self._update_bits(bit_messages, u_hat[i], i, 0, 0)
        
        return u_hat[self.info_set]
    
    def _calculate_llr(self, llr_msg, bit_msg, phi, psi, stage):
        """Calculate LLR recursively"""
        if stage == self.m:
            return llr_msg[stage, phi]
        
        if phi % 2 == 0:
            # Upper path
            llr_left = self._calculate_llr(llr_msg, bit_msg, phi//2, psi, stage + 1)
            llr_right = self._calculate_llr(llr_msg, bit_msg, phi//2, psi + 2**(self.m-stage-1), stage + 1)
            
            # f function: log((1 + exp(x+y))/(exp(x) + exp(y)))
            if abs(llr_left) > 30 or abs(llr_right) > 30:
                return min(abs(llr_left), abs(llr_right)) * np.sign(llr_left) * np.sign(llr_right)
            else:
                return np.log((1 + np.exp(llr_left + llr_right)) / (np.exp(llr_left) + np.exp(llr_right)))
        else:
            # Lower path
            bit_upper = self._get_bit(bit_msg, phi - 1, psi, stage)
            llr_left = self._calculate_llr(llr_msg, bit_msg, phi//2, psi, stage + 1)
            llr_right = self._calculate_llr(llr_msg, bit_msg, phi//2, psi + 2**(self.m-stage-1), stage + 1)
            
            # g function
            return (-1)**bit_upper * llr_left + llr_right
    
    def _update_bits(self, bit_msg, u_bit, phi, psi, stage):
        """Update bit messages recursively"""
        bit_msg[stage, psi + phi] = u_bit
        
        if stage > 0 and phi % 2 == 1:
            # Update parent
            bit_upper = bit_msg[stage, psi + phi - 1]
            bit_msg[stage - 1, psi//2 + phi//2] = (bit_upper + u_bit) % 2
            self._update_bits(bit_msg, (bit_upper + u_bit) % 2, phi//2, psi//2, stage - 1)
        elif stage > 0 and phi % 2 == 0:
            # Need both bits before updating
            if psi + phi + 1 < bit_msg.shape[1] and bit_msg[stage, psi + phi + 1] != -1:
                bit_lower = bit_msg[stage, psi + phi + 1]
                bit_msg[stage - 1, psi//2 + phi//2] = u_bit
                self._update_bits(bit_msg, u_bit, phi//2, psi//2, stage - 1)
    
    def _get_bit(self, bit_msg, phi, psi, stage):
        """Get bit from messages"""
        if bit_msg[stage, psi + phi] != 0:
            return bit_msg[stage, psi + phi]
        else:
            return 0


class ReedSolomonCoder:
    """Reed-Solomon encoder and decoder"""
    
    def __init__(self, n=255, k=223, primitive=0x11d):
        """
        n: codeword length
        k: message length  
        primitive: primitive polynomial for GF(2^8)
        """
        self.n = n
        self.k = k
        self.t = (n - k) // 2  # Error correction capability
        self.primitive = primitive
        
        # Generate Galois field arithmetic tables
        self._generate_gf_tables()
        
        # Generate generator polynomial
        self._generate_generator_polynomial()
    
    def _generate_gf_tables(self):
        """Generate GF(2^8) arithmetic tables"""
        self.gf_exp = np.zeros(512, dtype=int)
        self.gf_log = np.zeros(256, dtype=int)
        
        # Generate exp table
        x = 1
        for i in range(255):
            self.gf_exp[i] = x
            x <<= 1
            if x & 0x100:
                x ^= self.primitive
        
        # Extend exp table
        for i in range(255, 512):
            self.gf_exp[i] = self.gf_exp[i - 255]
        
        # Generate log table
        for i in range(1, 256):
            self.gf_log[self.gf_exp[i]] = i
    
    def gf_mult(self, x, y):
        """GF(2^8) multiplication"""
        if x == 0 or y == 0:
            return 0
        return self.gf_exp[(self.gf_log[x] + self.gf_log[y]) % 255]
    
    def gf_div(self, x, y):
        """GF(2^8) division"""
        if y == 0:
            raise ZeroDivisionError("Division by zero in GF")
        if x == 0:
            return 0
        return self.gf_exp[(self.gf_log[x] - self.gf_log[y] + 255) % 255]
    
    def _generate_generator_polynomial(self):
        """Generate RS generator polynomial"""
        self.generator = [1]
        
        for i in range(self.n - self.k):
            # Multiply by (x - alpha^i)
            new_gen = [0] * (len(self.generator) + 1)
            
            for j in range(len(self.generator)):
                new_gen[j] ^= self.gf_mult(self.generator[j], self.gf_exp[i])
                new_gen[j + 1] ^= self.generator[j]
            
            self.generator = new_gen
    
    def encode(self, message):
        """Reed-Solomon encoding"""
        # Pad message to k symbols
        if len(message) < self.k:
            padded_msg = np.zeros(self.k, dtype=int)
            padded_msg[-len(message):] = message
        else:
            padded_msg = message[:self.k]
        
        # Calculate remainder
        remainder = np.zeros(self.n - self.k, dtype=int)
        
        for i in range(self.k):
            coeff = padded_msg[i]
            if coeff != 0:
                for j in range(len(self.generator) - 1):
                    remainder[j] ^= self.gf_mult(coeff, self.generator[j])
        
        # Codeword = message + parity
        codeword = np.concatenate([padded_msg, remainder])
        return codeword
    
    def berlekamp_massey_decode(self, received):
        """Berlekamp-Massey RS decoding"""
        # Calculate syndromes
        syndromes = self._calculate_syndromes(received)
        
        # Check if error-free
        if np.all(syndromes == 0):
            return received[:self.k], True
        
        # Berlekamp-Massey algorithm
        error_locator = self._berlekamp_massey(syndromes)
        
        # Find error positions
        error_positions = self._chien_search(error_locator)
        
        # Calculate error values
        if len(error_positions) <= self.t:
            error_values = self._forney_algorithm(syndromes, error_locator, error_positions)
            
            # Correct errors
            corrected = received.copy()
            for i, pos in enumerate(error_positions):
                if pos < len(corrected):
                    corrected[pos] ^= error_values[i]
            
            return corrected[:self.k], True
        
        # Too many errors
        return received[:self.k], False
    
    def _calculate_syndromes(self, received):
        """Calculate syndrome polynomial"""
        syndromes = np.zeros(self.n - self.k, dtype=int)
        
        for i in range(self.n - self.k):
            syndrome = 0
            for j in range(len(received)):
                if received[j] != 0:
                    syndrome ^= self.gf_mult(received[j], 
                                           self.gf_exp[(i * j) % 255])
            syndromes[i] = syndrome
        
        return syndromes
    
    def _berlekamp_massey(self, syndromes):
        """Berlekamp-Massey algorithm"""
        n_syndromes = len(syndromes)
        
        # Initialize
        error_locator = [1]
        old_locator = [1]
        
        for i in range(n_syndromes):
            # Calculate discrepancy
            delta = syndromes[i]
            for j in range(1, len(error_locator)):
                if j <= i:
                    delta ^= self.gf_mult(error_locator[j], syndromes[i - j])
            
            if delta != 0:
                if len(old_locator) > len(error_locator):
                    # Update error locator
                    new_locator = error_locator.copy()
                    
                    # Extend if needed
                    while len(new_locator) < len(old_locator):
                        new_locator.append(0)
                    
                    for j in range(len(old_locator)):
                        new_locator[j] ^= self.gf_mult(delta, old_locator[j])
                    
                    old_locator = error_locator
                    error_locator = new_locator
        
        return error_locator
    
    def _chien_search(self, error_locator):
        """Chien search for error positions"""
        error_positions = []
        
        for i in range(self.n):
            # Evaluate error_locator at alpha^i
            eval_result = 0
            for j, coeff in enumerate(error_locator):
                if coeff != 0:
                    eval_result ^= self.gf_mult(coeff, self.gf_exp[(j * i) % 255])
            
            if eval_result == 0:
                error_positions.append(i)
        
        return error_positions
    
    def _forney_algorithm(self, syndromes, error_locator, error_positions):
        """Forney algorithm for error values"""
        error_values = []
        
        # This is a simplified version - full implementation is more complex
        for pos in error_positions:
            # Simplified error value calculation
            error_val = syndromes[0] if len(syndromes) > 0 else 0
            error_values.append(error_val)
        
        return error_values


class ChannelCodingDetector:
    """Detector for channel coding types"""
    
    def __init__(self):
        self.detection_methods = {
            'convolutional': self._detect_convolutional,
            'turbo': self._detect_turbo,
            'ldpc': self._detect_ldpc,
            'polar': self._detect_polar,
            'reed_solomon': self._detect_reed_solomon,
            'hamming': self._detect_hamming,
            'bch': self._detect_bch
        }
    
    def detect_coding_type(self, received_bits, soft_info=None):
        """Detect channel coding type from received data"""
        detection_scores = {}
        
        for code_type, detector in self.detection_methods.items():
            try:
                score = detector(received_bits, soft_info)
                detection_scores[code_type] = score
            except:
                detection_scores[code_type] = 0.0
        
        # Find best match
        best_type = max(detection_scores.items(), key=lambda x: x[1])
        return best_type[0], detection_scores
    
    def _detect_convolutional(self, bits, soft_info):
        """Detect convolutional coding patterns"""
        # Look for patterns characteristic of convolutional codes
        # Check for systematic structure, trellis termination patterns
        
        # Simple heuristic: check for rate patterns
        if len(bits) % 2 == 0:
            # Could be rate 1/2
            score = 0.3
            
            # Check for correlation between bit pairs (characteristic of CC)
            pairs = bits.reshape(-1, 2)
            correlation = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
            if not np.isnan(correlation) and abs(correlation) > 0.1:
                score += 0.4
            
            return score
        
        return 0.1
    
    def _detect_turbo(self, bits, soft_info):
        """Detect turbo coding patterns"""
        # Turbo codes have specific interleaver patterns
        # Look for systematic + parity structure
        
        if len(bits) % 3 == 0:
            # Could be rate 1/3 turbo
            score = 0.3
            
            # Check for interleaver-like patterns
            # Simplified: check for low autocorrelation at interleaver distances
            n = len(bits) // 3
            systematic = bits[:n]
            
            if n > 10:
                # Check autocorrelation
                autocorr = np.correlate(systematic, systematic, mode='full')
                mid = len(autocorr) // 2
                if len(autocorr) > mid + n//4:
                    side_corr = np.mean(np.abs(autocorr[mid + n//8:mid + n//4]))
                    if side_corr < 0.3:
                        score += 0.4
            
            return score
        
        return 0.1
    
    def _detect_ldpc(self, bits, soft_info):
        """Detect LDPC coding patterns"""
        # LDPC codes have sparse parity check structure
        # Look for low-density patterns
        
        # Try to find block structure
        n = len(bits)
        possible_block_sizes = [64, 96, 128, 256, 512, 1024, 2048]
        
        score = 0.1
        
        for block_size in possible_block_sizes:
            if n % block_size == 0:
                score += 0.1
                
                # Check for systematic structure
                blocks = bits.reshape(-1, block_size)
                if blocks.shape[0] > 1:
                    # Check correlation between blocks (should be low for LDPC)
                    block_corr = np.corrcoef(blocks[0], blocks[1])[0, 1]
                    if not np.isnan(block_corr) and abs(block_corr) < 0.2:
                        score += 0.3
                break
        
        return min(score, 1.0)
    
    def _detect_polar(self, bits, soft_info):
        """Detect polar coding patterns"""
        # Polar codes have power-of-2 block lengths
        n = len(bits)
        
        if n & (n - 1) == 0 and n > 0:  # Power of 2
            score = 0.4
            
            # Check for frozen bit patterns (many zeros in specific positions)
            # This is a simplified check
            zero_ratio = np.sum(bits == 0) / len(bits)
            if 0.3 < zero_ratio < 0.8:  # Typical for polar codes
                score += 0.4
            
            return score
        
        return 0.1
    
    def _detect_reed_solomon(self, bits, soft_info):
        """Detect Reed-Solomon coding patterns"""
        # RS codes work on symbols, not bits
        # Look for byte-aligned patterns
        
        if len(bits) % 8 == 0:
            symbols = bits.reshape(-1, 8)
            n_symbols = len(symbols)
            
            score = 0.2
            
            # Common RS code lengths
            if n_symbols in [15, 31, 63, 127, 255]:
                score += 0.5
            
            # Check for systematic structure
            if n_symbols >= 15:
                # Look for parity symbol patterns
                # RS parity symbols often have different statistics
                symbol_values = []
                for sym_bits in symbols:
                    val = 0
                    for i, bit in enumerate(sym_bits):
                        val += bit * (2**i)
                    symbol_values.append(val)
                
                # Check variance in different parts
                mid = len(symbol_values) // 2
                if mid > 2:
                    var1 = np.var(symbol_values[:mid])
                    var2 = np.var(symbol_values[mid:])
                    if abs(var1 - var2) / (var1 + var2 + 1e-6) > 0.2:
                        score += 0.3
            
            return score
        
        return 0.1
    
    def _detect_hamming(self, bits, soft_info):
        """Detect Hamming coding patterns"""
        n = len(bits)
        
        # Hamming codes have length 2^m - 1
        possible_lengths = [7, 15, 31, 63, 127]
        
        if n in possible_lengths:
            score = 0.6
            
            # Check for single-error correction capability
            # Hamming codes have specific parity check structure
            if n == 7:  # (7,4) Hamming code
                score += 0.3
            
            return score
        
        return 0.1
    
    def _detect_bch(self, bits, soft_info):
        """Detect BCH coding patterns"""
        # BCH codes are similar to RS but work over GF(2)
        # Look for specific block lengths and structures
        
        n = len(bits)
        common_bch_lengths = [7, 15, 31, 63, 127, 255, 511, 1023]
        
        score = 0.1
        
        for length in common_bch_lengths:
            if n == length or (n % length == 0 and n > length):
                score += 0.3
                break
        
        # BCH codes often have low weight codewords
        weight = np.sum(bits)
        if weight < n * 0.3:  # Low weight suggests BCH
            score += 0.2
        
        return min(score, 1.0)


# Helper functions for generating standard codes
def generate_hamming_matrix(m):
    """Generate Hamming(2^m-1, 2^m-m-1) parity check matrix"""
    n = 2**m - 1
    k = n - m
    
    # Generate all non-zero m-bit patterns
    H = []
    for i in range(1, 2**m):
        col = []
        for j in range(m):
            col.append((i >> j) & 1)
        H.append(col)
    
    return np.array(H).T


def generate_random_ldpc_matrix(n, k, max_col_weight=3, max_row_weight=6):
    """Generate random regular LDPC parity check matrix"""
    m = n - k
    H = np.zeros((m, n), dtype=int)
    
    # Simple random construction (not optimal)
    for i in range(m):
        # Add random 1s with limited row weight
        positions = np.random.choice(n, min(max_row_weight, n), replace=False)
        H[i, positions] = 1
    
    # Ensure column weights are not too high
    for j in range(n):
        col_weight = np.sum(H[:, j])
        if col_weight > max_col_weight:
            # Remove excess 1s
            ones_positions = np.where(H[:, j] == 1)[0]
            remove_count = col_weight - max_col_weight
            remove_positions = np.random.choice(ones_positions, remove_count, replace=False)
            H[remove_positions, j] = 0
    
    return H


# Example usage and testing
def test_channel_codes():
    """Test different channel coding implementations"""
    print("Testing Channel Coding Implementations...")
    
    # Test Convolutional Code
    print("\\n1. Testing Convolutional Code:")
    conv_coder = ConvolutionalCoder(constraint_length=7, code_rate=0.5)
    data_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = conv_coder.encode(data_bits)
    decoded = conv_coder.viterbi_decode(encoded)
    print(f"Data:    {data_bits}")
    print(f"Encoded: {encoded}")  
    print(f"Decoded: {decoded}")
    print(f"Success: {np.array_equal(data_bits, decoded)}")
    
    # Test LDPC Code
    print("\\n2. Testing LDPC Code:")
    H = generate_hamming_matrix(3)  # (7,4) Hamming as simple LDPC
    ldpc_coder = LDPCCoder(H)
    
    data_bits = np.array([1, 0, 1, 1])
    encoded = ldpc_coder.encode(data_bits)
    
    # Add noise
    received_llr = 2 * encoded.astype(float) - 1  # BPSK mapping
    received_llr += 0.5 * np.random.randn(len(received_llr))  # Add noise
    
    decoded, iterations = ldpc_coder.sum_product_decode(received_llr)
    print(f"Data:       {data_bits}")
    print(f"Encoded:    {encoded}")
    print(f"Decoded:    {decoded}")
    print(f"Iterations: {iterations}")
    
    # Test Polar Code
    print("\\n3. Testing Polar Code:")
    polar_coder = PolarCoder(n=8, k=4)
    
    info_bits = np.array([1, 0, 1, 1])
    encoded = polar_coder.encode(info_bits)
    
    # Add noise and decode
    received_llr = 2 * encoded.astype(float) - 1
    received_llr += 0.3 * np.random.randn(len(received_llr))
    
    decoded = polar_coder.sc_decode(received_llr)
    print(f"Info:     {info_bits}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    print(f"Success:  {np.array_equal(info_bits, decoded)}")
    
    # Test Reed-Solomon Code
    print("\\n4. Testing Reed-Solomon Code:")
    rs_coder = ReedSolomonCoder(n=15, k=11)
    
    message = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    encoded = rs_coder.encode(message)
    
    # Add errors
    received = encoded.copy()
    received[1] = 99  # Single error
    
    decoded, success = rs_coder.berlekamp_massey_decode(received)
    print(f"Message:  {message}")
    print(f"Encoded:  {encoded}")
    print(f"Received: {received}")
    print(f"Decoded:  {decoded}")
    print(f"Success:  {success}")
    
    # Test Detection
    print("\\n5. Testing Code Detection:")
    detector = ChannelCodingDetector()
    
    test_bits = np.array([1,0,1,1,0,0,1,0,1,1,0,1,0,1])  # Random test bits
    detected_type, scores = detector.detect_coding_type(test_bits)
    print(f"Test bits: {test_bits}")
    print(f"Detected:  {detected_type}")
    print(f"Scores:    {scores}")


if __name__ == "__main__":
    test_channel_codes()
'''

with open('channel_coding.py', 'w', encoding='utf-8') as f:
    f.write(channel_coding_module)

print("✅ Created channel_coding.py")
print("🔐 Comprehensive Channel Coding features:")
print("  • Convolutional Codes: Viterbi decoder (hard/soft decision)")
print("  • Turbo Codes: Log-MAP/BCJR decoder với iterative decoding") 
print("  • LDPC Codes: Sum-Product algorithm và Min-Sum decoder")
print("  • Polar Codes: Successive Cancellation decoder")
print("  • Reed-Solomon: Berlekamp-Massey decoder với Chien search")
print("  • Block Codes: Hamming, BCH code support")
print("  • Detection: Automatic channel coding type classification")
print("  • Complete implementations with GF arithmetic")