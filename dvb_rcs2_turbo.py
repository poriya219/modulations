import numpy as np
from numba import njit, int32, float32

# =============================================================================
# DVB-RCS2 Constants & Parameters
# Based on ETSI EN 301 545-2 & TR 101 545-4
# =============================================================================

# Interleaver Parameters (Table 7-2)
# N_couples: (P, Q0, Q1, Q2, Q3)
# Target is Waveform ID 14 -> 1616 symbols -> 1504 info bits -> 752 couples
INTERLEAVER_PARAMS = {
    48:  (31, 4, 2, 0, 3),   64:  (41, 2, 6, 4, 1),
    212: (137, 0, 6, 4, 9),  220: (143, 4, 2, 8, 5),
    424: (277, 2, 4, 0, 7),  752: (491, 0, 8, 2, 5),
    848: (553, 4, 6, 0, 3)
}

# Puncturing Patterns (Table 7-5)
# '1' = Transmitted, '0' = Punctured
PUNCTURE_PATTERNS = {
    '1/3': {'period': 1, 'W1': [1], 'Y1': [1], 'W2': [1], 'Y2': [1]},
    '1/2': {'period': 2, 'W1': [1, 0], 'Y1': [0, 1], 'W2': [1, 0], 'Y2': [0, 1]},
    '2/3': {'period': 3, 'W1': [1, 0, 0], 'Y1': [0, 1, 0], 'W2': [0, 0, 1], 'Y2': [0, 0, 0]},
    '3/4': {'period': 4, 'W1': [1,0,0,0], 'Y1': [0,1,0,0], 'W2': [0,0,1,0], 'Y2': [0,0,0,0]},
}

# =============================================================================
# NUMBA KERNELS (High Performance Calculation)
# =============================================================================

@njit(cache=True)
def max_star(a, b):
    """Max-Log approximation: max(a, b)"""
    return a if a > b else b

@njit(cache=True)
def mat_mul_gf2(A, B):
    """Matrix Multiplication over GF(2)"""
    dim = 4
    C = np.zeros((dim, dim), dtype=np.int32)
    for i in range(dim):
        for j in range(dim):
            val = 0
            for k in range(dim):
                val ^= (A[i, k] & B[k, j])
            C[i, j] = val
    return C

@njit(cache=True)
def mat_pow_gf2(A, power):
    """Matrix Exponentiation over GF(2)"""
    dim = 4
    res = np.eye(dim, dtype=np.int32)
    base = A.copy()
    while power > 0:
        if power % 2 == 1:
            res = mat_mul_gf2(res, base)
        base = mat_mul_gf2(base, base)
        power //= 2
    return res

@njit(cache=True)
def solve_circular_state_gf2(G_pow_N, Z_N):
    """
    Solve equation: (I + G^N) * S_c = Z_N over GF(2)
    to find circular state S_c.
    Uses Gaussian elimination.
    """
    dim = 4
    I = np.eye(dim, dtype=np.int32)
    A = (I + G_pow_N) % 2
    b = np.zeros(dim, dtype=np.int32)
    
    # Convert Z_N integer to bit vector
    for i in range(dim):
        b[i] = (Z_N >> i) & 1

    # Augmented Matrix [A|b]
    M = np.zeros((4, 5), dtype=np.int32)
    M[:, :4] = A
    M[:, 4] = b

    # Gaussian Elimination
    for i in range(dim):
        # Pivot
        if M[i, i] == 0:
            for k in range(i + 1, dim):
                if M[k, i] == 1:
                    # Swap rows
                    row_tmp = M[i, :].copy()
                    M[i, :] = M[k, :]
                    M[k, :] = row_tmp
                    break
        
        if M[i, i] == 1:
            for k in range(i + 1, dim):
                if M[k, i] == 1:
                    M[k, :] ^= M[i, :]

    # Back Substitution
    x = np.zeros(dim, dtype=np.int32)
    for i in range(dim - 1, -1, -1):
        sum_val = M[i, 4]
        for j in range(i + 1, dim):
            sum_val ^= (M[i, j] & x[j])
        x[i] = sum_val

    # Convert bit vector back to integer state
    state = 0
    for i in range(dim):
        if x[i]:
            state |= (1 << i)
    return state

@njit(cache=True)
def bcjr_max_log_map(Lc_A, Lc_B, Lc_W, Lc_Y, La_A, La_B, 
                     next_st, out_W, out_Y, prev_st, prev_inp, 
                     N, scaling_factor):
    """
    Max-Log-MAP Decoder Kernel
    Implements Double Pass for Circular Trellis and Extrinsic Scaling.
    """
    n_states = 16
    NEG_INF_VAL = -1e9
    
    # 1. Compute Branch Metrics (Gamma)
    # Dimensions: [Time, State, Input_Symbol(0..3)]
    gamma = np.zeros((N, n_states, 4), dtype=np.float32)
    
    for k in range(N):
        # Combined Input LLR (Channel Systematic + A-priori)
        # Note: DVB uses duo-binary symbols u = (A, B)
        # A is MSB (bit 1), B is LSB (bit 0)
        in_A = Lc_A[k] + La_A[k]
        in_B = Lc_B[k] + La_B[k]
        
        par_W = Lc_W[k]
        par_Y = Lc_Y[k]
        
        for s in range(n_states):
            for inp in range(4):
                # Decompose input symbol index to bits
                bit_A = (inp >> 1) & 1
                bit_B = inp & 1
                
                # Coded bits from Trellis LUT
                bit_W = out_W[s, inp]
                bit_Y = out_Y[s, inp]
                
                # Metric: LLR definition log(P(0)/P(1))
                # Correlation metric: 0.5 * LLR * (1 - 2*bit)
                # If bit=0 -> +0.5*LLR, if bit=1 -> -0.5*LLR
                m = 0.0
                m += in_A * (0.5 if bit_A == 0 else -0.5)
                m += in_B * (0.5 if bit_B == 0 else -0.5)
                m += par_W * (0.5 if bit_W == 0 else -0.5)
                m += par_Y * (0.5 if bit_Y == 0 else -0.5)
                
                gamma[k, s, inp] = m

    # 2. Forward Recursion (Alpha) - Double Pass
    alpha = np.zeros((N + 1, n_states), dtype=np.float32)
    
    # Pass 1: Convergence
    # Initialize alpha[0] to 0 (equal prob)
    for k in range(N):
        for ns in range(n_states):
            max_val = NEG_INF_VAL
            for idx in range(4): # 4 incoming branches per state
                ps = prev_st[ns, idx]
                inp = prev_inp[ns, idx]
                tmp = alpha[k, ps] + gamma[k, ps, inp]
                if tmp > max_val: max_val = tmp
            alpha[k+1, ns] = max_val
        
        # Normalize to avoid overflow
        norm = alpha[k+1, 0]
        for s in range(n_states): alpha[k+1, s] -= norm
        
    # Set Circular Boundary: alpha[0] = alpha[N]
    for s in range(n_states):
        alpha[0, s] = alpha[N, s]
        
    # Pass 2: Calculation
    for k in range(N):
        for ns in range(n_states):
            max_val = NEG_INF_VAL
            for idx in range(4):
                ps = prev_st[ns, idx]
                inp = prev_inp[ns, idx]
                tmp = alpha[k, ps] + gamma[k, ps, inp]
                if tmp > max_val: max_val = tmp
            alpha[k+1, ns] = max_val
        
        norm = alpha[k+1, 0]
        for s in range(n_states): alpha[k+1, s] -= norm

    # 3. Backward Recursion (Beta) - Double Pass
    beta = np.zeros((N + 1, n_states), dtype=np.float32)
    
    # Pass 1: Convergence
    for k in range(N - 1, -1, -1):
        for s in range(n_states):
            max_val = NEG_INF_VAL
            for inp in range(4):
                ns = next_st[s, inp]
                tmp = beta[k+1, ns] + gamma[k, s, inp]
                if tmp > max_val: max_val = tmp
            beta[k, s] = max_val
        
        norm = beta[k, 0]
        for s in range(n_states): beta[k, s] -= norm
        
    # Set Circular Boundary
    for s in range(n_states):
        beta[N, s] = beta[0, s]
        
    # Pass 2: Calculation
    for k in range(N - 1, -1, -1):
        for s in range(n_states):
            max_val = NEG_INF_VAL
            for inp in range(4):
                ns = next_st[s, inp]
                tmp = beta[k+1, ns] + gamma[k, s, inp]
                if tmp > max_val: max_val = tmp
            beta[k, s] = max_val
        
        norm = beta[k, 0]
        for s in range(n_states): beta[k, s] -= norm

    # 4. Extrinsic Information Calculation
    Le_A = np.zeros(N, dtype=np.float64)
    Le_B = np.zeros(N, dtype=np.float64)
    
    # APP for symbol u in {00, 01, 10, 11}
    # Index 0->00, 1->01, 2->10, 3->11
    
    for k in range(N):
        app_sym = np.full(4, NEG_INF_VAL, dtype=np.float32)
        
        for s in range(n_states):
            for inp in range(4):
                ns = next_st[s, inp]
                # P(transition) = alpha * gamma * beta
                metric = alpha[k, s] + gamma[k, s, inp] + beta[k+1, ns]
                if metric > app_sym[inp]:
                    app_sym[inp] = metric
        
        # Marginalize for Bit A (MSB)
        # P(A=0) = max(P(00), P(01))
        # P(A=1) = max(P(10), P(11))
        prob_A0 = max_star(app_sym[0], app_sym[1])
        prob_A1 = max_star(app_sym[2], app_sym[3])
        
        # Marginalize for Bit B (LSB)
        # P(B=0) = max(P(00), P(10))
        # P(B=1) = max(P(01), P(11))
        prob_B0 = max_star(app_sym[0], app_sym[2])
        prob_B1 = max_star(app_sym[1], app_sym[3])
        
        # Posterior LLR = log(P(0)/P(1))
        L_post_A = prob_A0 - prob_A1
        L_post_B = prob_B0 - prob_B1
        
        # Extrinsic = Posterior - (Channel + A-priori)
        Le_A[k] = L_post_A - (Lc_A[k] + La_A[k])
        Le_B[k] = L_post_B - (Lc_B[k] + La_B[k])
        
        # Apply Scaling Factor (Brandt 2013 / TR 101 545-4)
        Le_A[k] *= scaling_factor
        Le_B[k] *= scaling_factor
        
        # Clipping for stability
        limit = 300.0
        if Le_A[k] > limit: Le_A[k] = limit
        if Le_A[k] < -limit: Le_A[k] = -limit
        if Le_B[k] > limit: Le_B[k] = limit
        if Le_B[k] < -limit: Le_B[k] = -limit
        
    return Le_A, Le_B

# =============================================================================
# DVBRCS2 TURBO CODE CLASS
# =============================================================================

class DVBRCS2_Turbo:
    def __init__(self, N_couples, code_rate, iterations=8):
        self.N = N_couples
        self.k_info = N_couples * 2
        self.iterations = iterations
        self.punct = PUNCTURE_PATTERNS[code_rate]
        
        # 1. Initialize Interleaver
        if self.N not in INTERLEAVER_PARAMS:
            raise ValueError(f"Block size {self.N} not in standard tables.")
        self._init_interleaver()
        
        # 2. Initialize Trellis (16-State CRSC)
        self._init_trellis()
        
        # 3. Calculate Coded Size
        self._calc_coded_size()
        
        # 4. JIT Warmup
        try:
            self.decode(np.zeros(self.n_coded))
        except:
            pass

    def _init_interleaver(self):
        """Standard DVB-RCS2 Interleaver Generation"""
        P, Q0, Q1, Q2, Q3 = INTERLEAVER_PARAMS[self.N]
        self.perm = np.zeros(self.N, dtype=np.int32)
        for i in range(self.N):
            r = i % 4
            d = 0
            if r == 1: d = Q0
            elif r == 2: d = Q1
            elif r == 3: d = Q2
            
            # Equation: pi(i) = (P * (i + d + Q3 * floor(i/4))) % N
            idx = (P * (i + d + Q3 * (i // 4))) % self.N
            self.perm[i] = idx
        self.inv_perm = np.argsort(self.perm).astype(np.int32)

    def _init_trellis(self):
        """
        16-State Trellis Generation
        Polynomials (Octal): Feedback=23, W=35, Y=27
        States: S = (s0, s1, s2, s3)
        """
        self.next_state = np.zeros((16, 4), dtype=np.int32)
        self.out_W = np.zeros((16, 4), dtype=np.int32)
        self.out_Y = np.zeros((16, 4), dtype=np.int32)
        self.G_matrix = np.zeros((4, 4), dtype=np.int32) # For tail-biting
        
        for s in range(16):
            s0 = (s >> 0) & 1
            s1 = (s >> 1) & 1
            s2 = (s >> 2) & 1
            s3 = (s >> 3) & 1
            
            for inp in range(4):
                # Input symbols u = (A, B)
                A = (inp >> 1) & 1
                B = inp & 1
                
                # Feedback Poly 23 (10011 -> 1 + D^3 + D^4)
                # Taps at Input, D3(s2), D4(s3)
                dk = A ^ B ^ s2 ^ s3
                
                # Parity W Poly 35 (11101 -> 1 + D + D^2 + D^4)
                # Taps at dk, D1(s0), D2(s1), D4(s3)
                w = dk ^ s0 ^ s1 ^ s3
                
                # Parity Y Poly 27 (10111 -> 1 + D^2 + D^3 + D^4)
                # Taps at dk, D2(s1), D3(s2), D4(s3)
                y = dk ^ s1 ^ s2 ^ s3
                
                # Shift Register Update
                # s0_new = dk
                # s1_new = s0
                # s2_new = s1
                # s3_new = s2
                ns = (s2 << 3) | (s1 << 2) | (s0 << 1) | dk
                
                self.next_state[s, inp] = ns
                self.out_W[s, inp] = w
                self.out_Y[s, inp] = y
                
                # Build G matrix (Zero Input Response) for Tail-biting
                if inp == 0:
                    # dk = s2 ^ s3
                    # ns0 = s2 ^ s3
                    # ns1 = s0
                    # ns2 = s1
                    # ns3 = s2
                    # Columns: s0, s1, s2, s3
                    self.G_matrix[0, 2] = 1; self.G_matrix[0, 3] = 1
                    self.G_matrix[1, 0] = 1
                    self.G_matrix[2, 1] = 1
                    self.G_matrix[3, 2] = 1

        # Reverse Trellis for Decoder
        self.prev_state = np.full((16, 4), -1, dtype=np.int32)
        self.prev_input = np.full((16, 4), -1, dtype=np.int32)
        counts = np.zeros(16, dtype=int)
        for s in range(16):
            for inp in range(4):
                ns = self.next_state[s, inp]
                idx = counts[ns]
                if idx < 4:
                    self.prev_state[ns, idx] = s
                    self.prev_input[ns, idx] = inp
                    counts[ns] += 1

    def _calc_coded_size(self):
        period = self.punct['period']
        bits_per_period = 2 * period + sum(self.punct['W1']) + sum(self.punct['Y1']) + \
                          sum(self.punct['W2']) + sum(self.punct['Y2'])
        self.n_coded = (self.N // period) * bits_per_period

    def _encode_component(self, A, B):
        # Tail-biting Encoding
        
        # 1. Calculate Zero State Response
        state = 0
        for i in range(self.N):
            inp = (A[i] << 1) | B[i]
            state = self.next_state[state, inp]
        Z_N = state
        
        # 2. Calculate Circular Start State
        # S_c = (I + G^N)^-1 * Z_N
        G_pow_N = mat_pow_gf2(self.G_matrix, self.N)
        start_state = solve_circular_state_gf2(G_pow_N, Z_N)
        
        # 3. Encode with Circular State
        W = np.zeros(self.N, dtype=np.int32)
        Y = np.zeros(self.N, dtype=np.int32)
        state = start_state
        for i in range(self.N):
            inp = (A[i] << 1) | B[i]
            W[i] = self.out_W[state, inp]
            Y[i] = self.out_Y[state, inp]
            state = self.next_state[state, inp]
            
        return W, Y

    def encode(self, bits):
        """Encode bits into DVB-RCS2 Turbo Codeword"""
        bits = np.array(bits, dtype=np.int32)
        # Duo-binary split
        A = bits[0::2]
        B = bits[1::2]
        
        # 1. Encoder 1 (Natural Order)
        W1, Y1 = self._encode_component(A, B)
        
        # 2. Interleave
        A_int = A[self.perm]
        B_int = B[self.perm]
        
        # 3. Encoder 2 (Interleaved Order)
        W2, Y2 = self._encode_component(A_int, B_int)
        
        # 4. Puncturing and Multiplexing
        coded = []
        period = self.punct['period']
        for i in range(self.N):
            p = i % period
            # Systematic (Always transmitted)
            coded.append(A[i])
            coded.append(B[i])
            # Parity
            if self.punct['W1'][p]: coded.append(W1[i])
            if self.punct['Y1'][p]: coded.append(Y1[i])
            if self.punct['W2'][p]: coded.append(W2[i])
            if self.punct['Y2'][p]: coded.append(Y2[i])
            
        return np.array(coded, dtype=np.int32)

    def decode(self, llr):
        """Decode LLRs into info bits"""
        llr = np.array(llr, dtype=np.float32)
        
        # 1. Depuncturing
        Lc_A = np.zeros(self.N, dtype=np.float32)
        Lc_B = np.zeros(self.N, dtype=np.float32)
        Lc_W1 = np.zeros(self.N, dtype=np.float32)
        Lc_Y1 = np.zeros(self.N, dtype=np.float32)
        Lc_W2 = np.zeros(self.N, dtype=np.float32)
        Lc_Y2 = np.zeros(self.N, dtype=np.float32)
        
        idx = 0
        period = self.punct['period']
        for i in range(self.N):
            p = i % period
            # Systematic
            Lc_A[i] = llr[idx]; idx += 1
            Lc_B[i] = llr[idx]; idx += 1
            # Parity
            if self.punct['W1'][p]: Lc_W1[i] = llr[idx]; idx += 1
            if self.punct['Y1'][p]: Lc_Y1[i] = llr[idx]; idx += 1
            if self.punct['W2'][p]: Lc_W2[i] = llr[idx]; idx += 1
            if self.punct['Y2'][p]: Lc_Y2[i] = llr[idx]; idx += 1
            
        # 2. Iterative Decoding
        La_A = np.zeros(self.N, dtype=np.float64)
        La_B = np.zeros(self.N, dtype=np.float64)
        
        for i in range(self.iterations):
            # Extrinsic Scaling Factor (Crucial for Max-Log-MAP)
            # Typically 0.7 for all iterations except the last one
            sf = 0.7 if i < self.iterations - 1 else 1.0
            
            # Decoder 1
            Le1_A, Le1_B = bcjr_max_log_map(
                Lc_A, Lc_B, Lc_W1, Lc_Y1, La_A, La_B,
                self.next_state, self.out_W, self.out_Y,
                self.prev_state, self.prev_input,
                self.N, sf
            )
            
            # Interleave Extrinsic Info -> Apriori for Dec 2
            La2_A = Le1_A[self.perm]
            La2_B = Le1_B[self.perm]
            
            # Interleave Systematic LLRs for Dec 2
            Lc_A_int = Lc_A[self.perm]
            Lc_B_int = Lc_B[self.perm]
            
            # Decoder 2
            Le2_A, Le2_B = bcjr_max_log_map(
                Lc_A_int, Lc_B_int, Lc_W2, Lc_Y2, La2_A, La2_B,
                self.next_state, self.out_W, self.out_Y,
                self.prev_state, self.prev_input,
                self.N, sf
            )
            
            # Deinterleave Extrinsic Info -> Apriori for Dec 1
            La_A = Le2_A[self.inv_perm]
            La_B = Le2_B[self.inv_perm]
            
        # 3. Final Hard Decision
        # Total LLR = Channel + Apriori + Extrinsic(Dec1)
        # Note: In the final step, we effectively sum up info from both decoders
        L_final_A = Lc_A + La_A + Le1_A
        L_final_B = Lc_B + La_B + Le1_B
        
        decoded = np.zeros(self.k_info, dtype=np.int32)
        # LLR definition: log(P(0)/P(1)). Positive -> 0, Negative -> 1
        decoded[0::2] = np.where(L_final_A < 0, 1, 0)
        decoded[1::2] = np.where(L_final_B < 0, 1, 0)
        
        return decoded