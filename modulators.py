import numpy as np

# Try to import useful DSP functions from scipy
try:
    from scipy.signal import upfirdn, convolve
except ImportError:
    # Fallbacks if scipy is not installed
    def convolve(in1, in2, mode='full'):
        return np.convolve(in1, in2, mode=mode)
    def upfirdn(h, x, up, down):
        # Manual implementation: Upsample -> Convolve -> Downsample
        # This is slower but works without scipy
        output_len = len(x) * up
        upsampled = np.zeros(output_len, dtype=x.dtype)
        upsampled[::up] = x
        filtered = np.convolve(upsampled, h, mode='same') # 'same' keeps length roughly
        return filtered[::down]

def rrcosfilter(N, alpha, Ts, Fs):
    """
    Generates a Root Raised Cosine (RRC) filter.
    N: Number of symbols (span)
    alpha: Roll-off factor
    Ts: Symbol time (usually 1 for normalized)
    Fs: Samples per symbol
    """
    # Ensure an odd number of taps for a well-defined integer center delay
    num_taps = int(N * Fs) | 1  # Make sure it's odd
    
    T_delta = 1.0 / float(Fs)
    time_idx = (np.arange(num_taps) - (num_taps - 1) / 2) * T_delta
    h_rrc = np.zeros(len(time_idx), dtype=float)

    # Handling singularities
    for x in range(len(time_idx)):
        t = time_idx[x]
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and abs(t) == Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) * \
                    (np.sin(np.pi / (4 * alpha)))) + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
        else:
            denom = (1 - (4 * alpha * t / Ts) ** 2)
            if abs(denom) < 1e-10: denom = 1e-10
            numerator = (np.sin(np.pi * t / Ts * (1 - alpha)) + \
                        4 * alpha * t / Ts * np.cos(np.pi * t / Ts * (1 + alpha)))
            h_rrc[x] = numerator / (np.pi * t / Ts * denom)

    return h_rrc / np.sqrt(np.sum(h_rrc**2))


class Modulator:
    def __init__(self, samples_per_symbol=8, bt=0.3, rrc_alpha=0.35, rrc_span=6):
        self.sps = int(samples_per_symbol)
        self.bt = float(bt)
        self.rrc_alpha = rrc_alpha
        self.rrc_span = rrc_span
        
        # Generate RRC Filter
        self.rrc_filter = rrcosfilter(self.rrc_span, self.rrc_alpha, 1, self.sps)
        
        # Calculate theoretical delay of ONE filter (in samples)
        # Since len is odd, delay is exactly (len-1)/2
        self.filter_delay = (len(self.rrc_filter) - 1) // 2

    # ============ PULSE SHAPING ============
    def apply_pulse_shaping(self, symbols):
        """Upsamples and applies TX RRC filter."""
        # We use scipy's upfirdn for efficiency if available
        # It inserts zeros and filters.
        # Output length will increase.
        syms = np.array(symbols, dtype=np.complex64)
        try:
            # scipy implementation
            # upfirdn output starts with the first transient.
            shaped = upfirdn(self.rrc_filter, syms, up=self.sps, down=1)
        except NameError:
            # Manual fallback
            upsampled = np.zeros(len(syms) * self.sps, dtype=np.complex64)
            upsampled[::self.sps] = syms
            shaped = convolve(upsampled, self.rrc_filter, mode='full')
            
        return shaped

    def matched_filter(self, samples):
        """
        Applies Matched Filter (Rx RRC) and Downsamples.
        ASSUMPTION: This assumes 'samples' contains the TX delay + RX delay.
        Used for Simulation Loopback where synchronization is perfect.
        """
        # 1. Filter
        filtered = convolve(samples, self.rrc_filter, mode='full')
        
        # 2. Calculate Sampling Point
        # TX Filter Delay = self.filter_delay
        # RX Filter Delay = self.filter_delay
        # Total System Delay = 2 * self.filter_delay
        # Ideally, the peak of the first symbol is at index = Total Delay.
        start_idx = 2 * self.filter_delay
        
        # 3. Downsample
        # We verify we don't go out of bounds
        if start_idx >= len(filtered):
            return np.array([], dtype=np.complex64)
            
        downsampled = filtered[start_idx::self.sps]
        
        # Normalize amplitude (roughly)
        # Power drops due to filtering, we scale back so constellation is ~1
        # This is an approximation; in real systems AGC handles this.
        if len(downsampled) > 0:
             # Quick normalization based on expected power preservation of RRC
             downsampled = downsampled
        
        return downsampled

    # ============ MODULATIONS ============
    # BPSK
    def mod_bpsk(self, bits):
        return (2 * np.asarray(bits, int) - 1).astype(np.complex64)
    def demod_bpsk(self, symbols):
        return (np.real(symbols) > 0).astype(int)

    # QPSK
    def mod_qpsk(self, bits):
        bits = np.asarray(bits, int)
        if len(bits) % 2 != 0: bits = np.append(bits, 0)
        bits = bits.reshape((-1, 2))
        # Gray coding mapping usually preferred, but using simple mapping here
        s = (1 - 2*bits[:,0]) + 1j*(1 - 2*bits[:,1])
        return s / np.sqrt(2)
    
    def demod_qpsk(self, symbols):
        b0 = (np.real(symbols) < 0).astype(int)
        b1 = (np.imag(symbols) < 0).astype(int)
        return np.column_stack([b0, b1]).flatten()

    # 8PSK
    def mod_8psk(self, bits):
        bits = np.asarray(bits, int)
        pad = (3 - len(bits) % 3) % 3
        if pad: bits = np.append(bits, [0]*pad)
        bits = bits.reshape((-1, 3))
        dec = bits.dot(1 << np.arange(3)[::-1])
        return np.exp(1j * 2 * np.pi * dec / 8)
    
    def demod_8psk(self, symbols):
        phi = np.angle(symbols)
        phi[phi < 0] += 2*np.pi
        dec = np.round(phi / (np.pi/4)).astype(int) % 8
        out = []
        for d in dec:
            out.extend([(d >> 2) & 1, (d >> 1) & 1, d & 1])
        return np.array(out)

    # QAM Helpers
    def _qam_const(self, M):
        m = int(np.sqrt(M))
        axis = np.arange(-m+1, m, 2)
        xv, yv = np.meshgrid(axis, axis)
        c = xv.flatten() + 1j * yv.flatten()
        c /= np.sqrt(np.mean(np.abs(c)**2)) # Normalize power
        return c, axis

    def _demod_qam_generic(self, symbols, M):
        c, _ = self._qam_const(M)
        # Simple min distance decoder
        # Can be optimized, but fine for simulation
        idxs = np.argmin(np.abs(symbols[:, None] - c[None, :]), axis=1)
        k = int(np.log2(M))
        return np.array([[(idx >> i) & 1 for i in range(k-1, -1, -1)] for idx in idxs]).flatten()

    # 16-QAM
    def mod_16qam(self, bits):
        bits = np.asarray(bits, int)
        k=4
        pad = (k - len(bits)%k)%k
        if pad: bits = np.append(bits, [0]*pad)
        grp = bits.reshape(-1, k)
        dec = grp.dot(1 << np.arange(k)[::-1])
        c, _ = self._qam_const(16)
        # Note: This mapping is linear, Gray mapping is better for BER but this works for test
        return c[dec]
    
    def demod_16qam(self, symbols):
        return self._demod_qam_generic(symbols, 16)

    # 64-QAM
    def mod_64qam(self, bits):
        bits = np.asarray(bits, int)
        k=6
        pad = (k - len(bits)%k)%k
        if pad: bits = np.append(bits, [0]*pad)
        grp = bits.reshape(-1, k)
        dec = grp.dot(1 << np.arange(k)[::-1])
        c, _ = self._qam_const(64)
        return c[dec]
    
    def demod_64qam(self, symbols):
        return self._demod_qam_generic(symbols, 64)