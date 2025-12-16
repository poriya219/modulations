import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from modulators import Modulator

def estimate_frequency_offset(samples, fs):
    """Coarse Frequency Offset Estimation using FFT"""
    # Use FFT to find the center of the signal energy
    N = len(samples)
    X = np.fft.fft(samples)
    freqs = np.fft.fftfreq(N, 1/fs)
    # Find peak magnitude
    idx = np.argmax(np.abs(X))
    peak_freq = freqs[idx]
    return peak_freq

def coarse_sync(rx_samples, sps):
    """
    Simple Coarse Frequency and Timing Recovery.
    Note: In a real blind receiver, this is much more complex (Costas Loop, Gardner).
    Here we use a simplified approach for static burst analysis.
    """
    # 1. Remove DC component
    rx_samples = rx_samples - np.mean(rx_samples)
    
    # 2. Power normalization
    rx_samples = rx_samples / np.sqrt(np.mean(np.abs(rx_samples)**2))

    # 3. Rough Frequency Correction (Fourth Power for QPSK/QAM)
    # Raising to 4th power reveals 4x frequency offset for QPSK
    # This is a classic trick for QPSK/QAM carrier recovery
    rx_4th = rx_samples**4
    fft_spec = np.fft.fft(rx_4th)
    freqs = np.fft.fftfreq(len(rx_4th), 1.0)
    peak_idx = np.argmax(np.abs(fft_spec))
    freq_est = freqs[peak_idx] / 4.0
    
    # Apply correction
    t = np.arange(len(rx_samples))
    rx_corrected = rx_samples * np.exp(-1j * 2 * np.pi * freq_est * t)
    
    print(f"  Estimated Frequency Offset: {freq_est * 2e6:.2f} Hz (Normalized: {freq_est:.5f})")
    
    return rx_corrected

def analyze_signal(filename, mod_type='qpsk'):
    print(f"Loading {filename}...")
    try:
        rx_raw = np.load(filename)
    except FileNotFoundError:
        print("File not found! Run test_hardware.py --mode rx first.")
        return

    mod = Modulator(samples_per_symbol=16, rrc_alpha=0.35)
    
    # 1. Limit samples to avoid slow processing (e.g., first 50k samples)
    # Usually valid data is in the middle if captured via manual start
    N_PROCESS = min(len(rx_raw), 100000)
    rx_data = rx_raw[10000:10000+N_PROCESS] # Skip start transient
    
    if len(rx_data) == 0:
        print("Error: Not enough data captured.")
        return

    print("Performing Coarse Synchronization...")
    rx_synced = coarse_sync(rx_data, mod.sps)
    
    # 2. Matched Filter (RRC)
    print("Applying Matched Filter...")
    rx_filtered = mod.matched_filter(rx_synced)
    
    if len(rx_filtered) == 0:
        print("Error: Matched filter output empty.")
        return

    # 3. Symbol Timing Recovery (Simple Peak Picking / Downsampling)
    # In non-blind test, we just try all SPS offsets and pick best constellation power
    best_var = float('inf')
    best_offset = 0
    
    # Try offsets 0 to SPS-1 to find optimal sampling point (Eye Diagram opening)
    for i in range(mod.sps):
        candidate = rx_filtered[i::mod.sps]
        # Minimize variance of amplitude (for PSK) or cluster variance
        # Simple metric: Variance of |sample|^2 - 1 (CMA - Constant Modulus)
        # Works well for PSK and QAM corner points
        metric = np.var(np.abs(candidate)**2 - 1) 
        if metric < best_var:
            best_var = metric
            best_offset = i
            
    rx_symbols = rx_filtered[best_offset::mod.sps]
    
    # 4. Fine Phase Rotation (Blind Phase Search)
    # We try rotating the constellation to lock it to axes
    best_phase = 0
    best_cost = float('inf')
    phases = np.linspace(0, 2*np.pi, 100)
    
    # Standard targets for QPSK (normalized)
    targets = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    
    for ph in phases:
        rotated = rx_symbols * np.exp(1j * ph)
        # Calculate distance to nearest valid symbol
        # Just take first 100 symbols for speed
        sample_subset = rotated[:200]
        dist = np.min(np.abs(sample_subset[:, None] - targets[None, :]), axis=1)
        cost = np.mean(dist)
        if cost < best_cost:
            best_cost = cost
            best_phase = ph
            
    rx_symbols = rx_symbols * np.exp(1j * best_phase)
    
    print(f"  Optimal Sampling Offset: {best_offset}")
    print(f"  Phase Correction: {np.degrees(best_phase):.2f} degrees")

    # --- PLOTTING ---
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Time Domain (Magnitude)
    plt.subplot(1, 2, 1)
    plt.plot(np.abs(rx_data[:500]), label='Raw Magnitude')
    plt.title("Received Signal Magnitude (First 500 samples)")
    plt.grid(True)
    
    # Plot 2: Constellation Diagram
    plt.subplot(1, 2, 2)
    # Plot all points
    plt.scatter(np.real(rx_symbols), np.imag(rx_symbols), alpha=0.3, s=2, c='blue', label='Rx Symbols')
    
    # Plot Reference Points (QPSK)
    if mod_type == 'qpsk':
        ref_pts = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        plt.scatter(np.real(ref_pts), np.imag(ref_pts), c='red', marker='x', s=100, label='Ideal')
    
    plt.title(f"Recovered Constellation ({mod_type.upper()})")
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    print("Displaying plots...")
    plt.show()
    
    # BER Calculation requires knowing the EXACT transmitted sequence.
    # In async hardware test without a specific header, calculating exact BER is hard.
    # But we can estimate EVM (Error Vector Magnitude).
    if mod_type == 'qpsk':
        # Find nearest ideal symbol
        targets = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        distances = np.min(np.abs(rx_symbols[:, None] - targets[None, :]), axis=1)
        evm_rms = np.sqrt(np.mean(distances**2))
        print(f"\n--- Performance Metrics ---")
        print(f"  EVM (RMS): {evm_rms:.4f}")
        print(f"  Estimated SNR: {10*np.log10(1/evm_rms**2):.2f} dB")
        if evm_rms < 0.4:
            print("  Result: EXCELLENT Reception")
        elif evm_rms < 0.7:
            print("  Result: GOOD/FAIR Reception (Noisy)")
        else:
            print("  Result: POOR Reception (Sync failed or high noise)")

if __name__ == "__main__":
    # Replace 'my_test_signal.npy' with your actual filename from previous step
    analyze_signal("my_record.npy", mod_type='qpsk')