#!/usr/bin/env python3
"""
SDR Modulation Test - V3 (Based on working V2 + small optimizations)
"""
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import time
from scipy.signal import correlate

# ============== CONFIG ==============
FC = 433e6
FS = 2e6
SPS = 4  # Keep SPS=4 which worked
TX_GAIN = 47
RX_GAIN = 49

# ============== MODULATION FUNCTIONS ==============
def bpsk_mod(bits):
    return 2.0 * np.array(bits, dtype=np.complex64) - 1.0

def bpsk_demod(syms):
    return (np.real(syms) > 0).astype(int)

def qpsk_mod(bits):
    bits = np.array(bits)
    if len(bits) % 2: bits = np.append(bits, 0)
    I = 1 - 2 * bits[0::2]
    Q = 1 - 2 * bits[1::2]
    return (I + 1j * Q).astype(np.complex64) / np.sqrt(2)

def qpsk_demod(syms):
    bits = np.zeros(len(syms) * 2, dtype=int)
    bits[0::2] = (np.real(syms) < 0).astype(int)
    bits[1::2] = (np.imag(syms) < 0).astype(int)
    return bits

def psk8_mod(bits):
    bits = np.array(bits)
    pad = (3 - len(bits) % 3) % 3
    if pad: bits = np.append(bits, [0] * pad)
    bits = bits.reshape(-1, 3)
    gray_map = [0, 1, 3, 2, 6, 7, 5, 4]
    symbols = []
    for b in bits:
        idx = b[0] * 4 + b[1] * 2 + b[2]
        phase = gray_map[idx] * np.pi / 4
        symbols.append(np.exp(1j * phase))
    return np.array(symbols, dtype=np.complex64)

def psk8_demod(syms):
    gray_map = [0, 1, 3, 2, 6, 7, 5, 4]
    inv_gray = [gray_map.index(i) for i in range(8)]
    bits = []
    for s in syms:
        phase = np.angle(s)
        if phase < 0: phase += 2 * np.pi
        idx = int(np.round(phase / (np.pi / 4))) % 8
        orig_idx = inv_gray[idx]
        bits.extend([(orig_idx >> 2) & 1, (orig_idx >> 1) & 1, orig_idx & 1])
    return np.array(bits)

def qam16_mod(bits):
    bits = np.array(bits)
    pad = (4 - len(bits) % 4) % 4
    if pad: bits = np.append(bits, [0] * pad)
    bits = bits.reshape(-1, 4)
    gray2 = [0, 1, 3, 2]
    symbols = []
    for b in bits:
        i_idx = b[0] * 2 + b[1]
        q_idx = b[2] * 2 + b[3]
        I = (2 * gray2[i_idx] - 3) / np.sqrt(10)
        Q = (2 * gray2[q_idx] - 3) / np.sqrt(10)
        symbols.append(I + 1j * Q)
    return np.array(symbols, dtype=np.complex64)

def qam16_demod(syms):
    gray2 = [0, 1, 3, 2]
    inv_gray2 = [gray2.index(i) for i in range(4)]
    bits = []
    for s in syms:
        I = np.real(s) * np.sqrt(10)
        Q = np.imag(s) * np.sqrt(10)
        i_idx = int(np.clip(np.round((I + 3) / 2), 0, 3))
        q_idx = int(np.clip(np.round((Q + 3) / 2), 0, 3))
        orig_i = inv_gray2[i_idx]
        orig_q = inv_gray2[q_idx]
        bits.extend([(orig_i >> 1) & 1, orig_i & 1, (orig_q >> 1) & 1, orig_q & 1])
    return np.array(bits)

def qam64_mod(bits):
    bits = np.array(bits)
    pad = (6 - len(bits) % 6) % 6
    if pad: bits = np.append(bits, [0] * pad)
    bits = bits.reshape(-1, 6)
    gray3 = [0, 1, 3, 2, 6, 7, 5, 4]
    symbols = []
    for b in bits:
        i_idx = b[0] * 4 + b[1] * 2 + b[2]
        q_idx = b[3] * 4 + b[4] * 2 + b[5]
        I = (2 * gray3[i_idx] - 7) / np.sqrt(42)
        Q = (2 * gray3[q_idx] - 7) / np.sqrt(42)
        symbols.append(I + 1j * Q)
    return np.array(symbols, dtype=np.complex64)

def qam64_demod(syms):
    gray3 = [0, 1, 3, 2, 6, 7, 5, 4]
    inv_gray3 = [gray3.index(i) for i in range(8)]
    bits = []
    for s in syms:
        I = np.real(s) * np.sqrt(42)
        Q = np.imag(s) * np.sqrt(42)
        i_idx = int(np.clip(np.round((I + 7) / 2), 0, 7))
        q_idx = int(np.clip(np.round((Q + 7) / 2), 0, 7))
        orig_i = inv_gray3[i_idx]
        orig_q = inv_gray3[q_idx]
        bits.extend([(orig_i >> 2) & 1, (orig_i >> 1) & 1, orig_i & 1,
                     (orig_q >> 2) & 1, (orig_q >> 1) & 1, orig_q & 1])
    return np.array(bits)

def qam256_mod(bits):
    bits = np.array(bits)
    pad = (8 - len(bits) % 8) % 8
    if pad: bits = np.append(bits, [0] * pad)
    bits = bits.reshape(-1, 8)
    gray4 = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]
    symbols = []
    for b in bits:
        i_idx = b[0] * 8 + b[1] * 4 + b[2] * 2 + b[3]
        q_idx = b[4] * 8 + b[5] * 4 + b[6] * 2 + b[7]
        I = (2 * gray4[i_idx] - 15) / np.sqrt(170)
        Q = (2 * gray4[q_idx] - 15) / np.sqrt(170)
        symbols.append(I + 1j * Q)
    return np.array(symbols, dtype=np.complex64)

def qam256_demod(syms):
    gray4 = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]
    inv_gray4 = [gray4.index(i) for i in range(16)]
    bits = []
    for s in syms:
        I = np.real(s) * np.sqrt(170)
        Q = np.imag(s) * np.sqrt(170)
        i_idx = int(np.clip(np.round((I + 15) / 2), 0, 15))
        q_idx = int(np.clip(np.round((Q + 15) / 2), 0, 15))
        orig_i = inv_gray4[i_idx]
        orig_q = inv_gray4[q_idx]
        bits.extend([(orig_i >> 3) & 1, (orig_i >> 2) & 1, (orig_i >> 1) & 1, orig_i & 1,
                     (orig_q >> 3) & 1, (orig_q >> 2) & 1, (orig_q >> 1) & 1, orig_q & 1])
    return np.array(bits)

# Optimized alpha values for each modulation
MODULATIONS = {
    'BPSK': {'mod': bpsk_mod, 'demod': bpsk_demod, 'bps': 1, 'order': 2, 'alpha': 0.02},
    'QPSK': {'mod': qpsk_mod, 'demod': qpsk_demod, 'bps': 2, 'order': 4, 'alpha': 0.015},
    '8PSK': {'mod': psk8_mod, 'demod': psk8_demod, 'bps': 3, 'order': 8, 'alpha': 0.01},
    '16QAM': {'mod': qam16_mod, 'demod': qam16_demod, 'bps': 4, 'order': 16, 'alpha': 0.008},
    '64QAM': {'mod': qam64_mod, 'demod': qam64_demod, 'bps': 6, 'order': 64, 'alpha': 0.005},
    '256QAM': {'mod': qam256_mod, 'demod': qam256_demod, 'bps': 8, 'order': 256, 'alpha': 0.003},
}

# ============== DSP ==============
def rrc_taps(sps, alpha=0.35, ntaps=101):
    n = np.arange(ntaps) - (ntaps - 1) / 2
    h = np.zeros(ntaps)
    for i, t in enumerate(n / sps):
        if t == 0:
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(t) - 1/(4*alpha)) < 1e-8:
            h[i] = alpha/np.sqrt(2) * ((1+2/np.pi)*np.sin(np.pi/4/alpha) + 
                                        (1-2/np.pi)*np.cos(np.pi/4/alpha))
        else:
            num = np.sin(np.pi*t*(1-alpha)) + 4*alpha*t*np.cos(np.pi*t*(1+alpha))
            den = np.pi*t*(1-(4*alpha*t)**2)
            h[i] = num / den if abs(den) > 1e-8 else 0
    return h / np.linalg.norm(h)

def upsample_filter(syms, sps, taps):
    up = np.zeros(len(syms) * sps, dtype=np.complex64)
    up[::sps] = syms
    return np.convolve(up, taps, mode='same')

def estimate_freq_offset_cw(sig, fs):
    """Original working frequency estimation"""
    N = min(len(sig), 2**16)
    sig = sig[:N]
    win = np.hanning(N)
    fft = np.fft.fft(sig * win)
    freqs = np.fft.fftfreq(N, 1/fs)
    mag = np.abs(fft)
    mag[:N//100] = 0
    mag[-N//100:] = 0
    return freqs[np.argmax(mag)]

def correct_freq(sig, f_off, fs):
    t = np.arange(len(sig)) / fs
    return sig * np.exp(-1j * 2 * np.pi * f_off * t)

def carrier_recovery_bpsk(syms, alpha=0.02):
    """BPSK carrier recovery with frequency tracking"""
    out = np.zeros_like(syms)
    phase = 0.0
    freq = 0.0
    beta = alpha * 0.1
    
    for i in range(len(syms)):
        out[i] = syms[i] * np.exp(-1j * phase)
        err = out[i].imag * np.sign(out[i].real)
        freq += beta * err
        phase += alpha * err + freq
    
    return out, phase, freq

def carrier_recovery_data(syms, init_phase, init_freq, mod_type, alpha=0.01):
    """Carrier recovery for data with frequency tracking"""
    out = np.zeros_like(syms)
    phase = init_phase
    freq = init_freq
    beta = alpha * 0.1
    
    for i in range(len(syms)):
        out[i] = syms[i] * np.exp(-1j * phase)
        
        if mod_type == 'BPSK':
            err = out[i].imag * np.sign(out[i].real)
        elif mod_type in ['QPSK', '16QAM', '64QAM', '256QAM']:
            err = np.sign(out[i].real) * out[i].imag - np.sign(out[i].imag) * out[i].real
        else:  # 8PSK
            phase_s = np.angle(out[i])
            nearest = np.round(phase_s / (np.pi/4)) * (np.pi/4)
            err = np.sin(phase_s - nearest)
        
        freq += beta * err
        phase += alpha * err + freq
    
    return out

# ============== FILE I/O ==============
def save_iq(sig, fname):
    sig = sig / (np.max(np.abs(sig)) + 1e-10) * 0.95
    iq = np.zeros(len(sig) * 2, dtype=np.int8)
    iq[0::2] = np.clip(np.real(sig) * 127, -127, 127).astype(np.int8)
    iq[1::2] = np.clip(np.imag(sig) * 127, -127, 127).astype(np.int8)
    iq.tofile(fname)

def load_iq(fname):
    raw = np.fromfile(fname, dtype=np.uint8)
    I = (raw[0::2].astype(np.float32) - 127.5) / 127.5
    Q = (raw[1::2].astype(np.float32) - 127.5) / 127.5
    return I + 1j * Q

# ============== MAIN TEST ==============
def run_test(mod_type):
    print(f"\n{'='*60}")
    print(f"Testing {mod_type}")
    print(f"{'='*60}")
    
    mod_cfg = MODULATIONS[mod_type]
    mod_func = mod_cfg['mod']
    demod_func = mod_cfg['demod']
    bps = mod_cfg['bps']
    order = mod_cfg['order']
    cr_alpha = mod_cfg['alpha']
    
    sps = SPS
    taps = rrc_taps(sps)
    
    # ===== CREATE SIGNAL =====
    preamble_len = int(0.01 * FS)
    preamble = np.ones(preamble_len, dtype=np.complex64)
    
    # Longer sync for higher-order modulations
    sync_len = 240 if order >= 64 else 160
    sync_bits = np.array([1,0,1,0,1,0,1,0,1,1,0,0,1,1,0,0] * (sync_len // 16))
    sync_syms = bpsk_mod(sync_bits)
    
    # More data bits for higher-order modulations
    n_data_bits = 480 if order >= 64 else 240
    data_bits = np.random.randint(0, 2, n_data_bits)
    data_syms = mod_func(data_bits)
    
    # Frame
    frame_syms = np.concatenate([sync_syms, data_syms])
    frame_wave = upsample_filter(frame_syms, sps, taps)
    
    full_sig = np.concatenate([preamble, frame_wave])
    pad = np.zeros(int(0.02 * FS), dtype=np.complex64)
    burst = np.concatenate([pad, full_sig, pad])
    tx_signal = np.tile(burst, 20)
    
    print(f"TX: {len(tx_signal)} samples, Sync: {len(sync_syms)} sym, Data: {len(data_syms)} sym")
    
    # ===== HARDWARE TX/RX =====
    tx_file = '/tmp/tx.iq'
    rx_file = '/tmp/rx.iq'
    
    for f in [tx_file, rx_file]:
        if os.path.exists(f): os.remove(f)
    
    save_iq(tx_signal, tx_file)
    
    rx_proc = subprocess.Popen(
        ['rtl_sdr', '-f', str(int(FC)), '-s', str(int(FS)), '-g', str(RX_GAIN), rx_file],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(1.5)
    
    tx_proc = subprocess.Popen(
        ['hackrf_transfer', '-t', tx_file, '-f', str(int(FC)), '-s', str(int(FS)),
         '-x', str(TX_GAIN), '-a', '1', '-R'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    time.sleep(5)
    
    tx_proc.terminate(); tx_proc.wait()
    time.sleep(0.3)
    rx_proc.terminate(); rx_proc.wait()
    time.sleep(0.3)
    
    # ===== LOAD RX DATA =====
    if not os.path.exists(rx_file) or os.path.getsize(rx_file) < 10000:
        print("ERROR: No RX data!")
        return {'mod': mod_type, 'ber': 1.0, 'sync_ber': 1.0, 'bps': bps}
    
    rx_raw = load_iq(rx_file)
    print(f"RX: {len(rx_raw)} samples")
    
    for f in [tx_file, rx_file]:
        if os.path.exists(f): os.remove(f)
    
    # ===== RECEIVER =====
    rx = rx_raw - np.mean(rx_raw)
    
    # Find burst
    power = np.abs(rx) ** 2
    power_smooth = np.convolve(power, np.ones(1000)/1000, mode='same')
    threshold = np.max(power_smooth) * 0.3
    signal_present = power_smooth > threshold
    
    burst_start = None
    for i in range(len(signal_present) - preamble_len):
        if signal_present[i:i+preamble_len//2].all():
            burst_start = i
            break
    
    if burst_start is None:
        print("ERROR: Could not find burst!")
        return {'mod': mod_type, 'ber': 1.0, 'sync_ber': 1.0, 'bps': bps}
    
    print(f"Burst at {burst_start}")
    
    # Freq offset
    preamble_rx = rx[burst_start:burst_start + preamble_len]
    f_offset = estimate_freq_offset_cw(preamble_rx, FS)
    print(f"Freq offset: {f_offset:.1f} Hz")
    
    rx_corrected = correct_freq(rx, f_offset, FS)
    rx_filt = np.convolve(rx_corrected, taps, mode='same')
    
    # Find sync
    sync_wave = upsample_filter(sync_syms, sps, taps)
    search_start = burst_start + preamble_len - 100
    search_end = min(len(rx_filt), search_start + len(burst) * 2)
    search_region = rx_filt[search_start:search_end]
    
    corr = np.abs(correlate(search_region, sync_wave, mode='valid'))
    peak_idx = np.argmax(corr)
    sync_start = search_start + peak_idx
    
    print(f"Sync peak: {corr[peak_idx]:.1f} at {sync_start}")
    
    # Extract frame
    total_syms = len(sync_syms) + len(data_syms)
    frame_samples = total_syms * sps + sps * 4
    
    if sync_start + frame_samples > len(rx_filt):
        print("ERROR: Frame past buffer!")
        return {'mod': mod_type, 'ber': 1.0, 'sync_ber': 1.0, 'bps': bps}
    
    rx_frame = rx_filt[sync_start:sync_start + frame_samples]
    
    # Find best timing
    best_ber = 1.0
    best_data_rx = None
    best_offset = 0
    best_sync_ber = 1.0
    
    for offset in range(sps):
        syms = rx_frame[offset::sps]
        if len(syms) < total_syms:
            continue
        
        sync_rx_raw = syms[:len(sync_syms)]
        data_rx_raw = syms[len(sync_syms):len(sync_syms) + len(data_syms)]
        
        # BPSK carrier recovery with frequency tracking
        sync_rec, final_phase, final_freq = carrier_recovery_bpsk(sync_rx_raw, alpha=0.03)
        
        # Normalize sync
        pwr = np.mean(np.abs(sync_rec) ** 2)
        if pwr > 0:
            sync_rec = sync_rec / np.sqrt(pwr)
        
        for sync_rot in range(2):
            sync_phase = np.exp(1j * sync_rot * np.pi)
            sync_test = sync_rec * sync_phase
            
            sync_bits_rx = bpsk_demod(sync_test)
            sync_ber = np.mean(sync_bits_rx != sync_bits)
            
            if sync_ber > 0.2:
                continue
            
            # Data carrier recovery with tracked frequency
            adjusted_phase = final_phase + sync_rot * np.pi
            data_rec = carrier_recovery_data(data_rx_raw, adjusted_phase, final_freq, mod_type, alpha=cr_alpha)
            
            # Normalize data
            pwr = np.mean(np.abs(data_rec) ** 2)
            if pwr > 0:
                data_rec = data_rec / np.sqrt(pwr)
            
            # Phase rotations
            if mod_type == 'BPSK':
                n_rot, rot_step = 2, np.pi
            elif mod_type == '8PSK':
                n_rot, rot_step = 8, np.pi / 4
            else:
                n_rot, rot_step = 4, np.pi / 2
            
            for rot in range(n_rot):
                data_test = data_rec * np.exp(1j * rot * rot_step)
                data_bits_rx = demod_func(data_test)
                L = min(len(data_bits_rx), len(data_bits))
                data_ber = np.mean(data_bits_rx[:L] != data_bits[:L])
                
                if data_ber < best_ber:
                    best_ber = data_ber
                    best_data_rx = data_test.copy()
                    best_offset = offset
                    best_sync_ber = sync_ber
    
    print(f"Best: offset={best_offset}, sync_ber={best_sync_ber:.4f}")
    
    if best_data_rx is None:
        print("ERROR: Could not decode!")
        return {'mod': mod_type, 'ber': 1.0, 'sync_ber': 1.0, 'bps': bps}
    
    data_bits_rx = demod_func(best_data_rx)
    L = min(len(data_bits), len(data_bits_rx))
    errors = np.sum(data_bits[:L] != data_bits_rx[:L])
    ber = errors / L
    
    print(f"Data BER: {ber:.5f} ({errors}/{L})")
    
    # ===== PLOT =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{mod_type} - BER: {ber:.4f} ({ber*100:.2f}%)', fontsize=14)
    
    ax = axes[0, 0]
    ax.scatter(data_syms.real, data_syms.imag, s=50, c='blue', alpha=0.7)
    ax.set_title(f'TX Constellation ({order} points)')
    ax.grid(True)
    lim = 1.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    
    ax = axes[0, 1]
    ax.scatter(best_data_rx.real, best_data_rx.imag, s=10, c='red', alpha=0.5)
    ax.set_title('RX Constellation')
    ax.grid(True)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    
    ax = axes[1, 0]
    L_show = min(50, len(data_bits))
    ax.stem(range(L_show), data_bits[:L_show], 'b-', markerfmt='bo', basefmt=' ', label='TX')
    ax.stem(range(L_show), data_bits_rx[:L_show] + 0.1, 'r-', markerfmt='rx', basefmt=' ', label='RX')
    ax.set_title('Bit Comparison')
    ax.legend()
    ax.set_ylim(-0.5, 1.5)
    
    ax = axes[1, 1]
    ax.axis('off')
    status = "PASS" if ber < 0.1 else "MARGINAL" if ber < 0.2 else "FAIL"
    ax.text(0.1, 0.5, 
            f"Modulation: {mod_type}\n"
            f"Bits/Symbol: {bps}\n"
            f"Order: {order}\n\n"
            f"Data BER: {ber:.5f}\n"
            f"Sync BER: {best_sync_ber:.4f}\n\n"
            f"Status: {status}",
            fontsize=14, transform=ax.transAxes, va='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'result_{mod_type}.png', dpi=150)
    plt.close()
    print(f"Saved result_{mod_type}.png")
    
    return {'mod': mod_type, 'ber': ber, 'sync_ber': best_sync_ber, 'bps': bps}

def main():
    print("\n" + "="*60)
    print("SDR MODULATION TEST - V3")
    print("="*60)
    print(f"Frequency: {FC/1e6:.1f} MHz")
    print(f"Sample rate: {FS/1e6:.1f} MSps")
    print(f"Samples per symbol: {SPS}")
    
    results = []
    for mod in ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '256QAM']:
        r = run_test(mod)
        results.append(r)
        time.sleep(3)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Modulation':<12} {'BPS':<5} {'BER':<12} {'Status':<10}")
    print("-" * 40)
    for r in results:
        status = "PASS" if r['ber'] < 0.1 else "MARGINAL" if r['ber'] < 0.2 else "FAIL"
        print(f"{r['mod']:<12} {r['bps']:<5} {r['ber']:<12.5f} {status:<10}")
    print("="*60)

if __name__ == "__main__":
    main()