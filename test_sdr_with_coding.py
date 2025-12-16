#!/usr/bin/env python3
"""
SDR Modulation Test with DVB-RCS2 Turbo Coding - Optimized for 16QAM
"""
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import time
from scipy.signal import correlate
from dvb_rcs2_turbo import DVB_RCS2_TurboCodec

# ============== CONFIG ==============
FC = 433e6
FS = 2e6
SPS = 4
TX_GAIN = 47
RX_GAIN = 49

TURBO_BLOCK_LENGTH = 212
TURBO_CODE_RATE = '1/2'
TURBO_ITERATIONS = 10

# ============== MODULATION ==============
def bpsk_mod(bits):
    return 2.0 * np.array(bits, dtype=np.complex64) - 1.0

def bpsk_demod(syms):
    return (np.real(syms) > 0).astype(int)

def qpsk_mod(bits):
    bits = np.array(bits)
    if len(bits) % 2:
        bits = np.append(bits, 0)
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
    if pad:
        bits = np.append(bits, [0] * pad)
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
        if phase < 0:
            phase += 2 * np.pi
        idx = int(np.round(phase / (np.pi / 4))) % 8
        orig_idx = inv_gray[idx]
        bits.extend([(orig_idx >> 2) & 1, (orig_idx >> 1) & 1, orig_idx & 1])
    return np.array(bits)

def qam16_mod(bits):
    bits = np.array(bits)
    pad = (4 - len(bits) % 4) % 4
    if pad:
        bits = np.append(bits, [0] * pad)
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

MODULATIONS = {
    'BPSK': {'mod': bpsk_mod, 'demod': bpsk_demod, 'bps': 1, 'order': 2, 'alpha': 0.02},
    'QPSK': {'mod': qpsk_mod, 'demod': qpsk_demod, 'bps': 2, 'order': 4, 'alpha': 0.015},
    '8PSK': {'mod': psk8_mod, 'demod': psk8_demod, 'bps': 3, 'order': 8, 'alpha': 0.012},
    '16QAM': {'mod': qam16_mod, 'demod': qam16_demod, 'bps': 4, 'order': 16, 'alpha': 0.006},
}

# ============== DSP ==============
def rrc_taps(sps, alpha=0.35, ntaps=101):
    n = np.arange(ntaps) - (ntaps - 1) / 2
    h = np.zeros(ntaps)
    for i, t in enumerate(n / sps):
        if t == 0:
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(t) - 1 / (4 * alpha)) < 1e-8:
            h[i] = alpha / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / 4 / alpha) +
                                          (1 - 2 / np.pi) * np.cos(np.pi / 4 / alpha))
        else:
            num = np.sin(np.pi * t * (1 - alpha)) + 4 * alpha * t * np.cos(np.pi * t * (1 + alpha))
            den = np.pi * t * (1 - (4 * alpha * t) ** 2)
            h[i] = num / den if abs(den) > 1e-8 else 0
    return h / np.linalg.norm(h)

def upsample_filter(syms, sps, taps):
    up = np.zeros(len(syms) * sps, dtype=np.complex64)
    up[::sps] = syms
    return np.convolve(up, taps, mode='same')

def estimate_freq_offset_cw(sig, fs):
    N = min(len(sig), 2 ** 16)
    sig = sig[:N]
    win = np.hanning(N)
    fft = np.fft.fft(sig * win)
    freqs = np.fft.fftfreq(N, 1 / fs)
    mag = np.abs(fft)
    mag[:N // 100] = 0
    mag[-N // 100:] = 0
    return freqs[np.argmax(mag)]

def correct_freq(sig, f_off, fs):
    t = np.arange(len(sig)) / fs
    return sig * np.exp(-1j * 2 * np.pi * f_off * t)

def carrier_recovery_bpsk(syms, alpha=0.02):
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

def carrier_recovery_qam(syms, init_phase, init_freq, alpha=0.005):
    """Optimized carrier recovery for QAM"""
    out = np.zeros_like(syms)
    phase = init_phase
    freq = init_freq
    beta = alpha * 0.05
    
    for i in range(len(syms)):
        out[i] = syms[i] * np.exp(-1j * phase)
        
        # Decision-directed error for QAM
        re = np.real(out[i])
        im = np.imag(out[i])
        
        # Error using sign of real/imag parts
        err = np.sign(re) * im - np.sign(im) * re
        
        freq += beta * err
        freq = np.clip(freq, -0.01, 0.01)
        phase += alpha * err + freq
    
    return out

def carrier_recovery_data(syms, init_phase, init_freq, mod_type, alpha=0.01):
    out = np.zeros_like(syms)
    phase = init_phase
    freq = init_freq
    beta = alpha * 0.1
    for i in range(len(syms)):
        out[i] = syms[i] * np.exp(-1j * phase)
        if mod_type == 'BPSK':
            err = out[i].imag * np.sign(out[i].real)
        elif mod_type in ['QPSK', '16QAM']:
            err = np.sign(out[i].real) * out[i].imag - np.sign(out[i].imag) * out[i].real
        else:
            phase_s = np.angle(out[i])
            nearest = np.round(phase_s / (np.pi / 4)) * (np.pi / 4)
            err = np.sin(phase_s - nearest)
        freq += beta * err
        phase += alpha * err + freq
    return out

# ============== SOFT DEMODULATION ==============
def compute_llr(syms, mod_type, noise_var):
    """Compute LLR using max-log approximation"""
    noise_var = max(noise_var, 0.005)
    bps = MODULATIONS[mod_type]['bps']
    mod_func = MODULATIONS[mod_type]['mod']
    order = MODULATIONS[mod_type]['order']
    
    all_bits = np.array([list(map(int, format(i, f'0{bps}b'))) for i in range(order)])
    constellation = mod_func(all_bits.flatten()).reshape(-1)
    
    n_syms = len(syms)
    llr = np.zeros(n_syms * bps)
    
    for i, s in enumerate(syms):
        distances = np.abs(s - constellation) ** 2
        
        for b in range(bps):
            idx0 = np.where(all_bits[:, b] == 0)[0]
            idx1 = np.where(all_bits[:, b] == 1)[0]
            
            min_d0 = np.min(distances[idx0])
            min_d1 = np.min(distances[idx1])
            
            llr[i * bps + b] = (min_d0 - min_d1) / noise_var
    
    return np.clip(llr, -30, 30)

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
def run_test(mod_type, use_coding=True):
    print(f"\n{'=' * 60}")
    print(f"Testing {mod_type} {'WITH' if use_coding else 'WITHOUT'} Turbo Coding")
    print(f"{'=' * 60}")
    
    mod_cfg = MODULATIONS[mod_type]
    mod_func = mod_cfg['mod']
    demod_func = mod_cfg['demod']
    bps = mod_cfg['bps']
    order = mod_cfg['order']
    cr_alpha = mod_cfg['alpha']
    
    sps = SPS
    taps = rrc_taps(sps)
    
    if use_coding:
        codec = DVB_RCS2_TurboCodec(
            block_length=TURBO_BLOCK_LENGTH,
            code_rate=TURBO_CODE_RATE,
            n_iterations=TURBO_ITERATIONS
        )
        n_info_bits = codec.k_info
        print(f"Turbo: {codec.k_info} info -> ~{codec.n_coded} coded (rate {TURBO_CODE_RATE})")
    else:
        codec = None
        n_info_bits = 424
    
    preamble_len = int(0.01 * FS)
    preamble = np.ones(preamble_len, dtype=np.complex64)
    
    # Longer sync for 16QAM
    sync_len = 320 if order >= 16 else 160
    sync_bits = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0] * (sync_len // 16))
    sync_syms = bpsk_mod(sync_bits)
    
    np.random.seed(42)
    info_bits = np.random.randint(0, 2, n_info_bits)
    
    if use_coding:
        coded_bits = codec.encode(info_bits)
        tx_data_bits = coded_bits
        print(f"Info: {len(info_bits)}, Coded: {len(coded_bits)}")
    else:
        tx_data_bits = info_bits
        print(f"Data: {len(tx_data_bits)} (uncoded)")
    
    data_syms = mod_func(tx_data_bits)
    
    frame_syms = np.concatenate([sync_syms, data_syms])
    frame_wave = upsample_filter(frame_syms, sps, taps)
    
    full_sig = np.concatenate([preamble, frame_wave])
    pad = np.zeros(int(0.02 * FS), dtype=np.complex64)
    burst = np.concatenate([pad, full_sig, pad])
    tx_signal = np.tile(burst, 25)
    
    print(f"TX: {len(tx_signal)} samples, Sync: {len(sync_syms)}, Data: {len(data_syms)} sym")
    
    tx_file = '/tmp/tx.iq'
    rx_file = '/tmp/rx.iq'
    
    for f in [tx_file, rx_file]:
        if os.path.exists(f):
            os.remove(f)
    
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
    
    time.sleep(6)
    
    tx_proc.terminate()
    tx_proc.wait()
    time.sleep(0.3)
    rx_proc.terminate()
    rx_proc.wait()
    time.sleep(0.3)
    
    if not os.path.exists(rx_file) or os.path.getsize(rx_file) < 10000:
        print("ERROR: No RX data!")
        return {'mod': mod_type, 'coded': use_coding, 'ber_coded': 1.0, 'ber_info': 1.0, 'bps': bps}
    
    rx_raw = load_iq(rx_file)
    print(f"RX: {len(rx_raw)} samples")
    
    for f in [tx_file, rx_file]:
        if os.path.exists(f):
            os.remove(f)
    
    rx = rx_raw - np.mean(rx_raw)
    
    power = np.abs(rx) ** 2
    power_smooth = np.convolve(power, np.ones(1000) / 1000, mode='same')
    threshold = np.max(power_smooth) * 0.3
    signal_present = power_smooth > threshold
    
    burst_start = None
    for i in range(len(signal_present) - preamble_len):
        if signal_present[i:i + preamble_len // 2].all():
            burst_start = i
            break
    
    if burst_start is None:
        print("ERROR: No burst!")
        return {'mod': mod_type, 'coded': use_coding, 'ber_coded': 1.0, 'ber_info': 1.0, 'bps': bps}
    
    print(f"Burst at {burst_start}")
    
    preamble_rx = rx[burst_start:burst_start + preamble_len]
    f_offset = estimate_freq_offset_cw(preamble_rx, FS)
    print(f"Freq offset: {f_offset:.1f} Hz")
    
    rx_corrected = correct_freq(rx, f_offset, FS)
    rx_filt = np.convolve(rx_corrected, taps, mode='same')
    
    sync_wave = upsample_filter(sync_syms, sps, taps)
    search_start = burst_start + preamble_len - 100
    search_end = min(len(rx_filt), search_start + len(burst) * 2)
    search_region = rx_filt[search_start:search_end]
    
    corr = np.abs(correlate(search_region, sync_wave, mode='valid'))
    peak_idx = np.argmax(corr)
    sync_start = search_start + peak_idx
    
    print(f"Sync peak: {corr[peak_idx]:.1f}")
    
    total_syms = len(sync_syms) + len(data_syms)
    frame_samples = total_syms * sps + sps * 4
    
    if sync_start + frame_samples > len(rx_filt):
        print("ERROR: Frame past buffer!")
        return {'mod': mod_type, 'coded': use_coding, 'ber_coded': 1.0, 'ber_info': 1.0, 'bps': bps}
    
    rx_frame = rx_filt[sync_start:sync_start + frame_samples]
    
    best_ber = 1.0
    best_data_rx = None
    best_offset = 0
    
    for offset in range(sps):
        syms = rx_frame[offset::sps]
        if len(syms) < total_syms:
            continue
        
        sync_rx_raw = syms[:len(sync_syms)]
        data_rx_raw = syms[len(sync_syms):len(sync_syms) + len(data_syms)]
        
        sync_rec, final_phase, final_freq = carrier_recovery_bpsk(sync_rx_raw, alpha=0.04)
        
        pwr = np.mean(np.abs(sync_rec) ** 2)
        if pwr > 0:
            sync_rec = sync_rec / np.sqrt(pwr)
        
        for sync_rot in range(2):
            sync_phase = np.exp(1j * sync_rot * np.pi)
            sync_test = sync_rec * sync_phase
            
            sync_bits_rx = bpsk_demod(sync_test)
            sync_ber = np.mean(sync_bits_rx != sync_bits)
            
            if sync_ber > 0.15:
                continue
            
            adjusted_phase = final_phase + sync_rot * np.pi
            
            # Use optimized carrier recovery for 16QAM
            if mod_type == '16QAM':
                data_rec = carrier_recovery_qam(data_rx_raw, adjusted_phase, final_freq, alpha=cr_alpha)
            else:
                data_rec = carrier_recovery_data(data_rx_raw, adjusted_phase, final_freq, mod_type, alpha=cr_alpha)
            
            pwr = np.mean(np.abs(data_rec) ** 2)
            if pwr > 0:
                data_rec = data_rec / np.sqrt(pwr)
            
            if mod_type == 'BPSK':
                n_rot, rot_step = 2, np.pi
            elif mod_type == '8PSK':
                n_rot, rot_step = 8, np.pi / 4
            else:
                n_rot, rot_step = 4, np.pi / 2
            
            for rot in range(n_rot):
                data_test = data_rec * np.exp(1j * rot * rot_step)
                data_bits_rx = demod_func(data_test)
                L = min(len(data_bits_rx), len(tx_data_bits))
                data_ber = np.mean(data_bits_rx[:L] != tx_data_bits[:L])
                
                if data_ber < best_ber:
                    best_ber = data_ber
                    best_data_rx = data_test.copy()
                    best_offset = offset
    
    print(f"Best offset: {best_offset}")
    
    if best_data_rx is None:
        print("ERROR: Could not decode!")
        return {'mod': mod_type, 'coded': use_coding, 'ber_coded': 1.0, 'ber_info': 1.0, 'bps': bps}
    
    rx_coded_bits = demod_func(best_data_rx)
    L = min(len(tx_data_bits), len(rx_coded_bits))
    coded_errors = np.sum(tx_data_bits[:L] != rx_coded_bits[:L])
    coded_ber = coded_errors / L
    
    print(f"Coded/Raw BER: {coded_ber:.5f} ({coded_errors}/{L})")
    
    if use_coding and codec is not None:
        # Better noise variance estimation using constellation distance
        const = mod_func(rx_coded_bits[:len(best_data_rx) * bps])
        if len(const) <= len(best_data_rx):
            noise_var = np.mean(np.abs(best_data_rx[:len(const)] - const) ** 2)
        else:
            noise_var = 0.05
        
        # Scale noise_var appropriately for LLR computation
        noise_var = max(noise_var, 0.02)
        print(f"Estimated noise var: {noise_var:.4f}")
        
        # Compute LLR
        llr = compute_llr(best_data_rx, mod_type, noise_var)
        
        # Ensure correct length
        expected_len = codec.n_coded
        if len(llr) < expected_len:
            llr = np.pad(llr, (0, expected_len - len(llr)))
        else:
            llr = llr[:expected_len]
        
        try:
            decoded_bits = codec.decode(llr)
            
            info_L = min(len(info_bits), len(decoded_bits))
            info_errors = np.sum(info_bits[:info_L] != decoded_bits[:info_L])
            info_ber = info_errors / info_L
            
            print(f"Info BER (decoded): {info_ber:.5f} ({info_errors}/{info_L})")
        except Exception as e:
            print(f"Decoding error: {e}")
            info_ber = coded_ber
            decoded_bits = rx_coded_bits[:n_info_bits]
    else:
        info_ber = coded_ber
        decoded_bits = rx_coded_bits[:n_info_bits]
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    coding_str = "Turbo" if use_coding else "Uncoded"
    fig.suptitle(f'{mod_type} {coding_str} - BER: {info_ber:.4f}', fontsize=14)
    
    ax = axes[0, 0]
    ax.scatter(data_syms.real, data_syms.imag, s=50, c='blue', alpha=0.7)
    ax.set_title(f'TX ({order}-point)')
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    
    ax = axes[0, 1]
    ax.scatter(best_data_rx.real, best_data_rx.imag, s=10, c='red', alpha=0.5)
    ax.set_title('RX')
    ax.grid(True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    
    ax = axes[0, 2]
    L_show = min(50, len(info_bits), len(decoded_bits))
    ax.stem(range(L_show), info_bits[:L_show], 'b-', markerfmt='bo', basefmt=' ', label='TX')
    ax.stem(range(L_show), decoded_bits[:L_show] + 0.1, 'r-', markerfmt='rx', basefmt=' ', label='RX')
    ax.set_title('Info Bits')
    ax.legend()
    ax.set_ylim(-0.5, 1.5)
    
    ax = axes[1, 0]
    L_show = min(50, len(tx_data_bits), len(rx_coded_bits))
    ax.stem(range(L_show), tx_data_bits[:L_show], 'b-', markerfmt='bo', basefmt=' ')
    ax.stem(range(L_show), rx_coded_bits[:L_show] + 0.1, 'r-', markerfmt='rx', basefmt=' ')
    ax.set_title('Coded Bits')
    ax.set_ylim(-0.5, 1.5)
    
    ax = axes[1, 1]
    info_L = min(len(info_bits), len(decoded_bits))
    errs = (info_bits[:info_L] != decoded_bits[:info_L]).astype(int)
    ax.bar(range(len(errs)), errs, color='red', alpha=0.7)
    ax.set_title(f'Errors ({np.sum(errs)})')
    ax.set_ylim(0, 1.5)
    
    ax = axes[1, 2]
    ax.axis('off')
    status = "PASS" if info_ber < 0.1 else "FAIL"
    gain = 10 * np.log10(coded_ber / info_ber) if coded_ber > 0 and info_ber > 0 and info_ber < coded_ber else 0
    ax.text(0.1, 0.9, f"Mod: {mod_type}\nCoding: {coding_str}\nCoded BER: {coded_ber:.5f}\n"
                      f"Info BER: {info_ber:.5f}\nGain: {gain:.1f} dB\nStatus: {status}",
            transform=ax.transAxes, fontsize=12, va='top', fontfamily='monospace',
            bbox=dict(facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'result_{mod_type}_{"coded" if use_coding else "uncoded"}.png', dpi=150)
    plt.close()
    print(f"Saved plot")
    
    return {'mod': mod_type, 'coded': use_coding, 'ber_coded': coded_ber, 'ber_info': info_ber, 'bps': bps}


def main():
    print("\n" + "=" * 70)
    print("SDR TEST WITH DVB-RCS2 TURBO CODING")
    print("=" * 70)
    print(f"Frequency: {FC / 1e6:.1f} MHz, Sample rate: {FS / 1e6:.1f} MSps")
    print(f"Turbo Code Rate: {TURBO_CODE_RATE}, Iterations: {TURBO_ITERATIONS}")
    
    results = []
    
    # Only test BPSK, QPSK, 8PSK, 16QAM
    for mod in ['BPSK', 'QPSK', '8PSK', '16QAM']:
        r_uncoded = run_test(mod, use_coding=False)
        results.append(r_uncoded)
        time.sleep(3)
        
        r_coded = run_test(mod, use_coding=True)
        results.append(r_coded)
        time.sleep(3)
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Mod':<10} {'Coding':<8} {'Coded BER':<12} {'Info BER':<12} {'Gain':<8} {'Status':<8}")
    print("-" * 60)
    
    for i in range(0, len(results), 2):
        u, c = results[i], results[i + 1]
        
        status_u = "PASS" if u['ber_info'] < 0.1 else "FAIL"
        print(f"{u['mod']:<10} {'None':<8} {u['ber_coded']:<12.5f} {u['ber_info']:<12.5f} {'-':<8} {status_u:<8}")
        
        if c['ber_coded'] > 0 and c['ber_info'] > 0 and c['ber_info'] < c['ber_coded']:
            gain = 10 * np.log10(c['ber_coded'] / c['ber_info'])
        else:
            gain = 0
        status_c = "PASS" if c['ber_info'] < 0.1 else "FAIL"
        print(f"{c['mod']:<10} {'Turbo':<8} {c['ber_coded']:<12.5f} {c['ber_info']:<12.5f} {gain:<8.1f} {status_c:<8}")
        print("-" * 60)
    
    print("=" * 70)


if __name__ == "__main__":
    main()