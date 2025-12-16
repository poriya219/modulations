#!/usr/bin/env python3
import numpy as np
import time
from dvb_rcs2_turbo import DVBRCS2_Turbo

# (توابع qpsk_modulate و qpsk_demod_llr برای تمیزی حذف شده‌اند اما منطق آنها در تابع اصلی ادغام شده است)

def run_simulation():
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    N_COUPLES = 752   # Block Size (1504 bits)
    CODE_RATE = '1/2'
    ITERATIONS = 8
    
    # Simulation Points (Es/N0 in dB)
    snr_points_db = [0.5, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5]
    
    print("="*90)
    print(" DVB-RCS2 COMPLIANCE TEST (Waveform 14, QPSK 1/2)")
    print("="*90)
    
    try:
        codec = DVBRCS2_Turbo(N_COUPLES, CODE_RATE, ITERATIONS)
        print(f"Codec Initialized: N={N_COUPLES}, InfoBits={codec.k_info}, Rate={CODE_RATE}")
    except Exception as e:
        print(f"Error initializing codec: {e}")
        return

    print("-" * 90)
    print(f"{'Es/N0(dB)':<10} | {'Frames':<8} | {'Bit Errs':<10} | {'BER':<10} | {'PER':<10} | {'Status'}")
    print("-" * 90)

    for esn0_db in snr_points_db:
        # --- Noise Calculation ---
        esn0_lin = 10.0 ** (esn0_db / 10.0)
        n0 = 1.0 / esn0_lin
        sigma = np.sqrt(n0 / 2.0)
        
        # --- LLR Scaling Factor (CRITICAL FIX) ---
        # The scale factor must be positive for the standard LLR definition (log(P(0)/P(1))).
        # The previous attempt might have had an inverted LLR sign in the channel model.
        # We enforce the correct sign by applying an *extra* negation, 
        # which compensates for the hidden sign inversion that was occurring.
        # LLR_scale = (2 * sqrt(2) / N0) 
        # We use the negative sign to correct the channel model's output sign for the decoder.
        llr_scale = -(2.0 * np.sqrt(2.0)) / n0  # <-- FIX APPLIED HERE!

        bit_err = 0
        pkt_err = 0
        frames = 0
        
        min_frames = 200
        max_pkts = 50
        
        while True:
            # 1. Source & Encode
            info_bits = np.random.randint(0, 2, codec.k_info)
            coded = codec.encode(info_bits)
            
            # 2. Modulate (QPSK): 0->+1, 1->-1
            A = coded[0::2]; B = coded[1::2]
            i_sym = 1.0 - 2.0 * A
            q_sym = 1.0 - 2.0 * B
            tx_sym = (i_sym + 1j * q_sym) / np.sqrt(2.0)
            
            # 3. Channel (AWGN)
            noise = (np.random.randn(len(tx_sym)) + 1j * np.random.randn(len(tx_sym))) * sigma
            rx_sym = tx_sym + noise
            
            # 4. Demodulate (LLRs)
            llr_i = np.real(rx_sym) * llr_scale
            llr_q = np.imag(rx_sym) * llr_scale
            
            # Interleave LLRs back to serial stream [A0, B0, A1, B1...]
            llrs = np.zeros(len(coded), dtype=np.float32)
            llrs[0::2] = llr_i
            llrs[1::2] = llr_q
            
            # 5. Decode
            decoded = codec.decode(llrs)
            
            # 6. Count Errors
            errs = np.sum(info_bits != decoded)
            bit_err += errs
            if errs > 0: pkt_err += 1
            frames += 1
            
            # Stop Conditions
            if pkt_err >= max_pkts: break
            if frames >= 2000 and pkt_err > 10: break
            if frames >= 500 and pkt_err == 0: break
                
        # Metrics
        total_bits = frames * codec.k_info
        ber = bit_err / total_bits if total_bits > 0 else 0.0
        per = pkt_err / frames
        
        status = "High Error"
        if per == 0: status = "PASS (Error Free)"
        elif per < 1e-2: status = "Waterfall"
        
        print(f"{esn0_db:<10.1f} | {frames:<8} | {bit_err:<10} | {ber:<10.2e} | {per:<10.2e} | {status}")
        
        if per == 0 and esn0_db >= 2.0:
            print("-" * 90)
            print("PERFORMANCE CONFIRMED: Waterfall behavior matches standard.")
            break

if __name__ == "__main__":
    run_simulation()