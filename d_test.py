#!/usr/bin/env python3
"""
DEBUG:  Find why extrinsic info is not working
"""
import numpy as np
from numba import njit

# Import the codec
from dvb_rcs2_turbo import DVB_RCS2_TurboCodec, bcjr_decode


def debug_turbo_iteration():
    """Debug single turbo iteration"""
    print("="*70)
    print("DEBUG: TURBO ITERATION ANALYSIS")
    print("="*70)
    
    codec = DVB_RCS2_TurboCodec(212, '1/2', 1)
    
    np.random.seed(42)
    info = np.random.randint(0, 2, codec.k_info)
    coded = codec.encode(info)
    
    # Es/N0 = 3 dB
    esn0_lin = 10 ** (3.0 / 10)
    noise_var = 1.0 / esn0_lin
    
    tx = 1.0 - 2.0 * coded.astype(float)
    np.random.seed(123)
    rx = tx + np.sqrt(noise_var) * np.random.randn(len(tx))
    llr = 2.0 * rx / noise_var
    
    # Manual de-puncture
    N = codec.N
    Lc_A = np.zeros(N)
    Lc_B = np.zeros(N)
    Lc_W1 = np.zeros(N)
    Lc_Y1 = np.zeros(N)
    Lc_W2 = np.zeros(N)
    Lc_Y2 = np.zeros(N)
    
    idx = 0
    for i in range(N):
        Lc_A[i] = llr[idx]; idx += 1
        Lc_B[i] = llr[idx]; idx += 1
        if i % 2 == 0:
            Lc_W1[i] = llr[idx]; idx += 1
            Lc_Y2[i] = llr[idx]; idx += 1
        else:
            Lc_Y1[i] = llr[idx]; idx += 1
            Lc_W2[i] = llr[idx]; idx += 1
    
    print(f"\nChannel LLR statistics:")
    print(f"  Lc_A: mean={np.mean(Lc_A):.2f}, std={np.std(Lc_A):.2f}")
    print(f"  Lc_B: mean={np.mean(Lc_B):.2f}, std={np.std(Lc_B):.2f}")
    print(f"  Lc_W1: mean={np.mean(Lc_W1):.2f}, std={np.std(Lc_W1):.2f}, zeros={np.sum(Lc_W1==0)}")
    print(f"  Lc_Y1: mean={np.mean(Lc_Y1):.2f}, std={np.std(Lc_Y1):.2f}, zeros={np.sum(Lc_Y1==0)}")
    print(f"  Lc_W2: mean={np.mean(Lc_W2):.2f}, std={np.std(Lc_W2):.2f}, zeros={np.sum(Lc_W2==0)}")
    print(f"  Lc_Y2: mean={np.mean(Lc_Y2):.2f}, std={np.std(Lc_Y2):.2f}, zeros={np.sum(Lc_Y2==0)}")
    
    # Run decoder 1
    La_A = np.zeros(N)
    La_B = np.zeros(N)
    
    Le1_A, Le1_B = codec.decoder1.decode(Lc_A, Lc_B, Lc_W1, Lc_Y1, La_A, La_B)
    
    print(f"\nExtrinsic from Decoder 1:")
    print(f"  Le1_A: mean={np.mean(Le1_A):.2f}, std={np.std(Le1_A):.2f}, min={np.min(Le1_A):.2f}, max={np.max(Le1_A):.2f}")
    print(f"  Le1_B: mean={np.mean(Le1_B):.2f}, std={np.std(Le1_B):.2f}, min={np.min(Le1_B):.2f}, max={np.max(Le1_B):.2f}")
    
    # Check if extrinsic is meaningful
    print(f"\n  Non-zero Le1_A: {np.sum(np.abs(Le1_A) > 0.1)}/{N}")
    print(f"  Non-zero Le1_B: {np.sum(np.abs(Le1_B) > 0.1)}/{N}")
    
    # Interleave
    Le1_A_int = Le1_A[codec.interleaver.perm]
    Le1_B_int = Le1_B[codec.interleaver.perm]
    Lc_A_int = Lc_A[codec.interleaver.perm]
    Lc_B_int = Lc_B[codec.interleaver.perm]
    
    # Run decoder 2
    Le2_A_int, Le2_B_int = codec.decoder2.decode(
        Lc_A_int, Lc_B_int, Lc_W2, Lc_Y2, Le1_A_int, Le1_B_int
    )
    
    print(f"\nExtrinsic from Decoder 2:")
    print(f"  Le2_A:  mean={np.mean(Le2_A_int):.2f}, std={np.std(Le2_A_int):.2f}")
    print(f"  Le2_B: mean={np.mean(Le2_B_int):.2f}, std={np.std(Le2_B_int):.2f}")
    
    # De-interleave
    La_A_new = Le2_A_int[codec.interleaver.inv_perm]
    La_B_new = Le2_B_int[codec.interleaver.inv_perm]
    
    print(f"\nA priori for next iteration:")
    print(f"  La_A: mean={np.mean(La_A_new):.2f}, std={np.std(La_A_new):.2f}")
    print(f"  La_B:  mean={np.mean(La_B_new):.2f}, std={np.std(La_B_new):.2f}")
    
    # Second iteration
    Le1_A_2, Le1_B_2 = codec.decoder1.decode(Lc_A, Lc_B, Lc_W1, Lc_Y1, La_A_new, La_B_new)
    
    print(f"\nExtrinsic from Decoder 1 (iteration 2):")
    print(f"  Le1_A:  mean={np.mean(Le1_A_2):.2f}, std={np.std(Le1_A_2):.2f}")
    print(f"  Le1_B: mean={np.mean(Le1_B_2):.2f}, std={np.std(Le1_B_2):.2f}")
    
    # Compare
    print(f"\nChange in extrinsic (iteration 1 vs 2):")
    print(f"  ΔLe_A: {np.mean(np.abs(Le1_A_2 - Le1_A)):.4f}")
    print(f"  ΔLe_B: {np.mean(np.abs(Le1_B_2 - Le1_B)):.4f}")
    
    # Final decision comparison
    L_total_A = Lc_A + Le1_A + La_A_new
    L_total_B = Lc_B + Le1_B + La_B_new
    
    decoded_A = (L_total_A < 0).astype(int)
    decoded_B = (L_total_B < 0).astype(int)
    
    A_true = info[0:: 2]
    B_true = info[1::2]
    
    errors_A = np.sum(decoded_A != A_true)
    errors_B = np.sum(decoded_B != B_true)
    
    print(f"\nErrors after 1 full iteration:")
    print(f"  A errors: {errors_A}")
    print(f"  B errors: {errors_B}")
    print(f"  Total:  {errors_A + errors_B}")


def check_puncturing():
    """Verify puncturing pattern is correct"""
    print("\n" + "="*70)
    print("DEBUG:  PUNCTURING PATTERN CHECK")
    print("="*70)
    
    codec = DVB_RCS2_TurboCodec(212, '1/2', 1)
    
    # The DVB-RCS2 puncturing for rate 1/2 should be:
    # Even positions: A, B, W1, Y2
    # Odd positions:   A, B, Y1, W2
    
    # This means:
    # - W1 is transmitted at even positions, punctured at odd
    # - Y1 is transmitted at odd positions, punctured at even
    # - W2 is transmitted at odd positions, punctured at even  
    # - Y2 is transmitted at even positions, punctured at odd
    
    print("\nExpected puncturing pattern (DVB-RCS2 rate 1/2):")
    print("  Even i: transmit W1, Y2 (puncture Y1, W2)")
    print("  Odd i:  transmit Y1, W2 (puncture W1, Y2)")
    
    # Check what we're doing
    print("\nCurrent implementation:")
    N = 4  # Just check first 4 positions
    for i in range(N):
        if i % 2 == 0:
            print(f"  i={i} (even): W1, Y2")
        else:
            print(f"  i={i} (odd):  Y1, W2")
    
    print("\nThis looks correct!")
    
    # But let's verify the parity bits are correctly computed
    np.random.seed(42)
    info = np.random.randint(0, 2, codec.k_info)
    
    A = info[0::2]
    B = info[1::2]
    
    # Get encoder outputs
    W1, Y1 = codec.encoder1.encode(A, B)
    A_int, B_int = codec.interleaver.interleave(A, B)
    W2, Y2 = codec.encoder2.encode(A_int, B_int)
    
    print(f"\nEncoder outputs (first 8):")
    print(f"  W1: {W1[: 8]}")
    print(f"  Y1: {Y1[:8]}")
    print(f"  W2: {W2[:8]}")
    print(f"  Y2: {Y2[:8]}")
    
    # Check coded output
    coded = codec.encode(info)
    
    print(f"\nCoded output structure (first 16):")
    for i in range(4):
        base = i * 4
        print(f"  Symbol {i}: A={coded[base]}, B={coded[base+1]}, P1={coded[base+2]}, P2={coded[base+3]}")
        
        # Verify
        if i % 2 == 0:
            expected_P1 = W1[i]
            expected_P2 = Y2[i]
        else:
            expected_P1 = Y1[i]
            expected_P2 = W2[i]
        
        match_P1 = "✓" if coded[base+2] == expected_P1 else "✗"
        match_P2 = "✓" if coded[base+3] == expected_P2 else "✗"
        print(f"           Expected P1={expected_P1}{match_P1}, P2={expected_P2}{match_P2}")


def check_interleaver():
    """Verify interleaver is working"""
    print("\n" + "="*70)
    print("DEBUG: INTERLEAVER CHECK")
    print("="*70)
    
    codec = DVB_RCS2_TurboCodec(212, '1/2', 1)
    
    print(f"\nInterleaver parameters:")
    print(f"  N = {codec.interleaver.N}")
    print(f"  First 10 perm: {codec.interleaver.perm[:10]}")
    print(f"  First 10 inv:   {codec.interleaver.inv_perm[:10]}")
    
    # Verify inverse
    test = np.arange(codec.N)
    interleaved = test[codec.interleaver.perm]
    deinterleaved = interleaved[codec.interleaver.inv_perm]
    
    print(f"\n  Inverse check:  {np.all(test == deinterleaved)}")
    
    # Check spread
    diffs = np.abs(np.diff(codec.interleaver.perm))
    print(f"\n  Spread statistics:")
    print(f"    Min diff: {np.min(diffs)}")
    print(f"    Max diff: {np.max(diffs)}")
    print(f"    Mean diff: {np.mean(diffs):.1f}")


def main():
    debug_turbo_iteration()
    check_puncturing()
    check_interleaver()
    
    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()