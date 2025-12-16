import time
from rtlsdr import RtlSdr

def test_tuner():
    print("--- RTL-SDR Diagnostic ---")
    try:
        sdr = RtlSdr()
        sdr.sample_rate = 2.048e6
        sdr.center_freq = 100e6 # FM Band (Strong signals)
        sdr.gain = 'auto'
        
        print("1. Device Opened Successfully")
        
        # Try to read samples
        samples = sdr.read_samples(1024*256)
        print(f"2. Read {len(samples)} samples successfully")
        
        # Check signal strength (Mean Magnitude)
        avg_pwr = sum(abs(samples))/len(samples)
        print(f"3. Average Signal Magnitude: {avg_pwr:.4f}")
        
        if avg_pwr < 0.01:
            print("   WARNING: Signal is extremely weak (Is Antenna connected?)")
        else:
            print("   Signal strength looks normal.")

        sdr.close()
        print("--- DIAGNOSTIC PASS ---")
        print("Your RTL-SDR is working correctly in isolation.")
        
    except Exception as e:
        print("\n!!! DIAGNOSTIC FAIL !!!")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("- Unplug and replug the device.")
        print("- Try a different USB port (USB 2.0 is often more stable than 3.0 for these).")

if __name__ == "__main__":
    test_tuner()