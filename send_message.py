#!/usr/bin/env python3
"""
Interactive Message Sender/Receiver using SDR Modem
Sends a text message over HackRF and receives it with RTL-SDR
"""
import numpy as np
import matplotlib.pyplot as plt
from sdr_modem import SDRModem
import sys


def text_to_bits(text: str) -> np.ndarray:
    """Convert text string to bit array"""
    bits = []
    for char in text:
        # Convert each character to 8 bits
        char_bits = format(ord(char), '08b')
        bits.extend([int(b) for b in char_bits])
    return np.array(bits)


def bits_to_text(bits: np.ndarray) -> str:
    """Convert bit array back to text string"""
    # Ensure bits length is multiple of 8
    n_chars = len(bits) // 8
    text = ""
    for i in range(n_chars):
        char_bits = bits[i*8:(i+1)*8]
        char_val = int(''.join(str(b) for b in char_bits), 2)
        if 32 <= char_val <= 126:  # Printable ASCII
            text += chr(char_val)
        else:
            text += '?'
    return text


def plot_results(tx_bits, rx_bits, tx_symbols, rx_symbols, modulation, message, stats):
    """Create visualization of transmission results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Calculate BER
    L = min(len(tx_bits), len(rx_bits))
    errors = np.sum(tx_bits[:L] != rx_bits[:L])
    ber = errors / L if L > 0 else 1.0
    
    # Title
    fig.suptitle(f'SDR Message Transmission - {modulation}\n"{message[:50]}{"..." if len(message) > 50 else ""}"', 
                 fontsize=14, fontweight='bold')
    
    # 1.TX Constellation
    ax = axes[0, 0]
    ax.scatter(tx_symbols.real, tx_symbols.imag, s=50, c='blue', alpha=0.7, label='TX')
    ax.set_title(f'TX Constellation ({modulation})')
    ax.set_xlabel('In-Phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # 2.RX Constellation
    ax = axes[0, 1]
    if rx_symbols is not None:
        ax.scatter(rx_symbols.real, rx_symbols.imag, s=10, c='red', alpha=0.5, label='RX')
    ax.set_title('RX Constellation (After Recovery)')
    ax.set_xlabel('In-Phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # 3. Constellation Comparison (overlay)
    ax = axes[0, 2]
    ax.scatter(tx_symbols.real, tx_symbols.imag, s=100, c='blue', alpha=0.5, marker='x', label='TX (Ideal)')
    if rx_symbols is not None:
        ax.scatter(rx_symbols.real, rx_symbols.imag, s=10, c='red', alpha=0.3, label='RX')
    ax.set_title('TX vs RX Constellation')
    ax.set_xlabel('In-Phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.legend()
    
    # 4. Bit Comparison
    ax = axes[1, 0]
    L_show = min(80, len(tx_bits), len(rx_bits))
    x = np.arange(L_show)
    ax.stem(x, tx_bits[:L_show], 'b-', markerfmt='bo', basefmt=' ', label='TX Bits')
    ax.stem(x + 0.3, rx_bits[:L_show], 'r-', markerfmt='rx', basefmt=' ', label='RX Bits')
    ax.set_title('Bit Comparison (First 80 bits)')
    ax.set_xlabel('Bit Index')
    ax.set_ylabel('Bit Value')
    ax.legend()
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True, alpha=0.3)
    
    # 5.Bit Errors
    ax = axes[1, 1]
    bit_errors = (tx_bits[:L] != rx_bits[:L]).astype(int)
    error_positions = np.where(bit_errors)[0]
    ax.bar(range(len(bit_errors)), bit_errors, color='red', alpha=0.7)
    ax.set_title(f'Bit Errors ({errors} errors in {L} bits)')
    ax.set_xlabel('Bit Index')
    ax.set_ylabel('Error (1=Error)')
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3)
    
    # 6.Statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    # Decode received message
    rx_message = bits_to_text(rx_bits)
    
    stats_text = f"""
    ══════════════════════════════════════
                TRANSMISSION RESULTS
    ══════════════════════════════════════
    
    Modulation:     {modulation}
    Bits/Symbol:    {SDRModem.MODULATIONS[modulation]['bps']}
    
    TX Message:     "{message[:30]}{'...' if len(message) > 30 else ''}"
    RX Message:     "{rx_message[:30]}{'...' if len(rx_message) > 30 else ''}"
    
    TX Bits:        {len(tx_bits)}
    RX Bits:        {len(rx_bits)}
    
    Bit Errors:     {errors}
    BER:            {ber:.6f} ({ber*100:.2f}%)
    
    Freq Offset:    {stats.get('freq_offset', 0):.1f} Hz
    Sync BER:       {stats.get('sync_ber', 0):.4f}
    
    Status:         {'✓ SUCCESS' if ber < 0.1 else '✗ FAILED'}
    ══════════════════════════════════════
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save and show
    filename = f'message_result_{modulation}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {filename}")
    
    plt.show()


def main():
    print("\n" + "="*60)
    print("      📡 SDR MESSAGE TRANSMITTER / RECEIVER 📡")
    print("="*60)
    print("\nThis program sends a text message over the air using")
    print("HackRF (TX) and RTL-SDR (RX) with your chosen modulation.\n")
    
    # Get message from user
    print("─" * 60)
    message = input("📝 Enter your message: ").strip()
    
    if not message:
        message = "Hello SDR World!"
        print(f"   (Using default message: '{message}')")
    
    # Show modulation options
    print("\n─" * 60)
    print("📻 Available Modulations:")
    print("   1.BPSK   (1 bit/symbol)  - Most robust")
    print("   2.QPSK   (2 bits/symbol) - Good balance")
    print("   3.8PSK   (3 bits/symbol)")
    print("   4.16QAM  (4 bits/symbol)")
    print("   5.64QAM  (6 bits/symbol)")
    print("   6.256QAM (8 bits/symbol) - Highest throughput")
    
    mod_options = {
        '1': 'BPSK', '2': 'QPSK', '3': '8PSK',
        '4': '16QAM', '5': '64QAM', '6': '256QAM',
        'bpsk': 'BPSK', 'qpsk': 'QPSK', '8psk': '8PSK',
        '16qam': '16QAM', '64qam': '64QAM', '256qam': '256QAM'
    }
    
    choice = input("\n🔢 Select modulation (1-6 or name): ").strip().lower()
    modulation = mod_options.get(choice, 'QPSK')
    print(f"   Selected: {modulation}")
    
    # Convert message to bits
    tx_bits = text_to_bits(message)
    
    # Pad to ensure proper symbol alignment
    bps = SDRModem.MODULATIONS[modulation]['bps']
    pad_len = (bps - len(tx_bits) % bps) % bps
    if pad_len:
        tx_bits = np.append(tx_bits, [0] * pad_len)
    
    print("\n─" * 60)
    print("📊 Transmission Info:")
    print(f"   Message length: {len(message)} characters")
    print(f"   Total bits: {len(tx_bits)}")
    print(f"   Symbols: {len(tx_bits) // bps}")
    
    # Create modem
    print("\n─" * 60)
    print("🔧 Initializing SDR Modem...")
    modem = SDRModem(fc=400e6, fs=2e6, sps=4, tx_gain=40, rx_gain=40)
    
    # Get TX symbols for plotting
    tx_symbols = modem.modulate(tx_bits, modulation)
    
    # Transmit and receive
    print("\n─" * 60)
    print("📡 Starting transmission...")
    print("   ⏳ Please wait (5 seconds)...")
    
    rx_bits, stats = modem.receive_and_transmit(
        tx_bits, 
        modulation=modulation, 
        duration=5.0
    )
    
    # Process results
    print("\n─" * 60)
    if rx_bits is not None and stats.get('success', False):
        # Calculate BER
        L = min(len(tx_bits), len(rx_bits))
        errors = np.sum(tx_bits[:L] != rx_bits[:L])
        ber = errors / L
        
        # Decode received message
        rx_message = bits_to_text(rx_bits)
        
        print("📨 RESULTS:")
        print(f"   TX Message: \"{message}\"")
        print(f"   RX Message: \"{rx_message}\"")
        print(f"\n   Frequency Offset: {stats.get('freq_offset', 0):.1f} Hz")
        print(f"   Sync BER: {stats.get('sync_ber', 0):.4f}")
        print(f"   Data BER: {ber:.5f} ({errors}/{L} errors)")
        
        if ber < 0.01:
            print("\n   ✅ PERFECT RECEPTION!")
        elif ber < 0.1:
            print("\n   ✅ GOOD RECEPTION!")
        elif ber < 0.3:
            print("\n   ⚠️  MARGINAL RECEPTION")
        else:
            print("\n   ❌ POOR RECEPTION")
        
        # Get RX symbols for plotting
        rx_symbols = stats.get('rx_symbols', None)
        
        # Plot results
        print("\n─" * 60)
        print("📈 Generating plots...")
        plot_results(tx_bits, rx_bits, tx_symbols, rx_symbols, modulation, message, stats)
        
    else:
        print("❌ TRANSMISSION FAILED!")
        print(f"   Error: {stats.get('error', 'Unknown error')}")
        
        # Still show TX constellation
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(tx_symbols.real, tx_symbols.imag, s=50, c='blue', alpha=0.7)
        ax.set_title(f'TX Constellation - {modulation}\n(Reception Failed)')
        ax.set_xlabel('In-Phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        plt.savefig(f'message_failed_{modulation}.png', dpi=150)
        plt.show()
    
    print("\n" + "="*60)
    print("Done! 👋")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)