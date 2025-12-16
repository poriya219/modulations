#!/usr/bin/env python3
"""
SDR Modem Class - Reusable modulation/demodulation for HackRF + RTL-SDR
Supports: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM
"""
import numpy as np
import subprocess
import os
import time
from scipy.signal import correlate
from typing import Optional, Tuple, Dict, Any


class SDRModem:
    """
    Software Defined Radio Modem for HackRF (TX) and RTL-SDR (RX)
    
    Usage:
        modem = SDRModem(fc=433e6, fs=2e6)
        
        # Transmit
        modem.transmit(data_bits, modulation='QPSK')
        
        # Receive
        rx_bits, stats = modem.receive(modulation='QPSK', n_expected_bits=240)
    """
    
    # Supported modulations with their parameters
    MODULATIONS = {
        'BPSK':   {'bps': 1, 'order': 2,   'alpha': 0.02,  'n_rot': 2, 'rot_step': np.pi},
        'QPSK':   {'bps': 2, 'order': 4,   'alpha': 0.015, 'n_rot': 4, 'rot_step': np.pi/2},
        '8PSK':   {'bps': 3, 'order': 8,   'alpha': 0.01,  'n_rot': 8, 'rot_step': np.pi/4},
        '16QAM':  {'bps': 4, 'order': 16,  'alpha': 0.008, 'n_rot': 4, 'rot_step': np.pi/2},
        '64QAM':  {'bps': 6, 'order': 64,  'alpha': 0.005, 'n_rot': 4, 'rot_step': np.pi/2},
        '256QAM': {'bps': 8, 'order': 256, 'alpha': 0.003, 'n_rot': 4, 'rot_step': np.pi/2},
    }
    
    def __init__(self, fc: float = 433e6, fs: float = 2e6, sps: int = 4,
                 tx_gain: int = 47, rx_gain: int = 49):
        """
        Initialize SDR Modem
        
        Args:
            fc: Center frequency in Hz (default: 433 MHz)
            fs: Sample rate in Hz (default: 2 MHz)
            sps: Samples per symbol (default: 4)
            tx_gain: HackRF TX gain 0-47 (default: 47)
            rx_gain: RTL-SDR RX gain 0-49 (default: 49)
        """
        self.fc = fc
        self.fs = fs
        self.sps = sps
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        
        # Generate RRC filter taps
        self.rrc_taps = self._rrc_filter(sps)
        
        # Default sync sequence (BPSK)
        self.sync_bits = np.array([1,0,1,0,1,0,1,0,1,1,0,0,1,1,0,0] * 10)
        self.sync_syms = self._bpsk_mod(self.sync_bits)
        
        # Gray coding tables
        self._init_gray_tables()
    
    def _init_gray_tables(self):
        """Initialize Gray coding lookup tables"""
        self.gray2 = [0, 1, 3, 2]
        self.gray3 = [0, 1, 3, 2, 6, 7, 5, 4]
        self.gray4 = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]
        self.inv_gray2 = [self.gray2.index(i) for i in range(4)]
        self.inv_gray3 = [self.gray3.index(i) for i in range(8)]
        self.inv_gray4 = [self.gray4.index(i) for i in range(16)]
    
    # ==================== FILTERS ====================
    
    def _rrc_filter(self, sps: int, alpha: float = 0.35, ntaps: int = 101) -> np.ndarray:
        """Root Raised Cosine filter"""
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
    
    def _upsample_filter(self, syms: np.ndarray) -> np.ndarray:
        """Upsample and apply pulse shaping"""
        up = np.zeros(len(syms) * self.sps, dtype=np.complex64)
        up[::self.sps] = syms
        return np.convolve(up, self.rrc_taps, mode='same')
    
    # ==================== MODULATION ====================
    
    def _bpsk_mod(self, bits: np.ndarray) -> np.ndarray:
        return 2.0 * np.array(bits, dtype=np.complex64) - 1.0
    
    def _bpsk_demod(self, syms: np.ndarray) -> np.ndarray:
        return (np.real(syms) > 0).astype(int)
    
    def _qpsk_mod(self, bits: np.ndarray) -> np.ndarray:
        bits = np.array(bits)
        if len(bits) % 2: bits = np.append(bits, 0)
        I = 1 - 2 * bits[0::2]
        Q = 1 - 2 * bits[1::2]
        return (I + 1j * Q).astype(np.complex64) / np.sqrt(2)
    
    def _qpsk_demod(self, syms: np.ndarray) -> np.ndarray:
        bits = np.zeros(len(syms) * 2, dtype=int)
        bits[0::2] = (np.real(syms) < 0).astype(int)
        bits[1::2] = (np.imag(syms) < 0).astype(int)
        return bits
    
    def _psk8_mod(self, bits: np.ndarray) -> np.ndarray:
        bits = np.array(bits)
        pad = (3 - len(bits) % 3) % 3
        if pad: bits = np.append(bits, [0] * pad)
        bits = bits.reshape(-1, 3)
        symbols = []
        for b in bits:
            idx = b[0] * 4 + b[1] * 2 + b[2]
            phase = self.gray3[idx] * np.pi / 4
            symbols.append(np.exp(1j * phase))
        return np.array(symbols, dtype=np.complex64)
    
    def _psk8_demod(self, syms: np.ndarray) -> np.ndarray:
        bits = []
        for s in syms:
            phase = np.angle(s)
            if phase < 0: phase += 2 * np.pi
            idx = int(np.round(phase / (np.pi / 4))) % 8
            orig_idx = self.inv_gray3[idx]
            bits.extend([(orig_idx >> 2) & 1, (orig_idx >> 1) & 1, orig_idx & 1])
        return np.array(bits)
    
    def _qam16_mod(self, bits: np.ndarray) -> np.ndarray:
        bits = np.array(bits)
        pad = (4 - len(bits) % 4) % 4
        if pad: bits = np.append(bits, [0] * pad)
        bits = bits.reshape(-1, 4)
        symbols = []
        for b in bits:
            i_idx = b[0] * 2 + b[1]
            q_idx = b[2] * 2 + b[3]
            I = (2 * self.gray2[i_idx] - 3) / np.sqrt(10)
            Q = (2 * self.gray2[q_idx] - 3) / np.sqrt(10)
            symbols.append(I + 1j * Q)
        return np.array(symbols, dtype=np.complex64)
    
    def _qam16_demod(self, syms: np.ndarray) -> np.ndarray:
        bits = []
        for s in syms:
            I = np.real(s) * np.sqrt(10)
            Q = np.imag(s) * np.sqrt(10)
            i_idx = int(np.clip(np.round((I + 3) / 2), 0, 3))
            q_idx = int(np.clip(np.round((Q + 3) / 2), 0, 3))
            orig_i = self.inv_gray2[i_idx]
            orig_q = self.inv_gray2[q_idx]
            bits.extend([(orig_i >> 1) & 1, orig_i & 1, (orig_q >> 1) & 1, orig_q & 1])
        return np.array(bits)
    
    def _qam64_mod(self, bits: np.ndarray) -> np.ndarray:
        bits = np.array(bits)
        pad = (6 - len(bits) % 6) % 6
        if pad: bits = np.append(bits, [0] * pad)
        bits = bits.reshape(-1, 6)
        symbols = []
        for b in bits:
            i_idx = b[0] * 4 + b[1] * 2 + b[2]
            q_idx = b[3] * 4 + b[4] * 2 + b[5]
            I = (2 * self.gray3[i_idx] - 7) / np.sqrt(42)
            Q = (2 * self.gray3[q_idx] - 7) / np.sqrt(42)
            symbols.append(I + 1j * Q)
        return np.array(symbols, dtype=np.complex64)
    
    def _qam64_demod(self, syms: np.ndarray) -> np.ndarray:
        bits = []
        for s in syms:
            I = np.real(s) * np.sqrt(42)
            Q = np.imag(s) * np.sqrt(42)
            i_idx = int(np.clip(np.round((I + 7) / 2), 0, 7))
            q_idx = int(np.clip(np.round((Q + 7) / 2), 0, 7))
            orig_i = self.inv_gray3[i_idx]
            orig_q = self.inv_gray3[q_idx]
            bits.extend([(orig_i >> 2) & 1, (orig_i >> 1) & 1, orig_i & 1,
                         (orig_q >> 2) & 1, (orig_q >> 1) & 1, orig_q & 1])
        return np.array(bits)
    
    def _qam256_mod(self, bits: np.ndarray) -> np.ndarray:
        bits = np.array(bits)
        pad = (8 - len(bits) % 8) % 8
        if pad: bits = np.append(bits, [0] * pad)
        bits = bits.reshape(-1, 8)
        symbols = []
        for b in bits:
            i_idx = b[0] * 8 + b[1] * 4 + b[2] * 2 + b[3]
            q_idx = b[4] * 8 + b[5] * 4 + b[6] * 2 + b[7]
            I = (2 * self.gray4[i_idx] - 15) / np.sqrt(170)
            Q = (2 * self.gray4[q_idx] - 15) / np.sqrt(170)
            symbols.append(I + 1j * Q)
        return np.array(symbols, dtype=np.complex64)
    
    def _qam256_demod(self, syms: np.ndarray) -> np.ndarray:
        bits = []
        for s in syms:
            I = np.real(s) * np.sqrt(170)
            Q = np.imag(s) * np.sqrt(170)
            i_idx = int(np.clip(np.round((I + 15) / 2), 0, 15))
            q_idx = int(np.clip(np.round((Q + 15) / 2), 0, 15))
            orig_i = self.inv_gray4[i_idx]
            orig_q = self.inv_gray4[q_idx]
            bits.extend([(orig_i >> 3) & 1, (orig_i >> 2) & 1, (orig_i >> 1) & 1, orig_i & 1,
                         (orig_q >> 3) & 1, (orig_q >> 2) & 1, (orig_q >> 1) & 1, orig_q & 1])
        return np.array(bits)
    
    def modulate(self, bits: np.ndarray, modulation: str = 'QPSK') -> np.ndarray:
        """
        Modulate bits to symbols
        
        Args:
            bits: Input bit array
            modulation: Modulation type ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '256QAM')
        
        Returns:
            Complex symbol array
        """
        mod_funcs = {
            'BPSK': self._bpsk_mod,
            'QPSK': self._qpsk_mod,
            '8PSK': self._psk8_mod,
            '16QAM': self._qam16_mod,
            '64QAM': self._qam64_mod,
            '256QAM': self._qam256_mod,
        }
        if modulation not in mod_funcs:
            raise ValueError(f"Unknown modulation: {modulation}")
        return mod_funcs[modulation](bits)
    
    def demodulate(self, symbols: np.ndarray, modulation: str = 'QPSK') -> np.ndarray:
        """
        Demodulate symbols to bits
        
        Args:
            symbols: Complex symbol array
            modulation: Modulation type
        
        Returns:
            Bit array
        """
        demod_funcs = {
            'BPSK': self._bpsk_demod,
            'QPSK': self._qpsk_demod,
            '8PSK': self._psk8_demod,
            '16QAM': self._qam16_demod,
            '64QAM': self._qam64_demod,
            '256QAM': self._qam256_demod,
        }
        if modulation not in demod_funcs:
            raise ValueError(f"Unknown modulation: {modulation}")
        return demod_funcs[modulation](symbols)
    
    # ==================== DSP ====================
    
    def _estimate_freq_offset(self, sig: np.ndarray) -> float:
        """Estimate frequency offset from CW preamble"""
        N = min(len(sig), 2**16)
        sig = sig[:N]
        win = np.hanning(N)
        fft = np.fft.fft(sig * win)
        freqs = np.fft.fftfreq(N, 1/self.fs)
        mag = np.abs(fft)
        mag[:N//100] = 0
        mag[-N//100:] = 0
        return freqs[np.argmax(mag)]
    
    def _correct_freq(self, sig: np.ndarray, f_offset: float) -> np.ndarray:
        """Apply frequency correction"""
        t = np.arange(len(sig)) / self.fs
        return sig * np.exp(-1j * 2 * np.pi * f_offset * t)
    
    def _carrier_recovery_bpsk(self, syms: np.ndarray, alpha: float = 0.03) -> Tuple[np.ndarray, float, float]:
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
    
    def _carrier_recovery_data(self, syms: np.ndarray, init_phase: float, 
                                init_freq: float, modulation: str, alpha: float) -> np.ndarray:
        """Carrier recovery for data symbols"""
        out = np.zeros_like(syms)
        phase = init_phase
        freq = init_freq
        beta = alpha * 0.1
        
        for i in range(len(syms)):
            out[i] = syms[i] * np.exp(-1j * phase)
            
            if modulation == 'BPSK':
                err = out[i].imag * np.sign(out[i].real)
            elif modulation in ['QPSK', '16QAM', '64QAM', '256QAM']:
                err = np.sign(out[i].real) * out[i].imag - np.sign(out[i].imag) * out[i].real
            else:  # 8PSK
                phase_s = np.angle(out[i])
                nearest = np.round(phase_s / (np.pi/4)) * (np.pi/4)
                err = np.sin(phase_s - nearest)
            
            freq += beta * err
            phase += alpha * err + freq
        
        return out
    
    # ==================== FILE I/O ====================
    
    def _save_iq(self, sig: np.ndarray, filename: str):
        """Save IQ data to file for HackRF"""
        sig = sig / (np.max(np.abs(sig)) + 1e-10) * 0.95
        iq = np.zeros(len(sig) * 2, dtype=np.int8)
        iq[0::2] = np.clip(np.real(sig) * 127, -127, 127).astype(np.int8)
        iq[1::2] = np.clip(np.imag(sig) * 127, -127, 127).astype(np.int8)
        iq.tofile(filename)
    
    def _load_iq(self, filename: str) -> np.ndarray:
        """Load IQ data from RTL-SDR file"""
        raw = np.fromfile(filename, dtype=np.uint8)
        I = (raw[0::2].astype(np.float32) - 127.5) / 127.5
        Q = (raw[1::2].astype(np.float32) - 127.5) / 127.5
        return I + 1j * Q
    
    # ==================== FRAME BUILDING ====================
    
    def build_frame(self, data_bits: np.ndarray, modulation: str = 'QPSK',
                    n_repeats: int = 20) -> np.ndarray:
        """
        Build complete TX frame with preamble, sync, and data
        
        Args:
            data_bits: Data bits to transmit
            modulation: Modulation type for data
            n_repeats: Number of frame repetitions
        
        Returns:
            Complete baseband signal ready for transmission
        """
        # Preamble (CW for frequency estimation)
        preamble_len = int(0.01 * self.fs)
        preamble = np.ones(preamble_len, dtype=np.complex64)
        
        # Use longer sync for higher-order modulations
        order = self.MODULATIONS[modulation]['order']
        if order >= 64:
            sync_bits = np.array([1,0,1,0,1,0,1,0,1,1,0,0,1,1,0,0] * 15)
        else:
            sync_bits = self.sync_bits
        sync_syms = self._bpsk_mod(sync_bits)
        
        # Modulate data
        data_syms = self.modulate(data_bits, modulation)
        
        # Build frame
        frame_syms = np.concatenate([sync_syms, data_syms])
        frame_wave = self._upsample_filter(frame_syms)
        
        # Combine with preamble
        full_sig = np.concatenate([preamble, frame_wave])
        
        # Add padding and repeat
        pad = np.zeros(int(0.02 * self.fs), dtype=np.complex64)
        burst = np.concatenate([pad, full_sig, pad])
        
        return np.tile(burst, n_repeats)
    
    # ==================== TRANSMIT ====================
    
    def transmit(self, data_bits: np.ndarray, modulation: str = 'QPSK',
                 duration: float = 5.0, blocking: bool = True) -> bool:
        """
        Transmit data over HackRF
        
        Args:
            data_bits: Data bits to transmit
            modulation: Modulation type
            duration: Transmission duration in seconds
            blocking: Wait for transmission to complete
        
        Returns:
            True if successful
        """
        tx_file = '/tmp/sdr_modem_tx.iq'
        
        # Build and save frame
        tx_signal = self.build_frame(data_bits, modulation)
        self._save_iq(tx_signal, tx_file)
        
        # Start HackRF
        cmd = ['hackrf_transfer', '-t', tx_file, '-f', str(int(self.fc)),
               '-s', str(int(self.fs)), '-x', str(self.tx_gain), '-a', '1', '-R']
        
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if blocking:
            time.sleep(duration)
            proc.terminate()
            proc.wait()
            if os.path.exists(tx_file):
                os.remove(tx_file)
        
        return True
    
    # ==================== RECEIVE ====================
    
    def receive(self, modulation: str = 'QPSK', n_expected_bits: int = 240,
                duration: float = 5.0) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Receive and demodulate data from RTL-SDR
        
        Args:
            modulation: Expected modulation type
            n_expected_bits: Expected number of data bits
            duration: Receive duration in seconds
        
        Returns:
            Tuple of (received_bits or None, statistics dict)
        """
        rx_file = '/tmp/sdr_modem_rx.iq'
        
        if os.path.exists(rx_file):
            os.remove(rx_file)
        
        # Start RTL-SDR
        cmd = ['rtl_sdr', '-f', str(int(self.fc)), '-s', str(int(self.fs)),
               '-g', str(self.rx_gain), rx_file]
        
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(duration)
        proc.terminate()
        proc.wait()
        time.sleep(0.3)
        
        # Load and process
        if not os.path.exists(rx_file) or os.path.getsize(rx_file) < 10000:
            return None, {'error': 'No RX data', 'success': False}
        
        rx_raw = self._load_iq(rx_file)
        os.remove(rx_file)
        
        return self._process_received(rx_raw, modulation, n_expected_bits)
    
    def receive_and_transmit(self, data_bits: np.ndarray, modulation: str = 'QPSK',
                             duration: float = 5.0) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Simultaneous transmit and receive (for testing)
        
        Args:
            data_bits: Data bits to transmit
            modulation: Modulation type
            duration: Duration in seconds
        
        Returns:
            Tuple of (received_bits or None, statistics dict)
        """
        tx_file = '/tmp/sdr_modem_tx.iq'
        rx_file = '/tmp/sdr_modem_rx.iq'
        
        # Clean up
        for f in [tx_file, rx_file]:
            if os.path.exists(f):
                os.remove(f)
        
        # Build and save TX frame
        tx_signal = self.build_frame(data_bits, modulation)
        self._save_iq(tx_signal, tx_file)
        
        # Start RX first
        rx_cmd = ['rtl_sdr', '-f', str(int(self.fc)), '-s', str(int(self.fs)),
                  '-g', str(self.rx_gain), rx_file]
        rx_proc = subprocess.Popen(rx_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
        
        # Start TX
        tx_cmd = ['hackrf_transfer', '-t', tx_file, '-f', str(int(self.fc)),
                  '-s', str(int(self.fs)), '-x', str(self.tx_gain), '-a', '1', '-R']
        tx_proc = subprocess.Popen(tx_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait
        time.sleep(duration)
        
        # Stop
        tx_proc.terminate()
        tx_proc.wait()
        time.sleep(0.3)
        rx_proc.terminate()
        rx_proc.wait()
        time.sleep(0.3)
        
        # Clean TX file
        if os.path.exists(tx_file):
            os.remove(tx_file)
        
        # Load and process RX
        if not os.path.exists(rx_file) or os.path.getsize(rx_file) < 10000:
            return None, {'error': 'No RX data', 'success': False}
        
        rx_raw = self._load_iq(rx_file)
        os.remove(rx_file)
        
        return self._process_received(rx_raw, modulation, len(data_bits))
    
    def _process_received(self, rx_raw: np.ndarray, modulation: str,
                          n_expected_bits: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Process received IQ data and extract bits"""
        stats = {'success': False, 'n_samples': len(rx_raw)}
        
        # DC removal
        rx = rx_raw - np.mean(rx_raw)
        
        # Find burst using power detection
        preamble_len = int(0.01 * self.fs)
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
            stats['error'] = 'Could not find burst'
            return None, stats
        
        stats['burst_start'] = burst_start
        
        # Frequency offset estimation and correction
        preamble_rx = rx[burst_start:burst_start + preamble_len]
        f_offset = self._estimate_freq_offset(preamble_rx)
        stats['freq_offset'] = f_offset
        
        rx_corrected = self._correct_freq(rx, f_offset)
        
        # Matched filter
        rx_filt = np.convolve(rx_corrected, self.rrc_taps, mode='same')
        
        # Get sync sequence for this modulation
        order = self.MODULATIONS[modulation]['order']
        if order >= 64:
            sync_bits = np.array([1,0,1,0,1,0,1,0,1,1,0,0,1,1,0,0] * 15)
        else:
            sync_bits = self.sync_bits
        sync_syms = self._bpsk_mod(sync_bits)
        
        # Find sync using correlation
        sync_wave = self._upsample_filter(sync_syms)
        search_start = burst_start + preamble_len - 100
        search_end = min(len(rx_filt), burst_start + int(0.5 * self.fs))
        search_region = rx_filt[search_start:search_end]
        
        corr = np.abs(correlate(search_region, sync_wave, mode='valid'))
        peak_idx = np.argmax(corr)
        sync_start = search_start + peak_idx
        
        stats['sync_peak'] = corr[peak_idx]
        stats['sync_start'] = sync_start
        
        # Calculate expected data symbols
        bps = self.MODULATIONS[modulation]['bps']
        n_data_syms = (n_expected_bits + bps - 1) // bps
        
        # Extract frame
        total_syms = len(sync_syms) + n_data_syms
        frame_samples = total_syms * self.sps + self.sps * 4
        
        if sync_start + frame_samples > len(rx_filt):
            stats['error'] = 'Frame past buffer'
            return None, stats
        
        rx_frame = rx_filt[sync_start:sync_start + frame_samples]
        
        # Get modulation parameters
        mod_cfg = self.MODULATIONS[modulation]
        cr_alpha = mod_cfg['alpha']
        n_rot = mod_cfg['n_rot']
        rot_step = mod_cfg['rot_step']
        
        # Find best timing and phase
        best_ber = 1.0
        best_data_rx = None
        best_sync_ber = 1.0
        
        for offset in range(self.sps):
            syms = rx_frame[offset::self.sps]
            if len(syms) < total_syms:
                continue
            
            sync_rx_raw = syms[:len(sync_syms)]
            data_rx_raw = syms[len(sync_syms):len(sync_syms) + n_data_syms]
            
            # BPSK carrier recovery for sync
            sync_rec, final_phase, final_freq = self._carrier_recovery_bpsk(sync_rx_raw)
            
            # Normalize
            pwr = np.mean(np.abs(sync_rec) ** 2)
            if pwr > 0:
                sync_rec = sync_rec / np.sqrt(pwr)
            
            for sync_rot in range(2):
                sync_phase = np.exp(1j * sync_rot * np.pi)
                sync_test = sync_rec * sync_phase
                
                sync_bits_rx = self._bpsk_demod(sync_test)
                sync_ber = np.mean(sync_bits_rx != sync_bits)
                
                if sync_ber > 0.2:
                    continue
                
                # Data carrier recovery
                adjusted_phase = final_phase + sync_rot * np.pi
                data_rec = self._carrier_recovery_data(data_rx_raw, adjusted_phase, 
                                                        final_freq, modulation, cr_alpha)
                
                # Normalize
                pwr = np.mean(np.abs(data_rec) ** 2)
                if pwr > 0:
                    data_rec = data_rec / np.sqrt(pwr)
                
                # Try phase rotations
                for rot in range(n_rot):
                    data_test = data_rec * np.exp(1j * rot * rot_step)
                    data_bits_rx = self.demodulate(data_test, modulation)
                    
                    # For BER calculation, we need known data - skip if not available
                    # Just store the result with lowest sync BER
                    if sync_ber < best_sync_ber:
                        best_sync_ber = sync_ber
                        best_data_rx = data_test.copy()
        
        if best_data_rx is None:
            stats['error'] = 'Could not decode'
            return None, stats
        
        # Final demodulation
        data_bits_rx = self.demodulate(best_data_rx, modulation)
        
        stats['success'] = True
        stats['sync_ber'] = best_sync_ber
        stats['rx_symbols'] = best_data_rx
        
        return data_bits_rx[:n_expected_bits], stats


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("="*60)
    print("SDR Modem Test")
    print("="*60)
    
    # Create modem
    modem = SDRModem(fc=433e6, fs=2e6)
    
    # Test each modulation
    for mod in ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '256QAM']:
        print(f"\n--- Testing {mod} ---")
        
        # Generate random data
        n_bits = 240
        tx_bits = np.random.randint(0, 2, n_bits)
        
        # Transmit and receive
        rx_bits, stats = modem.receive_and_transmit(tx_bits, modulation=mod, duration=5.0)
        
        if rx_bits is not None:
            # Calculate BER
            L = min(len(tx_bits), len(rx_bits))
            errors = np.sum(tx_bits[:L] != rx_bits[:L])
            ber = errors / L
            
            print(f"  Freq offset: {stats.get('freq_offset', 0):.1f} Hz")
            print(f"  Sync BER: {stats.get('sync_ber', 1.0):.4f}")
            print(f"  Data BER: {ber:.5f} ({errors}/{L})")
            print(f"  Status: {'PASS' if ber < 0.1 else 'FAIL'}")
        else:
            print(f"  ERROR: {stats.get('error', 'Unknown')}")
        
        time.sleep(3)
    
    print("\n" + "="*60)
    print("Done!")



# How To Use:
# from sdr_modem import SDRModem
# import numpy as np

# # ایجاد modem
# modem = SDRModem(fc=433e6, fs=2e6)

# # ارسال و دریافت همزمان (برای تست)
# tx_bits = np.random.randint(0, 2, 240)
# rx_bits, stats = modem.receive_and_transmit(tx_bits, modulation='QPSK')

# # فقط ارسال
# modem.transmit(tx_bits, modulation='16QAM', duration=5.0)

# # فقط دریافت
# rx_bits, stats = modem.receive(modulation='16QAM', n_expected_bits=240)

# # مدولاسیون/دمدولاسیون مستقیم (بدون SDR)
# symbols = modem.modulate(tx_bits, 'QPSK')
# bits = modem.demodulate(symbols, 'QPSK')