#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import signal as sp_signal

# ───────────────────────────────────────────────────────────
# Utility
# ───────────────────────────────────────────────────────────
def shift_edge(x: np.ndarray, k: int) -> np.ndarray:
    """Shift by k with edge replication (phase-agnostic helper)."""
    y = np.empty_like(x)
    if k > 0:
        y[:k] = x[0]
        y[k:] = x[:-k]
    elif k < 0:
        k = -k
        y[-k:] = x[-1]
        y[:-k] = x[k:]
    else:
        y[:] = x
    return y

# ───────────────────────────────────────────────────────────
# Build the 3-chip complex templates (phase-agnostic via |corr|)
# ───────────────────────────────────────────────────────────
def build_fsk_template(pattern_bits: str, fs: float, Lchip: int, f0: float, f1: float) -> np.ndarray:
    """
    pattern_bits: 3-char string '000'..'111', one char per *chip*
    Lchip: samples per chip
    Template is complex, continuous-phase across chips, start phase=0.
    We take |correlation| later, so global phase is irrelevant.
    """
    freqs = np.array([f0 if c == '0' else f1 for c in pattern_bits], dtype=float)
    freq_samples = np.repeat(freqs, Lchip)
    phase = 2 * np.pi * np.cumsum(freq_samples) / fs
    s = np.exp(1j * phase)
    # Energy norm so all templates respond on the same scale
    s = s / np.sqrt(np.sum(np.abs(s) ** 2) + 1e-12)
    return s

def build_filter_bank(fs: float, Lchip: int, f0: float, f1: float):
    """
    Returns dict: { '000': template, ..., '111': template }
    """
    bank = {}
    for v in range(8):
        pat = format(v, '03b')  # '000'..'111'
        bank[pat] = build_fsk_template(pat, fs, Lchip, f0, f1)
    return bank

# ───────────────────────────────────────────────────────────
# FFT correlation for each template (frequency-domain)
# ───────────────────────────────────────────────────────────
def correlate_bank_fft(sig: np.ndarray, bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Uses scipy.signal.correlate(..., method='fft') for speed.
    Returns |correlation| per template (phase-agnostic score).
    """
    scores = {}
    for pat, tmpl in bank.items():
        # 'same' → each sample index corresponds to the template centered there
        r = sp_signal.correlate(sig, tmpl, mode='same', method='fft')
        scores[pat] = np.abs(r)
    return scores

# ───────────────────────────────────────────────────────────
# Peak / clock alignment & bit decisions from the bank
# ───────────────────────────────────────────────────────────
@dataclass
class BankDetectSettings:
    neighbor_alpha: float = 0.5   # suppress peaks 1 chip away (boundaries)
    min_peak_dist_bits: float = 0.85  # minimum spacing ≈ 0.85 * 1 bit
    energy_gate_frac: float = 0.10    # V must exceed this * median(V) to count
    refine_halfchip: float = 0.5      # refine center ± half-chip for best metric

def detect_bits_from_bank(scores: dict[str, np.ndarray],
                          fs: float,
                          Lchip: int,
                          settings: BankDetectSettings = BankDetectSettings()):
    """
    Use the filter bank to (1) find bit-center peaks, (2) align the clock,
    and (3) decode bits (0/1) at those centers.

    Chip→tone mapping is implicit in the templates:
      chip '0' → f0   |  chip '1' → f1
    Center-of-bit templates (valid):
      bit 0 (10): '010' or '110'
      bit 1 (01): '001' or '101'
    '000','111','011','100' act as "invalid" (boundary/no-center) suppressors.
    """
    N = len(next(iter(scores.values())))
    Lbit = 2 * Lchip

    # Combine valid + invalid score maps
    V0 = np.maximum(scores['010'], scores['110'])  # best for bit 0 centers
    V1 = np.maximum(scores['001'], scores['101'])  # best for bit 1 centers
    I  = np.maximum.reduce([scores['000'], scores['111'], scores['011'], scores['100']])

    V = np.maximum(V0, V1) - I  # center-likeness
    # suppress candidates that look like bit-boundaries: strong neighbor at ±Lchip
    neighbor = np.maximum(shift_edge(V, +Lchip), shift_edge(V, -Lchip))
    center_metric = V - settings.neighbor_alpha * neighbor

    # Energy/scale gate (robust to overall gain)
    med = np.median(center_metric[center_metric > 0]) if np.any(center_metric > 0) else 0.0
    thr = settings.energy_gate_frac * (med + 1e-12)
    cm_gated = np.where(center_metric >= thr, center_metric, 0.0)

    # Peak picking with ~bit spacing
    min_dist = max(1, int(round(settings.min_peak_dist_bits * Lbit)))
    peaks, props = sp_signal.find_peaks(cm_gated, distance=min_dist)

    # Refine each peak within ± half-chip to the local max of center_metric
    r = int(round(settings.refine_halfchip * Lchip))
    centers = []
    for p in peaks:
        a = max(0, p - r)
        b = min(N, p + r + 1)
        if b <= a: 
            continue
        loc = a + int(np.argmax(center_metric[a:b]))
        centers.append(loc)

    centers = np.array(sorted(set(centers)), dtype=int)

    # Decide bit values at the refined centers (0 if V0>=V1 else 1)
    bits = []
    for c in centers:
        b = 0 if V0[c] >= V1[c] else 1
        bits.append(b)
    bits = np.array(bits, dtype=int)

    # Clock estimate from center spacing
    if len(centers) >= 2:
        Pbit_est = int(round(np.median(np.diff(centers))))
    else:
        Pbit_est = Lbit
    Pchip_est = Pbit_est // 2

    return {
        "centers": centers,
        "rx_bits": bits,
        "center_metric": center_metric,
        "V0": V0, "V1": V1, "I": I,
        "Pbit_est": Pbit_est,
        "Pchip_est": Pchip_est
    }

# ───────────────────────────────────────────────────────────
# Simple FSK modulator for testing
# ───────────────────────────────────────────────────────────
def manchester_encode(bits: np.ndarray) -> np.ndarray:
    enc = np.empty(bits.size * 2, dtype=int)
    enc[0::2] = 1 - bits
    enc[1::2] = bits
    return enc

def fsk_modulate(bits: np.ndarray, fs: float, baud: float, f0: float, f1: float, manchester: bool = True):
    chips = manchester_encode(bits) if manchester else bits
    chip_rate = baud * 2 if manchester else baud
    Lchip = int(fs / chip_rate)
    freq_stream  = np.where(chips == 0, f0, f1)
    freq_samples = np.repeat(freq_stream, Lchip)
    phase = 2 * np.pi * np.cumsum(freq_samples) / fs
    sig = np.sin(phase)
    return sig, Lchip, chips

# ───────────────────────────────────────────────────────────
# Plots
# ───────────────────────────────────────────────────────────
def plot_bank_results(sig: np.ndarray, fs: float, result: dict, title_prefix="3-chip filter bank"):
    t = np.arange(len(sig)) / fs
    centers = result["centers"]
    Pbit = result["Pbit_est"]
    Lchip = result["Pchip_est"]

    # Make a bit-clock overlay aligned to the detected centers
    clk = np.zeros_like(sig)
    if len(centers) > 0:
        # build a square wave that flips each chip, centered on the bit centers
        # start half-bit before first center so the first flip lands on the first center
        start = max(0, centers[0] - Pbit // 2)
        k = start
        state = 0.0
        while k < len(sig):
            next_k = min(len(sig), k + Lchip)
            clk[k:next_k] = state
            state = 1.0 - state
            k = next_k

    pos_max = float(np.max(sig))
    overlay = clk * (pos_max * 1.12)

    # 1) Time-domain with clock & centers
    fig1, ax1 = plt.subplots(figsize=(12, 4.6))
    ax1.plot(t, sig, lw=1.0, label="FSK signal")
    ax1.step(t, overlay, where="post", lw=1.0, alpha=0.9, label="Chip clock (aligned)")
    for c in centers:
        ax1.axvline(c / fs, color='k', linestyle='--', alpha=0.35)
    ax1.set_title(f"{title_prefix}: time signal + clock + centers")
    ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3); ax1.legend(loc="upper right")
    plt.tight_layout(); plt.show()

    # 2) Center metric & components
    fig2, ax2 = plt.subplots(figsize=(12, 4.6))
    ax2.plot(t, result["center_metric"], lw=1.0, label="center_metric = max(V0,V1) - I")
    ax2.plot(t, result["V0"], lw=0.8, alpha=0.8, label="V0 (best of 010/110)")
    ax2.plot(t, result["V1"], lw=0.8, alpha=0.8, label="V1 (best of 001/101)")
    ax2.plot(t, result["I"],  lw=0.8, alpha=0.8, label="Invalid max (000/111/011/100)")
    for c in centers:
        ax2.axvline(c / fs, color='k', linestyle='--', alpha=0.3)
    ax2.set_title(f"{title_prefix}: filter responses & detected centers")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Correlation magnitude")
    ax2.grid(True, alpha=0.3); ax2.legend(loc="upper right", ncols=2, fontsize=9)
    plt.tight_layout(); plt.show()

# ───────────────────────────────────────────────────────────
# Demo / usage
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Parameters
    fs   = 44_100
    baud = 100
    f0, f1 = 1_000, 3_000

    # Example payload (LSB-first unpack)
    np.random.seed(0)
    byte_vals = np.array([0x55, 0x3F], dtype=np.uint8)
    tx_bits = np.unpackbits(byte_vals, bitorder='little').astype(int)

    # Generate test FSK (Manchester)
    sig, Lchip, chips = fsk_modulate(tx_bits, fs, baud, f0, f1, manchester=True)

    # Build 3-chip filter bank and correlate (FFT)
    bank   = build_filter_bank(fs, Lchip, f0, f1)
    scores = correlate_bank_fft(sig, bank)

    # Detect centers + bits from bank outputs
    settings = BankDetectSettings(
        neighbor_alpha=0.5,
        min_peak_dist_bits=0.85,
        energy_gate_frac=0.10,
        refine_halfchip=0.5
    )
    result = detect_bits_from_bank(scores, fs, Lchip, settings)

    # Report & plots
    print("Estimated samples per bit (from centers):", result["Pbit_est"])
    print("Estimated samples per chip:", result["Pchip_est"])
    print("TX bits:     ", tx_bits.tolist())
    print("RX bits (FB):", result["rx_bits"].tolist())

    plot_bank_results(sig, fs, result,
                      title_prefix=f"3-chip filter bank ({f0}/{f1} Hz, baud={baud})")
