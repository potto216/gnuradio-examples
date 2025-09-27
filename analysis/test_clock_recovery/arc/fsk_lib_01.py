#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import signal as sp_signal

# ───────────────────────────────────────
# Manchester helpers
# ───────────────────────────────────────
def manchester_encode(bits: np.ndarray) -> np.ndarray:
    encoded = np.empty(bits.size * 2, dtype=int)
    encoded[0::2] = 1 - bits  # 0→10, 1→01
    encoded[1::2] = bits
    return encoded

# ───────────────────────────────────────
# Modulator
# ───────────────────────────────────────
def fsk_modulate(bits: np.ndarray, fs: float, baud: float, f0: float, f1: float, manchester: bool = True):
    chips = manchester_encode(bits) if manchester else bits
    chip_rate        = baud * 2 if manchester else baud
    samples_per_chip = int(fs / chip_rate)
    freq_stream  = np.where(chips == 0, f0, f1)
    freq_samples = np.repeat(freq_stream, samples_per_chip)
    phase = 2 * np.pi * np.cumsum(freq_samples) / fs
    waveform = np.sin(phase)
    return waveform, samples_per_chip, chips

# ───────────────────────────────────────
# Utilities (edge-safe smoothing & correlation)
# ───────────────────────────────────────
def moving_avg_edge(x: np.ndarray, L: int) -> np.ndarray:
    if L <= 1: return x
    pad_left = L // 2
    pad_right = L - 1 - pad_left
    xpad = np.pad(x, (pad_left, pad_right), mode='edge')
    h = np.ones(L) / L
    return np.convolve(xpad, h, mode='valid')

def correlate_same_edge(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Centered cross-correlation with edge padding:
      y[n] ≈ sum_k x[n + k - center] * template[k]
    """
    L = len(template)
    pad_left = L // 2
    pad_right = L - 1 - pad_left
    xpad = np.pad(x, (pad_left, pad_right), mode='edge')
    # scipy.signal.correlate centers the template for 'valid' on the padded signal
    return sp_signal.correlate(xpad, template, mode='valid')

def shift_edge(x: np.ndarray, k: int) -> np.ndarray:
    """Shift by k with edge replication (not zero)."""
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

# ───────────────────────────────────────
# Noncoherent tone envelopes
# ───────────────────────────────────────
def tone_envelope_noncoherent(sig: np.ndarray, fs: float, f: float, lp_len: int) -> np.ndarray:
    n = np.arange(len(sig))
    w = 2 * np.pi * f / fs
    i = sig * np.cos(w * n)
    q = sig * (-np.sin(w * n))  # exp(-jwt)
    i_lp = moving_avg_edge(i, lp_len)
    q_lp = moving_avg_edge(q, lp_len)
    return np.sqrt(i_lp**2 + q_lp**2)

# ───────────────────────────────────────
# Demod settings
# ───────────────────────────────────────
@dataclass
class DemodSettings:
    lp_len: int = None            # default: spp_chip//2 (set later)
    invalid_margin: float = 0.15  # valid - invalid ≥ margin*|valid|
    energy_factor: float = 0.75   # valid^2 ≥ energy_factor * local_energy
    center_alpha: float = 0.85    # neighbor suppression weight at ±Lchip
    peak_distance_bits: float = 0.95
    normalize_corr: bool = True
    refine_halfchip: float = 0.5  # search ±half chip for local max refinement
    fill_search_halfchip: float = 0.6  # comb-fit search window in chips

# ───────────────────────────────────────
# Demod: 4 matched filters + center-of-bit selection + refinement + comb-fit
# ───────────────────────────────────────
def demodulate_matched_filters(sig: np.ndarray,
                               fs: float,
                               baud: float,
                               f0: float,
                               f1: float,
                               samples_per_chip: int,
                               settings: DemodSettings = DemodSettings()):
    Lchip = samples_per_chip
    Lbit  = 2 * Lchip
    n = len(sig)

    if settings.lp_len is None:
        settings.lp_len = max(4, Lchip // 2)

    # 1) Noncoherent envelopes & decision variable
    env0 = tone_envelope_noncoherent(sig, fs, f0, settings.lp_len)
    env1 = tone_envelope_noncoherent(sig, fs, f1, settings.lp_len)
    d    = env1 - env0

    # 2) Two-chip templates (valid & invalid)
    t10 = np.r_[ np.ones(Lchip), -np.ones(Lchip) ]   # bit 0 center
    t01 = -t10                                       # bit 1 center
    t11 = np.ones(Lbit)                              # invalid (no transition)
    t00 = -np.ones(Lbit)                             # invalid (no transition)

    if settings.normalize_corr:
        t10 /= np.linalg.norm(t10); t01 /= np.linalg.norm(t01)
        t11 /= np.linalg.norm(t11); t00 /= np.linalg.norm(t00)

    # Cross-correlations, properly centered
    c10 = correlate_same_edge(d, t10)
    c01 = correlate_same_edge(d, t01)
    c11 = correlate_same_edge(d, t11)
    c00 = correlate_same_edge(d, t00)

    valid   = np.maximum(c10, c01)
    invalid = np.maximum(c11, c00)

    # 3) Local energy gate
    energy_ma = moving_avg_edge(sig**2, Lbit)
    local_thr = settings.energy_factor * energy_ma  # per-sample threshold

    # 4) Center-of-bit score that suppresses boundaries (neighbors at ±Lchip)
    neighbor = np.maximum(shift_edge(valid, +Lchip), shift_edge(valid, -Lchip))
    center_score = valid - settings.center_alpha * neighbor

    # Initial peaks ~ one bit apart
    min_dist = int(max(1, settings.peak_distance_bits * Lbit))
    centers, _ = sp_signal.find_peaks(center_score, distance=min_dist)

    # 5) Refine centers to local argmax of VALID corr (±half chip), then re-check
    r = int(round(settings.refine_halfchip * Lchip))
    refined = []
    for c in centers:
        w0 = max(0, c - r); w1 = min(n, c + r + 1)
        if w1 <= w0: 
            continue
        # maximize the valid correlation, not just center_score
        loc = w0 + int(np.argmax(valid[w0:w1]))
        # energy + invalid guards
        v0, v1 = c10[loc], c01[loc]
        vmax   = max(v0, v1); vinv = invalid[loc]
        if (vmax >= vinv + settings.invalid_margin * abs(vmax)) and ((valid[loc]**2) >= local_thr[loc]):
            refined.append(loc)

    centers = np.array(sorted(set(refined)), dtype=int)

    # 6) Comb-fit fill: follow median bit spacing and add missing last/first centers
    if len(centers) >= 2:
        mu = int(round(np.median(np.diff(centers))))
        mu = int(np.clip(mu, int(0.8*Lbit), int(1.2*Lbit)))
    else:
        mu = Lbit

    half_win = int(round(settings.fill_search_halfchip * Lchip))

    def try_fill_at(c_exp: int) -> int | None:
        w0 = max(0, c_exp - half_win); w1 = min(n, c_exp + half_win + 1)
        if w1 <= w0:
            return None
        # choose best by valid correlation
        loc = w0 + int(np.argmax(valid[w0:w1]))
        if any(abs(loc - c) < min_dist for c in centers):
            return None
        v0, v1 = c10[loc], c01[loc]
        vmax   = max(v0, v1); vinv = invalid[loc]
        if (vmax >= vinv + settings.invalid_margin * abs(vmax)) and ((valid[loc]**2) >= local_thr[loc]):
            return loc
        return None

    if len(centers) > 0:
        # forward fill
        last = int(centers[-1])
        while last + int(0.7*mu) < n - Lchip:
            cand = try_fill_at(last + mu)
            if cand is None: break
            centers = np.append(centers, cand)
            centers.sort()
            last = int(centers[-1])

        # backward fill
        first = int(centers[0])
        while first - int(0.7*mu) > Lchip:
            cand = try_fill_at(first - mu)
            if cand is None: break
            centers = np.append(centers, cand)
            centers.sort()
            first = int(centers[0])

    # 7) Decide bits at final centers (after refinement/fill)
    rx_bits = []
    bit_centers = []
    for idx in centers:
        v0, v1 = c10[idx], c01[idx]
        vmax   = max(v0, v1); vinv = invalid[idx]
        if (vmax >= vinv + settings.invalid_margin * abs(vmax)) and ((valid[idx]**2) >= local_thr[idx]):
            bit = 0 if v0 >= v1 else 1
            rx_bits.append(bit)
            bit_centers.append(idx)

    return {
        "env0": env0, "env1": env1, "d": d,
        "c10": c10, "c01": c01, "c11": c11, "c00": c00,
        "valid": valid, "invalid": invalid,
        "center_score": center_score,
        "bit_centers": np.array(bit_centers, dtype=int),
        "rx_bits": np.array(rx_bits, dtype=int),
        "samples_per_bit": Lbit,
        "energy_ma": energy_ma,
    }

# ───────────────────────────────────────
# Plot helpers (unchanged from your previous version)
# ───────────────────────────────────────
def upsample_steps(values: np.ndarray, samples_per_symbol: int, n_target: int) -> np.ndarray:
    up = np.repeat(values.astype(float), samples_per_symbol)
    if len(up) < n_target:
        up = np.pad(up, (0, n_target - len(up)))
    else:
        up = up[:n_target]
    return up

def plot_fsk_with_overlays(sig, fs, baud, tx_bits, chips, samples_per_chip, title="2-FSK with Bit & Manchester Overlays"):
    n = len(sig); t = np.arange(n)/fs
    pos_max = float(np.max(sig))
    level_bits  = pos_max * 1.05 if pos_max > 0 else 1.05
    level_chips = pos_max * 1.15 if pos_max > 0 else 1.15
    chips_per_bit   = len(chips) // len(tx_bits)
    samples_per_bit = samples_per_chip * chips_per_bit
    bits_up  = upsample_steps(tx_bits, samples_per_bit, n)
    chips_up = upsample_steps(chips,   samples_per_chip, n)
    y_bits  = level_bits  * bits_up
    y_chips = level_chips * chips_up
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t, sig, lw=1.0, label="Received / 2-FSK waveform")
    ax.step(t, y_bits,  where='post', lw=1.2, label="TX bits (overlay)")
    ax.step(t, y_chips, where='post', lw=1.0, alpha=0.9, label="Manchester chips (overlay)")
    ymin = min(np.min(sig) * 1.1, -1.1); ymax = max(level_chips * 1.1,  1.1)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title); ax.set_xlabel("Time [s]"); ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", ncols=3, fontsize=9)
    plt.tight_layout(); plt.show()

def plot_demod_results(sig, fs, demod, samples_per_chip, title_prefix="Matched-filter demod"):
    n = len(sig); t = np.arange(n)/fs
    Lchip = samples_per_chip; Lbit = 2*Lchip

    # 1) MF outputs + center score
    fig1, ax1 = plt.subplots(figsize=(12, 5.2))
    ax1.plot(t, sig, lw=0.9, label="Received signal")
    ax1.plot(t, demod["c10"], lw=1.0, label="MF: valid 10 (bit=0)")
    ax1.plot(t, demod["c01"], lw=1.0, label="MF: valid 01 (bit=1)")
    ax1.plot(t, demod["c11"], lw=1.0, label="MF: invalid 11")
    ax1.plot(t, demod["c00"], lw=1.0, label="MF: invalid 00")
    ax1.plot(t, demod["center_score"], lw=1.0, alpha=0.9, label="Center score")
    for c in demod["bit_centers"]:
        ax1.axvline(c / fs, color='k', alpha=0.25, linestyle='--')
    ax1.set_title(f"{title_prefix}: MF outputs, center-score (centers dashed)")
    ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Amplitude / Correlation")
    ax1.grid(True, alpha=0.3); ax1.legend(loc="upper right", ncols=3, fontsize=9)
    plt.tight_layout(); plt.show()

    # 2) Recovered steps
    rx_bits = demod["rx_bits"]; centers = demod["bit_centers"]
    rec_bits_step  = np.zeros(n); rec_chips_step = np.zeros(n)
    for bit, c in zip(rx_bits, centers):
        start = int(max(0, c - Lbit // 2))
        mid   = int(min(n, c))
        end   = int(min(n, c + Lbit // 2))
        if start < mid:
            rec_chips_step[start:mid] = 1 if bit == 0 else 0  # bit=0→"10"
        if mid < end:
            rec_chips_step[mid:end]   = 0 if bit == 0 else 1
        if start < end:
            rec_bits_step[start:end]  = bit

    pos_max = float(np.max(sig))
    level_bits  = pos_max * 1.05 if pos_max > 0 else 1.05
    level_chips = pos_max * 1.15 if pos_max > 0 else 1.15
    y_bits  = level_bits  * rec_bits_step
    y_chips = level_chips * rec_chips_step

    fig2, ax2 = plt.subplots(figsize=(12, 4.8))
    ax2.plot(t, sig, lw=0.9, label="Received signal")
    ax2.step(t, y_bits,  where='post', lw=1.2, label="Recovered bits (overlay)")
    ax2.step(t, y_chips, where='post', lw=1.0, alpha=0.9, label="Recovered chips (overlay)")
    for c in centers:
        ax2.axvline(c / fs, color='k', alpha=0.25, linestyle='--')
    ymin = min(np.min(sig) * 1.1, -1.1); ymax = max(level_chips * 1.1, 1.1)
    ax2.set_ylim(ymin, ymax)
    ax2.set_title(f"{title_prefix}: recovered bits & chips (centers dashed)")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3); ax2.legend(loc="upper right", ncols=3, fontsize=9)
    plt.tight_layout(); plt.show()

    # 3) Decision variable 'd' with recovered bits
    fig3, ax3 = plt.subplots(figsize=(12, 4.8))
    ax3.plot(t, sig, lw=0.9, alpha=0.6, label="Received signal")
    ax3.plot(t, demod["d"], lw=1.2, label="Decision variable (d = env1-env0)")
    ax3.step(t, y_bits, where='post', lw=1.2, label="Recovered bits (overlay)")
    ax3.axhline(0, color='k', lw=0.8, alpha=0.4, linestyle='--')
    for c in centers:
        ax3.axvline(c / fs, color='k', alpha=0.25, linestyle='--')
    ymin = min(np.min(sig) * 1.1, np.min(demod["d"])*1.1, -1.1)
    ymax = max(np.max(sig) * 1.1, np.max(demod["d"])*1.1, 1.1)
    ax3.set_ylim(ymin, ymax)
    ax3.set_title(f"{title_prefix}: decision variable & recovered bits")
    ax3.set_xlabel("Time [s]"); ax3.set_ylabel("Amplitude")
    ax3.grid(True, alpha=0.3); ax3.legend(loc="upper right", ncols=3, fontsize=9)
    plt.tight_layout(); plt.show()

# ───────────────────────────────────────
# Demo
# ───────────────────────────────────────
if __name__ == "__main__":
    fs   = 44_100
    baud = 100
    f0, f1 = 1_000, 3_000

    np.random.seed(0)
    byte_vals = np.array([0x55, 0x3F], dtype=np.uint8)   # your example
    tx_bits = np.unpackbits(byte_vals, bitorder='little').astype(int)

    sig, spp_chip, chips = fsk_modulate(tx_bits, fs, baud, f0, f1, manchester=True)

    # Visualize TX overlays
    plot_fsk_with_overlays(sig, fs, baud, tx_bits, chips, spp_chip,
                           title=f"2-FSK ({f0} Hz / {f1} Hz), baud={baud}, fs={fs}")

    # Demod
    settings = DemodSettings(
        lp_len=None,
        invalid_margin=0.15,
        energy_factor=0.75,
        center_alpha=0.85,       # robust default
        peak_distance_bits=0.95,
        normalize_corr=True,
        refine_halfchip=0.5,
        fill_search_halfchip=0.6
    )
    demod = demodulate_matched_filters(sig, fs, baud, f0, f1, spp_chip, settings)
    plot_demod_results(sig, fs, demod, spp_chip,
                       title_prefix=f"MF demod ({f0}/{f1} Hz, baud={baud})")

    print("TX bits:     ", tx_bits.tolist())
    print("RX bits (MF):", demod["rx_bits"].tolist())
