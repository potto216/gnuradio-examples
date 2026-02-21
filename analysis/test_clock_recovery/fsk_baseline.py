# fsk_baseline.py
# Baseline narrowband 2-FSK modulator & offline packet detector/demod with Plotly viz.
# Author: ChatGPT (GNURadioSound project)
#
# Design goals:
# - Real-valued 2-FSK (f0, f1), default fs=44100, baud=100, 32-byte fixed payload (no header).
# - No Manchester. Optional short Hann edge taper (5–10%) per symbol.
# - Parameter checks (hard errors):
#     * Samples/symbol sps = fs/baud must be an integer.
#     * On-bin check: f0*N/fs and f1*N/fs (with N=sps) must be integers.
#     * Orthogonality: (f1 - f0) * Tsym must be an integer (m/Tsym spacing), Tsym = N/fs.
# - Guard time configurable; warn if too short.
# - Offline whole-packet detection/demod (not real-time).
# - Packet detection via sliding-window CFAR on |X(f0)|^2 + |X(f1)|^2 vs off-tone noise mean.
# - Demod via coherent per-symbol DFT (Goertzel-equivalent), timing search (coarse->fine).
# - Convert raw real input once to analytic signal (FFT Hilbert) at the start.
#
from __future__ import annotations

import math
import warnings
import logging
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Optional

from fsk_cfar import ca_cfar_alpha, analyze_cfar_window

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ----------------------------- Utility & checks ----------------------------- #

@dataclass
class FSKConfig:
    fs: int = 44100
    baud: int = 100
    f0: float = 1000.0
    f1: float = 2000.0
    amp: float = 0.9
    taper_frac: float = 0.1      # 5–10% typical; default 10%
    guard_time: float = 0.05     # seconds before & after
    payload_bytes: int = 32      # exactly 32 bytes per spec
    check_onbin: bool = True

@dataclass
class FSKMeta:
    sps: int
    Tsym: float
    Nsym: int
    guard_samples: int
    k0: int
    k1: int
    m_orth: int
    fs: int
    baud: int
    f0: float
    f1: float

def bytes_to_bits(data: bytes, msb_first: bool = True) -> np.ndarray:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes or bytearray")
    bits = []
    for b in data:
        if msb_first:
            bits.extend([(b >> (7 - i)) & 1 for i in range(8)])
        else:
            bits.extend([(b >> i) & 1 for i in range(8)])
    return np.array(bits, dtype=np.uint8)

def bits_to_bytes(bits: Iterable[int], msb_first: bool = True) -> bytes:
    bits_arr = np.array(list(bits), dtype=np.uint8)
    if len(bits_arr) % 8 != 0:
        raise ValueError("Number of bits must be a multiple of 8")
    out = bytearray()
    for i in range(0, len(bits_arr), 8):
        chunk = bits_arr[i:i + 8]
        b = 0
        if msb_first:
            for j in range(8):
                b |= int(chunk[j] & 1) << (7 - j)
        else:
            for j in range(8):
                b |= int(chunk[j] & 1) << j
        out.append(b)
    return bytes(out)

def _is_integer(x: float, tol: float = 1e-12) -> bool:
    return abs(x - round(x)) < tol

def check_params(cfg: FSKConfig) -> FSKMeta:
    # Samples per symbol must be integer
    sps_f = cfg.fs / cfg.baud
    if not _is_integer(sps_f):
        raise ValueError(
            f"sps = fs/baud = {cfg.fs}/{cfg.baud} = {sps_f:.6f} is not an integer. "
            "Choose fs and baud so that fs/baud is an exact integer."
        )
    sps = int(round(sps_f))
    Tsym = sps / cfg.fs

    # On-bin check for the symbol-length DFT (N = sps)
    k0_f = cfg.f0 * sps / cfg.fs
    k1_f = cfg.f1 * sps / cfg.fs
    if cfg.check_onbin and (not _is_integer(k0_f) or not _is_integer(k1_f)):
        raise ValueError(
            "On-bin check failed: f0*N/fs or f1*N/fs is not an integer with N = sps.\n"
            f"  f0*N/fs = {k0_f:.9f}, f1*N/fs = {k1_f:.9f}\n"
            "Pick f0, f1, fs, baud so tones land exactly on symbol DFT bins."
        )
    k0 = int(round(k0_f))
    k1 = int(round(k1_f))

    # Orthogonality check: (f1 - f0) * Tsym must be integer
    m_orth_f = (cfg.f1 - cfg.f0) * Tsym
    if not _is_integer(m_orth_f):
        raise ValueError(
            "Orthogonality check failed: (f1 - f0) * Tsym is not an integer.\n"
            f"  (f1 - f0) * Tsym = {m_orth_f:.9f}\n"
            "This ensures tones are orthogonal over a symbol."
        )
    m_orth = int(round(m_orth_f))

    # Guard-time sanity
    if cfg.guard_time < Tsym:
        warnings.warn(
            f"Guard time {cfg.guard_time:.6f}s is less than one symbol ({Tsym:.6f}s). "
            "This may be too short for robust separation between packets.",
            RuntimeWarning
        )

    meta = FSKMeta(
        sps=sps,
        Tsym=Tsym,
        Nsym=cfg.payload_bytes * 8,
        guard_samples=int(round(cfg.guard_time * cfg.fs)),
        k0=k0,
        k1=k1,
        m_orth=m_orth,
        fs=cfg.fs,
        baud=cfg.baud,
        f0=cfg.f0,
        f1=cfg.f1,
    )
    return meta

def make_edge_taper(N: int, frac: float) -> np.ndarray:
    """Return a per-symbol taper with Hann ramps of length int(frac*N) at both edges."""
    if frac <= 0:
        return np.ones(N, dtype=np.float64)
    L = int(round(frac * N))
    if L == 0:
        return np.ones(N, dtype=np.float64)
    if 2 * L > N:
        L = N // 2
    plateau = np.ones(N, dtype=np.float64)
    n = np.arange(L, dtype=np.float64)
    hann_rise = 0.5 * (1 - np.cos(np.pi * (n + 1) / (L + 1)))
    plateau[:L] = hann_rise
    plateau[-L:] = hann_rise[::-1]
    return plateau

# ----------------------------- Modulator ----------------------------- #

def modulate_fsk(
    data: bytes,
    cfg: FSKConfig = FSKConfig()
) -> Tuple[np.ndarray, FSKMeta]:
    """
    Modulate exactly 32 payload bytes into a real-valued 2-FSK packet with guard zeros.
    Each bit maps to one symbol: 0->f0, 1->f1. Phase is NOT continuous between symbols.
    Returns (signal_float32, meta).
    """
    if len(data) != cfg.payload_bytes:
        raise ValueError(f"Payload must be exactly {cfg.payload_bytes} bytes, got {len(data)}")

    meta = check_params(cfg)
    bits = bytes_to_bits(data, msb_first=True)
    N = meta.sps
    t = np.arange(N, dtype=np.float64) / meta.fs
    taper = make_edge_taper(N, cfg.taper_frac)

    # Build per-bit symbol and concatenate
    sig_syms = np.zeros((meta.Nsym, N), dtype=np.float64)
    for i, b in enumerate(bits):
        f = cfg.f1 if b else cfg.f0
        sig_syms[i, :] = cfg.amp * np.cos(2 * np.pi * f * t) * taper

    packet = sig_syms.reshape(-1)
    guard = np.zeros(meta.guard_samples, dtype=np.float64)
    tx = np.concatenate([guard, packet, guard]).astype(np.float32)
    return tx, meta

# ------------------------- Analytic signal (Hilbert) ------------------------ #

def analytic_signal_fft(x: np.ndarray) -> np.ndarray:
    """
    Compute analytic signal via FFT method (no SciPy dependency).
    For real x[n], returns complex y[n] whose FFT has zero negative freqs,
    doubled positive freqs, and (optionally) DC/Nyquist untouched.
    """
    x = np.asarray(x)
    N = x.size
    X = np.fft.fft(x)
    H = np.zeros(N, dtype=np.float64)
    if N % 2 == 0:
        H[0] = 1.0
        H[N // 2] = 1.0
        H[1:N // 2] = 2.0
    else:
        H[0] = 1.0
        H[1:(N + 1) // 2] = 2.0
    y = np.fft.ifft(X * H)
    return y.astype(np.complex128)

# --------------------------- Packet detection (CFAR) ------------------------ #

@dataclass
class DetectionResult:
    start_idx: int
    stop_idx: int
    centers: np.ndarray # center sample index of each CFAR window (relative to xa)
    stat: np.ndarray
    thresh: np.ndarray
    k0: int
    k1: int
    Nw: int
    hop: int
    packet_found: bool = False
    global_start: int = 0  # new field to track global start index

def _ca_cfar_alpha(pfa: float, Ntrain: int, m_sig: int=1) -> float:
    return ca_cfar_alpha(pfa, Ntrain, m_sig=m_sig)

def detect_packet(
    x_real: np.ndarray,
    cfg: FSKConfig,
    meta: Optional[FSKMeta] = None,
    pfa: float = 1e-1,
    win_symbols: int = 2,
    hop_symbols: int = 1,
    guard_bins: int = 2,
    global_start: int = 0
) -> Tuple[DetectionResult, np.ndarray]:
    """
    Sliding-window CFAR on |X(f0)|^2 + |X(f1)|^2 against noise mean excluding ±guard_bins
    around tone bins. Window length is win_symbols * sps and hop is hop_symbols * sps.
    Returns (DetectionResult, analytic_signal_used).
    """
    if meta is None:
        meta = check_params(cfg)

    # Convert ONCE to analytic (requirement)
    xa = analytic_signal_fft(np.asarray(x_real, dtype=np.float64))

    N = meta.sps
    Nw = win_symbols * N
    hop = hop_symbols * N
    if Nw > xa.size:
        raise ValueError("Detection window longer than signal. Provide a longer capture.")

    k0 = int(round(cfg.f0 * Nw / meta.fs))
    k1 = int(round(cfg.f1 * Nw / meta.fs))

    stat = []
    thr = []
    centers = []
    idx = 0
    while idx + Nw <= xa.size:
        seg = xa[idx:idx + Nw]
        X = np.fft.fft(seg)
        P = (X * np.conj(X)).real  # |X|^2

        cfar = analyze_cfar_window(
            P,
            k0=k0,
            k1=k1,
            guard_bins=guard_bins,
            pfa=pfa,
            m_sig=1,
            threshold_scale=10.0,
        )
        S = cfar["stat"]
        T = cfar["threshold"]
        noise_mean = cfar["noise_mean"]
        alpha = cfar["alpha"]
        # include idx also in seconds
        print(f"global idx_start={idx+global_start} ({(idx+global_start) / meta.fs:.3f}s) , global idx_end={idx + Nw + global_start} ({(idx + Nw + global_start) / meta.fs:.3f}s), stat(S)={S:.3f}, noise_mean={noise_mean:.6f}, thr(T)={T:.3f}, alpha={alpha:.3f}")

        stat.append(S)
        thr.append(T)
        centers.append(idx + Nw // 2)
        idx += hop

    stat = np.array(stat, dtype=np.float64)
    thr = np.array(thr, dtype=np.float64)
    centers = np.array(centers, dtype=np.int64)

    # Find a run above threshold; estimate span ~ packet length
    above = stat > thr
    runs = []
    i = 0
    packet_found = False
    # Expected packet length in samples
    Npkt = meta.Nsym * N
    # Expected number of CFAR windows that overlap the packet
    if Npkt <= Nw:
        expected_windows = 1
    else:
        expected_windows = ((Npkt - Nw) // hop) + 1
    # Require (e.g.) 90% of expected windows
    min_windows = max(1, int(math.ceil(0.9 * expected_windows)))

    while i < above.size and packet_found==False:
        if above[i]:
            j = i
            while j < above.size and above[j]:
                j += 1
            if (j - i) >= min_windows:
                packet_found = True
                runs.append((i, j))  # window index span [i, j)
            i = j
        else:
            i += 1
            
    if runs:
        i0, i1 = max(runs, key=lambda r: (centers[r[1]-1] - centers[r[0]]))
        start_idx_est = max(0, int(centers[i0] - Nw)) # for bose file
        #start_idx_est = max(0, int(centers[i0]+Nw//2)) # for loopback
        stop_idx_est = min(xa.size, start_idx_est + Npkt)
        print(f"Packet detected: windows [{i0}, {i1}), centers [{centers[i0]}, {centers[i1-1]}], "
              f"stat(S)={stat[i0]:.3f}, thr(T)={thr[i0]:.3f}")
    else:
        imax = int(np.argmax(stat))
        #start_idx_est = max(0, int(centers[imax] - Nw))
        start_idx_est = max(0, int(centers[imax]))
        stop_idx_est = min(xa.size, start_idx_est + Npkt)
        print("No packet detected.")

    det = DetectionResult(
        start_idx=start_idx_est,
        stop_idx=stop_idx_est,
        centers=centers,
        stat=stat,
        thresh=thr,
        k0=k0,
        k1=k1,
        Nw=Nw,
        hop=hop,
        packet_found=packet_found,
        global_start=global_start
    )
    return det, xa

# ------------------------------ Demodulation ------------------------------- #

@dataclass
class DemodResult:
    bits_hat: np.ndarray
    tau_best: int
    R0: np.ndarray  # complex correlation per symbol (best tau)
    R1: np.ndarray
    metric: float
    k0: int
    k1: int

def _dft_vectors(N: int, k_bins) -> Dict[int, np.ndarray]:
    n = np.arange(N, dtype=np.float64)
    vecs = {}
    for k in k_bins:
        vecs[k] = np.exp(-1j * 2 * np.pi * k * n / N)
    return vecs

def demodulate_from_start(
    xa: np.ndarray,
    cfg: FSKConfig,
    meta: Optional[FSKMeta],
    start_idx: int,
    refine: bool = True,
    max_lead_syms: int = 24,
) -> DemodResult:
    """
    Demodulate assuming 'start_idx' is near/before the packet start (from detect_packet).
    First search several whole-symbol offsets to skip any extra leading region, then search
    tau in [0, sps) (coarse-to-fine).
    """
    if meta is None:
        meta = check_params(cfg)

    N = meta.sps
    Npkt = meta.Nsym * N

    # Symbol-DFT exact bins
    k0 = meta.k0
    k1 = meta.k1
    v = _dft_vectors(N, [k0, k1])
    v0 = v[k0]
    v1 = v[k1]

    def metric_for_tau(base_idx: int, tau: int):
        R0 = np.empty(meta.Nsym, dtype=np.complex128)
        R1 = np.empty(meta.Nsym, dtype=np.complex128)
        absdiff = np.empty(meta.Nsym, dtype=np.float64)
        for i in range(meta.Nsym):
            a = base_idx + tau + i * N
            b = a + N
            if b > xa.size:
                w = np.zeros(N, dtype=np.complex128)
                w_part = xa[a:xa.size] if a < xa.size else np.array([], dtype=np.complex128)
                w[:w_part.size] = w_part
            else:
                w = xa[a:b]
            # For analytic x, use sum(w * exp(-j2πkn/N))
            R0[i] = np.dot(w, v0)
            R1[i] = np.dot(w, v1)
            absdiff[i] = abs(abs(R1[i]) - abs(R0[i]))
        score = float(np.sum(absdiff))
        return score, R0, R1

    # Coarse search over symbol offsets and tau
    coarse_step = max(1, N // 16)
    taus = list(range(0, N, coarse_step))
    best = (-1.0, start_idx, 0, None, None)  # (score, base_idx, tau, R0, R1)

    max_syms = max(0, int(max_lead_syms))
    for off_syms in range(0, max_syms + 1):
        base_idx = start_idx + off_syms * N
        if base_idx + Npkt <= xa.size:
            for tau in taus:
                score, R0, R1 = metric_for_tau(base_idx, tau)
                if score > best[0]:
                    best = (score, base_idx, tau, R0, R1)

    score_c, base_c, tau_c, R0_c, R1_c = best

    # Fine tau refine for the chosen base offset
    if refine:
        lo = max(0, tau_c - coarse_step)
        hi = min(N - 1, tau_c + coarse_step)
        best = (score_c, base_c, tau_c, R0_c, R1_c)
        for tau in range(lo, hi + 1):
            score, R0, R1 = metric_for_tau(base_c, tau)
            if score > best[0]:
                best = (score, base_c, tau, R0, R1)
        score_b, base_b, tau_b, R0_b, R1_b = best
    else:
        score_b, base_b, tau_b, R0_b, R1_b = score_c, base_c, tau_c, R0_c, R1_c

    # Decisions
    mags0 = np.abs(R0_b)
    mags1 = np.abs(R1_b)
    bits_hat = (mags1 > mags0).astype(np.uint8)

    return DemodResult(
        bits_hat=bits_hat,
        tau_best=tau_b,
        R0=R0_b,
        R1=R1_b,
        metric=score_b,
        k0=k0,
        k1=k1,
    )

# ------------------------------ High-level API ------------------------------ #

@dataclass
class DecodeOutput:
    det: DetectionResult
    dem: DemodResult
    meta: FSKMeta
    xa: np.ndarray

def decode_packet(
    x_real: np.ndarray,
    cfg: FSKConfig = FSKConfig(),
    pfa: float = 1e-3,
    win_symbols: int = 16,
    hop_symbols: int = 2,
    guard_bins: int = 2,
    global_start: int = 0
) -> DecodeOutput:
    """
    Convenience: analytic once, detect, then demod.
    """
    meta = check_params(cfg)
    det, xa = detect_packet(
        x_real=x_real,
        cfg=cfg,
        meta=meta,
        pfa=pfa,
        win_symbols=win_symbols,
        hop_symbols=hop_symbols,
        guard_bins=guard_bins,
        global_start=global_start
    )
    # print where the packet was detected
    # print(f"Packet detected from {det.start_idx} to {det.stop_idx}")
    dem = demodulate_from_start(xa, cfg, meta, det.start_idx, refine=True)
    return DecodeOutput(det=det, dem=dem, meta=meta, xa=xa)

# ------------------------------ Plotly Figures ------------------------------ #

def fig_time_with_bits(
    x_real: np.ndarray,
    fs: int,
    start_idx: int,
    sps: int,
    bits_hat: Optional[np.ndarray] = None,
    bits_true: Optional[np.ndarray] = None,
    title: str = "FSK Time-Domain Signal",
):
    if go is None:
        raise RuntimeError("Plotly is not available in this environment.")
    N = x_real.size
    t = np.arange(N) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x_real, name="signal", mode="lines"))
    ymax = float(np.max(np.abs(x_real))) if N > 0 else 1.0
    # Symbol boundaries (for decoded packet region)
    if bits_hat is not None:
        n_bits = len(bits_hat)
        for i in range(n_bits + 1):
            s = start_idx + i * sps
            if 0 <= s < N:
                fig.add_vline(x=s / fs, line=dict(width=1, dash="dot"))
        bit_track = np.repeat((2 * bits_hat - 1) * 0.7 * ymax, sps)
        overlay = np.zeros_like(x_real, dtype=float)
        end = min(N, start_idx + bit_track.size)
        overlay[start_idx:end] = bit_track[: end - start_idx]
        fig.add_trace(go.Scatter(x=t, y=overlay, name="bits (±)", mode="lines"))
    if bits_true is not None:
        bit_track2 = np.repeat((2 * bits_true - 1) * 0.4 * ymax, sps)
        overlay2 = np.zeros_like(x_real, dtype=float)
        end2 = min(N, start_idx + bit_track2.size)
        overlay2[start_idx:end2] = bit_track2[: end2 - start_idx]
        fig.add_trace(go.Scatter(x=t, y=overlay2, name="ground truth (±)", mode="lines"))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude")
    return fig

def fig_detection_metric(det,
                         fs: int,
                         x_real: np.ndarray,
                         *,
                         bits_hat: Optional[np.ndarray] = None,
                         bits_true: Optional[np.ndarray] = None,
                         sps: Optional[int] = None,
                         pkt_start: Optional[int] = None,
                         title: str = "Detection CFAR",
                         true_start: int | None = None,
                         true_stop: int | None = None,
                         bit_text_y_frac: float = 0.85,
                         det_overlay_time_adjust_sec: float = 0.0,
                         show_bit_text: bool = True,
                         show_bit_outline: bool = True):
    """
    bit_text_y_frac: vertical position for bit text as fraction of y-range (0=bottom, 1=top).
    det_overlay_time_adjust_sec: time offset in seconds for detection overlays.
    show_bit_text: whether to show bit value text annotations.
    show_bit_outline: whether to show bit box outlines.
    """
    if go is None or make_subplots is None:
        raise RuntimeError("Plotly is not available in this environment.")
    # Apply offset so detection times align to the full time-domain x-axis
    t_det = (det.centers + int(det.global_start)) / fs
    t_full = np.arange(x_real.size) / fs

    # Two rows, shared x-axis; top has secondary y for CFAR threshold
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{}]]
    )

    # Row 1: Detection statistic (primary Y)
    fig.add_trace(
        go.Scatter(
            x=t_det,
            y=det.stat,
            name="S = |X(f0)|^2+|X(f1)|^2",
            mode="lines+markers"
        ),
        row=1, col=1, secondary_y=False
    )

    # Row 1: CFAR threshold (secondary Y)
    fig.add_trace(
        go.Scatter(
            x=t_det,
            y=det.thresh,
            name="CFAR threshold",
            mode="lines"
        ),
        row=1, col=1, secondary_y=True
    )

    # Row 2: Time-domain signal
    fig.add_trace(
        go.Scatter(
            x=t_full,
            y=x_real,
            name="time signal",
            mode="lines"
        ),
        row=2, col=1
    )

    # Compute y position for bit text in row 2
    ymin = float(np.min(x_real)) if x_real.size > 0 else 0.0
    ymax = float(np.max(x_real)) if x_real.size > 0 else 1.0
    bit_text_y = ymin + bit_text_y_frac * (ymax - ymin)

    # Row 2: Per-bit background coloring (correct vs error) + bit text
    # Requires bits_true and bits_hat and sps and pkt_start
    if (bits_true is not None) and (bits_hat is not None) and (sps is not None) and (pkt_start is not None):
        n_bits = min(len(bits_true), len(bits_hat))
        N = x_real.size
        tmin = 0.0
        tmax = (N - 1) / fs if N > 0 else 0.0

        # Add legend proxies once (shapes don't appear in legend)
        legend_added = False
                
        for i in range(n_bits):
            a = pkt_start + i * sps
            b = a + sps
            if b <= 0 or a >= N:
                continue  # completely out of range
            a = max(0, a)
            b = min(N, b)
            t0 = a / fs
            t1 = b / fs
            if t1 <= t0:
                continue
            ok = (int(bits_hat[i]) == int(bits_true[i]))
            color = "rgba(0,255,0,0.7)" if ok else "rgba(255,0,0,0.7)"
            
            # Draw bit box with optional outline
            line_width = 1 if show_bit_outline else 0
            fig.add_vrect(
                x0=max(t0, tmin) + det_overlay_time_adjust_sec,
                x1=min(t1, tmax) + det_overlay_time_adjust_sec,
                fillcolor=color,
                opacity=0.7,
                line_width=line_width,
                line_color="black" if show_bit_outline else color,
                layer="below",
                row=2, col=1
            )
            
            # Add bit text at center of interval (if enabled)
            if show_bit_text:
                t_center = (t0 + t1) / 2.0
                bit_val = f"b[{i}] = {bits_hat[i]}"
                fig.add_annotation(
                    x=t_center + det_overlay_time_adjust_sec,
                    y=bit_text_y,
                    text=str(bit_val),
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    row=2, col=1
                )
            
            if not legend_added:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                         marker=dict(color="rgba(0,255,0,0.8)"),
                                         name="bit OK"),
                              row=2, col=1)
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                         marker=dict(color="rgba(255,0,0,0.7)"),
                                         name="bit error"),
                              row=2, col=1)
                legend_added = True

    # Detected packet region (both subplots) — shift by same offset
    fig.add_vrect(x0=(det.start_idx + det.global_start) / fs + det_overlay_time_adjust_sec,
                  x1=(det.stop_idx + det.global_start) / fs + det_overlay_time_adjust_sec,
                  fillcolor="LightGreen",
                  opacity=0.30,
                  line_width=0,
                  annotation_text="Detected",
                  annotation_position="top left",
                  row=1, col=1)

    fig.add_vrect(x0=(det.start_idx + det.global_start) / fs + det_overlay_time_adjust_sec,
                  x1=(det.stop_idx + det.global_start) / fs + det_overlay_time_adjust_sec,
                  fillcolor="LightGreen",
                  opacity=0.20,
                  line_width=0,
                  row=2, col=1)

    # Optional true packet region (both subplots)
    if (true_start is not None) and (true_stop is not None):
        fig.add_vrect(x0=true_start / fs,
                      x1=true_stop / fs,
                      fillcolor="LightBlue",
                      opacity=0.25,
                      line_width=0,
                      annotation_text="True",
                      annotation_position="top right",
                      layer="below",
                      row=1, col=1)
        fig.add_vrect(x0=true_start / fs,
                      x1=true_stop / fs,
                      fillcolor="LightBlue",
                      opacity=0.20,
                      line_width=0,
                      layer="below",
                      row=2, col=1)
        # Legend proxies
        fig.add_trace(go.Scatter(x=[None], y=[None],
                                 mode="markers",
                                 marker=dict(color="LightGreen"),
                                 name="Detected region"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=[None], y=[None],
                                 mode="markers",
                                 marker=dict(color="LightBlue"),
                                 name="True region"),
                      row=1, col=1)

    fig.update_layout(
        title=title,
        legend_title_text=None
    )
    # Axes titles
    fig.update_yaxes(title_text="Energy (S)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="CFAR threshold", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    return fig



def fig_spectrum(x: np.ndarray, fs: int, f0: float, f1: float, title: str = "Spectrum"):
    if go is None:
        raise RuntimeError("Plotly is not available in this environment.")
    N = int(2 ** math.ceil(math.log2(len(x)))) if len(x) > 0 else 1
    X = np.fft.rfft(x, n=N)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    P = 20 * np.log10(np.maximum(np.abs(X), 1e-12))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=P, name="|X(f)| (dB)", mode="lines"))
    fig.add_vline(x=f0, line=dict(color="firebrick", dash="dash"),
                  annotation_text="f0", annotation_position="top")
    fig.add_vline(x=f1, line=dict(color="royalblue", dash="dash"),
                  annotation_text="f1", annotation_position="top")
    fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)")
    return fig

def fig_symbol_magnitudes(dem, title: str = "Per-symbol Magnitudes"):
    if go is None:
        raise RuntimeError("Plotly is not available in this environment.")
    n = np.arange(len(dem.R0))
    m0 = np.abs(dem.R0)
    m1 = np.abs(dem.R1)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=n, y=m0, name="|R0| (f0)"))
    fig.add_trace(go.Bar(x=n, y=m1, name="|R1| (f1)"))
    fig.update_layout(barmode="group", title=title, xaxis_title="Symbol index", yaxis_title="Magnitude")
    return fig

