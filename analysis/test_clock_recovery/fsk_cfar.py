"""CFAR helpers shared between baseline decode and analysis tooling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CfarWindowBands:
    idx_start: int
    idx_end: int
    idx_center: int
    stat: float
    threshold: float
    alpha: float
    noise_mean: float
    e0: float
    e1: float
    detection_bins: np.ndarray
    guard_bins: np.ndarray
    noise_bins: np.ndarray
    detection_vals: np.ndarray
    guard_vals: np.ndarray
    noise_vals: np.ndarray


def ca_cfar_alpha(pfa: float, ntrain: int, m_sig: int = 1) -> float:
    if ntrain <= 0:
        return float("inf")
    ntrain_eff = max(4, min(ntrain, 256))
    return ntrain_eff * m_sig * (pfa ** (-1.0 / ntrain_eff) - 1.0)


def analyze_cfar_window(
    seg_fft_power: np.ndarray,
    k0: int,
    k1: int,
    guard_bins: int,
    pfa: float,
    m_sig: int = 1,
    threshold_scale: float = 10.0,
) -> dict:
    n = seg_fft_power.size
    detect_bins = np.array([k0, k1], dtype=np.int64)

    guard_mask = np.zeros(n, dtype=bool)
    for k in detect_bins:
        lo = max(k - guard_bins, 0)
        hi = min(k + guard_bins, n - 1)
        guard_mask[lo : hi + 1] = True

    noise_mask = ~guard_mask
    noise_mask[0] = False

    detection_vals = seg_fft_power[detect_bins]
    guard_vals = seg_fft_power[guard_mask]
    noise_vals = seg_fft_power[noise_mask]

    e0 = float(seg_fft_power[k0])
    e1 = float(seg_fft_power[k1])
    stat = e0 + e1
    noise_mean = float(np.mean(noise_vals)) if noise_vals.size else 1e-12
    ntrain = int(np.sum(noise_mask))
    alpha = ca_cfar_alpha(pfa, ntrain, m_sig=m_sig)
    threshold = alpha * noise_mean * threshold_scale

    return {
        "stat": stat,
        "threshold": float(threshold),
        "alpha": float(alpha),
        "noise_mean": noise_mean,
        "e0": e0,
        "e1": e1,
        "detection_bins": detect_bins,
        "guard_bins": np.flatnonzero(guard_mask),
        "noise_bins": np.flatnonzero(noise_mask),
        "detection_vals": detection_vals,
        "guard_vals": guard_vals,
        "noise_vals": noise_vals,
    }
