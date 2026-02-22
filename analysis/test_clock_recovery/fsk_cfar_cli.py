#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None

import fsk_baseline as fsk
from fsk_cfar import analyze_cfar_window, ca_cfar_alpha
from fsk_common import now_iso_utc, read_wav, save_json


def _to_sample(v_time: float | None, fs: int, default: int) -> int:
    if v_time is None:
        return default
    return int(round(v_time * fs))


def _analysis_bounds(args, nsamp: int, fs: int) -> tuple[int, int, int]:
    start = args.start_sample if args.start_sample is not None else _to_sample(args.start_time, fs, 0)
    end = args.end_sample if args.end_sample is not None else _to_sample(args.end_time, fs, nsamp)

    if args.center_sample is not None or args.center_time is not None:
        center = args.center_sample if args.center_sample is not None else _to_sample(args.center_time, fs, 0)
        span = args.span_samples if args.span_samples is not None else _to_sample(args.span_time, fs, fs)
        half = max(1, span // 2)
        start = max(0, center - half)
        end = min(nsamp, center + half)
    else:
        center = (start + end) // 2

    start = max(0, min(start, nsamp))
    end = max(start + 1, min(end, nsamp))
    center = max(start, min(center, end - 1))
    return start, end, center


def run_analysis(args):
    fs_file, x = read_wav(Path(args.wav))
    fs = fs_file if args.fs <= 0 else args.fs
    if fs != fs_file:
        print(f"[WARN] WAV fs={fs_file}; using --fs={fs} for analysis math.")

    cfg = fsk.FSKConfig(fs=fs, baud=args.baud, f0=args.f0, f1=args.f1)
    meta = fsk.check_params(cfg)

    start, end, center = _analysis_bounds(args, x.size, fs)
    x_slice = x[start:end]
    xa = fsk.analytic_signal_fft(x_slice)

    Nw = args.win_symbols * meta.sps
    hop = args.hop_symbols * meta.sps
    if Nw > xa.size:
        raise ValueError("Analysis slice shorter than CFAR window. Increase span or reduce win-symbols.")

    k0 = int(round(cfg.f0 * Nw / fs))
    k1 = int(round(cfg.f1 * Nw / fs))

    rows = []
    idx = 0
    while idx + Nw <= xa.size:
        seg = xa[idx : idx + Nw]
        P = (np.fft.fft(seg) * np.conj(np.fft.fft(seg))).real
        cfar = analyze_cfar_window(P, k0, k1, args.guard_bins, args.pfa, m_sig=1, threshold_scale=args.threshold_scale)
        center_idx = start + idx + (Nw // 2)
        rows.append({
            "idx_start": start + idx,
            "idx_end": start + idx + Nw,
            "idx_center": center_idx,
            "time_center_s": center_idx / fs,
            "stat": cfar["stat"],
            "threshold": cfar["threshold"],
            "alpha": cfar["alpha"],
            "noise_mean": cfar["noise_mean"],
            "e0": cfar["e0"],
            "e1": cfar["e1"],
            "detect_mean": float(np.mean(cfar["detection_vals"])),
            "guard_mean": float(np.mean(cfar["guard_vals"])) if cfar["guard_vals"].size else 0.0,
            "noise_band_mean": float(np.mean(cfar["noise_vals"])) if cfar["noise_vals"].size else 0.0,
            "fft_power": P,
            "cfar": cfar,
        })
        idx += hop

    Z = np.stack([r["fft_power"] for r in rows], axis=0).T
    t = np.array([r["time_center_s"] for r in rows])
    bins = np.arange(Z.shape[0])
    use_frequency_axis = bool(getattr(args, "frequency_axis", False))

    # NOTE: Plotly heatmaps can fail to render (or render as blank) if the coordinate
    # axis is non-monotonic. np.fft.fftfreq() returns frequencies in wrap-around order
    # [0..+..,-..], so when plotting in Hz we fftshift the frequency axis and reorder Z
    # accordingly.
    if use_frequency_axis:
        heatmap_y = np.fft.fftshift(np.fft.fftfreq(Z.shape[0], d=1.0 / fs))
        Z_heatmap = np.fft.fftshift(Z, axes=0)
        heatmap_y_label = "Frequency (Hz)"
    else:
        heatmap_y = bins
        Z_heatmap = Z
        heatmap_y_label = "FFT Bin"

    centers = np.array([r["idx_center"] for r in rows])
    focus_i = int(np.argmin(np.abs(centers - center)))
    focus = rows[focus_i]

    base = Path(args.out_base) if args.out_base else Path(args.wav).with_suffix(".cfar")
    base.parent.mkdir(parents=True, exist_ok=True)

    summary_json = {
        "timestamp": now_iso_utc(),
        "wav": args.wav,
        "analysis_region": {
            "start_sample": start,
            "end_sample": end,
            "center_sample": center,
            "start_time_s": start / fs,
            "end_time_s": end / fs,
            "center_time_s": center / fs,
        },
        "config": {
            "fs": fs,
            "baud": args.baud,
            "f0": args.f0,
            "f1": args.f1,
            "pfa": args.pfa,
            "guard_bins": args.guard_bins,
            "win_symbols": args.win_symbols,
            "hop_symbols": args.hop_symbols,
            "threshold_scale": args.threshold_scale,
        },
        "focus_window": {
            "window_index": focus_i,
            "idx_start": focus["idx_start"],
            "idx_end": focus["idx_end"],
            "idx_center": focus["idx_center"],
            "stat": focus["stat"],
            "threshold": focus["threshold"],
            "alpha": focus["alpha"],
            "noise_mean": focus["noise_mean"],
        },
        "num_windows": len(rows),
    }

    save_json(base.with_suffix(".json"), summary_json)

    with open(base.with_suffix(".windows.csv"), "w") as fh:
        fh.write("win_idx,idx_start,idx_end,idx_center,time_center_s,stat,threshold,alpha,noise_mean,e0,e1,detect_mean,guard_mean,noise_band_mean\n")
        for i, r in enumerate(rows):
            fh.write(
                f"{i},{r['idx_start']},{r['idx_end']},{r['idx_center']},{r['time_center_s']:.9f},{r['stat']:.9f},{r['threshold']:.9f},{r['alpha']:.9f},{r['noise_mean']:.9f},{r['e0']:.9f},{r['e1']:.9f},{r['detect_mean']:.9f},{r['guard_mean']:.9f},{r['noise_band_mean']:.9f}\n"
            )

    cfar = focus["cfar"]
    with open(base.with_suffix(".focus_bins.csv"), "w") as fh:
        fh.write("bin,power,band\n")
        for b, v in zip(cfar["noise_bins"], cfar["noise_vals"]):
            fh.write(f"{int(b)},{float(v):.9f},noise\n")
        for b, v in zip(cfar["guard_bins"], cfar["guard_vals"]):
            fh.write(f"{int(b)},{float(v):.9f},guard\n")
        for b, v in zip(cfar["detection_bins"], cfar["detection_vals"]):
            fh.write(f"{int(b)},{float(v):.9f},detect\n")

    if args.plots != "none" and go is not None and make_subplots is not None:
        fig = make_subplots(rows=4, cols=1, vertical_spacing=0.08, subplot_titles=(
            "CFAR statistic vs threshold",
            "CFAR FFT Bin Power vs Time",
            "Focus window FFT bins by CFAR band",
            "alpha vs Pfa",
        ))
        fig.add_trace(go.Scatter(x=t, y=[r["stat"] for r in rows], name="stat", mode="lines+markers"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=[r["threshold"] for r in rows], name="threshold", mode="lines+markers"), row=1, col=1)
        fig.add_vline(x=focus["idx_center"] / fs, line_dash="dot", line_color="black", row=1, col=1)

        fig.add_trace(
            go.Heatmap(
                x=t,
                y=heatmap_y,
                z=Z_heatmap,
                colorbar=dict(title="Power (linear)"),
                name="fft_power",
                hovertemplate=(
                    "Time: %{x:.6f} s<br>"
                    + ("Frequency: %{y:.2f} Hz<br>" if use_frequency_axis else "FFT Bin: %{y:d}<br>")
                    + "Power: %{z:.6g}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(go.Scatter(x=cfar["noise_bins"], y=cfar["noise_vals"], name="noise", mode="markers", marker=dict(size=5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=cfar["guard_bins"], y=cfar["guard_vals"], name="guard", mode="markers", marker=dict(size=6)), row=3, col=1)
        fig.add_trace(go.Scatter(x=cfar["detection_bins"], y=cfar["detection_vals"], name="detect", mode="markers", marker=dict(size=9, symbol="diamond")), row=3, col=1)

        pfa_vals = np.logspace(np.log10(args.pfa_min), np.log10(args.pfa_max), args.pfa_points)
        ntrain = int(cfar["noise_bins"].size)
        fig.add_trace(
            go.Scatter(
                x=pfa_vals,
                y=[ca_cfar_alpha(p, ntrain, m_sig=1) for p in pfa_vals],
                name="alpha m=1",
                mode="lines+markers",
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=pfa_vals,
                y=[ca_cfar_alpha(p, ntrain, m_sig=2) for p in pfa_vals],
                name="alpha m=2",
                mode="lines+markers",
            ),
            row=4,
            col=1,
        )
        fig.update_xaxes(type="log", row=4, col=1, title_text="Pfa")
        fig.update_yaxes(title_text="alpha", row=4, col=1)
        fig.update_xaxes(title_text="time (s)", row=1, col=1)
        fig.update_yaxes(title_text="power (linear)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text=heatmap_y_label, row=2, col=1)
        fig.update_xaxes(title_text="FFT bin", row=3, col=1)
        fig.update_yaxes(title_text="FFT bin power (linear)", row=3, col=1)
        fig.update_layout(title="FSK CFAR Bin Power Over Time", height=1300)

        if args.plots == "html":
            fig.write_html(str(base.with_suffix(".plot.html")), include_plotlyjs="cdn", full_html=True)
        else:
            fig.write_image(str(base.with_suffix(".plot.png")), scale=2)

    print(f"Analysis outputs written with base: {base}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CFAR analysis CLI for FSK packet detector")
    p.add_argument("--wav", required=True)
    p.add_argument("--out-base", default=None)
    p.add_argument("--fs", type=int, default=0, help="Override sample rate (0 = WAV rate)")
    p.add_argument("--baud", type=int, default=100)
    p.add_argument("--f0", type=float, default=1000.0)
    p.add_argument("--f1", type=float, default=2000.0)
    p.add_argument("--pfa", type=float, default=1e-3)
    p.add_argument("--win-symbols", type=int, default=2)
    p.add_argument("--hop-symbols", type=int, default=1)
    p.add_argument("--guard-bins", type=int, default=2)
    p.add_argument("--threshold-scale", type=float, default=10.0)

    p.add_argument("--start-time", type=float, default=None)
    p.add_argument("--end-time", type=float, default=None)
    p.add_argument("--center-time", type=float, default=None)
    p.add_argument("--span-time", type=float, default=0.5)
    p.add_argument("--start-sample", type=int, default=None)
    p.add_argument("--end-sample", type=int, default=None)
    p.add_argument("--center-sample", type=int, default=None)
    p.add_argument("--span-samples", type=int, default=None)

    p.add_argument("--frequency-axis", action="store_true", help="Use frequency (Hz) instead of FFT bin index for heatmap y-axis")

    p.add_argument("--pfa-min", type=float, default=1e-7)
    p.add_argument("--pfa-max", type=float, default=1e-1)
    p.add_argument("--pfa-points", type=int, default=60)
    p.add_argument("--plots", choices=["none", "html", "png"], default="html")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_analysis(args)


if __name__ == "__main__":
    main()
