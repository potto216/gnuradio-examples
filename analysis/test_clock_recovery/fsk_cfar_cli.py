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
    n_fft = Z.shape[0]

    if use_frequency_axis:
        # Audio input is real-valued and we operate on an analytic signal; in both cases
        # the physically meaningful spectrum is the one-sided (nonnegative) frequencies.
        # Using a one-sided axis also guarantees a monotonic y-axis for Plotly.
        n_pos = (n_fft // 2) + 1
        heatmap_y = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        Z_heatmap = Z[:n_pos, :]
        heatmap_y_label = "Frequency (Hz)"
    else:
        heatmap_y = bins
        Z_heatmap = Z
        heatmap_y_label = "FFT Bin"

    if not getattr(args, "heatmap_linear", False):
        # Convert linear power to dB for better visibility of noise vs signal.
        # Avoid -inf by flooring at a small positive value.
        eps = 1e-12
        Z_vis = 10.0 * np.log10(Z_heatmap + eps)
        heatmap_units = "dB"
        heatmap_colorbar_title = "Power (dB)"
    else:
        Z_vis = Z_heatmap
        heatmap_units = "linear"
        heatmap_colorbar_title = "Power (linear)"

    focus_use_db = bool(getattr(args, "focus_db", False))
    if focus_use_db:
        eps = 1e-12

        def _focus_power(vals: np.ndarray) -> np.ndarray:
            return 10.0 * np.log10(vals + eps)

        focus_units = "dB"
        focus_y_label = "FFT bin power (dB)"
    else:

        def _focus_power(vals: np.ndarray) -> np.ndarray:
            return vals

        focus_units = "linear"
        focus_y_label = "FFT bin power (linear)"

    centers = np.array([r["idx_center"] for r in rows])
    focus_i = int(np.argmin(np.abs(centers - center)))
    focus = rows[focus_i]
    focus_axis = np.fft.fftfreq(n_fft, d=1.0 / fs) if use_frequency_axis else bins
    focus_axis_label = "Frequency (Hz)" if use_frequency_axis else "FFT Bin"
    focus_xaxis_range = [0.0, float(args.focus_freq_max_hz)] if use_frequency_axis else None

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
            "frequency_axis": args.frequency_axis,
            "heatmap_scale": "linear" if args.heatmap_linear else "dB",
            "focus_scale": "dB" if args.focus_db else "linear",
            "focus_region_overlay": args.focus_region_overlay,
            "include_stat_plot": not args.no_stat_plot,
            "include_heatmap_plot": not args.no_heatmap_plot,
            "include_focus_plot": not args.no_focus_plot,
            "include_alpha_plot": not args.no_alpha_plot,
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

    include_stat_plot = not args.no_stat_plot
    include_heatmap_plot = not args.no_heatmap_plot
    include_focus_plot = not args.no_focus_plot
    include_alpha_plot = not args.no_alpha_plot

    plot_sections = []
    if include_stat_plot:
        plot_sections.append(("stat", "CFAR statistic vs threshold", 0.18))
    if include_heatmap_plot:
        plot_sections.append(("heatmap", "CFAR FFT Bin Power vs Time", 0.46))
    if include_focus_plot:
        plot_sections.append(("focus", "Focus window FFT bins by CFAR band", 0.20))
    if include_alpha_plot:
        plot_sections.append(("alpha", "alpha vs Pfa", 0.16))

    if args.plots != "none" and go is not None and make_subplots is not None and plot_sections:
        total_weight = sum(weight for _, _, weight in plot_sections)
        row_numbers = {key: idx + 1 for idx, (key, _, _) in enumerate(plot_sections)}
        fig = make_subplots(
            rows=len(plot_sections),
            cols=1,
            row_heights=[weight / total_weight for _, _, weight in plot_sections],
            vertical_spacing=0.09,
            subplot_titles=tuple(title for _, title, _ in plot_sections),
        )

        if include_stat_plot:
            stat_row = row_numbers["stat"]
            fig.add_trace(go.Scatter(x=t, y=[r["stat"] for r in rows], name="stat", mode="lines+markers"), row=stat_row, col=1)
            fig.add_trace(go.Scatter(x=t, y=[r["threshold"] for r in rows], name="threshold", mode="lines+markers"), row=stat_row, col=1)
            fig.add_vline(x=focus["idx_center"] / fs, line_dash="dot", line_color="black", row=stat_row, col=1)
            fig.update_xaxes(title_text="Time (s)", row=stat_row, col=1)
            fig.update_yaxes(title_text="Power (linear)", row=stat_row, col=1)

        if include_heatmap_plot:
            heatmap_row = row_numbers["heatmap"]
            fig.add_trace(
                go.Heatmap(
                    x=t,
                    y=heatmap_y,
                    z=Z_vis,
                    colorscale=args.heatmap_colorscale,
                    zmin=args.heatmap_zmin,
                    zmax=args.heatmap_zmax,
                    colorbar=dict(title=heatmap_colorbar_title, x=1.08, len=0.46, y=0.63),
                    name="fft_power",
                    hovertemplate=(
                        "Time: %{x:.6f} s<br>"
                        + ("Frequency: %{y:.2f} Hz<br>" if use_frequency_axis else "FFT Bin: %{y:d}<br>")
                        + f"Power ({heatmap_units}): %{{z:.6g}}<extra></extra>"
                    ),
                ),
                row=heatmap_row,
                col=1,
            )
            fig.update_xaxes(title_text="Time (s)", row=heatmap_row, col=1)
            fig.update_yaxes(title_text=heatmap_y_label, row=heatmap_row, col=1)

            focus_start_t = focus["idx_start"] / fs
            focus_end_t = focus["idx_end"] / fs
            if args.focus_region_overlay == "lines":
                fig.add_vline(x=focus_start_t, line_dash="dash", line_color="white", row=heatmap_row, col=1)
                fig.add_vline(x=focus_end_t, line_dash="dash", line_color="white", row=heatmap_row, col=1)
            elif args.focus_region_overlay == "box":
                fig.add_vrect(
                    x0=focus_start_t,
                    x1=focus_end_t,
                    fillcolor="white",
                    opacity=0.18,
                    line_width=1,
                    line_dash="dash",
                    line_color="white",
                    row=heatmap_row,
                    col=1,
                )

        if include_focus_plot:
            focus_row = row_numbers["focus"]
            fig.add_trace(
                go.Scatter(
                    x=focus_axis[cfar["noise_bins"]],
                    y=_focus_power(cfar["noise_vals"]),
                    name="noise",
                    mode="markers",
                    marker=dict(size=5),
                    hovertemplate=(
                        f"{focus_axis_label}: %{{x:.6g}}<br>Power ({focus_units}): %{{y:.6g}}<extra>noise</extra>"
                        if use_frequency_axis
                        else f"FFT Bin: %{{x:d}}<br>Power ({focus_units}): %{{y:.6g}}<extra>noise</extra>"
                    ),
                ),
                row=focus_row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=focus_axis[cfar["guard_bins"]],
                    y=_focus_power(cfar["guard_vals"]),
                    name="guard",
                    mode="markers",
                    marker=dict(size=6),
                    hovertemplate=(
                        f"{focus_axis_label}: %{{x:.6g}}<br>Power ({focus_units}): %{{y:.6g}}<extra>guard</extra>"
                        if use_frequency_axis
                        else f"FFT Bin: %{{x:d}}<br>Power ({focus_units}): %{{y:.6g}}<extra>guard</extra>"
                    ),
                ),
                row=focus_row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=focus_axis[cfar["detection_bins"]],
                    y=_focus_power(cfar["detection_vals"]),
                    name="detect",
                    mode="markers",
                    marker=dict(size=9, symbol="diamond"),
                    hovertemplate=(
                        f"{focus_axis_label}: %{{x:.6g}}<br>Power ({focus_units}): %{{y:.6g}}<extra>detect</extra>"
                        if use_frequency_axis
                        else f"FFT Bin: %{{x:d}}<br>Power ({focus_units}): %{{y:.6g}}<extra>detect</extra>"
                    ),
                ),
                row=focus_row,
                col=1,
            )
            fig.update_xaxes(title_text=focus_axis_label, range=focus_xaxis_range, row=focus_row, col=1)
            fig.update_yaxes(title_text=focus_y_label, row=focus_row, col=1)

        if include_alpha_plot:
            alpha_row = row_numbers["alpha"]
            pfa_vals = np.logspace(np.log10(args.pfa_min), np.log10(args.pfa_max), args.pfa_points)
            ntrain = int(cfar["noise_bins"].size)
            fig.add_trace(
                go.Scatter(
                    x=pfa_vals,
                    y=[ca_cfar_alpha(p, ntrain, m_sig=1) for p in pfa_vals],
                    name="alpha m=1",
                    mode="lines+markers",
                ),
                row=alpha_row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=pfa_vals,
                    y=[ca_cfar_alpha(p, ntrain, m_sig=2) for p in pfa_vals],
                    name="alpha m=2",
                    mode="lines+markers",
                ),
                row=alpha_row,
                col=1,
            )
            fig.update_xaxes(type="log", row=alpha_row, col=1, title_text="Pfa")
            fig.update_yaxes(title_text="alpha", row=alpha_row, col=1)

        fig.update_layout(
            title="FSK CFAR Bin Power Over Time",
            height=max(420, 320 * len(plot_sections) + 80),
            margin=dict(t=120, b=90, l=80, r=180),
            legend=dict(orientation="h", x=0.0, y=1.08, xanchor="left", yanchor="bottom"),
        )

        for annotation in fig.layout.annotations:
            annotation["yshift"] = 8

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

    p.add_argument("--frequency-axis", action="store_true", help="Use frequency (Hz) instead of FFT bin index for the heatmap and focus-window y-axes")
    heatmap_scale = p.add_mutually_exclusive_group()
    heatmap_scale.add_argument("--heatmap-db", action="store_true", help="Plot CFAR bin-power heatmap in dB (10*log10); this is the default")
    heatmap_scale.add_argument("--heatmap-linear", action="store_true", help="Plot CFAR bin-power heatmap in linear power instead of the default dB view")
    focus_scale = p.add_mutually_exclusive_group()
    focus_scale.add_argument("--focus-db", action="store_true", help="Plot focus-window FFT bin power in dB (10*log10)")
    focus_scale.add_argument("--focus-linear", action="store_true", help="Plot focus-window FFT bin power in linear scale (default)")
    p.add_argument("--focus-freq-max-hz", type=float, default=10000.0, help="Upper x-axis limit in Hz for the focus plot when --frequency-axis is enabled")
    p.add_argument(
        "--focus-region-overlay",
        choices=["none", "lines", "box"],
        default="none",
        help="Overlay focus-window coverage on heatmap: none, boundary lines, or shaded box",
    )
    p.add_argument("--heatmap-colorscale", default="Viridis", help="Plotly colorscale name for CFAR heatmap")
    p.add_argument("--heatmap-zmin", type=float, default=None, help="Lower color scale bound for heatmap")
    p.add_argument("--heatmap-zmax", type=float, default=None, help="Upper color scale bound for heatmap")
    p.add_argument("--no-stat-plot", action="store_true", help="Skip the CFAR statistic vs threshold subplot")
    p.add_argument("--no-heatmap-plot", action="store_true", help="Skip the CFAR FFT Bin Power vs Time heatmap subplot")
    p.add_argument("--no-focus-plot", action="store_true", help="Skip the focus-window CFAR band subplot")
    p.add_argument("--no-alpha-plot", action="store_true", help="Skip the alpha vs Pfa subplot")

    p.add_argument("--pfa-min", type=float, default=1e-7)
    p.add_argument("--pfa-max", type=float, default=1e-1)
    p.add_argument("--pfa-points", type=int, default=60)
    p.add_argument("--plots", choices=["none", "html", "png"], default="html")
    args = p.parse_args(argv)
    if args.heatmap_zmin is not None and args.heatmap_zmax is not None and not (args.heatmap_zmin < args.heatmap_zmax):
        raise ValueError("Invalid heatmap range: --heatmap-zmin must be less than --heatmap-zmax.")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_analysis(args)


if __name__ == "__main__":
    main()
