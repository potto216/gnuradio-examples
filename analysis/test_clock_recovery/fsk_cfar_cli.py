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


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.zeros_like(x, dtype=float)
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _exponential_pdf(x: np.ndarray, lam: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    if lam <= 0.0:
        return y
    mask = x >= 0.0
    y[mask] = lam * np.exp(-lam * x[mask])
    return y


def _hist_transform(vals: np.ndarray, use_db: bool) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    if not use_db:
        return arr
    eps = 1e-12
    return 10.0 * np.log10(arr + eps)


def _apply_hist_upper_limit(vals: np.ndarray, hist_max: float | None, hist_drop_above_max: bool) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    if hist_max is None:
        return arr
    if hist_drop_above_max:
        return arr[arr <= hist_max]
    return np.minimum(arr, hist_max)


def _fit_distribution_band(
    vals: np.ndarray,
    hist_bins: int,
    hist_density: bool,
    fit_models: str,
    hist_db: bool,
    hist_max: float | None,
    hist_drop_above_max: bool,
) -> dict:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"sample_count": 0}

    arr = _hist_transform(arr, hist_db)
    arr = _apply_hist_upper_limit(arr, hist_max, hist_drop_above_max)
    if arr.size == 0:
        return {
            "sample_count": 0,
            "histogram": {
                "bins": int(hist_bins),
                "density": bool(hist_density),
                "y_axis_mode": "percent" if hist_density else "count",
                "scale": "dB" if hist_db else "linear",
                "upper_limit": hist_max,
                "drop_above_limit": bool(hist_drop_above_max),
            },
        }

    hist_counts_raw, hist_edges = np.histogram(arr, bins=hist_bins, density=False)
    hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    hist_bin_width = float(hist_edges[1] - hist_edges[0]) if hist_edges.size > 1 else 0.0
    if hist_density:
        hist_counts = 100.0 * hist_counts_raw / float(arr.size)
    else:
        hist_counts = hist_counts_raw.astype(float)

    use_gaussian = fit_models in ("gaussian", "both")
    use_exponential = fit_models in ("exponential", "both")
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=0))

    result = {
        "sample_count": int(arr.size),
        "sample_mean": mu,
        "sample_std": sigma,
        "histogram": {
            "bins": int(hist_bins),
            "density": bool(hist_density),
            "y_axis_mode": "percent" if hist_density else "count",
            "scale": "dB" if hist_db else "linear",
            "upper_limit": hist_max,
            "drop_above_limit": bool(hist_drop_above_max),
            "counts": hist_counts.tolist(),
            "counts_raw": hist_counts_raw.tolist(),
            "bin_edges": hist_edges.tolist(),
            "bin_centers": hist_centers.tolist(),
            "bin_width": hist_bin_width,
        },
    }

    if use_gaussian:
        g_pdf = _gaussian_pdf(hist_centers, mu, sigma)
        g_y = g_pdf * hist_bin_width * 100.0 if hist_density else g_pdf * hist_bin_width * float(arr.size)
        result["gaussian"] = {
            "mu": mu,
            "sigma": sigma,
            "sse_hist_y": float(np.sum((hist_counts - g_y) ** 2)),
        }

    if use_exponential:
        pos = arr[arr >= 0.0]
        if pos.size > 0:
            mean_pos = float(np.mean(pos))
            lam = (1.0 / mean_pos) if mean_pos > 0 else 0.0
            exp_cdf = 1.0 - np.exp(-lam * np.sort(pos)) if lam > 0 else np.zeros(pos.size, dtype=float)
            emp_cdf = np.arange(1, pos.size + 1, dtype=float) / float(pos.size)
            ks_stat = float(np.max(np.abs(emp_cdf - exp_cdf))) if pos.size else 0.0
            e_pdf = _exponential_pdf(hist_centers, lam)
            e_y = e_pdf * hist_bin_width * 100.0 if hist_density else e_pdf * hist_bin_width * float(arr.size)
            result["exponential"] = {
                "lambda": float(lam),
                "domain": "x>=0",
                "nonnegative_sample_count": int(pos.size),
                "ks_statistic_nonnegative": ks_stat,
                "sse_hist_y": float(np.sum((hist_counts - e_y) ** 2)),
            }
        else:
            result["exponential"] = {
                "lambda": None,
                "domain": "x>=0",
                "nonnegative_sample_count": 0,
                "ks_statistic_nonnegative": None,
                "sse_hist_y": None,
            }

    return result


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
    focus_window_symbols = args.focus_win_symbols if args.focus_win_symbols is not None else args.win_symbols
    focus_nw = int(focus_window_symbols * meta.sps)
    if focus_nw <= 0:
        raise ValueError("Focus window must have positive length. Increase --focus-win-symbols.")
    if focus_nw > xa.size:
        raise ValueError("Focus window longer than analysis slice. Reduce --focus-win-symbols or increase the analysis span.")

    focus_center_rel = center - start
    focus_start_rel = max(0, min(focus_center_rel - (focus_nw // 2), xa.size - focus_nw))
    focus_end_rel = focus_start_rel + focus_nw
    focus_seg = xa[focus_start_rel:focus_end_rel]
    focus_power = (np.fft.fft(focus_seg) * np.conj(np.fft.fft(focus_seg))).real
    focus_k0 = int(round(cfg.f0 * focus_nw / fs))
    focus_k1 = int(round(cfg.f1 * focus_nw / fs))
    focus_cfar = analyze_cfar_window(
        focus_power,
        focus_k0,
        focus_k1,
        args.guard_bins,
        args.pfa,
        m_sig=1,
        threshold_scale=args.threshold_scale,
    )
    focus = {
        "window_index": focus_i,
        "idx_start": start + focus_start_rel,
        "idx_end": start + focus_end_rel,
        "idx_center": start + focus_start_rel + (focus_nw // 2),
        "stat": focus_cfar["stat"],
        "threshold": focus_cfar["threshold"],
        "alpha": focus_cfar["alpha"],
        "noise_mean": focus_cfar["noise_mean"],
        "fft_power": focus_power,
        "cfar": focus_cfar,
        "window_symbols": focus_window_symbols,
        "window_samples": focus_nw,
    }
    focus_axis = np.fft.fftfreq(focus_nw, d=1.0 / fs) if use_frequency_axis else np.arange(focus_nw)
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
            "focus_win_symbols": focus_window_symbols,
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
            "window_index": focus["window_index"],
            "idx_start": focus["idx_start"],
            "idx_end": focus["idx_end"],
            "idx_center": focus["idx_center"],
            "window_symbols": focus["window_symbols"],
            "window_samples": focus["window_samples"],
            "stat": focus["stat"],
            "threshold": focus["threshold"],
            "alpha": focus["alpha"],
            "noise_mean": focus["noise_mean"],
        },
        "distribution_fit_config": {
            "hist_bins": args.hist_bins,
            "hist_density": args.hist_density,
            "hist_scale": "dB" if args.hist_db else "linear",
            "hist_max": args.hist_max,
            "hist_drop_above_max": args.hist_drop_above_max,
            "fit_models": args.fit_models,
        },
        "num_windows": len(rows),
    }

    cfar = focus["cfar"]
    distribution_fits = {
        "noise_vals": _fit_distribution_band(cfar["noise_vals"], args.hist_bins, args.hist_density, args.fit_models, args.hist_db, args.hist_max, args.hist_drop_above_max),
        "guard_vals": _fit_distribution_band(cfar["guard_vals"], args.hist_bins, args.hist_density, args.fit_models, args.hist_db, args.hist_max, args.hist_drop_above_max),
        "detection_vals": _fit_distribution_band(cfar["detection_vals"], args.hist_bins, args.hist_density, args.fit_models, args.hist_db, args.hist_max, args.hist_drop_above_max),
    }
    summary_json["distribution_fits"] = distribution_fits

    save_json(base.with_suffix(".json"), summary_json)

    with open(base.with_suffix(".windows.csv"), "w") as fh:
        fh.write("win_idx,idx_start,idx_end,idx_center,time_center_s,stat,threshold,alpha,noise_mean,e0,e1,detect_mean,guard_mean,noise_band_mean\n")
        for i, r in enumerate(rows):
            fh.write(
                f"{i},{r['idx_start']},{r['idx_end']},{r['idx_center']},{r['time_center_s']:.9f},{r['stat']:.9f},{r['threshold']:.9f},{r['alpha']:.9f},{r['noise_mean']:.9f},{r['e0']:.9f},{r['e1']:.9f},{r['detect_mean']:.9f},{r['guard_mean']:.9f},{r['noise_band_mean']:.9f}\n"
            )

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

        if any(fit_data.get("sample_count", 0) > 0 for fit_data in distribution_fits.values()):
            fit_fig = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=("noise_vals distribution", "guard_vals distribution", "detection_vals distribution"),
                vertical_spacing=0.08,
            )
            band_order = ["noise_vals", "guard_vals", "detection_vals"]
            band_names = {"noise_vals": "noise", "guard_vals": "guard", "detection_vals": "detect"}
            for row_idx, band_key in enumerate(band_order, start=1):
                fit_data = distribution_fits[band_key]
                if fit_data.get("sample_count", 0) == 0:
                    continue
                hist = fit_data["histogram"]
                x_centers = np.array(hist["bin_centers"], dtype=float)
                y_hist = np.array(hist["counts"], dtype=float)
                bin_width = float(hist.get("bin_width", 0.0))

                fit_fig.add_trace(
                    go.Bar(
                        x=x_centers,
                        y=y_hist,
                        width=bin_width if bin_width > 0 else None,
                        name=f"{band_names[band_key]} hist",
                        opacity=0.45,
                    ),
                    row=row_idx,
                    col=1,
                )

                if "gaussian" in fit_data:
                    g = fit_data["gaussian"]
                    g_pdf = _gaussian_pdf(x_centers, g["mu"], g["sigma"])
                    g_y = g_pdf * bin_width * (100.0 if args.hist_density else float(fit_data["sample_count"]))
                    fit_fig.add_trace(go.Scatter(x=x_centers, y=g_y, mode="lines", name=f"{band_names[band_key]} gaussian"), row=row_idx, col=1)
                if "exponential" in fit_data and fit_data["exponential"].get("lambda") is not None:
                    e = fit_data["exponential"]
                    e_pdf = _exponential_pdf(x_centers, e["lambda"])
                    e_y = e_pdf * bin_width * (100.0 if args.hist_density else float(fit_data["sample_count"]))
                    fit_fig.add_trace(go.Scatter(x=x_centers, y=e_y, mode="lines", name=f"{band_names[band_key]} exponential"), row=row_idx, col=1)
                fit_fig.update_yaxes(title_text="Percent of samples (%)" if args.hist_density else "Count", row=row_idx, col=1)
                fit_fig.update_xaxes(title_text=("Power (dB)" if args.hist_db else "Power (linear)"), row=row_idx, col=1)

            fit_fig.update_layout(
                title="CFAR band distribution fits (focus window)",
                barmode="overlay",
                height=1000,
                margin=dict(t=90, b=70, l=80, r=60),
            )
            if args.plots == "html":
                distfit_path = base.with_suffix(".distfit.html")
                fit_fig.write_html(str(distfit_path), include_plotlyjs="cdn", full_html=True)
            else:
                distfit_path = base.with_suffix(".distfit.png")
                fit_fig.write_image(str(distfit_path), scale=2)
            print(f"Distribution-fit plot written: {distfit_path}")

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
    p.add_argument("--focus-win-symbols", type=int, default=None, help="Focus-window width in symbols for the focus plot and histogram analysis (defaults to --win-symbols)")
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
    p.add_argument("--hist-bins", type=int, default=50, help="Number of histogram bins for distribution-fit analysis")
    p.add_argument("--hist-density", action="store_true", help="Use density-normalized histogram values instead of raw counts")
    p.add_argument("--hist-db", action="store_true", help="Transform histogram distribution-fit values to dB before binning to compress dynamic range")
    p.add_argument("--hist-max", type=float, default=None, help="Upper limit for histogram values after any optional dB transform")
    p.add_argument("--hist-drop-above-max", action="store_true", help="Drop histogram values above --hist-max instead of saturating them down to the limit")
    p.add_argument(
        "--fit-models",
        choices=["none", "gaussian", "exponential", "both"],
        default="both",
        help="Which distribution models to fit on CFAR focus-window band samples",
    )
    args = p.parse_args(argv)
    if args.focus_win_symbols is not None and args.focus_win_symbols <= 0:
        raise ValueError("Invalid focus window: --focus-win-symbols must be positive.")
    if args.hist_max is not None and args.hist_drop_above_max and not np.isfinite(args.hist_max):
        raise ValueError("Invalid histogram maximum: --hist-max must be finite when used with --hist-drop-above-max.")
    if args.heatmap_zmin is not None and args.heatmap_zmax is not None and not (args.heatmap_zmin < args.heatmap_zmax):
        raise ValueError("Invalid heatmap range: --heatmap-zmin must be less than --heatmap-zmax.")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_analysis(args)


if __name__ == "__main__":
    main()
