#!/usr/bin/env python3
import argparse, json, sys, os, math, datetime
from pathlib import Path
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None
try:
    import plotly.io as pio
except ImportError:
    pio = None

import fsk_baseline as fsk


def _now_iso():
    # Use timezone-aware UTC (avoids deprecation warning for utcnow)
    ts = datetime.datetime.now(datetime.UTC)
    # Ensure trailing 'Z' (strip offset)
    return ts.isoformat().replace("+00:00", "Z")


def write_wav(path: Path, data: np.ndarray, fs: int):
    if sf is None:
        from scipy.io import wavfile
        wavfile.write(str(path), fs, data.astype(np.float32))
    else:
        sf.write(str(path), data.astype(np.float32), fs)


def read_wav(path: Path):
    if sf is None:
        from scipy.io import wavfile
        fs, x = wavfile.read(str(path))
        if x.dtype != np.float32:
            x = x.astype(np.float32) / np.max(np.abs(x))
    else:
        x, fs = sf.read(str(path), dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
    return fs, x.astype(np.float32)


def save_json(path: Path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def build_tx_signal(cfg: fsk.FSKConfig,
                    nbytes: int,
                    snr_db: float,
                    pad_pre_sec: float,
                    pad_post_sec: float,
                    seed: int | None,
                    hex_payload: str | None):
    rng = np.random.default_rng(seed)
    if hex_payload:
        payload = bytes.fromhex(hex_payload.replace(" ", ""))
        if len(payload) != nbytes:
            print(f"[WARN] hex payload length {len(payload)} != nbytes ({nbytes}), using actual length.")
            nbytes = len(payload)
    else:
        payload = rng.integers(0, 256, nbytes, dtype=np.uint8).tobytes()

    tx, meta = fsk.modulate_fsk(payload, cfg)

    # Add AWGN at requested SNR (per signal power)
    p_sig = float(np.mean(tx ** 2))
    sigma = math.sqrt(p_sig / (10 ** (snr_db / 10)))
    noise = sigma * rng.standard_normal(tx.shape)
    tx_noisy = tx + noise

    fs = cfg.fs
    n_pre = int(round(pad_pre_sec * fs))
    n_post = int(round(pad_post_sec * fs))
    noise_pre = sigma * rng.standard_normal(n_pre)
    noise_post = sigma * rng.standard_normal(n_post)
    full = np.concatenate([noise_pre, tx_noisy, noise_post])
    true_start = n_pre
    true_end = n_pre + tx.size
    return {
        "signal": full,
        "payload": payload,
        "true_start": true_start,
        "true_end": true_end,
        "sigma": sigma,
        "meta": meta
    }


def decode_one(rx: np.ndarray, cfg: fsk.FSKConfig, pfa: float):
    out = fsk.decode_packet(rx, cfg=cfg, pfa=pfa)
    bits_hat = out.dem.bits_hat
    return out, bits_hat


def slice_and_decode_all(rx: np.ndarray,
                         cfg: fsk.FSKConfig,
                         pfa: float,
                         max_packets: int | None,
                         min_gap_symbols: int = 2):
    """
    Repeatedly run detection+decode on remaining tail of signal.
    After each packet, advance index by detected length + gap.
    """
    # Derive samples-per-symbol (sps) from config (FSKConfig itself has no sps field)
    meta = fsk.check_params(cfg)
    sps = meta.sps
    packets = []
    cursor = 0
    total = rx.size
    gap = min_gap_symbols * sps
    count = 0
    global_start = 0
    while cursor < total and (max_packets is None or count < max_packets):
        sub = rx[cursor:]
        try:
            out = fsk.decode_packet(sub, cfg=cfg, pfa=pfa, global_start=cursor)
        except Exception:
            break
        start_rel = out.det.start_idx
        if start_rel >= sub.size:
            break
        packet_len_samples = len(out.dem.bits_hat) * out.meta.sps
        global_start = cursor + start_rel
        global_end = global_start + packet_len_samples
        # Print start/end in samples and seconds
        t_start = global_start / cfg.fs
        t_end = global_end / cfg.fs
        print(f"[decode] Packet {count}: samples [{global_start}, {global_end}) "
              f"time [{t_start:.6f}s, {t_end:.6f}s) dur={(t_end - t_start):.6f}s")
        
        packets.append({
            "out": out,
            "start": global_start,
            "end": global_end,
            "bits_hat": out.dem.bits_hat
        })
        count += 1
        cursor = global_end + gap
        if packet_len_samples <= 0:
            break
    return packets


def make_plots_tx(rx_full, tx_obj, cfg, base: Path, mode: str,
                  plot_kind: str, overwrite: bool):
    if pio is None or plot_kind == "none":
        return []
    outputs = []
    # Time plot with overlay
    fig_t = fsk.fig_time_with_bits(
        rx_full,
        cfg.fs,
        tx_obj["true_start"],
        cfg.sps,
        bits_hat=None,
        bits_true=fsk.bytes_to_bits(tx_obj["payload"]),
        title=f"TX waveform (mode={mode})"
    )
    if plot_kind == "html":
        path = base.with_suffix(".time.html")
        fig_t.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    else:
        path = base.with_suffix(".time.png")
        fig_t.write_image(str(path), scale=2)
    outputs.append(str(path))
    return outputs


def make_plots_rx(rx, cfg, packets, base: Path, plot_kind: str, plot_list: str = "det", truth_bits=None):
    if pio is None or plot_kind == "none":
        return []
    # Parse desired plots: comma-separated string (e.g., "det,time") or "all"
    if isinstance(plot_list, str):
        tokens = [t.strip().lower() for t in plot_list.split(",")] if plot_list else []
    else:
        tokens = [str(t).strip().lower() for t in plot_list]
    if not tokens:
        tokens = ["det"]
    valid = {"det", "time", "mag"}
    wanted = valid if ("all" in tokens) else {t for t in tokens if t in valid}

    out_paths = []
    for i, pkt in enumerate(packets):
        out = pkt["out"]
        suffix = f".p{i}"

        figs = {}
        if "det" in wanted:
            figs["det"] = fsk.fig_detection_metric(
                out.det,
                cfg.fs,
                rx,
                bits_hat=out.dem.bits_hat,
                bits_true=truth_bits,
                sps=out.meta.sps,
                pkt_start=pkt["start"],
                true_start=None,
                true_stop=None
            )
        if "time" in wanted:
            figs["time"] = fsk.fig_time_with_bits(
                rx,
                cfg.fs,
                pkt["start"],
                out.meta.sps,
                bits_hat=out.dem.bits_hat,
                bits_true=None,
                title=f"Packet {i} time-domain"
            )
        if "mag" in wanted:
            figs["mag"] = fsk.fig_symbol_magnitudes(out.dem)

        if plot_kind == "html":
            for label, fig in figs.items():
                path = base.with_suffix(f"{suffix}.{label}.html")                
                fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
                print(f"Wrote plot: {path}")
                out_paths.append(str(path))
        else:
            for label, fig in figs.items():
                path = base.with_suffix(f"{suffix}.{label}.png")
                fig.write_image(str(path), scale=2)
                print(f"Wrote plot: {path}")
                out_paths.append(str(path))
    return out_paths


def cmd_tx(args):
    cfg = fsk.FSKConfig(
        fs=args.fs,
        baud=args.baud,
        f0=args.f0,
        f1=args.f1,
    )
    tx_obj = build_tx_signal(cfg,
                             nbytes=args.nbytes,
                             snr_db=args.snr_db,
                             pad_pre_sec=args.pad_pre,
                             pad_post_sec=args.pad_post,
                             seed=args.seed,
                             hex_payload=args.hex)
    base = Path(args.out_base)
    wav_path = base.with_suffix(".wav")
    write_wav(wav_path, tx_obj["signal"], cfg.fs)

    payload_hex = tx_obj["payload"].hex()
    meta = tx_obj["meta"]
    info = {
        "mode": "tx",
        "timestamp": _now_iso(),
        "wav": str(wav_path),
        "config": {
            "fs": cfg.fs,
            "baud": cfg.baud,
            "f0": cfg.f0,
            "f1": cfg.f1,
            "nbytes": args.nbytes
        },
        "snr_db": args.snr_db,
        "pad_pre_sec": args.pad_pre,
        "pad_post_sec": args.pad_post,
        "true_start": tx_obj["true_start"],
        "true_end": tx_obj["true_end"],
        "payload_hex": payload_hex,
        "seed": args.seed,
        "sigma": tx_obj["sigma"],
        "samples": int(tx_obj["signal"].size),
        "sps": meta.sps
    }
    json_path = base.with_suffix(".json")
    save_json(json_path, info)

    plots = make_plots_tx(tx_obj["signal"], tx_obj, cfg, base, "tx",
                          args.plots, args.overwrite)
    if plots:
        info["plots"] = plots
        save_json(json_path, info)

    print(f"TX written: {wav_path}")
    print(f"JSON: {json_path}")


# Add this new helper function
def find_best_bit_shift(bits_true, bits_hat):
    """Find best circular shift using correlation."""
    # Convert bits 0/1 to -1/+1 for correlation
    bits_true_pm = 2 * np.array(bits_true, dtype=float) - 1
    bits_hat_pm = 2 * np.array(bits_hat, dtype=float) - 1
    
    # Ensure same length
    min_len = min(len(bits_true), len(bits_hat))
    bits_true_pm = bits_true_pm[:min_len]
    bits_hat_pm = bits_hat_pm[:min_len]
    
    # Calculate circular correlation for all possible shifts
    n = len(bits_true_pm)
    corr = np.zeros(n)
    for shift in range(n):
        # Circular shift
        shifted = np.roll(bits_hat_pm, shift)
        # Correlation = sum of products
        corr[shift] = np.sum(bits_true_pm * shifted)
    
    # Find best shift (maximum correlation)
    best_shift = np.argmax(corr)
    max_corr = corr[best_shift]
    
    # Calculate how many bits match at this shift
    shifted_bits = np.roll(bits_hat, best_shift)
    matches = np.sum(bits_true[:min_len] == shifted_bits[:min_len])
    match_rate = matches / min_len
    
    return {
        "best_shift": best_shift,
        "max_corr": float(max_corr),
        "matches": int(matches),
        "total_bits": min_len,
        "match_rate": float(match_rate)
    }


def cmd_rx(args):
    # Load truth data if provided
    truth_data = None
    if args.truth_json:
        try:
            with open(args.truth_json, "r") as f:
                truth_data = json.load(f)
            print(f"Loaded truth data from: {args.truth_json}")
        except Exception as e:
            print(f"Warning: Failed to load truth JSON: {e}")

    fs, x = read_wav(Path(args.wav))
    
    # Use configuration from truth file if available and requested
    if truth_data and args.use_truth_config:
        cfg_t = truth_data.get("config", {})
        fs = cfg_t.get("fs", fs) if args.fs == 0 else args.fs
        baud = cfg_t.get("baud", args.baud)
        f0 = cfg_t.get("f0", args.f0)
        f1 = cfg_t.get("f1", args.f1)
        print(f"Using config from truth file: fs={fs}, baud={baud}, f0={f0}, f1={f1}")
    else:
        if fs != args.fs and args.fs > 0:
            print(f"[WARN] WAV fs={fs} differs from provided --fs={args.fs}; using file fs.")
        baud, f0, f1 = args.baud, args.f0, args.f1
    
    cfg = fsk.FSKConfig(fs=fs, baud=baud, f0=f0, f1=f1)
    
    # Default out-base to wav basename (without extension) if not supplied
    if not args.out_base:
        args.out_base = str(Path(args.wav).with_suffix(""))
    
    packets = slice_and_decode_all(x, cfg, pfa=args.pfa,
                                   max_packets=args.max_pkts)
    base = Path(args.out_base)
    results = []
    
    # Get truth bits/bytes if available
    truth_bits = None
    truth_bytes = None
    truth_hex = None
    if truth_data and "payload_hex" in truth_data:
        truth_hex = truth_data["payload_hex"]
        truth_bytes = bytes.fromhex(truth_hex)
        truth_bits = fsk.bytes_to_bits(truth_bytes)
        print(f"Truth payload ({len(truth_bytes)} bytes): {truth_hex}")
    
    # Process each detected packet
    for i, pkt in enumerate(packets):
        out = pkt["out"]
        bits_hat = out.dem.bits_hat
        bytes_demod = fsk.bits_to_bytes(bits_hat)
        hex_rx = bytes_demod.hex()
        
        print(f"\nPacket {i}:")
        print(f"  Detected at samples [{pkt['start']}, {pkt['end']})")
        print(f"  Decoded ({len(bytes_demod)} bytes): {hex_rx}")
        
        # Compare with truth data if available
        truth_metrics = {}
        if truth_bits is not None:
            # Calculate bit error rate (BER)
            min_len = min(len(bits_hat), len(truth_bits))
            if min_len > 0:
                bit_errors = np.sum(bits_hat[:min_len] != truth_bits[:min_len])
                ber = float(bit_errors / min_len)
            else:
                bit_errors = 0
                ber = 1.0
                
            # Calculate byte error rate
            min_bytes = min(len(bytes_demod), len(truth_bytes))
            if min_bytes > 0:
                byte_errors = sum(1 for a, b in zip(bytes_demod, truth_bytes) if a != b)
                byte_error_rate = byte_errors / min_bytes
            else:
                byte_errors = 0
                byte_error_rate = 1.0
                
            # Add correlation analysis if there are any byte errors
            print(f"  Byte errors: {byte_errors}")
            if byte_errors > 0:
                print(f"\n  Checking for bit shifts (using circular correlation)...")
                corr_results = find_best_bit_shift(truth_bits, bits_hat)
                best_shift = corr_results["best_shift"]
                match_rate = corr_results["match_rate"]
                matches = corr_results["matches"]
                total = corr_results["total_bits"]
                
                print(f"  Best circular bit shift: {best_shift} bits")
                print(f"  After shift match rate: {match_rate:.4f} ({matches}/{total} bits)")
                
                # If match rate after shift is significantly better, likely a sync issue
                if match_rate > 0.95 and ber > 0.05:
                    print(f"  \033[93mPossible sync issue detected!\033[0m Bits match at {best_shift}-bit offset.")
                
                # Add to metrics
                truth_metrics["bit_correlation"] = {
                    "best_shift": best_shift,
                    "match_rate_after_shift": match_rate,
                    "matches_after_shift": matches,
                    "bits_compared_after_shift": total
                }
                        
                
            # Add to metrics
            truth_metrics = {
                "truth_hex": truth_hex,
                "ber": ber,
                "byte_error_rate": byte_error_rate,
                "bit_errors": int(bit_errors),
                "byte_errors": int(byte_errors),
                "bits_compared": int(min_len),
                "bytes_compared": int(min_bytes)
            }
            
            # Print comparison
            print(f"  BER: {ber:.6f} ({bit_errors}/{min_len} bits)")
            print(f"  Byte errors: {byte_errors}/{min_bytes} bytes")
            
            # Show visual difference (mark differing bytes)
            if min_bytes > 0:
                print("  Truth:    ", end="")
                for j in range(min_bytes):
                    byte_match = truth_bytes[j] == bytes_demod[j]
                    color = "" if byte_match else "\033[91m"  # Red for errors
                    end_color = "\033[0m" if not byte_match else ""
                    print(f"{color}{truth_bytes[j]:02x}{end_color}", end=" ")
                print()
                
                print("  Received: ", end="")
                for j in range(min_bytes):
                    byte_match = truth_bytes[j] == bytes_demod[j]
                    color = "" if byte_match else "\033[91m"  # Red for errors
                    end_color = "\033[0m" if not byte_match else ""
                    print(f"{color}{bytes_demod[j]:02x}{end_color}", end=" ")
                print()
        
        # Build packet entry
        entry = {
            "index": i,
            "detected_start": int(pkt["start"]),
            "detected_end": int(pkt["end"]),
            "num_bits": int(len(bits_hat)),
            "num_bytes": int(len(bytes_demod)),
            "bytes_hex": hex_rx,
            "sps": int(out.meta.sps),
            "symbol_freqs_hz": {
                "f0": cfg.f0,
                "f1": cfg.f1
            }
        }
        
        # Add truth comparison metrics if available
        if truth_metrics:
            entry["truth_comparison"] = truth_metrics
        
        results.append(entry)

    # Create overall results
    top = {
        "mode": "rx",
        "timestamp": _now_iso(),
        "wav": args.wav,
        "config": {
            "fs": cfg.fs,
            "baud": cfg.baud,
            "f0": cfg.f0,
            "f1": cfg.f1,
            "expected_nbytes": args.nbytes
        },
        "pfa": args.pfa,
        "packets_found": len(results),
        "packets": results
    }
    
    # Add truth reference if available
    if truth_data:
        top["truth_source"] = args.truth_json
        # Include key truth metadata
        if "true_start" in truth_data and "true_end" in truth_data:
            top["truth_metadata"] = {
                "true_start": truth_data["true_start"],
                "true_end": truth_data["true_end"],
                "payload_size": len(truth_bytes) if truth_bytes else None,
                "snr_db": truth_data.get("snr_db"),
                "pad_pre_sec": truth_data.get("pad_pre_sec"),
                "pad_post_sec": truth_data.get("pad_post_sec")
            }

    json_path = base.with_suffix(".json")
    save_json(json_path, top)

    # Pass truth_bits into plot builder so per-bit coloring can be applied
    plot_paths = make_plots_rx(x, cfg, packets, base, args.plots, plot_list="det", truth_bits=truth_bits)
    if plot_paths:
        top["plots"] = plot_paths
        save_json(json_path, top)

    print(f"\nRX complete. Packets: {len(results)}  JSON: {json_path}")
    if not packets:
        print("No packets decoded.")


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="FSK TX/RX CLI: generate or demodulate FSK wav packets."
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # TX
    tx = sub.add_parser("tx", help="Create FSK transmit WAV + JSON")
    tx.add_argument("--out-base", required=True,
                    help="Output base filename (no extension).")
    tx.add_argument("--fs", type=int, default=44100)
    tx.add_argument("--baud", type=int, default=100)
    tx.add_argument("--f0", type=float, default=1000)
    tx.add_argument("--f1", type=float, default=2000)
    tx.add_argument("--nbytes", type=int, default=32)
    tx.add_argument("--snr-db", type=float, default=96.0)
    tx.add_argument("--pad-pre", type=float, default=0.05,
                    help="Seconds of noise before packet.")
    tx.add_argument("--pad-post", type=float, default=0.05,
                    help="Seconds of noise after packet.")
    tx.add_argument("--seed", type=int, default=0)
    tx.add_argument("--hex", type=str, default=None,
                    help="Optional exact hex payload (overrides random payload).")
    tx.add_argument("--plots", choices=["none", "png", "html"], default="none",
                    help="Generate plots for TX signal.")
    tx.add_argument("--overwrite", action="store_true")
    tx.set_defaults(func=cmd_tx)

    # RX
    rx = sub.add_parser("rx", help="Demodulate one or more packets from a WAV")
    rx.add_argument("--wav", required=True, help="Input WAV file.")
    rx.add_argument("--truth-json", type=str, default=None,
                    help="Optional path to TX-generated JSON for ground truth comparison.")
    rx.add_argument("--use-truth-config", action="store_true",
                    help="Use configuration from truth JSON file if provided.")
    rx.add_argument("--out-base", default=None,
                    help="Output base filename (default: WAV basename without extension).")
    rx.add_argument("--fs", type=int, default=0,
                    help="Override sample rate (0 = use file).")
    rx.add_argument("--baud", type=int, default=100)
    rx.add_argument("--f0", type=float, default=1000)
    rx.add_argument("--f1", type=float, default=2000)
    rx.add_argument("--nbytes", type=int, default=32,
                    help="Expected payload size (for config; decoding scans independently).")
    rx.add_argument("--pfa", type=float, default=1e-3,
                    help="CFAR false alarm probability.")
    rx.add_argument("--max-pkts", type=int, default=None,
                    help="Limit number of packets to decode.")
    rx.add_argument("--plots", choices=["none", "png", "html"], default="none",
                    help="Generate plots for each decoded packet.")
    rx.set_defaults(func=cmd_rx)

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.func(args)


if __name__ == "__main__":
    main()