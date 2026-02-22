# Narrowband 2-FSK Offline TX/RX & Visualization

This directory contains a baseline 2-FSK modulator plus offline packet detection / demodulation utilities with Plotly visualization (bit overlays, CFAR metric, etc.).

## Features
- Real-valued 2-FSK (f0, f1) audio‑rate operation (default fs=44100 Hz, baud=100).
- Fixed payload length (default 32 bytes).
- Sliding-window CFAR detection using |X(f0)|² + |X(f1)|².
- Coherent per‑symbol DFT (Goertzel‑equivalent) demod with timing search.
- Plotly interactive figures:
  - Detection metric + CFAR threshold (dual y-axis) + packet region.
  - Time-domain waveform with per‑bit correctness coloring and optional bit text.
  - Symbol magnitudes.
- CLI for transmit (tx) and receive (rx) workflows.
- Flexible plotting controls (enable/disable overlays).

---

## 1. Installation

Create and activate a virtual Python environment (recommended) using uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install
uv venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv pip install --upgrade pip
uv pip install -r requirements.txt
```

(Optional) If you use a helper to configure a Chrome renderer:

```bash
plotly_get_chrome
```

---

## 2. Transmit (Generate a Test Packet)

Generate a synthetic FSK packet WAV file + JSON metadata with 100 milliseconds of blank before and :

```bash
python fsk_cli.py tx \
  --out-base packet_02 \
  --pad-pre 0.1 \
  --pad-post 0.5
```

Outputs:
- packet_02.wav (FSK waveform with leading/trailing silence)
- packet_02.json (configuration + payload hex)

You can now play this WAV through a speaker or transmit through another path, and record the result (loopback, microphone capture, etc.).

---

## 3. Record / Capture

Record the transmitted audio under different acoustic / channel conditions.  
Example resulting filenames (your setup may differ):

- packet_02_gnuradio_c01_clean_with_silence_rcv_bose_speaker.wav
- packet_02_gnuradio_c01_clean_with_silence_rcv_loopback.wav
- packet_02_gnuradio_c01_clean_with_silence_rcv_samsung_speaker.wav

---

## 4. Receive / Demodulate

Basic receive (no plots):

```bash
python fsk_cli.py rx \
  --wav packet_02_gnuradio_c01_clean_with_silence_rcv_bose_speaker.wav \
  --truth-json packet_02.json
```

### 4.1 Plot Generation

Enable HTML plots (interactive):

```bash
python ./analysis/test_clock_recovery/fsk_cli.py rx \
  --wav /home/user/data/audio/fsk/rx/packet_02_gnuradio_c01_clean_with_silence_rcv_bose_speaker.wav \
  --out-base /home/user/data/audio/fsk/rx/results/bose/packet_02_bit_overlay \
  --truth-json /home/user/data/audio/fsk/tx/packet_02.json \
  --plots html
```

python fsk_cli.py rx \
  --wav packet_02_gnuradio_c01_clean_with_silence_rcv_bose_speaker.wav \
  --out-base results/bose/packet_02_bit_overlay \
  --truth-json packet_02.json \
  --plots htmlls 


PNG export instead:

```bash
python fsk_cli.py rx ... --plots png
```

---

## 5. Bit Overlay & Timing Adjustment Options

The detection metric figure supports tuning of overlay alignment and visual layers.

| Option | Default | Description |
|--------|---------|-------------|
| `--det-time-adjust <secs>` | 0.0 | Shifts detection & bit overlay boxes horizontally (use if CFAR metric and waveform are offset). |
| `--no-bit-text` | (disabled by default) | Suppress per-bit text annotations (`b[i] = 0/1`). |
| `--no-bit-outline` | (disabled by default) | Remove rectangle outlines (keeps fill color). |
| `--no-bit-overlay` | (disabled by default) | Disable all bit-region coloring & text. |

If you do nothing, all overlays (boxes + outlines + text) are shown.

### 5.1 Computing a Good `--det-time-adjust`
You may observe a slight offset between where bits are colored and the apparent symbol energy in the waveform. This can happen if:
- The detection windowing created an internal start index.
- The recorded audio has leading silence trimming differences.

You can empirically compute the adjustment by comparing:
```
(det_time_reference) - (waveform_reference_samples / fs)
```
Example from the comment:
```
23.5 / 100.0  -  34.0 / 44100.0  ≈ 0.234229025
```
Here:
- 23.5/100.0 seconds might be a symbol or packet boundary derived from symbol timing (23.5 symbols at 100 baud).
- 34.0/44100.0 seconds is a raw sample index anchor in the full stream.

Pass that result into `--det-time-adjust`.

---

## 6. Example End-to-End Commands

Bit overlay (with alignment):

```bash
python fsk_cli.py rx \
  --wav packet_02_gnuradio_c01_clean_with_silence_rcv_bose_speaker.wav \
  --out-base results/bose/packet_02_bit_overlay \
  --det-time-adjust 0.234229025 \
  --truth-json packet_02.json \
  --plots html > results/bose/packet_02_bit_overlay.ansi
```

No bit overlay at all:

```bash
python fsk_cli.py rx \
  --wav packet_02_gnuradio_c01_clean_with_silence_rcv_bose_speaker.wav \
  --out-base results/bose/packet_02_no_bit_overlay \
  --truth-json packet_02.json \
  --no-bit-text --no-bit-outline --no-bit-overlay \
  --det-time-adjust 0.234229025 \
  --plots html > results/bose/packet_02_no_bit_overlay.ansi
```

Loopback capture:

```bash
python fsk_cli.py rx \
  --wav packet_02_gnuradio_c01_clean_with_silence_rcv_loopback.wav \
  --out-base results/loopback/packet_02_bit_overlay \
  --det-time-adjust 0.227857125 \
  --truth-json packet_02.json \
  --plots html > results/loopback/packet_02_bit_overlay.ansi
```

Samsung speaker capture:

```bash
python fsk_cli.py rx \
  --wav packet_02_gnuradio_c01_clean_with_silence_rcv_samsung_speaker.wav \
  --out-base results/samsung/packet_02_bit_overlay \
  --det-time-adjust 0.235861025 \
  --truth-json packet_02.json \
  --plots html > results/samsung/packet_02_bit_overlay.ansi
```

---

## 7. Output Artifacts
For each RX run (with plots enabled):
- `*.json`: Demod/decode report (bit errors, timing, config).
- `*.html` or `*.png`: Interactive or static plots:
  - Detection & CFAR
  - Time-domain with bit correctness overlay
  - Symbol magnitudes (if enabled)
- `*.ansi`: (Optional) console log redirection.

---

## 8. CFAR Deep-Dive (Using `fsk_cfar_cli.py`)

`fsk_cli.py rx` computes a CFAR-style detection metric as part of packet finding. If you want to inspect and tune CFAR behavior in isolation (window length, hop size, guard bins, Pfa, threshold scaling), use `fsk_cfar_cli.py`.

This tool slides a short FFT window over a chosen time region, computes:
- `stat = |X(f0)|^2 + |X(f1)|^2`
- A CA-CFAR threshold from the remaining FFT bins (excluding guard bins around the detection bins)

### 8.1 Quick Start

Analyze an entire WAV (or at least a long default span) and write outputs under `results/cfar/...`:

```bash
python fsk_cfar_cli.py \
  --wav packet_02_gnuradio_c01_clean_with_silence_rcv_loopback.wav \
  --out-base results/cfar/packet_02_loopback \
  --baud 100 \
  --f0 1000 \
  --f1 2000 \
  --plots html
```

Focus the analysis on a specific region (recommended). You can specify a time-centered region with `--center-time` + `--span-time`:

```bash
python fsk_cfar_cli.py \
  --wav packet_02_gnuradio_c01_clean_with_silence_rcv_loopback.wav \
  --out-base results/cfar/packet_02_loopback_centered \
  --center-time 0.30 \
  --span-time 0.80 \
  --win-symbols 2 \
  --hop-symbols 1 \
  --guard-bins 2 \
  --pfa 1e-3 \
  --threshold-scale 10.0 \
  --plots html
```

### 8.2 Common Options

CFAR / windowing:
- `--win-symbols`: FFT window length in symbols (default 2)
- `--hop-symbols`: step size between windows in symbols (default 1)
- `--guard-bins`: number of FFT bins excluded around each detection bin (default 2)
- `--pfa`: desired false alarm probability for CA-CFAR alpha calculation (default 1e-3)
- `--threshold-scale`: extra multiplier to make the threshold more/less conservative (default 10.0)

Analysis region selection (choose one style):
- `--start-time`, `--end-time` (seconds)
- `--start-sample`, `--end-sample`
- `--center-time` + `--span-time`
- `--center-sample` + `--span-samples`

Plots:
- `--plots none|html|png` (default `html`)
- `--frequency-axis`: Use frequency (Hz) instead of FFT bin index on the heatmap y-axis (one-sided: 0..Nyquist)
- `--heatmap-db`: Plot the heatmap in dB instead of linear

### 8.3 Outputs

Given `--out-base results/cfar/packet_02_loopback`, the tool writes:
- `results/cfar/packet_02_loopback.json`: Summary + chosen focus window
- `results/cfar/packet_02_loopback.windows.csv`: One row per sliding window (stat, threshold, etc.)
- `results/cfar/packet_02_loopback.focus_bins.csv`: FFT bins in the focus window labeled as `noise|guard|detect`
- `results/cfar/packet_02_loopback.plot.html` or `.plot.png`: CFAR plots (if enabled)

### 8.4 Notes / Gotchas
- If you see: `Analysis slice shorter than CFAR window`, increase `--span-time/--span-samples` or reduce `--win-symbols`.
- If your WAV header rate differs from the analysis rate you want to assume, use `--fs` to override.
- PNG export requires Plotly + kaleido; HTML export does not.

---

## 9. Tips / Troubleshooting
- If plots are blank: ensure Plotly installed (`pip show plotly`) and kaleido (for PNG).
- If bit overlays misalign: adjust `--det-time-adjust` gradually (e.g. ±0.01 s).
- If no packets are found: inspect detection plot (enable `--plots html`) to confirm thresholds vs energy.
- Large `--pad-pre` / `--pad-post` help avoid edge truncation during playback/recording.

---

## 10. Future Enhancements (Ideas)
- Adaptive timing refinement after first decode.
- Soft metrics export (LLR-like).
- Multi-packet auto batch decode + summary CSV.


