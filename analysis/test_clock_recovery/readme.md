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

For a parameter-by-parameter guide that explains how the CLI settings relate to packet placement, sliding windows, guard bins, and threshold behavior, see `fsk_cfar_cli_parameter_guide.md`.

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

### 8.2 Command-Line Options, Units, and Tuning Notes

The CFAR CLI options mainly control four things:
- Which part of the recording gets analyzed
- How wide the sliding FFT windows are
- Which bins are treated as signal, guard, or noise
- How conservative the threshold becomes

#### Analysis region selection

Choose either an absolute range or a centered range:
- `--start-time`, `--end-time`: analysis bounds in seconds
- `--start-sample`, `--end-sample`: analysis bounds in samples
- `--center-time`, `--span-time`: centered analysis region in seconds
- `--center-sample`, `--span-samples`: centered analysis region in samples

Notes:
- `--span-time` is in seconds. `--span-samples` is in raw samples.
- `--center-time` / `--center-sample` with a span override the corresponding start/end settings in the implementation.
- If you omit explicit start/end and center/span, the tool defaults to a broad region based on the file length and default span.
- A centered region is usually the fastest way to inspect a suspected packet location.

Practical hints:
- Start with `--center-time` plus `--span-time` when you know roughly where the packet is.
- Make the span wide enough to include the full packet plus some silence or background on both sides. This helps the heatmap and threshold trace stay interpretable.
- If the selected region is too tight, the packet can appear clipped and the focus window may land on the wrong portion of the packet.
- If the region is too wide, plots become harder to read and tuning gets slower, but the detector still runs.

#### Signal and sample-rate options

- `--fs`: analysis sample rate in Hz. Default `0`, which means use the WAV header sample rate.
- `--baud`: symbol rate in symbols/second. Default `100`.
- `--f0`, `--f1`: target FSK tones in Hz. Defaults `1000` and `2000`.

Notes:
- `--fs` only changes the analysis math. It does not resample the WAV.
- `--baud` affects samples-per-symbol, which then affects both the window length and hop size in samples.
- `--f0` and `--f1` are converted to FFT bins using the current window length, so changing `--win-symbols`, `--baud`, or `--fs` can shift the exact detection bins.

Practical hints:
- Only override `--fs` when the WAV metadata is wrong or you intentionally want different analysis math.
- Keep `--baud` matched to the transmitter. A wrong baud rate changes window sizing and can make the detector look at the wrong bins.
- If the packet is visible in the heatmap but `stat` is weak, turn on `--frequency-axis --heatmap-db` and verify that the energy ridges line up with `--f0` and `--f1`.

#### Windowing and CFAR controls

- `--win-symbols`: FFT window length in symbols. Default `2`.
- `--hop-symbols`: step size between analysis windows in symbols. Default `1`.
- `--guard-bins`: number of FFT bins excluded on each side of each detection bin. Default `2`.
- `--pfa`: desired false alarm probability for CA-CFAR alpha calculation. Default `1e-3`.
- `--threshold-scale`: extra multiplier applied after the CFAR threshold is computed. Default `10.0`.

Units and side effects:
- `--win-symbols` and `--hop-symbols` are not seconds; they are converted to samples through the configured baud rate.
- Larger `--win-symbols` gives more energy averaging and often smoother traces, but it also smears packet edges in time.
- Smaller `--win-symbols` sharpens timing but can make the statistic noisier.
- Smaller `--hop-symbols` gives denser time sampling and smoother-looking curves, but increases the number of windows and output rows.
- Larger `--hop-symbols` runs faster, but can skip over short transitions or make the plots look coarse.
- Larger `--guard-bins` protects the noise estimate from tone leakage, but also removes more training bins from the CFAR noise pool.
- Smaller `--pfa` raises the threshold and makes detection more conservative.
- Larger `--pfa` lowers the threshold and makes detection easier to trigger.
- `--threshold-scale` is the quickest coarse sensitivity knob: higher is more conservative, lower is more permissive.

Practical hints:
- Start with `--win-symbols 2 --hop-symbols 1`. That is the current default tuning baseline.
- Increase `--win-symbols` to `3` or `4` if the statistic is too noisy.
- Keep `--hop-symbols 1` while tuning so you do not hide timing details.
- Increase `--guard-bins` if the focus-window plot shows strong energy spilling into neighboring bins around `f0` or `f1`.
- Lower `--threshold-scale` before changing `--pfa` when you just need a quick sensitivity adjustment.

#### Plot and output controls

- `--plots none|html|png`: output plot format. Default `html`.
- `--out-base`: output file prefix and directory.
- `--frequency-axis`: use frequency in Hz instead of FFT bin number on the heatmap and focus plot.
- `--heatmap-db`: render the heatmap in dB. This is the default visual behavior.
- `--heatmap-linear`: render the heatmap in linear power.
- `--focus-db`: render the focus-window spectrum in dB.
- `--focus-linear`: render the focus-window spectrum in linear power. This is the default.
- `--focus-freq-max-hz`: upper x-axis limit in Hz for the focus plot when `--frequency-axis` is enabled. Default `10000`.
- `--focus-win-symbols`: width of the focus window in symbols for the focus plot and histogram analysis. By default it uses `--win-symbols`.
- `--focus-region-overlay none|lines|box`: mark the selected focus-window time range on the heatmap. Default `none`.
- `--heatmap-colorscale`: Plotly colorscale name. Default `Viridis`.
- `--heatmap-zmin`, `--heatmap-zmax`: lower and upper heatmap color limits.
- `--hist-bins`: number of histogram bins used in the focus-window distribution-fit analysis. Default `50`.
- `--hist-density`: normalize the histogram to percent-of-samples per bin instead of plotting raw counts.
- `--hist-db`: convert the focus-window histogram input values to dB before binning, which compresses dynamic range.
- `--hist-max`: upper limit for histogram values after any optional `--hist-db` transform.
- `--hist-drop-above-max`: drop values above `--hist-max` instead of saturating them down to that limit.
- `--fit-models none|gaussian|exponential|both`: choose which model curves to fit to the focus-window band histograms. Default `both`.
- `--no-stat-plot`, `--no-heatmap-plot`, `--no-focus-plot`, `--no-alpha-plot`: disable individual subplot panels.

Units and side effects:
- `--heatmap-zmin` and `--heatmap-zmax` use the same units as the selected heatmap scale: dB if the heatmap is in dB, linear power otherwise.
- `--focus-freq-max-hz` only affects the focus plot, and only when `--frequency-axis` is enabled.
- `--focus-win-symbols` changes the width of the dedicated focus window used for the focus plot and histogram analysis. If you do not set it, that width follows `--win-symbols`.
- `--hist-bins`, `--hist-density`, and `--fit-models` affect the focus-window distribution-fit analysis written into the JSON summary and `*.distfit.html` or `*.distfit.png` when plots are enabled.
- `--hist-db` only affects the histogram/distribution-fit analysis. It does not change the main focus plot, heatmap, or CFAR statistic.
- `--hist-max` is applied after any optional `--hist-db` conversion.
- Without `--hist-drop-above-max`, values above `--hist-max` are saturated down to the limit. With `--hist-drop-above-max`, those values are excluded from the histogram and fit analysis entirely.
- The histogram and fit output is written as a separate `*.distfit.html` or `*.distfit.png` file; it is not embedded inside the main `*.plot.html` or `*.plot.png` figure.
- `--hist-density` changes the histogram y-axis from count to percent of samples in each bin, so the histogram bars sum to about `100%` across bins.
- `--hist-db` changes the histogram x-axis units from linear power to dB. This usually makes broad power spreads easier to visualize, but it also changes the domain on which the Gaussian or exponential fits are computed.
- `--fit-models none` disables the Gaussian and exponential overlay curves, but the histogram summaries are still computed and written.
- If both `--heatmap-zmin` and `--heatmap-zmax` are provided, `zmin` must be less than `zmax`.
- `--plots png` requires Plotly image export support such as `kaleido`; `html` does not.
- Disabling subplots reduces plot size and clutter, but the corresponding information is still available in the JSON and CSV outputs where applicable.

Practical hints:
- Use `--plots html` while tuning because hover labels make it much easier to inspect time, bin, and power values.
- Use `--frequency-axis --heatmap-db` when you are reasoning about `f0`, `f1`, frequency offset, or guard-bin width.
- Use `--focus-region-overlay box` when presenting results because it makes the selected focus window obvious in the heatmap.
- Increase `--focus-win-symbols` when you want the focus plot and histogram to summarize a wider slice around the selected center. Reduce it when you want the histogram to reflect a more local, time-concentrated view.
- Start with the default `--hist-bins 50`; increase it only if you have enough samples in the selected band and want finer histogram detail.
- Use `--hist-density` when you want a normalized histogram whose bins sum to about `100%` instead of raw sample counts.
- Use `--hist-db` when a few large values dominate the linear-power histogram and you want better visibility into the lower-power structure.
- Use `--hist-max` to prevent a few very large values from dominating the right edge of the histogram. Add `--hist-drop-above-max` when those high-end values are outliers you want removed entirely instead of clipped.
- Use `--fit-models gaussian` or `--fit-models exponential` when you want to inspect one family at a time without the extra overlay clutter.
- Set `--out-base` for every experiment so each run writes to a distinct result set.

#### Alpha sweep options

- `--pfa-min`: minimum Pfa used in the alpha sweep subplot. Default `1e-7`.
- `--pfa-max`: maximum Pfa used in the alpha sweep subplot. Default `1e-1`.
- `--pfa-points`: number of log-spaced points used in the alpha sweep subplot. Default `60`.

Notes:
- These options only affect the alpha-vs-Pfa subplot.
- They do not change the actual packet-analysis threshold unless you also change `--pfa`.
- If you disable the alpha plot with `--no-alpha-plot`, these settings have no visible plotting effect.

Quick rule of thumb:
- Use `--center-time` plus `--span-time` for the first pass.
- Put the packet fully inside the selected region with margin on both sides.
- Verify `f0` and `f1` in `--frequency-axis --heatmap-db` mode.
- Tune `--threshold-scale` before making finer CFAR changes.

Example (focus in dB + shaded focus window on heatmap):

```bash
python fsk_cfar_cli.py \
  --wav packet_02_gnuradio_c01_clean_with_silence_rcv_loopback.wav \
  --out-base results/cfar/packet_02_loopback_focus_db \
  --center-time 0.30 \
  --span-time 0.80 \
  --focus-db \
  --focus-region-overlay box \
  --plots html
```

### 8.3 Outputs

Given `--out-base results/cfar/packet_02_loopback`, the tool writes:
- `results/cfar/packet_02_loopback.json`: Summary + chosen focus window
- `results/cfar/packet_02_loopback.windows.csv`: One row per sliding window (stat, threshold, etc.)
- `results/cfar/packet_02_loopback.focus_bins.csv`: FFT bins in the focus window labeled as `noise|guard|detect`
- `results/cfar/packet_02_loopback.plot.html` or `.plot.png`: CFAR plots (if enabled)
- `results/cfar/packet_02_loopback.distfit.html` or `.distfit.png`: Focus-window histogram and model-fit plots for noise, guard, and detection bands (if plots are enabled)

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

---

## Appendix A. Choosing Parameters For Offline 256-Bit Packet Search

This appendix summarizes how to choose CLI parameters with the current code when the packet size is fixed at 256 bits (32 bytes), processing is offline, and there is no preamble.

### A.1 Key Point

Even though processing is offline, the best `--win-symbols` value is usually not the full packet length.

Why:
- A packet-length window does maximize total accumulated tone energy when the window fully overlaps the packet.
- But in the current detector, packet finding is based on a run of above-threshold sliding windows, not on a single exact packet-center estimate.
- A very long window smears the packet edges in time, so it is good for saying "there is packet energy around here" but not as good for saying "the packet starts exactly here".
- After CFAR detection, the code still does a separate timing search to refine symbol alignment, so a packet-length detection window does not replace the later start refinement.

### A.2 Why Not Use `--win-symbols 256`

With the current implementation, a packet-length window has several drawbacks:
- The CFAR statistic becomes broad because many nearby window positions overlap most of the same packet energy.
- Start and stop localization get worse because the transition from noise-only to packet-overlap is spread over a long interval.
- The detector's run-length logic becomes less informative when the window is as long as or longer than the packet, because only a small number of windows can fully overlap the packet.
- Any small frequency mismatch, sample-rate mismatch, or trimming error is accumulated over the entire packet-length FFT.

In short: a full-packet window is a stronger energy integrator, but a weaker boundary locator.

### A.3 What Works Better In The Current Code

Use the detector in two stages:
- First, use a moderate CFAR window to identify the packet region.
- Then let the existing demodulation timing search refine the start index.

That matches how the current baseline code is structured:
- CFAR detection finds a plausible packet region from overlapping windows.
- Demodulation then searches symbol offsets and intra-symbol timing to improve alignment.

### A.4 Recommended Starting Parameters

For `fsk_cfar_cli.py` when visually tuning an offline capture:
- `--center-time` plus `--span-time` to bracket the suspected packet with margin on both sides
- `--win-symbols 8` to `16`
- `--hop-symbols 1`
- `--guard-bins 2`
- `--pfa 1e-3`
- `--threshold-scale 10.0` as a first pass, then lower it if the packet is visible but does not cross threshold
- `--frequency-axis --heatmap-db` when checking whether the tones line up with `f0` and `f1`

Why this range:
- `--win-symbols 8` to `16` usually gives enough averaging to make the packet stand out clearly.
- It is still short enough to preserve useful timing information near the packet edges.
- `--hop-symbols 1` is appropriate offline because runtime is usually not the limiting factor and denser windows make the plots easier to interpret.

### A.5 Practical Guidance For `--span-time`

Set `--span-time` large enough to include:
- the full 256-bit packet
- some non-packet time before it
- some non-packet time after it

This helps because:
- the threshold trace is easier to interpret when you can compare packet and non-packet regions
- the heatmap is more useful when the packet is not clipped at the plot edges
- the selected focus window is less likely to land on an ambiguous edge case

If the span is too small, the packet can be clipped. If it is too large, the packet still appears, but the plots become less convenient to inspect.

### A.6 Suggested Offline Tuning Workflow

1. Start with a centered region around the suspected packet using `--center-time` and `--span-time`.
2. Use `--win-symbols 8` or `16` and `--hop-symbols 1`.
3. Turn on `--frequency-axis --heatmap-db` and confirm that the visible ridges align with `--f0` and `--f1`.
4. Check whether the packet region rises above threshold in the stat plot.
5. If the packet is visible but does not cross threshold, lower `--threshold-scale` before making more aggressive changes.
6. If nearby leakage is inflating the noise estimate, increase `--guard-bins` slightly.
7. Once the CFAR region looks correct, use the normal decode path to let the current demodulator refine the start timing.

### A.7 Example Starting Command

```bash
python fsk_cfar_cli.py \
  --wav packet_02_gnuradio_c01_clean_with_silence_rcv_loopback.wav \
  --out-base results/cfar/packet_02_loopback_offline_search \
  --center-time 0.30 \
  --span-time 0.80 \
  --win-symbols 16 \
  --hop-symbols 1 \
  --guard-bins 2 \
  --pfa 1e-3 \
  --threshold-scale 10.0 \
  --frequency-axis \
  --heatmap-db \
  --plots html
```

### A.8 Rule Of Thumb

For the current code and a fixed 256-bit packet with no preamble:
- Do not use a full-packet CFAR window as the default search setting.
- Use a moderate window to get a reliable packet region.
- Let the existing timing-refinement logic handle the exact symbol alignment.
- Optimize first for robust packet-region detection, not for a single-window energy maximum.


