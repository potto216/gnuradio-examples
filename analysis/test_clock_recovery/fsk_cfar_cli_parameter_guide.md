# CFAR CLI Parameter Guide for Packet Detection

This note explains how the parameters in `fsk_cfar_cli.py` affect packet detection and what to look for when tuning them.

The CLI does not directly decode bits. It inspects a selected time region of a WAV file, slides an FFT window across that region, and computes a CFAR-style packet score:

```text
stat = |X(f0)|^2 + |X(f1)|^2
threshold = alpha(Pfa, training_bins) * mean(noise_bins) * threshold_scale

packet-like window if:
    stat > threshold
```

That means every option is really controlling one of four stages:

1. Which part of the recording gets analyzed.
2. How large each sliding FFT window is.
3. Which FFT bins count as signal, guard, or noise.
4. How hard it is for a window to exceed the detection threshold.

## 1. Detection Flow

```text
WAV file
  |
  +--> choose analysis region
  |      using --start-time/--end-time
  |      or    --center-time/--span-time
  |
  +--> convert selected region to analytic signal
  |
  +--> slide FFT window of length:
  |        Nw = win_symbols * samples_per_symbol
  |      with hop:
  |        hop = hop_symbols * samples_per_symbol
  |
  +--> in each window
  |        detect bins: k0, k1 from f0, f1
  |        guard bins:  +/- guard_bins around k0 and k1
  |        noise bins:  everything else except DC bin 0
  |
  +--> compute stat and threshold
  |
  +--> generate outputs
         .json           summary
         .windows.csv    one row per sliding window
         .focus_bins.csv signal/guard/noise bins for the focus window
         .plot.html/png  4-panel visualization
```

## 2. Where the Time-Range Options Should Sit Relative to a Packet

The analysis region should include the packet and some surrounding non-packet time. If the selected region is too tight, the heatmap and threshold trend are hard to interpret. If it is too wide, the packet becomes visually small and tuning gets slower.

### Good mental model

```text
time -------------------------------------------------------------->

recording:     noise/silence      packet energy present      noise/silence
               .............. [==========================] ..............

analysis:          |---------------------------------------------|
                    start-time                             end-time

focus point:                     ^
                                 center-time

span-time:          <------------------------------- -->
```

### Recommended placement

Use `--start-time` and `--end-time` so the packet sits well inside the selected region, not touching the edges:

```text
bad:
|==========================|..........
^                          ^
start-time                 end-time

better:
.....|==========================|.....
     ^                          ^
     packet starts after        packet ends before
     the analysis start         the analysis end

best for first pass:
........|==========================|........
        ^                          ^
        keep some margin on both sides
```

### When to use `--center-time` and `--span-time`

These are the quickest way to inspect a suspected packet location.

```text
time -------------------------------------------------------------->
.......... [========= packet =========] ..........
                    ^
              center-time

span-time covers both sides:
        |<----------------------------->|
```

`--center-time` with `--span-time` overrides `--start-time` and `--end-time` in the code path. Use it when you know roughly where the packet is and want a symmetric zoom around it.

### Sample-based equivalents

If another tool gives offsets in samples instead of seconds, use:

- `--start-sample`, `--end-sample`
- `--center-sample`, `--span-samples`

They behave the same way as the time-based versions.

## 3. How the Sliding Window Relates to the Packet

The CLI evaluates many short windows, not the entire selected region at once.

```text
selected analysis region
|---------------------------------------------------------------|

window 0: [------ Nw ------]
window 1:           [------ Nw ------]
window 2:                     [------ Nw ------]
window 3:                               [------ Nw ------]
                 step size = hop
```

In the implementation:

```text
Nw  = win_symbols * sps
hop = hop_symbols * sps
```

where `sps = fs / baud`.

### `--win-symbols`

Controls how much packet energy is collected in each FFT.

```text
small window
[--]
high time precision, less energy averaging

large window
[--------]
more energy averaging, more temporal smearing
```

Effects:

- Smaller values make the time plot more responsive to packet edges.
- Larger values usually stabilize the score, but can blur the packet start and stop.
- The value also changes the FFT size, which changes the exact FFT bin index used for `f0` and `f1`.

Starting point:

- Use `--win-symbols 2` first. That is also the current default.
- Try 3 or 4 if the score is too noisy.
- Try 1 if you need better edge localization and the tones are still visible.

### `--hop-symbols`

Controls how far the window moves between evaluations.

```text
hop-symbols = 1
[------]
   [------]
      [------]

hop-symbols = 2
[------]
      [------]
            [------]
```

Effects:

- Smaller hops give denser time samples in the stat/threshold trace.
- Larger hops run faster but can skip over short transitions.

Starting point:

- Use `--hop-symbols 1` when tuning.
- Increase only after you already understand where the packet sits.

## 4. How `f0`, `f1`, and `guard-bins` Define Signal vs Noise

For each sliding window, the code converts the requested tones into FFT bins:

```text
k0 = round(f0 * Nw / fs)
k1 = round(f1 * Nw / fs)
```

Then it assigns bins to three groups.

```text
frequency / FFT bin axis ---------------------------------------------->

           noise    guard   detect   guard        noise      guard   detect   guard    noise
...... ..... ..... [ g ]   [ f0 ]   [ g ] .... ............ [ g ]   [ f1 ]   [ g ] .......
```

The code marks every bin inside the guard range as excluded from noise estimation:

```text
around each tone bin k:

k-guard_bins ... k-1   k   k+1 ... k+guard_bins
      excluded    excluded signal excluded    excluded
```

### Why guard bins matter

If the guard region is too small, nearby tone leakage can contaminate the noise estimate. That raises `noise_mean` and can make true packets harder to detect.

If the guard region is too large, too many bins are removed from the noise pool. That leaves fewer training bins, making the threshold estimate less stable.

### Frequency-time view for understanding guard bins

Use the heatmap with frequency units when tuning guard bins:

```bash
python fsk_cfar_cli.py \
  --wav your_capture.wav \
  --center-time 0.30 \
  --span-time 0.80 \
  --frequency-axis \
  --heatmap-db \
  --plots html
```

What to look for in the heatmap:

```text
frequency
  ^
  |                          packet interval
  |                 |<------------------------------>|
  |
  |  f1  --------------------------------------------- bright ridge
  |
  |  f0  --------------------------------------------- bright ridge
  |
  +---------------------------------------------------------------> time
```

If energy is visibly spreading into adjacent frequencies around `f0` or `f1`, increase `--guard-bins` a little. If the focus-window plot shows the guard region eating too much spectrum, reduce it.

### `--f0` and `--f1`

These must match the transmitted tone pair closely enough that the detector lands on the correct FFT bins.

Effects of wrong values:

- `stat` drops because the detector is measuring the wrong bins.
- The heatmap may still show a ridge, but not at the expected `f0` and `f1` bins.
- The packet can disappear even when it is visually obvious in the spectrum.

If you suspect frequency offset, first inspect with `--frequency-axis --heatmap-db`, then adjust `--f0` and `--f1` toward the visible ridges.

## 5. How the Threshold Is Controlled

The threshold is built from three main controls:

1. `--pfa`
2. `--threshold-scale`
3. The number of training bins left after guard-bin exclusion

### `--pfa`

The code computes an alpha value from the requested false-alarm probability.

```text
smaller Pfa  -> larger alpha  -> higher threshold  -> harder to trigger
larger Pfa   -> smaller alpha -> lower threshold   -> easier to trigger
```

Practical meaning:

- Lower `--pfa` if noise-only windows are crossing the threshold too often.
- Raise `--pfa` if the packet windows stay below threshold even though the tones are present.

Starting point:

- `--pfa 1e-3` is a reasonable first pass.

### `--threshold-scale`

This is a direct multiplier applied after the CA-CFAR alpha and noise mean are computed.

```text
threshold = alpha * noise_mean * threshold_scale
```

Practical meaning:

- Increase it to be more conservative.
- Decrease it if the packet is clearly present but the threshold sits above the stat trace.

This is often the fastest knob to adjust once `f0`, `f1`, and the time region are already correct.

## 6. What Each Plot Tells You

The HTML or PNG output contains four panels.

### Panel 1: CFAR statistic vs threshold

```text
power
  ^
  |      stat  /
  |           /  \__
  | threshold -------\-----------------
  +------------------------------------> time
```

Use it to answer:

- Does the packet region rise above threshold?
- Are there many false peaks outside the packet?
- Is the threshold obviously too high or too low?

### Panel 2: FFT power heatmap over time

Use this to confirm the packet really contains energy near `f0` and `f1`, and to determine whether the selected time range properly brackets the packet.

Best settings for interpretation:

- `--frequency-axis` to view Hz instead of raw FFT bin index.
- `--heatmap-db` to make weak structure visible.

### Panel 3: Focus window FFT bins by CFAR band

This is the clearest picture of how the code split the spectrum in one selected window.

The focus window is not an overlap-add spectrum and it is not an average across many FFTs. It is one FFT taken from one sliding analysis window chosen from the region you selected.

In the implementation, the CLI first builds many short overlapping analysis windows. The selected focus window $w^*$ is the one whose center sample is closest to the requested analysis center $c$:

$$w^* = \underset{w}{\arg\min} \left| \left( \text{start} + \text{idx}_w + \frac{N_w}{2} \right) - c \right|$$

```text
Nw  = win_symbols * sps
hop = hop_symbols * sps

window 0 samples: [0 ................ Nw-1]
window 1 samples: [hop .............. hop+Nw-1]
window 2 samples: [2*hop ............ 2*hop+Nw-1]
...
```

For each window it computes one complex FFT of the analytic-signal slice and then converts that to per-bin power:

```text
seg = xa[idx : idx + Nw]
P   = |FFT(seg)|^2
```

So the focus plot shows the bin powers from exactly one of those windows. The selected window is the sliding window whose center sample is closest to the requested analysis center.

```text
window center sample = start_sample + idx + Nw/2
focus window = window whose center is nearest to:
               center-time or center-sample

if no explicit center was given:
    use the midpoint of the selected analysis region
```

That means:

- If `hop < Nw`, neighboring windows overlap in time, but each plotted point in Panel 3 still comes from one FFT only.
- The plot is instantaneous for that chosen window, not time-averaged across the packet.
- Noise, guard, and detect markers are just different CFAR labels applied to bins from that one power spectrum.

```text
bin power
  ^
  |         o noise
  |    x x x x x x x   guard bins
  |           D             detection bin at f0
  |                               D   detection bin at f1
  +----------------------------------------------------> FFT bin
```

Use it to answer:

- Are the chosen detection bins sitting on real peaks?
- Are guard bins wide enough to exclude leakage?
- Are there too few remaining noise bins?
- Is the selected focus window actually centered on the packet portion you meant to inspect?

### Panel 4: alpha vs Pfa

This helps explain how sensitive the threshold is to `--pfa`. It is mainly a tuning aid, not the primary packet-interpretation plot.

## 7. Parameter-by-Parameter Tuning Summary

| Parameter | What it changes | If detection is poor, try this |
| --- | --- | --- |
| `--start-time`, `--end-time` | Absolute analysis window in time | Expand them so the packet is inside with margin on both sides |
| `--center-time`, `--span-time` | Symmetric zoom around a suspected packet | Use this first when you know the approximate packet location |
| `--start-sample`, `--end-sample` | Sample-based version of start/end | Use when another tool gives sample indices |
| `--center-sample`, `--span-samples` | Sample-based version of center/span | Same as above, but for centered zoom |
| `--fs` | Overrides WAV sample rate in analysis math | Only use if the WAV metadata is wrong |
| `--baud` | Samples per symbol and window/hop conversion | Match the transmitter exactly |
| `--f0`, `--f1` | Detection tone locations | Align them with visible ridges in the spectrum |
| `--win-symbols` | FFT window duration | Increase for smoother scores, decrease for sharper timing |
| `--hop-symbols` | Step between windows | Keep at 1 while tuning |
| `--guard-bins` | Excluded bins around each tone | Increase if tone leakage pollutes the noise estimate |
| `--pfa` | CA-CFAR aggressiveness through alpha | Lower for fewer false alarms, raise for more sensitivity |
| `--threshold-scale` | Extra threshold multiplier | Lower if the packet never crosses threshold |
| `--frequency-axis` | Heatmap y-axis and focus-window x-axis in Hz | Turn on when reasoning about `f0`, `f1`, and guard width |
| `--heatmap-db` | Log-scaled heatmap visibility | Turn on when packet ridges are hard to see |
| `--heatmap-zmin`, `--heatmap-zmax` | Heatmap contrast limits | Use to stabilize color contrast between runs |
| `--plots` | Whether the figure is written | Use `html` while tuning, `png` for reports, `none` for batch runs |
| `--out-base` | Output file prefix | Set this so each experiment writes to a distinct result set |

## 8. Practical Tuning Workflow

Start with a frequency-aware view around the suspected packet:

```bash
python fsk_cfar_cli.py \
  --wav your_capture.wav \
  --center-time 0.30 \
  --span-time 0.80 \
  --baud 100 \
  --f0 1000 \
  --f1 2000 \
  --win-symbols 2 \
  --hop-symbols 1 \
  --guard-bins 2 \
  --pfa 1e-3 \
  --threshold-scale 10.0 \
  --frequency-axis \
  --heatmap-db \
  --plots html
```

Then tune in this order:

1. Confirm the time range contains the whole packet with some margin.
2. Confirm the spectral ridges line up with `f0` and `f1`.
3. Check the focus-window plot to see whether `guard-bins` excludes nearby leakage but still leaves enough noise bins.
4. Adjust `--threshold-scale` for quick coarse sensitivity changes.
5. Refine `--pfa` if you want a more principled shift in CFAR aggressiveness.
6. Only then experiment with larger `--win-symbols` or `--hop-symbols`.

## 9. Quick Failure Patterns

### Packet visible in heatmap, but `stat` does not cross threshold

Likely causes:

- `--threshold-scale` too high
- `--pfa` too small
- `--f0` or `--f1` slightly wrong
- `--guard-bins` too small, causing leakage to inflate `noise_mean`

### Many false triggers outside the packet

Likely causes:

- `--threshold-scale` too low
- `--pfa` too large
- Analysis region contains unrelated tones or interference

### Packet is cut off in the plots

Likely cause:

- `--start-time`/`--end-time` or `--center-time`/`--span-time` selected too tightly

### Focus-window plot does not show clear peaks at the detection bins

Likely causes:

- Wrong `--f0` or `--f1`
- Wrong `--baud`, which changes `Nw` and therefore the derived FFT bins
- The selected focus time is not actually inside the packet

## 10. Minimal Rule of Thumb

If you only remember four things, remember these:

1. Put the packet fully inside the selected time range, with margin on both sides.
2. Use `--frequency-axis --heatmap-db` when checking tone placement and guard bins.
3. Tune `--threshold-scale` first for quick sensitivity changes.
4. Treat `--guard-bins` as protection against tone leakage contaminating the noise estimate.