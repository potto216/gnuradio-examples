
import numpy as np
import plotly.graph_objs as go

# import soundfile as sf
import wave
import sys

try:
    from scipy.signal import fftconvolve
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    # If scipy isn't available, we'll only use np.correlate
    HAS_SCIPY = False

def find_delay_by_amplitude(sc_output, sc_input, sample_rate=None, 
                            noise_window=1000, threshold_factor=3):
    """
    Estimates the delay in sc_input by detecting when the low-amplitude noise ends
    and the actual signal starts.

    Parameters
    ----------
    sc_output : np.ndarray
        The original signal played out (not directly used in this amplitude-based method,
        but included for consistency).
    sc_input : np.ndarray
        The delayed recording of the signal, which starts with noise and then
        eventually contains a delayed copy of sc_output.
    sample_rate : float, optional
        Sampling rate in Hz. If provided, the function also returns delay in seconds.
    noise_window : int, optional
        Number of initial samples in sc_input to consider for estimating noise. 
        If sc_input is shorter than this, uses the entire array.
    threshold_factor : float, optional
        Multiplier for the noise standard deviation to set the detection threshold.

    Returns
    -------
    delay_samples : int
        The delay in samples at which sc_input begins to exceed the noise threshold.
    delay_seconds : float, optional
        The delay in seconds, if sample_rate is provided.
    """

    # 1. Estimate noise level from the first `noise_window` samples.
    #    If sc_input is shorter, just use the entire sc_input.
    window_end = min(noise_window, len(sc_input))
    noise_section = sc_input[:window_end]
    
    noise_std = np.std(noise_section)
    threshold = threshold_factor * noise_std
    # print threshold, noise_std, noise_section, window_end
    print(f"Threshold: {threshold:.4f}, Noise Std: {noise_std:.4f}, Window End: {window_end}")

    # 2. Find the first index where the signal amplitude exceeds the threshold
    # delay_idx = None
    # for i in range(len(sc_input)):
    #     if abs(sc_input[i]) > threshold:
    #         delay_idx = i
    #         break
    # 2. Find the first index where the signal amplitude exceeds the threshold (vectorized)
    above_threshold_indices = np.where(np.abs(sc_input) > threshold)[0]
    if len(above_threshold_indices) > 0:
        delay_idx = above_threshold_indices[0]
    else:
        delay_idx = 0

    # If we never exceed threshold, assume no signal was found
    if delay_idx is None:
        delay_idx = 0

    # 3. Convert to seconds if sample_rate is provided
    if sample_rate is not None:
        delay_seconds = delay_idx / sample_rate
        return delay_idx, delay_seconds
    else:
        return delay_idx


def cut_signals(sc_output, sc_input, 
                sc_output_start_index=0, sc_input_start_index=0):
    """
    Cuts both signals starting at the specified indices so that they are the same length.

    Parameters
    ----------
    sc_output : np.ndarray
        The output signal (array).
    sc_input : np.ndarray
        The input signal (array).
    sc_output_start_index : int
        Start index for sc_output.
    sc_input_start_index : int
        Start index for sc_input.

    Returns
    -------
    sc_output_cut : np.ndarray
        The sliced portion of sc_output.
    sc_input_cut : np.ndarray
        The sliced portion of sc_input.

    Raises
    ------
    ValueError
        If any start index is out of valid range or there is no overlap to slice.
    """

    # 1. Basic index checks
    if not (0 <= sc_output_start_index < len(sc_output)):
        raise ValueError(
            f"sc_output_start_index={sc_output_start_index} is out of range for sc_output "
            f"with length {len(sc_output)}."
        )
    if not (0 <= sc_input_start_index < len(sc_input)):
        raise ValueError(
            f"sc_input_start_index={sc_input_start_index} is out of range for sc_input "
            f"with length {len(sc_input)}."
        )

    # 2. Determine the number of samples we can slice
    #    We subtract each start index from the respective signal lengths
    #    to see how many samples remain from that point. Then take the minimum.
    available_output = len(sc_output) - sc_output_start_index
    available_input = len(sc_input) - sc_input_start_index
    num_samples = min(available_output, available_input)

    if num_samples <= 0:
        raise ValueError(
            "No valid overlapping region to slice. "
            f"Computed num_samples={num_samples}, which is non-positive."
        )

    # 3. Slice both signals
    sc_output_cut = sc_output[sc_output_start_index : sc_output_start_index + num_samples]
    sc_input_cut = sc_input[sc_input_start_index : sc_input_start_index + num_samples]

    return sc_output_cut, sc_input_cut



def fractional_time_shift(signal, shift_in_samples, method='linear', fill_value=0.0):
    """
    Shifts a 1D signal by a (possibly fractional) amount of samples, using the
    specified interpolation method.

    Parameters
    ----------
    signal : array_like
        The input signal (1D numpy array).
    shift_in_samples : float
        The shift in samples. Can be fractional.
        - Positive shift  => signal is delayed (moves to the right).
        - Negative shift  => signal is advanced (moves to the left).
    method : str, optional
        Interpolation method. Options (when SciPy is available) include:
          - 'linear'
          - 'nearest'
          - 'zero'
          - 'slinear'
          - 'quadratic'
          - 'cubic'
          - etc.
        If SciPy is not installed, only 'linear' is supported via NumPy's np.interp.
        Default is 'linear'.
    fill_value : float, optional
        Value to use for samples that fall outside the original signal range.
        Default is 0.0

    Returns
    -------
    shifted_signal : ndarray
        The time-shifted signal, the same length as the original.
        Samples outside the original range are filled with `fill_value`.
    """

    n = len(signal)
    original_x = np.arange(n, dtype=float)

    # We want to evaluate the original signal at a "shifted" axis
    # For a positive shift_in_samples, we effectively want to sample
    # the signal from earlier times => x - shift_in_samples.
    shifted_x = original_x - shift_in_samples

    # If SciPy isn't installed, fall back to np.interp only for 'linear'.
    if not HAS_SCIPY:
        if method != 'linear':
            raise ImportError(
                "SciPy is required for interpolation methods other than 'linear'. "
                "Please install SciPy or use method='linear'."
            )
        # Linear interpolation fallback (NumPy)
        shifted_signal = np.interp(
            original_x, 
            shifted_x, 
            signal, 
            left=fill_value, 
            right=fill_value
        )
    else:
        # Use SciPy's interp1d for the specified interpolation method
        interpolator = interp1d(
            original_x,
            signal,
            kind=method,
            bounds_error=False,
            fill_value=fill_value
        )
        shifted_signal = interpolator(original_x - shift_in_samples)

    return shifted_signal

# shift a signal by a known number of samples which could be a fractional amount
def shift_signal(signal, shift_samples):
    """
    Shift a signal by a known number of samples (which could be a fractional amount).

    Parameters
    ----------
    signal : array_like
        Input signal (1D array).
    shift_samples : float
        Number of samples to shift the signal by.
        Positive values shift the signal to the right (delay),
        while negative values shift the signal to the left (advance).

    Returns
    -------
    shifted_signal : ndarray
        The shifted signal.
    """

    # If the shift is an integer, we can use np.roll for a fast shift
    if shift_samples.is_integer():
        shift_samples = int(shift_samples)
        return np.roll(signal, shift_samples)

    # Otherwise, we'll interpolate the signal to shift by a fractional amount
    # This is a simple linear interpolation, which can introduce artifacts
    # (e.g., if the signal is not band-limited or has sharp edges)
    x = np.arange(len(signal))
    x_shifted = x - shift_samples
    return np.interp(x, x_shifted, signal, left=0.0, right=0.0)



def find_shift_between_signals(
    signal1,
    signal2,
    sample_rate,
    interpolation_factor=1,
    corr_mode='full',
    method='correlate',
    plot=True
):
    """
    Compute the correlation of two signals and its amplitude spectrum in frequency domain,
    with an option to interpolate for higher resolution. Also returns the shift_in_samples and a
    shifted version of signal2 aligned to signal1 at the original sample rate.

    Parameters
    ----------
    signal1 : array_like
        First input signal (1D array).
    signal2 : array_like
        Second input signal (1D array).
    sample_rate : float
        The sampling rate of the original signals (in Hz).
    interpolation_factor : int, optional
        Factor by which to up-sample (interpolate) the signals.
        If 1, no interpolation is performed. Default is 1.
    corr_mode : str, optional
        Mode for correlation (e.g. 'full', 'same', 'valid'). Default is 'full'.
    method : {'correlate', 'fftconvolve'}, optional
        Correlation method:
            - 'correlate': use np.correlate
            - 'fftconvolve': use scipy.signal.fftconvolve (requires SciPy)
        Default is 'correlate'.
    plot : bool, optional
        If True, show the correlation and spectrum plots. Default is True.

    Returns
    -------
    shift_in_samples : float
        The shift (peak correlation index offset) in original samples
        (i.e. after dividing out the interpolation factor).
    """

    # ---------------------------
    # 1. Interpolate if needed
    # ---------------------------
    # Assume signal1 and signal2 are the same length for simplicity
    n_original = len(signal1)
    x_original = np.arange(n_original)

    if interpolation_factor > 1:
        # Create a higher-resolution index
        # length = n_original * interpolation_factor
        x_upsampled = np.linspace(0, n_original - 1, n_original * interpolation_factor)
        interp_signal1 = np.interp(x_upsampled, x_original, signal1)
        interp_signal2 = np.interp(x_upsampled, x_original, signal2)
    else:
        # No interpolation
        x_upsampled = x_original
        interp_signal1 = signal1
        interp_signal2 = signal2

    # ---------------------------
    # 2. Correlation
    # ---------------------------
    if method == 'correlate':
        correlation = np.correlate(interp_signal1, interp_signal2, mode=corr_mode)
    elif method == 'fftconvolve':
        if not HAS_SCIPY:
            raise ImportError("scipy.signal.fftconvolve is not available. Install scipy or switch method.")
        # For correlation via fftconvolve, convolve one signal with the time-reversed other
        correlation = fftconvolve(interp_signal1, interp_signal2[::-1], mode=corr_mode)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    # ---------------------------
    # 3. Find the shift_in_samples
    # ---------------------------
    # For 'full' mode, correlation length = len(A) + len(B) - 1
    # The "zero-shift" index is (len(interp_signal1) - 1) if signals have same length in upsampled domain
    zero_shift_index = len(interp_signal1) - 1
    peak_index = np.argmax(correlation)
    shift_index = peak_index - zero_shift_index  # in upsampled samples

    # Convert shift to original sampling
    shift_in_samples = shift_index / interpolation_factor

    # ---------------------------
    # 7. Optionally plot
    # ---------------------------
    if plot:
        # ---------------------------
        # 6. Compute and plot FFT of the correlation
        # ---------------------------
        corr_fft = np.fft.fft(correlation)
        corr_spectrum = np.abs(corr_fft)

        # Keep only the positive frequencies
        half_len = len(corr_spectrum) // 2
        corr_spectrum = corr_spectrum[:half_len]

        # Create frequency axis in Hz
        interp_sample_rate = sample_rate * interpolation_factor
        freqs = np.fft.fftfreq(len(correlation), d=1.0 / interp_sample_rate)
        freqs = freqs[:half_len]
        # --- Plot correlation in the upsampled domain ---
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(
            x=np.arange(len(correlation)),
            y=correlation,
            mode='lines',
            name='Correlation'
        ))
        fig_corr.update_layout(
            title='Correlation (Upsampled Domain)',
            xaxis_title='Correlation Array Index (Upsampled)',
            yaxis_title='Amplitude'
        )
        fig_corr.show()

        # --- Plot correlation spectrum ---
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(
            x=freqs,
            y=corr_spectrum,
            mode='lines',
            name='Correlation Spectrum'
        ))
        fig_spec.update_layout(
            title='Correlation Spectrum (Frequency Domain)',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Amplitude'
        )
        fig_spec.show()

    # ---------------------------
    # 8. Return shift
    # ---------------------------
    return shift_in_samples


# Example usage
def test_find_shift_between_signals_and_shift_signal():
       
        # Create sample data
    fs = 100.0  # original sample rate (Hz)
    t = np.linspace(0, 1, int(fs), endpoint=False)
    # Two signals with a known shift of 5 samples
    sig1 = np.sin(2 * np.pi * 5 * t)  # 5 Hz tone
    shift_samples = 5
    sig2 = np.roll(sig1, shift_samples)

    # # Create a simple signal (e.g., a sine wave)
    # fs = 50  # 50 Hz sample rate
    # t = np.linspace(0, 1, fs, endpoint=False)
    # sig = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

    # # Shift by 3.7 samples using cubic interpolation
    # shift = 3.7

    # Estimate correlation-based shift and get the shifted signal2
    shift_in_samples = find_shift_between_signals(
        sig1, sig2,
        sample_rate=fs,
        interpolation_factor=2,
        corr_mode='full',
        method='correlate',
        plot=True
    )

    print(f"Estimated shift_in_samples (in original samples): {shift_in_samples:.3f}")

    # Compare the final aligned signal2 with sig1 if desired
    # (They should be nearly identical apart from edge effects.)
    method = 'cubic'

    shifted_sig2 = fractional_time_shift(sig2, shift_in_samples, method=method)

    # Plot for comparison
    fig = go.Figure()

    # Plot sig1
    fig.add_trace(
        go.Scatter(
            x=t,
            y=sig1,
            mode='lines',
            name='sig1'
        )
    )

    # Plot sig2
    fig.add_trace(
        go.Scatter(
            x=t,
            y=sig2,
            mode='lines',
            name='sig2'
        )
    )

    # Plot shifted_sig2 with dashed line
    fig.add_trace(
        go.Scatter(
            x=t,
            y=shifted_sig2,
            mode='lines',
            name=f"sig2 shifted by {shift_in_samples:.1f} samples ({method})",
            line=dict(dash='dash')
        )
    )

    # Update layout
    fig.update_layout(
        title="Fractional Time Shift with Interpolation",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude"
    )

    # Show the interactive plot
    fig.show()


import wave
import numpy as np
import csv
import os
from scipy.signal import hilbert
from pathlib import Path

def process_large_wav_in_chunks(
    filename,
    channel=0,
    chunk_size=65536,
    threshold=0.0001,
    min_below_count=100,
    override_sample_rate=None,
    output_csv_path=None,
    output_binary_dir=None,
    chunk_index_padding=6
):
    """
    Process a large .wav file in chunks to compute instantaneous phase (radians)
    and instantaneous frequency (rad/s) for non-silent chunks. Phase is kept
    continuous from one chunk to the next by adjusting each new chunk’s
    initial phase offset.

    Parameters
    ----------
    filename : str or Path
        Path to the WAV file.
    channel : int, optional
        Zero-based index of the channel to process.
    chunk_size : int, optional
        Number of frames to read at once from the file.
    threshold : float, optional
        Absolute amplitude threshold for considering a chunk "silent."
        (Used in a simple chunk-level test, see note below.)
        If None, no silence check is applied and all chunks are processed.
    min_below_count : int, optional
        If at least this many consecutive samples within the chunk are below
        threshold, we consider the entire chunk silent.
    override_sample_rate : float, optional
        If provided, use this sample rate (samples per second).
        Otherwise, use the WAV file's sample rate.
    output_csv_path : str or None, optional
        If provided, writes each chunk's results to a single CSV file.
        Rows: [chunk_index, time_seconds, signal_amplitude, phase_radians, frequency_rad_s]
    output_binary_dir : str or None, optional
        If provided, creates one .npz file per chunk in this directory. Each .npz
        will contain arrays: chunk_index, time, signal, phase, frequency.

    Returns
    -------
    results : list of dict
        Each dict corresponds to a chunk that was processed (non-silent).
        Fields:
            - 'chunk_index': int
            - 'start_time': float (seconds)
            - 'end_time': float (seconds)
            - 'time': np.ndarray (1D, seconds)
            - 'signal': np.ndarray (1D, the chunk’s data after channel extraction)
            - 'phase': np.ndarray (1D, unwrapped phase in radians)
            - 'frequency': np.ndarray (1D, instantaneous angular frequency in rad/s)
              (length matches the chunk; freq[0] is copied from freq[1] for continuity)

    Notes
    -----
    1) The Hilbert transform on each chunk introduces some boundary effects. For
       most applications, using a large chunk_size is usually acceptable. For very
       precise phase measurements, consider using overlap between chunks.
    2) If you only want CSV output or only binary output, set the other output
       parameter to None.
    3) If you do not need to keep all chunks in memory, you can remove the
       'results.append(...)' part and rely solely on the CSV or binary files.
    """

    filename = Path(filename)

    # Open the WAV file
    with wave.open(str(filename), 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        wav_sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        # Use WAV file's sample rate unless overridden
        sample_rate = wav_sample_rate if override_sample_rate is None else override_sample_rate

        # Check channel validity
        if channel < 0 or channel >= n_channels:
            raise ValueError(
                f"Requested channel {channel} out of range; file has {n_channels} channel(s)."
            )

        # --------------------------------------------------
        # Prepare optional CSV output
        # --------------------------------------------------
        csv_file = None
        csv_writer = None
        if output_csv_path is not None:
            output_csv_path = Path(output_csv_path)
            # Create folder if needed
            os.makedirs(str(output_csv_path.parent), exist_ok=True)
            csv_file = open(output_csv_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            # Write a header row
            csv_writer.writerow([
                "chunk_index",
                "time_seconds",
                "signal_amplitude",
                "phase_radians",
                "frequency_rad_s"
            ])

        # --------------------------------------------------
        # Prepare optional Binary (.npz) output
        # --------------------------------------------------
        if output_binary_dir is not None:
            output_binary_dir = Path(output_binary_dir)
            os.makedirs(output_binary_dir, exist_ok=True)

        # Keep phase continuous across chunks
        prev_phase_end = 0.0  # last sample's phase from the previous chunk
        first_chunk = True

        results = []
        frames_read = 0
        chunk_index = 0

        while frames_read < n_frames:
            frames_to_read = min(chunk_size, n_frames - frames_read)
            raw_data = wf.readframes(frames_to_read)
            frames_read += frames_to_read
            chunk_index += 1

            # Convert bytes -> int16 array
            data_int16 = np.frombuffer(raw_data, dtype=np.int16)

            # Reshape to (samples_in_chunk, n_channels)
            samples_in_chunk = len(data_int16) // n_channels
            data_int16 = data_int16.reshape((samples_in_chunk, n_channels))

            # Extract selected channel as float in [-1, 1]
            channel_data = data_int16[:, channel].astype(np.float32) / 32768.0

            # --------------------------------------------------
            # Silence Check (chunk-level, optional)
            # --------------------------------------------------
            is_silent_chunk = False
            if threshold is not None:
                silent_count = 0
                for sample_val in channel_data:
                    if abs(sample_val) < threshold:
                        silent_count += 1
                        if silent_count >= min_below_count:
                            # If a chunk is considered "silent," skip it
                            is_silent_chunk = True
                            break
                    else:
                        silent_count = 0

            if is_silent_chunk:
                # Skip updating prev_phase_end; phase continuity will resume
                # with the next non-silent chunk.
                continue

            # --------------------------------------------------
            # Hilbert Transform -> Unwrapped Phase
            # --------------------------------------------------
            analytic = hilbert(channel_data)
            raw_phase = np.unwrap(np.angle(analytic))  # in radians

            # --------------------------------------------------
            # Adjust phase for continuity across chunks
            # --------------------------------------------------
            if first_chunk:
                first_chunk = False
            else:
                # Shift new chunk so that raw_phase[0] continues from prev_phase_end
                offset = prev_phase_end - raw_phase[0]
                raw_phase += offset

            # Update for next chunk
            prev_phase_end = raw_phase[-1]

            # --------------------------------------------------
            # Instantaneous Frequency (rad/s)
            # --------------------------------------------------
            freq = np.zeros_like(raw_phase, dtype=np.float32)
            if len(raw_phase) > 1:
                freq[1:] = np.diff(raw_phase) * sample_rate / (2 * np.pi)  # Convert to Hz
                freq[0] = freq[1]  # or some continuity approach

            # --------------------------------------------------
            # Time Axis for this chunk
            # --------------------------------------------------
            chunk_start_frame = frames_read - frames_to_read
            chunk_end_frame = chunk_start_frame + samples_in_chunk
            chunk_start_time = chunk_start_frame / sample_rate
            chunk_end_time = chunk_end_frame / sample_rate
            t = np.linspace(chunk_start_time, chunk_end_time, samples_in_chunk, endpoint=False)

            # --------------------------------------------------
            # (Optional) Write to CSV
            # --------------------------------------------------
            if csv_writer is not None:
                for i in range(samples_in_chunk):
                    csv_writer.writerow([
                        chunk_index,
                        f"{t[i]:.6f}",
                        f"{channel_data[i]:.6f}",
                        f"{raw_phase[i]:.6f}",
                        f"{freq[i]:.6f}"
                    ])

            # --------------------------------------------------
            # (Optional) Write to Binary: one .npz per chunk
            # --------------------------------------------------
            if output_binary_dir is not None:
                # e.g., chunk_1.npz, chunk_2.npz, ...
                chunk_npz_path = output_binary_dir / f"chunk_{chunk_index:0{chunk_index_padding}d}.npz"
                np.savez_compressed(
                    chunk_npz_path,
                    chunk_index=chunk_index,
                    time=t,
                    signal=channel_data,
                    phase=raw_phase,
                    frequency=freq
                )

            # --------------------------------------------------
            # Store results (in memory)
            # --------------------------------------------------
            results.append({
                "chunk_index": chunk_index,
                "start_time": chunk_start_time,
                "end_time": chunk_end_time,
                "time": t,
                "signal": channel_data,
                "phase": raw_phase,
                "frequency": freq
            })

        # Close CSV file if it was opened
        if csv_file is not None:
            csv_file.close()

    return results

import csv
import numpy as np
from pathlib import Path

def load_csv_results_merged(csv_file, fields=None):
    """
    Load chunked results from a CSV file produced by process_large_wav_in_chunks
    and merge them into single arrays.

    Parameters
    ----------
    csv_file : str or Path
        Path to the CSV file generated by process_large_wav_in_chunks(..., output_csv_path=...).
    fields : list of str or None
        Which fields to include in the merged output. Valid options are:
            ["time", "signal", "phase", "frequency"].
        If None, defaults to all four fields.

    Returns
    -------
    merged_data : dict
        A dictionary with only the requested fields as keys. For instance:
            {
                "time": <1D ndarray>,
                "signal": <1D ndarray>,
                "phase": <1D ndarray>,
                "frequency": <1D ndarray>
            }
        The arrays are concatenated in ascending chunk_index order.
        Any fields not requested are omitted.
    """
    if fields is None:
        fields = ["time", "signal", "phase", "frequency"]

    valid_fields = {"time", "signal", "phase", "frequency"}
    # Validate requested fields
    for f in fields:
        if f not in valid_fields:
            raise ValueError(f"Invalid field '{f}'. Valid fields: {valid_fields}")

    csv_file = Path(csv_file)
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    # We'll group data by chunk_index first, then merge in ascending order.
    chunks_data = {}  # chunk_idx -> dict of lists

    with open(csv_file, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError("CSV file is empty or malformed.")

        # We expect columns:
        #  "chunk_index", "time_seconds", "signal_amplitude", "phase_radians", "frequency_rad_s"
        try:
            ci_col = header.index("chunk_index")
            t_col  = header.index("time_seconds")
            s_col  = header.index("signal_amplitude")
            p_col  = header.index("phase_radians")
            f_col  = header.index("frequency_rad_s")
        except ValueError as e:
            raise ValueError(
                "CSV must contain columns: "
                "'chunk_index', 'time_seconds', 'signal_amplitude', "
                "'phase_radians', 'frequency_rad_s'"
            ) from e

        for row in reader:
            chunk_idx = int(row[ci_col])

            # If the chunk hasn't been seen yet, init structure
            if chunk_idx not in chunks_data:
                chunks_data[chunk_idx] = {
                    "time": [],
                    "signal": [],
                    "phase": [],
                    "frequency": []
                }

            # If a field is requested, parse and store it
            if "time" in fields:
                chunks_data[chunk_idx]["time"].append(float(row[t_col]))

            if "signal" in fields:
                chunks_data[chunk_idx]["signal"].append(float(row[s_col]))

            if "phase" in fields:
                chunks_data[chunk_idx]["phase"].append(float(row[p_col]))

            if "frequency" in fields:
                chunks_data[chunk_idx]["frequency"].append(float(row[f_col]))

    # Now we concatenate data in ascending chunk_index order
    merged_data = {}
    # Prepare lists to later convert into arrays
    merged_lists = {f: [] for f in fields}

    # Sort chunks by chunk_idx
    for chunk_idx in sorted(chunks_data.keys()):
        cdata = chunks_data[chunk_idx]
        for f in fields:
            merged_lists[f].extend(cdata[f])

    # Convert to numpy arrays
    for f in fields:
        # Decide on a dtype per field
        if f == "time":
            merged_data[f] = np.array(merged_lists[f], dtype=np.float64)
        else:
            # signal, phase, frequency
            merged_data[f] = np.array(merged_lists[f], dtype=np.float32)

    return merged_data


import re
import numpy as np
from pathlib import Path

def load_chunked_npz_merged(directory, fields=None, chunk_indices=None):
    """
    Load results from one or more .npz chunk files (e.g. chunk_1.npz, chunk_2.npz, ...)
    in `directory` and merge them into single arrays.

    Parameters
    ----------
    directory : str or Path
        Folder containing chunk_{i}.npz files, as produced by process_large_wav_in_chunks(..., output_binary_dir=...).
    fields : list of str or None
        Which fields to include in the merged output. Valid options:
            ["time", "signal", "phase", "frequency"].
        If None, defaults to all four.
    chunk_indices : list of int or None
        If None, load all chunk_X.npz found. Otherwise, only load those chunk indices.

    Returns
    -------
    merged_data : dict
        A dictionary with only the requested fields as keys, each containing a 1D ndarray
        with data concatenated in ascending chunk_index order. For example:
            {
                "time": <np.ndarray>,
                "signal": <np.ndarray>,
                "phase": <np.ndarray>,
                "frequency": <np.ndarray>
            }
        If you omit some fields from `fields`, those keys won't appear in merged_data.
    """
    if fields is None:
        fields = ["time", "signal", "phase", "frequency"]

    valid_fields = {"time", "signal", "phase", "frequency"}
    for f in fields:
        if f not in valid_fields:
            raise ValueError(f"Invalid field '{f}'. Valid fields: {valid_fields}")

    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {directory}")

    pattern = re.compile(r"^chunk_(\d+)\.npz$")

    # Gather all matching chunk files
    chunk_files = []
    for f in directory.iterdir():
        if f.is_file():
            match = pattern.match(f.name)
            if match:
                cidx = int(match.group(1))
                if chunk_indices is None or cidx in chunk_indices:
                    chunk_files.append((cidx, f))

    # Sort by chunk index
    chunk_files.sort(key=lambda x: x[0])

    # We'll build lists for each requested field, then concatenate
    merged_lists = {f: [] for f in fields}

    for cidx, npz_file in chunk_files:
        data = np.load(npz_file)
        # Typically, data keys: "chunk_index", "time", "signal", "phase", "frequency"
        # We'll only retrieve what's requested in `fields`.
        for f in fields:
            if f in data:
                # 'time' -> float64, others -> float32
                if f == "time":
                    merged_lists[f].extend(data[f].astype(np.float64))
                else:
                    merged_lists[f].extend(data[f].astype(np.float32))
            else:
                # If the NPZ doesn't contain the requested field, we have a mismatch
                raise KeyError(f"Field '{f}' not found in file: {npz_file}")

    # Create final dictionary of concatenated arrays
    merged_data = {}
    for f in fields:
        if f == "time":
            merged_data[f] = np.array(merged_lists[f], dtype=np.float64)
        else:
            merged_data[f] = np.array(merged_lists[f], dtype=np.float32)

    return merged_data


import os
import numpy as np
import wave

def create_sine_wave_wav(filename, file_path, amplitude, frequency, duration,
                           sample_rate=44100, bit_depth=16, stereo=False):
    """
    Creates a WAV file containing a sine wave.

    Parameters:
      filename (str): The name of the WAV file to create ('.wav' will be appended if not present).
      file_path (str): The directory path where the WAV file will be saved.
      amplitude (float): The amplitude of the sine wave (a value between 0 and 1).
      frequency (float): The frequency of the sine wave in Hertz (must be positive).
      duration (float): The length of the sine wave in seconds (must be positive).
      sample_rate (int): The sample rate in Hz (default is 44100). Must be a positive integer.
      bit_depth (int): Bits of resolution (default 16). Supported values: 8 or 16.
      stereo (bool): If True, create a stereo file (with the same sine wave duplicated on both channels);
                     if False, create a mono file.
    
    Raises:
      ValueError: If any of the parameters are invalid.
    
    On success, the function prints a success message with the full file path.
    """
    
    # --- Input Validation ---
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Invalid filename provided.")
    
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("Invalid file path provided.")
    
    if not os.path.isdir(file_path):
        raise ValueError("The specified file path does not exist.")
    
    if not (isinstance(sample_rate, int) and sample_rate > 0):
        raise ValueError("Sample rate must be a positive integer.")
    
    if not (isinstance(bit_depth, int) and bit_depth in [8, 16]):
        raise ValueError("Bit depth must be either 8 or 16.")
    
    if not (isinstance(amplitude, (int, float)) and 0 <= amplitude <= 1):
        raise ValueError("Amplitude must be a number between 0 and 1.")
    
    if not (isinstance(frequency, (int, float)) and frequency > 0):
        raise ValueError("Frequency must be a positive number.")
    
    if not (isinstance(duration, (int, float)) and duration > 0):
        raise ValueError("Duration must be a positive number.")
    
    if not isinstance(stereo, bool):
        raise ValueError("Stereo parameter must be a boolean.")
    
    # Ensure the filename ends with .wav
    if not filename.lower().endswith('.wav'):
        filename += '.wav'
    full_file_path = os.path.join(file_path, filename)
    
    # --- Generate the Sine Wave Data ---
    # Calculate the number of samples.
    n_samples = int(sample_rate * duration)
    # Create an array of time values.
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # Generate the sine wave (values will be in the range [-1, 1]).
    sine_wave = np.sin(2 * np.pi * frequency * t)
    
    # Scale the sine wave data to the appropriate integer range.
    if bit_depth == 16:
        # For 16-bit PCM, the values are in the range -32768 to 32767.
        max_int_value = np.iinfo(np.int16).max  # 32767
        samples = (sine_wave * amplitude * max_int_value).astype(np.int16)
    elif bit_depth == 8:
        # For 8-bit PCM, the values are unsigned [0, 255] with 128 as the midpoint.
        samples = ((sine_wave * amplitude * 127) + 128).astype(np.uint8)
    
    # If stereo, duplicate the mono samples into two channels.
    if stereo:
        # Stack the same data side-by-side.
        samples = np.column_stack((samples, samples))
    
    # --- Write the Data to a WAV File ---
    try:
        with wave.open(full_file_path, 'wb') as wav_file:
            n_channels = 2 if stereo else 1
            sampwidth = bit_depth // 8  # bytes per sample
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())
        print(f"WAV file created successfully: {full_file_path}")
    except Exception as e:
        print(f"An error occurred while writing the WAV file: {e}")


# --------------------------------------------------------------------------
# Example usage (assuming you have sc_output, sc_input, and sample_rate)

# delay_samples, delay_seconds = find_delay_by_amplitude(sc_output, sc_input, sample_rate=44100)
# print(f"Detected delay: {delay_samples} samples, which is {delay_seconds:.3f} seconds.")

def test_find_delay_by_amplitude():
    # Open the wav file
    file_path = '/home/user/data/sound_test_with_chirp.wav'
    show_plots = False
                        

    # Open the wav file
    wav_file = wave.open(file_path, 'r')

    # Extract Raw Audio from Wav File
    signal = wav_file.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int16)

    # Get the number of channels
    channels = wav_file.getnchannels()

    # Split the data into channels
    if channels == 2:
        signal = np.reshape(signal, (-1, 2))
        sc_output = signal[:, 0]
        sc_input = signal[:, 1]
    else:
        print('The file does not have 2 channels')
        sys.exit(0)
    # get the sample rate of the wav file
    sample_rate = wav_file.getframerate()

    # Create interactive plot
    if show_plots:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=sc_output, mode='lines', name='Sound Card Output'))
        fig.add_trace(go.Scatter(y=sc_input, mode='lines', name='Sound Card Input'))
        fig.update_layout(title='Channels Plot', xaxis_title='Sample Index', yaxis_title='Amplitude')
        fig.show()

    [delay_idx, delay_seconds]= find_delay_by_amplitude(sc_output, sc_input, sample_rate=sample_rate, noise_window=1000, threshold_factor=6)
    # print delay in samples and seconds
    print(f"Detected delay: {delay_idx} samples, which is {delay_seconds:.5f} seconds.")


from pathlib import Path  # Add at top of file if not present

def test_process_large_wav_in_chunks():
    filename = 'sound_test_with_sine_baseline_20250112.wav'
    filename = 'sound_test_with_sine_ice_20250112.wav'
    filename = 'sine_wave_120_sec_4KHz.wav'
    # output_csv_path has the base name of filename without the extension and with a postfix of _phases.csv
    output_csv_path = Path(filename).stem + "_phases.csv"
    
    
    file_path = Path('/home/user/data')
    full_file_path = file_path / filename 
    full_output_csv_path = file_path / output_csv_path
    results = process_large_wav_in_chunks(
        filename=full_file_path,
        channel=0,
        chunk_size=65536*16,
        threshold=None,
        min_below_count=200,
        override_sample_rate=None,
        output_csv_path=full_output_csv_path  # or None, if you don't want file output
    )

    # Print how many chunks were processed
    print(f"Processed {len(results)} non-silent chunks.")
    # For instance, inspect the first chunk's data:
    if len(results) > 0:
        first_chunk = results[0]
        print("First chunk index:", first_chunk["chunk_index"])
        print("Time array shape:", first_chunk["time"].shape)
        print("Phase array shape:", first_chunk["phase"].shape)
        print("Frequency array shape:", first_chunk["frequency"].shape)
        print("Start time:", first_chunk["start_time"])
        print("End time:", first_chunk["end_time"])



# Example usage:
def test_create_sine_wave_wav():
    # Parameters for the sine wave:
    filename = "/home/user/data/sine_wave_120_sec_4KHz.wav"
    file_path = "."  # current directory
    amplitude = 0.8       # 80% of maximum amplitude
    frequency = 4000       # A4 note, 440 Hz
    duration = 120          # 2 seconds long
    sample_rate = 44100   # CD quality
    bit_depth = 16        # 16-bit resolution
    stereo = True         # stereo output

    create_sine_wave_wav(filename, file_path, amplitude, frequency, duration,
                           sample_rate=sample_rate, bit_depth=bit_depth, stereo=stereo)



if __name__ == '__main__':
    # test_create_sine_wave_wav()
    test_process_large_wav_in_chunks()
    # test_find_delay_by_amplitude()
    # test_find_shift_between_signals_and_shift_signal()
