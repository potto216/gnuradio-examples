#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from numpy.fft import fft, fftshift, fftfreq

show_plots = False
save_plots = True

# =======================
# User-Specified Settings
# =======================
# User-specified flag to include FIR Hilbert transform in the combined frequency plot.
include_fir_in_combined = False  # Set to False to hide FIR Hilbert plot in the combined graph

model_type = "wavfile"  # Use "wavfile" to load a WAV file signal
#model_type = "narrowband"
#model_type = "wideband"
if model_type=="wavfile":
    import soundfile as sf
    filename = "/home/user/data/sound_test_with_sine_baseline_20250112.wav"  # Ensure this WAV file exists in the working directory
    filename = "/home/user/data/sound_test_12KHz_20250213.wav"
    x_real, fs = sf.read(filename)
    # If the WAV file has more than one channel, take the first one.
    if x_real.ndim > 1:
        x_real = x_real[:, 0]
    N = len(x_real)
    t = np.arange(N) / fs
    f0 = 12000
    # Ideal frequency for the WAV file signal (Hz)
    frequency_zoom_hz = [f0-5, f0+5]
    frequency_zoom_hz = None
    # User-defined frequency range where the signal is present.
    signal_freq_range = [f0-5, f0+5]
elif model_type=="narrowband":
    fs = 44100                # Sampling rate (Hz)
    duration = 1.0            # Signal duration (seconds)
    N = int(fs * duration)    # Number of samples
    t = np.arange(N) / fs     # Time vector

    f0 = 12000  # Nominal frequency (Hz)
    amp_mod = 1.0 + 0.5 * np.sin(1 * np.pi * 2 * t)
    phase_mod = 0.2 * np.sin(1 * np.pi * 5 * t)
    noise_std = 0.0001
    signal_freq_range = [f0-5, f0+5]
    frequency_zoom_hz = None
elif model_type=="wideband":
    fs = 44100
    duration = 1.0
    N = int(fs * duration)
    t = np.arange(N) / fs

    f0 = 12000  # Nominal frequency (Hz)
    amp_mod = 1.0 + 0.5 * np.sin(2 * np.pi * 2 * t)
    phase_mod = 0.2 * np.sin(70 * np.pi * 5 * t)
    noise_std = 0.0001
    signal_freq_range = [f0-5, f0+5]
    frequency_zoom_hz = None
else:
    raise ValueError("Invalid model type. Choose 'wavfile', 'narrowband' or 'wideband'.")


# FIR Hilbert transformer parameter (number of taps, must be odd)
M = 101
if M % 2 == 0:
    raise ValueError("FIR filter length M must be odd.")

# ==============================
# Signal Generation or Loading
# ==============================
if model_type=="wavfile":
    # The WAV file has been loaded above. (x_real, fs, N, and t are defined.)
    ideal_wavfile_frequency_hz=f0
    pass
else:
    # Generate a synthetic complex signal for narrowband or wideband models.
    true_phase = 2 * np.pi * f0 * t + phase_mod
    # Note: The true instantaneous frequency is not used in the wavfile case.
    x_complex = amp_mod * np.exp(1j * true_phase)
    x_real = np.real(x_complex) + np.random.normal(0, noise_std, size=t.shape)
    ideal_wavfile_frequency_hz=f0

# ==============================
# FFT Utility Function
# ==============================
def compute_fft(signal, fs):
    """Compute FFT and return frequency axis, magnitude (linear) and magnitude in dB."""
    N = len(signal)
    X = fft(signal)
    X_shifted = fftshift(X)
    freqs = fftshift(fftfreq(N, d=1/fs))
    eps = 1e-12
    mag = np.abs(X_shifted)
    mag_dB = 20 * np.log10(mag + eps)
    return freqs, mag, mag_dB

# ==============================
# Time-Domain Plot: Signal Before Filtering
# ==============================
num_cycles_to_show = 5  # Number of cycles to display    
T0 = 1 / f0  # Period of one cycle
if model_type=="wavfile":
    # For a wavfile, plot a portion of the signal.
    # For synthetic signals, plot both the real and complex parts.
    samples_to_show = min(int(num_cycles_to_show * T0 * fs),len(t))
    t_zoom = t[:samples_to_show]
    x_real_zoom = x_real[:samples_to_show]
       
    plt.figure(figsize=(10, 6))
    plt.plot(t_zoom, x_real_zoom, label=None, color='b', lw=2)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(f"Time-Domain Plot of Recorded {f0} Hz Sine Wave")
    #plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if show_plots:
        plt.show()
    if save_plots:
        image_filepath = f"images/{model_type}"
        if not os.path.exists(image_filepath):
            os.makedirs(image_filepath)
        plt.savefig(f"{image_filepath}/wavfile_signal.png", dpi=300)
        plt.savefig(f"{image_filepath}/wavfile_signal.svg", format='svg')    
else:
    # For synthetic signals, plot both the real and complex parts.
    samples_to_show = min(int(num_cycles_to_show * T0 * fs),len(t))
    t_zoom = t[:samples_to_show]
    x_real_zoom = x_real[:samples_to_show]
    x_complex_zoom = x_complex[:samples_to_show]

    plt.figure(figsize=(10, 6))
    plt.plot(t_zoom, x_real_zoom, label="Real Part of Signal", color='b', lw=2)
    plt.plot(t_zoom, np.real(x_complex_zoom), '--', label="Real Part of Complex Signal", color='g', lw=1.5)
    plt.plot(t_zoom, np.imag(x_complex_zoom), ':', label="Imaginary Part of Complex Signal", color='r', lw=1.5)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Time-Domain Plot: Real and Complex Signal Before Filtering")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if show_plots:
        plt.show()
    if save_plots:
        image_filepath = f"images/{model_type}"
        if not os.path.exists(image_filepath):
            os.makedirs(image_filepath)
        plt.savefig(f"{image_filepath}/real_vs_complex_signal.png", dpi=300)
        plt.savefig(f"{image_filepath}/real_vs_complex_signal.svg", format='svg')    


# Compute FFT of the real signal (and complex signal if available)
freqs, X_real_mag, X_real_dB = compute_fft(x_real, fs)
if model_type!="wavfile":
    _, X_complex_mag, X_complex_dB = compute_fft(x_complex, fs)

# ==============================
# Plot 1: Signal Spectrum
# ==============================
plt.figure(figsize=(10, 6))

peak_magnitude = np.max(X_real_dB)
    # Add vertical dotted line at ideal_wavfile_frequency_hz
plt.axvline(ideal_wavfile_frequency_hz, color='r', linestyle='--', lw=1.5, label=f"{f0} Hz")
plt.plot(freqs, X_real_dB, label="Real Signal Spectrum", lw=1.5)
if model_type!="wavfile":
    plt.plot(freqs, X_complex_dB, label="Original Complex Signal Spectrum", lw=1.5, linestyle='--')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title(f"Spectrum of  of Recorded {f0} Hz Sine Wave")
plt.legend()
plt.grid(True)
if frequency_zoom_hz is not None:
    plt.xlim(frequency_zoom_hz[0], frequency_zoom_hz[1])
else:
    plt.xlim(-fs/2, fs/2)
plt.tight_layout()
if show_plots:
    plt.show()
if save_plots:
    image_filepath = f"images/{model_type}"
    if not os.path.exists(image_filepath):
        os.makedirs(image_filepath)
    plt.savefig(f"{image_filepath}/real_vs_complex_spectrum.png", dpi=300)
    plt.savefig(f"{image_filepath}/real_vs_complex_spectrum.svg", format='svg')    


# =============================
# Plot 2: Hilbert FFT filter  (Using scipy.signal.hilbert)
# =============================
hilbert_fft_signal = hilbert(x_real)
freqs, analytic_mag, analytic_dB = compute_fft(hilbert_fft_signal, fs)

# Revised Ideal Hilbert Frequency Response (using a delta input)
delta = np.zeros_like(x_real)
delta[len(delta)//2] = 1.0
delta_analytic = hilbert(delta)
_, delta_analytic_mag, delta_analytic_dB = compute_fft(delta_analytic, fs)
scale_factor = np.max(analytic_mag)
H_ideal = delta_analytic_mag * (scale_factor / np.max(delta_analytic_mag))
H_ideal_dB = 20 * np.log10(H_ideal + 1e-12)

plt.figure(figsize=(10, 6))
plt.plot(freqs, X_real_dB, label="Recorded Signal", lw=1.0, alpha=0.7)
plt.plot(freqs, analytic_dB, label="Filtered Signal", lw=1.5)
plt.plot(freqs, H_ideal_dB, 'k--', label="Hilbert Filter (FFT) Response", lw=1.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Spectrum of Recorded Signal and Filtered Signal by FFT Hilbert Transform")
plt.legend()
plt.grid(True)

if frequency_zoom_hz is not None:
    plt.xlim(frequency_zoom_hz[0], frequency_zoom_hz[1])
    # Get indices for the specified frequency range.
    idx = (freqs >= frequency_zoom_hz[0]) & (freqs <= frequency_zoom_hz[1])
    # Combine dB data from all signals within the zoom window.
    combined_dB = np.concatenate((X_real_dB[idx], analytic_dB[idx], H_ideal_dB[idx]))
    y_min = np.min(combined_dB)
    y_max = np.max(combined_dB)
    # Set margin as 10% of the dynamic range.
    margin = 0.1 * (y_max - y_min)
    plt.ylim(y_min - margin, y_max + margin)
else:
    plt.xlim(-fs/2, fs/2)

plt.tight_layout()
if show_plots:
    plt.show()
if save_plots:
    image_filepath = f"images/{model_type}"
    if not os.path.exists(image_filepath):
        os.makedirs(image_filepath)
    plt.savefig(f"{image_filepath}/hilbert_transform_analysis.png", dpi=300)
    plt.savefig(f"{image_filepath}/hilbert_transform_analysis.svg", format='svg')    
        

# ==============================
# Plot 3: FIR Hilbert Transformer Analysis (Complex FIR for Analytic Signal)
# ==============================
m = np.arange(-((M-1)//2), ((M-1)//2) + 1)
h_real = np.zeros(M)
h_imag = np.zeros(M)
for i, mi in enumerate(m):
    if mi == 0:
        h_real[i] = 1.0
    elif mi % 2 != 0:
        h_imag[i] = 2 / (np.pi * mi)
    else:
        h_imag[i] = 0.0

window = np.hamming(M)
h_imag_windowed = h_imag * window
h_fir = h_real + 1j * h_imag_windowed

x_fir = np.convolve(x_real, h_fir, mode='same')
freqs, x_fir_mag, x_fir_dB = compute_fft(x_fir, fs)

H_fir = fftshift(fft(h_fir, n=1024))
freq_h = np.linspace(-fs/2, fs/2, 1024)
scale_fir = np.max(np.abs(x_fir_dB))
H_fir_dB = 20 * np.log10(np.abs(H_fir) + 1e-12)
H_fir_scaled = H_fir_dB + (scale_fir - np.max(H_fir_dB))
H_fir_dB = H_fir_scaled

plt.figure(figsize=(10, 6))
plt.plot(freqs, X_real_dB, label="Recorded Signal", lw=1.0, alpha=0.7)
plt.plot(freqs, x_fir_dB, label="Filtered Signal", lw=1.5)
plt.plot(freq_h, H_fir_dB, 'k--', label=" Hilbert FIR Filter Response", lw=1.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Spectrum of Recorded Signal and Filtered Signal by FIR Hilbert Transform")
plt.legend()
plt.grid(True)

if frequency_zoom_hz is not None:
    plt.xlim(frequency_zoom_hz[0], frequency_zoom_hz[1])
    # Get indices for the specified frequency range.
    idx = (freqs >= frequency_zoom_hz[0]) & (freqs <= frequency_zoom_hz[1])
    # Combine dB data from all signals within the zoom window.
    combined_dB = np.concatenate((X_real_dB[idx], analytic_dB[idx], H_ideal_dB[idx]))
    y_min = np.min(combined_dB)
    y_max = np.max(combined_dB)
    # Set margin as 10% of the dynamic range.
    margin = 0.1 * (y_max - y_min)
    plt.ylim(y_min - margin, y_max + margin)
else:
    plt.xlim(-fs/2, fs/2)

plt.tight_layout()
if show_plots:
    plt.show()
if save_plots:
    image_filepath = f"images/{model_type}"
    if not os.path.exists(image_filepath):
        os.makedirs(image_filepath)
    plt.savefig(f"{image_filepath}/fir_hilbert_transformer_analysis.png", dpi=300)
    plt.savefig(f"{image_filepath}/fir_hilbert_transformer_analysis.svg", format='svg')    

# ==============================
# Combined Frequency Plot: Signal, FFT Hilbert Transform, and FIR Hilbert Transform
# ==============================
plt.figure(figsize=(10, 6))

# Plot the real signal spectrum
plt.plot(freqs, X_real_dB, label="Recorded Signal", lw=1.0, alpha=0.7)

# Plot the FFT-based Hilbert transform spectrum
plt.plot(freqs, analytic_dB, label="FFT Hilbert Transform", lw=1.5)

if include_fir_in_combined:
    # Plot the FIR-based Hilbert transform spectrum
    plt.plot(freqs, x_fir_dB, label="FIR Hilbert Transform", lw=1.5, linestyle="--")

# Add vertical dotted line at ideal_wavfile_frequency_hz if available
if 'ideal_wavfile_frequency_hz' in globals():
    plt.axvline(ideal_wavfile_frequency_hz, color='r', linestyle='--', lw=1.5, label=f"{ideal_wavfile_frequency_hz} Hz")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Spectrum Comparison: Original Signal vs. Hilbert Transform Methods")
plt.legend()
plt.grid(True)

# Set x-axis limits based on user-defined zoom range
if frequency_zoom_hz is not None:
    plt.xlim(frequency_zoom_hz[0], frequency_zoom_hz[1])
    
    # Get indices for the specified frequency range
    idx = (freqs >= frequency_zoom_hz[0]) & (freqs <= frequency_zoom_hz[1])
    
    # Combine dB data from all signals within the zoom window
    combined_dB = np.concatenate((X_real_dB[idx], analytic_dB[idx], x_fir_dB[idx]))
    
    # Set y-axis limits with a margin of 10% of the dynamic range
    y_min = np.min(combined_dB)
    y_max = np.max(combined_dB)
    margin = 0.1 * (y_max - y_min)
    plt.ylim(y_min - margin, y_max + margin)
else:
    plt.xlim(-fs/2, fs/2)

plt.tight_layout()
if show_plots:
    plt.show()
if save_plots:
    image_filepath = f"images/{model_type}"
    if not os.path.exists(image_filepath):
        os.makedirs(image_filepath)
    plt.savefig(f"{image_filepath}/combined_hilbert_transform_analysis.png", dpi=300)
    plt.savefig(f"{image_filepath}/combined_hilbert_transform_analysis.svg", format='svg')    

# ==============================
# Noise Spectrum and Noise Statistics Investigation
# ==============================


# Compute the FFT of the original real signal
N = len(x_real)
X_orig = np.fft.fft(x_real)
f_axis = np.fft.fftfreq(N, d=1/fs)

# Create a copy of the FFT to extract the noise
X_noise = X_orig.copy()

# Zero out the frequency components in the user-defined signal range.
# This zeros out both positive and negative frequencies within the range.
mask = (np.abs(f_axis) >= (signal_freq_range[0]-2000)) & (np.abs(f_axis) <= (signal_freq_range[1]+2000))
X_noise[mask] = 0

# Compute the noise spectrum (shift for plotting)
X_noise_shifted = fftshift(X_noise)
f_axis_shifted = fftshift(f_axis)
noise_mag = np.abs(X_noise_shifted)
noise_dB = 20 * np.log10(noise_mag + 1e-12)

# Plot the noise frequency spectrum
plt.figure(figsize=(10, 6))
plt.plot(f_axis_shifted, noise_dB, label="Noise Spectrum", lw=1.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Noise Frequency Spectrum (Signal Frequency Range Removed)")
plt.legend()
plt.grid(True)
# if frequency_zoom_hz is not None:
#     plt.xlim(frequency_zoom_hz[0], frequency_zoom_hz[1])
# else:
plt.xlim(-fs/2, fs/2)

plt.tight_layout()
if show_plots:
    plt.show()
if save_plots:
    image_filepath = f"images/{model_type}"
    if not os.path.exists(image_filepath):
        os.makedirs(image_filepath)
    plt.savefig(f"{image_filepath}/noise_spectrum.png", dpi=300)
    plt.savefig(f"{image_filepath}/noise_spectrum.svg", format='svg')


# Inverse FFT to convert the noise spectrum back to the time domain
noise_time = np.fft.ifft(X_noise)
noise_time = np.real(noise_time)

# Plot a histogram to investigate the noise statistics in the time domain
# Compute the 2.5th and 97.5th percentiles to define the central 95% range
low_bound = np.percentile(noise_time, 0.05)
high_bound = np.percentile(noise_time, 99.95)
noise_time_95 = noise_time[(noise_time >= low_bound) & (noise_time <= high_bound)]

# Create a figure with two subplots (vertical layout)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Top subplot: Histogram of the full noise_time signal
ax1.hist(noise_time, bins=150, color='g', alpha=0.7, label="Noise Histogram")
ax1.set_xlabel("Amplitude")
ax1.set_ylabel("Count")
ax1.set_title("Histogram of Noise Time-Domain Signal")
ax1.legend()
ax1.grid(True)

# Bottom subplot: Histogram of the central 95% data (without the 5% outliers)
ax2.hist(noise_time_95, bins=150, color='b', alpha=0.7, label="Noise Histogram (Central 95%)")
ax2.set_xlabel("Amplitude")
ax2.set_ylabel("Count")
ax2.set_title("Histogram of Noise Time-Domain Signal (Central 95% Data)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
if show_plots:
    plt.show()
if save_plots:
    image_filepath = f"images/{model_type}"
    if not os.path.exists(image_filepath):
        os.makedirs(image_filepath)
    plt.savefig(f"{image_filepath}/noise_time_histogram.png", dpi=300)
    plt.savefig(f"{image_filepath}/noise_time_histogram.svg", format='svg')


# ==============================
# Plot 4: Instantaneous Frequency Comparison and Difference (Two Subplots)
# ==============================
# Compute instantaneous frequency from the Hilbert transform and FIR-filtered signal.
phase_analytic = np.unwrap(np.angle(hilbert_fft_signal))
inst_freq_analytic = np.gradient(phase_analytic) * fs / (2 * np.pi)

phase_fir = np.unwrap(np.angle(x_fir))
inst_freq_fir = np.gradient(phase_fir) * fs / (2 * np.pi)

# Trim a fraction of the data to reduce endpoint boundary effects.
trim_ratio = 0.05  # Remove 5% of samples at beginning and end
trim_samples = int(trim_ratio * len(t))
t_trim = t[trim_samples:-trim_samples]
inst_freq_analytic_trim = inst_freq_analytic[trim_samples:-trim_samples]
inst_freq_fir_trim = inst_freq_fir[trim_samples:-trim_samples]

# Compute the difference between FIR and Hilbert estimates.
error_diff = inst_freq_fir_trim - inst_freq_analytic_trim

# Create two subplots:
#   Top: Instantaneous frequency vs. time.
#   Bottom: Difference (error) between FIR and Hilbert instantaneous frequencies vs. time.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

if include_fir_in_combined:
    ax1.plot(t_trim, inst_freq_fir_trim, linestyle=':', lw=1.5, label="FIR Implementation")
    ax1.plot(t_trim, inst_freq_analytic_trim, linestyle='--', lw=1.5, label="FFT Implementation")
else:
    ax1.plot(t_trim, inst_freq_analytic_trim,  color='orange',linestyle='--', lw=1.5, label=None)
    
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Frequency (Hz)")
if include_fir_in_combined:
    ax1.set_title("Instantaneous Frequency Comparison")
else:
    ax1.set_title("Instantaneous Frequency")
ax1.legend()
ax1.grid(True)

if include_fir_in_combined:
    ax2.plot(t_trim, inst_freq_fir_trim, linestyle=':', lw=1.5, label="FIR Implementation")
    ax2.plot(t_trim, inst_freq_analytic_trim, linestyle='--', lw=1.5, label="FFT Implementation")
else:
    ax2.plot(t_trim, inst_freq_analytic_trim, color='orange', linestyle='-', lw=1.5, label=None)

ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Frequency (Hz)")
if include_fir_in_combined:
    ax2.set_title("Instantaneous Frequency Comparison (Zoomed)")
else:
    ax2.set_title("Instantaneous Frequency (Zoomed)")
ax2.legend()
ax2.grid(True)
# zoom in the time axis in the midpoint of time for a few cycles
midpoint = len(t_trim)//2
zoom_samples = int(num_cycles_to_show * T0 * fs*10)
ax2.set_xlim(t_trim[midpoint-zoom_samples], t_trim[midpoint+zoom_samples])


plt.tight_layout()
if show_plots:
    plt.show()
if save_plots:
    image_filepath = f"images/{model_type}"
    if not os.path.exists(image_filepath):
        os.makedirs(image_filepath)
    plt.savefig(f"{image_filepath}/instantaneous_frequency_comparison_subplots.png", dpi=300)
    plt.savefig(f"{image_filepath}/instantaneous_frequency_comparison_subplots.svg", format='svg')    
    
    

# ==============================
# Plot enerate a histogram of inst_freq_analytic_trim and compute relevant statistics:
# ==============================

mean_freq = np.mean(inst_freq_analytic_trim)
std_freq = np.std(inst_freq_analytic_trim)
min_freq = np.min(inst_freq_analytic_trim)
max_freq = np.max(inst_freq_analytic_trim)
median_freq = np.median(inst_freq_analytic_trim)

# Print statistics
print(f"Instantaneous Frequency Statistics Model: {model_type}:")
print(f"Mean: {mean_freq:.2f} Hz")
print(f"Standard Deviation: {std_freq:.2f} Hz")
print(f"Minimum: {min_freq:.2f} Hz")
print(f"Maximum: {max_freq:.2f} Hz")
print(f"Median: {median_freq:.2f} Hz")

# Create histogram plot
plt.figure(figsize=(10, 6))
plt.hist(inst_freq_analytic_trim, bins=50, color='orange', alpha=0.7, label="Instantaneous Frequency Histogram")
plt.xlabel("Instantaneous Frequency (Hz)")
plt.ylabel("Count")
plt.title("Histogram of Instantaneous Frequency (Hilbert Transform)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show and save the plot
if show_plots:
    plt.show()
if save_plots:
    image_filepath = f"images/{model_type}"
    if not os.path.exists(image_filepath):
        os.makedirs(image_filepath)
    plt.savefig(f"{image_filepath}/inst_freq_analytic_histogram.png", dpi=300)
    plt.savefig(f"{image_filepath}/inst_freq_analytic_histogram.svg", format='svg')
