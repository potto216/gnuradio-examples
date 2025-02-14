#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from numpy.fft import fft, fftshift, fftfreq

show_plots = False
save_plots = True

# =======================
# User-Specified Settings
# =======================
model_type = "wideband"  # "sinusoidal" or "complex"
if model_type=="narrowband":
    fs = 44100                # Sampling rate (Hz)
    duration = 1.0            # Signal duration (seconds)
    N = int(fs * duration)    # Number of samples
    t = np.arange(N) / fs     # Time vector

    # Nominal frequency of the complex exponential (Hz)
    f0 = 12000

    # Amplitude modulation function (user specifiable)
    amp_mod = 1.0 + 0.5 * np.sin(1 * np.pi * 2 * t)

    # Phase modulation function (user specifiable)
    phase_mod = 0.2 * np.sin(1 * np.pi * 5 * t)

    # Gaussian noise standard deviation (user specifiable)
    noise_std = 0.0001
elif model_type=="wideband":
    fs = 44100                # Sampling rate (Hz)
    duration = 1.0            # Signal duration (seconds)
    N = int(fs * duration)    # Number of samples
    t = np.arange(N) / fs     # Time vector

    # Nominal frequency of the complex exponential (Hz)
    f0 = 12000

    # Amplitude modulation function (user specifiable)
    amp_mod = 1.0 + 0.5 * np.sin(2 * np.pi * 2 * t)

    # Phase modulation function (user specifiable)
    phase_mod = 0.2 * np.sin(70 * np.pi * 5 * t)

    # Gaussian noise standard deviation (user specifiable)
    noise_std = 0.0001
else:
    raise ValueError("Invalid model type. Choose 'narrowband' or 'wideband'.")

# FIR Hilbert transformer parameter (number of taps, must be odd)
M = 101
if M % 2 == 0:
    raise ValueError("FIR filter length M must be odd.")

# ==============================
# Generate the modified complex exponential
# ==============================
true_phase = 2 * np.pi * f0 * t + phase_mod
true_instantaneous_frequency = np.gradient(true_phase) * fs / (2 * np.pi)
x_complex = amp_mod * np.exp(1j * true_phase)
x_real = np.real(x_complex) + np.random.normal(0, noise_std, size=t.shape)

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
# Time-Domain Plot: Real and Complex Signal Before Filtering
# ==============================
num_cycles_to_show = 5  # Number of cycles to display
T0 = 1 / f0  # Period of one cycle
samples_to_show = int(num_cycles_to_show * T0 * fs)

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
    plt.savefig("images/real_vs_complex_signal.png", dpi=300)
    plt.savefig("images/real_vs_complex_signal.svg", format='svg')

# Compute FFTs for later plotting.
freqs, X_complex_mag, X_complex_dB = compute_fft(x_complex, fs)
_, X_real_mag, X_real_dB = compute_fft(x_real, fs)

# ==============================
# Plot 1: Real Signal vs. Original Complex Signal Spectrum
# ==============================
plt.figure(figsize=(10, 6))
plt.plot(freqs, X_complex_dB, label="Original Complex Signal Spectrum", lw=1.5, linestyle='--')
plt.plot(freqs, X_real_dB, label="Real Signal Spectrum", lw=1.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Spectrum: Real Signal vs. Original Complex Signal")
plt.legend()
plt.grid(True)
plt.xlim(-fs/2, fs/2)
plt.tight_layout()
if show_plots:
    plt.show()
if save_plots:
    plt.savefig("images/real_vs_complex_spectrum.png", dpi=300)
    plt.savefig("images/real_vs_complex_spectrum.svg", format='svg')

# ==============================
# Plot 2: Hilbert Transform Analysis (Using scipy.signal.hilbert)
# ==============================
analytic_signal = hilbert(x_real)
freqs, analytic_mag, analytic_dB = compute_fft(analytic_signal, fs)

# Revised Ideal Hilbert Frequency Response
delta = np.zeros_like(x_real)
delta[len(delta)//2] = 1.0
delta_analytic = hilbert(delta)
_, delta_analytic_mag, delta_analytic_dB = compute_fft(delta_analytic, fs)
scale_factor = np.max(analytic_mag)
H_ideal = delta_analytic_mag * (scale_factor / np.max(delta_analytic_mag))
H_ideal_dB = 20 * np.log10(H_ideal + 1e-12)

plt.figure(figsize=(10, 6))
plt.plot(freqs, X_real_dB, label="Original Real Signal Spectrum", lw=1.0, alpha=0.7)
plt.plot(freqs, analytic_dB, label="Filtered Signal", lw=1.5)
plt.plot(freqs, H_ideal_dB, 'k--', label="Hilbert Filter (FFT) Response", lw=1.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Spectrum of Recorded Signal and Filtered Signal by FFT Hilbert Transform")
plt.legend()
plt.grid(True)
plt.xlim(-fs/2, fs/2)
plt.tight_layout()
if show_plots:
    plt.show()
if save_plots:
    plt.savefig("images/hilbert_transform_analysis.png", dpi=300)
    plt.savefig("images/hilbert_transform_analysis.svg", format='svg')

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
plt.plot(freqs, X_real_dB, label="Original Real Signal Spectrum", lw=1.0, alpha=0.7)
plt.plot(freqs, x_fir_dB, label="FIR Filter Output (Analytic Signal)", lw=1.5)
plt.plot(freq_h, H_fir_dB, 'k--', label="FIR Hilbert Filter Response (scaled)", lw=1.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("FIR Hilbert Transformer Frequency Domain Analysis")
plt.legend()
plt.grid(True)
plt.xlim(-fs/2, fs/2)
plt.tight_layout()
if show_plots:
    plt.show()
if save_plots:
    plt.savefig("images/fir_hilbert_transformer_analysis.png", dpi=300)
    plt.savefig("images/fir_hilbert_transformer_analysis.svg", format='svg')

# ==============================
# Plot 4: Instantaneous Frequency Comparison and Error (Two Subplots)
# ==============================
# Compute instantaneous frequency from analytic_signal and FIR-filtered signal
phase_analytic = np.unwrap(np.angle(analytic_signal))
inst_freq_analytic = np.gradient(phase_analytic) * fs / (2 * np.pi)

phase_fir = np.unwrap(np.angle(x_fir))
inst_freq_fir = np.gradient(phase_fir) * fs / (2 * np.pi)

# Remove boundary spikes by trimming a fraction of the data
trim_ratio = 0.05  # Remove 5% of samples at beginning and end
trim_samples = int(trim_ratio * len(t))
t_trim = t[trim_samples:-trim_samples]
true_if_trim = true_instantaneous_frequency[trim_samples:-trim_samples]
inst_freq_fir_trim = inst_freq_fir[trim_samples:-trim_samples]
inst_freq_analytic_trim = inst_freq_analytic[trim_samples:-trim_samples]

# Compute error (estimated minus true)
error_fir = inst_freq_fir_trim - true_if_trim
error_analytic = inst_freq_analytic_trim - true_if_trim

# Create a figure with two subplots:
#   Top: Instantaneous frequencies vs. time (trimmed)
#   Bottom: Y-Y error plot (error vs. true instantaneous frequency)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Top subplot: Time-domain plot of instantaneous frequencies
ax1.plot(t_trim, inst_freq_fir_trim, linestyle=':', lw=1.5, label="FIR Inst. Frequency")
ax1.plot(t_trim, inst_freq_analytic_trim, linestyle='--', lw=1.5, label="Hilbert Inst. Frequency")
ax1.plot(t_trim, true_if_trim, lw=1.5, label="True Instantaneous Frequency")
ax1.set_ylabel("Frequency (Hz)")
ax1.set_title("Instantaneous Frequency Comparison (Time Domain)")
ax1.legend()
ax1.grid(True)

# Bottom subplot: Y-Y error plot (error vs. true instantaneous frequency)
ax2.plot(t_trim, error_fir, linestyle=':', lw=1.5, label="FIR Error")
ax2.plot(t_trim, error_analytic, linestyle='--', lw=1.5, label="Hilbert Error")
ax2.axhline(0, color='k', lw=1, linestyle='--')  # Zero error reference
ax2.set_xlabel("True Instantaneous Frequency (Hz)")
ax2.set_ylabel("Time (seconds)")
ax2.set_title("Instantaneous Frequency Error Comparison")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
if show_plots:
    plt.show()
if save_plots:
    plt.savefig("images/instantaneous_frequency_comparison_subplots.png", dpi=300)
    plt.savefig("images/instantaneous_frequency_comparison_subplots.svg", format='svg')
