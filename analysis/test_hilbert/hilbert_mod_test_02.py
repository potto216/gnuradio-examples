#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from numpy.fft import fft, fftshift, fftfreq

# =======================
# User-Specified Settings
# =======================
fs = 44100                # Sampling rate (Hz)
duration = 1.0            # Signal duration (seconds)
N = int(fs * duration)    # Number of samples
t = np.arange(N) / fs     # Time vector

# Nominal frequency of the complex exponential (Hz)
f0 = 3000

# Amplitude modulation function (user specifiable)
# Example: a slow sinusoidal fluctuation about 1.0
amp_mod = 1.0 + 0.5 * np.sin(2 * np.pi * 2 * t)

# Phase modulation function (user specifiable)
# Example: a 5 Hz sinusoidal phase fluctuation with amplitude in radians
phase_mod = 0.2 * np.sin(70 * np.pi * 5 * t)

# Gaussian noise standard deviation (user specifiable)
noise_std = 0.001

# FIR Hilbert transformer parameter (number of taps, must be odd)
M = 101
if M % 2 == 0:
    raise ValueError("FIR filter length M must be odd.")

# ==============================
# Generate the modified complex exponential
# ==============================
# The complex signal with amplitude and phase modulation:
#   x_complex = A(t) * exp(j*(2*pi*f0*t + phase_mod(t)))
x_complex = amp_mod * np.exp(1j * (2 * np.pi * f0 * t + phase_mod))
# Take the real part and add Gaussian noise:
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
    # Avoid log(0) by adding a small epsilon.
    eps = 1e-12
    mag = np.abs(X_shifted)
    mag_dB = 20 * np.log10(mag + eps)
    return freqs, mag, mag_dB



# ==============================
# Time-Domain Plot of Real and Complex Signal Before Filtering
# ==============================

# Choose a window that contains only a few cycles of the signal
num_cycles_to_show = 15  # Number of cycles to display
T0 = 1 / f0  # Period of one cycle
samples_to_show = int(num_cycles_to_show * T0 * fs)  # Convert to number of samples

# Extract a small section of the signal
t_zoom = t[:samples_to_show]
x_real_zoom = x_real[:samples_to_show]
x_complex_zoom = x_complex[:samples_to_show]  # This is complex, plot real & imaginary parts

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
plt.show()


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
plt.show()

# ==============================
# Plot 2: Hilbert Transform Analysis (Using scipy.signal.hilbert)
# ==============================
# Generate the analytic signal via Hilbert transform:
analytic_signal = hilbert(x_real)
freqs, analytic_mag, analytic_dB = compute_fft(analytic_signal, fs)

# --- Revised Ideal Hilbert Frequency Response ---
# Here we generate a delta impulse (centered) and compute its analytic signal.
# This “delta response” shows the filter (frequency) behavior of the Hilbert transform used.
delta = np.zeros_like(x_real)
delta[len(delta)//2] = 1.0  # Centered delta impulse
delta_analytic = hilbert(delta)
_, delta_analytic_mag, delta_analytic_dB = compute_fft(delta_analytic, fs)
# Scale the ideal response so its maximum equals that of the analytic signal.
scale_factor = np.max(analytic_mag)
H_ideal = delta_analytic_mag * (scale_factor / np.max(delta_analytic_mag))
H_ideal_dB = 20 * np.log10(H_ideal + 1e-12)

plt.figure(figsize=(10, 6))
plt.plot(freqs, X_real_dB, label="Original Real Signal Spectrum", lw=1.0, alpha=0.7)
plt.plot(freqs, analytic_dB, label="Analytic Signal (Hilbert Transform Output)", lw=1.5)
plt.plot(freqs, H_ideal_dB, 'k--', label="Ideal Hilbert Response (delta input, scaled)", lw=1.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Hilbert Transform Frequency Domain Analysis")
plt.legend()
plt.grid(True)
plt.xlim(-fs/2, fs/2)
plt.tight_layout()
plt.show()

# ==============================
# Plot 3: FIR Hilbert Transformer Analysis (Complex FIR for Analytic Signal)
# ==============================
# The standard FIR Hilbert transformer (truncated ideal impulse response) yields a real filter.
# For analytic signal generation, we design a complex filter with impulse response:
#    h[n] = delta[n - mid] + j * h_H[n]
# where h_H[n] is the ideal Hilbert impulse response (for odd indices only).
m = np.arange(-((M-1)//2), ((M-1)//2) + 1)
h_real = np.zeros(M)
h_imag = np.zeros(M)
for i, mi in enumerate(m):
    if mi == 0:
        h_real[i] = 1.0  # delta impulse for the real part
    elif mi % 2 != 0:
        h_imag[i] = 2 / (np.pi * mi)
    else:
        h_imag[i] = 0.0

# Apply a window only to the imaginary (Hilbert) part to smooth its response.
window = np.hamming(M)
h_imag_windowed = h_imag * window
# Construct the complex filter.
h_fir = h_real + 1j * h_imag_windowed

# Convolve the real signal with the complex FIR filter.
# The output should approximate the analytic signal.
x_fir = np.convolve(x_real, h_fir, mode='same')

# Compute FFT of the FIR–filtered output.
freqs, x_fir_mag, x_fir_dB = compute_fft(x_fir, fs)

# Compute the FIR filter's frequency response (using zero-padding for smoothness).
H_fir = fftshift(fft(h_fir, n=1024))
freq_h = np.linspace(-fs/2, fs/2, 1024)

# Scale the FIR frequency response to the maximum magnitude of the FIR output.
scale_fir = np.max(np.abs(x_fir_dB))
H_fir_dB = 20 * np.log10(np.abs(H_fir) + 1e-12)
H_fir_scaled = H_fir_dB + (scale_fir - np.max(H_fir_dB))
H_fir_dB = H_fir_scaled

# scale_fir = np.max(np.abs(x_fir))
# H_fir_scaled = np.abs(H_fir) * (scale_fir / np.max(np.abs(H_fir)))
# H_fir_dB = 20 * np.log10(H_fir_scaled + 1e-12)


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
plt.show()
