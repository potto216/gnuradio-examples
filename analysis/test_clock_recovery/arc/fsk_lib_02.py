
# ───────────────────────────────────────
# DSP building blocks
# ───────────────────────────────────────
def design_bandpass(fs: float, center: float, bw: float, order: int = 4):
    """Return IIR band-pass coefficients (Butterworth)."""
    nyq = fs / 2.0
    low  = (center - bw / 2) / nyq
    high = (center + bw / 2) / nyq
    return sp_signal.butter(order, [low, high], btype="band")


def envelope(signal_in: np.ndarray, fs: float, lp_cut: float):
    """Magnitude envelope via Hilbert transform + low-pass filter."""
    analytic = sp_signal.hilbert(signal_in)           # complex analytic signal
    mag = np.abs(analytic)
    b_lp, a_lp = sp_signal.butter(2, lp_cut / (fs / 2), btype="low")
    return sp_signal.filtfilt(b_lp, a_lp, mag)        # zero-phase LPF


# ───────────────────────────────────────
# FSK demodulator
# ───────────────────────────────────────
def fsk_demodulate_extra(signal_in: np.ndarray,
                   fs: float,
                   baud: float,
                   f0: float,
                   f1: float,
                   bw: float = 400,
                   lp_cut: float | None = None):
    """
    Demodulate Manchester-coded 2-FSK.
    Returns (decoded_bits, env_f0, env_f1, diff).
    """
    if lp_cut is None:
        lp_cut = baud * 2             # good default for envelope LPF

    # Tone-selective band-pass filtering (zero phase)
    b0, a0 = design_bandpass(fs, f0, bw)
    b1, a1 = design_bandpass(fs, f1, bw)
    tone0  = sp_signal.filtfilt(b0, a0, signal_in)
    tone1  = sp_signal.filtfilt(b1, a1, signal_in)

    # Magnitude envelopes
    env0 = envelope(tone0, fs, lp_cut)
    env1 = envelope(tone1, fs, lp_cut)
    diff = env1 - env0                # decision variable

    # Manchester decoding
    samples_per_chip = int(fs / (baud * 2))
    bits = decode_manchester_avg(diff, samples_per_chip)

    return bits, env0, env1, diff


# ───────────────────────────────────────
# BER helper
# ───────────────────────────────────────
def ber_with_shift(tx: np.ndarray,
                    rx: np.ndarray,
                    max_shift: int = 2) -> float:
    """Best BER after trying ±max_shift bit alignment."""
    best = 1.0
    for s in range(-max_shift, max_shift + 1):
        if s < 0:
            ref, est = tx[:s], rx[-s:]
        else:
            ref, est = tx[s:], rx[: tx.size - s]
        if est.size:
            # limit comparison to the shorter array to avoid shape mismatches
            n = min(ref.size, est.size)
            best = min(best, np.mean(ref[:n] != est[:n]))
    return best


# ───────────────────────────────────────
# Quick demo / self-test
# ───────────────────────────────────────
if __name__ == "__main__":
    # Parameters
    fs   = 44_100           # samples/s
    baud = 100              # data bits per second
    f0, f1 = 1_000, 3_000   # FSK tones
    np.random.seed(0)
    tx_bits = np.random.randint(0, 2, 50)
    byte_vals = np.array([0x01, 0x02, 0x03, 0x04, 0x05, 0x06], dtype=np.uint8)
    # …and unpack to bits (LSB first on a little‐endian host)
    tx_bits = np.unpackbits(byte_vals, bitorder='little').astype(int)

    # --- Modulate -----------------------------------------------------------
    sig, spp_chip, chips = fsk_modulate(tx_bits, fs, baud, f0, f1, manchester=True)
    
    wav_filename = "/home/user/src/gnuradio/gnuradio-examples/analysis/test_clock_recovery/pkt_fsk_xmt_make_wav_v1.wav"
    from scipy.io import wavfile
    # read wavfile
    fs_file, sig = wavfile.read(wav_filename)
    if sig.ndim > 1:  # stereo to mono
        sig = np.mean(sig, axis=1)
    # trim the first 1000 samples
    sig = sig[2700:]        
    #verify sample rate
    if fs_file != fs:
        raise ValueError(f"Sample rate mismatch: expected {fs}, got {fs_file}") 
        

    # --- Demodulate ---------------------------------------------------------
    rx_bits, env0, env1, diff = fsk_demodulate(sig, fs, baud, f0, f1)

    # BER
    ber = ber_with_shift(tx_bits, rx_bits)
    print(f"Recovered {rx_bits.size}/{tx_bits.size} bits  |  BER ≈ {ber:.3%}")

    # -----------------------------------------------------------------------
    # Plots (first ~10 data bits for clarity)
    show_samples = spp_chip * 20                   # 10 bits → 20 chips
    t = np.arange(show_samples) / fs

    plt.figure()
    plt.title("Band-limited envelopes (first 10 bits)")
    plt.plot(t, env0[:show_samples], label="Envelope f0")
    plt.plot(t, env1[:show_samples], label="Envelope f1")
    plt.xlabel("Time [s]"); plt.ylabel("Magnitude"); plt.legend()

    plt.figure()
    plt.title("Envelope difference  f1 − f0")
    plt.plot(t, diff[:show_samples])
    plt.xlabel("Time [s]"); plt.ylabel("Env₁ − Env₀")

    plt.figure()
    plt.title("Manchester decoded bits (TX vs RX)")
    tx_step = np.repeat(tx_bits, 2)
    rx_step = np.repeat(rx_bits, 2)
    plt.step(np.arange(tx_step.size) / baud / 2, tx_step, where="post", label="TX")
    plt.step(np.arange(rx_step.size) / baud / 2, rx_step, where="post", label="RX", linestyle=':')
    plt.yticks([0, 1]); plt.ylim(-0.2, 1.2)
    plt.xlabel("Time [s]"); plt.legend()

    plt.tight_layout()
    plt.show()
