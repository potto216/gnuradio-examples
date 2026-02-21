"""Shared utilities for FSK analysis CLIs."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None


def now_iso_utc() -> str:
    ts = datetime.datetime.now(datetime.UTC)
    return ts.isoformat().replace("+00:00", "Z")


def write_wav(path: Path, data: np.ndarray, fs: int) -> None:
    if sf is None:
        from scipy.io import wavfile

        wavfile.write(str(path), fs, data.astype(np.float32))
    else:
        sf.write(str(path), data.astype(np.float32), fs)


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    if sf is None:
        from scipy.io import wavfile

        fs, x = wavfile.read(str(path))
        if x.dtype != np.float32:
            denom = np.max(np.abs(x))
            if denom == 0:
                denom = 1.0
            x = x.astype(np.float32) / denom
    else:
        x, fs = sf.read(str(path), dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
    return fs, x.astype(np.float32)


def save_json(path: Path, obj) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
