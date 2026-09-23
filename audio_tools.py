#!/usr/bin/env python
"""Shared audio helpers for the voice environment (.venv-voice): Demucs stems, loudness,
peak limiting, fades and MP3 encoding. Imported by voice_convert.py, stems.py,
export_audio.py and lyrics_sync.py; not used by the YuE2 environment.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
HF_CACHE = HERE / "tools" / "seed-vc" / "checkpoints" / "hf_cache"   # Seed-VC's Whisper already lives here
STEM_NAMES = ("vocals", "drums", "bass", "other")


def log(message):
    print(f"[audio] {message}", flush=True)


def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path):
    """-> (tensor [channels, samples] float32, rate)"""
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), rate


def resample(wave, source, target):
    import torchaudio
    return torchaudio.functional.resample(wave, source, target) if source != target else wave


def to_stereo(wave):
    return wave if wave.shape[0] == 2 else wave[:1].repeat(2, 1)


def write_flac(path, wave, rate):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = wave.T.numpy() if isinstance(wave, torch.Tensor) else wave
    sf.write(str(path), data, rate, subtype="PCM_24")


def separate_stems(song_path, stems_dir, device=None):
    """Demucs htdemucs: vocals / drums / bass / other, plus accompaniment (everything but the
    vocals), cached as 44.1 kHz FLAC in stems_dir. Returns {name: path}."""
    stems_dir = Path(stems_dir)
    expected = {name: stems_dir / f"{name}.flac" for name in (*STEM_NAMES, "accompaniment")}
    if all(p.is_file() for p in expected.values()):
        log("reusing cached stems")
        return expected
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    device = device or device_name()
    song, rate = load_audio(song_path)
    t = time.perf_counter()
    model = get_model("htdemucs").to(device).eval()
    wave = resample(to_stereo(song), rate, model.samplerate)
    ref = wave.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.inference_mode():
        sources = apply_model(model, ((wave - mean) / std)[None].to(device), device=device,
                              shifts=1, split=True, overlap=0.25, progress=False)[0]
    sources = (sources * std + mean).cpu()
    names = list(model.sources)
    stems = {name: sources[names.index(name)] for name in STEM_NAMES}
    stems["accompaniment"] = sum(stems[n] for n in STEM_NAMES if n != "vocals")
    for name, out in stems.items():
        write_flac(expected[name], out.clamp(-1, 1), model.samplerate)
    log(f"separated in {time.perf_counter() - t:.0f}s")
    del model, sources
    if device == "cuda":
        torch.cuda.empty_cache()
    return expected


# ── mastering helpers (numpy [samples, channels]) ────────────────────────────
def integrated_lufs(data, rate):
    import pyloudnorm as pyln
    value = pyln.Meter(rate).integrated_loudness(data)
    return float(value) if np.isfinite(value) else None


def limit_peaks(data, ceiling=0.985, window_ms=10.0, rate=44100):
    """Simple lookahead peak limiter: the gain each sample needs, min-filtered over a short
    window (lookahead + release) then smoothed with an equal window, so no sample exceeds the
    ceiling and the gain never jumps."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    peak = np.abs(data).max(axis=1)
    needed = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    if needed.min() >= 1.0:
        return data
    size = max(3, int(rate * window_ms / 1000) | 1)
    gain = minimum_filter1d(needed, size=size, mode="nearest")
    gain = uniform_filter1d(gain, size=size, mode="nearest")
    return np.clip(data * gain[:, None], -ceiling, ceiling)


def fade(data, rate, fade_in=0.0, fade_out=0.0):
    n = data.shape[0]
    if fade_in > 0:
        k = min(n, int(fade_in * rate))
        data[:k] *= (np.sin(np.linspace(0, np.pi / 2, k)) ** 2)[:, None]
    if fade_out > 0:
        k = min(n, int(fade_out * rate))
        data[n - k:] *= (np.cos(np.linspace(0, np.pi / 2, k)) ** 2)[:, None]
    return data


def trim_silence(data, rate, threshold_db=-55.0, pad=0.25):
    level = np.abs(data).max(axis=1)
    above = np.flatnonzero(level > 10 ** (threshold_db / 20))
    if above.size == 0:
        return data
    start = max(0, int(above[0] - pad * rate))
    end = min(data.shape[0], int(above[-1] + pad * rate))
    return data[start:end]


def encode_mp3(data, rate, bitrate=320):
    import lameenc
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(rate)
    encoder.set_channels(data.shape[1])
    encoder.set_quality(2)
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()
    return bytes(encoder.encode(pcm)) + bytes(encoder.flush())
