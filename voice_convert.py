#!/usr/bin/env python
"""Sing a generated song in another voice (runs in .venv-voice).

    .venv-voice\\Scripts\\python.exe voice_convert.py --song outputs\\x\\audio.flac --reference voices\\me.wav --name me --output outputs\\x

1. Demucs (htdemucs) splits the song into the vocal and the accompaniment; the stems are cached
   in <output>/stems so switching voices later skips this step.
2. Seed-VC (zero-shot, f0-conditioned singing model) re-renders the vocal in the reference voice.
3. The converted vocal is level-matched to the original one and mixed back over the accompaniment.

Writes audio-voice-<name>.flac and vocal-<name>.flac into --output and prints one RESULT line.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
SEEDVC = HERE / "tools" / "seed-vc"
FFMPEG = HERE / "tools" / "ffmpeg" / "bin"


def log(message):
    print(f"[voice] {message}", flush=True)


def load_audio(path):
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), rate            # [channels, samples]


def resample(wave, source, target):
    import torchaudio
    return torchaudio.functional.resample(wave, source, target) if source != target else wave


def separate(song, rate, device, stems_dir):
    """Return (vocals, accompaniment, sample_rate) as [2, N] tensors, cached on disk."""
    vocals_file, acc_file = stems_dir / "vocals.wav", stems_dir / "accompaniment.wav"
    if vocals_file.is_file() and acc_file.is_file():
        vocals, rate_v = load_audio(vocals_file)
        acc, _ = load_audio(acc_file)
        log(f"reusing cached stems ({rate_v} Hz)")
        return vocals, acc, rate_v
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    t = time.perf_counter()
    model = get_model("htdemucs").to(device).eval()
    wave = song if song.shape[0] == 2 else song[:1].repeat(2, 1)
    wave = resample(wave, rate, model.samplerate)
    ref = wave.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.inference_mode():
        sources = apply_model(model, ((wave - mean) / std)[None].to(device), device=device,
                              shifts=1, split=True, overlap=0.25, progress=False)[0]
    sources = (sources * std + mean).cpu()
    names = list(model.sources)
    vocals = sources[names.index("vocals")]
    acc = sum(sources[i] for i, name in enumerate(names) if name != "vocals")
    stems_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(vocals_file), vocals.T.numpy(), model.samplerate, subtype="FLOAT")
    sf.write(str(acc_file), acc.T.numpy(), model.samplerate, subtype="FLOAT")
    log(f"separated in {time.perf_counter() - t:.0f}s")
    sample_rate = model.samplerate
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return vocals, acc, sample_rate


def convert(vocals_path, reference, out_dir, steps, semitones, cfg_rate):
    """Run Seed-VC's singing model; return the converted mono file."""
    command = [sys.executable, "-X", "utf8", str(SEEDVC / "inference.py"), "--source", str(vocals_path), "--target", str(reference),
               "--output", str(out_dir), "--diffusion-steps", str(steps), "--length-adjust", "1.0",
               "--inference-cfg-rate", str(cfg_rate), "--f0-condition", "True", "--auto-f0-adjust", "False",
               "--semi-tone-shift", str(semitones), "--fp16", "True"]
    t = time.perf_counter()
    result = subprocess.run(command, cwd=str(SEEDVC), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        tail = (result.stderr or result.stdout).strip().splitlines()[-15:]
        raise RuntimeError("Seed-VC failed:\n" + "\n".join(tail))
    produced = sorted(out_dir.glob("vc_*.wav"), key=lambda p: p.stat().st_mtime)
    if not produced:
        raise RuntimeError("Seed-VC produced no output file")
    log(f"converted in {time.perf_counter() - t:.0f}s")
    return produced[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--song", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="a clean recording of the target voice (20-60 s)")
    parser.add_argument("--name", required=True, help="voice name used in the output file names")
    parser.add_argument("--output", type=Path, required=True, help="the song's folder")
    parser.add_argument("--semitones", type=int, default=0, help="shift the vocal pitch (e.g. -3 for a lower voice)")
    parser.add_argument("--steps", type=int, default=30, help="diffusion steps (quality vs time)")
    parser.add_argument("--cfg-rate", type=float, default=0.7)
    parser.add_argument("--vocal-gain-db", type=float, default=0.0, help="extra level for the converted vocal in the mix")
    args = parser.parse_args()
    if FFMPEG.is_dir():
        os.environ["PATH"] = str(FFMPEG) + os.pathsep + os.environ.get("PATH", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    song, rate = load_audio(args.song)
    stems = args.output / "stems"
    vocals, acc, stem_rate = separate(song, rate, device, stems)
    vocals_mono = stems / "vocals-mono.wav"
    sf.write(str(vocals_mono), vocals.mean(0).numpy(), stem_rate, subtype="PCM_16")
    work = args.output / "stems" / f"vc-{args.name}"
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("vc_*.wav"):
        old.unlink()
    converted_file = convert(vocals_mono, args.reference.resolve(), work, args.steps, args.semitones, args.cfg_rate)
    converted, conv_rate = load_audio(converted_file)
    converted = resample(converted, conv_rate, stem_rate)
    n = acc.shape[1]
    converted = converted[:, :n] if converted.shape[1] >= n else torch.nn.functional.pad(converted, (0, n - converted.shape[1]))
    original_rms = vocals.pow(2).mean().sqrt()
    converted_rms = converted.pow(2).mean().sqrt().clamp_min(1e-6)
    gain = float(original_rms / converted_rms) * (10 ** (args.vocal_gain_db / 20))
    vocal_stereo = converted.repeat(2, 1) * gain
    mix = acc + vocal_stereo
    peak = mix.abs().max()
    if peak > 0.98:
        mix = mix / peak * 0.98
    mix = resample(mix, stem_rate, rate)
    vocal_out = resample(vocal_stereo, stem_rate, rate)
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in args.name).strip("-") or "voice"
    mix_path = args.output / f"audio-voice-{safe}.flac"
    sf.write(str(mix_path), mix.T.numpy(), rate, subtype="PCM_24")
    sf.write(str(args.output / f"vocal-{safe}.flac"), vocal_out.T.numpy(), rate, subtype="PCM_24")
    summary = {"audio": mix_path.name, "vocal": f"vocal-{safe}.flac", "voice": safe, "seconds": round(n / stem_rate, 1),
               "semitones": args.semitones, "steps": args.steps, "device": device, "total_seconds": round(time.perf_counter() - start, 1)}
    print("RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
