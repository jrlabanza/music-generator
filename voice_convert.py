#!/usr/bin/env python
"""Sing a generated song in another voice (runs in .venv-voice).

    .venv-voice\\Scripts\\python.exe voice_convert.py --song outputs\\x\\audio.flac --reference voices\\me.wav --name me --output outputs\\x

1. Demucs (htdemucs) splits the song into stems; they are cached in <output>/stems, so a second
   voice, a harmony or a karaoke sync skips this step.
2. Seed-VC (zero-shot, f0-conditioned singing model) re-renders the vocal in the reference voice.
   --semitones shifts the pitch, --auto-f0 moves it into the reference singer's range first.
   --harmonies "4,7" renders extra copies shifted by those intervals and blends them under the lead.
   --duet-reference/--duet-name sing the sections named by --duet-sections (from karaoke.json's
   line timings) in a second voice.
3. The converted vocal is level-matched to the original one and mixed back over the accompaniment.

Writes audio-voice-<label>.flac and vocal-<label>.flac into --output and prints one RESULT line.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_tools import HERE, device_name, load_audio, log, resample, separate_stems, write_flac

SEEDVC = HERE / "tools" / "seed-vc"
FFMPEG = HERE / "tools" / "ffmpeg" / "bin"


def convert(vocals_path, reference, out_dir, steps, semitones, cfg_rate, auto_f0, force=False):
    """Run Seed-VC's singing model (or reuse an earlier render in out_dir); return the mono file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("vc_*.wav"), key=lambda p: p.stat().st_mtime)
    if existing and not force:
        log(f"reusing {existing[-1].name}")
        return existing[-1]
    for old in existing:
        old.unlink()
    command = [sys.executable, "-X", "utf8", str(SEEDVC / "inference.py"), "--source", str(vocals_path), "--target", str(reference),
               "--output", str(out_dir), "--diffusion-steps", str(steps), "--length-adjust", "1.0",
               "--inference-cfg-rate", str(cfg_rate), "--f0-condition", "True", "--auto-f0-adjust", "True" if auto_f0 else "False",
               "--semi-tone-shift", str(semitones), "--fp16", "True"]
    t = time.perf_counter()
    result = subprocess.run(command, cwd=str(SEEDVC), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        tail = (result.stderr or result.stdout).strip().splitlines()[-15:]
        raise RuntimeError("Seed-VC failed:\n" + "\n".join(tail))
    produced = sorted(out_dir.glob("vc_*.wav"), key=lambda p: p.stat().st_mtime)
    if not produced:
        raise RuntimeError("Seed-VC produced no output file")
    log(f"converted ({semitones:+d} st{', auto f0' if auto_f0 else ''}) in {time.perf_counter() - t:.0f}s")
    return produced[-1]


def load_mono_at(path, rate, length):
    wave, source_rate = load_audio(path)
    wave = resample(wave.mean(0, keepdim=True), source_rate, rate)[0]
    return wave[:length] if wave.shape[0] >= length else torch.nn.functional.pad(wave, (0, length - wave.shape[0]))


def section_key(name):
    return re.sub(r"[^a-z]", "", name.lower())


def duet_mask(karaoke, sections, rate, length, ramp=0.08, margin=0.15):
    """1.0 where the duet voice sings (the named sections' lines), 0.0 elsewhere, soft edges."""
    from scipy.ndimage import uniform_filter1d
    lines = karaoke.get("lines", [])
    wanted = {section_key(s) for s in sections}
    alternate = "alternate" in wanted
    mask = np.zeros(length, dtype=np.float32)
    order, seen = [], {}
    for line in lines:                                        # section index for "alternate"
        key = line.get("section", "")
        if key not in seen:
            seen[key] = len(order)
            order.append(key)
    for line in lines:
        key = section_key(line.get("section", ""))
        pick = (seen[line.get("section", "")] % 2 == 1) if alternate else any(w and (w == key or key.startswith(w)) for w in wanted)
        if pick and line.get("start") is not None:
            a = max(0, int((line["start"] - margin) * rate))
            b = min(length, int((line["end"] + margin) * rate))
            mask[a:b] = 1.0
    size = max(3, int(ramp * rate) | 1)
    return torch.from_numpy(uniform_filter1d(mask, size=size, mode="nearest"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--song", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="a clean recording of the target voice (20-60 s)")
    parser.add_argument("--name", required=True, help="voice name used in the output file names")
    parser.add_argument("--output", type=Path, required=True, help="the song's folder")
    parser.add_argument("--semitones", type=int, default=0, help="shift the vocal pitch (e.g. -3 for a lower voice)")
    parser.add_argument("--auto-f0", action="store_true", help="move the melody into the reference voice's range first")
    parser.add_argument("--steps", type=int, default=30, help="diffusion steps (quality vs time)")
    parser.add_argument("--cfg-rate", type=float, default=0.7)
    parser.add_argument("--vocal-gain-db", type=float, default=0.0, help="extra level for the converted vocal in the mix")
    parser.add_argument("--harmonies", default="", help="comma-separated semitone intervals for harmony voices, e.g. 4,7 or -5,4")
    parser.add_argument("--harmony-gain-db", type=float, default=-6.0, help="level of each harmony relative to the lead")
    parser.add_argument("--duet-reference", type=Path, help="second voice's reference clip")
    parser.add_argument("--duet-name", default="")
    parser.add_argument("--duet-sections", default="chorus", help="sections the duet voice sings: names, or 'alternate'")
    parser.add_argument("--karaoke", type=Path, help="karaoke.json with line timings (needed for a duet)")
    parser.add_argument("--force", action="store_true", help="re-render even when a conversion is cached")
    args = parser.parse_args()
    if FFMPEG.is_dir():
        os.environ["PATH"] = str(FFMPEG) + os.pathsep + os.environ.get("PATH", "")
    device = device_name()
    start = time.perf_counter()
    song_rate = sf.info(str(args.song)).samplerate
    stems = separate_stems(args.song, args.output / "stems", device)
    vocals, stem_rate = load_audio(stems["vocals"])
    acc, _ = load_audio(stems["accompaniment"])
    n = acc.shape[1]
    vocals_mono = args.output / "stems" / "vocals-mono.wav"
    if not vocals_mono.is_file():
        sf.write(str(vocals_mono), vocals.mean(0).numpy(), stem_rate, subtype="PCM_16")
    safe = lambda text: "".join(c if c.isalnum() or c in "-_" else "-" for c in text).strip("-") or "voice"
    name = safe(args.name)
    work = args.output / "stems"
    reference = args.reference.resolve()
    tag = f"s{args.semitones:+d}" + ("-auto" if args.auto_f0 else "")
    lead_file = convert(vocals_mono, reference, work / f"vc-{name}-{tag}", args.steps, args.semitones, args.cfg_rate, args.auto_f0, args.force)
    lead = load_mono_at(lead_file, stem_rate, n)
    original_rms = vocals.pow(2).mean().sqrt()

    def matched(wave):                                     # bring a render to the original vocal's level
        return wave * float(original_rms / wave.pow(2).mean().sqrt().clamp_min(1e-6))
    lead = matched(lead)
    label = name
    harmonies = [int(h) for h in re.split(r"[,\s]+", args.harmonies.strip()) if h.strip()]
    harmony_mix = torch.zeros(n)
    for interval in harmonies:
        shift = args.semitones + interval
        folder = work / (f"vc-{name}-s{shift:+d}" + ("-auto" if args.auto_f0 else "") + "-h")
        file = convert(vocals_mono, reference, folder, args.steps, shift, args.cfg_rate, args.auto_f0, args.force)
        harmony_mix += matched(load_mono_at(file, stem_rate, n)) * (10 ** (args.harmony_gain_db / 20))
    if harmonies:
        label += "-harmony"
    duet = None
    if args.duet_reference:
        if not args.karaoke or not args.karaoke.is_file():
            raise SystemExit("A duet needs karaoke.json line timings (run the karaoke sync first)")
        duet = safe(args.duet_name or args.duet_reference.stem)
        duet_file = convert(vocals_mono, args.duet_reference.resolve(), work / f"vc-{duet}-{tag}", args.steps, args.semitones,
                            args.cfg_rate, args.auto_f0, args.force)
        second = matched(load_mono_at(duet_file, stem_rate, n))
        mask = duet_mask(json.loads(args.karaoke.read_text(encoding="utf-8")), re.split(r"[,\s]+", args.duet_sections.strip()), stem_rate, n)
        lead = lead * (1 - mask) + second * mask
        label += f"-duet-{duet}"
    vocal = (lead + harmony_mix) * (10 ** (args.vocal_gain_db / 20))
    vocal_stereo = vocal.repeat(2, 1)
    mix = acc + vocal_stereo
    peak = mix.abs().max()
    if peak > 0.98:
        mix = mix / peak * 0.98
    mix = resample(mix, stem_rate, song_rate)
    vocal_out = resample(vocal_stereo, stem_rate, song_rate)
    mix_path = args.output / f"audio-voice-{label}.flac"
    write_flac(mix_path, mix, song_rate)
    write_flac(args.output / f"vocal-{label}.flac", vocal_out, song_rate)
    summary = {"audio": mix_path.name, "vocal": f"vocal-{label}.flac", "voice": name, "label": label, "seconds": round(n / stem_rate, 1),
               "semitones": args.semitones, "auto_f0": args.auto_f0, "steps": args.steps, "harmonies": harmonies, "duet": duet,
               "device": device, "total_seconds": round(time.perf_counter() - start, 1)}
    print("RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
