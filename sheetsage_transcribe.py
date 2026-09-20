#!/usr/bin/env python
"""Transcribe a recording to a YuE2 melody score with SheetSage2 (runs in .venv-sheetsage2).

    .venv-sheetsage2\\Scripts\\python.exe sheetsage_transcribe.py song.mp3 --output transcriptions\\x [--full]

Writes score.abc (+ MIDI, events) into --output and prints one JSON line with the
ABC, warnings and timing. Audio is decoded with soundfile (WAV/FLAC/MP3/OGG), so
FFmpeg is only needed for other formats; tools/ffmpeg/bin is added to PATH when present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models" / "SheetSage2"
FFMPEG = HERE / "tools" / "ffmpeg" / "bin"


def decode_audio(path: Path):
    """Return (waveform[channels, samples] float32, rate) via soundfile, else None."""
    try:
        import soundfile as sf
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return data.T.copy(), int(rate)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="keep chord symbols (default: melody only, chord-free)")
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    if FFMPEG.is_dir():
        os.environ["PATH"] = str(FFMPEG) + os.pathsep + os.environ.get("PATH", "")
    start = time.perf_counter()
    import torch
    from transformers import AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True).eval().to(device)
    loaded = time.perf_counter() - start
    decoded = decode_audio(args.audio)
    kwargs = {"melody_only": not args.full}
    if args.max_seconds:
        kwargs["max_seconds"] = args.max_seconds
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        if decoded is not None:
            waveform, rate = decoded
            result = model.transcribe(waveform, output_dir=str(args.output), sampling_rate=rate, **kwargs)
            seconds = waveform.shape[1] / rate
        else:                                    # unusual container: let SheetSage2 use ffmpeg
            result = model.transcribe(str(args.audio), output_dir=str(args.output), **kwargs)
            seconds = None
    abc = result.get("abc")
    if not abc or result.get("abc_error"):
        raise SystemExit("Transcription did not produce a usable score: " + str(result.get("abc_error") or "empty ABC"))
    (args.output / "score.abc").write_bytes(abc.encode("utf-8"))
    summary = {"abc": abc, "warnings": list(result.get("warnings", [])), "seconds": seconds,
               "load_seconds": round(loaded, 1), "total_seconds": round(time.perf_counter() - start, 1),
               "melody_only": not args.full, "device": device}
    (args.output / "transcription.json").write_text(json.dumps({k: v for k, v in summary.items() if k != "abc"}, indent=2), encoding="utf-8")
    sys.stdout.write("RESULT " + json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
