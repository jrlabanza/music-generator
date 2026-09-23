#!/usr/bin/env python
"""Master and export a song (runs in .venv-voice).

    .venv-voice\\Scripts\\python.exe export_audio.py --song outputs\\x\\audio.flac --output outputs\\x --format mp3 --lufs -14 --fade-out 2

Trims leading/trailing silence (optional), applies fades, normalises the integrated loudness to a
streaming target (BS.1770 via pyloudnorm), limits peaks, and writes MP3 (320 kbps, ID3 tags),
WAV (16-bit) or FLAC (24-bit) into <output>/export. Prints one RESULT line.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_tools import encode_mp3, fade, integrated_lufs, limit_peaks, trim_silence


def tag_file(path, fmt, title, artist, album, comment):
    from mutagen.flac import FLAC
    from mutagen.id3 import COMM, ID3, TALB, TIT2, TPE1
    if fmt == "mp3":
        tags = ID3()
        tags.add(TIT2(encoding=3, text=title))
        tags.add(TPE1(encoding=3, text=artist))
        tags.add(TALB(encoding=3, text=album))
        if comment:
            tags.add(COMM(encoding=3, lang="eng", desc="", text=comment))
        tags.save(str(path))
    elif fmt == "flac":
        tags = FLAC(str(path))
        tags["title"], tags["artist"], tags["album"] = title, artist, album
        if comment:
            tags["comment"] = comment
        tags.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--song", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="the song's folder; files go to <output>/export")
    parser.add_argument("--name", default="audio", help="output base name")
    parser.add_argument("--format", choices=("mp3", "wav", "flac"), default="mp3")
    parser.add_argument("--lufs", type=float, default=None, help="target integrated loudness (e.g. -14 for streaming); omit to keep the level")
    parser.add_argument("--ceiling-db", type=float, default=-1.0, help="peak ceiling in dBFS")
    parser.add_argument("--fade-in", type=float, default=0.0)
    parser.add_argument("--fade-out", type=float, default=0.0)
    parser.add_argument("--trim", action="store_true", help="cut silence at the start and end")
    parser.add_argument("--title", default="")
    parser.add_argument("--artist", default="Music Gen Studio")
    parser.add_argument("--album", default="Music Gen Studio")
    parser.add_argument("--comment", default="")
    args = parser.parse_args()
    start = time.perf_counter()
    data, rate = sf.read(str(args.song), dtype="float32", always_2d=True)
    if args.trim:
        data = trim_silence(data, rate)
    data = fade(np.array(data, dtype=np.float64), rate, args.fade_in, args.fade_out)
    before = integrated_lufs(data, rate)
    ceiling = 10 ** (args.ceiling_db / 20)
    if args.lufs is not None and before is not None:
        gain_db = float(np.clip(args.lufs - before, -30, 24))
        data = data * (10 ** (gain_db / 20))
    data = limit_peaks(data, ceiling=ceiling, rate=rate)
    after = integrated_lufs(data, rate)
    peak = float(np.abs(data).max()) if data.size else 0.0
    out_dir = args.output / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.name}.{args.format}"
    if args.format == "mp3":
        target.write_bytes(encode_mp3(data, rate))
    elif args.format == "wav":
        sf.write(str(target), data, rate, subtype="PCM_16")
    else:
        sf.write(str(target), data, rate, subtype="PCM_24")
    if args.format != "wav":
        tag_file(target, args.format, args.title or args.name, args.artist, args.album, args.comment)
    summary = {"file": target.relative_to(args.output).as_posix(), "format": args.format, "seconds": round(data.shape[0] / rate, 1),
               "lufs_before": None if before is None else round(before, 1), "lufs_after": None if after is None else round(after, 1),
               "peak_dbfs": round(20 * np.log10(max(peak, 1e-9)), 2), "bytes": target.stat().st_size,
               "total_seconds": round(time.perf_counter() - start, 1)}
    print("RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
