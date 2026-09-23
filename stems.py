#!/usr/bin/env python
"""Split a song into stems with Demucs (runs in .venv-voice).

    .venv-voice\\Scripts\\python.exe stems.py --song outputs\\x\\audio.flac --output outputs\\x

Writes vocals / drums / bass / other / accompaniment as 44.1 kHz FLAC into <output>/stems
(the same cache the voice and karaoke features use) and prints one RESULT line.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf

from audio_tools import separate_stems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--song", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="the song's folder")
    args = parser.parse_args()
    start = time.perf_counter()
    stems = separate_stems(args.song, args.output / "stems")
    seconds = sf.info(str(stems["vocals"])).duration
    print("RESULT " + json.dumps({"stems": {name: path.relative_to(args.output).as_posix() for name, path in stems.items()},
                                  "seconds": round(seconds, 1), "total_seconds": round(time.perf_counter() - start, 1)}), flush=True)


if __name__ == "__main__":
    main()
