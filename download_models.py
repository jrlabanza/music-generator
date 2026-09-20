#!/usr/bin/env python
"""Download the YuE2 weights (about 7.3 GB) into models/ as plain files.

    YuE\\.venv\\Scripts\\python.exe download_models.py

Plain files are used instead of the Hugging Face cache because the cache
relies on symlinks, which Windows blocks unless Developer Mode is on.
Only the files the pipeline needs are fetched (no demo audio).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOS = {"YuE2-3B": "m-a-p/YuE2-3B", "YuE2-Vae": "m-a-p/YuE2-Vae"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, default=HERE / "models")
    parser.add_argument("--legacy-vae", action="store_true",
                        help="also fetch m-a-p/YuE2-Vae-legacy, the decoder used for the published benchmarks")
    parser.add_argument("--sheetsage2", action="store_true",
                        help="also fetch m-a-p/SheetSage2 and its MERT-v2-FullSong encoder (2.6 GB) for the cover feature")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    from yue2.storage import MODEL_FILES, MODEL_LICENSES
    patterns = sorted(MODEL_FILES) + ["model-?????-of-?????.safetensors"] + ["licenses/" + n for n in sorted(MODEL_LICENSES)]
    repos = dict(REPOS)
    if args.legacy_vae:
        repos["YuE2-Vae-legacy"] = "m-a-p/YuE2-Vae-legacy"
    extra = {}
    if args.sheetsage2:
        extra = {"SheetSage2": ("m-a-p/SheetSage2", ["*.py", "*.json", "*.txt", "*.safetensors"]),
                 "MERT-v2-FullSong": ("m-a-p/MERT-v2-FullSong", ["*.py", "*.json", "*.safetensors"])}
    start = time.perf_counter()
    for name, repo in repos.items():
        target = args.models_dir / name
        print(f"{repo} -> {target}", flush=True)
        snapshot_download(repo, local_dir=target, allow_patterns=patterns)
        print(f"  done ({time.perf_counter() - start:.0f}s elapsed)", flush=True)
    for name, (repo, allow) in extra.items():
        target = args.models_dir / name
        print(f"{repo} -> {target}", flush=True)
        snapshot_download(repo, local_dir=target, allow_patterns=allow)
        print(f"  done ({time.perf_counter() - start:.0f}s elapsed)", flush=True)
    if args.sheetsage2:                      # SheetSage2 loads its encoder by hub id; point it at the local copy
        import json
        config = args.models_dir / "SheetSage2" / "config.json"
        data = json.loads(config.read_text(encoding="utf-8"))
        local = str((args.models_dir / "MERT-v2-FullSong").resolve())

        def patch(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if value == "m-a-p/MERT-v2-FullSong":
                        obj[key] = local
                    else:
                        patch(value)
            elif isinstance(obj, list):
                for value in obj:
                    patch(value)
        patch(data)
        config.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("SheetSage2 config points at", local)
    print("All models downloaded.")


if __name__ == "__main__":
    main()
