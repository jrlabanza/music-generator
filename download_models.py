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
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    from yue2.storage import MODEL_FILES, MODEL_LICENSES
    patterns = sorted(MODEL_FILES) + ["model-?????-of-?????.safetensors"] + ["licenses/" + n for n in sorted(MODEL_LICENSES)]
    repos = dict(REPOS)
    if args.legacy_vae:
        repos["YuE2-Vae-legacy"] = "m-a-p/YuE2-Vae-legacy"
    start = time.perf_counter()
    for name, repo in repos.items():
        target = args.models_dir / name
        print(f"{repo} -> {target}", flush=True)
        snapshot_download(repo, local_dir=target, allow_patterns=patterns)
        print(f"  done ({time.perf_counter() - start:.0f}s elapsed)", flush=True)
    print("All models downloaded.")


if __name__ == "__main__":
    main()
