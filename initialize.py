#!/usr/bin/env python
"""Set up Music Gen Studio in one go: every step of the README's "Setup (Windows)".

    Setup Music Gen Studio.bat           double-click; finds Python 3.10-3.13 and runs this
    py -3.12 initialize.py               the same from a console
    py -3.12 initialize.py --cuda cu126  another PyTorch CUDA build (default cu128, as in the README)

YuE submodule -> YuE\\.venv -> CUDA PyTorch -> yue2 + requirements.txt -> model weights -> GPU check.
Safe to run again, e.g. after a git pull: finished steps are skipped, whatever the repo now pins
differently is brought in line, and interrupted downloads resume.

The CUDA PyTorch wheel is about 2.9 GB and the PyTorch CDN can be slow on a single connection, so it
is fetched over parallel ranged requests and checked against the index's sha256 before pip installs
it. If that route fails for any reason, the plain pip command from the README is used instead.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / "YuE"
VENV = REPO / ".venv"
VENV_PY = VENV / "Scripts" / "python.exe"
PYTHONS = ((3, 10), (3, 13))            # upstream needs 3.10+; its numpy pin has no wheels for 3.14
TORCH_INDEX = "https://download.pytorch.org/whl/{cuda}"
AGENT = {"User-Agent": "music-gen-studio-setup/1.0"}      # the PyTorch CDN refuses urllib's default agent
CONNECTIONS = 8
DOWNLOADS = Path(tempfile.gettempdir()) / "music-gen-setup"

CHECK = """
import sys
import torch
import python_multipart, qrcode, uvicorn, webui          # the app and everything it needs import cleanly
from run_lowvram import resolve_vram_mode
print(f"  torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
if not torch.cuda.is_available():
    print("  PyTorch cannot see an NVIDIA GPU: install or update the NVIDIA driver, then run this again.")
    sys.exit(3)
gpu = torch.cuda.get_device_properties(0)
print(f"  {gpu.name}: {gpu.total_memory / 2**30:.1f} GiB, compute capability {gpu.major}.{gpu.minor}")
if (gpu.major, gpu.minor) < (8, 0):
    print("  YuE2 needs BF16, i.e. compute capability 8.0 or newer (RTX 30-series or later).")
    sys.exit(3)
print(f"  The app will run in {resolve_vram_mode('auto')} VRAM mode"
      + ("" if (gpu.major, gpu.minor) >= (8, 9) else "; --quantization fp8 needs compute capability 8.9+, so leave it off"))
"""


class SetupError(Exception):
    """A step failed in a way the user has to fix; the message says how."""


def run(*args):
    """Run a command in the repo folder, echoing it README-style; raise if it fails."""
    shown = [str(a.relative_to(HERE)) if isinstance(a, Path) and a.is_relative_to(HERE) else str(a) for a in args]
    print("  > " + subprocess.list2cmdline(shown), flush=True)
    code = subprocess.call([str(a) for a in args], cwd=HERE)
    if code:
        raise SetupError(f"that command failed (exit code {code}); its output is above")


def venv_python(code):
    """Run a snippet in the venv: its stripped stdout, or None if it fails."""
    result = subprocess.run([str(VENV_PY), "-c", code], cwd=HERE, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def nvidia_gpu():
    """The driver's view of the GPU ('NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB'), or None."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    result = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                            capture_output=True, text=True)
    lines = result.stdout.strip().splitlines() if result.returncode == 0 else []
    return lines[0] if lines else None


# ── steps ─────────────────────────────────────────────────────────────────────

def step_submodule():
    git = shutil.which("git")
    if (HERE / ".git").exists() and git:
        state = subprocess.run([git, "submodule", "status", "YuE"], cwd=HERE, capture_output=True, text=True).stdout
        if state.startswith(" "):
            print("  YuE is checked out at the commit this repo pins")
            return
        if state.startswith("+"):
            print("  YuE is at a different commit than the one this repo pins; checking out the pinned one")
        run("git", "submodule", "update", "--init", "--recursive")
    elif (REPO / "pyproject.toml").is_file():
        print("  YuE is present")                           # not a git clone, or no git: use what is there
    elif not (HERE / ".git").exists():
        raise SetupError("YuE\\ is empty and this folder is not a git clone (ZIP downloads leave submodules out). "
                         "Clone it instead: git clone --recurse-submodules "
                         "https://github.com/jrlabanza/music-generator.git")
    else:
        raise SetupError("git is needed to fetch the YuE submodule: install Git for Windows from "
                         "https://git-scm.com/download/win and run this again.")
    if not (REPO / "pyproject.toml").is_file():
        raise SetupError("YuE\\pyproject.toml is still missing after fetching the submodule")


def step_venv():
    if VENV_PY.is_file():
        version = venv_python("import sys; print('%d.%d' % sys.version_info[:2])")
        if version is None:
            raise SetupError("YuE\\.venv exists but its Python does not start (was that Python uninstalled?). "
                             "Delete the YuE\\.venv folder and run this again.")
        if not PYTHONS[0] <= tuple(map(int, version.split("."))) <= PYTHONS[1]:
            raise SetupError(f"YuE\\.venv uses Python {version}, but 3.10-3.13 is needed. "
                             "Delete the YuE\\.venv folder and run this again.")
        print(f"  YuE\\.venv exists (Python {version})")
    else:
        if not PYTHONS[0] <= sys.version_info[:2] <= PYTHONS[1]:
            raise SetupError(f"this is Python {sys.version.split()[0]}; the environment needs 3.10-3.13 "
                             "(3.12 recommended)")
        run(sys.executable, "-m", "venv", VENV)
    run(VENV_PY, "-m", "pip", "install", "--upgrade", "--quiet", "pip")


def step_torch(cuda):
    match = re.search(r'"torch==([^"]+)"', (REPO / "pyproject.toml").read_text(encoding="utf-8"))
    if not match:
        raise SetupError("no torch==... pin found in YuE\\pyproject.toml")
    pin = match.group(1)
    have = venv_python("import torch; print(torch.__version__)")
    if have and have.startswith(pin + "+cu"):
        print(f"  torch {have} is installed")
        return
    if have:
        print(f"  torch {have} is installed; replacing it with the CUDA build of {pin}")
    try:
        wheel = fetch_torch_wheel(pin, cuda)
    except Exception as exc:                            # any trouble on the fast route: the README's pip command
        print(f"\n  parallel download unavailable ({exc}); letting pip download it instead, which can be slow",
              flush=True)
        run(VENV_PY, "-m", "pip", "install", f"torch=={pin}+{cuda}", "--index-url", TORCH_INDEX.format(cuda=cuda))
    else:
        run(VENV_PY, "-m", "pip", "install", wheel)    # a failure here keeps the verified wheel for the next run
        shutil.rmtree(DOWNLOADS, ignore_errors=True)
    have = venv_python("import torch; print(torch.__version__)")
    if not (have and have.startswith(pin + "+cu")):
        raise SetupError(f"expected torch {pin}+{cuda}, found {have or 'nothing'}")


def step_packages():
    run(VENV_PY, "-m", "pip", "install", "-e", REPO)
    run(VENV_PY, "-m", "pip", "install", "-r", HERE / "requirements.txt")
    check = subprocess.run([str(VENV_PY), "-m", "pip", "check"], cwd=HERE, capture_output=True, text=True)
    print("  pip check: no broken requirements" if check.returncode == 0 else
          "  warning, pip check reports:\n" + check.stdout.rstrip())


def models_verified():
    return venv_python("from yue2.storage import model_identity\n"
                       "for name in ('YuE2-3B', 'YuE2-Vae'): model_identity('models/' + name)") is not None


def step_models():
    try:
        run(VENV_PY, HERE / "download_models.py")       # resumes, and skips files that are already complete
    except SetupError:
        if not models_verified():
            raise
        print("  could not reach Hugging Face, but the weights already here are complete")
    if not models_verified():
        raise SetupError("the model weights fail their sha256 check: delete models\\YuE2-3B and models\\YuE2-Vae "
                         "and run this again")
    print("  weights match the sha256 in their weights_manifest.json")


def step_check():
    if subprocess.call([str(VENV_PY), "-c", CHECK], cwd=HERE):
        raise SetupError("everything is installed, but the check above failed")


# ── fast PyTorch download ─────────────────────────────────────────────────────

def urlopen(url, **headers):
    return urllib.request.urlopen(urllib.request.Request(url, headers={**AGENT, **headers}), timeout=60)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_torch_wheel(pin, cuda):
    """Download this venv's CUDA wheel of torch==pin into DOWNLOADS; its path once sha256-verified."""
    tags = venv_python("import sys, sysconfig; print('cp%d%d' % sys.version_info[:2], sysconfig.get_platform())")
    python_tag, platform = tags.split()
    name = f"torch-{pin}+{cuda}-{python_tag}-{python_tag}-{platform.replace('-', '_')}.whl"
    index = TORCH_INDEX.format(cuda=cuda) + "/torch/"
    with urlopen(index) as response:
        page = response.read().decode("utf-8", "replace")
    link = re.search(r'href="((?:[^"#]*/)?' + re.escape(urllib.parse.quote(name)) + r')#sha256=([0-9a-f]{64})"',
                     page, re.IGNORECASE)
    if not link:
        raise RuntimeError(f"{name} is not listed on {index}")
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    target = DOWNLOADS / name
    parallel_download(urllib.parse.urljoin(index, link.group(1)), target, link.group(2).lower())
    return target


def parallel_download(url, target, sha256):
    """Fetch url into target over CONNECTIONS ranged requests; resumable, sha256-checked."""
    if target.is_file():
        if sha256_of(target) == sha256:
            print(f"  using {target.name}, downloaded and verified earlier")
            return
        target.unlink()
    with urlopen(url, Range="bytes=0-0") as probe:
        if probe.status != 206:
            raise RuntimeError("the server does not support ranged downloads")
        total = int(probe.headers["Content-Range"].rsplit("/", 1)[1])
    size = total // CONNECTIONS
    spans = [(i * size, total - 1 if i == CONNECTIONS - 1 else (i + 1) * size - 1) for i in range(CONNECTIONS)]
    parts = [target.with_name(f"{target.name}.part{i}") for i in range(CONNECTIONS)]
    done = [part.stat().st_size if part.is_file() else 0 for part in parts]
    for i, (start, end) in enumerate(spans):
        if done[i] > end - start + 1:                   # left over from an interrupted join: start that span again
            parts[i].unlink()
            done[i] = 0
    problems = []

    def fetch(i):
        start, end = spans[i]
        stalls = 0
        while start + done[i] <= end:
            before, error = done[i], None
            try:
                with urlopen(url, Range=f"bytes={start + done[i]}-{end}") as response, open(parts[i], "ab") as out:
                    if response.status != 206:
                        raise RuntimeError(f"HTTP {response.status} instead of a partial response")
                    while block := response.read(1 << 20):
                        out.write(block)
                        done[i] += len(block)
            except Exception as exc:                    # dropped connection: carry on from where it stopped
                error = exc
            if done[i] > before:
                stalls = 0
                continue
            stalls += 1
            if stalls == 10:
                problems.append(f"part {i + 1}: {error or 'no data'}")
                return
            time.sleep(3)

    print(f"  downloading {target.name} ({total / 2**20:.0f} MiB) over {CONNECTIONS} connections", flush=True)
    threads = [threading.Thread(target=fetch, args=(i,), daemon=True) for i in range(CONNECTIONS)]
    for thread in threads:
        thread.start()
    started, resumed = time.time(), sum(done)
    while any(thread.is_alive() for thread in threads):
        time.sleep(1)
        got = sum(done)
        rate = (got - resumed) / (time.time() - started)
        left = f"~{(total - got) / rate / 60:.0f} min left" if rate > 0 else ""
        print(f"\r  {got / 2**20:6.0f} / {total / 2**20:.0f} MiB  {rate / 2**20:5.1f} MiB/s  {left}   ",
              end="", flush=True)
    print()
    if problems:
        raise RuntimeError("; ".join(problems))
    with open(parts[0], "ab") as out:                   # join onto the first part: no second full-size copy
        for part in parts[1:]:
            with open(part, "rb") as stream:
                shutil.copyfileobj(stream, out, 16 << 20)
            part.unlink()
    parts[0].replace(target)
    if sha256_of(target) != sha256:
        target.unlink()
        raise RuntimeError("the downloaded wheel failed its sha256 check")
    print("  sha256 matches the PyTorch index", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cuda", default="cu128", help="PyTorch CUDA build to install (default: cu128)")
    args = parser.parse_args()
    if sys.version_info < PYTHONS[0]:
        print(f"Run this with Python 3.10-3.13 (this is {sys.version.split()[0]}), e.g. py -3.12 initialize.py")
        return 1
    steps = [("YuE submodule", step_submodule),
             ("Python environment (YuE\\.venv)", step_venv),
             (f"PyTorch with CUDA ({args.cuda})", lambda: step_torch(args.cuda)),
             ("yue2 and the web app's packages", step_packages),
             ("Model weights (about 7.3 GB)", step_models),
             ("Check", step_check)]
    gpu = nvidia_gpu()
    print(f"GPU: {gpu}" if gpu else
          "Warning: no NVIDIA driver found (nvidia-smi is missing). Music Gen Studio needs an NVIDIA GPU; setup "
          "continues, but songs can only be generated once the driver is installed.", flush=True)
    started = time.time()
    try:
        for number, (title, action) in enumerate(steps, 1):
            print(f"\n[{number}/{len(steps)}] {title}", flush=True)
            action()
    except SetupError as exc:
        print(f"\nSetup stopped: {exc}", flush=True)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Run setup again to continue where it stopped.", flush=True)
        return 130
    print(f"\nSetup complete ({(time.time() - started) / 60:.0f} min).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
