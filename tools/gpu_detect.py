#!/usr/bin/env python
"""Which graphics card is this, and which PyTorch build does it need?

Python port of the AI Studio Hub's reference detection (docs/gpu-detect.ps1 / docs/gpu.md); this
studio's Windows initialiser is Python, so it uses this module instead of the PowerShell one. Standard
library only, so it runs before anything is installed.

    python tools/gpu_detect.py                  # print the detection as JSON
    python tools/gpu_detect.py --write .gpu.json  # also write the studio's .gpu.json
    python tools/gpu_detect.py --gpu amd --gfx gfx1100    # force a vendor / target

    from gpu_detect import detect, write_gpu_json
    gpu = detect()            # {vendor, backend, name, gfx, vram_mb, driver, cuda_tag, torch_args, note}
    write_gpu_json(".gpu.json", gpu, torch_version="2.12.0+rocm7.14.1")

Detection order: nvidia-smi (vendor nvidia, backend cuda) -> an AMD/Radeon display adapter (Windows:
Win32_VideoController; Linux: rocm-smi, else /sys/class/drm vendor 0x1002) -> cpu. ``torch_args`` is
the argument list for ``pip install``: on AMD Windows it is AMD's native PyTorch-on-ROCm wheel set.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROCM_WIN_VERSION = "7.14.1"
ROCM_WIN_TORCH = "2.12.0"
ROCM_WIN_VISION = "0.27.0"
ROCM_WIN_AUDIO = "2.11.0"
ROCM_WIN_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"
ROCM_LINUX_INDEX = "https://download.pytorch.org/whl/rocm7.1"
CUDA_INDEX = "https://download.pytorch.org/whl/{tag}"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

VENDORS = ("nvidia", "amd", "cpu")
WINDOWS = sys.platform == "win32"

# ── AMD: marketing name -> LLVM target (docs/gpu.md has the table) ───────────
GFX_TABLE = (
    (r"rx 9070|r9700", "gfx1201"),                  # RDNA 4, FP8 capable
    (r"rx 9060", "gfx1200"),                        # RDNA 4
    (r"rx 7900|w7900|w7800", "gfx1100"),            # RDNA 3
    (r"rx 7800|rx 7700|w7700", "gfx1101"),
    (r"rx 7600|rx 7650", "gfx1102"),
    (r"780m|760m", "gfx1103"),                      # Ryzen 7040/8040 iGPU
    (r"880m|890m", "gfx1150"),                      # Ryzen AI 300
    (r"8060s|8050s|8040s", "gfx1151"),              # Ryzen AI MAX
    (r"rx 6950|rx 6900|rx 6800|w6800", "gfx1030"),  # RDNA 2, in the multi-arch wheel only
)


def amd_gfx(name):
    """The LLVM target from the adapter's marketing name; "" when unknown."""
    lowered = (name or "").lower()
    for pattern, gfx in GFX_TABLE:
        if re.search(pattern, lowered):
            return gfx
    return ""


def amd_support(name, gfx):
    """ok (a known target) | all (use the device-all wheel) | unsupported (not in AMD's wheels)."""
    if gfx:
        return "ok"
    lowered = (name or "").lower()
    if re.search(r"rx 6700|rx 6750", lowered):
        return "all"
    if re.search(r"rx 6[0-6]\d\d|rx 5\d\d\d|vega|rx 5\d0|radeon r", lowered):
        return "unsupported"
    return "all"


def cpu_args():
    return ["torch", "torchvision", "torchaudio", "--index-url", CPU_INDEX]


def rocm_windows_args(gfx):
    device = f"device-{gfx}" if gfx else "device-all"
    return ["--index-url", ROCM_WIN_INDEX,
            f"torch[{device}]=={ROCM_WIN_TORCH}+rocm{ROCM_WIN_VERSION}",
            f"torchvision[{device}]=={ROCM_WIN_VISION}+rocm{ROCM_WIN_VERSION}",
            f"torchaudio=={ROCM_WIN_AUDIO}+rocm{ROCM_WIN_VERSION}"]


def _run(args, timeout=30):
    """stdout of a command, or "" if it is missing or fails."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                                encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


# ── 1. NVIDIA ─────────────────────────────────────────────────────────────────
def _nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if exe is None and WINDOWS:
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(candidate) if candidate.is_file() else None
    return exe


def detect_nvidia():
    exe = _nvidia_smi()
    if not exe:
        return None
    out = _run([exe, "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader,nounits"])
    lines = [line for line in out.splitlines() if line.strip()]
    if not lines:
        return None
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 3:
        return None
    driver, name, vram = parts[0], parts[1], parts[2]
    try:
        major = int(driver.split(".")[0])
    except ValueError:
        major = 0
    tag = "cu130" if major >= 580 else "cu126"
    try:
        vram_mb = int(float(vram))
    except ValueError:
        vram_mb = 0
    return {"vendor": "nvidia", "backend": "cuda", "name": name, "gfx": "", "vram_mb": vram_mb,
            "driver": driver, "cuda_tag": tag,
            "torch_args": ["torch", "torchvision", "torchaudio", "--index-url", CUDA_INDEX.format(tag=tag)],
            "note": ""}


# ── 2. AMD ────────────────────────────────────────────────────────────────────
def _windows_adapters():
    """[{Name, AdapterRAM, DriverVersion}] from Win32_VideoController (AMD/Radeon only)."""
    out = _run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json"])
    try:
        data = json.loads(out) if out.strip() else []
    except ValueError:
        return []
    if isinstance(data, dict):
        data = [data]
    adapters = []
    for item in data or []:
        name = str((item or {}).get("Name") or "")
        if re.search(r"amd|radeon", name, re.IGNORECASE):
            adapters.append({"Name": name, "AdapterRAM": int((item.get("AdapterRAM") or 0)),
                             "DriverVersion": str(item.get("DriverVersion") or "")})
    return adapters


def _windows_registry_vram_mb(name):
    """The display class key's HardwareInformation.qwMemorySize: AdapterRAM caps at 4 GB."""
    try:
        import winreg
    except ImportError:
        return 0
    root = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root) as cls:
            count = winreg.QueryInfoKey(cls)[0]
            for index in range(count):
                sub = winreg.EnumKey(cls, index)
                if not re.fullmatch(r"\d{4}", sub):
                    continue
                try:
                    with winreg.OpenKey(cls, sub) as key:
                        desc = winreg.QueryValueEx(key, "DriverDesc")[0]
                        if not desc or (desc != name and desc not in name):
                            continue
                        value = winreg.QueryValueEx(key, "HardwareInformation.qwMemorySize")[0]
                except OSError:
                    continue
                if isinstance(value, bytes):
                    value = int.from_bytes(value, "little")
                if value:
                    return int(int(value) // 2**20)
    except OSError:
        pass
    return 0


def _linux_amd():
    """(name, vram_mb, driver) for an amdgpu card, or None."""
    name, vram_mb, driver = "", 0, ""
    out = _run(["rocm-smi", "--showproductname"]) or _run(["amd-smi", "static"])
    for line in out.splitlines():
        if re.search(r"card series|market name|product name", line, re.IGNORECASE) and ":" in line:
            candidate = line.split(":", 1)[1].strip()
            if candidate and not name:
                name = candidate
    cards = sorted(Path("/sys/class/drm").glob("card[0-9]*/device/vendor"))
    found = False
    for vendor_file in cards:
        try:
            if vendor_file.read_text().strip().lower() != "0x1002":
                continue
        except OSError:
            continue
        found = True
        device = vendor_file.parent
        try:
            vram_mb = max(vram_mb, int((device / "mem_info_vram_total").read_text().strip()) // 2**20)
        except (OSError, ValueError):
            pass
        if not name:
            try:
                name = (device / "product_name").read_text().strip()
            except OSError:
                pass
    if not found and not name:
        return None
    try:
        driver = Path("/sys/module/amdgpu/version").read_text().strip()
    except OSError:
        pass
    return (name or "AMD Radeon (model unknown)", vram_mb, driver)


def detect_amd(force=False, gfx=""):
    gpu = {"vendor": "amd", "backend": "rocm", "name": "", "gfx": "", "vram_mb": 0, "driver": "", "cuda_tag": "",
           "torch_args": [], "note": ""}
    if WINDOWS:
        adapters = _windows_adapters()
        if not adapters and force:
            adapters = [{"Name": "AMD (forced)", "AdapterRAM": 0, "DriverVersion": ""}]
        if not adapters:
            return None
        best = max(adapters, key=lambda a: a["AdapterRAM"])
        gpu["name"], gpu["driver"] = best["Name"], best["DriverVersion"]
        gpu["vram_mb"] = _windows_registry_vram_mb(best["Name"]) or int(best["AdapterRAM"] // 2**20)
    else:
        card = _linux_amd()
        if card is None and force:
            card = ("AMD (forced)", 0, "")
        if card is None:
            return None
        gpu["name"], gpu["vram_mb"], gpu["driver"] = card
    gpu["gfx"] = gfx or amd_gfx(gpu["name"])
    support = amd_support(gpu["name"], gpu["gfx"])
    if support == "unsupported":
        gpu["backend"] = "cpu"
        gpu["torch_args"] = cpu_args()
        gpu["note"] = (f"{gpu['name']} is not in AMD's PyTorch-on-Windows wheels (RDNA 3 / RDNA 4 / RX 6800+ only)"
                       " - installing the CPU build")
        return gpu
    if WINDOWS:
        gpu["torch_args"] = rocm_windows_args(gpu["gfx"])
        if not gpu["gfx"]:
            gpu["note"] = (f"unknown Radeon model '{gpu['name']}': installing the multi-architecture build "
                           "(device-all, larger download)")
    else:
        gpu["torch_args"] = ["torch", "torchvision", "torchaudio", "--index-url", ROCM_LINUX_INDEX]
    return gpu


# ── 3. detect() ───────────────────────────────────────────────────────────────
def detect(force="", gfx=""):
    """The card and the PyTorch build it needs. ``force`` is nvidia|amd|cpu (an override), ``gfx`` an
    AMD target override. Never raises: with nothing found it returns the cpu entry with a note."""
    force = (force or "").lower()
    if force and force not in VENDORS:
        raise ValueError(f"--gpu must be one of {', '.join(VENDORS)}, not {force!r}")
    if force in ("", "nvidia"):
        gpu = detect_nvidia()
        if gpu:
            return gpu
        if force == "nvidia":
            gpu = {"vendor": "nvidia", "backend": "cuda", "name": "NVIDIA (forced)", "gfx": "", "vram_mb": 0,
                   "driver": "", "cuda_tag": "cu126",
                   "torch_args": ["torch", "torchvision", "torchaudio", "--index-url", CUDA_INDEX.format(tag="cu126")],
                   "note": "nvidia-smi did not answer; NVIDIA forced on the command line"}
            return gpu
    if force in ("", "amd"):
        gpu = detect_amd(force=(force == "amd"), gfx=gfx)
        if gpu:
            return gpu
    return {"vendor": "cpu", "backend": "cpu", "name": "", "gfx": "", "vram_mb": 0, "driver": "", "cuda_tag": "",
            "torch_args": cpu_args(),
            "note": "no NVIDIA or AMD graphics card found - installing the CPU build; generation will be very slow"}


def describe(gpu):
    """One line for the initialiser's log: card, VRAM and the build it maps to."""
    if gpu["backend"] == "cuda":
        what = f"PyTorch CUDA {gpu['cuda_tag']}"
    elif gpu["backend"] == "rocm":
        what = f"PyTorch ROCm {ROCM_WIN_VERSION} ({gpu['gfx'] or 'device-all'})" if WINDOWS else "PyTorch ROCm (Linux)"
    else:
        what = "PyTorch CPU build"
    line = f"GPU: {gpu['name'] or 'none'}  {gpu['vram_mb']} MB  ->  {what}"
    return line + (f"\n      note: {gpu['note']}" if gpu["note"] else "")


# ── .gpu.json ─────────────────────────────────────────────────────────────────
def write_gpu_json(path, gpu, torch_version=""):
    """The studio's .gpu.json (docs/gpu.md schema); launchers and the app read it."""
    record = {"vendor": gpu["vendor"], "name": gpu["name"], "gfx": gpu["gfx"], "vram_mb": gpu["vram_mb"],
              "backend": gpu["backend"], "torch": torch_version, "driver": gpu.get("driver", ""),
              "platform": "windows" if WINDOWS else "linux",
              "detected": datetime.datetime.now().replace(microsecond=0).isoformat()}
    Path(path).write_text(json.dumps(record) + "\n", encoding="utf-8")
    return record


def read_gpu_json(path):
    """The record written above, or None when the file is missing or unreadable."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and record.get("backend") in ("cuda", "rocm", "cpu") else None


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Detect the graphics card and the PyTorch build it needs")
    parser.add_argument("--gpu", choices=VENDORS, default="", help="override the detected vendor")
    parser.add_argument("--gfx", default="", help="override the AMD target (e.g. gfx1100)")
    parser.add_argument("--write", metavar="PATH", help="also write .gpu.json there (torch from this interpreter)")
    args = parser.parse_args(argv)
    gpu = detect(args.gpu, args.gfx)
    print(json.dumps(gpu, indent=2))
    if args.write:
        try:
            import torch
            version = torch.__version__
        except Exception:                                   # noqa: BLE001 - no torch here is fine
            version = ""
        write_gpu_json(args.write, gpu, version)
        print(f"wrote {args.write}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
