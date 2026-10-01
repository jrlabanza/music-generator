# YuE 2 – Music Gen Studio on Linux (Docker)

Everything in this directory runs the app in a container with **its own Python
3.12 and PyTorch build**, so it cannot collide with anything else on the
machine. The repo itself is bind-mounted into the container: code, models and
outputs stay right here on disk and the image only holds the environment.

Tested on Ubuntu 26.04, RTX 3070 Laptop 8 GB, NVIDIA driver 595 / CUDA 13.2.
AMD Radeon cards get their own ROCm image (see *AMD / ROCm* below; built and
booted here without an AMD card, awaiting verification on AMD hardware).

## One-time setup

One command does everything - driver check, Docker (+ the NVIDIA toolkit on
NVIDIA), the image, the models, a boot test and an app-menu entry. Safe to re-run; it skips what is
already done and resumes interrupted downloads.

```bash
./linux/initialize.sh                 # essentials
./linux/initialize.sh --all-models    # every model the project knows about
./linux/initialize.sh --help          # --no-models --repair --no-test --no-desktop
./linux/initialize.sh --gpu amd       # force the AMD / ROCm setup (--gpu nvidia for CUDA)
```

On **Windows** the equivalent is `initialize.bat` in the repo root (same flags,
`-AllModels` style).

## Run

```bash
./linux/run.sh                # console in this window; Ctrl+C / close = stop
./linux/run.sh --background   # detached; browser opens when ready
./linux/stop.sh
```

First run builds the image (several GB of downloads). The app is at
**http://localhost:7863**. Ports bind to `127.0.0.1` only; to share on your
LAN put `AI_BIND=0.0.0.0` in `linux/.env`.

`./linux/install-desktop.sh` adds an app-menu entry that behaves like the old
`.bat` launcher (a console window; closing it stops the app).

Only one AI tool at a time fits in 8 GB of VRAM; `run.sh` stops any other
`ai.tool` container before starting.

## What is pinned, and why

`constraints.txt` is every package version read out of the working Windows
venv, and the Dockerfile installs `torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128`. Nothing was guessed: the
container reproduces the environment that was already known to work. Rebuild
only after editing `Dockerfile` or `constraints.txt`:

```bash
docker compose -f linux/compose.yml build
./linux/test.sh        # environment + real boot check
```

(`compose.rocm.yml` / `Dockerfile.rocm` for the AMD image; `test.sh` follows
`.gpu.json` or `AI_GPU=rocm`.)

## AMD / ROCm

AMD cards run the same app from a second image, **`ai/yue2:rocm`**, following
the AI Studio Hub's shared GPU contract (`docs/gpu.md` in the hub):

| | NVIDIA (CUDA) | AMD (ROCm) |
|---|---|---|
| files | `Dockerfile`, `compose.yml` | `Dockerfile.rocm`, `compose.rocm.yml` |
| base image | `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` | `rocm/dev-ubuntu-24.04:7.1.1` |
| torch (Python 3.12) | `2.10.0+cu128` | `2.10.0+rocm7.0` (same version, official PyTorch ROCm wheel) |
| helper venvs `/opt/venv-voice`, `/opt/venv-sheetsage2` (Python 3.11) | `torch 2.8.0+cu126` | `torch 2.8.0+rocm6.4` - the exact pin is still on the rocm6.4 index, so Seed-VC, demucs, Export and SheetSage2 keep their known-good versions |
| GPU access | `runtime: nvidia`, `NVIDIA_*` | `/dev/kfd` + `/dev/dri`, the host's `video`/`render` groups (`AI_VIDEO_GID`/`AI_RENDER_GID`, read with `getent` by the scripts), `seccomp=unconfined` |
| allocator | `PYTORCH_CUDA_ALLOC_CONF` | `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` |
| not in the image | - | nothing CUDA-only was in it anyway (no flash-attn / triton-windows / bitsandbytes); `TORCH_CUDA_ARCH_LIST` is dropped |

`./linux/initialize.sh` detects the card (`nvidia-smi`, else `rocm-smi`/`amd-smi`
or an `amdgpu` device in sysfs; `--gpu amd` forces it) and on AMD: installs
Docker without the NVIDIA toolkit (`install-docker.sh --no-nvidia`), checks
`/dev/kfd` and `/dev/dri` (the amdgpu driver is in the Ubuntu kernel - nothing
ROCm is installed on the host), adds you to the `video` and `render` groups,
builds the image, reads the gfx target with `rocminfo` inside the container
(`--gfx gfx1100` to force it) and writes **`.gpu.json`** (`"vendor": "amd",
"backend": "rocm"`, the `gfx`, `"torch": "2.10.0+rocm7.0"`). `run.sh`, `stop.sh`,
`test.sh` and `cli.sh` read that file to pick the compose file and image;
**`AI_GPU=rocm`** or **`AI_GPU=cuda`** in the environment forces one
(`AI_GPU=rocm ./linux/test.sh`). `test.sh` checks that torch is the HIP build
(`torch.version.hip`) on ROCm.

**Cards outside AMD's Linux matrix.** ROCm only ships kernels for some chips;
others run with a sibling's code through `HSA_OVERRIDE_GFX_VERSION`, which
`compose.rocm.yml` passes through from **`linux/.env`**. `initialize.sh` writes
it when it knows the chip: `10.3.0` for gfx1031/gfx1032 (RX 6700 / 6650 / 6600),
`11.0.0` for gfx1103 (Radeon 780M/760M iGPU). For another card put the line in
`linux/.env` yourself (`HSA_OVERRIDE_GFX_VERSION=11.0.0`) and restart.

**What is different in the app on AMD.** `torch.cuda.*` works as is (ROCm
presents itself as the `cuda` device), so the low-VRAM placement is unchanged.
There are no CUDA graphs and no fp8 before RDNA 4: the decoder runs the eager
path and `YUE_FP8=1` is switched off with a note in the log
(`run_lowvram.effective_quantization`). The 8 GB "low VRAM" swap through RAM
works unchanged.

**Status.** The image builds on this NVIDIA machine and the web UI boots in it
without any GPU device (HTTP 200, `backend: rocm`, eager decoder, fp8 off); the
model load itself needs a device (upstream `pipeline.py` asks for
`torch.cuda.current_device()` before placing anything), so with no card the
page shows "No HIP GPUs are available" instead of a loaded model. Generation
on an AMD card is still **awaiting verification on AMD hardware**.

## Model / cache location

HuggingFace and torch caches live in `~/.cache/ai-tools` on the host (override
with `AI_CACHE=/some/path`), shared with the other tools so nothing is
downloaded twice.

## When an extension needs a Python package

The container runs as your user and the venv is root-owned, so runtime
self-installs fail on purpose. Install into the running container:

```bash
docker exec -it -u root ai-yue2 /opt/venv/bin/pip install <pkg>
```

That survives stop/start but not a rebuild; to make it permanent add it to
`linux/Dockerfile` (and `linux/Dockerfile.rocm`).

## Troubleshooting

- **"cannot reach the docker daemon"** – you are in the `docker` group but this
  session predates it. Log out and back in (Ubuntu 26.04 has no `newgrp`).
  Until then `run.sh` falls back to passwordless `sudo` if you have it.
- **Out of VRAM** – make sure no other tool is running (`docker ps`).
- **AMD: "No GPU" / `torch.cuda.is_available()` is False in the container** –
  `/dev/kfd` missing (amdgpu driver not loaded), the container not in the
  `render` group (the scripts export `AI_RENDER_GID` from `getent group render`;
  check it matches `ls -l /dev/kfd`), or a chip outside AMD's matrix: set
  `HSA_OVERRIDE_GFX_VERSION` in `linux/.env` (see *AMD / ROCm*).
