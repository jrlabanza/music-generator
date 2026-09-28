# YuE 2 – Music Gen Studio on Linux (Docker)

Everything in this directory runs the app in a container with **its own Python
3.12 and PyTorch build**, so it cannot collide with anything else on the
machine. The repo itself is bind-mounted into the container: code, models and
outputs stay right here on disk and the image only holds the environment.

Tested on Ubuntu 26.04, RTX 3070 Laptop 8 GB, NVIDIA driver 595 / CUDA 13.2.

## One-time setup

```bash
sudo bash linux/install-docker.sh     # Docker Engine + NVIDIA Container Toolkit
# then log out and back in so the docker group applies
```

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
`linux/Dockerfile`.

## Troubleshooting

- **"cannot reach the docker daemon"** – you are in the `docker` group but this
  session predates it. Log out and back in (Ubuntu 26.04 has no `newgrp`).
  Until then `run.sh` falls back to passwordless `sudo` if you have it.
- **Out of VRAM** – make sure no other tool is running (`docker ps`).
