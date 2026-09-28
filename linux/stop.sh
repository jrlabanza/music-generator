#!/usr/bin/env bash
# Stop YuE 2 – Music Gen Studio (and Chatterbox if it is running).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Docker access: docker group, or passwordless sudo until the group is active
# (Ubuntu 26.04 has no newgrp/sg - a new group needs a fresh login). sudo resets
# HOME and drops the environment, so the variables compose needs are passed
# through explicitly - otherwise AI_CACHE would resolve to /root and Docker
# would create it root-owned, unwritable by the non-root container.
export AI_UID="$(id -u)" AI_GID="$(id -g)"
export AI_CACHE="${AI_CACHE:-$HOME/.cache/ai-tools}"
mkdir -p "$AI_CACHE"/{hf,torch,xdg,mpl,numba,gradio}
if docker info >/dev/null 2>&1; then DK=(docker)
elif sudo -n docker info >/dev/null 2>&1; then DK=(sudo -n --preserve-env=AI_UID,AI_GID,AI_CACHE,AI_BIND docker)
else echo "cannot reach the docker daemon (installed? in the docker group? logged out and in?)" >&2; exit 1; fi
"${DK[@]}" compose -f "$HERE/compose.yml" --profile chatterbox stop
