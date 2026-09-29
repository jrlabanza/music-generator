#!/usr/bin/env bash
# Music Gen Studio (YuE 2). Port 7863, not webui.py's default 7860, which
# would collide with Forge.
#   --gpu-reserve-gib 2  headroom on the 3070's 8 GB (matches its own default)
#
# Anything that is not a webui.py flag runs instead of the server, inside the
# same environment - that is how linux/cli.sh gives you the README's Windows
# command lines:  entrypoint.sh python run_lowvram.py --request songs/x.json ...
set -euo pipefail
cd /app
export PYTHONUTF8=1
# YuE's package lives in YuE/src; bind-mounted source means no editable install.
export PYTHONPATH="/app/YuE/src:${PYTHONPATH:-}"
# The helper environments the README calls .venv-voice and .venv-sheetsage2
# are baked into the image (see Dockerfile); webui.py reads these overrides.
export MUSICGEN_VOICE_PY="${VOICE_ENV:-/opt/venv-voice}/bin/python"
export MUSICGEN_SHEETSAGE_PY="${SHEETSAGE_ENV:-/opt/venv-sheetsage2}/bin/python"

if [[ $# -gt 0 && $1 != --* ]]; then
  exec "$@"
fi

ARGS=(--host 0.0.0.0 --port 7863 --gpu-reserve-gib 2)
[[ -n "${YUE_VRAM:-}" ]] && ARGS+=(--vram "$YUE_VRAM")
[[ "${YUE_FP8:-0}" == "1" ]] && ARGS+=(--quantization fp8)
[[ -n "${YUE_EXTRA_ARGS:-}" ]] && read -ra EXTRA <<< "$YUE_EXTRA_ARGS" && ARGS+=("${EXTRA[@]}")

echo "[yue2] python webui.py ${ARGS[*]}"
exec python webui.py "${ARGS[@]}" "$@"
