#!/usr/bin/env bash
# Music Gen Studio (YuE 2). Port 7863, not webui.py's default 7860, which
# would collide with Forge.
#   --gpu-reserve-gib 2  headroom on the 3070's 8 GB (matches its own default)
set -euo pipefail
cd /app
export PYTHONUTF8=1
# YuE's package lives in YuE/src; bind-mounted source means no editable install.
export PYTHONPATH="/app/YuE/src:${PYTHONPATH:-}"

ARGS=(--host 0.0.0.0 --port 7863 --gpu-reserve-gib 2)
[[ -n "${YUE_VRAM:-}" ]] && ARGS+=(--vram "$YUE_VRAM")
[[ "${YUE_FP8:-0}" == "1" ]] && ARGS+=(--quantization fp8)
[[ -n "${YUE_EXTRA_ARGS:-}" ]] && read -ra EXTRA <<< "$YUE_EXTRA_ARGS" && ARGS+=("${EXTRA[@]}")

echo "[yue2] python webui.py ${ARGS[*]}"
exec python webui.py "${ARGS[@]}" "$@"
