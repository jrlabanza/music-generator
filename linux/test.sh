#!/usr/bin/env bash
# Verify the yue2 image: right Python/torch, GPU visible, key imports;
# then actually boot it and wait for HTTP 200. Exit 0 = all good.
# Tests the CUDA image (ai/yue2:latest) or, when .gpu.json says rocm or
# AI_GPU=rocm, the ROCm one (ai/yue2:rocm; torch must be the HIP build).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="yue2"; PORT=7863; WANT_PY="3.12"; WANT_TORCH="2.10.0"; MODS="torch transformers accelerate soundfile tiktoken"
# GPU backend (AI Studio Hub contract, docs/gpu.md): .gpu.json says cuda or rocm,
# AI_GPU=rocm|cuda overrides. Anything else (no file yet) is the CUDA image.
GPU="${AI_GPU:-$(sed -n 's/.*"backend": *"\([a-z]*\)".*/\1/p' "$HERE/../.gpu.json" 2>/dev/null)}"
if [[ $GPU == rocm ]]; then CF="$HERE/compose.rocm.yml"; TAG=rocm; else CF="$HERE/compose.yml"; TAG=latest; fi
export AI_VIDEO_GID="$(getent group video | cut -d: -f3)" AI_RENDER_GID="$(getent group render | cut -d: -f3)"
: "${AI_VIDEO_GID:=44}" "${AI_RENDER_GID:=992}"
# Docker access: docker group, or passwordless sudo until the group is active
# (Ubuntu 26.04 has no newgrp/sg - a new group needs a fresh login). sudo resets
# HOME and drops the environment, so the variables compose needs are passed
# through explicitly - otherwise AI_CACHE would resolve to /root and Docker
# would create it root-owned, unwritable by the non-root container.
export AI_UID="$(id -u)" AI_GID="$(id -g)"
export AI_CACHE="${AI_CACHE:-$HOME/.cache/ai-tools}"
mkdir -p "$AI_CACHE"/{hf,torch,xdg,mpl,numba,gradio}
if docker info >/dev/null 2>&1; then DK=(docker)
elif sudo -n docker info >/dev/null 2>&1; then DK=(sudo -n --preserve-env=AI_UID,AI_GID,AI_VIDEO_GID,AI_RENDER_GID,AI_CACHE,AI_BIND docker)
else echo "cannot reach the docker daemon (installed? in the docker group? logged out and in?)" >&2; exit 1; fi
COMPOSE=("${DK[@]}" compose -f "$CF" --profile chatterbox)
fail=0

# How a bare `docker run` reaches the GPU: the NVIDIA runtime, or on AMD the
# kfd/dri device nodes plus the video/render groups (only when the host has them).
if [[ $TAG == rocm ]]; then
  GPUARGS=(--group-add "$AI_VIDEO_GID" --group-add "$AI_RENDER_GID" --security-opt seccomp=unconfined)
  [[ -e /dev/kfd ]] && GPUARGS+=(--device /dev/kfd); [[ -e /dev/dri ]] && GPUARGS+=(--device /dev/dri)
  # compose reads HSA_OVERRIDE_GFX_VERSION from linux/.env by itself; a bare docker run needs it passed.
  [[ -z ${HSA_OVERRIDE_GFX_VERSION:-} && -f $HERE/.env ]] && HSA_OVERRIDE_GFX_VERSION=$(sed -n 's/^HSA_OVERRIDE_GFX_VERSION=//p' "$HERE/.env" | tail -1)
  [[ -n ${HSA_OVERRIDE_GFX_VERSION:-} ]] && GPUARGS+=(-e "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION")
else
  GPUARGS=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all)
fi

echo "=== 1/2 environment ($TAG) ==="
out=$("${DK[@]}" run --rm "${GPUARGS[@]}" \
      --entrypoint /opt/venv/bin/python "ai/$TOOL:$TAG" -c "
import sys, importlib, importlib.metadata as md, torch
print('python', '.'.join(map(str, sys.version_info[:3])))
print('torch', torch.__version__)
print('hip', torch.version.hip or '-')
print('gpu_available', torch.cuda.is_available())
if torch.cuda.is_available(): print('gpu', torch.cuda.get_device_name(0))
for m in '$MODS'.split():
    try: importlib.import_module(m); print('ok', m)
    except Exception as e: print('IMPORTFAIL', m, e)
" 2>&1); echo "$out" | sed 's/^/  /'
grep -q "^python $WANT_PY\." <<< "$out" || { echo "  >> WRONG PYTHON"; fail=1; }
grep -q "^torch $WANT_TORCH" <<< "$out"  || { echo "  >> WRONG TORCH"; fail=1; }
if [[ $TAG == rocm ]]; then grep -qE "^hip [0-9]" <<< "$out" || { echo "  >> NOT A ROCM TORCH (torch.version.hip unset)"; fail=1; }
else grep -q "^hip -$" <<< "$out" || { echo "  >> ROCM TORCH IN THE CUDA IMAGE"; fail=1; }; fi
grep -q "^gpu_available True" <<< "$out" || { echo "  >> GPU NOT AVAILABLE"; fail=1; }
grep -q IMPORTFAIL <<< "$out" && fail=1

echo "=== 1b/2 helper environments (voice + SheetSage2) ==="
out=$("${DK[@]}" run --rm --entrypoint bash "ai/$TOOL:$TAG" -c '
/opt/venv-voice/bin/python -c "import torch, torchaudio, demucs, lameenc, pyloudnorm, mutagen, transformers; print(\"voice ok torch\", torch.__version__)" 2>&1 || echo IMPORTFAIL voice
/opt/venv-sheetsage2/bin/python -c "import torch, transformers, mir_eval, pretty_midi; print(\"sheetsage ok torch\", torch.__version__)" 2>&1 || echo IMPORTFAIL sheetsage
' 2>&1); echo "$out" | sed 's/^/  /'
grep -q "^voice ok torch 2.8.0" <<< "$out"     || { echo "  >> VOICE ENV BROKEN"; fail=1; }
grep -q "^sheetsage ok torch 2.8.0" <<< "$out" || { echo "  >> SHEETSAGE ENV BROKEN"; fail=1; }
grep -q IMPORTFAIL <<< "$out" && fail=1

echo "=== 2/2 boot ==="
"${COMPOSE[@]}" stop >/dev/null 2>&1
# --force-recreate: a reused container keeps its old log, and the previous
# run's shutdown noise (e.g. "cannot schedule new futures after interpreter
# shutdown") would be scanned as if it were this boot's error.
"${COMPOSE[@]}" up -d --force-recreate "$TOOL" >/dev/null 2>&1 || { echo "  container did not start"; exit 1; }
t0=$(date +%s); code=""
while (( $(date +%s) - t0 < 300 )); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/" 2>/dev/null)
  [[ $code == 200 ]] && break; sleep 5
done
logs=$("${COMPOSE[@]}" logs --no-color --no-log-prefix "$TOOL" 2>/dev/null)
errs=$(grep -E 'Traceback \(most recent call last\)|ModuleNotFoundError|ImportError|CUDA error|HIP error|out of memory|Address already in use' <<< "$logs" | head -5)
if [[ $code == 200 && -z $errs ]]; then echo "  UP in $(( $(date +%s) - t0 ))s, no errors";
else echo "  FAILED (http=$code)"; [[ -n $errs ]] && sed 's/^/  ! /' <<< "$errs"; echo "$logs" | tail -15 | sed 's/^/  | /'; fail=1; fi
"${COMPOSE[@]}" stop >/dev/null 2>&1
echo; (( fail )) && echo "RESULT: FAIL" || echo "RESULT: PASS"
exit $fail
