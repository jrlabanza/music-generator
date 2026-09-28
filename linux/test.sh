#!/usr/bin/env bash
# Verify the yue2 image: right Python/torch, GPU visible, key imports;
# then actually boot it and wait for HTTP 200. Exit 0 = all good.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="yue2"; PORT=7863; WANT_PY="3.12"; WANT_TORCH="2.10.0"; MODS="torch transformers accelerate soundfile tiktoken"
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
COMPOSE=("${DK[@]}" compose -f "$HERE/compose.yml" --profile chatterbox)
fail=0

echo "=== 1/2 environment ==="
out=$("${DK[@]}" run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
      --entrypoint /opt/venv/bin/python "ai/$TOOL:latest" -c "
import sys, importlib, importlib.metadata as md, torch
print('python', '.'.join(map(str, sys.version_info[:3])))
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available(): print('gpu', torch.cuda.get_device_name(0))
for m in '$MODS'.split():
    try: importlib.import_module(m); print('ok', m)
    except Exception as e: print('IMPORTFAIL', m, e)
" 2>&1); echo "$out" | sed 's/^/  /'
grep -q "^python $WANT_PY\." <<< "$out" || { echo "  >> WRONG PYTHON"; fail=1; }
grep -q "^torch $WANT_TORCH" <<< "$out"  || { echo "  >> WRONG TORCH"; fail=1; }
grep -q "^cuda_available True" <<< "$out" || { echo "  >> CUDA NOT AVAILABLE"; fail=1; }
grep -q IMPORTFAIL <<< "$out" && fail=1

echo "=== 2/2 boot ==="
"${COMPOSE[@]}" stop >/dev/null 2>&1
"${COMPOSE[@]}" up -d "$TOOL" >/dev/null 2>&1 || { echo "  container did not start"; exit 1; }
t0=$(date +%s); code=""
while (( $(date +%s) - t0 < 300 )); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/" 2>/dev/null)
  [[ $code == 200 ]] && break; sleep 5
done
logs=$("${COMPOSE[@]}" logs --no-color --no-log-prefix "$TOOL" 2>/dev/null)
errs=$(grep -E 'Traceback \(most recent call last\)|ModuleNotFoundError|ImportError|CUDA error|out of memory|Address already in use' <<< "$logs" | head -5)
if [[ $code == 200 && -z $errs ]]; then echo "  UP in $(( $(date +%s) - t0 ))s, no errors";
else echo "  FAILED (http=$code)"; [[ -n $errs ]] && sed 's/^/  ! /' <<< "$errs"; echo "$logs" | tail -15 | sed 's/^/  | /'; fail=1; fi
"${COMPOSE[@]}" stop >/dev/null 2>&1
echo; (( fail )) && echo "RESULT: FAIL" || echo "RESULT: PASS"
exit $fail
