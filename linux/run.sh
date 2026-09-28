#!/usr/bin/env bash
# Start YuE 2 – Music Gen Studio in a container.
#
#   ./run.sh                  foreground: log in this window, Ctrl+C / close = stop
#   ./run.sh --background     start detached, open the browser, return
#   ./run.sh --no-open        don't open the browser
#   ./stop.sh                 stop it
#
# Needs Docker + NVIDIA Container Toolkit (see install-docker.sh). Builds the
# image on first run. Stops any other ai.tool container first: one tool at a
# time on an 8 GB GPU.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="yue2"; PORT=7863; URL="http://localhost:$PORT"

BG=0; OPEN=1; SERVICES=("$TOOL"); PROFILE=()
for a in "$@"; do
  case $a in
    --background|-d) BG=1 ;;
    --no-open) OPEN=0 ;;
    --with-chatterbox) [[ $TOOL == qwen-tts ]] || { echo "only for qwen-tts" >&2; exit 1; }
                       SERVICES+=(chatterbox); PROFILE=(--profile chatterbox) ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done

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
COMPOSE=("${DK[@]}" compose -f "$HERE/compose.yml" "${PROFILE[@]}")

# One tool at a time.
others=$("${DK[@]}" ps --filter label=ai.tool --format '{{.Label "ai.tool"}} {{.Names}}' | grep -v "^$TOOL " | awk '{print $2}')
if [[ -n $others ]]; then
  echo "==> 8 GB VRAM: stopping $(tr '\n' ' ' <<< "$others")first"
  # shellcheck disable=SC2086
  "${DK[@]}" stop $others >/dev/null
fi

"${DK[@]}" image inspect "ai/$TOOL:latest" >/dev/null 2>&1 || { echo "==> first run: building the image (this takes a while)"; "${COMPOSE[@]}" build || exit 1; }

open_when_ready(){
  (( OPEN )) || return; command -v xdg-open >/dev/null || return
  ( for _ in $(seq 1 150); do curl -sf -o /dev/null --max-time 2 "$URL/" && { xdg-open "$URL" >/dev/null 2>&1; exit; }; sleep 2; done ) & disown
}

echo; echo "    YuE 2 – Music Gen Studio"; echo "    $URL"; echo
if (( BG )); then
  "${COMPOSE[@]}" up -d "${SERVICES[@]}" || exit 1
  echo "    running in the background; browser opens when ready.  stop: ./stop.sh"
  open_when_ready
else
  echo "    Ctrl+C or close this window to stop."
  echo "-----------------------------------------------------------------"
  trap '"${COMPOSE[@]}" stop >/dev/null 2>&1' EXIT HUP
  open_when_ready
  "${COMPOSE[@]}" up --no-log-prefix "${SERVICES[@]}"
fi
