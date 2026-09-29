#!/usr/bin/env bash
# One-time setup / repair for YuE 2 – Music Gen Studio on Linux - the counterpart of
# initialize.bat on Windows. Safe to re-run: finished steps are skipped and
# interrupted downloads resume.
#
#   ./initialize.sh               essentials (default models only), then a boot test
#   ./initialize.sh --all-models  every model the project knows about
#   ./initialize.sh --no-models   environment only
#   ./initialize.sh --repair      rebuild the image even if it exists
#   ./initialize.sh --no-test     skip the boot test
#   ./initialize.sh --no-desktop  skip the app-menu entry
#   ./initialize.sh --civitai-token KEY   (Forge: for Civitai downloads; or put it
#                                          in tools/civitai_token.txt / $CIVITAI_TOKEN)
#
# Steps: 1 GPU driver  2 Docker + NVIDIA toolkit  3 image  4 models (verified)
#        4b Seed-VC code  5 boot test  6 app-menu entry
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
TOOL="yue2"; SVC="yue2"
MODELS=essentials; REPAIR=0; TEST=1; DESKTOP=1; TOKEN="${CIVITAI_TOKEN:-}"
while (( $# )); do
  case $1 in
    --all-models) MODELS=all ;; --no-models) MODELS=none ;; --essentials) MODELS=essentials ;;
    --repair|--rebuild) REPAIR=1 ;; --no-test) TEST=0 ;; --no-desktop) DESKTOP=0 ;;
    --civitai-token) TOKEN="$2"; shift ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac; shift
done
say(){ printf '\n\033[1m[%s] %s\033[0m\n' "$1" "$2"; }
ok(){ printf '      %s\n' "$*"; }
bad(){ printf '      \033[31m%s\033[0m\n' "$*"; }
fail(){ bad "$*"; echo; echo "Setup did not complete. Fix the error above and run initialize.sh again."; exit 1; }

say 1/6 "NVIDIA driver"
if command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >/dev/null 2>&1; then
  ok "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
else
  fail "no working NVIDIA driver (nvidia-smi). Install the driver first: sudo ubuntu-drivers install"
fi

say 2/6 "Docker + NVIDIA Container Toolkit"
if ! command -v docker >/dev/null; then
  ok "not installed - installing (needs sudo)"
  sudo bash "$HERE/install-docker.sh" || fail "Docker install failed"
  ok "installed. NOTE: the docker group applies after you log out and back in; until then sudo is used."
fi
if docker info >/dev/null 2>&1; then DK=(docker); ok "docker OK (as $USER)"
elif sudo -n docker info >/dev/null 2>&1; then DK=(sudo -n --preserve-env=AI_UID,AI_GID,AI_CACHE,AI_BIND docker); ok "docker OK (via sudo until you re-login)"
else fail "cannot reach the docker daemon"; fi
"${DK[@]}" info 2>/dev/null | grep -q 'Runtimes:.*nvidia' || fail "NVIDIA runtime not registered in Docker: sudo bash $HERE/install-docker.sh"
export AI_UID="$(id -u)" AI_GID="$(id -g)" AI_CACHE="${AI_CACHE:-$HOME/.cache/ai-tools}"
mkdir -p "$AI_CACHE"/{hf,torch,xdg,mpl,numba,gradio}
COMPOSE=("${DK[@]}" compose -f "$HERE/compose.yml" --profile chatterbox)
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'); ok "disk free: ${free_gb} GB"

say 3/6 "Image ai/$TOOL"
if (( REPAIR )) || ! "${DK[@]}" image inspect "ai/$TOOL:latest" >/dev/null 2>&1; then
  ok "building (first time: several GB of downloads, 10-30 min)"
  "${COMPOSE[@]}" build || fail "image build failed"
else ok "present ($("${DK[@]}" image inspect "ai/$TOOL:latest" --format '{{.Size}}' | awk '{printf "%.1f GB", $1/1e9}'))"; fi
# run a python one-liner inside the tool's environment, repo mounted at /app
# via the entrypoint, so PYTHONPATH and the helper-env overrides are set
inpy(){ "${COMPOSE[@]}" run --rm --no-deps -T "$SVC" python "$@"; }

say 4/6 "Prerequisites ($MODELS)"
if [[ $MODELS == none ]]; then ok "skipped (--no-models)"; else
  # YuE2-3B + YuE2-Vae into models/ (download_models.py resumes and verifies).
  # --all-models adds the SheetSage2 cover-feature encoder (2.6 GB).
  extra=(); [[ $MODELS == all ]] && extra=(--sheetsage2 --legacy-vae)
  # Every weights file is checked against its weights_manifest.json sha256, like
  # initialize.py does on Windows - and a mismatch (a bad copy, an interrupted
  # download) is repaired here rather than reported: the file is removed and
  # fetched again.
  verify='
import json, sys
from pathlib import Path
from yue2.storage import model_identity
bad = []
for name in ("YuE2-3B", "YuE2-Vae"):
    folder = Path("/app/models") / name
    try:
        model_identity(folder)
    except FileNotFoundError:
        bad.append(name + ": missing")
    except ValueError as exc:
        manifest = json.loads((folder / "weights_manifest.json").read_text())
        for f in folder.glob("*.safetensors"):
            f.unlink()
            for meta in (folder / ".cache/huggingface/download").glob(f.name + ".metadata"):
                meta.unlink()
        bad.append(f"{name}: {exc} - removed, will download again")
print("\n".join(bad)); sys.exit(1 if bad else 0)
'
  for attempt in 1 2; do
    if [[ -d $ROOT/models/YuE2-3B && -d $ROOT/models/YuE2-Vae ]] && inpy -c "$verify" 2>/dev/null | sed 's/^/      /'; then
      [[ ${#extra[@]} -eq 0 ]] && { ok "YuE2-3B and YuE2-Vae present and verified"; break; }
    fi
    (( attempt == 1 )) || fail "the model weights still fail their sha256 check after a fresh download"
    inpy /app/download_models.py "${extra[@]}" || fail "model download failed"; ok "models downloaded"
  done
fi

say 4b/6 "Voice conversion engine (Seed-VC)"
# The README's "git clone https://github.com/Plachtaa/seed-vc tools\seed-vc": its
# Python environment is already in the image, only the code is needed here.
if [[ -f $ROOT/tools/seed-vc/inference.py ]]; then ok "tools/seed-vc present"
else
  mkdir -p "$ROOT/tools"
  git clone --depth 1 https://github.com/Plachtaa/seed-vc "$ROOT/tools/seed-vc" 2>&1 | tail -1 | sed 's/^/      /' \
    && ok "cloned into tools/seed-vc (checkpoints download on the first conversion)" \
    || bad "could not clone seed-vc (no network?) - Voices… stays disabled until you re-run this"
fi

say 5/6 "Boot test"
if (( TEST )); then "$HERE/test.sh" | sed 's/^/      /' || fail "boot test failed"; else ok "skipped (--no-test)"; fi

say 6/6 "App-menu entry"
if (( DESKTOP )) && command -v xdg-open >/dev/null && [[ -n ${DISPLAY:-}${WAYLAND_DISPLAY:-} ]]; then "$HERE/install-desktop.sh" | sed 's/^/      /'; else ok "skipped"; fi

echo; echo "Done. Start with: $HERE/run.sh   (or the \"AI – YuE 2 – Music Gen Studio\" app-menu entry)"
