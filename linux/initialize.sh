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
#   ./initialize.sh --gpu nvidia|amd   override the card detection (AI Studio Hub
#                                 GPU contract, docs/gpu.md); --gfx gfx1100 the AMD target
#   ./initialize.sh --civitai-token KEY   (Forge: for Civitai downloads; or put it
#                                          in tools/civitai_token.txt / $CIVITAI_TOKEN)
#
# Steps: 1 GPU driver  2 Docker (+ NVIDIA toolkit on NVIDIA)  3 image  4 models (verified)
#        4b Seed-VC code  5 boot test  6 app-menu entry
# NVIDIA builds ai/yue2:latest (compose.yml, CUDA); AMD builds ai/yue2:rocm
# (compose.rocm.yml, ROCm) and writes .gpu.json so run.sh/test.sh pick it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
TOOL="yue2"; SVC="yue2"
MODELS=essentials; REPAIR=0; TEST=1; DESKTOP=1; TOKEN="${CIVITAI_TOKEN:-}"; FORCE_GPU=""; FORCE_GFX=""
while (( $# )); do
  case $1 in
    --all-models) MODELS=all ;; --no-models) MODELS=none ;; --essentials) MODELS=essentials ;;
    --repair|--rebuild) REPAIR=1 ;; --no-test) TEST=0 ;; --no-desktop) DESKTOP=0 ;;
    --civitai-token) TOKEN="$2"; shift ;;
    --gpu) FORCE_GPU="${2,,}"; shift ;; --gpu=*) FORCE_GPU="${1#*=}"; FORCE_GPU="${FORCE_GPU,,}" ;;
    --gfx) FORCE_GFX="$2"; shift ;; --gfx=*) FORCE_GFX="${1#*=}" ;;
    -h|--help) sed -n '2,19p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac; shift
done
say(){ printf '\n\033[1m[%s] %s\033[0m\n' "$1" "$2"; }
ok(){ printf '      %s\n' "$*"; }
bad(){ printf '      \033[31m%s\033[0m\n' "$*"; }
fail(){ bad "$*"; echo; echo "Setup did not complete. Fix the error above and run initialize.sh again."; exit 1; }
case $FORCE_GPU in ""|nvidia|amd) ;; cpu) fail "--gpu cpu: the Linux containers need a GPU (NVIDIA or AMD); there is no CPU image" ;;
  *) fail "--gpu must be nvidia or amd, not '$FORCE_GPU'" ;; esac

say 1/6 "GPU driver"
# Detection order from the shared contract: nvidia-smi answers -> NVIDIA; else an
# AMD card (rocm-smi/amd-smi, or an amdgpu device in sysfs) -> AMD; --gpu overrides.
is_amd(){ command -v rocm-smi >/dev/null || command -v amd-smi >/dev/null \
          || grep -qis '^0x1002$' /sys/class/drm/card*/device/vendor 2>/dev/null; }
VENDOR=""
if [[ $FORCE_GPU != amd ]] && command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >/dev/null 2>&1; then
  VENDOR=nvidia; ok "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
elif [[ $FORCE_GPU == amd ]] || { [[ $FORCE_GPU != nvidia ]] && is_amd; }; then
  VENDOR=amd
  # The amdgpu driver is in the Ubuntu kernel; nothing ROCm is installed on the host.
  # The container needs the compute node (/dev/kfd) and the render nodes (/dev/dri).
  name=$(lspci -d 1002: 2>/dev/null | grep -iE 'vga|display|3d' | head -1 | sed 's/.*: //')
  ok "AMD Radeon${name:+: $name}${FORCE_GPU:+ (forced with --gpu amd)}"
  [[ -e /dev/kfd ]] || fail "/dev/kfd is missing: the amdgpu kernel driver is not loaded (try: sudo modprobe amdgpu; the card must be in AMD's ROCm Linux matrix, or set HSA_OVERRIDE_GFX_VERSION in linux/.env)"
  [[ -d /dev/dri ]] || fail "/dev/dri is missing: no DRM render node (amdgpu driver not loaded?)"
  ok "/dev/kfd and /dev/dri present"
else
  fail "no working NVIDIA driver (nvidia-smi) and no AMD card found. NVIDIA: sudo ubuntu-drivers install; AMD: the amdgpu driver must be loaded (/dev/kfd)"
fi
if [[ $VENDOR == nvidia ]]; then CF="$HERE/compose.yml"; TAG=latest; else CF="$HERE/compose.rocm.yml"; TAG=rocm; fi
IMAGE="ai/$TOOL:$TAG"

if [[ $VENDOR == nvidia ]]; then say 2/6 "Docker + NVIDIA Container Toolkit"; else say 2/6 "Docker (AMD: no container runtime needed)"; fi
if ! command -v docker >/dev/null; then
  ok "not installed - installing (needs sudo)"
  if [[ $VENDOR == nvidia ]]; then sudo bash "$HERE/install-docker.sh" || fail "Docker install failed"
  else sudo bash "$HERE/install-docker.sh" --no-nvidia || fail "Docker install failed"; fi
  ok "installed. NOTE: the docker group applies after you log out and back in; until then sudo is used."
fi
# ROCm containers join the host's video/render groups to reach /dev/kfd and /dev/dri.
export AI_VIDEO_GID="$(getent group video | cut -d: -f3)" AI_RENDER_GID="$(getent group render | cut -d: -f3)"
: "${AI_VIDEO_GID:=44}" "${AI_RENDER_GID:=992}"
if [[ $VENDOR == amd ]]; then
  missing=(); for g in video render; do id -nG | tr ' ' '\n' | grep -qx "$g" || missing+=("$g"); done
  if (( ${#missing[@]} )); then
    sudo usermod -aG "$(IFS=,; echo "${missing[*]}")" "$USER" && ok "added $USER to the ${missing[*]} group(s) (applies after you log out and back in; the container is unaffected)" \
      || bad "could not add $USER to the ${missing[*]} group(s); the container still reaches the card through group_add"
  else ok "$USER is in the video and render groups"; fi
  ok "video gid $AI_VIDEO_GID, render gid $AI_RENDER_GID"
fi
if docker info >/dev/null 2>&1; then DK=(docker); ok "docker OK (as $USER)"
elif sudo -n docker info >/dev/null 2>&1; then DK=(sudo -n --preserve-env=AI_UID,AI_GID,AI_VIDEO_GID,AI_RENDER_GID,AI_CACHE,AI_BIND docker); ok "docker OK (via sudo until you re-login)"
else fail "cannot reach the docker daemon"; fi
if [[ $VENDOR == nvidia ]]; then
  "${DK[@]}" info 2>/dev/null | grep -q 'Runtimes:.*nvidia' || fail "NVIDIA runtime not registered in Docker: sudo bash $HERE/install-docker.sh"
fi
export AI_UID="$(id -u)" AI_GID="$(id -g)" AI_CACHE="${AI_CACHE:-$HOME/.cache/ai-tools}"
mkdir -p "$AI_CACHE"/{hf,torch,xdg,mpl,numba,gradio}
COMPOSE=("${DK[@]}" compose -f "$CF" --profile chatterbox)
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'); ok "disk free: ${free_gb} GB"

say 3/6 "Image $IMAGE"
if (( REPAIR )) || ! "${DK[@]}" image inspect "$IMAGE" >/dev/null 2>&1; then
  ok "building (first time: several GB of downloads, 10-30 min)"
  "${COMPOSE[@]}" build || fail "image build failed"
else ok "present ($("${DK[@]}" image inspect "$IMAGE" --format '{{.Size}}' | awk '{printf "%.1f GB", $1/1e9}'))"; fi
# run a python one-liner inside the tool's environment, repo mounted at /app
# via the entrypoint, so PYTHONPATH and the helper-env overrides are set
inpy(){ "${COMPOSE[@]}" run --rm --no-deps -T "$SVC" python "$@"; }
# Record the card and the torch build the image carries in .gpu.json (the hub's
# shared GPU contract, docs/gpu.md): the check-up page and the app read it.
DETECT=(/app/tools/gpu_detect.py)
if [[ $VENDOR == amd ]]; then
  # The gfx target comes from rocminfo inside the container (the image carries
  # ROCm's tools, the host does not). Tolerated when it fails: .gpu.json then
  # says "gfx": "" and the app reads the target from the device at runtime.
  GFX="$FORCE_GFX"
  if [[ -z $GFX ]]; then
    GFX=$("${COMPOSE[@]}" run --rm --no-deps -T --entrypoint /opt/rocm/bin/rocminfo "$SVC" 2>/dev/null \
          | grep -oE '\bgfx[0-9a-f]+\b' | head -1 || true)
    [[ -n $GFX ]] && ok "rocminfo: $GFX" || bad "rocminfo could not read the gfx target (the card may be outside AMD's matrix; HSA_OVERRIDE_GFX_VERSION in linux/.env may help)"
  else ok "gfx target forced: $GFX"; fi
  DETECT+=(--gpu amd); [[ -n $GFX ]] && DETECT+=(--gfx "$GFX")
  # Cards outside AMD's Linux matrix run with a compatible target's code:
  # RX 6700/6650/6600 (gfx1031/1032) as gfx1030, the 780M/760M iGPU (gfx1103) as gfx1100.
  HSA=""; case $GFX in gfx1031|gfx1032) HSA=10.3.0 ;; gfx1103) HSA=11.0.0 ;; esac
  if [[ -n $HSA ]]; then
    touch "$HERE/.env"
    if grep -q '^HSA_OVERRIDE_GFX_VERSION=' "$HERE/.env"; then sed -i "s/^HSA_OVERRIDE_GFX_VERSION=.*/HSA_OVERRIDE_GFX_VERSION=$HSA/" "$HERE/.env"
    else printf 'HSA_OVERRIDE_GFX_VERSION=%s\n' "$HSA" >> "$HERE/.env"; fi
    export HSA_OVERRIDE_GFX_VERSION="$HSA"; ok "$GFX is outside AMD's Linux matrix: HSA_OVERRIDE_GFX_VERSION=$HSA written to linux/.env"
  fi
fi
if inpy "${DETECT[@]}" --write /app/.gpu.json >/dev/null 2>&1 && [[ -f $ROOT/.gpu.json ]]; then
  ok ".gpu.json: $(tr -d '\n' < "$ROOT/.gpu.json" | cut -c1-110)"
else bad "could not write .gpu.json (the app falls back to runtime detection)"; fi

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
# AI_GPU pins test.sh to the image built above even if .gpu.json could not be written.
if (( TEST )); then AI_GPU=$([[ $VENDOR == amd ]] && echo rocm || echo cuda) "$HERE/test.sh" | sed 's/^/      /' || fail "boot test failed"; else ok "skipped (--no-test)"; fi

say 6/6 "App-menu entry"
if (( DESKTOP )) && command -v xdg-open >/dev/null && [[ -n ${DISPLAY:-}${WAYLAND_DISPLAY:-} ]]; then "$HERE/install-desktop.sh" | sed 's/^/      /'; else ok "skipped"; fi

echo; echo "Done. Start with: $HERE/run.sh   (or the \"AI – YuE 2 – Music Gen Studio\" app-menu entry)"
