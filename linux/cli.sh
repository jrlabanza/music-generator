#!/usr/bin/env bash
# Run a command inside the YuE 2 environment, repo mounted at /app - the Linux
# form of the README's "YuE\.venv\Scripts\python.exe ..." lines:
#
#   ./linux/cli.sh python run_lowvram.py --request songs/my-song.json --output outputs/my-song
#   ./linux/cli.sh python download_models.py --sheetsage2 --legacy-vae
#   ./linux/cli.sh voice python export_audio.py --song outputs/x/audio.flac --output outputs/x --format mp3 --lufs -14
#   ./linux/cli.sh sheetsage python sheetsage_transcribe.py uploads/song.mp3 --output transcriptions/song
#   ./linux/cli.sh bash                     # a shell in the environment
#
# "voice" / "sheetsage" pick the helper environment (README: .venv-voice /
# .venv-sheetsage2); plain "python" is the YuE2 one. Runs alongside the web
# app if it is up (they share the GPU, so expect a slower song).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AI_UID="$(id -u)" AI_GID="$(id -g)"
export AI_CACHE="${AI_CACHE:-$HOME/.cache/ai-tools}"
mkdir -p "$AI_CACHE"/{hf,torch,xdg,mpl,numba,gradio}
if docker info >/dev/null 2>&1; then DK=(docker)
elif sudo -n docker info >/dev/null 2>&1; then DK=(sudo -n --preserve-env=AI_UID,AI_GID,AI_CACHE,AI_BIND docker)
else echo "cannot reach the docker daemon (installed? in the docker group? logged out and in?)" >&2; exit 1; fi
[[ $# -gt 0 ]] || { sed -n '2,13p' "$0" | sed 's/^# \?//'; exit 1; }
case ${1:-} in
  voice)     shift; set -- env PATH="/opt/venv-voice/bin:$PATH" "$@" ;;
  sheetsage) shift; set -- env PATH="/opt/venv-sheetsage2/bin:$PATH" "$@" ;;
esac
tty=(); [[ -t 0 && -t 1 ]] && tty=(-it) || tty=(-T)
exec "${DK[@]}" compose -f "$HERE/compose.yml" run --rm --no-deps "${tty[@]}" --name "ai-yue2-cli-$$" yue2 "$@"
