#!/usr/bin/env bash
# Best-effort install of every package that was present in the working Windows
# venv but not covered by the curated install above. Maximises fidelity for
# the long tail (43 Forge extensions, 8 ComfyUI custom node packs) without
# letting one Linux-unavailable package fail the whole build.
#
# Usage: install-pins.sh <pins-file> <torch-lock-file>
#
# Both files are passed to pip as constraints, so nothing here can upgrade or
# downgrade torch or drag a dependency off its pinned version.
set -uo pipefail
PINS="${1:?pins file required}"
LOCK="${2:?torch lock file required}"

CONSTRAIN=(-c "$PINS" -c "$LOCK")

echo "[pins] attempting single resolve of $(grep -cvE '^\s*(#|$)' "$PINS") packages"
if pip install "${CONSTRAIN[@]}" -r "$PINS"; then
  echo "[pins] batch install succeeded"
else
  echo "[pins] batch resolve failed - falling back to per-package best effort"
  fails=()
  while read -r spec; do
    [[ -z $spec || $spec == \#* ]] && continue
    pip install "${CONSTRAIN[@]}" "$spec" >/dev/null 2>&1 || fails+=("$spec")
  done < <(grep -vE '^\s*(#|$)' "$PINS")
  if (( ${#fails[@]} )); then
    echo "[pins] ${#fails[@]} package(s) unavailable on Linux (expected; skipped):"
    printf '   %s\n' "${fails[@]}"
  else
    echo "[pins] all packages installed individually"
  fi
fi

# Guard: the curated step above installed a very specific torch build. If
# anything here moved it, fail the build now rather than at runtime.
echo "[pins] verifying torch was not disturbed"
python - "$LOCK" <<'PY'
import sys, re, importlib.metadata as md
lock = sys.argv[1]
bad = []
for line in open(lock):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name, want = line.split("==", 1)
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        continue
    # Installed version carries a +cuXXX local tag; compare the base only.
    if got.split("+")[0] != want.split("+")[0]:
        bad.append(f"{name}: expected {want}, found {got}")
if bad:
    sys.exit("[pins] FATAL - torch stack was modified:\n  " + "\n  ".join(bad))
print("[pins] torch stack intact")
PY
