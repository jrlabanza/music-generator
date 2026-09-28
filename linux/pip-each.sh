#!/usr/bin/env bash
# Install packages ONE AT A TIME, best effort.
#
# pip's resolver is all-or-nothing: in a batch install, a single package that
# fails to build (pycairo without cairo headers) or whose metadata conflicts
# (gradio 4.40.0 wants pillow<11 while Forge pins 12.3.0) aborts every other
# package in the same command. That is how gradio went missing from the first
# build. Installing individually contains the damage to the offending package.
#
# Usage: pip-each.sh <constraints-file> <pkg> [pkg...]
set -uo pipefail
CONSTRAINTS="${1:?constraints file required}"; shift
ok=(); bad=()
for p in "$@"; do
  if pip install -c "$CONSTRAINTS" "$p" >/dev/null 2>&1; then ok+=("$p"); else bad+=("$p"); fi
done
echo "[pip-each] installed ${#ok[@]}/$(( ${#ok[@]} + ${#bad[@]} ))"
(( ${#bad[@]} )) && { echo "[pip-each] failed (continuing):"; printf '   %s\n' "${bad[@]}"; }
exit 0
