#!/usr/bin/env bash
# Fail the BUILD if a module the tool cannot run without is missing, so a broken
# image never reaches runtime. Usage: assert-imports.sh mod1 mod2 ...
set -euo pipefail
python - "$@" <<'PY'
import sys, importlib, importlib.metadata as md
missing=[]
for m in sys.argv[1:]:
    try:
        importlib.import_module(m)
        try: v=md.version(m)
        except Exception: v='?'
        print(f"  ok      {m} {v}")
    except Exception as e:
        print(f"  MISSING {m}: {type(e).__name__}: {e}")
        missing.append(m)
if missing:
    sys.exit("BUILD FAILED - required modules missing: " + ", ".join(missing))
print("  all required modules present")
PY
