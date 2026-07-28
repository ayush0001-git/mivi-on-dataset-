"""Pre-flight check — run this first on a clean checkout.

    python check_setup.py

Verifies, in plain English, that everything the prototype needs is in place
BEFORE you run answer.py, so a missing key or dependency is a clear message
here instead of a stack trace later. Exits 0 when ready, 1 when not.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ok = True


def check(label, passed, hint=""):
    global ok
    ok = ok and passed
    mark = "OK  " if passed else "MISS"
    line = f"[{mark}] {label}"
    if not passed and hint:
        line += f"\n        -> {hint}"
    print(line)


# 1. Python version
check(f"Python {sys.version_info.major}.{sys.version_info.minor} (need 3.10+)",
      sys.version_info >= (3, 10),
      "Install Python 3.10 or newer.")

# 2. Dependencies importable
for mod in ("sentence_transformers", "numpy", "openai", "dotenv"):
    check(f"package '{mod}' installed", importlib.util.find_spec(mod) is not None,
          "Run: pip install -r requirements.txt")

# 2b. Optional extras (demo UI / web mode) — informational, never fail setup
for mod, why in (("fastapi", "demo UI (uvicorn app:app)"),
                 ("uvicorn", "demo UI (uvicorn app:app)"),
                 ("requests", "optional web mode")):
    present = importlib.util.find_spec(mod) is not None
    print(f"[{'OK  ' if present else 'opt '}] optional package '{mod}' ({why})"
          + ("" if present else " — install only if you need it"))

# 3. Dataset present
check("data/sample_colleges.csv present", (ROOT / "data" / "sample_colleges.csv").exists(),
      "The dataset CSV is missing from data/.")

# 4. API key configured
key = ""
try:
    from dotenv import dotenv_values
    env = dotenv_values(ROOT / ".env")
    key = env.get("OPENAI_API_KEY") or env.get("GROQ_API_KEY") or ""
except Exception:
    pass
import os
key = key or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or ""
check("LLM API key set (.env or environment)", bool(key),
      "Copy .env.example to .env and set your key. See README 'Quickstart'.")

print()
if ok:
    print("All set. Try:  python answer.py \"Which colleges offer an MBA, and what do they cost?\"")
    sys.exit(0)
else:
    print("Not ready yet — fix the MISS lines above, then re-run: python check_setup.py")
    sys.exit(1)
