"""Extrai o JavaScript inline do frontend.html e valida a sintaxe com Node."""
import re
import subprocess
import sys
from pathlib import Path

html = Path("frontend/frontend.html").read_text(encoding="utf-8")

scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S | re.I)
print(f"Scripts inline encontrados: {len(scripts)}")

out = Path("frontend/_inline_script.js")
try:
    out.write_text("\n".join(scripts), encoding="utf-8")
    result = subprocess.run(
        ["node", "--check", str(out)],
        capture_output=True,
        text=True,
    )
finally:
    out.unlink(missing_ok=True)

if result.returncode == 0:
    print("OK: sintaxe JavaScript válida")
    sys.exit(0)
print("ERRO DE SINTAXE JS:")
print(result.stderr)
sys.exit(1)
