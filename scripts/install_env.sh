#!/usr/bin/env bash
# Reproducible environment install for NullShift. Everything stays inside the project folder.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export TMPDIR="$ROOT/.tmp"; mkdir -p "$TMPDIR"
PY="$ROOT/nullshift-venv/bin/python"
"$PY" -m pip install --no-cache-dir --upgrade pip
"$PY" -m pip install --no-cache-dir -r "$ROOT/requirements.txt"
# triton is only needed for torch.compile; we do not use it -> save ~0.6 GB
"$PY" -m pip uninstall -y triton || true
[ -n "$ROOT" ] && [ -d "$ROOT/.tmp" ] && find "$ROOT/.tmp" -mindepth 1 -maxdepth 1 -name "pip-*" -exec rm -rf {} +
