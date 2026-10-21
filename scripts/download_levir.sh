#!/usr/bin/env bash
# Download LEVIR-CD test + val (original 1024x1024 PNG, ~0.75 GB) into data/raw, verify, extract, delete zips.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"; mkdir -p "$RAW/LEVIR-CD"
URL=https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main
for s in test val; do
  if [ -d "$RAW/LEVIR-CD/$s/A" ]; then echo "$s already present"; continue; fi
  curl -L --fail -sS -o "$RAW/$s.zip" "$URL/$s.zip"
  sha256sum "$RAW/$s.zip" | sed "s#$RAW/##" >> "$RAW/SHA256SUMS.txt"
  "$ROOT/nullshift-venv/bin/python" -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); print(len(z.namelist()),'entries'); print('\n'.join(z.namelist()[:6]))" "$RAW/$s.zip"
  "$ROOT/nullshift-venv/bin/python" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$RAW/$s.zip" "$RAW/LEVIR-CD/_extract_$s"
  rm -f "$RAW/$s.zip"
done
"$ROOT/scripts/storage_check.sh" || true
