"""Download a timm encoder into the project HF cache, save an fp16 local copy in models/, delete the cache (storage).
Usage: PYTHONPATH=src nullshift-venv/bin/python scripts/fetch_encoder.py dinov3_vitb16_web"""
import shutil
import sys

from nullshift import paths
from nullshift.encoder import ENCODERS
import timm
from safetensors.torch import save_file

name = sys.argv[1]
arch = ENCODERS[name]
out = paths.MODELS / f"{arch}.fp16.safetensors"
if not out.exists():
    m = timm.create_model(arch, pretrained=True, num_classes=0).eval()
    save_file({k: (v.half() if v.is_floating_point() else v).contiguous() for k, v in m.state_dict().items()}, str(out))
shutil.rmtree(paths.MODELS / "hf_cache", ignore_errors=True)
print(out, round(out.stat().st_size / 1e6), "MB")
