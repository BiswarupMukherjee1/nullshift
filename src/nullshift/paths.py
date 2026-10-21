"""Project paths and cache redirection (everything must stay inside the project folder)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
LEVIR = DATA_RAW / "LEVIR-CD"
CREATED = ROOT / "Data_Created"
NSB = CREATED / "NSB-v1"
MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"
MLOPS = ROOT / "mlops"
TMP = ROOT / ".tmp"


def setup_env():
    """Redirect every library cache into the project. Call before importing timm / huggingface_hub / matplotlib."""
    env = {
        "HF_HOME": MODELS / "hf_cache",
        "TORCH_HOME": MODELS / "torch_cache",
        "MPLCONFIGDIR": TMP / "mpl",
        "XDG_CACHE_HOME": TMP / "xdg",
        "TMPDIR": TMP,
        "CUDA_CACHE_PATH": TMP / "cuda_cache",      # default would be ~/.nv (outside the project)
        "PYTHONPYCACHEPREFIX": TMP / "pycache",
    }
    for k, v in env.items():
        v.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(v)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")      # plain HTTPS; xet also keeps a duplicate chunk cache
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


setup_env()
