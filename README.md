# NullShift — telling real change from instrument change, without retraining

Training-free operators that remove acquisition nuisance (sub-pixel misregistration, parallax, radiometry) from the
feature residual of a frozen satellite foundation model (DINOv3 ViT-L/16 SAT-493M) before the change decision,
evaluated on **null pairs** (image vs. physically perturbed copy of itself, zero true change by construction).

| Where | What |
|---|---|
| `PROPOSALS_28_09/` | original proposal + external critique |
| `docs/01_proposal_review.md` | citation verification + technical review + protocol decisions |
| `docs/02_work_plan.md` | plan, experiment registry, status |
| `docs/03_results_phase1.md` | Phase-1 results and interpretation |
| `docs/00_GUIDE.md` | plain-language guide (start here) |
| `docs/04_independent_review.md` | independent critical review of the work |
| `docs/05_novelty_assessment.md` | novelty assessment |
| `Data_Created/NSB-v1/` | the created null-pair benchmark (manifests, seeds, previews, subspaces, DATASHEET) |
| `Data_Created/MXR-LA-v1/`, `MXR-TX-v1/` | created real-pair benchmarks from Maxar Open Data (LA 2025, Texas 2024) |
| `src/nullshift/` | library: nuisances, encoder, operators, baselines, metrics, sweep, tracking |
| `experiments/E{NN}_*/` | one folder per experiment: `run.py`, `README.md`, `runs/<timestamp>_<tag>/` |
| `results/` | consolidated summary + headline figures |
| `mlops/mlflow.db` | MLflow tracking store (sqlite) |

## Reproduce
```bash
cd nullshift_cd
scripts/install_env.sh                        # venv nullshift-venv (CUDA torch, timm, mlflow)
scripts/download_levir.sh                     # LEVIR-CD test+val (~0.75 GB) -> data/raw
export PYTHONPATH=src; PY=nullshift-venv/bin/python
$PY -m pytest -q -p no:cacheprovider tests    # unit tests
$PY scripts/create_nsb.py                     # NSB-v1 benchmark (manifests/seeds/previews)
$PY scripts/fit_subspaces.py                  # global subspaces on val
$PY experiments/E00_sanity/run.py
$PY experiments/E01_linearity/run.py          # -> eps*
$PY experiments/E02_selectivity/run.py --eps <eps*>
$PY experiments/E03_nullpair_dose/run.py --eps <eps*>
$PY experiments/E04_perturbed_pair/run.py --eps <eps*> --e03-run experiments/E03_nullpair_dose/runs/<run>
nullshift-venv/bin/mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db   # browse runs
scripts/storage_check.sh                      # 8 GB target, 10 GB HARD CAP
```

## Data not included
Imagery is not redistributed: LEVIR-CD (academic licence, Google Earth terms) and Maxar Open Data (CC BY-NC 4.0).
Rebuild it with `scripts/download_levir.sh`, `scripts/create_nsb.py` and `scripts/create_maxar_event.py`; manifests and seeds are included.
