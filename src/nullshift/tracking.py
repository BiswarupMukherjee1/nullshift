"""Run bookkeeping: every run writes a self-contained folder (config, metrics, logs, figures) and mirrors
params/metrics/figures to a local MLflow store (sqlite, inside the project). MLflow failures never kill a run."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from .paths import EXPERIMENTS, MLOPS, ROOT

TRACKING_URI = f"sqlite:///{MLOPS / 'mlflow.db'}"


def _git_rev():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


class Run:
    def __init__(self, experiment: str, tag: str, config: dict, resume_dir: str | None = None):
        self.exp = experiment
        if resume_dir:                      # continue an interrupted run in place (checkpointed sweeps)
            self.dir = Path(resume_dir).resolve()
            assert self.dir.exists(), f"resume dir missing: {self.dir}"
            config = dict(config, _resumed_at=time.strftime("%Y%m%d-%H%M"))
        else:
            stamp = time.strftime("%Y%m%d-%H%M")
            self.dir = EXPERIMENTS / experiment / "runs" / f"{stamp}_{tag}"
            assert not self.dir.exists(), f"run dir exists, refusing to overwrite: {self.dir}"
            self.dir.mkdir(parents=True)
        self.config = dict(config, _git=_git_rev(), _python=sys.version.split()[0], _host=platform.node(), _tag=tag)
        (self.dir / ("config_resume.json" if resume_dir else "config.json")).write_text(
            json.dumps(self.config, indent=2, default=str))
        self.metrics: dict = {}
        self._log = open(self.dir / "log.txt", "a")
        self._t0 = time.time()
        self._ml = None
        try:
            import mlflow
            MLOPS.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(TRACKING_URI)
            if mlflow.get_experiment_by_name(experiment) is None:
                mlflow.create_experiment(experiment, artifact_location=(MLOPS / "artifacts" / experiment).as_uri())
            mlflow.set_experiment(experiment)
            self._ml = mlflow.start_run(run_name=self.dir.name)
            flat = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in self.config.items()}
            mlflow.log_params({k: str(v)[:500] for k, v in flat.items()})
        except Exception as e:  # pragma: no cover
            self.log(f"[mlflow disabled] {e!r}")

    def log(self, msg: str):
        line = f"[{time.time() - self._t0:8.1f}s] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def metric(self, key: str, value, step: int | None = None):
        self.metrics[key] = value
        if self._ml is not None and isinstance(value, (int, float)):
            try:
                import mlflow
                mlflow.log_metric(key.replace("@", "_at_").replace("+", "_plus_"), float(value), step=step)
            except Exception:
                pass

    def save_json(self, name: str, obj):
        (self.dir / name).write_text(json.dumps(obj, indent=2, default=float))

    def figure(self, fig, name: str):
        p = self.dir / name
        fig.savefig(p, dpi=130, bbox_inches="tight")
        if self._ml is not None:
            try:
                import mlflow
                mlflow.log_artifact(str(p))
            except Exception:
                pass
        return p

    def close(self, status: str = "FINISHED"):
        self.metrics["_wallclock_s"] = round(time.time() - self._t0, 1)
        self.save_json("metrics.json", self.metrics)
        self.log(f"run closed: {status}")
        self._log.close()
        if self._ml is not None:
            try:
                import mlflow
                mlflow.log_artifact(str(self.dir / "metrics.json"))
                mlflow.end_run(status=status)
            except Exception:
                pass
