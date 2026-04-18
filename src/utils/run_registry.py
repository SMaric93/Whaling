"""
Lightweight run registry for VLM extraction and downstream pipeline stages.

Tracks what you need to reproduce a result across runs: model id, prompt hash,
preprocessor version, post-processor version, extraction config, and aggregate
metrics (event counts, confidence, fill rates). Writes append-only JSONL to
`data/derived/runs.jsonl` and a per-run detail file under `data/derived/runs/`.

If MLflow is installed and `MLFLOW_TRACKING_URI` is set, runs are also mirrored
to MLflow. Otherwise the JSONL registry is the single source of truth — no
extra infrastructure required.

Usage:
    from src.utils.run_registry import log_extraction_run

    log_extraction_run(
        stage="wsl_vlm_postprocess",
        config={"extractor_version": "v4.0", "model_id": "Qwen3-VL-32B"},
        metrics={"n_events": 272972, "mean_confidence": 0.96},
        artifacts={"events_parquet": "data/staging/wsl_events_v4.parquet"},
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

RUNS_DIR = PROJECT_ROOT / "data" / "derived" / "runs"
RUNS_REGISTRY = PROJECT_ROOT / "data" / "derived" / "runs.jsonl"


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
        return bool(out.decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def hash_config(config: Dict[str, Any]) -> str:
    """Deterministic hash of a config dict. Keys sorted, JSON-serialized."""
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _mlflow_log(run_id: str, stage: str, config: Dict, metrics: Dict, artifacts: Dict) -> None:
    """Mirror the run to MLflow if available and configured. Silent on absence."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
    except ImportError:
        return

    mlflow.set_experiment(stage)
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({k: str(v)[:500] for k, v in config.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                            if isinstance(v, (int, float))})
        for key, path in artifacts.items():
            p = Path(path)
            if p.exists() and p.stat().st_size < 50_000_000:
                mlflow.log_artifact(str(p), artifact_path=key)


def log_extraction_run(
    stage: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record a pipeline run.

    Args:
        stage: Logical stage name, e.g. "wsl_vlm_postprocess", "stage2_clean".
        config: Hyperparameters / version info. Hashed for reproducibility.
        metrics: Aggregate metrics (counts, means, rates). Numeric preferred.
        artifacts: {name: path} pointers to output files.
        tags: Free-form tags (git branch, host, user, etc.).
        run_id: Optional caller-supplied ID; otherwise generated.

    Returns:
        The run record that was written (also available as file on disk).
    """
    artifacts = artifacts or {}
    tags = tags or {}

    run_id = run_id or f"{stage}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"

    record = {
        "run_id": run_id,
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "config": config,
        "config_hash": hash_config(config),
        "metrics": metrics,
        "artifacts": artifacts,
        "tags": tags,
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = RUNS_DIR / f"{run_id}.json"
    with open(detail_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    RUNS_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_REGISTRY, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    logger.info("Logged run %s (stage=%s, config_hash=%s)",
                run_id, stage, record["config_hash"])

    _mlflow_log(run_id, stage, config, metrics, artifacts)

    return record


def list_runs(stage: Optional[str] = None, limit: int = 20) -> Iterable[Dict]:
    """Iterate run records (newest last) from the registry, optionally filtered."""
    if not RUNS_REGISTRY.exists():
        return []
    runs = []
    with open(RUNS_REGISTRY) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if stage and rec.get("stage") != stage:
                continue
            runs.append(rec)
    return runs[-limit:]
