"""Local tracking configuration and byte-level provenance helpers."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/evaluation/session20"
TRACKING = ROOT / "outputs/tracking"
EXPERIMENT = "SentinelGraph_Session20_Ablations"
MODEL_NAMES = ("AmountOnly", "AmountPlusActivity", "AmountPlusRelationships", "AllFive")

# This workflow is local; do not enable MLflow usage telemetry or autologging.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "true")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")


def tracking_uri(store_dir=TRACKING):
    return "sqlite:///" + (Path(store_dir).resolve() / "mlflow.db").as_posix()


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git_state(root=ROOT):
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    try:
        head, status = git("rev-parse", "HEAD"), git("status", "--porcelain=v1")
    except FileNotFoundError:
        return {"registration_commit": None, "dirty": None, "reason": "git unavailable",
                "training_commit": None}
    return {"registration_commit": head.stdout.strip() if head.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "status_porcelain": status.stdout,
            "reason": None if head.returncode == 0 else "No readable HEAD commit; this repository may have no commits",
            "training_commit": None,
            "revision_scope": "registration-time state only; original training revision was not captured"}


def artifact_record(source, artifact_path):
    source = Path(source)
    return {"artifact_path": artifact_path, "sha256": sha256_file(source), "bytes": source.stat().st_size}


def verify_bytes(path, expected):
    if not Path(path).is_file():
        raise ValueError(f"Missing artifact: {path}")
    if sha256_file(path) != expected["sha256"] or Path(path).stat().st_size != expected["bytes"]:
        raise ValueError(f"Artifact bytes differ: {path}")
