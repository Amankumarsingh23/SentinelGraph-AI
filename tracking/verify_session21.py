"""Retrieve the exact registered runs and check every logged artifact's bytes."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tracking.common import MODEL_NAMES, TRACKING, verify_bytes, write_json

import numpy as np


def verify_manifest(manifest):
    if manifest["status"] != "FINISHED" or len(manifest["runs"]) != 4:
        raise ValueError("Manifest must describe four successfully registered runs")
    if {row["model"] for row in manifest["runs"]} != set(MODEL_NAMES):
        raise ValueError("Manifest run names do not match the four ablation groups")
    if len({row["run_id"] for row in manifest["runs"]}) != 4:
        raise ValueError("Manifest repeats a run ID")
    if manifest.get("test_access") is not False:
        raise ValueError("Unexpected test-access scope")


def main():
    from mlflow.tracking import MlflowClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=TRACKING / "experiment_manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    verify_manifest(manifest)
    client = MlflowClient(tracking_uri=manifest["tracking_uri"])
    experiment = client.get_experiment_by_name(manifest["experiment_name"])
    if experiment is None or experiment.experiment_id != manifest["experiment_id"]:
        raise ValueError("Registered experiment was not found at the recorded tracking URI")
    checks = []
    for row in manifest["runs"]:
        run = client.get_run(row["run_id"])
        if run.info.status != "FINISHED" or run.info.experiment_id != experiment.experiment_id:
            raise ValueError("Run is not a finished member of the expected experiment")
        if run.data.tags.get("mlflow.runName") != row["model"]:
            raise ValueError("Run name differs from the manifest")
        for key, value in row["tags"].items():
            if run.data.tags.get(key) != value:
                raise ValueError(f"Unexpected tag {key}")
        for key, value in row["parameters"].items():
            if run.data.params.get(key) != value:
                raise ValueError(f"Parameter mismatch: {key}")
        for key, value in row["metrics"].items():
            if key not in run.data.metrics:
                raise ValueError(f"Missing metric: {key}")
            np.testing.assert_allclose(run.data.metrics[key], value, rtol=1e-12, atol=1e-12)
        with tempfile.TemporaryDirectory(prefix="sentinelgraph_mlflow_verify_") as temporary:
            downloaded = Path(client.download_artifacts(row["run_id"], "", dst_path=temporary))
            for artifact in row["artifacts"]:
                target = (downloaded / artifact["artifact_path"]).resolve()
                if not target.is_relative_to(downloaded.resolve()):
                    raise ValueError("Artifact path escapes its download directory")
                verify_bytes(target, artifact)
        checks.append({"model": row["model"], "run_id": row["run_id"], "status": "PASS",
                       "verified_artifacts": len(row["artifacts"]), "average_precision": run.data.metrics["average_precision"]})
        print(f"{row['model']}: run, parameters, metrics and {len(row['artifacts'])} artifact hashes PASS", flush=True)
    report = {"status": "PASS", "registration_id": manifest["registration_id"], "runs": checks,
              "training_reexecuted": False, "test_access": False}
    write_json(args.manifest.parent / "verification.json", report)
    print("Four exact manifest run IDs verified; additional historical registrations are allowed.")


if __name__ == "__main__":
    main()
