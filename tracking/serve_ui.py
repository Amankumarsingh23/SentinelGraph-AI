"""Open a loopback-only MLflow UI against the same local SQLite store."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tracking.common import TRACKING, tracking_uri


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--store-dir", type=Path, default=TRACKING)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    if not (args.store_dir / "mlflow.db").is_file():
        parser.error("Tracking database is missing; register the Session 20 results first")
    subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", tracking_uri(args.store_dir),
                    "--host", "127.0.0.1", "--port", str(args.port), "--workers", "1"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
