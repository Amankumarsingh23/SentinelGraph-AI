"""Build a new immutable development release; never retrain or read reserved test."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.release import OUTPUT, build_release


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    path, manifest = build_release(parser.parse_args().output)
    print(f"Built development release {manifest['release_id']}: {path}")
    print(f"Frozen {len(manifest['artifacts'])} artifacts and {len(manifest['source_files'])} source/document files.")
    print("Matched earlier Session 21 model, predictions, config and input-file fingerprints.")
    print("Candidate requires verification; final-test approval/execution remain pending.")
