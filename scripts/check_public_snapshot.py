"""Read-only review of the explicit candidate public source snapshot (stdlib only)."""

import argparse
import ast
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = "publication/source_allowlist.txt"
PRIVATE_PARTS = {"data", "outputs", ".venv", ".git", "venv", "mlruns", "mlartifacts",
                 "__pycache__", ".pytest_cache"}
EXTENSIONS = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".ini"}
SECRETS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    re.compile(r"(?i)(?:password|api_key|access_token|client_secret)\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']"),
    re.compile(r"[a-zA-Z]+://[^/\s:@]+:[^/\s@]+@"),
)


def read_allowlist(root=ROOT):
    root = Path(root).resolve()
    rows = [line.strip() for line in (root / ALLOWLIST).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("The public allowlist must be nonempty and unique")
    return rows


def public_path(relative, root=ROOT):
    root = Path(root).resolve()
    path = PurePosixPath(relative)
    if (path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative
            or any(part in PRIVATE_PARTS or part.startswith(".venv-") for part in path.parts)
            or any(part in {".env", "kaggle.json", "private"} for part in path.parts)
            or path.name.startswith(".env.") or path.name.endswith((".local.json", ".local.xml", ".local.log"))
            or relative in {"tests/fixtures/session22_golden.json", "practice.ipynb"}
            or (path.suffix not in EXTENSIONS and relative != ".gitignore")):
        raise ValueError("Excluded or unsafe public path")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Public file is missing or escapes the snapshot")
    # No symlinked path component may smuggle private material into a clean copy.
    if any(p.is_symlink() for p in [root / relative, *(root / relative).parents] if p != root):
        raise ValueError("Public files must not use symlinks")
    return resolved


def inspect_snapshot(root=ROOT):
    root = Path(root).resolve()
    rows = read_allowlist(root)
    problems, total = [], 0
    for relative in rows:
        try:
            path = public_path(relative, root)
            if path.stat().st_size > 1_000_000:
                raise ValueError("Oversized public source file")
            raw = path.read_bytes()
            total += len(raw)
            text = raw.decode("utf-8-sig")
            if "\x00" in text:
                raise ValueError("Binary content in public source")
            if any(pattern.search(text) for pattern in SECRETS):
                raise ValueError("Potential credential pattern; inspect privately")
            if path.suffix == ".py":
                ast.parse(text, filename=relative)
            if path.suffix == ".json":
                payload = json.loads(text)
                if isinstance(payload, dict) and "vectors" in payload:
                    raise ValueError("Golden-vector JSON needs explicit provenance review; excluded by policy")
        except (ValueError, OSError, SyntaxError) as error:
            # Do not print secret contents, parser excerpts or source lines.
            problems.append({"path": relative, "reason": type(error).__name__})
    if (root / ".git").exists():
        indexed = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
        for name in indexed.decode("utf-8").split("\x00"):
            if name and name not in rows:
                problems.append({"path": name, "reason": "indexed file outside allowlist"})
    return {"status": "FAIL" if problems else "PASS", "files": len(rows), "bytes": total,
            "problems": problems, "scope": "candidate working files; no Git staging or rights certification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = inspect_snapshot(args.root)
    print(json.dumps(report, indent=2))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
