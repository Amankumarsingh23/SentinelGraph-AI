# Public snapshot policy

`source_allowlist.txt` lists exact repository-relative public candidate files,
one per line. It is a review manifest, not an automatic staging command.
No publication tool stages, commits or pushes files.

```powershell
python -B scripts/check_public_snapshot.py
```

The checker rejects private/fitted artifacts, unsafe paths, oversized/binary files,
credential-pattern matches and literal dataset golden vectors in JSON. It reports
paths/categories without exposing possible secret values. If Git metadata is
available, already-indexed paths must also belong to the allowlist. Scanning is
heuristic; new fixtures still need manual provenance review.

Keep data, outputs, environments, tracking stores, credentials and the original
`tests/fixtures/session22_golden.json` private. Its exact bytes remain available
for `pytest --run-private`. The personal practice notebook is excluded.

`verification.local.json` is an ignored local publication record, separate from
Session 26. It may contain private integrity metadata and must not be staged.
It does not assert hosted CI success or final model approval.

Preserve the current index until owner review. Do not use `git add .`. Review
`git diff --cached` and `git diff --cached --name-only` after any later explicit
staging. Source licensing is still an owner choice; none is silently granted.
