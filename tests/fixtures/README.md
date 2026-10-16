# Test fixture provenance

Public tests construct synthetic examples in their source files. Their amounts,
IDs, labels and proxy values are hand-created test inputs, not exported IEEE-CIS
observations. Toy fits test algorithms; they do not reproduce the saved research
models or establish research performance.

`session22_golden.json` is **private and explicitly ignored**. It contains six
real IEEE-CIS-derived validation observations and independently saved Session 20
expected scores. Redistribution permission has not been established. Preserve
the existing local file and its bytes: it belongs to the historical release.
It must not appear in the public staging allowlist or a public artifact bundle.

`pytest --run-private` includes the original golden assertions, without changing
their expected values or tolerances. Missing private artifacts cause failures;
they are not replaced by invented predictions or silently skipped. These checks
require authorized local evidence as described in `docs/REPRODUCING.md`.

The public tier intentionally has no substitute golden claim for the private
research model. It tests causal features, metrics, contracts and algorithms with
synthetic inputs. Run it with `python -B -m pytest -q` from the repository root.
