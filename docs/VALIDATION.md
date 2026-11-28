# Verification scope

This is an experimental 0.2 release. Functional tests, independent review and external adoption are different kinds of evidence.

## Executable checks

```sh
python -m pip install -e ".[torch]"
python -m unittest discover -s tests -v
python -m build
```

The suite checks independently specified causal/packed/prefix relations, boolean encoding polarity, additive-mask semantics, per-batch/head identity, partial coverage, empty-row policy, rejection before large array operations, real CPU SDPA/MHA effects, reducer identity and budget behavior, report injection handling, CLI exit codes, input preservation and the binding of fingerprints to loaded bytes. The 0.2 additions cover floating MHA attention plus key-padding summation against an actual `MultiheadAttention` call, positive-overflow rejection, mixed-dtype rejection, tiled exact counts across tile sizes, late out-of-preview witnesses with their own batch/head context, exact counters under an evidence cap, broadcast workloads without full expanded temporaries, and real hard-link/race publication controls. The compatibility branch adds a 4,194,304-cell broadcast relation that must raise `ContractError` before either relation implementation is constructed.

Mutation checks operate only on temporary source copies. A baseline must pass; the intended edit must be unique and nonempty; the mutated run must fail at a named behavioral assertion. A syntax/import error is not accepted as catching a silent bug.

## Compatibility matrix

The v0.2.1 public contract declares Python `>=3.10` and NumPy `>=2.0`; the optional torch adapter uses the installed torch build. The current branch configures the following CI matrix, installing the package before the suite and building both distributions in every job:

| Python | Ubuntu | Windows |
|---|---|---|
| 3.10 | Configured | Configured |
| 3.11 | Configured | Configured |
| 3.12 | Configured | Configured |

Checkout and setup-python Actions are pinned to full commit SHAs (`actions/checkout` v4.2.2 and `actions/setup-python` v5.6.0). This is a statement of configured coverage, not a claim that six remote jobs have passed. The local task verification ran Python 3.12.3 on Windows with NumPy 2.4.6 and torch 2.12.0 CPU: the focused broadcast-budget regression passed and all 69 tests passed in 10.874 seconds. The wheel and sdist both built successfully at version 0.2.1. Python 3.10/3.11, Linux, and the remote matrix remain evidence gaps until their jobs run. The older four-job release evidence below remains valid only for its recorded tag and does not supply Python 3.10 coverage.

## What the proof does and does not cover

A completed audit checks every logical cell exactly once: `checked_cells == total_cells == batch_size * heads * length²`, with `sampling=false`. The full logical relation must fit `max_cells` before expected or actual relation setup, even when the input mask uses broadcasting. Rejection is not a PASS. Accepted work uses relation tiles of at most `tile_size²` cells per batch/head; this is not a total-memory bound on caller-owned arrays, intermediate/aggregate outputs or retained evidence. Tiling changes neither exact counts nor the verdict.

A PASS with `complete_relation_contract=true` covers every declared cell and is complete for that hard-visibility contract only. With a partial `MatrixContract`, `0` cells are unconstrained: PASS covers only constrained cells and is reported by `complete_relation_contract=false`. Neither form validates model quality, training correctness or whole-model causality.

The `preview` is capped to batch/head 0 and the first 16 positions; `preview.truncated` exposes omissions. `max_findings` caps stored findings, while `violation_count` and per-category `counts` remain exact. `max_evidence` separately caps retained context windows and `evidence_truncated` compares their number with retained findings. Neither cap limits cells scanned. Empty-query evidence retains a full-row summary even when its displayed context window is small. See the README evidence/reduction table for defaults and accepted ranges.

`reduce_case` is bounded to one sequence of 1..64 tokens and trusted deterministic callbacks. The initial reproduction and each single-token deletion attempt consume one call before user code runs. Exhaustion preserves the last verified witness with `complete=false` and no minimality claim. A completed result is 1-minimal only for nonempty single-token deletions with original IDs, segments, positions and prefix roles retained; it is not globally minimal or a minimized builder program.

## Review provenance

The contract core received an independent initial review. Missing-data handling and validation-before-allocation issues were reproduced, fixed and independently re-reviewed as passing. The torch/reducer review found a test gap around prefix/segment identity; targeted regressions and executable mutations were added. Its final independent re-review could not run because the review service reported its daily limit. Later CLI/report review and final integration checking were performed by the implementation owner, not a new independent reviewer.

No claim of complete independent final sign-off is made. No known runtime defect was found by the completed torch/reducer review; its remaining finding concerned regression protection, which is now exercised by the suite.

## Environment and limits

Development verification used Windows, Python 3.12, NumPy 2.x and PyTorch 2.12 CPU. The released 0.2.1 build runs 68 tests with no failures; a fresh environment installed the built wheel and ran its console demo outside the source checkout, reporting installed version 0.2.1. The normal and deliberately broken CLI examples also ran outside the checkout, with their JSON/HTML reread and exit codes checked. That fresh environment inherited the machine's optional PyTorch installation rather than downloading a separate CPU build.

[GitHub CI run 44411040290](https://github.com/gedouluhui/attnwitness/actions/runs/44411040290) passed on tag commit `2b1ea9f`: all four Python 3.11/3.12 × Windows/Linux jobs succeeded. [Release v0.2.1](https://github.com/gedouluhui/attnwitness/releases/tag/v0.2.1) published the wheel and sdist read back with the recorded SHA256 values. The original v0.2.0 tag CI failed because Windows checked out recorded JSON with CRLF; 0.2.1 only adds LF checkout rules and changes no runtime behavior. This is evidence for those commits and environments, not every possible dependency version. The public Pages example was opened in a browser and its actual cell-click handler verified; responsive checks covered 1440 and 360 CSS pixels. These are owner-run checks, not an external user study.

The recorded HTML example is generated by the CLI with synthetic data. It is not a live model, user study, performance benchmark or evidence of adoption. Its matrix and findings come from the actual audit result.

No CUDA, Safari, Firefox, real mobile device, screen-reader or external-user study is claimed. Browser responsive checks are not equivalent to all-device certification. A complete contract PASS is still not a whole-model correctness guarantee.

## Report security boundary

Reports escape user-controlled text and use static local scripts with a restrictive no-network CSP. Numeric NPY input is bounded and never permits pickle. Arbitrary Python callbacks supplied to the reducer remain trusted executable code, not sandboxed code. Review any report before publishing it: even without training text, user-chosen token labels can contain information you consider private.
