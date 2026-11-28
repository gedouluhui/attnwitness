# Changelog


## Unreleased

## 0.2.2 — 2026-11-28

- Exercise Python 3.10/3.11/3.12 on Ubuntu and Windows with immutable Action pins and both distribution builds.
- Reject broadcast-budget violations before relation construction and document the checker-side tiling memory boundary.
- Align public governance, bilingual quickstarts, issue templates, documentation validation, and current release evidence.
- Current source version. Supported scope and limitations are documented in [README.md](README.md) and [validation](docs/validation-v0.2.2.md).

## 0.2.1

- Checkout recorded examples with LF line endings on every platform so the byte-for-byte reproduction test also passes on Windows CI. No runtime behavior changed from 0.2.0.

## 0.2.0

- MHA floating attention and key-padding masks are summed in their declared dtype before hard-visibility classification, matching actual backend composition; overflow to `-inf` blocks, while `NaN` and positive infinity make the input invalid. Mixed floating dtypes are rejected rather than guessed.
- Output publication uses same-directory staging plus atomic exclusive create or entry replacement. Hard links to inputs are detected by file identity, and a file created after preflight is not overwritten without `--force`.
- The relation check runs in bounded tiles with exact counters and deterministic global witness order. Large logical broadcasts no longer require an equally large expanded mask or `argwhere` array.
- Retained findings carry original-coordinate context windows for their own batch and head, including late witnesses outside the legacy preview; empty-query windows record how many keys were checked.
- Finding and evidence truncation are reported separately. CLI exposes bounded `--tile-size`, `--max-evidence`, `--window-radius` and `--max-findings` options.
- Generated examples cover prefix visibility, combined float overflow and a late batch/head witness.

## 0.1.0

- Independent causal, packed, prefix and partial visibility contracts.
- Explicit SDPA/MHA mask encodings, bounded relation checking and original-position witnesses.
- Optional CPU torch adapter and real backend characterization tests.
- Trusted-builder single-token reduction with an explicit incomplete result at budget exhaustion.
- Offline interactive evidence reports, JSON/NPY CLI input, source fingerprints and protected output paths.
- Regression tests and temporary-copy silent-bug mutations.

This is an experimental release. Supported-scope and review limitations are documented in `docs/VALIDATION.md`; no model-quality or whole-model safety guarantee is made.
