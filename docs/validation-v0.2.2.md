# Validation — v0.2.2

Recorded 2026-11-28 on Windows x64 / Python 3.12.3. These are local results; hosted matrix execution is not claimed.

## Source gate

```text
PYTHONPATH=src PYTHONUTF8=1 python -B -m unittest discover -s tests -v
Ran 69 tests in 13.266s
OK
```

The suite covers explicit contracts, real CPU torch controls, tiled relation work, broadcast-budget refusal before materialization, witness ordering/context, output protection, report escaping and named silent-bug mutations. Python 3.10 grammar compatibility and the Ubuntu/Windows 3.10–3.12 matrix are configured in CI.

## Installed wheel gate

The current wheel was installed into a separate target outside the checkout. Imports resolved only to that target and reported version 0.2.2. The installed causal demo returned PASS/exit 0; the installed float-mask demo returned VIOLATION/exit 1 with one forbidden-edge witness. Both JSON and HTML outputs were written outside the checkout.

## Release boundaries

The optional torch adapter uses the installed CPU torch build. Checker-side relation tiling bounds its own work; caller inputs, unrelated intermediates and retained outputs are outside that bound. This is not a model-quality or whole-model safety proof. Historical v0.2.1 evidence remains in [VALIDATION.md](VALIDATION.md).
