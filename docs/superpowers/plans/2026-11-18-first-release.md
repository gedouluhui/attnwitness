# AttnWitness First Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Audit explicitly declared attention visibility and emit independently verifiable counterexamples, without loading a language model.

**Architecture:** A NumPy contract engine interprets actual masks separately from token metadata policies. Optional CPU PyTorch adapters validate backend semantics; a bounded reducer reruns a trusted builder. A CLI and self-contained report expose the same result objects.

**Tech Stack:** Python >=3.10, NumPy >=2.0; PyTorch is an optional integration and required release-test dependency. Standard-library unittest; setuptools packaging.

**Spec:** Approved contract in the following section; this public plan contains no private handover data.

## Global Constraints / Approved Contract

- CPU, square training-time self-attention only; no KV-cache, cross-attention, GPU kernels or whole-model causality claims.
- Explicit metadata: stable token IDs, segment, logical order, prefix membership, query/key validity. Contract cells are must-allow, must-block, or unconstrained.
- SDPA bool True permits; MHA explicit bool True blocks. Additive finite values are biases, only negative infinity is a hard block. Reject NaN and positive infinity.
- SDPA implicit is_causal is supported without an explicit mask. MHA causal hints are unsupported in v0.1. Invalid flag combinations cannot PASS.
- Default maximum expanded relation count 1,048,576; validate before allocating. Input errors are typed errors, never no-findings PASS.
- An empty row only violates an explicit require-nonempty policy for that query. Allowed empty rows do not imply numerical safety.
- Expected policy cannot call the actual mask builder. Fixed tensors provide edge witnesses only. Builder reduction may claim 1-minimal only after completed single-deletion checks; max_calls exhaustion is explicitly incomplete.
- No global hooks, downloads, credentials, external report requests, pickle loading, user text in demos, invented benchmark claims or broad scans.
- Public source, examples, license, English README and Chinese quickstart are required. Agents do not commit/push or modify sibling projects. Main owner performs final publication review.

## Task 1: Contract and mask semantics

**Files:** `src/attnwitness/contracts.py`, `src/attnwitness/masks.py`, `src/attnwitness/audit.py`, `src/attnwitness/__init__.py`, `tests/test_core.py`, `pyproject.toml`.

**Interfaces:** `Token(id: str, segment: str, position: int, prefix: bool=False, query_valid: bool=True, key_valid: bool=True)`; `Policy(tokens, mode='causal', require_nonempty=True)` with modes causal/packed/prefix; `audit(mask, policy, *, backend='sdpa', batch_size=1, heads=1, is_causal=False, max_cells=1048576) -> AuditResult`. `AuditResult` exposes `status`, `findings`, `coverage`, `to_dict()`; findings retain batch/head/query/key original IDs and category. Also expose explicit tri-state matrix contracts for partially constrained policies. Broadcast dimensions are validated; no inferred batch/head semantics.

- [ ] Write actual unittest cases before engine implementation. Begin with a callable API stub that raises NotImplementedError and watch the behavioral case fail.

```python
p = Policy([Token('a', 's', 0), Token('b', 's', 1)])
self.assertEqual(audit(np.tril(np.ones((2, 2), dtype=bool)), p).status, 'PASS')
bad = audit(np.tril(np.ones((2, 2), dtype=np.float32)), p)
self.assertEqual(bad.status, 'VIOLATION')
self.assertTrue(any(f.query_id == 'a' and f.key_id == 'b' for f in bad.findings))
```

- [ ] Run `python -m unittest discover -s tests -v`; retain concise red evidence in the local implementation log, then implement masks and independent policy comparisons.
- [ ] Add packed diagonal, prefix first-two-tokens, explicit partial constraints, allowed padding empty rows, MHA inversion, head-specific broadcasting, invalid metadata, NaN/+inf, no-mask flags, and budget tests. Each test names the production defect it catches.
- [ ] Implement semantics with explicit validity checks before NumPy allocation; produce structured coverage and counts independently from capped witness lists. Re-run all tests.

## Task 2: Real backend checks and bounded reduction

**Files:** `src/attnwitness/torch_adapter.py`, `src/attnwitness/reduce.py`, `tests/test_torch.py`, `tests/test_reduce.py`.

**Interfaces:** torch adapter converts a CPU supported tensor without mutating it and delegates to audit; `reduce_case(tokens, builder, policy_factory, *, category, max_calls=256) -> ReductionResult`, retaining original token IDs, calls, completed flag, and minimality scope. Callables are trusted user code, never imported from a JSON field.

- [ ] Write real SDPA tests with Q/K zero, distinct V entries, and dropout_p=0.0. A forbidden future V perturbation must not change q0 with a valid mask; the 0/1 floating mask must visibly change it. Also exercise actual MHA with explicit masks and is_causal=False.
- [ ] Write reducer tests that fail when deletion breaks prefix/segment identity, when the returned example stops reproducing, or when a max_calls=1 result falsely claims minimality.

```python
self.assertFalse(reduce_case(tokens, builder, policy_factory,
                             category='forbidden_edge', max_calls=1).complete)
```

- [ ] Implement greedy single-deletion minimization, restarting deletion passes after each accepted change; preserve the violation category and account for every builder call. If budget runs out, return incomplete with a still-verified last witness.
- [ ] Run torch tests without skips for release verification. Mutate a copied engine to invert SDPA boolean meaning, treat finite negatives as hard blocks, and always mark reduction minimal; the named tests must fail for those reasons. Never mutate the live product directory.

## Task 3: Usable CLI, evidence report and distribution

**Files:** `src/attnwitness/cli.py`, `src/attnwitness/report.py`, `src/attnwitness/__main__.py`, `examples/`, `tests/test_cli.py`, `tests/test_report.py`, `README.md`, `README.zh-CN.md`, `LICENSE`, `.gitignore`, `.github/workflows/ci.yml`.

**Interfaces:** `attnwitness demo --case {causal,float-mask,packed,prefix} --json FILE --html FILE`; optional `check SPEC.json MASK.npy` reads explicit files with allow_pickle=False and bounded header/element checks. Exit 0 means checked PASS; 1 violation; 2 malformed/unsupported/incomplete. `render_report(result) -> str` must escape untrusted labels.

- [ ] Add subprocess tests for demo success/violation exit codes and JSON reread; malicious `</script>` token labels must not become executable HTML. Output paths cannot overwrite the input files.
- [ ] Implement a polished, offline, keyboard-accessible report with evidence grid, category filters, coverage and limitations. No CDN/fonts/analytics or arbitrary raw JSON insertion into script.
- [ ] Document installation, API, actual CLI examples, supported semantics and non-goals; use MIT for original code and list dependencies accurately. Include both failure and repaired example.
- [ ] Build wheel and install into an isolated venv; run the installed CLI outside the checkout. CI runs core, real CPU torch integration and build on Python 3.11/3.12, with a Windows smoke lane. No network claims beyond what CI actually tests.
- [ ] Main owner independently reviews the code, runs full tests and representative browser report flows, then explicitly stages reviewed source files and publishes only after privacy/license checks. No success badge or remote URL claim until the actual remote is verified.
