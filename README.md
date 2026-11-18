# AttnWitness

[![Tests](https://github.com/gedouluhui/attnwitness/actions/workflows/ci.yml/badge.svg)](https://github.com/gedouluhui/attnwitness/actions/workflows/ci.yml)

**Your mask runs. Does it express your attention contract?**

AttnWitness checks an actual attention mask against separately declared token visibility. It returns concrete query–key witnesses, an offline interactive evidence grid, and a bounded counterexample reducer.

No model download, GPU or API key is needed. Experimental v0.2: this is a **hard-visibility contract checker**, not a proof of model quality or whole-model causality.

[Open the recorded demo](https://gedouluhui.github.io/attnwitness/) · [中文快速开始](README.zh-CN.md) · [Verification scope](docs/VALIDATION.md)

## The two-token mistake

```python
import numpy as np
from attnwitness import Token, Policy, audit

policy = Policy([Token("first", "document", 0), Token("second", "document", 1)])
causal = np.array([[True, False], [True, True]])

assert audit(causal, policy).status == "PASS"
result = audit(causal.astype(np.float32), policy)
print(result.to_dict())
# VIOLATION: first -> second is visible, but the contract forbids it.
```

A floating SDPA mask is **added to attention scores**. Converting a boolean mask to 0/1 floats does not preserve hard blocking. Finite negative values remain biases; only negative infinity is recognized here as a hard block.

This is checked against actual CPU SDPA and MHA calls, not just against another copy of the mask-construction code.

## Run it

Python 3.10+ and NumPy 2+. From a virtual environment at the repository root:

```sh
python -m pip install .

# Expected: PASS / exit 0
attnwitness demo --case causal --html correct.html

# Expected: VIOLATION / exit 1, not a crashed demo
attnwitness demo --case float-mask --json evidence.json --html evidence.html

# A cross-document edge and a legitimate bidirectional prefix
attnwitness demo --case packed --html packed.html
attnwitness demo --case prefix --html prefix.html
```

Open the HTML locally. Click matrix cells, filter findings, inspect coverage and download the JSON. Reports contain no remote scripts, fonts, analytics or network calls. Existing reports are preserved unless you supply `--force`.

This project is not claimed to be on PyPI. Install the source checkout or a release wheel, not a guessed registry package.

### Your own contract and mask

```sh
python examples/make_mask.py
attnwitness check examples/causal.json examples/causal.npy --html custom.html
```

The explicit input is bounded JSON plus a numeric NPY file (formats 1.0/2.0, no pickle/object arrays). The input fingerprint is computed from the same bytes that are actually loaded. Output paths cannot overwrite either input, including through a hard link, and without `--force` a file created after preflight is not overwritten.

For batched input, `tokens` contains one token list per batch; `batch_size` and `heads` are explicit JSON fields. A missing mask argument is permitted only as an explicit no-mask case; an optional `is_causal` field is still interpreted and validated.

## Python interfaces

### Declare intent independently

`Policy(tokens, mode=..., require_nonempty=True)` supports `causal`, `packed`, `prefix`, and `bidirectional` modes. A `Token` carries its stable ID, segment, logical position, prefix membership and query/key validity.

Do not generate the expected contract with the mask builder being tested. Invalid queries are unconstrained; invalid keys are blocked for active queries. Prefix membership must form a prefix in each segment's logical order.

For partial declarations, use `MatrixContract(cells, token_ids, require_nonempty=False)` with an integer ndarray: `-1` must block, `1` must permit, `0` is deliberately unconstrained. Coverage reports the unconstrained cells; a partial PASS is not complete validation.

### Inspect real torch tensors

Install an appropriate CPU torch build first when you do not need CUDA:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[torch]"
```

```python
from attnwitness.torch_adapter import audit_torch
result = audit_torch(mask_tensor, policy, backend="sdpa")
```

The adapter accepts plain dense CPU boolean/float16/float32/float64 tensors. It does not mutate tensors, gradients, training flags or global hooks. Tensor subclasses and unsupported encodings are rejected.

### Reduce a builder failure

```python
from attnwitness import reduce_case

reduced = reduce_case(
    tokens,
    builder=my_deterministic_mask_builder,
    policy_factory=lambda subset: Policy(subset, mode="packed"),
    category="forbidden_edge",
    max_calls=256,
)
print(reduced.to_dict())
```

Callables are trusted code, not sandboxed. This mode is limited to 1–64 tokens and nonempty single-token deletions. Original Token objects, segment membership, positions and prefix roles are retained. A completed result is **1-minimal for those deletion operations**, not globally minimal. Exhausting the call budget retains a verified witness but does not claim minimality. A fixed matrix alone cannot produce a minimized builder program.

## Semantics and boundaries

| Input | Meaning |
|---|---|
| SDPA boolean `True` | Permit the edge |
| MHA explicit boolean `True` | Block the edge |
| Finite additive float | Bias, not unconditional blocking |
| Additive `-inf` | Hard block |
| Missing data / MaskedArray / unsupported subclass | Reject, do not silently fill |
| Empty query row | Violation only if the declaration requires it nonempty |

Square CPU self-attention only. MHA requires explicit masks and `is_causal=False`; causal hints are not modeled. When MHA attention and key-padding masks are both floating point, they are summed in their declared dtype before visibility is classified, matching the real backend merge; overflow to `-inf` blocks, while `NaN` and positive infinity are invalid. Mixed floating dtypes are rejected. KV-cache, cross-attention, NestedTensor and GPU-kernel correctness are outside v0.2. An allowed empty row does **not** imply numerical safety.

Default expanded-relation budget: 1,048,576 cells. Work runs in bounded tiles (`--tile-size`, default 128) so the largest temporary tile, not the full logical relation, controls working memory. CLI file loading permits `--max-cells` in 1..16,777,216; values outside that range are input errors. Shape compatibility is checked before elementwise allocation. Finding and preview caps do not cap the total violation count. Retained findings include original-coordinate context windows (`--max-evidence`, `--window-radius`) for their own batch and head; evidence truncation is reported separately from finding truncation. CLI exit codes: 0 PASS, 1 VIOLATION, 2 invalid/unsupported/error.

## Tests and comparisons

```sh
python -m pip install -e ".[torch]"
python -m unittest discover -s tests -v
python -m build
```

Tests include hand-calculated relations, actual torch calls, missing-data and allocation-order regressions, HTML injection checks, file fingerprint binding, and temporary-copy mutations that must reach named behavioral assertions. No live source is modified by mutation tests.

PyTorch executes masks; Transformers and Attention Gym already provide powerful constructors and backend adapters. AttnWitness adds a separately declared relation, concrete evidence and bounded reduction. It does not claim to invent masking or replace a well-tested constructor. See [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [Transformers mask handling](https://huggingface.co/docs/transformers/en/attention_interface), and [Attention Gym](https://meta-pytorch.org/attention-gym/masks/).

## License

Original code: MIT. NumPy and optional PyTorch keep their own licenses. Dependency binaries, model weights and private training data are not bundled.
