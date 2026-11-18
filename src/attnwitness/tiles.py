"""Shape-only validation and tile-local actual/expected relations."""
import numpy as np
from . import masks
from .contracts import ContractError, Policy, MatrixContract, plain_array


class ActualMask:
    def __init__(self, mask, padding, backend, batch_size, heads, length, is_causal):
        self.backend, self.length, self.causal = backend, length, is_causal
        self.mask, self.padding = None, None
        shape = (batch_size, heads, length, length)
        if backend not in ('sdpa', 'mha') or type(is_causal) is not bool:
            raise ContractError('unsupported backend or non-boolean is_causal')
        if backend == 'mha' and is_causal:
            raise ContractError('MHA causal hints are unsupported; supply an explicit mask')
        if backend == 'sdpa' and mask is not None and is_causal:
            raise ContractError('explicit SDPA mask and is_causal cannot be combined')
        if backend != 'mha' and padding is not None:
            raise ContractError('key_padding_mask is only supported for MHA')
        if mask is not None:
            array = plain_array(mask, 'mask')
            if array.dtype.kind not in 'bf':
                raise ContractError('only boolean and floating-point masks are supported')
            if backend == 'mha':
                if array.ndim == 3 and array.shape == (batch_size * heads, length, length):
                    # Index the flattened B*H dimension without copying a noncontiguous array.
                    self.flat_heads = heads
                elif array.ndim == 2:
                    self.flat_heads = None
                else:
                    raise ContractError('MHA mask must be (L,L) or (B*H,L,L)')
            else:
                self.flat_heads = None
            if not 2 <= array.ndim <= 4 or array.shape[-2:] != (length, length):
                raise ContractError('mask sequence dimensions must match the square contract')
            if self.flat_heads is None:
                try:
                    array = np.broadcast_to(array, shape)
                except ValueError as exc:
                    raise ContractError('mask cannot broadcast to declared batch/head dimensions') from exc
            self.mask = array
        if padding is not None:
            pad = plain_array(padding, 'key_padding_mask')
            if pad.shape != (batch_size, length) or pad.dtype.kind not in 'bf':
                raise ContractError('key_padding_mask must be boolean/float (B,L)')
            if self.mask is not None and pad.dtype != self.mask.dtype:
                raise ContractError('MHA attention and padding mask dtypes must match')
            self.padding = pad

    def block(self, b, h, q0, q1, k0, k1):
        if self.mask is None:
            if self.causal:
                visible = np.arange(q0, q1)[:, None] >= np.arange(k0, k1)[None, :]
            else:
                visible = np.ones((q1-q0, k1-k0), dtype=bool)
            raw = None
        else:
            raw = (self.mask[b*self.flat_heads+h, q0:q1, k0:k1] if self.flat_heads is not None
                   else self.mask[b, h, q0:q1, k0:k1])
            visible = masks._visible(raw, self.backend)
        if self.padding is not None:
            pad = self.padding[b, k0:k1][None, :]
            padding_visible = masks._visible(pad, 'mha')
            if raw is not None and raw.dtype.kind == 'f':
                with np.errstate(over='ignore', invalid='ignore'):
                    combined = np.add(raw, pad, dtype=raw.dtype)
                visible = masks._visible(combined, 'mha')
            else:
                visible = visible & padding_visible
        return visible


class ExpectedRelation:
    def __init__(self, policy, batches, heads):
        self.policy, self.length = policy, policy.length
        self.matrix = None
        if isinstance(policy, Policy):
            if len(policy.tokens) != batches:
                raise ContractError('one explicit token sequence per batch is required')
            self.ids = tuple(tuple(t.id for t in row) for row in policy.tokens)
            self.rows = []
            for row in policy.tokens:
                self.rows.append({'position': np.array([t.position for t in row], dtype=object),
                                  'segment': np.array([t.segment for t in row], dtype=object),
                                  'prefix': np.array([t.prefix for t in row]),
                                  'query': np.array([t.query_valid for t in row]),
                                  'key': np.array([t.key_valid for t in row])})
        elif isinstance(policy, MatrixContract):
            if policy.cells.dtype.kind not in 'iu':
                raise ContractError('contract cells must be integers -1,0,1')
            try:
                self.matrix = np.broadcast_to(policy.cells, (batches, heads, self.length, self.length))
            except ValueError as exc:
                raise ContractError('contract cannot broadcast to the declared batch/head shape') from exc
            self.ids = tuple(policy.token_ids for _ in range(batches))
        else:
            raise ContractError('unsupported relation contract')

    def required_rows(self, batch):
        if self.matrix is not None:
            return np.full(self.length, self.policy.require_nonempty, dtype=bool)
        return self.rows[batch]['query'] & self.policy.require_nonempty

    def block(self, b, h, q0, q1, k0, k1):
        if self.matrix is not None:
            result = self.matrix[b, h, q0:q1, k0:k1]
            if not np.isin(result, (-1, 0, 1)).all():
                raise ContractError('contract cells must be integers -1,0,1')
            return result
        row, mode = self.rows[b], self.policy.mode
        qvalid = row['query'][q0:q1, None]
        permit = np.broadcast_to(row['key'][None, k0:k1], (q1-q0, k1-k0)).copy()
        if mode in ('packed', 'prefix'):
            permit &= row['segment'][q0:q1, None] == row['segment'][None, k0:k1]
        if mode in ('causal', 'packed'):
            permit &= row['position'][q0:q1, None] >= row['position'][None, k0:k1]
        elif mode == 'prefix':
            permit &= row['prefix'][None, k0:k1] | (~row['prefix'][q0:q1, None] & (
                row['position'][q0:q1, None] >= row['position'][None, k0:k1]))
        return np.where(qvalid, np.where(permit, 1, -1), 0).astype(np.int8)
