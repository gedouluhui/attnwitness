"""Conservative hard-visibility semantics; finite biases are never hard masks."""
import numpy as np
from .contracts import ContractError, plain_array


def _visible(array, backend):
    if array.dtype.kind == 'b':
        return array if backend == 'sdpa' else ~array
    if array.dtype.kind != 'f':
        raise ContractError('only boolean or floating-point masks are supported')
    if np.isnan(array).any() or np.isposinf(array).any():
        raise ContractError('mask contains NaN or positive infinity')
    return ~np.isneginf(array)


def normalize_mask(mask, *, backend, batch_size, heads, length, is_causal, key_padding_mask):
    shape = (batch_size, heads, length, length)
    if backend not in ('sdpa', 'mha') or type(is_causal) is not bool:
        raise ContractError('unsupported backend or non-boolean is_causal')
    if backend == 'mha' and is_causal:
        raise ContractError('MHA causal hints are not modeled; supply an explicit mask')
    if backend == 'sdpa' and mask is not None and is_causal:
        raise ContractError('explicit SDPA mask and is_causal cannot be combined in this contract')
    if backend != 'mha' and key_padding_mask is not None:
        raise ContractError('key_padding_mask is only supported for MHA')
    array = None
    if mask is None:
        visible = np.tri(length, dtype=bool) if is_causal else np.ones((length, length), dtype=bool)
    else:
        array = plain_array(mask, 'mask')
        if backend == 'mha':
            if array.ndim == 3 and array.shape == (batch_size * heads, length, length):
                array = array.reshape(shape)
            elif array.ndim != 2:
                raise ContractError('MHA mask must be (L,L) or (B*H,L,L)')
        if array.ndim < 2 or array.ndim > 4 or array.shape[-2:] != (length, length):
            raise ContractError('mask sequence dimensions must match the square contract')
        # Shape-only validation precedes elementwise scans and their temporary arrays.
        try:
            array = np.broadcast_to(array, shape)
        except ValueError as exc:
            raise ContractError('mask cannot broadcast to declared batch/head dimensions') from exc
        visible = _visible(array, backend)
    visible = np.broadcast_to(visible, shape)
    if key_padding_mask is not None:
        padding = plain_array(key_padding_mask, 'key_padding_mask')
        if padding.shape != (batch_size, length):
            raise ContractError('key_padding_mask must have shape (B,L)')
        if array is not None and padding.dtype != array.dtype:
            raise ContractError('MHA attention and padding mask dtypes must match')
        padding_visible = _visible(padding, 'mha')[:, None, None, :]
        if array is not None and array.dtype.kind == 'f':
            with np.errstate(over='ignore', invalid='ignore'):
                combined = np.add(array, padding[:, None, None, :], dtype=array.dtype)
            visible = _visible(combined, 'mha')
        else:
            visible = visible & padding_visible
    return visible
