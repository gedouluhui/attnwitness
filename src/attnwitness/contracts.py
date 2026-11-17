"""User-owned visibility intent, independent of the mask being inspected."""
from dataclasses import dataclass
import numpy as np


class ContractError(ValueError):
    """Malformed, unsupported or over-budget audit input."""


def plain_array(value, name):
    # MaskedArray/other subclasses can carry semantics that np.asarray discards.
    # Explicit ndarray input also avoids unbounded materialization of arbitrary iterables.
    if type(value) is not np.ndarray:
        raise ContractError(f'{name} must be a plain numpy.ndarray with complete data')
    return value


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ContractError(f'{name} must be a positive integer')
    return value


@dataclass(frozen=True)
class Token:
    id: str
    segment: str
    position: int
    prefix: bool = False
    query_valid: bool = True
    key_valid: bool = True

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id or not isinstance(self.segment, str) or not self.segment:
            raise ContractError('token id and segment must be nonempty strings')
        if type(self.position) is not int or self.position < 0:
            raise ContractError('token position must be a nonnegative integer')
        if any(type(x) is not bool for x in (self.prefix, self.query_valid, self.key_valid)):
            raise ContractError('token flags must be booleans')


@dataclass(frozen=True)
class Policy:
    tokens: object
    mode: str = 'causal'
    require_nonempty: bool = True

    def __post_init__(self):
        if self.mode not in ('causal', 'packed', 'prefix', 'bidirectional'):
            raise ContractError('unsupported policy mode')
        if type(self.require_nonempty) is not bool:
            raise ContractError('require_nonempty must be boolean')
        try:
            tokens = tuple(self.tokens)
            batches = (tokens,) if tokens and isinstance(tokens[0], Token) else tuple(tuple(b) for b in tokens)
        except (TypeError, ValueError) as exc:
            raise ContractError('tokens must be a sequence or batch of token sequences') from exc
        if not batches or not batches[0]:
            raise ContractError('at least one token is required')
        length = len(batches[0])
        for batch in batches:
            if len(batch) != length or any(not isinstance(t, Token) for t in batch):
                raise ContractError('batches must contain equally sized Token sequences')
            if len({t.id for t in batch}) != length:
                raise ContractError('token IDs must be unique within each batch')
            if len({(t.segment, t.position) for t in batch}) != length:
                raise ContractError('positions must be unique within each segment')
            if self.mode == 'prefix':
                for segment in {t.segment for t in batch}:
                    role_order = [t.prefix for t in sorted((t for t in batch if t.segment == segment),
                                                          key=lambda t: t.position)]
                    if any(role_order[i] and not role_order[i - 1] for i in range(1, len(role_order))):
                        raise ContractError('prefix tokens must precede suffix tokens within a segment')
        object.__setattr__(self, 'tokens', batches)

    @property
    def length(self):
        return len(self.tokens[0])

    def expand(self, batch_size, heads):
        if len(self.tokens) != batch_size:
            raise ContractError('one explicit token sequence per batch is required')
        cells = np.zeros((batch_size, 1, self.length, self.length), dtype=np.int8)
        required = np.zeros((batch_size, 1, self.length), dtype=bool)
        for b, tokens in enumerate(self.tokens):
            for q, query in enumerate(tokens):
                if not query.query_valid:
                    continue
                required[b, 0, q] = self.require_nonempty
                for k, key in enumerate(tokens):
                    permit = key.key_valid
                    if self.mode in ('packed', 'prefix'):
                        permit = permit and query.segment == key.segment
                    if self.mode in ('causal', 'packed'):
                        permit = permit and key.position <= query.position
                    elif self.mode == 'prefix':
                        permit = permit and (key.prefix or (not query.prefix and key.position <= query.position))
                    cells[b, 0, q, k] = 1 if permit else -1
        return (np.broadcast_to(cells, (batch_size, heads, self.length, self.length)),
                np.broadcast_to(required, (batch_size, heads, self.length)),
                tuple(tuple(t.id for t in ts) for ts in self.tokens))


@dataclass(frozen=True)
class MatrixContract:
    """Cells: -1 must block, +1 must permit, 0 unconstrained. IDs shared across batches."""
    cells: object
    token_ids: object
    require_nonempty: bool = False

    def __post_init__(self):
        a = plain_array(self.cells, 'contract cells')
        try:
            if isinstance(self.token_ids, str):
                raise TypeError('IDs must be a sequence of strings')
            ids = tuple(self.token_ids)
        except TypeError as exc:
            raise ContractError('token_ids must be a sequence of strings') from exc
        if a.ndim not in (2, 4) or a.shape[-1] != a.shape[-2] or a.shape[-1] == 0:
            raise ContractError('matrix contract must be nonempty square 2D or 4D')
        if len(ids) != a.shape[-1] or any(not isinstance(s, str) or not s for s in ids) or len(set(ids)) != len(ids):
            raise ContractError('unique nonempty IDs must match contract length')
        if type(self.require_nonempty) is not bool:
            raise ContractError('require_nonempty must be boolean')
        object.__setattr__(self, 'cells', a)
        object.__setattr__(self, 'token_ids', ids)

    @property
    def length(self):
        return self.cells.shape[-1]

    def expand(self, batch_size, heads):
        if self.cells.dtype.kind not in 'iu':
            raise ContractError('contract cells must be integers -1, 0, or 1')
        try:
            cells = np.broadcast_to(self.cells, (batch_size, heads, self.length, self.length))
        except ValueError as exc:
            raise ContractError('contract cannot broadcast to the declared batch/head shape') from exc
        if not np.isin(cells, (-1, 0, 1)).all():
            raise ContractError('contract cells must be integers -1, 0, or 1')
        required = np.full((batch_size, heads, self.length), self.require_nonempty, dtype=bool)
        return cells, required, tuple(self.token_ids for _ in range(batch_size))
