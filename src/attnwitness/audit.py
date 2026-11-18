"""Exact tiled audit of declared relations, with bounded counterexample context."""
from dataclasses import asdict, dataclass, field
import heapq
import numpy as np
from .contracts import ContractError, Policy, MatrixContract, positive_int
from .tiles import ActualMask, ExpectedRelation


@dataclass(frozen=True)
class Finding:
    category: str
    batch: int
    head: int
    query_id: str
    key_id: str | None
    query_index: int
    key_index: int | None


@dataclass(frozen=True)
class AuditResult:
    status: str
    findings: tuple
    violation_count: int
    coverage: dict
    preview: dict
    counts: dict = field(default_factory=dict)
    evidence: tuple = ()

    def to_dict(self):
        return {'schema_version': 1, 'status': self.status,
                'scope': 'Hard visibility under the declared contract, not model or training correctness.',
                'findings': [asdict(f) for f in self.findings], 'violation_count': self.violation_count,
                'findings_truncated': self.violation_count > len(self.findings),
                'coverage': self.coverage, 'preview': self.preview, 'counts': self.counts,
                'evidence': list(self.evidence), 'evidence_truncated': len(self.evidence) < len(self.findings)}


TILE_SIZE_RANGE = (1, 1024)
MAX_EVIDENCE_RANGE = (0, 128)
WINDOW_RADIUS_RANGE = (0, 8)
MAX_FINDINGS_RANGE = (1, 4096)


def bounded_int(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ContractError(f'{name} must be an integer in {minimum}..{maximum}')
    return value


def audit(mask, policy, *, backend='sdpa', batch_size=1, heads=1, is_causal=False,
          max_cells=1048576, max_findings=128, key_padding_mask=None,
          tile_size=128, max_evidence=16, window_radius=3):
    for name, value in [('batch_size', batch_size), ('heads', heads), ('max_cells', max_cells)]:
        positive_int(value, name)
    bounded_int(max_findings, 'max_findings', *MAX_FINDINGS_RANGE)
    bounded_int(tile_size, 'tile_size', *TILE_SIZE_RANGE)
    bounded_int(max_evidence, 'max_evidence', *MAX_EVIDENCE_RANGE)
    bounded_int(window_radius, 'window_radius', *WINDOW_RADIUS_RANGE)
    if not isinstance(policy, (Policy, MatrixContract)):
        raise ContractError('policy must be a Policy or MatrixContract')
    length = policy.length
    cells_count = batch_size * heads * length * length
    if cells_count > max_cells:
        raise ContractError(f'expanded relation needs {cells_count} cells, budget is {max_cells}')
    expected = ExpectedRelation(policy, batch_size, heads)
    actual = ActualMask(mask, key_padding_mask, backend, batch_size, heads, length, is_causal)
    categories = ('forbidden_edge', 'missing_edge', 'empty_query')
    counts = dict.fromkeys(categories, 0)
    heaps = {category: [] for category in categories}
    constrained = nonempty_count = largest_tile = 0

    def retain(category, index, b, h, q, k):
        heap = heaps[category]
        entry = (-index, b, h, q, -1 if k is None else k)
        if len(heap) < max_findings:
            heapq.heappush(heap, entry)
        elif index < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    for b in range(batch_size):
        required = expected.required_rows(b)
        for h in range(heads):
            nonempty_count += int(np.count_nonzero(required))
            row_visible = np.zeros(length, dtype=bool)
            for q0 in range(0, length, tile_size):
                q1 = min(length, q0 + tile_size)
                for k0 in range(0, length, tile_size):
                    k1 = min(length, k0 + tile_size)
                    want = expected.block(b, h, q0, q1, k0, k1)
                    seen = actual.block(b, h, q0, q1, k0, k1)
                    largest_tile = max(largest_tile, want.size)
                    constrained += int(np.count_nonzero(want))
                    row_visible[q0:q1] |= seen.any(axis=-1)
                    for category, flags in (('forbidden_edge', (want == -1) & seen),
                                            ('missing_edge', (want == 1) & ~seen)):
                        count = int(np.count_nonzero(flags))
                        counts[category] += count
                        if not count:
                            continue
                        lowest = ((b * heads + h) * length + q0) * length + k0
                        heap = heaps[category]
                        if len(heap) >= max_findings and lowest > -heap[0][0]:
                            continue
                        # At most one tile's indices; first K within any tile suffice for global first K.
                        for local in np.flatnonzero(flags)[:max_findings]:
                            q, k = q0 + int(local) // (k1-k0), k0 + int(local) % (k1-k0)
                            index = ((b * heads + h) * length + q) * length + k
                            retain(category, index, b, h, q, k)
            empty = required & ~row_visible
            counts['empty_query'] += int(np.count_nonzero(empty))
            for q in np.flatnonzero(empty)[:max_findings]:
                q = int(q)
                retain('empty_query', (b * heads + h) * length + q, b, h, q, None)
    findings = []
    for category in categories:
        for _, b, h, q, k in sorted(heaps[category], key=lambda item: -item[0]):
            if len(findings) >= max_findings:
                break
            findings.append(Finding(category, b, h, expected.ids[b][q],
                                    None if k < 0 else expected.ids[b][k], q, None if k < 0 else k))
    windows = []
    for index, finding in enumerate(findings[:max_evidence]):
        b, h, q = finding.batch, finding.head, finding.query_index
        k = finding.key_index if finding.key_index is not None else 0
        q0, q1 = max(0, q-window_radius), min(length, q+window_radius+1)
        k0, k1 = max(0, k-window_radius), min(length, k+window_radius+1)
        window = {'finding_index': index, 'batch': b, 'head': h,
                  'query_indices': list(range(q0, q1)), 'key_indices': list(range(k0, k1)),
                  'query_ids': list(expected.ids[b][q0:q1]), 'key_ids': list(expected.ids[b][k0:k1]),
                  'expected': expected.block(b, h, q0, q1, k0, k1).tolist(),
                  'visible': actual.block(b, h, q0, q1, k0, k1).tolist()}
        if finding.key_index is None:
            window['row_summary'] = {'keys_checked': length, 'visible_keys': 0}
        windows.append(window)
    preview_length = min(length, 16)
    total = sum(counts.values())
    return AuditResult('VIOLATION' if total else 'PASS', tuple(findings), total,
                       {'backend': backend, 'batch_size': batch_size, 'heads': heads, 'length': length,
                        'total_cells': cells_count, 'checked_cells': cells_count, 'constrained_cells': constrained,
                        'complete_relation_contract': constrained == cells_count,
                        'nonempty_queries': nonempty_count, 'unconstrained_cells': cells_count - constrained,
                        'tile_size': tile_size, 'largest_tile_cells': largest_tile,
                        'sampling': False, 'max_evidence': max_evidence},
                       {'batch': 0, 'head': 0, 'ids': list(expected.ids[0][:preview_length]),
                        'visible': actual.block(0, 0, 0, preview_length, 0, preview_length).tolist(),
                        'expected': expected.block(0, 0, 0, preview_length, 0, preview_length).tolist(),
                        'truncated': length > preview_length or heads > 1 or batch_size > 1},
                       counts, tuple(windows))
