"""Audit an explicitly declared relation; findings are not whole-model proofs."""
from dataclasses import asdict, dataclass
import numpy as np
from .contracts import ContractError, Policy, MatrixContract, positive_int
from .masks import normalize_mask


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

    def to_dict(self):
        return {'schema_version': 1, 'status': self.status,
                'scope': 'Hard visibility under the declared contract, not model or training correctness.',
                'findings': [asdict(f) for f in self.findings],
                'violation_count': self.violation_count,
                'findings_truncated': self.violation_count > len(self.findings),
                'coverage': self.coverage, 'preview': self.preview}


def audit(mask, policy, *, backend='sdpa', batch_size=1, heads=1, is_causal=False,
          max_cells=1048576, max_findings=128, key_padding_mask=None):
    for name, value in [('batch_size', batch_size), ('heads', heads), ('max_cells', max_cells),
                        ('max_findings', max_findings)]:
        positive_int(value, name)
    if not isinstance(policy, (Policy, MatrixContract)):
        raise ContractError('policy must be a Policy or MatrixContract')
    length = policy.length
    cells_count = batch_size * heads * length * length
    if cells_count > max_cells:
        raise ContractError(f'expanded relation needs {cells_count} cells, budget is {max_cells}')
    expected, nonempty, ids = policy.expand(batch_size, heads)
    visible = normalize_mask(mask, backend=backend, batch_size=batch_size, heads=heads, length=length,
                             is_causal=is_causal, key_padding_mask=key_padding_mask)
    forbidden = (expected == -1) & visible
    missing = (expected == 1) & ~visible
    empty = nonempty & ~visible.any(axis=-1)
    total = int(np.count_nonzero(forbidden) + np.count_nonzero(missing) + np.count_nonzero(empty))
    findings = []
    for category, flags in [('forbidden_edge', forbidden), ('missing_edge', missing)]:
        # Temporary indices are bounded by max_cells, not by max_findings.
        for flat in np.flatnonzero(flags):
            if len(findings) >= max_findings:
                break
            b, h, q, k = (int(i) for i in np.unravel_index(int(flat), flags.shape))
            findings.append(Finding(category, b, h, ids[b][q], ids[b][k], q, k))
    if len(findings) < max_findings:
        for b, h, q in np.argwhere(empty):
            if len(findings) >= max_findings:
                break
            b, h, q = int(b), int(h), int(q)
            findings.append(Finding('empty_query', b, h, ids[b][q], None, q, None))
    constrained = int(np.count_nonzero(expected))
    preview_length = min(length, 16)
    return AuditResult('VIOLATION' if total else 'PASS', tuple(findings), total,
                       {'backend': backend, 'batch_size': batch_size, 'heads': heads, 'length': length,
                        'total_cells': cells_count, 'constrained_cells': constrained,
                        'complete_relation_contract': constrained == cells_count,
                        'nonempty_queries': int(np.count_nonzero(nonempty)),
                        'unconstrained_cells': cells_count - constrained},
                       {'batch': 0, 'head': 0, 'ids': list(ids[0][:preview_length]),
                        'visible': visible[0, 0, :preview_length, :preview_length].tolist(),
                        'expected': expected[0, 0, :preview_length, :preview_length].tolist(),
                        'truncated': length > preview_length or heads > 1 or batch_size > 1})
