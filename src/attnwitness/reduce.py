"""Single-token deletion for trusted, deterministic mask builders.

This is not code minimization or a sandbox. The builder and policy factory must
be deterministic and preserve the meaning of stable IDs and logical positions.
"""
from dataclasses import asdict, dataclass
from .audit import AuditResult, audit
from .contracts import ContractError, Token, positive_int


@dataclass(frozen=True)
class ReductionResult:
    tokens: tuple
    audit: AuditResult
    category: str
    calls: int
    complete: bool

    def to_dict(self):
        return {'tokens': [asdict(t) for t in self.tokens], 'audit': self.audit.to_dict(),
                'category': self.category, 'calls': self.calls, 'complete': self.complete,
                'minimality': '1-minimal under single-token deletion' if self.complete else None,
                'assumption': 'Trusted deterministic callbacks; nonempty single-sequence cases only.'}


def reduce_case(tokens, builder, policy_factory, *, category, max_calls=256, backend='sdpa'):
    """Keep a verified witness. Budget exhaustion never claims minimality.

At most 64 input tokens keep full-category witness enumeration bounded. Each
attempt is charged before calling user code. Only nonempty deletion candidates
are in the minimization domain; original Token objects are not renumbered.
"""
    positive_int(max_calls, 'max_calls')
    if type(category) is not str or category not in ('forbidden_edge', 'missing_edge', 'empty_query'):
        raise ContractError('unsupported reduction category')
    if type(tokens) not in (list, tuple) or not 1 <= len(tokens) <= 64:
        raise ContractError('reduction requires a list/tuple of 1..64 tokens')
    if any(not isinstance(t, Token) for t in tokens) or not callable(builder) or not callable(policy_factory):
        raise ContractError('reduction needs Tokens and explicit trusted callbacks')
    current = tuple(tokens)
    calls = 0

    def evaluate(candidate):
        nonlocal calls
        calls += 1
        try:
            mask = builder(candidate)
            policy = policy_factory(candidate)
            checked = audit(mask, policy, backend=backend,
                            max_findings=len(candidate) ** 2 + len(candidate))
        except Exception as exc:
            raise ContractError(f'reduction callback or audit failed ({type(exc).__name__})') from exc
        return checked, any(f.category == category for f in checked.findings)

    best, fails = evaluate(current)
    if not fails:
        raise ContractError('initial case does not reproduce the requested violation')
    while len(current) > 1:
        accepted = False
        for index in range(len(current)):
            if calls >= max_calls:
                return ReductionResult(current, best, category, calls, False)
            candidate = current[:index] + current[index + 1:]
            checked, fails = evaluate(candidate)
            if fails:
                current, best, accepted = candidate, checked, True
                break
        if not accepted:
            return ReductionResult(current, best, category, calls, True)
    return ReductionResult(current, best, category, calls, True)
