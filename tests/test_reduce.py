"""A reduced case must still reproduce and cannot overclaim global minimality."""
import unittest
import numpy as np
from attnwitness import Token, Policy, ContractError
from attnwitness.reduce import reduce_case


def leaking_builder(tokens):
    return np.ones((len(tokens), len(tokens)), dtype=bool)


class ReductionTests(unittest.TestCase):
    def setUp(self):
        self.tokens = [Token(f't{i}', 's', i * 2) for i in range(6)]

    def test_removes_tokens_but_preserves_original_identity_and_failure(self):
        result = reduce_case(self.tokens, leaking_builder, Policy, category='forbidden_edge')
        self.assertTrue(result.complete)
        self.assertEqual(len(result.tokens), 2)
        self.assertEqual(result.audit.status, 'VIOLATION')
        self.assertTrue(set(t.id for t in result.tokens) <= set(t.id for t in self.tokens))
        self.assertTrue(all(t.position == int(t.id[1:]) * 2 for t in result.tokens))
        self.assertEqual(result.to_dict()['minimality'], '1-minimal under single-token deletion')
        for i in range(len(result.tokens)):
            smaller = result.tokens[:i] + result.tokens[i+1:]
            from attnwitness import audit
            self.assertFalse(any(f.category == 'forbidden_edge' for f in
                                 audit(leaking_builder(smaller), Policy(smaller)).findings))

    def test_budget_exhaustion_keeps_a_verified_witness_but_is_not_minimal(self):
        result = reduce_case(self.tokens, leaking_builder, Policy, category='forbidden_edge', max_calls=1)
        self.assertFalse(result.complete)
        self.assertEqual(result.calls, 1)
        self.assertEqual(result.audit.status, 'VIOLATION')
        self.assertIsNone(result.to_dict()['minimality'])

    def test_missing_initial_violation_is_not_successful_reduction(self):
        def correct(tokens):
            return np.tril(np.ones((len(tokens), len(tokens)), dtype=bool))
        with self.assertRaises(ContractError):
            reduce_case(self.tokens, correct, Policy, category='forbidden_edge')

    def test_invalid_budgets_and_builder_failures_do_not_become_pass(self):
        for limit in (0, True, -1):
            with self.subTest(limit=limit), self.assertRaises(ContractError):
                reduce_case(self.tokens, leaking_builder, Policy, category='forbidden_edge', max_calls=limit)
        def raises(tokens):
            raise RuntimeError('a trusted builder failed')
        with self.assertRaises(ContractError):
            reduce_case(self.tokens, raises, Policy, category='forbidden_edge')

    def test_prefix_witness_preserves_original_roles_and_objects(self):
        tokens = [Token('p0', 'prefix-doc', 0, prefix=True),
                  Token('p1', 'prefix-doc', 1, prefix=True), Token('tail', 'prefix-doc', 2)]
        original = {t.id: t for t in tokens}
        candidates = []
        def causal_builder(ts):
            candidates.append(tuple(ts))
            return np.tril(np.ones((len(ts), len(ts)), dtype=bool))
        policy = lambda ts: Policy(ts, mode='prefix')
        r = reduce_case(tokens, causal_builder, policy, category='missing_edge')
        self.assertTrue(r.complete)
        self.assertEqual([t.id for t in r.tokens], ['p0', 'p1'])
        for candidate in candidates + [r.tokens]:
            for token in candidate:
                self.assertIs(token, original[token.id])
        from attnwitness import audit
        checked = audit(causal_builder(r.tokens), policy(r.tokens))
        self.assertTrue(any(f.category == 'missing_edge' and f.query_id == 'p0' and f.key_id == 'p1'
                            for f in checked.findings))

    def test_cross_segment_witness_preserves_original_segments(self):
        tokens = [Token('x', 'A', 0), Token('y', 'B', 2), Token('z', 'B', 4)]
        original = {t.id: t for t in tokens}
        candidates = []
        def causal_builder(ts):
            candidates.append(tuple(ts))
            return np.tril(np.ones((len(ts), len(ts)), dtype=bool))
        policy = lambda ts: Policy(ts, mode='packed')
        r = reduce_case(tokens, causal_builder, policy, category='forbidden_edge')
        self.assertTrue(r.complete)
        self.assertEqual(len(r.tokens), 2)
        self.assertEqual({t.segment for t in r.tokens}, {'A', 'B'})
        for candidate in candidates + [r.tokens]:
            for token in candidate:
                self.assertIs(token, original[token.id])
        from attnwitness import audit
        self.assertTrue(any(f.category == 'forbidden_edge' for f in
                            audit(causal_builder(r.tokens), policy(r.tokens)).findings))

    def test_prefix_reduction_rechecks_declared_semantics(self):
        tokens = [Token('p0', 's', 0, prefix=True), Token('p1', 's', 1, prefix=True),
                  Token('s2', 's', 2), Token('s3', 's', 3)]
        r = reduce_case(tokens, leaking_builder, lambda ts: Policy(ts, mode='prefix'),
                        category='forbidden_edge')
        self.assertTrue(r.complete)
        self.assertTrue(any(not t.prefix for t in r.tokens))
        self.assertEqual(r.audit.status, 'VIOLATION')


if __name__ == '__main__':
    unittest.main()
