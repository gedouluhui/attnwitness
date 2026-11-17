"""Independent small contracts: polarity, bias, partial scope and identity errors."""
import itertools
import unittest
import numpy as np
from attnwitness import Token, Policy, MatrixContract, audit, ContractError


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tokens = [Token('a', 's', 0), Token('b', 's', 1)]
        self.policy = Policy(self.tokens)

    def test_sdpa_polarity_and_float_conversion(self):
        mask = np.array([[True, False], [True, True]])
        self.assertEqual(audit(mask, self.policy).status, 'PASS')
        result = audit(mask.astype(np.float32), self.policy)
        self.assertEqual(result.status, 'VIOLATION')
        self.assertEqual([(f.category, f.query_id, f.key_id) for f in result.findings],
                         [('forbidden_edge', 'a', 'b')])

    def test_mha_uses_opposite_boolean_encoding(self):
        mask = np.array([[False, True], [False, False]])
        self.assertEqual(audit(mask, self.policy, backend='mha').status, 'PASS')
        self.assertEqual(audit(~mask, self.policy, backend='mha').status, 'VIOLATION')

    def test_finite_negative_bias_is_not_a_hard_block(self):
        for value in (-1, -1e9, -np.finfo(np.float32).max):
            with self.subTest(value=value):
                self.assertEqual(audit(np.array([[0., value], [0., 0.]]), self.policy).status,
                                 'VIOLATION')
        self.assertEqual(audit(np.array([[0., -np.inf], [0., 0.]]), self.policy).status, 'PASS')

    def test_missing_required_edge_is_detected(self):
        result = audit(np.eye(2, dtype=bool), self.policy)
        self.assertIn(('missing_edge', 'b', 'a'),
                      [(f.category, f.query_id, f.key_id) for f in result.findings])

    def test_packed_sequences_are_not_one_causal_sequence(self):
        p = Policy([Token('x', 'one', 0), Token('y', 'two', 0)], mode='packed')
        self.assertEqual(audit(np.eye(2, dtype=bool), p).status, 'PASS')
        result = audit(np.array([[1, 0], [1, 1]], dtype=bool), p)
        self.assertEqual([(f.query_id, f.key_id) for f in result.findings], [('y', 'x')])

    def test_prefix_bidirectionality_is_legitimate(self):
        p = Policy([Token('p0', 's', 0, prefix=True), Token('p1', 's', 1, prefix=True),
                    Token('tail', 's', 2)], mode='prefix')
        valid = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=bool)
        self.assertEqual(audit(valid, p).status, 'PASS')
        valid[0, 2] = True
        self.assertEqual([(f.query_id, f.key_id) for f in audit(valid, p).findings], [('p0', 'tail')])

    def test_padding_queries_are_not_required_to_be_nonempty(self):
        p = Policy([Token('x', 's', 0), Token('pad', 's', 1, query_valid=False, key_valid=False)])
        self.assertEqual(audit(np.array([[1, 0], [0, 0]], dtype=bool), p).status, 'PASS')
        r = audit(np.zeros((2, 2), dtype=bool), self.policy)
        self.assertEqual(sum(f.category == 'empty_query' for f in r.findings), 2)

    def test_partial_contract_discloses_coverage(self):
        c = MatrixContract(np.array([[0, -1], [0, 0]], dtype=np.int8), ['a', 'b'])
        r = audit(np.zeros((2, 2), dtype=bool), c)
        self.assertEqual(r.status, 'PASS')
        self.assertEqual(r.coverage['constrained_cells'], 1)
        self.assertFalse(r.coverage['complete_relation_contract'])
        self.assertIn('declared', r.to_dict()['scope'])

    def test_heads_broadcast_and_witness_identity(self):
        masks = np.array([[[1, 0], [1, 1]], [[1, 1], [1, 1]]], dtype=bool)
        r = audit(masks, self.policy, heads=2)
        self.assertEqual([(f.head, f.query_id, f.key_id) for f in r.findings], [(1, 'a', 'b')])
        self.assertEqual(audit(masks.reshape(2, 2, 2), self.policy, heads=2, backend='mha').status,
                         'VIOLATION')

    def test_batch_metadata_cannot_be_silently_broadcast(self):
        with self.assertRaises(ContractError):
            audit(np.ones((2, 2), dtype=bool), self.policy, batch_size=2)
        p = Policy([self.tokens, [Token('c', 's', 0), Token('d', 's', 1)]])
        r = audit(None, p, batch_size=2)
        self.assertEqual([(f.batch, f.query_id, f.key_id) for f in r.findings], [(0, 'a', 'b'), (1, 'c', 'd')])

    def test_mha_key_padding_is_combined_not_ignored(self):
        p = Policy([Token('a', 's', 0), Token('pad', 's', 1, key_valid=False, query_valid=False)])
        r = audit(np.zeros((2, 2), dtype=bool), p, backend='mha',
                  key_padding_mask=np.array([[False, True]]))
        self.assertEqual(r.status, 'PASS')

    def test_flags_unknown_backends_and_integer_masks_rejected(self):
        self.assertEqual(audit(None, self.policy, is_causal=True).status, 'PASS')
        for kw in ({'backend': 'mha', 'is_causal': True}, {'backend': 'mystery'},
                   {'is_causal': 1}, {'heads': True}, {'max_cells': 0}):
            with self.subTest(kw=kw), self.assertRaises(ContractError):
                audit(None, self.policy, **kw)
        with self.assertRaises(ContractError):
            audit(np.ones((2, 2), dtype=bool), self.policy, is_causal=True)
        with self.assertRaises(ContractError):
            audit(np.ones((2, 2), dtype=np.int64), self.policy)

    def test_budget_shape_and_nonfinite_are_fail_closed(self):
        for mask in (np.ones((3, 3), dtype=bool), np.array([[0., np.nan], [0., 0.]]),
                     np.array([[0., np.inf], [0., 0.]])):
            with self.subTest(mask=repr(mask)), self.assertRaises(ContractError):
                audit(mask, self.policy)
        with self.assertRaises(ContractError):
            audit(None, self.policy, heads=1_000_000, max_cells=10)

    def test_bad_metadata_and_contract_values_are_rejected(self):
        cases = [[('a', 's', 0), ('a', 's', 1)],
                 [('a', 's', 0), ('b', 's', 0)],
                 [('a', 's', True)], [('', 's', 0)]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ContractError):
                audit(None, Policy([Token(*row) for row in rows]))
        with self.assertRaises(ContractError):
            audit(None, MatrixContract(np.array([[2]]), ['a']))
        with self.assertRaises(ContractError):
            audit(None, Policy([Token('a', 's', 0), Token('b', 's', 1, prefix=True)], mode='prefix'))

    def test_witness_cap_does_not_cap_total_violations(self):
        r = audit(np.zeros((2, 2), dtype=bool), self.policy, max_findings=1)
        self.assertEqual(len(r.findings), 1)
        self.assertEqual(r.violation_count, 5)
        self.assertTrue(r.to_dict()['findings_truncated'])

    def test_missing_data_marks_are_rejected_at_all_array_inputs(self):
        p = Policy([Token('a', 's', 0)])
        bad_true = np.ma.array([[True]], mask=[[True]])
        bad_false = np.ma.array([[False]], mask=[[True]])
        cases = [lambda: audit(bad_true, p),
                 lambda: audit(bad_false, p, backend='mha'),
                 lambda: audit(None, p, backend='mha', key_padding_mask=bad_false),
                 lambda: audit(np.ones((1, 1), dtype=bool),
                               MatrixContract(np.ma.array([[1]], mask=[[True]]), ['a']))]
        for i, call in enumerate(cases):
            with self.subTest(entry=i), self.assertRaises(ContractError):
                call()

    def test_incompatible_shapes_are_rejected_before_elementwise_scans(self):
        from unittest.mock import patch
        p = Policy([Token('a', 's', 0)])
        big_mask = np.broadcast_to(np.array(0.0), (1_048_576, 1, 1))
        with patch('attnwitness.masks.np.isnan', side_effect=AssertionError('scan before shape gate')):
            with self.assertRaises(ContractError):
                audit(big_mask, p, max_cells=1)
        c = MatrixContract(np.broadcast_to(np.array(1, dtype=np.int8), (1_048_576, 1, 1, 1)), ['a'])
        with patch('attnwitness.contracts.np.isin', side_effect=AssertionError('scan before shape gate')):
            with self.assertRaises(ContractError):
                audit(None, c, max_cells=1)

    def test_conversion_errors_are_domain_errors(self):
        for call in (lambda: audit([[True, False], [True]], self.policy),
                     lambda: MatrixContract(np.array([[1]]), None)):
            with self.assertRaises(ContractError):
                call()

    def test_exact_batch_head_mapping_for_mha(self):
        p = Policy([self.tokens, [Token('c', 's', 0), Token('d', 's', 1)]])
        blocked = np.broadcast_to(np.array([[False, True], [False, False]]), (4, 2, 2)).copy()
        blocked[2, 0, 1] = False
        result = audit(blocked, p, backend='mha', batch_size=2, heads=2)
        self.assertEqual([(f.batch, f.head, f.query_id, f.key_id) for f in result.findings],
                         [(1, 0, 'c', 'd')])

    def test_inputs_are_not_mutated(self):
        mask = np.array([[0., -np.inf], [0., 0.]])
        original = mask.copy()
        audit(mask, self.policy)
        np.testing.assert_array_equal(mask, original)

    def test_exhaustive_two_token_relations_against_hand_literal(self):
        expected = (True, False, True, True)
        for values in itertools.product((False, True), repeat=4):
            with self.subTest(values=values):
                r = audit(np.array(values).reshape(2, 2), self.policy)
                self.assertEqual(r.status == 'PASS', values == expected)


if __name__ == '__main__':
    unittest.main()
