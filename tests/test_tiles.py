"""Independent scalar truth and bounded-allocation checks for exact tiled audits."""
import unittest
from unittest.mock import patch
import numpy as np
from attnwitness import Token, Policy, MatrixContract, ContractError, audit


class TileTests(unittest.TestCase):
    def test_tile_sizes_preserve_scalar_counts_and_global_witness_order(self):
        rng = np.random.default_rng(241)
        for length in (2, 7, 17):
            tokens = [[Token(f'b{b}t{i}', str(i // 4), i % 4) for i in range(length)] for b in range(2)]
            mask = rng.random((2, 3, length, length)) > .35
            expected = []; counts = {'forbidden_edge': 0, 'missing_edge': 0, 'empty_query': 0}
            for category in counts:
                for b in range(2):
                    for h in range(3):
                        for q in range(length):
                            if category == 'empty_query':
                                if not any(mask[b, h, q, k] for k in range(length)):
                                    expected.append((category, b, h, q, None)); counts[category] += 1
                                continue
                            for k in range(length):
                                wanted = q // 4 == k // 4 and k % 4 <= q % 4
                                bad = (category == 'forbidden_edge' and not wanted and mask[b, h, q, k]) or (
                                    category == 'missing_edge' and wanted and not mask[b, h, q, k])
                                if bad:
                                    expected.append((category, b, h, q, k)); counts[category] += 1
            for size in (1, 3, 16):
                with self.subTest(length=length, tile=size):
                    result = audit(mask, Policy(tokens, mode='packed'), batch_size=2, heads=3,
                                   max_findings=20, tile_size=size)
                    actual = [(f.category, f.batch, f.head, f.query_index, f.key_index) for f in result.findings]
                    self.assertEqual(actual, expected[:20])
                    self.assertEqual(result.violation_count, len(expected))
                    self.assertEqual(result.to_dict()['counts'], counts)

    def test_late_witness_has_its_own_batch_head_context(self):
        n = 24; tokens = [[Token(f'b{b}t{i}', 'doc', i) for i in range(n)] for b in range(2)]
        mask = np.broadcast_to(np.tril(np.ones((n, n), dtype=bool)), (2, 2, n, n)).copy()
        mask[1, 1, 20, 21] = True
        result = audit(mask, Policy(tokens), batch_size=2, heads=2, tile_size=3)
        window = result.to_dict()['evidence'][0]
        self.assertEqual((window['batch'], window['head']), (1, 1))
        qi = window['query_indices'].index(20); ki = window['key_indices'].index(21)
        self.assertEqual(window['query_ids'][qi], 'b1t20')
        self.assertEqual(window['key_ids'][ki], 'b1t21')
        self.assertTrue(window['visible'][qi][ki])
        self.assertEqual(window['expected'][qi][ki], -1)

    def test_evidence_limit_does_not_limit_detection_counts(self):
        p = Policy([Token(str(i), 's', i) for i in range(16)])
        r = audit(None, p, max_evidence=2, max_findings=5, tile_size=3)
        self.assertEqual(r.violation_count, 120)
        self.assertEqual(len(r.to_dict()['evidence']), 2)
        self.assertTrue(r.to_dict()['evidence_truncated'])
        self.assertEqual(r.coverage['checked_cells'], 256)

    def test_broadcast_workload_does_not_create_full_relation_temporaries(self):
        import attnwitness.masks as masks
        n = 512; p = Policy([Token(str(i), 's', i) for i in range(n)])
        mask = np.broadcast_to(np.array(0, dtype=np.float32), (1, 8, n, n))
        sizes = []; original = masks._visible
        def measured(array, backend):
            sizes.append(array.size)
            self.assertLessEqual(array.size, 32 ** 2)
            return original(array, backend)
        with patch.object(Policy, 'expand', side_effect=AssertionError('full policy expansion')), patch.object(
                masks, '_visible', side_effect=measured):
            r = audit(mask, p, heads=8, max_cells=8*n*n, tile_size=32, max_evidence=0)
        self.assertEqual(r.violation_count, 8*n*(n-1)//2)
        self.assertTrue(sizes)
        self.assertLessEqual(r.coverage['largest_tile_cells'], 32 ** 2)

    def test_empty_query_window_carries_full_row_evidence(self):
        p = MatrixContract(np.zeros((9, 9), dtype=np.int8), list(map(str, range(9))), require_nonempty=True)
        r = audit(np.zeros((9, 9), dtype=bool), p, tile_size=2, max_evidence=1)
        self.assertEqual(r.to_dict()['evidence'][0]['row_summary'], {'keys_checked': 9, 'visible_keys': 0})

    def test_invalid_late_tile_and_bad_budgets_cannot_pass(self):
        p = MatrixContract(np.ones((8, 8), dtype=np.int8), list(map(str, range(8))))
        mask = np.zeros((8, 8), dtype=np.float32); mask[-1, -1] = np.nan
        with self.assertRaises(ContractError):
            audit(mask, p, tile_size=2)
        for args in ({'tile_size': True}, {'tile_size': 0}, {'max_evidence': -1}, {'window_radius': 20}):
            with self.subTest(args=args), self.assertRaises(ContractError):
                audit(None, p, **args)


if __name__ == '__main__':
    unittest.main()
