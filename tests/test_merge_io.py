"""Real backend composition and real filesystem identity regressions."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from attnwitness import MatrixContract, ContractError, audit
from attnwitness.torch_adapter import audit_torch
from attnwitness import cli


class MergeTests(unittest.TestCase):
    def test_float_mask_sum_matches_actual_mha_overflow(self):
        contract = MatrixContract(np.ones((2, 2), dtype=np.int8), ['a', 'b'])
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                value = -torch.finfo(dtype).max * 0.75
                mask = torch.tensor([[0, value], [0, 0]], dtype=dtype)
                padding = mask[:1].clone()
                result = audit_torch(mask, contract, backend='mha', key_padding_mask=padding)
                self.assertEqual(result.status, 'VIOLATION')
                self.assertTrue(any(f.category == 'missing_edge' and f.query_id == 'a' and f.key_id == 'b'
                                    for f in result.findings))
                equivalent = audit_torch(mask + padding, contract, backend='mha')
                self.assertEqual(result.to_dict()['findings'], equivalent.to_dict()['findings'])
                model = torch.nn.MultiheadAttention(2, 1, bias=False, batch_first=True, dtype=dtype)
                with torch.no_grad():
                    model.in_proj_weight.zero_(); model.in_proj_weight[4:].copy_(torch.eye(2))
                    model.out_proj.weight.copy_(torch.eye(2))
                x = torch.eye(2, dtype=dtype)[None]
                _, weights = model(x, x, x, attn_mask=mask, key_padding_mask=padding)
                self.assertEqual(weights[0, 0].tolist(), [1., 0.])

    def test_finite_nonoverflow_sum_is_not_a_hard_block(self):
        contract = MatrixContract(np.ones((2, 2), dtype=np.int8), ['a', 'b'])
        for dtype in (np.float16, np.float32, np.float64):
            result = audit(np.full((2, 2), -10, dtype=dtype), contract, backend='mha',
                           key_padding_mask=np.full((1, 2), -20, dtype=dtype))
            self.assertEqual(result.status, 'PASS')

    def test_positive_overflow_is_not_valid_attention(self):
        contract = MatrixContract(np.ones((2, 2), dtype=np.int8), ['a', 'b'])
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype), self.assertRaises(ContractError):
                audit(np.full((2, 2), np.finfo(dtype).max * .75, dtype=dtype), contract, backend='mha',
                      key_padding_mask=np.full((1, 2), np.finfo(dtype).max * .75, dtype=dtype))

    def test_mixed_float_dtype_is_explicitly_unsupported(self):
        contract = MatrixContract(np.ones((2, 2), dtype=np.int8), ['a', 'b'])
        with self.assertRaises(ContractError):
            audit(np.zeros((2, 2), dtype=np.float32), contract, backend='mha',
                  key_padding_mask=np.zeros((1, 2), dtype=np.float16))


class OutputTests(unittest.TestCase):
    def run_cli(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.main(args)

    def test_force_cannot_overwrite_input_through_hard_link(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / 'source.json'; mask = root / 'mask.npy'; output = root / 'output.json'
            text = json.dumps({'tokens': [{'id': 'x', 'segment': 's', 'position': 0}]})
            source.write_text(text, encoding='utf-8'); np.save(mask, np.ones((1, 1), dtype=bool))
            os.link(source, output)
            code = self.run_cli(['check', str(source), str(mask), '--json', str(output), '--force'])
            self.assertEqual(code, 2)
            self.assertEqual(source.read_text(encoding='utf-8'), text)

    def test_late_created_output_is_not_overwritten_without_force(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'report.json'; original = cli.demo
            def late(case, **options):
                output.write_text('another-writer', encoding='utf-8')
                return original(case, **options)
            with patch.object(cli, 'demo', side_effect=late):
                code = self.run_cli(['demo', '--case', 'causal', '--json', str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(output.read_text(encoding='utf-8'), 'another-writer')

    def test_forced_report_replaces_entry_without_truncating_alias(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); original = root / 'other-user-file'; output = root / 'report.json'
            original.write_text('keep-this-inode', encoding='utf-8'); os.link(original, output)
            code = self.run_cli(['demo', '--case', 'causal', '--json', str(output), '--force'])
            self.assertEqual(code, 0)
            self.assertEqual(original.read_text(encoding='utf-8'), 'keep-this-inode')
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['status'], 'PASS')
            self.assertFalse(os.path.samefile(original, output))

    def test_renderer_failure_does_not_leave_half_published_json(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); output = root / 'report.json'; html = root / 'report.html'
            with patch.object(cli, 'render_report', side_effect=ValueError('bad report input')):
                code = self.run_cli(['demo', '--case', 'causal', '--json', str(output), '--html', str(html)])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
            self.assertFalse(html.exists())


if __name__ == '__main__':
    unittest.main()
