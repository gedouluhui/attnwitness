"""The auditor's interpretation must agree with actual CPU attention calls."""
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from attnwitness import Policy, Token, ContractError
from attnwitness.torch_adapter import audit_torch


class TorchTests(unittest.TestCase):
    def setUp(self):
        self.policy = Policy([Token('a', 's', 0), Token('b', 's', 1)])

    def test_sdpa_forbidden_value_perturbation_and_float_encoding(self):
        q = torch.zeros(1, 1, 2, 2)
        value = torch.eye(2).reshape(1, 1, 2, 2)
        changed = value.clone(); changed[0, 0, 1, 1] = 20
        good = torch.tensor([[True, False], [True, True]])
        self.assertEqual(audit_torch(good, self.policy).status, 'PASS')
        a = F.scaled_dot_product_attention(q, q, value, attn_mask=good, dropout_p=0.0)
        b = F.scaled_dot_product_attention(q, q, changed, attn_mask=good, dropout_p=0.0)
        self.assertTrue(torch.equal(a[0, 0, 0], b[0, 0, 0]))
        bad = good.float()
        self.assertEqual(audit_torch(bad, self.policy).status, 'VIOLATION')
        c = F.scaled_dot_product_attention(q, q, value, attn_mask=bad, dropout_p=0.0)
        d = F.scaled_dot_product_attention(q, q, changed, attn_mask=bad, dropout_p=0.0)
        self.assertFalse(torch.equal(c[0, 0, 0], d[0, 0, 0]))

    def test_actual_mha_boolean_mask_matches_audited_encoding(self):
        model = torch.nn.MultiheadAttention(2, 1, bias=False, batch_first=True)
        with torch.no_grad():
            model.in_proj_weight.zero_()
            model.in_proj_weight[4:6].copy_(torch.eye(2))
            model.out_proj.weight.copy_(torch.eye(2))
        x = torch.eye(2)[None]
        mask = torch.tensor([[False, True], [False, False]])
        self.assertEqual(audit_torch(mask, self.policy, backend='mha').status, 'PASS')
        result, _ = model(x, x, x, attn_mask=mask, is_causal=False)
        self.assertTrue(torch.equal(result[0, 0], torch.tensor([1., 0.])))
        wrong, _ = model(x, x, x, attn_mask=~mask, is_causal=False)
        self.assertTrue(torch.equal(wrong[0, 0], torch.tensor([0., 1.])))
        self.assertEqual(audit_torch(~mask, self.policy, backend='mha').status, 'VIOLATION')

    def test_tensor_inputs_and_grad_metadata_are_unchanged(self):
        mask = torch.tensor([[0., -float('inf')], [0., 0.]], requires_grad=True)
        before = mask.detach().clone()
        self.assertEqual(audit_torch(mask, self.policy).status, 'PASS')
        self.assertTrue(torch.equal(mask, before))
        self.assertTrue(mask.requires_grad)
        self.assertIsNone(mask.grad)

    def test_padding_tensor_and_unsupported_tensor_are_handled(self):
        p = Policy([Token('a', 's', 0), Token('pad', 's', 1, query_valid=False, key_valid=False)])
        r = audit_torch(None, p, backend='mha', key_padding_mask=torch.tensor([[False, True]]))
        self.assertEqual(r.status, 'PASS')
        with self.assertRaises(ContractError):
            audit_torch(torch.ones(2, 2, dtype=torch.int64), self.policy)
        with self.assertRaises(ContractError):
            audit_torch(np.ones((2, 2), dtype=bool), self.policy)

    def test_oversized_tensor_is_rejected_before_numpy_conversion(self):
        from unittest.mock import patch
        large = torch.ones(1).expand(1_048_576, 2, 2)
        with patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('conversion before budget')):
            with self.assertRaises(ContractError):
                audit_torch(large, self.policy, max_cells=4)


if __name__ == '__main__':
    unittest.main()
