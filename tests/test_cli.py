import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def call(self, *args):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONIOENCODING='utf-8')
        return subprocess.run([sys.executable, '-m', 'attnwitness', *map(str, args)],
                              env=env, capture_output=True, text=True, encoding='utf-8', timeout=15)

    def test_instantiable_demo_commands_write_real_reports(self):
        with tempfile.TemporaryDirectory() as d:
            for case, code, status in [('causal', 0, 'PASS'), ('float-mask', 1, 'VIOLATION'),
                                       ('packed', 1, 'VIOLATION'), ('prefix', 0, 'PASS')]:
                with self.subTest(case=case):
                    result = self.call('demo', '--case', case, '--json', Path(d) / (case + '.json'),
                                       '--html', Path(d) / (case + '.html'))
                    self.assertEqual(result.returncode, code, result.stderr)
                    data = json.loads((Path(d) / (case + '.json')).read_text(encoding='utf-8'))
                    self.assertEqual(data['status'], status)
                    self.assertIn('AttnWitness', (Path(d) / (case + '.html')).read_text(encoding='utf-8'))

    def test_explicit_json_contract_and_npy_mask_are_audited(self):
        with tempfile.TemporaryDirectory() as d:
            spec = Path(d) / 'spec.json'; mask = Path(d) / 'mask.npy'
            spec.write_text(json.dumps({'tokens': [{'id': 'x', 'segment': 's', 'position': 0}]}), encoding='utf-8')
            np.save(mask, np.array([[True]]))
            result = self.call('check', spec, mask)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'], 'PASS')

    def test_inputs_and_existing_output_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            spec = Path(d) / 'spec.json'; mask = Path(d) / 'mask.npy'; existing = Path(d) / 'saved.json'
            content = json.dumps({'tokens': [{'id': 'x', 'segment': 's', 'position': 0}]})
            spec.write_text(content, encoding='utf-8'); np.save(mask, np.array([[True]]))
            self.assertEqual(self.call('check', spec, mask, '--json', spec, '--force').returncode, 2)
            self.assertEqual(spec.read_text(encoding='utf-8'), content)
            existing.write_text('keep', encoding='utf-8')
            self.assertEqual(self.call('demo', '--json', existing).returncode, 2)
            self.assertEqual(existing.read_text(encoding='utf-8'), 'keep')

    def test_unrepresentable_cli_budget_is_an_input_error_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            spec = Path(d) / 'spec.json'; mask = Path(d) / 'mask.npy'
            spec.write_text(json.dumps({'tokens': [{'id': 'x', 'segment': 's', 'position': 0}]}), encoding='utf-8')
            np.save(mask, np.array([[True]]))
            result = self.call('check', spec, mask, '--max-cells', str(10 ** 100))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'], 'ERROR')

    def test_fingerprint_binds_the_bytes_that_were_actually_loaded(self):
        import hashlib
        from unittest.mock import patch
        from attnwitness.cli import _load_mask
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'mask.npy'
            np.save(path, np.array([[True]]))
            original = path.read_bytes()
            real_load = np.load
            def load_then_change(*args, **kwargs):
                value = real_load(*args, **kwargs)
                np.save(path, np.array([[False]]))
                return value
            with patch('attnwitness.cli.np.load', side_effect=load_then_change):
                value, digest = _load_mask(path, 4)
            self.assertTrue(value[0, 0])
            self.assertEqual(digest, hashlib.sha256(original).hexdigest())

    def test_object_npy_and_unknown_spec_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            spec = Path(d) / 'spec.json'; mask = Path(d) / 'mask.npy'
            spec.write_text(json.dumps({'tokens': [{'id': 'x', 'segment': 's', 'position': 0}]}), encoding='utf-8')
            np.save(mask, np.array([[object()]], dtype=object))
            self.assertEqual(self.call('check', spec, mask).returncode, 2)
            np.save(mask, np.array([[True]]))
            spec.write_text(json.dumps({'tokens': [{'id': 'x', 'segment': 's', 'position': 0}],
                                        'silently_ignore': True}), encoding='utf-8')
            self.assertEqual(self.call('check', spec, mask).returncode, 2)


if __name__ == '__main__':
    unittest.main()
