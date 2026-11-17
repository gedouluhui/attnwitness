"""Named silent bugs must be caught by actual behavioral tests in copies."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MutationTests(unittest.TestCase):
    def check_mutation(self, module, old, new, test_file, test_name):
        with tempfile.TemporaryDirectory(prefix='attnwitness-mutation-') as directory:
            copied = Path(directory)
            shutil.copytree(ROOT / 'src', copied / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
            (copied / 'tests').mkdir()
            shutil.copy2(ROOT / 'tests' / test_file, copied / 'tests' / test_file)
            env = dict(os.environ, PYTHONPATH=str(copied / 'src'), PYTHONIOENCODING='utf-8')
            command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', test_file,
                       '-k', test_name, '-v']
            baseline = subprocess.run(command, cwd=copied, env=env, capture_output=True, text=True,
                                      encoding='utf-8', timeout=30)
            self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
            target = copied / 'src' / 'attnwitness' / module
            source = target.read_text(encoding='utf-8')
            self.assertEqual(source.count(old), 1, 'mutation must have exactly one anchor')
            changed = source.replace(old, new, 1)
            self.assertNotEqual(changed, source, 'a no-op is not a mutation')
            target.write_text(changed, encoding='utf-8', newline='\n')
            # Avoid stale .pyc when replacement has the same size and timestamp granularity.
            for cache in (copied / 'src').rglob('__pycache__'):
                shutil.rmtree(cache)
            mutated = subprocess.run(command, cwd=copied, env=env, capture_output=True, text=True,
                                     encoding='utf-8', timeout=30)
            output = mutated.stdout + mutated.stderr
            self.assertNotEqual(mutated.returncode, 0, 'silent mutation escaped')
            self.assertRegex(output, rf'(?m)^FAIL: {re.escape(test_name)}\b')
            self.assertNotIn('FAILED (errors=', output, 'a crash is not the expected assertion failure')

    def test_inverted_sdpa_boolean_semantics_is_caught(self):
        self.check_mutation('masks.py', "return array if backend == 'sdpa' else ~array",
                            "return ~array if backend == 'sdpa' else ~array",
                            'test_core.py', 'test_sdpa_polarity_and_float_conversion')

    def test_finite_negative_bias_as_hard_block_is_caught(self):
        self.check_mutation('masks.py', 'return ~np.isneginf(array)', 'return array >= 0',
                            'test_core.py', 'test_finite_negative_bias_is_not_a_hard_block')

    def test_cleared_prefix_metadata_is_caught(self):
        self.check_mutation('reduce.py', 'candidate = current[:index] + current[index + 1:]',
                            "candidate = tuple(__import__('dataclasses').replace(t, prefix=False) "
                            "for t in current[:index] + current[index + 1:])",
                            'test_reduce.py', 'test_prefix_witness_preserves_original_roles_and_objects')

    def test_collapsed_segment_metadata_is_caught(self):
        self.check_mutation('reduce.py', 'candidate = current[:index] + current[index + 1:]',
                            "candidate = tuple(__import__('dataclasses').replace(t, segment='s') "
                            "for t in current[:index] + current[index + 1:])",
                            'test_reduce.py', 'test_cross_segment_witness_preserves_original_segments')

    def test_budget_exhaustion_claiming_minimality_is_caught(self):
        self.check_mutation('reduce.py', 'ReductionResult(current, best, category, calls, False)',
                            'ReductionResult(current, best, category, calls, True)',
                            'test_reduce.py', 'test_budget_exhaustion_keeps_a_verified_witness_but_is_not_minimal')


if __name__ == '__main__':
    unittest.main()
