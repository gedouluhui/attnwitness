import unittest
import numpy as np
from attnwitness import Token, Policy, MatrixContract, audit
from attnwitness.report import render_report


def late_case():
    n = 24
    tokens = [[Token(f'b{b}t{i}', 'doc', i) for i in range(n)] for b in range(2)]
    mask = np.broadcast_to(np.tril(np.ones((n, n), dtype=bool)), (2, 2, n, n)).copy()
    mask[1, 1, 20, 21] = True
    return audit(mask, Policy(tokens), batch_size=2, heads=2, tile_size=8)


class ReportTests(unittest.TestCase):
    def test_untrusted_ids_cannot_become_html_or_script(self):
        name = '</script><img src=x onerror=alert(1)>'
        result = audit(np.ones((2, 2), dtype=bool), Policy([Token(name, 's', 0), Token('b', 's', 1)]))
        html = render_report(result)
        self.assertNotIn('<img src=x', html)
        self.assertIn('&lt;img', html)
        self.assertIn('VIOLATION', html)
        self.assertIn('Content-Security-Policy', html)
        self.assertNotIn('<script src=', html)
        self.assertNotIn('<link ', html)

    def test_empty_success_and_error_reports_are_readable(self):
        success = audit(np.ones((1, 1), dtype=bool), Policy([Token('a', 's', 0)]))
        self.assertIn('PASS', render_report(success))
        rejected = render_report({'status': 'ERROR', 'error': 'Unsupported mask'})
        self.assertIn('ERROR', rejected)
        self.assertNotIn('No declared relation violation was found', rejected)

    def test_dictionary_metadata_is_escaped_too(self):
        payload = '<img src=x onerror=alert(1)>'
        result = audit(np.ones((2, 2), dtype=bool), Policy([Token('a', 's', 0), Token('b', 's', 1)])).to_dict()
        result['findings'][0]['batch'] = payload
        result['violation_count'] = payload
        self.assertNotIn(payload, render_report(result))

    def test_retained_witness_outside_preview_has_its_own_context(self):
        result = late_case()
        window = result.to_dict()['evidence'][0]
        self.assertEqual((window['batch'], window['head']), (1, 1))
        html = render_report(result)
        self.assertIn('data-witness="0"', html)
        self.assertIn('aria-controls="witness-0"', html)
        self.assertIn('B1', html)
        self.assertIn('H1', html)
        panel = html.split('id="witness-0"', 1)[1].split('</article>', 1)[0]
        self.assertIn('b1t20', panel)
        self.assertIn('b1t21', panel)
        self.assertIn('>20<', panel)
        self.assertIn('>21<', panel)
        self.assertIn('must block', panel)
        preview = html.split('class="matrix-wrap"', 1)[1].split('</section>', 1)[0]
        self.assertNotIn('b1t20', preview)
        self.assertIn('witness panel', panel.lower())

    def test_witness_panel_escapes_ids_and_explains_bias_semantics(self):
        n = 24
        payload = '</script><img src=x onerror=alert(1)>'
        tokens = [[Token(f'b{b}t{i}', 'doc', i) for i in range(n)] for b in range(2)]
        tokens[1][20] = Token(payload, 'doc', 20)
        mask = np.broadcast_to(np.tril(np.ones((n, n), dtype=bool)), (2, 2, n, n)).copy()
        mask[1, 1, 20, 21] = True
        html = render_report(audit(mask, Policy(tokens), batch_size=2, heads=2, tile_size=8))
        panel = html.split('id="witness-0"', 1)[1].split('</article>', 1)[0]
        self.assertNotIn('<img src=x', html)
        self.assertIn('&lt;img', panel)
        self.assertIn('only <code>-inf</code>', panel)
        self.assertIn('summed in the mask dtype', panel)

    def test_partial_declaration_and_filter_counts_are_explicit(self):
        cells = np.array([[1, -1, 0], [1, 1, -1], [1, 1, 1]], dtype=np.int8)
        policy = MatrixContract(cells, ['a', 'b', 'c'])
        mask = np.array([[True, True, False], [False, True, False], [True, True, True]])
        result = audit(mask, policy)
        data = result.to_dict()
        self.assertEqual(data['coverage']['constrained_cells'], 8)
        self.assertFalse(data['coverage']['complete_relation_contract'])
        html = render_report(result)
        self.assertIn('Partial declaration', html)
        self.assertIn('8 of 9', html)
        self.assertIn('1 unconstrained', html)
        for category, count in [('forbidden_edge', 1), ('missing_edge', 1), ('empty_query', 0)]:
            self.assertIn(f'data-filter="{category}" data-count="{count}"', html)
        self.assertIn('data-filter="all" data-count="2"', html)
        self.assertIn('not asserted', html)

    def test_evidence_and_finding_truncation_are_disclosed_separately(self):
        policy = Policy([Token(str(i), 's', i) for i in range(16)])
        html = render_report(audit(None, policy, max_evidence=2, max_findings=5))
        self.assertIn('Context windows are retained for the first', html)
        self.assertIn('Finding list truncated', html)
        self.assertIn('120', html)


if __name__ == '__main__':
    unittest.main()
