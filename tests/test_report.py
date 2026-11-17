import unittest
import numpy as np
from attnwitness import Token, Policy, audit
from attnwitness.report import render_report


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


if __name__ == '__main__':
    unittest.main()
