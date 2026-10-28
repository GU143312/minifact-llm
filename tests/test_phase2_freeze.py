import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_phase2_freeze import ROOT, audit, digest


class FreezeAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        directory = Path(self.tmp.name)
        self.dev_input = directory / 'dev.json'
        self.formal_input = directory / 'formal.json'
        self.dev_run = directory / 'dev_run.json'
        self.dev_input.write_text(json.dumps({'samples': [{'id': 'D1', 'source': 'Dev text.'}]}), encoding='utf-8')
        self.formal_input.write_text(json.dumps([{'id': 'F1', 'text': 'First.'}, {'id': 'F2', 'text': 'Second.'}]), encoding='utf-8')
        self.freeze = json.loads((ROOT / 'reports/phase2_frozen_protocol.json').read_text(encoding='utf-8'))
        self.freeze['developer_input_sha256'] = digest(self.dev_input)
        self.freeze['formal_input_sha256'] = digest(self.formal_input)
        self.freeze['frozen_at_utc'] = '2026-10-29T00:00:00+00:00'
        common = {'protocol': self.freeze['protocol'], 'max_rounds': self.freeze['max_rounds'],
                  'files_sha256': self.freeze['files_sha256'], 'model': self.freeze['model_config'],
                  'evidence': self.freeze['evidence'], 'roles': 'same model', 'verdict_scope': 'claim'}
        execution = {'actual_revision': self.freeze['model_config']['model_revision'],
                     'actual_dtype': 'torch.float16',
                     'attention_implementation': self.freeze['model_config']['attention_implementation']}
        self.dev = {'created_at_utc': '2026-10-28T22:00:00+00:00',
                    'completed_at_utc': '2026-10-28T23:00:00+00:00', 'completed': True,
                    'contract': {**common, 'input_kind': 'dev', 'input_sha256': digest(self.dev_input)},
                    'execution': execution, 'results': [{'id': 'D1', 'source': 'Dev text.'}]}
        self.dev_run.write_text(json.dumps(self.dev), encoding='utf-8')
        self.freeze['developer_run_sha256'] = digest(self.dev_run)
        self.formal = {'created_at_utc': '2026-10-29T00:00:01+00:00', 'completed': False,
                       'contract': {**copy.deepcopy(common), 'input_kind': 'formal',
                                    'input_sha256': digest(self.formal_input)},
                       'execution': copy.deepcopy(execution), 'results': [{'id': 'F1', 'source': 'First.'}]}

    def check(self):
        return audit(self.freeze, self.dev, self.formal, self.dev_run,
                     self.dev_input, self.formal_input)

    def test_partial_and_complete_prefix(self):
        result = self.check()
        self.assertEqual((result['status'], result['formal_samples_processed'], result['formal_samples_unprocessed']),
                         ('partial', 1, 1))
        self.formal['results'].append({'id': 'F2', 'source': 'Second.'})
        self.formal['completed'] = True
        self.assertEqual(self.check()['status'], 'complete')

    def test_equal_or_earlier_formal_timestamp_is_rejected(self):
        for time in ['2026-10-29T00:00:00+00:00', '2026-10-28T23:59:59+00:00', '2026-10-29T00:00:01']:
            with self.subTest(time=time):
                self.formal['created_at_utc'] = time
                with self.assertRaises(ValueError):
                    self.check()

    def test_developer_completion_after_freeze_is_rejected(self):
        self.dev['completed_at_utc'] = '2026-10-29T00:00:01+00:00'
        self.dev_run.write_text(json.dumps(self.dev), encoding='utf-8')
        self.freeze['developer_run_sha256'] = digest(self.dev_run)
        with self.assertRaisesRegex(ValueError, 'developer completion'):
            self.check()

    def test_input_and_developer_run_hash_mismatch_are_rejected(self):
        self.freeze['developer_run_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'developer run hash mismatch'):
            self.check()
        self.freeze['developer_run_sha256'] = digest(self.dev_run)
        self.freeze['formal_input_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'formal input hash mismatch'):
            self.check()

    def test_changed_protocol_and_incomplete_prefix_are_rejected(self):
        self.formal['contract']['max_rounds'] = 1
        with self.assertRaisesRegex(ValueError, 'max_rounds mismatch'):
            self.check()
        self.formal['contract']['max_rounds'] = self.freeze['max_rounds']
        self.formal['results'][0]['source'] = 'Wrong.'
        with self.assertRaisesRegex(ValueError, 'result prefix'):
            self.check()


if __name__ == '__main__':
    unittest.main()
