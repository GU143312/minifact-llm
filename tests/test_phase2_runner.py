"""Check input isolation and checkpoint safety without loading a model."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_phase2 import prepare_output, samples_from_file


class Phase2RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def test_developer_gold_is_not_passed_as_model_input(self):
        path = self.directory / 'dev.json'
        path.write_text(json.dumps({'samples': [{'id': 'D01', 'source': 'A source',
                                                'facts': ['secret expected fact'],
                                                'logical_checks': ['gold answer']}]}), encoding='utf-8')
        self.assertEqual(samples_from_file(path, 'dev'), [{'id': 'D01', 'text': 'A source'}])

    def test_resume_requires_exact_contract_and_completed_prefix(self):
        path = self.directory / 'checkpoint.json'
        cases = [{'id': 'a', 'text': 'one'}, {'id': 'b', 'text': 'two'}]
        contract = {'hash': 'fixed'}
        path.write_text(json.dumps({'completed': False, 'contract': contract,
                                    'results': [{'id': 'a'}]}), encoding='utf-8')
        self.assertEqual(len(prepare_output(path, contract, cases, True)['results']), 1)
        with self.assertRaisesRegex(ValueError, 'protocol'):
            prepare_output(path, {'hash': 'changed'}, cases, True)
        with self.assertRaisesRegex(ValueError, 'prefix'):
            prepare_output(path, contract, list(reversed(cases)), True)

    def test_prevents_overwrite_and_resuming_complete_result(self):
        path = self.directory / 'checkpoint.json'
        path.write_text(json.dumps({'completed': True, 'contract': {}, 'results': []}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'exists'):
            prepare_output(path, {}, [], False)
        with self.assertRaisesRegex(ValueError, 'already complete'):
            prepare_output(path, {}, [], True)


if __name__ == '__main__':
    unittest.main()
