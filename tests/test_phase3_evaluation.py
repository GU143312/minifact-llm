"""The scorer must not accept replacement gold with identical model inputs."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_phase3 import main
from scripts.run_phase3 import sha, write


class ScoringGoldBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = {'samples': [{'id': 'a', 'text': 'Original claim',
                                     'gold': {'verdict': 'REFUTED'}}]}
        self.inputs = [{'id': 'a', 'text': 'Original claim'}]
        write(self.root / 'canonical.json', self.dataset)
        self.run = {'contract': {'files_sha256': {
            'data/phase3/dataset.json': sha(self.root / 'canonical.json')}}}
        write(self.root / 'run.json', self.run)
        write(self.root / 'freeze.json', {})
        write(self.root / 'inputs.json', self.inputs)

    def invoke(self, dataset):
        write(self.root / 'scoring.json', dataset)
        argv = ['evaluate_phase3', '--run', str(self.root / 'run.json'),
                '--freeze', str(self.root / 'freeze.json'), '--inputs', str(self.root / 'inputs.json'),
                '--dataset', str(self.root / 'scoring.json'), '--output', str(self.root / 'metrics.json')]
        with patch.object(sys, 'argv', argv), patch('scripts.evaluate_phase3.audit', return_value={'passed': True}):
            main()

    def test_changed_gold_with_identical_id_and_text_fails_before_scoring(self):
        changed = copy.deepcopy(self.dataset)
        changed['samples'][0]['gold']['verdict'] = 'SUPPORTED'
        with patch('scripts.evaluate_phase3.summarize') as scorer:
            with self.assertRaisesRegex(ValueError, 'Scoring labels differ from frozen dataset bytes'):
                self.invoke(changed)
            scorer.assert_not_called()
        self.assertFalse((self.root / 'metrics.json').exists())

    def test_identical_frozen_gold_at_another_path_is_allowed(self):
        with patch('scripts.evaluate_phase3.summarize', return_value={'verified': True}) as scorer:
            self.invoke(self.dataset)
            scorer.assert_called_once()
        self.assertTrue((self.root / 'metrics.json').exists())

    def test_frozen_gold_still_must_project_exactly_to_model_inputs(self):
        write(self.root / 'inputs.json', [{'id': 'a', 'text': 'Different claim'}])
        with patch('scripts.evaluate_phase3.summarize') as scorer:
            with self.assertRaisesRegex(ValueError, 'Dataset does not reproduce label-free inputs'):
                self.invoke(self.dataset)
            scorer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
