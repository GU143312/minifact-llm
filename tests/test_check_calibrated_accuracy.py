"""The calibrated local CLI must preserve its teacher-forced scoring boundary."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.check_accuracy import digest
from scripts.check_calibrated_accuracy import (main, parser, prepare_request,
                                                validate_execution)


class CalibratedLocalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root / 'answer.json'
        self.calibration_path = self.root / 'threshold.json'
        self.contract = {'model': {'model_revision': 'pinned'},
                         'prediction': 'SUPPORTED iff margin >= threshold'}
        self.calibration = {'development_run': 'reports/dev.json',
                            'development_run_sha256': 'development-hash',
                            'holdout_input_sha256': 'protocol-hash',
                            'fit': {'rule': 'Development-only selection', 'threshold': 0.2},
                            'terminator_id': 9,
                            'candidate_token_ids': {'TRUE': [7, 9], 'FALSE': [8, 9]}}
        self.calibration_path.write_text(json.dumps(self.calibration), encoding='utf-8')
        self.execution = {'runtime': {'torch': 'fixed'}, 'actual_dtype': 'torch.float16',
                          'actual_revision': 'pinned', 'attention_implementation': 'eager',
                          'device_map': {'model': 'cuda:0'}, 'model_generation_config': {'eos_token_id': [9]}}
        self.calibration['execution'] = {k: self.execution[k] for k in self.execution if k != 'model_generation_config'}
        self.calibration['execution']['terminator_id'] = 9
        self.backend = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9])),
                                       tokenizer=SimpleNamespace(eos_token_id=9))

    def args(self, *extra):
        return parser().parse_args(['--text', 'A new local statement.',
                                    '--calibration', str(self.calibration_path),
                                    '--output', str(self.output), *extra])

    def prepared(self, statement='A new local statement.'):
        return statement, self.calibration, self.contract, {'calibration_sha256': 'frozen'}

    def test_user_statement_need_not_appear_in_frozen_holdout(self):
        with patch('scripts.check_calibrated_accuracy.code_contract', return_value=self.contract), \
                patch('scripts.check_calibrated_accuracy.check_calibration') as checked:
            statement, _, _, provenance = prepare_request(self.args(), self.root)
        self.assertEqual(statement, 'A new local statement.')
        checked.assert_called_once()
        self.assertEqual(checked.call_args.args[2], self.root / 'data/improvement/holdout_inputs.json')
        self.assertEqual(provenance['calibration_sha256'], digest(self.calibration_path))
        self.assertIn('Protocol binding only', provenance['holdout_hash_role'])
        self.assertFalse((self.root / 'data/improvement/holdout_dataset.json').exists())

    def test_changed_calibration_guard_fails_before_any_model_loading(self):
        with patch('scripts.check_calibrated_accuracy.code_contract', return_value=self.contract), \
                patch('scripts.check_calibrated_accuracy.check_calibration', side_effect=ValueError('changed fit')), \
                patch('scripts.check_calibrated_accuracy.load_backend') as loader:
            with self.assertRaisesRegex(ValueError, 'changed fit'):
                main(['--text', 'A statement.', '--calibration', str(self.calibration_path),
                      '--output', str(self.output)])
            loader.assert_not_called()

    def test_existing_output_fails_without_model_or_calibration_read(self):
        self.output.write_bytes(b'preserve')
        with patch('scripts.check_calibrated_accuracy.minifact.read_json') as read, \
                patch('scripts.check_calibrated_accuracy.load_backend') as loader:
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                main(['--text', 'A statement.', '--output', str(self.output)])
            read.assert_not_called()
            loader.assert_not_called()
        self.assertEqual(self.output.read_bytes(), b'preserve')

    def test_plain_text_unicode_and_crlf_are_preserved(self):
        source = self.root / 'claim.txt'
        source.write_bytes('\ufeff 原文。\r\n'.encode('utf-8'))
        args = parser().parse_args(['--input-file', str(source), '--output', str(self.output),
                                   '--calibration', str(self.calibration_path)])
        with patch('scripts.check_calibrated_accuracy.code_contract', return_value=self.contract), \
                patch('scripts.check_calibrated_accuracy.check_calibration'):
            statement, _, _, provenance = prepare_request(args, self.root)
        self.assertEqual(statement, ' 原文。\r\n')
        self.assertEqual(provenance['input']['file_sha256'], digest(source))

    def test_evidence_method_and_conflicting_inputs_are_not_cli_options(self):
        with patch('sys.stderr'):
            for argv in [[], ['--text', 'x', '--evidence-file', 'qa.json'],
                         ['--text', 'x', '--method', 'evidence'],
                         ['--text', 'x', '--input-file', 'claim.txt']]:
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    parser().parse_args([*argv, '--output', str(self.output)])
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            prepare_request(self.args('--text', ' \t'), self.root)

    def test_runtime_and_terminator_must_match_development_scoring(self):
        self.assertEqual(validate_execution(self.backend, self.execution, self.calibration), 9)
        for changed in [{**self.calibration, 'terminator_id': 8},
                        {**self.calibration, 'execution': {**self.calibration['execution'], 'device_map': {}}}]:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'differs'):
                validate_execution(self.backend, self.execution, changed)

    def test_candidate_tokenization_change_fails_before_forward(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()))
        with patch('scripts.check_calibrated_accuracy.prepare_request', return_value=self.prepared()), \
                patch('scripts.check_calibrated_accuracy.load_backend', return_value=(self.backend, self.execution, torch)), \
                patch('scripts.check_calibrated_accuracy.encode_candidates', return_value=([], [1], {'TRUE': [2, 9], 'FALSE': [8, 9]})), \
                patch('scripts.check_calibrated_accuracy.score_candidates') as score:
            with self.assertRaisesRegex(ValueError, 'token IDs differ'):
                main(['--text', 'A statement.', '--output', str(self.output)])
            score.assert_not_called()

    def test_output_is_explicitly_teacher_forced_and_uses_frozen_threshold(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()))
        messages = [{'role': 'user', 'content': '{"statement":"A new local statement."}'}]
        scoring = {'margin': 0.3, 'forward_calls': 2,
                   'method': 'teacher_forced_complete_binary_candidates',
                   'candidates': [{'label': 'TRUE', 'sequence_logprob': -1.},
                                  {'label': 'FALSE', 'sequence_logprob': -1.3}]}
        with patch('scripts.check_calibrated_accuracy.prepare_request', return_value=self.prepared()), \
                patch('scripts.check_calibrated_accuracy.load_backend', return_value=(self.backend, self.execution, torch)), \
                patch('scripts.check_calibrated_accuracy.encode_candidates', return_value=(messages, [1, 2], self.calibration['candidate_token_ids'])) as encode, \
                patch('scripts.check_calibrated_accuracy.score_candidates', return_value=scoring) as score:
            main(['--text', 'A statement.', '--output', str(self.output)])
        self.assertEqual(encode.call_args.args[2], 'A new local statement.')
        score.assert_called_once_with(self.backend.model, [1, 2], self.calibration['candidate_token_ids'], 9)
        saved = json.loads(self.output.read_bytes())
        self.assertEqual(saved['mode'], 'binary_teacher_forced_closed_book')
        self.assertEqual(saved['threshold'], 0.2)
        self.assertEqual(saved['verdict'], 'SUPPORTED')
        self.assertEqual(saved['candidate_scoring']['forward_calls'], 2)
        self.assertNotIn('generation', saved)
        self.assertNotIn('raw_response', saved['candidate_scoring'])
        self.assertNotIn(b'\r\n', self.output.read_bytes())


if __name__ == '__main__':
    unittest.main()
