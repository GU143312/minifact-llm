"""Local inference must isolate closed-book input and bind automatic selection."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.check_accuracy import (digest, load_selection, main, parser,
                                    prepare_request, validate_evidence_packet,
                                    write_new_output)


def packet():
    return {'context': {'claim_date': '2020-10-12', 'speaker': None, 'location_ISO_code': 'GB'},
            'evidence': [{'question': 'When did the harbor open?', 'answer': '12 October 2020.',
                          'url': 'https://example.org/source'}]}


class LocalInferenceInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root / 'out.json'
        self.evidence = self.root / 'evidence.json'
        self.evidence.write_text(json.dumps(packet()), encoding='utf-8')

    def args(self, *extra):
        return parser().parse_args(['--text', 'The harbor opened.', '--output', str(self.output),
                                    '--method', 'neutral', *extra])

    def test_closed_book_contains_only_statement_and_internal_id(self):
        method, row, provenance = prepare_request(self.args())
        self.assertEqual(method, 'neutral')
        self.assertEqual(row, {'id': 'local-statement', 'text': 'The harbor opened.'})
        self.assertIsNone(provenance['evidence_provenance'])

    def test_binary_prompt_is_available_without_reading_reference_evidence(self):
        method, row, provenance = prepare_request(self.args('--method', 'binary'))
        self.assertEqual(method, 'binary')
        self.assertEqual(set(row), {'id', 'text'})
        self.assertIsNone(provenance['evidence_provenance'])

    def test_closed_book_rejects_even_an_unreadable_evidence_argument(self):
        with patch('scripts.check_accuracy.minifact.read_json') as read:
            with self.assertRaisesRegex(ValueError, 'only permitted'):
                prepare_request(self.args('--evidence-file', str(self.root / 'missing.json')))
            read.assert_not_called()

    def test_evidence_method_requires_explicit_packet(self):
        with self.assertRaisesRegex(ValueError, 'requires --evidence-file'):
            prepare_request(self.args('--method', 'evidence'))
        method, row, provenance = prepare_request(self.args('--method', 'evidence',
                                                           '--evidence-file', str(self.evidence)))
        self.assertEqual(method, 'evidence')
        self.assertEqual(row['context'], packet()['context'])
        self.assertEqual(row['evidence'], packet()['evidence'])
        self.assertFalse(provenance['evidence_provenance']['retrieval_performed'])
        self.assertEqual(provenance['evidence_provenance']['file_sha256'], digest(self.evidence))

    def test_gold_and_justifications_cannot_be_passed_as_packet_fields(self):
        for field in ('gold', 'label', 'verdict', 'justification'):
            changed = {**packet(), field: 'SUPPORTED'}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'only context and evidence'):
                validate_evidence_packet(changed)
        changed = packet()
        changed['evidence'][0]['verdict'] = 'SUPPORTED'
        with self.assertRaisesRegex(ValueError, 'only question, answer and url'):
            validate_evidence_packet(changed)

    def test_invalid_context_qa_url_and_empty_packet_fail(self):
        cases = []
        changed = packet(); changed['context']['claim_date'] = 'yesterday'; cases.append(changed)
        changed = packet(); changed['context']['gold'] = 'SUPPORTED'; cases.append(changed)
        changed = packet(); changed['evidence'] = []; cases.append(changed)
        changed = packet(); changed['evidence'][0]['url'] = 'file:///secret'; cases.append(changed)
        changed = packet(); changed['evidence'][0]['answer'] = ''; cases.append(changed)
        for changed in cases:
            with self.subTest(packet=changed), self.assertRaises(ValueError):
                validate_evidence_packet(changed)

    def test_utf8_file_preserves_unicode_whitespace_and_crlf(self):
        source = self.root / 'statement.txt'
        source.write_bytes('\ufeff  新港开放。\r\n'.encode('utf-8'))
        args = parser().parse_args(['--input-file', str(source), '--output', str(self.output),
                                    '--method', 'reasoned'])
        _, row, provenance = prepare_request(args)
        self.assertEqual(row['text'], '  新港开放。\r\n')
        self.assertEqual(provenance['input']['file_sha256'], digest(source))

    def test_unknown_methods_and_conflicting_inputs_rejected(self):
        with patch('sys.stderr'):
            for argv in [['--text', 'x', '--method', 'arbitrary_python', '--output', str(self.output)],
                         ['--text', 'x', '--input-file', 'y', '--output', str(self.output)]]:
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    parser().parse_args(argv)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            prepare_request(self.args('--text', ' \t'))

    def test_existing_output_rejected_before_loading_model(self):
        self.output.write_bytes(b'preserve me')
        with patch('scripts.check_accuracy.load_backend') as loader:
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                main(['--text', 'A statement.', '--method', 'neutral', '--output', str(self.output)])
            loader.assert_not_called()
        self.assertEqual(self.output.read_bytes(), b'preserve me')

    def test_output_created_concurrently_cannot_be_overwritten(self):
        self.output.write_bytes(b'other writer')
        with self.assertRaises(FileExistsError):
            write_new_output(self.output, {'new': True})
        self.assertEqual(self.output.read_bytes(), b'other writer')

    def test_mocked_local_run_preserves_raw_result_and_condition(self):
        backend = Mock()
        backend.run_arm.return_value = {'verdict': 'NOT_ENOUGH_INFO',
                                       'generation': {'raw_response': 'UNKNOWN'}, 'calls': []}
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()))
        with patch('scripts.check_accuracy.load_backend', return_value=(backend, {'actual_revision': 'pinned'}, torch)):
            main(['--text', 'A statement.', '--method', 'structured', '--output', str(self.output)])
        saved = json.loads(self.output.read_bytes())
        backend.run_arm.assert_called_once_with({'id': 'local-statement', 'text': 'A statement.'}, 'structured')
        self.assertTrue(saved['completed'])
        self.assertEqual(saved['condition'], 'closed_book')
        self.assertEqual(saved['decision_policy'], 'forced_choice_prompt')
        self.assertEqual(saved['result']['generation']['raw_response'], 'UNKNOWN')
        self.assertNotIn(b'\r\n', self.output.read_bytes())


class SelectionBindingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.selection = self.root / 'selection.json'
        self.run = self.root / 'dev.json'
        for name in ('accuracy_backend.py', 'config.json'):
            (self.root / name).write_bytes(b'fixed-source')
        self.contract = {'split': 'dev', 'arms': ['neutral', 'reasoned'],
                         'files_sha256': {name: digest(self.root / name)
                                          for name in ('accuracy_backend.py', 'config.json')}}
        self.run.write_text(json.dumps({'completed': True, 'contract': self.contract}), encoding='utf-8')
        self.value = {'selected_closed_book': 'reasoned', 'development_run': 'dev.json',
                      'development_run_sha256': digest(self.run)}
        self.save()

    def save(self, value=None):
        self.selection.write_text(json.dumps(self.value if value is None else value), encoding='utf-8')

    def test_auto_selection_binds_method_run_and_source_hashes(self):
        method, provenance = load_selection(self.selection, self.root)
        self.assertEqual(method, 'reasoned')
        self.assertEqual(provenance['development_run_sha256'], digest(self.run))
        self.assertEqual(provenance['selection_sha256'], digest(self.selection))

    def test_development_output_or_method_source_changes_fail(self):
        self.run.write_bytes(self.run.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'run hash differs'):
            load_selection(self.selection, self.root)
        self.value['development_run_sha256'] = digest(self.run); self.save()
        (self.root / 'accuracy_backend.py').write_bytes(b'changed-source')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            load_selection(self.selection, self.root)

    def test_selection_cannot_execute_or_escape_through_method_or_path(self):
        for method in ['evidence', 'legacy_dual', '__import__("os").system("x")']:
            with self.subTest(method=method):
                self.save({**self.value, 'selected_closed_book': method})
                with self.assertRaisesRegex(ValueError, 'known closed-book'):
                    load_selection(self.selection, self.root)
        for value in ['../external.json', str(self.run.resolve())]:
            with self.subTest(path=value):
                self.save({**self.value, 'development_run': value})
                with self.assertRaisesRegex(ValueError, 'relative|escapes'):
                    load_selection(self.selection, self.root)

    def test_holdout_or_incomplete_run_is_not_a_development_selection(self):
        for run in [{'completed': False, 'contract': self.contract},
                    {'completed': True, 'contract': {**self.contract, 'split': 'holdout'}}]:
            self.run.write_text(json.dumps(run), encoding='utf-8')
            self.save({**self.value, 'development_run_sha256': digest(self.run)})
            with self.assertRaisesRegex(ValueError, 'completed development'):
                load_selection(self.selection, self.root)


if __name__ == '__main__':
    unittest.main()
