"""Check label isolation and immutable, exact-prefix checkpoint handling."""
import copy
import tempfile
import unittest
from pathlib import Path

import minifact
from scripts.run_phase3 import prepare_output, sha, validate_inputs, write


class Phase3RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.freeze_path = self.directory / 'freeze.json'
        self.path = self.directory / 'run.json'
        self.contract = {'protocol': 'test', 'input_sha256': 'frozen'}
        write(self.freeze_path, {'contract': self.contract, 'frozen_at_utc': '2026-11-12T00:00:00Z'})
        self.frozen = {'contract': self.contract, '_path': self.freeze_path}
        self.rows = [{'id': 'a', 'text': 'First claim'}, {'id': 'b', 'text': 'Second claim'}]

    def checkpoint(self, **changes):
        run = {'contract': copy.deepcopy(self.contract), 'freeze_sha256': sha(self.freeze_path),
               'completed': False, 'results': [{'id': 'a', 'source': 'First claim'}]}
        run.update(changes)
        write(self.path, run)
        return run

    def test_gold_and_other_extra_fields_are_rejected_not_silently_passed(self):
        validate_inputs(self.rows)
        for field, value in [('label', 'REFUTED'), ('gold', {'verdict': 'REFUTED'}),
                             ('evidence', 'Fact checker answer'), ('facts', ['Expected fact'])]:
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_inputs(rows)

    def test_duplicate_and_empty_text_inputs_are_rejected(self):
        for rows in [[], [self.rows[0], self.rows[0]], [{'id': 'a', 'text': ' \t'}],
                     [{'id': 1, 'text': 'Claim'}], [{'id': 'a', 'text': None}]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_inputs(rows)

    def test_resume_accepts_exact_id_and_source_prefix_without_mutation(self):
        run = self.checkpoint()
        original_bytes = self.path.read_bytes()
        before_frozen = copy.deepcopy(self.frozen)
        before_rows = copy.deepcopy(self.rows)
        self.assertEqual(prepare_output(self.path, self.frozen, self.rows, True), run)
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertEqual(self.frozen, before_frozen)
        self.assertEqual(self.rows, before_rows)

    def test_resume_rejects_changed_source_even_when_id_matches(self):
        self.checkpoint()
        changed = copy.deepcopy(self.rows)
        changed[0]['text'] = 'Changed first claim'
        with self.assertRaisesRegex(ValueError, 'ID/source prefix'):
            prepare_output(self.path, self.frozen, changed, True)

    def test_resume_rejects_changed_id_and_reordered_prefix(self):
        self.checkpoint()
        for rows in [list(reversed(self.rows)), [{'id': 'changed', 'text': 'First claim'}]]:
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, 'ID/source prefix'):
                prepare_output(self.path, self.frozen, rows, True)

    def test_resume_rejects_changed_contract_or_freeze_bytes(self):
        self.checkpoint()
        frozen = copy.deepcopy(self.frozen)
        frozen['contract']['input_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'protocol/freeze mismatch'):
            prepare_output(self.path, frozen, self.rows, True)
        # Even a whitespace-only freeze modification invalidates its recorded hash.
        self.freeze_path.write_bytes(self.freeze_path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'protocol/freeze mismatch'):
            prepare_output(self.path, self.frozen, self.rows, True)

    def test_existing_outputs_are_not_overwritten_and_complete_runs_are_not_resumed(self):
        self.checkpoint(completed=True)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            prepare_output(self.path, self.frozen, self.rows)
        with self.assertRaisesRegex(ValueError, 'Run complete'):
            prepare_output(self.path, self.frozen, self.rows, True)
        self.assertEqual(self.path.read_bytes(), before)

    def test_resume_requires_existing_file_and_bounded_prefix(self):
        with self.assertRaisesRegex(ValueError, 'Resume file missing'):
            prepare_output(self.path, self.frozen, self.rows, True)
        self.checkpoint(results=[{'id': 'a', 'source': 'First claim'}] * 3)
        with self.assertRaisesRegex(ValueError, 'invalid prefix'):
            prepare_output(self.path, self.frozen, self.rows, True)

    def test_new_run_is_bound_to_freeze_without_writing_output_or_mutating_inputs(self):
        run = prepare_output(self.path, self.frozen, self.rows)
        self.assertFalse(run['completed'])
        self.assertEqual(run['results'], [])
        self.assertEqual(run['freeze_sha256'], sha(self.freeze_path))
        self.assertEqual(run['contract'], self.contract)
        self.assertFalse(self.path.exists())

    def test_atomic_serialization_is_lf_and_roundtrips_unicode(self):
        write(self.path, {'text': '否定：并非如此。', 'items': [1, 2]})
        data = self.path.read_bytes()
        self.assertNotIn(b'\r', data)
        self.assertTrue(data.endswith(b'\n'))
        self.assertEqual(minifact.read_json(self.path)['text'], '否定：并非如此。')
        self.assertFalse(self.path.with_suffix('.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
