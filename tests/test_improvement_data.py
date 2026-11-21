"""Guard project holdout isolation and provenance of provided evidence."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.build_improvement_data import (
    ARTIFACTS, DATA, RULES, build, encoded, evidence_input, evidence_projection,
    exclusion, explicit_verdict, rank, sample, sha, validate,
)


class ImprovementDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        for name in (*ARTIFACTS, 'manifest.json', 'dataset.schema.json'):
            shutil.copyfile(DATA / name, self.directory / name)

    def read(self, name):
        return json.loads((self.directory / name).read_bytes())

    def write_rebound(self, name, value):
        """Test semantic guards even after an attacker rebinds artifact hashes."""
        raw = encoded(value)
        (self.directory / name).write_bytes(raw)
        manifest = self.read('manifest.json')
        manifest['artifacts_sha256'][name] = sha(raw)
        (self.directory / 'manifest.json').write_bytes(encoded(manifest))

    def test_shipped_balanced_sets_and_frozen_hashes(self):
        stats = validate(self.directory)
        self.assertEqual(stats['dev']['labels'], {'REFUTED': 16, 'SUPPORTED': 16})
        self.assertEqual(stats['holdout']['labels'], {'REFUTED': 40, 'SUPPORTED': 40})

    def test_changed_bytes_fail_frozen_artifact_binding(self):
        with (self.directory / 'holdout_inputs.json').open('ab') as stream:
            stream.write(b' ')
        with self.assertRaisesRegex(ValueError, 'Frozen artifact SHA256 mismatch'):
            validate(self.directory)

    def test_labels_cannot_be_added_to_closed_book_inputs(self):
        changed = self.read('dev_inputs.json')
        changed[0]['label'] = 'SUPPORTED'
        self.write_rebound('dev_inputs.json', changed)
        with self.assertRaisesRegex(ValueError, 'gold leakage'):
            validate(self.directory)

    def test_justification_cannot_be_added_to_evidence_inputs(self):
        changed = self.read('holdout_evidence_inputs.json')
        changed[0]['justification'] = 'The upstream conclusion is false.'
        self.write_rebound('holdout_evidence_inputs.json', changed)
        with self.assertRaisesRegex(ValueError, 'gold leakage'):
            validate(self.directory)

    def test_cannot_reuse_development_story_in_holdout(self):
        dev = self.read('dev_dataset.json')['samples'][0]
        changed = self.read('holdout_dataset.json')
        target = next(s for s in changed['samples'] if s['gold']['verdict'] == dev['gold']['verdict'])
        replacement = sample(dev['provenance']['source_record_index'], dev['upstream_record'],
                             'holdout', int(target['id'].rsplit('-', 1)[1]))
        changed['samples'][changed['samples'].index(target)] = replacement
        changed['samples'].sort(key=lambda s: rank(RULES['order_namespace'], s['id'], s['text']))
        self.write_rebound('holdout_dataset.json', changed)
        manifest = self.read('manifest.json')
        manifest['selected_train_indices']['holdout'] = sorted(
            s['provenance']['source_record_index'] for s in changed['samples'])
        (self.directory / 'manifest.json').write_bytes(encoded(manifest))
        with self.assertRaisesRegex(ValueError, 'Cross-set claim/article/ID/index overlap'):
            validate(self.directory)

    def test_preserved_upstream_record_change_is_detected(self):
        changed = self.read('dev_dataset.json')
        changed['samples'][0]['upstream_record']['justification'] += ' Changed.'
        self.write_rebound('dev_dataset.json', changed)
        with self.assertRaisesRegex(ValueError, 'Source record hash/eligibility mismatch'):
            validate(self.directory)

    def test_reannotation_flag_is_excluded(self):
        row = copy.deepcopy(self.read('dev_dataset.json')['samples'][0]['upstream_record'])
        row['required_reannotation'] = True
        self.assertEqual(exclusion(0, row), 'requires_reannotation')

    def test_wrong_upstream_bytes_cannot_be_used_for_rebuild(self):
        with self.assertRaisesRegex(ValueError, 'Pinned upstream source SHA256 mismatch'):
            build(b'[]', b'{}')

    def test_evidence_projection_never_reads_gold_label_or_justification(self):
        row = copy.deepcopy(self.read('dev_dataset.json')['samples'][0]['upstream_record'])
        first = evidence_projection(row)
        row['label'] = 'CHANGED GOLD'
        row['justification'] = 'CHANGED GOLD JUSTIFICATION'
        self.assertEqual(evidence_projection(row), first)

    def test_explicit_verdict_questions_and_curator_article_are_omitted(self):
        def answer(text, url='https://example.org/evidence'):
            return {'answer': text, 'source_url': url, 'answer_type': 'Extractive', 'source_medium': 'Web text'}
        row = {'fact_checking_article': 'https://checker.example/review',
               'questions': [
                   {'question': 'Is the claim true?', 'answers': [answer('False.')]},
                   {'question': 'When was the bridge opened?', 'answers': [
                       answer('The bridge opened in 2015.', 'https://checker.example/review#section'),
                       answer('The claim is false.'), answer('The bridge opened in 2015.')]}]}
        self.assertEqual(evidence_projection(row), [{'question': 'When was the bridge opened?',
                         'answer': 'The bridge opened in 2015.', 'url': 'https://example.org/evidence'}])
        self.assertTrue(explicit_verdict('A baseless claim about the bridge.'))

    def test_evidence_caps_are_character_based_and_deterministic(self):
        row = {'fact_checking_article': 'https://checker.example/review', 'questions': [
            {'question': 'What was recorded?', 'answers': [
                {'answer': '\u6865' * 1200, 'source_url': 'https://example.org/' + str(i),
                 'answer_type': 'Extractive', 'source_medium': 'Web text'} for i in range(12)]}]}
        projected = evidence_projection(row)
        self.assertEqual(len(projected), 6)
        self.assertEqual(sum(len(x['answer']) for x in projected), 5000)
        self.assertEqual(len(projected[0]['answer']), 900)
        self.assertEqual(projected, evidence_projection(row))

    def test_evidence_input_has_only_allowed_context_and_qa_fields(self):
        s = self.read('dev_dataset.json')['samples'][0]
        projected = evidence_input(s)
        self.assertEqual(set(projected), {'id', 'text', 'context', 'evidence'})
        self.assertEqual(set(projected['context']), {'claim_date', 'speaker', 'location_ISO_code'})
        self.assertTrue(all(set(e) == {'question', 'answer', 'url'} for e in projected['evidence']))


if __name__ == '__main__':
    unittest.main()
