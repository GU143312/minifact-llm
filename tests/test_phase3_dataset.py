"""Guard Week6 gold isolation, frozen inputs and inherited whole-sample labels."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import ValidationError

from scripts.build_phase3_testset import DATA, build, ordered_samples, validate
from scripts.curate_phase1 import canonical_article, record_hash, restore_upstream


class Phase3DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def load(name):
            return json.loads((DATA / name).read_text(encoding='utf-8'))
        cls.dataset = load('dataset.json')
        cls.inputs = load('inputs.json')
        cls.config = load('curation.json')
        cls.schema = load('dataset.schema.json')

    def check(self, dataset=None, inputs=None):
        return validate(self.dataset if dataset is None else dataset,
                        self.inputs if inputs is None else inputs,
                        self.config, self.schema)

    def test_frozen_dataset_and_label_free_projection(self):
        stats = self.check()
        self.assertEqual(stats['verdicts'], {'REFUTED': 60, 'SUPPORTED': 50})
        self.assertTrue(all(set(row) == {'id', 'text'} for row in self.inputs))

    def test_no_label_or_evidence_may_be_added_to_model_inputs(self):
        for key, value in [('gold', {'verdict': 'REFUTED'}), ('questions', []),
                           ('fact_checking_article', 'https://example.com/review')]:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.inputs)
                changed[0][key] = value
                with self.assertRaisesRegex(ValueError, 'gold leakage'):
                    self.check(inputs=changed)

    def test_original_phase1_sample_cannot_be_edited(self):
        changed = copy.deepcopy(self.dataset)
        sample = next(s for s in changed['samples'] if s['id'].startswith('MF-P1-'))
        sample['gold']['justification'] += ' altered'
        with self.assertRaisesRegex(ValueError, 'Original Phase1 sample modified'):
            self.check(changed)

    def test_control_row_cannot_be_replaced_even_with_a_rehashed_record(self):
        changed = copy.deepcopy(self.dataset)
        sample = next(s for s in changed['samples'] if s['id'].startswith('MF-P3-C-'))
        sample['provenance']['source_record_index'] = 0
        with self.assertRaisesRegex(ValueError, 'Frozen control source indices'):
            self.check(changed)

    def test_gold_is_whole_sample_and_not_atomic_annotation(self):
        changed = copy.deepcopy(self.dataset)
        changed['samples'][0]['annotations']['atomic_claims'] = [{'verdict': 'REFUTED'}]
        with self.assertRaises(ValidationError):
            self.check(changed)

    def test_control_label_mapping_cannot_be_flipped(self):
        changed = copy.deepcopy(self.dataset)
        sample = next(s for s in changed['samples'] if s['id'].startswith('MF-P3-C-'))
        sample['gold']['verdict'] = 'REFUTED'
        with self.assertRaisesRegex(ValueError, 'Whole-sample verdict mapping'):
            self.check(changed)

    def test_record_text_edit_cannot_keep_source_hash(self):
        changed = copy.deepcopy(self.dataset)
        sample = next(s for s in changed['samples'] if s['id'].startswith('MF-P3-C-'))
        sample['text'] += ' changed'
        changed['samples'] = ordered_samples(changed['samples'], self.config)
        inputs = [{'id': s['id'], 'text': s['text']} for s in changed['samples']]
        with self.assertRaisesRegex(ValueError, 'Upstream record hash mismatch'):
            self.check(changed, inputs)

    def test_rehashed_metadata_edit_still_breaks_frozen_dataset_digest(self):
        changed = copy.deepcopy(self.dataset)
        sample = next(s for s in changed['samples'] if s['id'].startswith('MF-P3-C-'))
        sample['gold']['justification'] += ' changed'
        sample['provenance']['source_record_sha256'] = record_hash(restore_upstream(sample))
        with self.assertRaisesRegex(ValueError, 'Frozen Phase3 dataset SHA256'):
            self.check(changed)

    def test_article_archive_variants_are_the_same_deduplication_key(self):
        self.assertEqual(canonical_article('https://web.archive.org/web/20200101/https://example.com/story/#part'),
                         canonical_article('https://example.com/story'))

    def test_wrong_upstream_version_is_rejected_before_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'dev.json'
            path.write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Source SHA256 mismatch'):
                build(path, self.config)


if __name__ == '__main__':
    unittest.main()
