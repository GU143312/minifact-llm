"""Protect dataset provenance, gold separation, and frozen evaluation scope."""
import copy
import json
import unittest

from jsonschema import ValidationError

from scripts.curate_phase1 import DATA, record_hash, restore_upstream
from scripts.validate_phase1 import validate


class Phase1DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def load(name):
            return json.loads((DATA / name).read_text(encoding='utf-8'))
        cls.master = load('dataset.json')
        cls.inputs = load('inputs.json')
        cls.config = load('curation.json')
        cls.schema = load('dataset.schema.json')

    def check(self, dataset=None, inputs=None):
        return validate(dataset if dataset is not None else self.master,
                        inputs if inputs is not None else self.inputs, self.config, self.schema)

    def test_delivered_dataset_is_valid(self):
        self.assertEqual(self.check()['sample_count'], 60)

    def test_changed_claim_cannot_keep_upstream_hash(self):
        changed = copy.deepcopy(self.master)
        changed['samples'][0]['text'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'record hash mismatch'):
            self.check(changed)

    def test_duplicate_text_is_rejected(self):
        changed = copy.deepcopy(self.master)
        changed['samples'][1]['text'] = changed['samples'][0]['text']
        with self.assertRaisesRegex(ValueError, 'Duplicate text'):
            self.check(changed)

    def test_gold_leakage_in_inputs_is_rejected(self):
        changed = copy.deepcopy(self.inputs)
        changed[0]['gold'] = {'verdict': 'REFUTED'}
        with self.assertRaisesRegex(ValueError, 'gold leakage'):
            self.check(inputs=changed)

    def test_sample_verdict_cannot_be_reported_as_atomic_gold(self):
        changed = copy.deepcopy(self.master)
        changed['samples'][0]['annotations']['atomic_claims'] = [{'verdict': 'REFUTED'}]
        with self.assertRaises(ValidationError):
            self.check(changed)

    def test_frozen_selection_cannot_be_replaced_by_another_row(self):
        changed = copy.deepcopy(self.master)
        changed['samples'][0]['provenance']['source_record_index'] = 1
        with self.assertRaisesRegex(ValueError, 'Frozen source indices'):
            self.check(changed)

    def test_wrong_normalized_date_is_rejected(self):
        changed = copy.deepcopy(self.master)
        changed['samples'][0]['claim_date'] = '2020-01-01'
        with self.assertRaisesRegex(ValueError, 'normalized date mismatch'):
            self.check(changed)

    def test_repeated_article_cannot_masquerade_as_another_sample(self):
        changed = copy.deepcopy(self.master)
        sample = changed['samples'][1]
        sample['provenance']['fact_checking_article'] = changed['samples'][0]['provenance']['fact_checking_article']
        sample['provenance']['source_record_sha256'] = record_hash(restore_upstream(sample))
        with self.assertRaisesRegex(ValueError, 'Duplicate fact-checking article'):
            self.check(changed)

    def test_reported_counts_must_match_contents(self):
        changed = copy.deepcopy(self.master)
        changed['statistics']['question_count'] += 1
        with self.assertRaisesRegex(ValueError, 'statistics mismatch'):
            self.check(changed)


if __name__ == '__main__':
    unittest.main()
