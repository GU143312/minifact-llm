"""Check label isolation and scores for the accuracy candidates without GPU."""
import copy
import json
import unittest

from accuracy_backend import AccuracyBackend, decision_from_generation


def generated(label='TRUE', eos=True):
    return {'raw_response': label, 'ended_with_eos': eos,
            'mean_nonspecial_generated_logprob': -.2,
            'token_trace': [
                {'decoded_piece': label, 'special': False, 'label_token': None,
                 'raw_logprob': -.2, 'full_vocab_entropy_nats': .4},
                {'decoded_piece': '', 'special': True, 'label_token': None,
                 'raw_logprob': -.9, 'full_vocab_entropy_nats': .1}]}


class RecordingBackend(AccuracyBackend):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen = []

    def _generate(self, system, payload, **params):
        self.seen.append((system, copy.deepcopy(payload), params))
        return next(self.responses)


class AccuracyBackendTests(unittest.TestCase):
    def test_label_scores_are_realigned_without_modifying_raw_numbers(self):
        raw = generated()
        result = decision_from_generation(raw)
        self.assertEqual(result['verdict'], 'SUPPORTED')
        self.assertEqual(result['raw_label'], 'TRUE')
        self.assertEqual(result['generation']['mean_label_token_logprob'], -.2)
        self.assertEqual(result['generation']['mean_label_token_full_vocab_entropy_nats'], .4)
        self.assertIsNone(raw['token_trace'][0]['label_token'])
        self.assertTrue(result['generation']['token_trace'][0]['label_token'])
        self.assertFalse(result['generation']['token_trace'][1]['label_token'])

    def test_unknown_is_not_a_false_prediction(self):
        self.assertEqual(decision_from_generation(generated('UNKNOWN'))['verdict'], 'NOT_ENOUGH_INFO')

    def test_extra_explanation_case_variants_and_missing_eos_are_invalid(self):
        for label, eos in [('True', True), ('TRUE.', True), ('FALSE because...', True), ('TRUE', False)]:
            self.assertIsNone(decision_from_generation(generated(label, eos))['verdict'])

    def test_decoding_mismatch_does_not_invent_available_score(self):
        g = generated()
        g['token_trace'][0]['decoded_piece'] = 'OTHER'
        r = decision_from_generation(g)['generation']
        self.assertIsNone(r['mean_label_token_logprob'])
        self.assertEqual(r['label_token_alignment'], 'unavailable')

    def row(self):
        return {'id': 'PRIVATE_COHORT', 'text': 'A claim.',
                'context': {'claim_date': '2020-01-01', 'speaker': 'Speaker', 'location_ISO_code': 'ZA'},
                'evidence': [{'question': 'What happened?', 'answer': 'An event.',
                              'url': 'https://example.org/VERDICT_FALSE'}]}

    def test_closed_book_only_passes_statement(self):
        b = RecordingBackend([generated()])
        b.run_arm(self.row(), 'neutral')
        self.assertEqual(b.seen[0][1], {'statement': 'A claim.'})
        self.assertEqual(b.seen[0][2]['max_new_tokens'], 16)

    def test_reasoned_review_is_untrusted_context_not_gold(self):
        b = RecordingBackend([generated('A review.'), generated('FALSE')])
        r = b.run_arm(self.row(), 'reasoned')
        self.assertEqual(r['verdict'], 'REFUTED')
        self.assertEqual(b.seen[1][1], {'statement': 'A claim.', 'review': 'A review.'})
        self.assertEqual(b.seen[0][2]['max_new_tokens'], 64)

    def test_truncated_review_is_preserved_but_not_consumed(self):
        b = RecordingBackend([generated('unfinished', False), generated()])
        r = b.run_arm(self.row(), 'reasoned')
        self.assertIsNone(b.seen[1][1]['review'])
        self.assertEqual(r['calls'][0]['generation']['raw_response'], 'unfinished')

    def test_evidence_payload_omits_urls_ids_and_gold_fields(self):
        b = RecordingBackend([generated()])
        r = b.run_arm(self.row(), 'evidence')
        p = b.seen[0][1]
        self.assertEqual(p['evidence'], [{'id': 'E1', 'question': 'What happened?', 'answer': 'An event.'}])
        self.assertNotIn('VERDICT_FALSE', json.dumps(p))
        self.assertNotIn('PRIVATE_COHORT', json.dumps(p))
        self.assertEqual(r['condition'], 'supplied_benchmark_evidence')

    def test_missing_evidence_is_an_error_not_closed_book_fallback(self):
        row = self.row()
        row['evidence'] = []
        with self.assertRaises(ValueError):
            RecordingBackend([]).run_arm(row, 'evidence')


if __name__ == '__main__':
    unittest.main()
