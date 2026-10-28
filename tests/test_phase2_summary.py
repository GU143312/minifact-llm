"""Offline aggregation tests; audit is mocked to isolate summary semantics."""
import copy
import unittest
from unittest.mock import patch

from scripts.summarize_phase2_run import summarize


def generation(ended=True, tokens=None):
    return {'ended_with_eos': ended, 'token_trace': tokens or []}


def make_step(number, last=False):
    token = {'label_token': True, 'raw_logprob': -.4, 'full_vocab_entropy_nats': .8}
    response = {'verdict': 'REFUTED', 'feedback': 'Check the number.',
                'verdict_generation': generation(tokens=[token]), 'feedback_generation': generation()}
    return {'round': number, 'accepted_claims': [{'claim': 'Count is 10', 'quote_index': 0,
                                                  'start_idx': 0, 'end_idx': 11}],
            'alignment_rejections': [{'reason': 'no_literal_match'}],
            'extract_response': generation(), 'extract_error': None,
            'verifications': [{'quote_index': 0, 'claim': 'Count is 10', 'status': 'ok', 'response': response}],
            'source_logic_review': {'status': 'no_feedback', 'response': {'feedback': None, 'generation': generation()}},
            'next_feedback': {'origin': 'verifier_model_output', 'entries': []},
            'stop_reason': 'round_limit' if last else None}


def fixture():
    return {'completed': True, 'results': [{'id': 'D01', 'source': 'Count is 10',
            'interaction': {'stop_reason': 'round_limit', 'rounds': [make_step(1), make_step(2, True)]}}]}


class Phase2SummaryTests(unittest.TestCase):
    def test_repeated_claims_are_separate_round_occurrences_not_unique_facts(self):
        with patch('scripts.summarize_phase2_run.audit', return_value={'status': 'complete'}) as check:
            result = summarize(fixture(), [], 'input.json')
        check.assert_called_once()
        r1, r2 = result['round_statistics']
        self.assertEqual((r1['accepted_quote_occurrences'], r2['accepted_quote_occurrences']), (1, 1))
        self.assertEqual(r1['rejection_reasons'], {'no_literal_match': 1})
        self.assertEqual(r1['verdicts'], {'REFUTED': 1})
        self.assertEqual(r1['label_token_count'], 1)
        self.assertAlmostEqual(r1['mean_label_token_logprob'], -.4)
        self.assertAlmostEqual(r1['mean_label_token_full_vocab_entropy_nats'], .8)
        self.assertEqual(r1['feedback_transitions_to_next_round'], 1)
        self.assertEqual(r2['feedback_transitions_to_next_round'], 0)
        self.assertEqual(result['samples'][0]['round_count'], 2)
        self.assertIn('not unique facts', result['limitations'][0])

    def test_missing_and_unterminated_records_stay_visible_without_fake_scores(self):
        run = fixture()
        step = run['results'][0]['interaction']['rounds'][0]
        step['extract_response'] = generation(False)
        step['verifications'][0] = {'quote_index': 0, 'claim': 'Count is 10', 'status': 'exception', 'response': None}
        step['source_logic_review'] = {'status': 'invalid_response', 'response': 'bad model output'}
        with patch('scripts.summarize_phase2_run.audit', return_value={'status': 'partial'}):
            result = summarize(run, [], 'input.json')
        first = result['samples'][0]['rounds'][0]
        self.assertEqual(first['generations_without_eos'], ['extract'])
        self.assertEqual(first['missing_generations'], ['verdict', 'feedback', 'source_logic'])
        self.assertIsNone(first['verifications'][0]['mean_label_token_logprob'])
        self.assertEqual(result['round_statistics'][0]['label_token_count'], 0)
        self.assertEqual(result['round_statistics'][0]['verdicts'], {'missing_or_invalid': 1})

    def test_audit_failure_prevents_summary(self):
        with patch('scripts.summarize_phase2_run.audit', side_effect=ValueError('corrupt')):
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                summarize(copy.deepcopy(fixture()), [], 'input.json')


if __name__ == '__main__':
    unittest.main()
