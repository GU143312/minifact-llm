"""Offline checks for the sample-level wrapper, with no model downloads."""
import copy
import json
import unittest
from unittest.mock import Mock

from phase3_experiment import (
    SAMPLE_PROMPT, Phase3Backend, analysis_from_interaction, prediction,
)


def generated(text='REFUTED', ended=True):
    return {
        'raw_response': text, 'ended_with_eos': ended,
        'mean_label_token_logprob': -0.25,
        'mean_label_token_full_vocab_entropy_nats': 0.75,
        'token_trace': [{'token_id': 1, 'raw_logprob': -0.25}],
    }


def round_record(number, *, empty=False):
    return {
        'round': number,
        'accepted_claims': [] if empty else [
            {'claim': 'Alice left today', 'start_idx': 0, 'end_idx': 16, 'quote_index': 0}],
        'alignment_rejections': [] if not empty else [
            {'quote_index': 0, 'reason': 'invalid_quote', 'quote': []}],
        'extract_error': None,
        'verifications': [] if empty else [{
            'claim': 'Alice left today', 'status': 'ok',
            'response': {
                'verdict': 'NOT_ENOUGH_INFO', 'feedback': 'Preserve the date.',
                'verdict_generation': generated('NOT_ENOUGH_INFO'),
                # Deliberately different: feedback generation is not a verdict score.
                'feedback_generation': {
                    'mean_label_token_logprob': -9,
                    'mean_label_token_full_vocab_entropy_nats': 9},
            },
        }],
        'source_logic_review': {
            'status': 'no_feedback', 'response': {'feedback': None}},
        'stop_reason': 'no_valid_claims' if empty else None,
    }


class SampleJudgeTests(unittest.TestCase):
    def backend(self, response=None):
        # _generate is the tested Phase 2 scoring boundary. The sample wrapper
        # itself can be checked without importing torch or loading any weights.
        backend = Phase3Backend.__new__(Phase3Backend)
        backend._generate = Mock(return_value=response or generated())
        return backend

    def test_both_arms_use_same_prompt_source_and_generation_parameters(self):
        backend = self.backend()
        analysis = {'origin': 'same_model_untrusted_analysis', 'rounds': []}
        backend.judge_sample('Alice left today')
        backend.judge_sample('Alice left today', analysis)
        direct, dual = backend._generate.call_args_list
        self.assertEqual(direct.args, (SAMPLE_PROMPT, {
            'source': 'Alice left today', 'analysis': None}))
        self.assertEqual(dual.args, (SAMPLE_PROMPT, {
            'source': 'Alice left today', 'analysis': analysis}))
        self.assertEqual(direct.kwargs, dual.kwargs)
        self.assertEqual(direct.kwargs, {'trace_logits': True, 'max_new_tokens': 16})

    def test_payload_does_not_include_case_id_gold_or_reference_evidence(self):
        interaction = {
            'id': 'PRIVATE_CASE_ID', 'label': 'PRIVATE_GOLD_LABEL',
            'evidence': 'PRIVATE_REFERENCE_EVIDENCE', 'rounds': [round_record(1)],
            'stop_reason': 'no_actionable_feedback',
        }
        interaction['rounds'][0]['gold'] = 'PRIVATE_GOLD_LABEL'
        interaction['rounds'][0]['verifications'][0]['id'] = 'PRIVATE_CASE_ID'
        interaction['rounds'][0]['verifications'][0]['response']['evidence'] = (
            'PRIVATE_REFERENCE_EVIDENCE')
        backend = self.backend()
        backend.judge_sample('Alice left today', analysis_from_interaction(interaction))
        payload = backend._generate.call_args.args[1]
        self.assertEqual(set(payload), {'source', 'analysis'})
        encoded = json.dumps(payload)
        for secret in ['PRIVATE_CASE_ID', 'PRIVATE_GOLD_LABEL', 'PRIVATE_REFERENCE_EVIDENCE']:
            self.assertNotIn(secret, encoded)

    def test_only_complete_exact_labels_are_valid(self):
        for label in ['SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO']:
            with self.subTest(label=label):
                result = self.backend(generated('  ' + label + '\n')).judge_sample('source')
                self.assertEqual(result['verdict'], label)
                self.assertIsNone(result['parse_error'])
        for text, ended in [
            ('REFUTED', False), ('refuted', True), ('REFUTED.', True),
            ('REFUTED because of the date', True), ('', True),
            ('SUPPORTED\nREFUTED', True),
        ]:
            with self.subTest(text=text, ended=ended):
                raw = generated(text, ended)
                result = self.backend(raw).judge_sample('source')
                self.assertIsNone(result['verdict'])
                self.assertTrue(result['parse_error'])
                self.assertEqual(result['generation'], raw)

    def test_empty_or_nonstring_source_never_calls_model(self):
        backend = self.backend()
        for value in ['', '  ', None, 5, ['source']]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                backend.judge_sample(value)
        backend._generate.assert_not_called()

    def test_prediction_retains_actual_final_verdict_generation_scores(self):
        direct = self.backend(generated('SUPPORTED')).judge_sample('source')
        dual_generation = generated('REFUTED')
        dual_generation['mean_label_token_logprob'] = -0.7
        dual_generation['mean_label_token_full_vocab_entropy_nats'] = 1.2
        dual = self.backend(dual_generation).judge_sample('source', {'rounds': []})
        row = {'id': 'case-1', 'direct': direct, 'dual': dual,
               'interaction': {'mean_label_token_logprob': -99}}
        self.assertEqual(prediction(row, 'direct'), {
            'id': 'case-1', 'verdict': 'SUPPORTED',
            'mean_label_token_logprob': -0.25,
            'mean_label_token_full_vocab_entropy_nats': 0.75})
        self.assertEqual(prediction(row, 'dual')['mean_label_token_logprob'], -0.7)
        self.assertEqual(prediction(row, 'dual')['mean_label_token_full_vocab_entropy_nats'], 1.2)


class AnalysisSummaryTests(unittest.TestCase):
    def test_empty_second_round_does_not_erase_first_round_evidence_of_behavior(self):
        interaction = {'rounds': [round_record(1), round_record(2, empty=True)],
                       'stop_reason': 'no_valid_claims'}
        original = copy.deepcopy(interaction)
        analysis = analysis_from_interaction(interaction)
        self.assertEqual(len(analysis['rounds']), 2)
        self.assertEqual(analysis['rounds'][0]['accepted_claims'][0]['claim'], 'Alice left today')
        self.assertEqual(analysis['rounds'][1]['accepted_claims'], [])
        self.assertEqual(analysis['stop_reason'], 'no_valid_claims')
        self.assertEqual(interaction, original)
        analysis['rounds'][0]['accepted_claims'][0]['claim'] = 'changed'
        self.assertEqual(interaction, original)

    def test_analysis_scores_come_from_verdict_not_feedback_or_extractor(self):
        step = round_record(1)
        step['mean_label_token_logprob'] = -88
        analysis = analysis_from_interaction({'rounds': [step], 'stop_reason': 'round_limit'})
        check = analysis['rounds'][0]['checks'][0]
        self.assertEqual(check['mean_label_token_logprob'], -0.25)
        self.assertEqual(check['mean_label_token_full_vocab_entropy_nats'], 0.75)

    def test_rejected_text_and_failed_verdict_do_not_become_accepted_or_supported(self):
        step = round_record(1)
        step['accepted_claims'] = []
        step['alignment_rejections'] = [{'quote': 'invented fact', 'reason': 'no_literal_match'}]
        step['verifications'] = [{'claim': 'earlier literal claim', 'status': 'exception',
                                  'response': None, 'error': {'type': 'RuntimeError'}}]
        step['extract_error'] = {'type': 'ValueError', 'message': 'malformed response'}
        analysis = analysis_from_interaction({'rounds': [step], 'stop_reason': 'verifier_failure'})
        self.assertEqual(analysis['rounds'][0]['accepted_claims'], [])
        self.assertEqual(analysis['rounds'][0]['alignment_rejections'], step['alignment_rejections'])
        self.assertIsNone(analysis['rounds'][0]['checks'][0]['verdict'])
        self.assertIsNone(analysis['rounds'][0]['checks'][0]['mean_label_token_logprob'])
        self.assertEqual(analysis['origin'], 'same_model_untrusted_analysis')
        self.assertIn('not truth probabilities', analysis['warning'])
        self.assertIn('untrusted', SAMPLE_PROMPT)
        self.assertIn('not independent evidence', SAMPLE_PROMPT)


if __name__ == '__main__':
    unittest.main()
