"""Offline behavior tests for actual-generation verdict scoring and feedback."""
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from phase2_backend import Phase2Backend, is_no_feedback


class FakeTokenizer:
    eos_token_id = 4
    all_special_ids = [4]
    text = {0: 'INVALID', 1: 'REFUTED', 2: '["Earth is flat"]', 3: 'Check evidence.', 4: ''}

    def __init__(self):
        self.messages = []
        self.text = dict(self.text)

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        system = messages[0]['content']
        task = (1 if system.startswith('Extract every') else 2 if 'exactly one label' in system
                else 4 if system.startswith('Compare the factual propositions') else 3)
        return {'input_ids': torch.tensor([[0, task]]), 'attention_mask': torch.tensor([[1, 1]])}

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(self.text[i] for i in ids)


class FakeModel(torch.nn.Module if torch is not None else object):
    def __init__(self, verdict_token=1, end_verdict=True, end_feedback=True, end_logic=True):
        super().__init__()
        self.emb = torch.nn.Embedding(5, 1)
        self.generation_config = SimpleNamespace(eos_token_id=4)
        self.verdict_token = verdict_token
        self.end_verdict = end_verdict
        self.end_feedback = end_feedback
        self.end_logic = end_logic
        self.calls = []

    def get_input_embeddings(self):
        return self.emb

    def generate(self, input_ids, attention_mask, output_logits, **kwargs):
        task = int(input_ids[0, 1])
        token = {1: 2, 2: self.verdict_token, 3: 3, 4: 3}[task]
        self.calls.append({'task': task, 'output_logits': output_logits, **kwargs})
        ended = ((task != 2 or self.end_verdict) and (task != 3 or self.end_feedback)
                 and (task != 4 or self.end_logic))
        generated = [token, 4] if ended else [token]
        sequences = torch.cat([input_ids, torch.tensor([generated])], dim=1)
        raw = (torch.tensor([[0., math.log(2), 0., 0., 0.]]),
               torch.tensor([[0., 0., 0., 0., math.log(4)]]))
        # Deliberately disagree with raw logits: the backend must not use scores.
        processed = (torch.tensor([[0., -50., 0., 0., 0.]]),) * 2
        return SimpleNamespace(sequences=sequences, logits=raw[:len(generated)] if output_logits else None, scores=processed)


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class Phase2BackendTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel().eval()
        self.config = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text(encoding='utf-8'))
        self.backend = Phase2Backend(self.model, self.tokenizer, self.config)

    def test_extractor_records_feedback_and_literal_slice(self):
        feedback = {'origin': 'verifier', 'warning': 'untrusted', 'entries': ['Preserve the subject.']}
        result = self.backend.extract('Earth is flat', feedback=feedback)
        self.assertEqual(result['raw_quotes'], ['Earth is flat'])
        self.assertEqual(result['aligned_claims'][0]['claim'], 'Earth is flat')
        request = json.loads(self.tokenizer.messages[-1][1]['content'])
        self.assertEqual(request['verifier_feedback'], feedback)
        self.assertIn('untrusted suggestions', self.tokenizer.messages[-1][0]['content'])

    def test_repeated_source_quote_is_not_assigned_an_arbitrary_position(self):
        result = self.backend.extract('Earth is flat. Earth is flat.')
        self.assertEqual(result['aligned_claims'], [])
        self.assertEqual(result['alignment_rejections'][0]['reason'], 'ambiguous_repeated_quote')

    def test_verdict_uses_raw_generation_logits_and_separate_feedback(self):
        result = self.backend.verify('Earth is flat', 'Earth is flat')
        self.assertEqual(result['verdict'], 'REFUTED')
        self.assertEqual(result['feedback'], 'Check evidence.')
        trace = result['verdict_generation']['token_trace']
        self.assertEqual([t['token_id'] for t in trace], [1, 4])
        self.assertEqual([t['special'] for t in trace], [False, True])
        self.assertEqual([t['label_token'] for t in trace], [True, False])
        self.assertAlmostEqual(trace[0]['raw_logprob'], math.log(2/6), places=6)
        self.assertAlmostEqual(trace[1]['raw_logprob'], math.log(.5), places=6)
        self.assertAlmostEqual(result['verdict_generation']['mean_label_token_logprob'], math.log(2/6), places=6)
        self.assertAlmostEqual(trace[0]['full_vocab_entropy_nats'], -(2/6*math.log(2/6)+4/6*math.log(1/6)), places=6)
        self.assertEqual(len(self.tokenizer.messages), 2)

    def test_invalid_label_is_retained_without_invented_verdict(self):
        backend = Phase2Backend(FakeModel(verdict_token=0).eval(), self.tokenizer, self.config)
        result = backend.verify('Earth is flat', 'Earth is flat')
        self.assertIsNone(result['verdict'])
        self.assertEqual(result['verdict_generation']['raw_response'], 'INVALID')
        self.assertIsNone(result['verdict_generation']['mean_label_token_logprob'])
        self.assertTrue(result['parse_error'])

    def test_role_limits_are_passed_to_generation_and_recorded(self):
        result = self.backend.verify('Earth is flat', 'Earth is flat')
        self.assertEqual([c['max_new_tokens'] for c in self.model.calls], [16, 64])
        self.assertEqual(result['verdict_generation']['generation_params']['max_new_tokens'], 16)
        self.assertEqual(result['feedback_generation']['generation_params']['max_new_tokens'], 64)
        short_config = dict(self.config, max_new_tokens=8)
        short_model = FakeModel().eval()
        short_result = Phase2Backend(short_model, self.tokenizer, short_config).verify('s', 'c')
        self.assertEqual([c['max_new_tokens'] for c in short_model.calls], [8, 8])
        self.assertEqual(short_result['feedback_generation']['generation_params']['max_new_tokens'], 8)

    def test_unterminated_verdict_does_not_become_valid_label(self):
        model = FakeModel(end_verdict=False).eval()
        result = Phase2Backend(model, self.tokenizer, self.config).verify('s', 'c')
        self.assertIsNone(result['verdict'])
        self.assertTrue(result['parse_error'])
        self.assertFalse(result['verdict_generation']['ended_with_eos'])
        self.assertEqual(result['verdict_generation']['raw_response'], 'REFUTED')
        self.assertEqual(result['verdict_generation']['generated_token_ids'], [1])

    def test_unterminated_feedback_is_not_used_but_raw_is_retained(self):
        model = FakeModel(end_feedback=False).eval()
        result = Phase2Backend(model, self.tokenizer, self.config).verify('s', 'c')
        self.assertEqual(result['verdict'], 'REFUTED')
        self.assertIsNone(result['feedback'])
        self.assertFalse(result['feedback_generation']['ended_with_eos'])
        self.assertEqual(result['feedback_generation']['raw_response'], 'Check evidence.')
        self.assertEqual(result['feedback_generation']['generated_token_ids'], [3])

    def test_no_issue_is_not_a_feedback_but_raw_is_retained(self):
        self.tokenizer.text[3] = 'NO_ISSUE'
        result = self.backend.verify('s', 'c')
        self.assertIsNone(result['feedback'])
        self.assertTrue(result['feedback_generation']['ended_with_eos'])
        self.assertEqual(result['feedback_generation']['raw_response'], 'NO_ISSUE')
        self.assertEqual(result['feedback_generation']['generated_token_ids'], [3, 4])

    def test_sentinel_variants_and_meaningful_suffixes(self):
        for marker in ['NO_ISSUE', 'NO_CONTRADICTION']:
            for suffix in ['', '.', '!']:
                self.assertTrue(is_no_feedback('  ' + marker.lower() + suffix + '  ', marker))
            for text in [marker + '. But the dates conflict.', marker + '..',
                         marker + ' is uncertain', 'prefix ' + marker]:
                self.assertFalse(is_no_feedback(text, marker))

    def test_punctuated_sentinels_keep_raw_but_suppress_feedback(self):
        self.tokenizer.text[3] = '  no_issue.  '
        verdict = self.backend.verify('s', 'c')
        self.assertIsNone(verdict['feedback'])
        self.assertEqual(verdict['feedback_generation']['raw_response'], '  no_issue.  ')
        self.tokenizer.text[3] = '  no_contradiction!  '
        logic = self.backend.review_source('s', ['c'])
        self.assertIsNone(logic['feedback'])
        self.assertEqual(logic['generation']['raw_response'], '  no_contradiction!  ')

    def test_sentinel_with_explanation_is_not_discarded(self):
        self.tokenizer.text[3] = 'NO_ISSUE. But the quantity may be inconsistent.'
        verdict = self.backend.verify('s', 'c')
        self.assertEqual(verdict['feedback'], self.tokenizer.text[3])
        self.tokenizer.text[3] = 'NO_CONTRADICTION. But 10 and 20 conflict.'
        logic = self.backend.review_source('s', ['c'])
        self.assertEqual(logic['feedback'], self.tokenizer.text[3])

    def test_source_logic_challenge_passes_complete_context_and_tracks_generation(self):
        self.tokenizer.text[3] = 'The count is both 10 and 20 for the same date.'
        source = 'On Monday the count was 10. On Monday the count was 20.'
        claims = ['On Monday the count was 10', 'On Monday the count was 20']
        result = self.backend.review_source(source, claims)
        self.assertEqual(result['feedback'], self.tokenizer.text[3])
        request = json.loads(self.tokenizer.messages[-1][1]['content'])
        self.assertEqual(request, {'source': source, 'claims': claims})
        self.assertIn('Do not infer external truth', self.tokenizer.messages[-1][0]['content'])
        self.assertEqual(result['generation']['generation_params']['max_new_tokens'], 64)
        self.assertTrue(result['generation']['ended_with_eos'])
        self.assertEqual(result['generation']['generated_token_ids'], [3, 4])

    def test_no_contradiction_is_not_feedback_and_raw_is_retained(self):
        self.tokenizer.text[3] = 'NO_CONTRADICTION'
        result = self.backend.review_source('Counts changed over time.', ['Counts changed over time'])
        self.assertIsNone(result['feedback'])
        self.assertEqual(result['generation']['raw_response'], 'NO_CONTRADICTION')
        self.assertTrue(result['generation']['ended_with_eos'])

    def test_unterminated_source_logic_response_is_not_feedback(self):
        self.tokenizer.text[3] = 'Both 10 and 20'
        model = FakeModel(end_logic=False).eval()
        result = Phase2Backend(model, self.tokenizer, self.config).review_source('source', ['claim'])
        self.assertIsNone(result['feedback'])
        self.assertEqual(result['generation']['raw_response'], 'Both 10 and 20')
        self.assertFalse(result['generation']['ended_with_eos'])
        self.assertEqual(result['generation']['generated_token_ids'], [3])

    def test_sampling_or_training_mode_rejected(self):
        config = dict(self.config, do_sample=True)
        with self.assertRaises(ValueError):
            Phase2Backend(self.model, self.tokenizer, config)
        self.model.train()
        with self.assertRaises(ValueError):
            Phase2Backend(self.model, self.tokenizer, self.config)


if __name__ == '__main__':
    unittest.main()
