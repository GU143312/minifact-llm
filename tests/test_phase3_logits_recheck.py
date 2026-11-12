"""Selection and tokenizer checks for the two final-verdict logit rechecks."""
import copy
import json
import math
import unittest
from types import SimpleNamespace

from phase3_experiment import SAMPLE_PROMPT, analysis_from_interaction
from scripts.recheck_phase3_logits import first_sample_traces, recompute, validate_tokenization

try:
    import torch
except ImportError:
    torch = None


class Tokenizer:
    all_special_ids = [9]

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2]

    def decode(self, ids, skip_special_tokens=True):
        values = {7: 'SUPPORTED', 9: '' if skip_special_tokens else '<eos>'}
        return ''.join(values[token] for token in ids)


def generation():
    return {
        'prompt_input_ids': [1, 2], 'generated_token_ids': [7, 9],
        'messages': [], 'raw_response': 'SUPPORTED',
        'raw_response_with_special_tokens': 'SUPPORTED<eos>', 'ended_with_eos': True,
        'label_token_alignment': 'decoded_piece_overlap',
        'token_trace': [
            {'step': 0, 'token_id': 7, 'decoded_piece': 'SUPPORTED',
             'special': False, 'label_token': True},
            {'step': 1, 'token_id': 9, 'decoded_piece': '',
             'special': True, 'label_token': False}],
    }


def run_record():
    sample = {'id': 'first', 'source': 'source',
              'interaction': {'rounds': [], 'stop_reason': 'no_valid_claims'}}
    for arm in ('direct', 'dual'):
        g = generation()
        analysis = None if arm == 'direct' else analysis_from_interaction(sample['interaction'])
        g['messages'] = [
            {'role': 'system', 'content': SAMPLE_PROMPT},
            {'role': 'user', 'content': json.dumps({'source': 'source', 'analysis': analysis},
                                                  ensure_ascii=False)}]
        sample[arm] = {'verdict': 'SUPPORTED', 'generation': g}
    return {'completed': True, 'results': [sample]}


class RecheckSelectionTests(unittest.TestCase):
    def test_selects_first_sample_both_final_arms_without_gold(self):
        run = run_record()
        extra = copy.deepcopy(run['results'][0])
        extra['id'] = 'second'
        run['results'].append(extra)
        selected = first_sample_traces(run)
        self.assertEqual([x[0]['arm'] for x in selected], ['direct', 'dual'])
        self.assertEqual([x[0]['sample_id'] for x in selected], ['first', 'first'])

    def test_incomplete_empty_or_missing_first_trace_does_not_fall_forward(self):
        for run in [{'completed': False, 'results': []}, {'completed': True, 'results': []}]:
            with self.subTest(run=run), self.assertRaises(ValueError):
                first_sample_traces(run)
        run = run_record()
        run['results'].append(copy.deepcopy(run['results'][0]))
        run['results'][0]['dual']['generation']['token_trace'] = []
        with self.assertRaisesRegex(ValueError, 'do not select another'):
            first_sample_traces(run)

    def test_altered_final_prompt_is_rejected(self):
        run = run_record()
        run['results'][0]['direct']['generation']['messages'][1]['content'] = '{}'
        with self.assertRaisesRegex(ValueError, 'messages differ'):
            first_sample_traces(run)


class RecheckTokenizerTests(unittest.TestCase):
    def test_prompt_ids_decode_eos_special_and_label_flags_all_match(self):
        g = generation()
        prompt, ids, trace = validate_tokenization(g, Tokenizer(), [9])
        self.assertEqual(prompt, [1, 2])
        self.assertEqual(ids, [7, 9])
        self.assertEqual(trace, g['token_trace'])

    def test_corrupted_ids_decodes_eos_or_flags_are_rejected(self):
        bad = []
        value = generation(); value['prompt_input_ids'] = [2, 1]; bad.append(value)
        value = generation(); value['generated_token_ids'] = [True, 9]; bad.append(value)
        value = generation(); value['raw_response'] = 'REFUTED'; bad.append(value)
        value = generation(); value['raw_response_with_special_tokens'] = 'SUPPORTED'; bad.append(value)
        value = generation(); value['ended_with_eos'] = False; bad.append(value)
        value = generation(); value['token_trace'][1]['special'] = False; bad.append(value)
        value = generation(); value['token_trace'][1]['decoded_piece'] = '<eos>'; bad.append(value)
        value = generation(); value['token_trace'][0]['label_token'] = False; bad.append(value)
        value = generation(); value['label_token_alignment'] = 'unavailable'; bad.append(value)
        for index, value in enumerate(bad):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_tokenization(value, Tokenizer(), [9])

    def test_no_eos_status_is_verified_without_inventing_an_eos_token(self):
        g = generation()
        g['generated_token_ids'] = [7]
        g['token_trace'] = g['token_trace'][:1]
        g['raw_response_with_special_tokens'] = 'SUPPORTED'
        g['ended_with_eos'] = False
        _, ids, _ = validate_tokenization(g, Tokenizer(), [9])
        self.assertEqual(ids, [7])


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class RecomputedTraceTests(unittest.TestCase):
    def test_forward_uses_entire_prefix_and_scores_targets_with_correct_offset(self):
        class Model:
            def get_input_embeddings(self):
                return SimpleNamespace(weight=torch.zeros(1))

            def __call__(model, **kwargs):
                model.call = kwargs
                return SimpleNamespace(logits=torch.tensor([
                    [[0., math.log(2), 0.], [0., 0., math.log(3)]]]))

        model = Model()
        fresh = recompute(model, torch, [100, 101, 102], [1, 2])
        self.assertEqual(model.call['input_ids'].tolist(), [[100, 101, 102, 1]])
        self.assertEqual(model.call['attention_mask'].tolist(), [[1, 1, 1, 1]])
        self.assertIs(model.call['use_cache'], False)
        self.assertEqual(model.call['logits_to_keep'], 2)
        self.assertEqual([row['token_id'] for row in fresh], [1, 2])
        self.assertAlmostEqual(fresh[0]['raw_logprob'], math.log(0.5), places=6)
        self.assertAlmostEqual(fresh[1]['raw_logprob'], math.log(0.6), places=6)
        self.assertAlmostEqual(fresh[0]['full_vocab_entropy_nats'],
                               -(0.5 * math.log(0.5) + 0.5 * math.log(0.25)), places=6)


if __name__ == '__main__':
    unittest.main()
