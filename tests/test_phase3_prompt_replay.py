"""Token replay tests without a real tokenizer, model, GPU or network."""
import copy
import unittest

from scripts.replay_phase3_prompts import replay_generation, replay_run


class FakeTokenizer:
    all_special_ids = [99]

    def apply_chat_template(self, messages, **kwargs):
        if kwargs != {'add_generation_prompt': True, 'tokenize': True}:
            raise AssertionError('Wrong chat template flags')
        return [ord(character) for message in messages for character in message['content']]

    def decode(self, ids, skip_special_tokens):
        return ''.join(
            ('' if skip_special_tokens else '<eos>') if token == 99 else chr(token) for token in ids)


def generation():
    return {'messages': [{'role': 'system', 'content': 'check'}, {'role': 'user', 'content': 'source'}],
            'prompt_input_ids': [ord(character) for character in 'checksource'],
            'generated_token_ids': [65, 99], 'raw_response': 'A',
            'raw_response_with_special_tokens': 'A<eos>', 'ended_with_eos': True,
            'token_trace': [{'step': 0, 'token_id': 65, 'decoded_piece': 'A', 'special': False},
                            {'step': 1, 'token_id': 99, 'decoded_piece': '', 'special': True}]}


def run_record():
    g = generation()
    return {'completed': True, 'execution': {'model_generation_config': {'eos_token_id': [99]}},
            'results': [{'id': 'sample', 'direct': {'generation': copy.deepcopy(g)},
                         'dual': {'generation': copy.deepcopy(g)},
                         'interaction': {'rounds': [{
                             'extract_response': copy.deepcopy(g),
                             'verifications': [{'response': {'verdict_generation': copy.deepcopy(g),
                                                             'feedback_generation': copy.deepcopy(g)}}],
                             'source_logic_review': {'response': {'generation': copy.deepcopy(g)}}}]}}]}


class Phase3PromptReplayTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_all_roles_are_replayed_and_counted(self):
        result = replay_run(run_record(), self.tokenizer)
        self.assertEqual(result['generation_calls_checked'], 6)
        self.assertEqual(result['calls_by_role'], {
            'direct/sample_judge': 1, 'dual/sample_judge': 1, 'dual/extract': 1,
            'dual/verdict_generation': 1, 'dual/feedback_generation': 1, 'dual/source_logic': 1})
        self.assertEqual(result['trace_token_pieces_checked'], 12)
        self.assertEqual(result['generated_tokens_checked_including_eos'], 12)

    def test_missing_and_invalid_prompt_ids_fail(self):
        for ids in [None, [], [True], [-1], '123']:
            g = generation()
            g['prompt_input_ids'] = ids
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'invalid token IDs'):
                replay_generation(g, self.tokenizer, [99], 'test')

    def test_modified_prompt_text_or_ids_fail(self):
        for field in ['messages', 'prompt_input_ids']:
            g = generation()
            if field == 'messages':
                g[field][1]['content'] = 'altered source'
            else:
                g[field][-1] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'prompt token IDs differ'):
                replay_generation(g, self.tokenizer, [99], 'test')

    def test_output_decode_and_special_decode_are_both_checked(self):
        for key in ['raw_response', 'raw_response_with_special_tokens']:
            g = generation()
            g[key] = 'modified'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'decoded output|special-token output'):
                replay_generation(g, self.tokenizer, [99], 'test')

    def test_generated_ids_and_eos_status_are_validated(self):
        g = generation()
        g['generated_token_ids'] = []
        with self.assertRaisesRegex(ValueError, 'invalid token IDs'):
            replay_generation(g, self.tokenizer, [99], 'test')
        g = generation()
        g['ended_with_eos'] = False
        with self.assertRaisesRegex(ValueError, 'EOS flag'):
            replay_generation(g, self.tokenizer, [99], 'test')

    def test_trace_piece_and_special_flags_are_checked_against_tokenizer(self):
        for key, value in [('special', False), ('decoded_piece', 'modified'), ('token_id', 100)]:
            g = generation()
            g['token_trace'][1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'trace'):
                replay_generation(g, self.tokenizer, [99], 'test')

    def test_untraced_generation_is_still_replayed(self):
        g = generation()
        del g['token_trace']
        result = replay_generation(g, self.tokenizer, [99], 'test')
        self.assertEqual(result['trace_tokens'], 0)
        self.assertEqual(result['prompt_tokens'], len('checksource'))

    def test_incomplete_empty_and_missing_arm_records_fail(self):
        for change in [('completed', False), ('results', [])]:
            run = run_record()
            run[change[0]] = change[1]
            with self.subTest(change=change), self.assertRaises(ValueError):
                replay_run(run, self.tokenizer)
        run = run_record()
        del run['results'][0]['direct']['generation']
        with self.assertRaisesRegex(ValueError, 'missing generation'):
            replay_run(run, self.tokenizer)

    def test_internal_modified_prompt_is_not_skipped(self):
        run = run_record()
        run['results'][0]['interaction']['rounds'][0]['source_logic_review']['response'][
            'generation']['prompt_input_ids'][0] = 1
        with self.assertRaisesRegex(ValueError, 'source_logic.*prompt token IDs differ'):
            replay_run(run, self.tokenizer)


if __name__ == '__main__':
    unittest.main()
