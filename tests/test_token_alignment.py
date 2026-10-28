import os
import unittest

from token_alignment import encode_prompt_offsets, locate_unique_source, map_char_span


class AlignmentTests(unittest.TestCase):
    def test_overlapping_occurrences_are_ambiguous(self):
        self.assertEqual(locate_unique_source('aaaa', 'aa'),
                         {'status': 'ambiguous', 'occurrences': [0, 1, 2]})
        self.assertEqual(locate_unique_source('abc', 'z')['status'], 'not_found')
        self.assertEqual(locate_unique_source('prefix 花 suffix', '花')['start'], 7)

    def test_special_tokens_unicode_and_duplicate_offsets(self):
        text = 'A猫🙂B'
        offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (2, 3), (3, 4), (0, 0)]
        result = map_char_span(text, 1, 3, offsets, [1, 0, 0, 0, 0, 0, 1])
        self.assertEqual((result['status'], result['token_start'], result['token_end']), ('ok', 2, 5))
        self.assertEqual(result['text'], '猫🙂')

    def test_partial_token_and_uncovered_gap(self):
        result = map_char_span('abcdef', 1, 3, [(0, 3), (3, 6)])
        self.assertEqual(result['status'], 'partial_token')
        self.assertEqual((result['covered_char_start'], result['covered_char_end']), (0, 3))
        self.assertEqual(map_char_span('a b', 0, 3, [(0, 1), (2, 3)])['status'], 'uncovered')
        self.assertEqual(map_char_span('a b', 1, 2, [(0, 1), (2, 3)])['status'], 'uncovered')

    def test_special_token_cannot_be_silently_inside_interval(self):
        result = map_char_span('ab', 0, 2, [(0, 1), (0, 0), (1, 2)], [0, 1, 0])
        self.assertEqual(result['status'], 'non_contiguous')

    def test_bad_ranges_and_offsets_fail(self):
        for start, end in [(0, 0), (-1, 1), (0, 4)]:
            with self.assertRaises(ValueError):
                map_char_span('abc', start, end, [(0, 1)])
        with self.assertRaises(ValueError):
            map_char_span('abc', 0, 1, [(0, 4)])
        with self.assertRaises(ValueError):
            map_char_span('abc', 0, 1, [(0, 1)], [])

    def test_actual_model_ids_must_match(self):
        class Tokenizer:
            is_fast = True
            def __call__(self, prompt, **kwargs):
                return {'input_ids': [4, 5], 'offset_mapping': [(0, 1), (1, 2)],
                        'special_tokens_mask': [0, 0]}
        self.assertEqual(encode_prompt_offsets(Tokenizer(), 'ab', [[4, 5]])['input_ids'], [4, 5])
        with self.assertRaisesRegex(ValueError, 'differ'):
            encode_prompt_offsets(Tokenizer(), 'ab', [4, 6])

    @unittest.skipUnless(os.environ.get('MINIFACT_TEST_TOKENIZER'), 'local tokenizer path not supplied')
    def test_real_qwen_fast_chat_template(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.environ['MINIFACT_TEST_TOKENIZER'],
                                             use_fast=True, local_files_only=True)
        source = 'Lina visited 北京 in May.'
        messages = [{'role': 'user', 'content': source}]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        encoded = encode_prompt_offsets(tok, rendered, model_ids)
        loc = locate_unique_source(rendered, source)
        self.assertEqual(loc['status'], 'ok')
        start = loc['start'] + source.index('北京')
        result = map_char_span(rendered, start, start + len('北京'), encoded['offset_mapping'],
                               encoded['special_tokens_mask'])
        # Qwen's leading-space token overlaps this mention; the tool must not
        # silently report an exact token boundary for the shorter substring.
        self.assertEqual(result['status'], 'partial_token')
        self.assertEqual(result['text'], '北京')
        whole = map_char_span(rendered, start - 1, start + len('北京'), encoded['offset_mapping'],
                              encoded['special_tokens_mask'])
        self.assertEqual((whole['status'], whole['text']), ('ok', ' 北京'))


if __name__ == '__main__':
    unittest.main()
