import unittest

from scripts.run_phase1_extraction import align_to_prompt


class RunnerAlignmentIntegrationTests(unittest.TestCase):
    def test_source_relative_chars_map_to_full_prompt_tokens(self):
        prompt = 'User: café is open.'
        source = 'café is open.'
        encoded = {'offset_mapping': [(0, 6), (6, 10), (10, 13), (13, 18), (18, 19)],
                   'special_tokens_mask': [0, 0, 0, 0, 0]}
        location, aligned = align_to_prompt(prompt, source, encoded,
                                            [{'quote_index': 0, 'start_idx': 0, 'end_idx': 4}])
        self.assertEqual(location['start'], 6)
        self.assertEqual(aligned[0]['status'], 'ok')
        self.assertEqual((aligned[0]['token_start'], aligned[0]['token_end']), (1, 2))
        self.assertEqual(aligned[0]['text'], 'café')

    def test_ambiguous_prompt_is_not_guessed(self):
        location, aligned = align_to_prompt('same same', 'same',
                                            {'offset_mapping': [], 'special_tokens_mask': []},
                                            [{'quote_index': 0, 'start_idx': 0, 'end_idx': 4}])
        self.assertEqual(location['status'], 'ambiguous')
        self.assertEqual(aligned[0]['status'], 'source_ambiguous')


if __name__ == '__main__':
    unittest.main()
