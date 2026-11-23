"""Preserve evidence spans separately from unverified claim reconstruction."""
import json
import unittest

from accuracy_claims import EXTRACTION_PROMPT, parse_extraction
from minifact import exact_span


def prediction(quote, proposition=None, contexts=None):
    return {'quote': quote, 'proposition': proposition,
            'context_quotes': [] if contexts is None else contexts}


class ContextualExtractionTests(unittest.TestCase):
    def parse(self, source, *items):
        return parse_extraction(source, json.dumps(list(items), ensure_ascii=False))

    def test_shared_predicate_keeps_literal_quote_and_distinct_reconstruction(self):
        source = 'Alice owns three boats and four cars.'
        result = self.parse(source,
            prediction('Alice owns three boats'),
            prediction('four cars', 'Alice owns four cars', ['Alice owns']))
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['rejected'], [])
        self.assertEqual(len(result['accepted']), 2)
        for claim in result['accepted']:
            self.assertTrue(exact_span(source, claim))
            self.assertTrue(all(exact_span(source, c) for c in claim['context_dependencies']))
        second = result['accepted'][1]
        self.assertEqual(second['claim'], 'four cars')
        self.assertEqual(second['proposition'], 'Alice owns four cars')
        self.assertNotIn(second['proposition'], source)
        self.assertEqual(second['proposition_status'], 'unverified_model_reconstruction')
        self.assertIn('declared_context_dependency', second['guard_flags'])

    def test_order_complement_omission_is_flagged_without_dropping_quote(self):
        source = 'Morgan issued an executive order to reopen the harbor on Monday.'
        result = self.parse(source, prediction('Morgan issued an executive order'))
        self.assertEqual(len(result['accepted']), 1)
        self.assertIn('possible_missing_complement', result['accepted'][0]['guard_flags'])
        self.assertEqual(result['rejected'], [])

    def test_complete_order_content_and_time_remain_literal(self):
        source = 'Morgan issued an executive order to reopen the harbor on Monday.'
        result = self.parse(source, prediction(source[:-1]))
        claim = result['accepted'][0]
        self.assertNotIn('possible_missing_complement', claim['guard_flags'])
        self.assertEqual(claim['proposition'], source[:-1])
        self.assertEqual(claim['proposition_status'], 'literal_quote_unverified_semantics')

    def test_dates_and_negation_are_not_silently_removed_from_reconstruction(self):
        source = 'The ferry did not sail from 12 May to 15 May.'
        result = self.parse(source, prediction(source[:-1], 'The ferry sailed on 12 May'))
        claim = result['accepted'][0]
        self.assertIn('possible_negation_loss', claim['guard_flags'])
        self.assertIn('possible_omitted_numeric_literal', claim['guard_flags'])
        self.assertEqual(claim['semantic_status'], 'not_independently_verified')
        self.assertTrue(exact_span(source, claim))

    def test_context_can_supply_time_and_subject_without_rewriting_span(self):
        source = 'On 12 May, Ada departed. She returned on 15 May.'
        result = self.parse(source, prediction('She returned on 15 May',
                                             'Ada returned on 15 May', ['Ada departed']))
        claim = result['accepted'][0]
        self.assertIn('possible_pronoun_dependency', claim['guard_flags'])
        self.assertEqual(claim['context_dependencies'][0]['claim'], 'Ada departed')
        self.assertTrue(exact_span(source, claim))

    def test_nonliteral_main_quote_is_rejected_even_if_proposition_is_plausible(self):
        result = self.parse('Ada hired three nurses and opened a ward.',
                            prediction('Ada opened a ward'))
        self.assertEqual(result['accepted'], [])
        self.assertEqual(result['rejected'][0]['reason'], 'no_literal_match')

    def test_nonliteral_or_ambiguous_context_rejects_the_whole_item(self):
        cases = [
            ('Ada opened a ward.', prediction('opened a ward', 'Ada opened a ward', ['Ada hired'])),
            ('Ada left. Ada returned.', prediction('Ada returned', 'Ada returned', ['Ada']))]
        for source, item in cases:
            with self.subTest(source=source):
                result = self.parse(source, item)
                self.assertEqual(result['accepted'], [])
                self.assertEqual(result['rejected'][0]['reason'], 'invalid_context_alignment')
                self.assertEqual(result['rejected'][0]['item'], item)

    def test_repeated_quote_is_not_given_a_guessed_position(self):
        result = self.parse('The hall has 80 seats. The hall has 80 seats.',
                            prediction('The hall has 80 seats'))
        self.assertEqual(result['accepted'], [])
        self.assertEqual(result['rejected'][0]['reason'], 'ambiguous_repeated_quote')
        self.assertEqual(len(result['rejected'][0]['matches']), 2)

    def test_duplicate_main_quote_is_rejected_but_shared_context_is_allowed(self):
        source = 'Ada owns two boats and three cars.'
        first = prediction('two boats', 'Ada owns two boats', ['Ada owns'])
        result = self.parse(source, first, prediction('three cars', 'Ada owns three cars', ['Ada owns']), first)
        self.assertEqual(len(result['accepted']), 2)
        self.assertEqual(result['rejected'][0]['index'], 2)
        self.assertEqual(result['rejected'][0]['reason'], 'duplicate_prediction')

    def test_added_number_is_flagged_not_treated_as_a_literal_fact(self):
        source = 'The harbor opened in May.'
        result = self.parse(source, prediction(source[:-1], 'The harbor opened on 12 May'))
        claim = result['accepted'][0]
        self.assertIn('possible_added_numeric_literal', claim['guard_flags'])
        self.assertEqual(claim['proposition_status'], 'unverified_model_reconstruction')

    def test_derived_positions_cannot_be_supplied_by_the_model(self):
        item = {**prediction('Ada left'), 'start_idx': 99}
        result = self.parse('Ada left.', item)
        self.assertEqual(result['accepted'], [])
        self.assertEqual(result['rejected'][0]['reason'], 'invalid_item_fields')

    def test_unicode_indices_are_python_codepoints(self):
        source = '🛳️新港有12艘船，旧港关闭。'
        result = self.parse(source, prediction('新港有12艘船'))
        claim = result['accepted'][0]
        self.assertEqual(claim['start_idx'], 2)
        self.assertEqual(claim['end_idx'], 9)
        self.assertTrue(exact_span(source, claim))

    def test_malformed_top_level_and_item_types_are_explicit(self):
        for raw in ['```json\n[]\n```', '{}', '[]', 'null', None]:
            with self.subTest(raw=raw):
                result = parse_extraction('Ada left.', raw)
                self.assertEqual(result['accepted'], [])
                self.assertTrue(result['errors'])
        malformed = [prediction('', None), prediction('Ada left', 5),
                     prediction('Ada left', None, ['']), 'Ada left']
        result = self.parse('Ada left.', *malformed)
        self.assertEqual(result['accepted'], [])
        self.assertEqual(len(result['rejected']), len(malformed))
        self.assertEqual(result['raw_items'], malformed)

    def test_valid_schema_is_not_a_proof_of_atomicity(self):
        source = 'Ada left and Bob arrived.'
        result = self.parse(source, prediction(source[:-1]))
        self.assertEqual(len(result['accepted']), 1)
        self.assertEqual(result['accepted'][0]['semantic_status'], 'not_independently_verified')

    def test_prompt_protects_complements_negation_and_distinguishes_reconstruction(self):
        self.assertIn('predicate, object or complement', EXTRACTION_PROMPT)
        self.assertIn('negation', EXTRACTION_PROMPT)
        self.assertIn('proposition: null', EXTRACTION_PROMPT)
        self.assertIn('unverified model output', EXTRACTION_PROMPT)


if __name__ == '__main__':
    unittest.main()
