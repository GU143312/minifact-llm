import unittest

from phase2_pipeline import run_sample


class Backend:
    def __init__(self, outputs, responses):
        self.outputs = iter(outputs)
        self.responses = responses
        self.extract_calls = []
        self.verify_calls = []

    def extract(self, source, feedback=None):
        self.extract_calls.append((source, feedback))
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value

    def verify(self, source, claim):
        self.verify_calls.append((source, claim))
        value = self.responses[claim]
        if isinstance(value, Exception):
            raise value
        return value


class InteractionTests(unittest.TestCase):
    def test_feedback_enters_second_extractor_with_original_verifier_output(self):
        b = Backend([{'raw_quotes': ['Alice left']}, {'raw_quotes': ['Alice left today']}],
                    {'Alice left': {'verdict': 'NOT_ENOUGH_INFO', 'feedback': 'Consider the date.'},
                     'Alice left today': {'verdict': 'SUPPORTED', 'feedback': ''}})
        result = run_sample('Alice left today', b)
        self.assertEqual(result['stop_reason'], 'no_actionable_feedback')
        self.assertIsNone(b.extract_calls[0][1])
        feedback = b.extract_calls[1][1]
        self.assertEqual(feedback['origin'], 'verifier_model_output')
        self.assertEqual(feedback['entries'][0]['verifier_feedback'], 'Consider the date.')
        self.assertEqual(result['rounds'][1]['extract_request']['feedback'], feedback)
        self.assertEqual(len(b.verify_calls), 2)

    def test_nonliteral_duplicate_and_ambiguous_are_preserved_and_not_verified(self):
        b = Backend([{'raw_quotes': ['Alice left', 'Alice left', 'Bob left', 'left']}],
                    {'Alice left': {'verdict': 'SUPPORTED', 'feedback': ''}})
        result = run_sample('Alice left and left again', b)
        step = result['rounds'][0]
        self.assertEqual([x['claim'] for x in step['accepted_claims']], ['Alice left'])
        self.assertEqual([x['reason'] for x in step['alignment_rejections']],
                         ['duplicate_prediction', 'no_literal_match', 'ambiguous_repeated_quote'])
        self.assertEqual(b.verify_calls, [('Alice left and left again', 'Alice left')])
        self.assertEqual(step['extract_response']['raw_quotes'][2], 'Bob left')

    def test_invalid_json_and_no_valid_claims_stop_without_guessing(self):
        b = Backend([{'raw_response': '[not json'}], {})
        result = run_sample('A claim', b)
        self.assertEqual(result['stop_reason'], 'malformed_extraction')
        self.assertEqual(result['rounds'][0]['extract_response']['raw_response'], '[not json')
        self.assertEqual(b.verify_calls, [])
        b = Backend([{'raw_quotes': ['rewritten']}], {})
        result = run_sample('A claim', b)
        self.assertEqual(result['stop_reason'], 'no_valid_claims')
        self.assertEqual(result['rounds'][0]['alignment_rejections'][0]['reason'], 'no_literal_match')

    def test_empty_feedback_and_uncertainty_do_not_invent_model_explanation(self):
        b = Backend([{'raw_quotes': ['A claim']}],
                    {'A claim': {'verdict': 'NOT_ENOUGH_INFO', 'feedback': ''}})
        result = run_sample('A claim', b)
        self.assertEqual(result['stop_reason'], 'no_actionable_feedback')
        self.assertIsNone(result['rounds'][0]['next_feedback'])
        self.assertEqual(len(b.extract_calls), 1)

    def test_supported_verdict_with_feedback_also_reaches_next_round(self):
        b = Backend([{'raw_quotes': ['A claim']}, {'raw_quotes': ['A claim']}],
                    {'A claim': {'verdict': 'SUPPORTED', 'feedback': 'Keep the subject.'}})
        result = run_sample('A claim', b)
        self.assertEqual(result['stop_reason'], 'repeated_extraction')
        self.assertEqual(b.extract_calls[1][1]['entries'][0]['verifier_feedback'], 'Keep the subject.')

    def test_repeated_extraction_and_round_limit(self):
        response = {'verdict': 'REFUTED', 'feedback': 'Review this quotation.'}
        b = Backend([{'raw_quotes': ['A claim']}, {'raw_quotes': ['A claim']}], {'A claim': response})
        result = run_sample('A claim', b)
        self.assertEqual(result['stop_reason'], 'repeated_extraction')
        self.assertEqual(len(result['rounds'][1]['verifications']), 1)
        b = Backend([{'raw_quotes': ['A claim']}], {'A claim': response})
        result = run_sample('A claim', b, max_rounds=1)
        self.assertEqual(result['stop_reason'], 'round_limit')
        self.assertIsNotNone(result['rounds'][0]['next_feedback'])

    def test_verifier_error_is_recorded_and_other_claims_are_still_checked(self):
        b = Backend([{'raw_quotes': ['Alice left', 'Bob stayed']}],
                    {'Alice left': RuntimeError('verification failed'),
                     'Bob stayed': {'verdict': 'SUPPORTED', 'feedback': ''}})
        result = run_sample('Alice left; Bob stayed', b)
        self.assertEqual(result['stop_reason'], 'verifier_failure')
        self.assertEqual(result['rounds'][0]['verifications'][0]['error']['type'], 'RuntimeError')
        self.assertEqual(len(b.verify_calls), 2)
        self.assertIsNone(result['rounds'][0]['next_feedback'])

    def test_invalid_verdict_and_extractor_exception(self):
        b = Backend([{'raw_quotes': ['A claim']}], {'A claim': {'verdict': 'MAYBE'}})
        result = run_sample('A claim', b)
        self.assertEqual(result['rounds'][0]['verifications'][0]['status'], 'invalid_verdict')
        self.assertEqual(result['stop_reason'], 'verifier_failure')
        b = Backend([RuntimeError('offline')], {})
        result = run_sample('A claim', b)
        self.assertEqual(result['stop_reason'], 'extractor_exception')
        self.assertEqual(result['rounds'][0]['extract_error']['message'], 'offline')

    def test_round_bounds_and_input_are_checked(self):
        for n in [0, 3, True]:
            with self.assertRaises(ValueError):
                run_sample('A claim', Backend([], {}), max_rounds=n)
        with self.assertRaises(ValueError):
            run_sample('', Backend([], {}))

    def test_source_logic_feedback_reaches_next_extraction(self):
        class LogicBackend(Backend):
            def review_source(self, source, claims):
                self.review_call = (source, claims)
                return {'feedback': 'The two dates conflict.', 'generation': {
                    'raw_response': 'The two dates conflict.', 'ended_with_eos': True}}

        b = LogicBackend([{'raw_quotes': ['The event was in May']},
                          {'raw_quotes': ['The event was in May']}],
                         {'The event was in May': {'verdict': 'SUPPORTED', 'feedback': ''}})
        result = run_sample('The event was in May and in June', b)
        self.assertEqual(b.review_call, ('The event was in May and in June', ['The event was in May']))
        review = result['rounds'][0]['source_logic_review']
        self.assertEqual(review['status'], 'ok')
        self.assertEqual(review['response']['generation']['raw_response'], 'The two dates conflict.')
        sent = b.extract_calls[1][1]['source_logic_feedback']
        self.assertEqual(sent['origin'], 'source_logic_model_output')
        self.assertEqual(sent['text'], 'The two dates conflict.')
        self.assertIn('Unverified', sent['warning'])
        self.assertEqual(result['stop_reason'], 'repeated_extraction')

    def test_source_review_optional_and_exception_preserved(self):
        b = Backend([{'raw_quotes': ['A claim']}], {'A claim': {'verdict': 'SUPPORTED', 'feedback': ''}})
        result = run_sample('A claim', b)
        self.assertEqual(result['rounds'][0]['source_logic_review']['status'], 'unavailable')
        class FailedBackend(Backend):
            def review_source(self, source, claims):
                raise RuntimeError('logic review failed')
        b = FailedBackend([{'raw_quotes': ['A claim']}], {'A claim': {'verdict': 'SUPPORTED', 'feedback': ''}})
        result = run_sample('A claim', b)
        review = result['rounds'][0]['source_logic_review']
        self.assertEqual(review['status'], 'exception')
        self.assertEqual(review['error']['message'], 'logic review failed')
        self.assertIsNone(result['rounds'][0]['next_feedback'])

    def test_source_review_truncated_and_empty_are_never_fabricated(self):
        class LogicBackend(Backend):
            def __init__(self, review):
                super().__init__([{'raw_quotes': ['A claim']}],
                                 {'A claim': {'verdict': 'SUPPORTED', 'feedback': ''}})
                self.review = review
            def review_source(self, source, claims):
                return self.review
        examples = [({'feedback': None, 'generation': {'raw_response': 'unfinished', 'ended_with_eos': False}}, 'truncated'),
                    ({'feedback': None, 'generation': {'raw_response': 'NO_CONTRADICTION', 'ended_with_eos': True}}, 'no_feedback'),
                    ({'feedback': None, 'generation': {'raw_response': ' no_contradiction. ', 'ended_with_eos': True}}, 'no_feedback'),
                    ({'feedback': 'fabricated', 'generation': {'raw_response': 'unfinished', 'ended_with_eos': False}}, 'invalid_response')]
        for response, status in examples:
            with self.subTest(status=status):
                result = run_sample('A claim', LogicBackend(response))
                step = result['rounds'][0]
                self.assertEqual(step['source_logic_review']['status'], status)
                self.assertIsNone(step['next_feedback'])
                self.assertEqual(step['source_logic_review']['response'], response)

    def test_sentinel_followed_by_explanation_is_retained(self):
        class LogicBackend(Backend):
            def review_source(self, source, claims):
                note = 'NO_CONTRADICTION. The dates nevertheless conflict.'
                return {'feedback': note, 'generation': {'raw_response': note, 'ended_with_eos': True}}
        b = LogicBackend([{'raw_quotes': ['A claim']}, {'raw_quotes': ['A claim']}],
                         {'A claim': {'verdict': 'SUPPORTED', 'feedback': None}})
        result = run_sample('A claim', b)
        self.assertEqual(result['rounds'][0]['source_logic_review']['status'], 'ok')
        self.assertEqual(b.extract_calls[1][1]['source_logic_feedback']['text'],
                         'NO_CONTRADICTION. The dates nevertheless conflict.')

    def test_source_review_can_request_retry_after_no_valid_claims(self):
        class LogicBackend(Backend):
            def review_source(self, source, claims):
                return {'feedback': 'Check the conflicting dates.', 'generation': {
                    'raw_response': 'Check the conflicting dates.', 'ended_with_eos': True}}
        b = LogicBackend([{'raw_quotes': ['rewritten']}, {'raw_quotes': ['A claim']}],
                         {'A claim': {'verdict': 'SUPPORTED', 'feedback': ''}})
        result = run_sample('A claim', b)
        self.assertEqual(result['rounds'][0]['source_logic_review']['request']['claims'], [])
        self.assertEqual(b.extract_calls[1][1]['source_logic_feedback']['text'], 'Check the conflicting dates.')
        self.assertEqual(result['rounds'][0]['alignment_rejections'][0]['reason'], 'no_literal_match')
        self.assertEqual(result['stop_reason'], 'round_limit')


if __name__ == '__main__':
    unittest.main()
