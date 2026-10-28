import copy
import unittest

from scripts.recheck_phase2_logits import compare_observations, first_trace


class RecheckTests(unittest.TestCase):
    def test_first_verdict_trace_selection_and_incomplete_run(self):
        g = {'token_trace': [{'step': 0, 'token_id': 5}]}
        run = {'completed': True, 'results': [{'id': 'F1', 'interaction': {'rounds': [
            {'round': 1, 'verifications': [{'quote_index': 2, 'claim': 'x',
                                           'response': {'verdict_generation': g}}]}]}}]}
        identity, found = first_trace(run)
        self.assertEqual((identity['sample_id'], identity['quote_index'], found), ('F1', 2, g))
        run['completed'] = False
        with self.assertRaisesRegex(ValueError, 'complete'):
            first_trace(run)

    def test_token_differences_include_every_token_and_eos(self):
        recorded = [{'step': 0, 'token_id': 10, 'raw_logprob': -0.25, 'full_vocab_entropy_nats': 0.6},
                    {'step': 1, 'token_id': 11, 'special': True,
                     'raw_logprob': -1.0, 'full_vocab_entropy_nats': 0.9}]
        fresh = [{'token_id': 10, 'raw_logprob': -0.3, 'full_vocab_entropy_nats': 0.61},
                 {'token_id': 11, 'raw_logprob': -1.2, 'full_vocab_entropy_nats': 1.0}]
        report = compare_observations([10, 11], recorded, fresh)
        self.assertEqual(len(report['tokens']), 2)
        self.assertAlmostEqual(report['max_absolute_logprob_error'], 0.2)
        self.assertAlmostEqual(report['max_absolute_entropy_error_nats'], 0.1)
        self.assertTrue(report['tokens'][-1]['special'])

    def test_bad_token_id_length_and_nan_fail(self):
        trace = [{'step': 0, 'token_id': 10, 'raw_logprob': -0.2, 'full_vocab_entropy_nats': 0.6}]
        fresh = [{'token_id': 10, 'raw_logprob': -0.3, 'full_vocab_entropy_nats': 0.6}]
        with self.assertRaisesRegex(ValueError, 'lengths'):
            compare_observations([10, 11], trace, fresh)
        bad = copy.deepcopy(fresh)
        bad[0]['token_id'] = 11
        with self.assertRaisesRegex(ValueError, 'token ID mismatch'):
            compare_observations([10], trace, bad)
        bad = copy.deepcopy(fresh)
        bad[0]['full_vocab_entropy_nats'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            compare_observations([10], trace, bad)


if __name__ == '__main__':
    unittest.main()
