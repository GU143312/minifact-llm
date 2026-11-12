import math
import unittest

from phase3_metrics import SELECTIVE_THRESHOLDS, compare_paired, evaluate


def gold(labels):
    return [{'id': str(i), 'label': label} for i, label in enumerate(labels)]


def predictions(labels, scores=None):
    if scores is None:
        scores = [-0.1] * len(labels)
    return [{'id': str(i), 'verdict': label, 'mean_label_token_logprob': score,
             'mean_label_token_full_vocab_entropy_nats': 0.2}
            for i, (label, score) in enumerate(zip(labels, scores))]


class Phase3MetricsTests(unittest.TestCase):
    def test_abstentions_and_invalid_errors_are_not_hidden_in_fnr(self):
        truth = gold(['REFUTED'] * 4 + ['SUPPORTED'] * 4)
        output = predictions(['REFUTED', 'SUPPORTED', 'NOT_ENOUGH_INFO', None,
                              'SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO', 'garbage'])
        result = evaluate(truth, output)
        self.assertEqual(result['confusion_matrix']['REFUTED'],
                         {'SUPPORTED': 1, 'REFUTED': 1, 'NOT_ENOUGH_INFO': 1, 'INVALID': 1})
        self.assertEqual(result['accuracy'], 2 / 8)
        self.assertEqual(result['binary_coverage'], 4 / 8)
        self.assertEqual(result['accuracy_on_binary_decisions'], 2 / 4)
        self.assertEqual(result['false_negative_rate'], 1 / 4)
        self.assertEqual(result['error_recall'], 1 / 4)
        self.assertEqual(result['missed_error_rate_including_abstentions'], 3 / 4)
        self.assertEqual(result['false_positive_rate'], 1 / 4)
        self.assertEqual(result['abstained_errors'], 1)
        self.assertEqual(result['invalid_errors'], 1)
        self.assertEqual(result['invalid_by_gold']['SUPPORTED'], 1)

    def test_all_abstain_has_no_binary_accuracy_but_zero_overall_accuracy(self):
        result = evaluate(gold(['SUPPORTED', 'REFUTED']),
                          predictions(['NOT_ENOUGH_INFO', None]))
        self.assertEqual(result['accuracy'], 0)
        self.assertIsNone(result['accuracy_on_binary_decisions'])
        self.assertEqual(result['binary_coverage'], 0)
        self.assertEqual(result['false_negative_rate'], 0)
        self.assertEqual(result['missed_error_rate_including_abstentions'], 1)

    def test_single_class_and_empty_denominators_are_explicit_null(self):
        result = evaluate(gold(['REFUTED']), predictions(['REFUTED']))
        self.assertIsNone(result['false_positive_rate'])
        self.assertIsNone(result['supported_recall'])
        result = evaluate(gold(['SUPPORTED']), predictions(['SUPPORTED']))
        self.assertIsNone(result['false_negative_rate'])
        self.assertIsNone(result['error_recall'])
        self.assertIsNone(result['missed_error_rate_including_abstentions'])
        empty = evaluate([], [])
        self.assertIsNone(empty['accuracy'])
        self.assertIsNone(empty['binary_coverage'])
        self.assertIsNone(empty['selective_results'][0]['coverage'])

    def test_prediction_order_does_not_change_matching(self):
        truth = gold(['SUPPORTED', 'REFUTED'])
        output = predictions(['SUPPORTED', 'REFUTED'])
        self.assertEqual(evaluate(truth, output), evaluate(truth, list(reversed(output))))

    def test_missing_extra_and_duplicate_ids_are_rejected(self):
        truth = gold(['SUPPORTED', 'REFUTED'])
        output = predictions(['SUPPORTED', 'REFUTED'])
        for rows in [output[:1], output + [{'id': 'extra', 'verdict': None}],
                     [output[0], output[0]]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate(truth, rows)
        with self.assertRaises(ValueError):
            evaluate([truth[0], truth[0]], output)

    def test_invalid_gold_and_missing_verdict_are_rejected(self):
        for row in [{'id': 'a', 'label': 'NOT_ENOUGH_INFO'},
                    {'id': 'a', 'label': 'Refuted'}, {'id': '', 'label': 'REFUTED'}, None]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                evaluate([row], [{'id': 'a', 'verdict': 'REFUTED'}])
        with self.assertRaises(ValueError):
            evaluate(gold(['SUPPORTED']), [{'id': '0'}])

    def test_threshold_is_inclusive_and_only_binary_finite_scores_are_selected(self):
        truth = gold(['REFUTED', 'REFUTED', 'SUPPORTED', 'SUPPORTED', 'REFUTED'])
        output = predictions(['REFUTED', 'SUPPORTED', 'SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO'],
                             [-0.1, -0.1000001, -0.05, math.nan, -0.001])
        result = evaluate(truth, output)
        selected = result['selective_results'][1]
        self.assertEqual(selected['minimum_mean_label_token_logprob'], -0.1)
        self.assertEqual(selected['retained_binary_decisions'], 2)
        self.assertEqual(selected['accuracy_on_retained_decisions'], 1)
        self.assertEqual(selected['coverage'], 2 / 5)
        self.assertEqual(selected['detected_errors'], 1)
        self.assertEqual(selected['false_negatives_supported'], 0)
        self.assertEqual(selected['abstained_or_invalid_errors'], 2)
        self.assertEqual(selected['missed_error_rate_including_abstentions'], 2 / 3)
        self.assertEqual(tuple(row['minimum_mean_label_token_logprob']
                               for row in result['selective_results']), SELECTIVE_THRESHOLDS)

    def test_bad_score_types_and_positive_logprobs_are_not_selected(self):
        scores = [True, '0', math.inf, -math.inf, math.nan, 0.2, None]
        result = evaluate(gold(['SUPPORTED'] * len(scores)),
                          predictions(['SUPPORTED'] * len(scores), scores))
        self.assertTrue(all(row['retained_binary_decisions'] == 0
                            for row in result['selective_results']))
        self.assertEqual(result['score_summaries']['mean_label_token_logprob']['samples_with_valid_score'], 0)
        self.assertIsNone(result['score_summaries']['mean_label_token_logprob']['sample_mean_of_label_token_means'])

    def test_score_mean_is_explicitly_sample_weighted(self):
        output = predictions(['SUPPORTED', 'REFUTED'], [-0.1, -0.3])
        output[0]['label_token_count'] = 1
        output[1]['label_token_count'] = 10
        result = evaluate(gold(['SUPPORTED', 'REFUTED']), output)
        summary = result['score_summaries']['mean_label_token_logprob']
        self.assertAlmostEqual(summary['sample_mean_of_label_token_means'], -0.2)
        self.assertEqual(summary['samples_with_valid_score'], 2)
        output[1]['mean_label_token_full_vocab_entropy_nats'] = -0.1
        result = evaluate(gold(['SUPPORTED', 'REFUTED']), output)
        entropy = result['score_summaries']['mean_label_token_full_vocab_entropy_nats']
        self.assertEqual(entropy['samples_with_valid_score'], 1)
        self.assertEqual(entropy['sample_mean_of_label_token_means'], 0.2)

    def test_all_correct_and_all_wrong(self):
        truth = gold(['SUPPORTED', 'REFUTED'])
        good = evaluate(truth, predictions(['SUPPORTED', 'REFUTED']))
        bad = evaluate(truth, predictions(['REFUTED', 'SUPPORTED']))
        self.assertEqual((good['accuracy'], good['false_negative_rate'], good['false_positive_rate']), (1, 0, 0))
        self.assertEqual((bad['accuracy'], bad['false_negative_rate'], bad['false_positive_rate']), (0, 1, 1))

    def test_paired_counts_include_abstention_as_incorrect(self):
        truth = gold(['SUPPORTED', 'REFUTED', 'SUPPORTED', 'REFUTED'])
        baseline = predictions(['SUPPORTED', 'SUPPORTED', 'NOT_ENOUGH_INFO', 'REFUTED'])
        dual = predictions(['SUPPORTED', 'REFUTED', None, 'SUPPORTED'])
        paired = compare_paired(truth, baseline, dual)
        self.assertEqual(paired['both_correct'], 1)
        self.assertEqual(paired['both_incorrect'], 1)
        self.assertEqual(paired['baseline_only_correct'], 1)
        self.assertEqual(paired['dual_only_correct'], 1)
        self.assertEqual(paired['accuracy_difference_dual_minus_baseline'], 0)
        self.assertEqual(paired['changed_verdicts'], 3)
        self.assertEqual(paired['mcnemar_exact_two_sided_p'], 1)

    def test_exact_mcnemar_and_generator_inputs(self):
        truth = gold(['REFUTED'] * 6)
        baseline = predictions(['SUPPORTED'] * 6)
        dual = predictions(['REFUTED'] * 6)
        paired = compare_paired(iter(truth), iter(baseline), iter(dual))
        self.assertEqual(paired['dual_only_correct'], 6)
        self.assertEqual(paired['accuracy_difference_dual_minus_baseline'], 1)
        self.assertEqual(paired['mcnemar_exact_two_sided_p'], 2 / 64)
        equal = compare_paired(truth, dual, dual)
        self.assertEqual(equal['mcnemar_exact_two_sided_p'], 1)

    def test_paired_requires_both_arms_to_have_exact_gold_ids(self):
        truth = gold(['REFUTED'])
        with self.assertRaises(ValueError):
            compare_paired(truth, predictions(['REFUTED']), [])
        empty = compare_paired([], [], [])
        self.assertIsNone(empty['accuracy_difference_dual_minus_baseline'])


if __name__ == '__main__':
    unittest.main()
