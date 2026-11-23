"""Sample-level agreement with benchmark labels, keeping abstention explicit.

REFUTED is the error-positive class. A generated-token log-probability is a
model score, not a calibrated probability that a statement is true. Gold labels
apply to whole samples; this module does not evaluate extracted atomic facts.
"""
import math
from collections import Counter


BINARY_LABELS = ('SUPPORTED', 'REFUTED')
PREDICTION_LABELS = (*BINARY_LABELS, 'NOT_ENOUGH_INFO', 'INVALID')
SELECTIVE_THRESHOLDS = (-0.05, -0.1, -0.25, -0.5)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _indexed(rows, kind):
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f'{kind} rows must be dictionaries')
        sample_id = row.get('id')
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f'{kind} IDs must be nonempty strings')
        if sample_id in result:
            raise ValueError(f'Duplicate {kind} ID: {sample_id}')
        if kind == 'gold' and row.get('label') not in BINARY_LABELS:
            raise ValueError(f'Gold sample {sample_id} must have SUPPORTED or REFUTED label')
        if kind == 'prediction' and 'verdict' not in row:
            raise ValueError(f'Prediction {sample_id} is missing its verdict field')
        result[sample_id] = row
    return result


def _matched(dataset_rows, prediction_rows):
    gold = _indexed(dataset_rows, 'gold')
    predictions = _indexed(prediction_rows, 'prediction')
    if set(gold) != set(predictions):
        missing = sorted(set(gold) - set(predictions))
        extra = sorted(set(predictions) - set(gold))
        raise ValueError(f'Prediction IDs differ from gold: missing={missing}, extra={extra}')
    return gold, predictions


def _label(row):
    label = row['verdict']
    return label if label in PREDICTION_LABELS[:3] else 'INVALID'


def _finite_score(value, *, entropy=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or (value < 0 if entropy else value > 0):
        return None
    return float(value)


def _score_summary(rows, field, *, entropy=False):
    values = [_finite_score(row.get(field), entropy=entropy) for row in rows]
    available = [value for value in values if value is not None]
    return {'samples_with_valid_score': len(available),
            'samples_without_valid_score': len(values) - len(available),
            'sample_mean_of_label_token_means': sum(available) / len(available) if available else None,
            'aggregation': 'Arithmetic mean of per-sample label-token means; not token-weighted.'}


def _selective(gold, predictions, threshold):
    selected = {sample_id: row for sample_id, row in predictions.items()
                if _label(row) in BINARY_LABELS and
                (score := _finite_score(row.get('mean_label_token_logprob'))) is not None and
                score >= threshold}
    correct = sum(_label(row) == gold[sample_id]['label'] for sample_id, row in selected.items())
    refuted = sum(row['label'] == 'REFUTED' for row in gold.values())
    supported = len(gold) - refuted
    detected = sum(gold[sample_id]['label'] == 'REFUTED' and _label(row) == 'REFUTED'
                   for sample_id, row in selected.items())
    false_negatives = sum(gold[sample_id]['label'] == 'REFUTED' and _label(row) == 'SUPPORTED'
                          for sample_id, row in selected.items())
    false_positives = sum(gold[sample_id]['label'] == 'SUPPORTED' and _label(row) == 'REFUTED'
                          for sample_id, row in selected.items())
    refuted_selected = detected + false_negatives
    return {'minimum_mean_label_token_logprob': threshold,
            'total_samples': len(gold), 'retained_binary_decisions': len(selected),
            'coverage': _ratio(len(selected), len(gold)),
            'correct_retained_decisions': correct,
            'accuracy_on_retained_decisions': _ratio(correct, len(selected)),
            'accuracy_on_all_samples': _ratio(correct, len(gold)),
            'gold_refuted': refuted, 'gold_supported': supported,
            'detected_errors': detected, 'false_negatives_supported': false_negatives,
            'abstained_or_invalid_errors': refuted - refuted_selected,
            'false_negative_rate': _ratio(false_negatives, refuted),
            'missed_errors_including_abstentions': refuted - detected,
            'missed_error_rate_including_abstentions': _ratio(refuted - detected, refuted),
            'false_positives': false_positives,
            'false_positive_rate': _ratio(false_positives, supported)}


def evaluate(dataset_rows, prediction_rows):
    """Evaluate one arm on exact ID-matched, whole-sample binary gold labels.

    Missing/extra/duplicate IDs fail instead of silently changing denominators.
    NOT_ENOUGH_INFO is an explicit abstention; any other verdict (including None)
    is INVALID. Neither is correct on binary gold. Undefined rates are None.
    FNR counts only erroneous SUPPORTED decisions among gold REFUTED samples;
    the separate missed-error rate also counts abstentions and invalid outputs.
    """
    gold, predictions = _matched(dataset_rows, prediction_rows)
    confusion = {label: {prediction: 0 for prediction in PREDICTION_LABELS}
                 for label in BINARY_LABELS}
    for sample_id, row in gold.items():
        confusion[row['label']][_label(predictions[sample_id])] += 1
    totals = {label: sum(row.values()) for label, row in confusion.items()}
    prediction_counts = Counter(_label(row) for row in predictions.values())
    total = len(gold)
    correct = sum(confusion[label][label] for label in BINARY_LABELS)
    binary = sum(prediction_counts[label] for label in BINARY_LABELS)
    detected = confusion['REFUTED']['REFUTED']
    false_negatives = confusion['REFUTED']['SUPPORTED']
    false_positives = confusion['SUPPORTED']['REFUTED']
    return {
        'evaluation_unit': 'whole_sample', 'positive_class': 'REFUTED',
        'total_samples': total, 'gold_counts': totals,
        'prediction_counts': {label: prediction_counts[label] for label in PREDICTION_LABELS},
        'confusion_matrix': confusion,
        'correct_samples': correct, 'accuracy': _ratio(correct, total),
        'binary_decisions': binary, 'binary_coverage': _ratio(binary, total),
        'accuracy_on_binary_decisions': _ratio(correct, binary),
        'detected_errors': detected,
        'error_recall': _ratio(detected, totals['REFUTED']),
        'false_negatives_supported': false_negatives,
        'false_negative_rate': _ratio(false_negatives, totals['REFUTED']),
        'abstained_errors': confusion['REFUTED']['NOT_ENOUGH_INFO'],
        'invalid_errors': confusion['REFUTED']['INVALID'],
        'missed_errors_including_abstentions': totals['REFUTED'] - detected,
        'missed_error_rate_including_abstentions': _ratio(totals['REFUTED'] - detected, totals['REFUTED']),
        'false_positives': false_positives,
        'false_positive_rate': _ratio(false_positives, totals['SUPPORTED']),
        'supported_recall': _ratio(confusion['SUPPORTED']['SUPPORTED'], totals['SUPPORTED']),
        'abstentions_by_gold': {label: confusion[label]['NOT_ENOUGH_INFO'] for label in BINARY_LABELS},
        'invalid_by_gold': {label: confusion[label]['INVALID'] for label in BINARY_LABELS},
        'score_summaries': {
            'mean_label_token_logprob': _score_summary(predictions.values(), 'mean_label_token_logprob'),
            'mean_label_token_full_vocab_entropy_nats': _score_summary(
                predictions.values(), 'mean_label_token_full_vocab_entropy_nats', entropy=True)},
        'selective_results': [_selective(gold, predictions, threshold) for threshold in SELECTIVE_THRESHOLDS],
        'definitions': {
            'accuracy': 'Correct whole-sample binary labels / all samples; abstentions and invalid outputs are not correct.',
            'false_negative_rate': 'Gold REFUTED predicted SUPPORTED / all gold REFUTED. Read with the separate abstention-inclusive missed-error rate.',
            'missed_error_rate_including_abstentions': 'Gold REFUTED not predicted REFUTED / all gold REFUTED, including NOT_ENOUGH_INFO and INVALID.',
            'selective_results': 'Fixed descriptive thresholds, applied equally to both arms. Rejected or unavailable scores abstain; no threshold is selected from test outcomes.',
            'scores': 'Actual generated-label-token model scores, not calibrated truth probabilities. Means of sample means are not token-weighted.'}}


def compare_paired(dataset_rows, baseline_rows, dual_rows):
    """Compare correctness on the same samples, counting abstentions as errors.

    The exact two-sided McNemar p-value is descriptive; no independence or
    general-population representativeness of benchmark samples is asserted.
    """
    # Materialize so caller-supplied generators can be reused by both matchers.
    dataset_rows = list(dataset_rows)
    gold, baseline = _matched(dataset_rows, baseline_rows)
    _, dual = _matched(dataset_rows, dual_rows)
    both_correct = both_wrong = baseline_only = dual_only = 0
    changed_verdicts = 0
    for sample_id, row in gold.items():
        b = _label(baseline[sample_id])
        d = _label(dual[sample_id])
        bc, dc = b == row['label'], d == row['label']
        changed_verdicts += b != d
        both_correct += bc and dc
        both_wrong += not bc and not dc
        baseline_only += bc and not dc
        dual_only += not bc and dc
    discordant = baseline_only + dual_only
    # Integer numerator/denominator avoid underflow from 0.5 ** discordant.
    pvalue = (min(1.0, 2 * sum(math.comb(discordant, k)
                              for k in range(min(baseline_only, dual_only) + 1)) / (1 << discordant))
              if discordant else 1.0)
    return {'total_samples': len(gold), 'both_correct': both_correct,
            'both_incorrect': both_wrong, 'baseline_only_correct': baseline_only,
            'dual_only_correct': dual_only, 'changed_verdicts': changed_verdicts,
            'accuracy_difference_dual_minus_baseline': _ratio(dual_only - baseline_only, len(gold)),
            'mcnemar_exact_two_sided_p': pvalue,
            'interpretation': 'Paired whole-sample benchmark-label agreement; abstentions/invalid outputs count as incorrect. No threshold selection or causal attribution to log-probability.'}
