"""Combine the frozen holdout conditions; never select a method on holdout."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minifact
from accuracy_calibration import predict
from phase3_metrics import evaluate, compare_paired
from scripts.summarize_accuracy import predictions
from scripts.run_phase3 import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--calibration-run', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing combined summary')
    run, metrics = minifact.read_json(args.run), minifact.read_json(args.metrics)
    scored, calibration = minifact.read_json(args.calibration_run), minifact.read_json(args.calibration)
    if (not run['completed'] or not scored['completed'] or run['contract']['split'] != 'holdout'
            or scored['stage'] != 'holdout' or metrics['run_sha256'] != sha(args.run)
            or scored['calibration_sha256'] != sha(args.calibration)):
        raise ValueError('Incomplete or different holdout records')
    data_path = ROOT / 'data/improvement/holdout_dataset.json'
    if sha(data_path) != run['contract']['files_sha256']['data/improvement/holdout_dataset.json']:
        raise ValueError('Holdout gold changed')
    dataset = minifact.read_json(data_path)['samples']
    expected = [(r['id'], r['text']) for r in dataset]
    if expected != [(r['id'], r['source']) for r in run['results']] or expected != [
            (r['id'], r['source']) for r in scored['results']]:
        raise ValueError('Holdout source/ID mismatch')
    threshold = calibration['fit']['threshold']
    if scored['threshold'] != threshold or any(r['verdict'] != predict(r['margin'], threshold) for r in scored['results']):
        raise ValueError('Threshold/predictions differ from frozen calibration')
    gold = [{'id': r['id'], 'label': r['gold']['verdict']} for r in dataset]
    candidate = [{'id': r['id'], 'verdict': r['verdict']} for r in scored['results']]
    zero = [{'id': r['id'], 'verdict': predict(r['margin'], 0.)} for r in scored['results']]
    calls = [c for r in scored['results'] for c in r['candidate_scoring']['candidates']]
    metrics['arms']['calibrated'] = {
        'metrics': evaluate(gold, candidate),
        'cost': {'calls': len(calls), 'forward_calls': len(calls),
                 'elapsed_seconds': sum(r['elapsed_seconds'] for r in scored['results']),
                 'prompt_tokens': sum(len(r['candidate_scoring']['prompt_token_ids'])*2 for r in scored['results']),
                 'generated_tokens_including_eos': 0,
                 'forced_candidate_tokens_including_eos': sum(len(c['token_ids']) for c in calls),
                 'definition': 'Two teacher-forced forward calls per case, no text generation; timings exclude model load/checkpoints.'},
        'score_kind': 'Teacher-forced candidate sums, not actual greedy label-token means or calibrated truth probabilities.'}
    metrics['paired_vs_legacy_dual']['calibrated'] = compare_paired(gold, predictions(run, 'legacy_dual'), candidate)
    metrics['uncalibrated_candidate_reference'] = {'threshold': 0., 'metrics': evaluate(gold, zero),
        'definition': 'Same saved candidate scores and zero margin; declared before first holdout run, no additional inference.'}
    metrics['calibration'] = {'threshold': threshold, 'development_fit': calibration['fit'],
                               'development_run_sha256': calibration['development_run_sha256']}
    flags, reject_reasons = Counter(), Counter()
    accepted = rejected = complete = 0
    for sample in run['results']:
        step = sample['arms'].get('structured', {}).get('extraction') or {}
        for item in step.get('accepted', []):
            if sample['source'][item['start_idx']:item['end_idx']] != item['claim']:
                raise ValueError('Accepted extraction is not a literal source slice')
            accepted += 1
            flags.update(item['guard_flags'])
        rejected += len(step.get('rejected', []))
        for item in step.get('rejected', []):
            reject_reasons.update([item.get('reason', 'unspecified')])
    metrics['structured_extraction'] = {'accepted_quote_occurrences': accepted, 'rejected_records': rejected,
        'guard_flags': dict(flags), 'rejection_reasons': dict(reject_reasons),
        'semantic_boundary': 'All accepted spans are literal, but atomic coverage and qualifier retention lack independent gold. Flags are heuristics.'}
    metrics['artifact_sha256'] = {name: sha(path) for name, path in [
        ('run', args.run), ('main_metrics', args.metrics), ('calibration_run', args.calibration_run),
        ('calibration', args.calibration), ('holdout_gold', data_path), ('summary_script', Path(__file__))]}
    metrics['constant_reference_accuracy'] = .5
    metrics['interpretation'] = 'Same balanced 80 cases for all conditions. Supplied QA plus context is a privileged diagnostic input condition. All holdout conditions are reported, not selected by holdout accuracy.'
    write(args.output, metrics)
    print('PASS combined holdout summary:', args.output)


if __name__ == '__main__':
    main()
