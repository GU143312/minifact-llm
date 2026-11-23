"""Independently audit complete-candidate scoring and dev-only threshold fitting.

Optional tokenizer replay is CPU-only. Saved logits-derived values are checked
arithmetically; this script does not reload model weights or recompute logits.
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minifact
from accuracy_backend import BINARY_PROMPT
from accuracy_calibration import sha, write
from scripts.run_accuracy_calibration import FILES, code_contract

AUDIT_DEPENDENCIES = ('scripts/audit_accuracy_calibration.py', *FILES)
LABELS = {'TRUE': 'SUPPORTED', 'FALSE': 'REFUTED'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)


def verdict(margin, threshold):
    require(finite(margin) and finite(threshold), 'Non-finite decision parameter')
    return 'SUPPORTED' if margin >= threshold else 'REFUTED'


def threshold_counts(rows, threshold):
    """Own standard-library counters; no scorer/threshold helper is imported."""
    matrix = {g: {p: 0 for p in ('SUPPORTED', 'REFUTED')} for g in ('SUPPORTED', 'REFUTED')}
    for row in rows:
        require(row['gold'] in matrix and finite(row['margin']), 'Invalid development label/margin')
        matrix[row['gold']][verdict(row['margin'], threshold)] += 1
    support, refute = (sum(matrix[g].values()) for g in ('SUPPORTED', 'REFUTED'))
    require(support > 0 and refute > 0, 'Both development classes are required')
    correct = matrix['SUPPORTED']['SUPPORTED'] + matrix['REFUTED']['REFUTED']
    fp, fn = matrix['SUPPORTED']['REFUTED'], matrix['REFUTED']['SUPPORTED']
    return {'threshold': threshold, 'samples': support+refute, 'correct': correct,
            'accuracy': correct/(support+refute), 'false_positive_rate': fp/support,
            'false_negative_rate': fn/refute, 'balanced_error': (fp/support+fn/refute)/2,
            'confusion_matrix': matrix}


def independent_fit(rows):
    require(rows and len({r['id'] for r in rows}) == len(rows), 'Duplicate/empty development rows')
    values = {float(r['margin']) for r in rows if finite(r['margin'])}
    require(len(values) > 0 and all(finite(r['margin']) for r in rows), 'Non-finite development margin')
    thresholds = sorted({0., *values, *(math.nextafter(v, math.inf) for v in values)})
    scored = [threshold_counts(rows, t) for t in thresholds if math.isfinite(t)]
    best = min(scored, key=lambda r: (-r['correct'], r['false_positive_rate'], r['balanced_error'],
                                      abs(r['threshold']), r['threshold']))
    return {**best, 'candidate_thresholds_evaluated': len(scored)}


def compare_fit(saved, actual):
    require(isinstance(saved, dict), 'Missing development fit')
    for key in ('threshold', 'samples', 'correct', 'accuracy', 'false_positive_rate',
                'false_negative_rate', 'balanced_error', 'candidate_thresholds_evaluated'):
        require(close(saved.get(key), actual[key]), f'Development fitted {key} differs')


def check_scoring(item, source, terminator, tokenizer=None):
    g = item['candidate_scoring']
    require(g.get('method') == 'teacher_forced_complete_binary_candidates' and g.get('forward_calls') == 2,
            'Not the declared two-forward candidate method')
    expected_messages = [{'role': 'system', 'content': BINARY_PROMPT},
                         {'role': 'user', 'content': json.dumps({'statement': source}, ensure_ascii=False)}]
    require(g.get('messages') == expected_messages, 'Candidate prompt contains changed or extra information')
    prompt = g.get('prompt_token_ids')
    require(isinstance(prompt, list) and prompt and all(type(t) is int and t >= 0 for t in prompt), 'Invalid prompt IDs')
    require(g.get('terminator_id') == terminator, 'Candidate terminator differs from execution')
    candidates = g.get('candidates')
    require(isinstance(candidates, list) and [c.get('label') for c in candidates] == ['TRUE', 'FALSE'],
            'Expected exactly TRUE/FALSE complete candidates')
    mapping = {}
    for candidate in candidates:
        ids, logprobs, entropies = (candidate.get(k) for k in
                                   ('token_ids', 'token_logprobs', 'token_full_vocab_entropy_nats'))
        require(isinstance(ids, list) and len(ids) >= 2 and all(type(t) is int and t >= 0 for t in ids), 'Invalid complete candidate IDs')
        require(ids[-1] == terminator and terminator not in ids[:-1], 'Candidate EOS must occur exactly once at the end')
        require(isinstance(logprobs, list) and isinstance(entropies, list) and len(ids) == len(logprobs) == len(entropies),
                'Candidate ID/score lengths differ')
        require(all(finite(p) and p <= 1e-7 for p in logprobs), 'Invalid candidate token logprob')
        require(all(finite(h) and h >= -1e-7 for h in entropies), 'Invalid candidate token entropy')
        require(close(candidate.get('sequence_logprob'), sum(logprobs)), 'Complete sequence score must sum all label and EOS tokens')
        mapping[candidate['label']] = ids
        if tokenizer is not None:
            require(terminator in tokenizer.all_special_ids and not any(i in tokenizer.all_special_ids for i in ids[:-1]),
                    'Unexpected special token in label continuation')
            require(tokenizer.decode(ids[:-1], skip_special_tokens=False) == candidate['label'] and
                    tokenizer.decode(ids, skip_special_tokens=True) == candidate['label'], 'Candidate IDs decode to a different label')
    require(mapping['TRUE'] != mapping['FALSE'], 'Duplicate candidate token sequences')
    margin = candidates[0]['sequence_logprob'] - candidates[1]['sequence_logprob']
    require(close(g.get('margin'), margin) and close(item.get('margin'), margin), 'TRUE-minus-FALSE margin differs')
    require(finite(item.get('elapsed_seconds')) and item['elapsed_seconds'] >= 0, 'Invalid scoring elapsed time')
    if tokenizer is not None:
        template = tokenizer.apply_chat_template(expected_messages, tokenize=False, add_generation_prompt=True)
        require(tokenizer.apply_chat_template(expected_messages, tokenize=True, add_generation_prompt=True) == prompt and
                tokenizer.encode(template, add_special_tokens=False) == prompt, 'Chat prompt tokenization differs')
        for label, ids in mapping.items():
            require(tokenizer.encode(template+label, add_special_tokens=False) == prompt+ids[:-1],
                    'Candidate changes the chat-prefix tokenization boundary')
    return mapping


def development_rows(run, root=ROOT):
    path = root/'data/improvement/dev_dataset.json'
    dataset = minifact.read_json(path)
    require([(s['id'], s['text']) for s in dataset['samples']] ==
            [(r['id'], r['source']) for r in run['results']], 'Development gold/source projection differs')
    gold = {s['id']: s['gold']['verdict'] for s in dataset['samples']}
    return [{'id': r['id'], 'margin': r['margin'], 'gold': gold[r['id']]} for r in run['results']]


def audit(run, freeze, freeze_path, input_path, calibration=None, calibration_path=None,
          tokenizer=None, allow_partial=False, root=ROOT):
    require(run.get('contract') == code_contract() and freeze.get('contract') == run['contract'], 'Frozen calibration code/config differs')
    require(run.get('freeze_sha256') == sha(freeze_path), 'Run freeze bytes differ')
    stage = run.get('stage')
    require(freeze.get('stage') == stage and stage in ('dev', 'holdout', 'counter', 'old110'), 'Unknown/mismatched scoring stage')
    require(freeze.get('inputs_sha256') == sha(input_path), 'Frozen input bytes differ')
    require(datetime.fromisoformat(freeze['frozen_at_utc']) <= datetime.fromisoformat(run['created_at_utc']), 'Run predates freeze')
    inputs = minifact.read_json(input_path)
    require(isinstance(inputs, list) and inputs and len({s['id'] for s in inputs}) == len(inputs) and
            all(set(s) == {'id', 'text'} for s in inputs), 'Inputs must be unique id/text only')
    results = run.get('results')
    require(type(run.get('completed')) is bool and isinstance(results, list) and len(results) <= len(inputs), 'Invalid scoring prefix')
    require(allow_partial or run['completed'], 'Scoring run incomplete')
    require(not run['completed'] or len(results) == len(inputs), 'Completed scoring run misses inputs')
    if run['completed']:
        require(datetime.fromisoformat(run['created_at_utc']) <= datetime.fromisoformat(run['completed_at_utc']), 'Completion predates start')
    config, execution = run['contract']['model'], run['execution']
    require(execution.get('actual_revision') == config['model_revision'] and
            str(execution.get('actual_dtype', '')).replace('torch.', '') == config['dtype'] and
            execution.get('attention_implementation') == config['attention_implementation'], 'Recorded model configuration differs')
    terminator = execution.get('terminator_id')
    require(type(terminator) is int and terminator >= 0, 'Invalid execution terminator')
    expected_candidates = None
    diagnostics = None
    if stage == 'dev':
        require(freeze.get('calibration_sha256') is None and freeze.get('threshold') is None, 'Development run used a preselected threshold')
        require(freeze.get('development_gold_sha256') == sha(root/'data/improvement/dev_dataset.json'), 'Development gold changed')
        require(freeze.get('future_holdout_input_sha256') == sha(root/'data/improvement/holdout_inputs.json'), 'Future holdout inputs changed')
    else:
        require(calibration is not None and calibration_path is not None, 'Evaluation scoring needs its frozen development calibration')
        require(calibration.get('contract') == run['contract'], 'Development calibration protocol differs')
        require(run.get('calibration_sha256') == freeze.get('calibration_sha256') == sha(calibration_path), 'Calibration bytes differ')
        require(run.get('threshold') == freeze.get('threshold') == calibration['fit']['threshold'], 'Frozen evaluation threshold differs')
        require(calibration['terminator_id'] == terminator and calibration['execution'] == execution, 'Development/evaluation runtime or terminator differs')
        require(datetime.fromisoformat(calibration['created_at_utc']) <= datetime.fromisoformat(freeze['frozen_at_utc']),
                'Threshold calibration postdates evaluation freeze')
        if stage == 'holdout':
            require(calibration['holdout_input_sha256'] == sha(input_path), 'Holdout input differs from development declaration')
        dev_path = root/calibration['development_run']
        require(dev_path.resolve().is_relative_to(root.resolve()) and sha(dev_path) == calibration['development_run_sha256'], 'Development scoring history differs')
        dev = minifact.read_json(dev_path)
        require(dev.get('completed') is True and dev.get('stage') == 'dev' and dev.get('contract') == run['contract'],
                'Calibration history is not the completed development stage')
        require(sha(root/'data/improvement/dev_dataset.json') == calibration['development_gold_sha256'], 'Calibration development gold differs')
        recomputed = independent_fit(development_rows(dev, root))
        compare_fit(calibration['fit'], recomputed)
        compare_fit(dev['development_fit'], recomputed)
        require(datetime.fromisoformat(dev['completed_at_utc']) <= datetime.fromisoformat(calibration['created_at_utc']),
                'Threshold predates development completion')
        expected_candidates = calibration['candidate_token_ids']
        require('development_fit' not in run, 'Evaluation stage must not refit the threshold')
    for sample, item in zip(inputs, results):
        require(item.get('id') == sample['id'] and item.get('source') == sample['text'], 'Scoring source/ID prefix differs')
        actual_candidates = check_scoring(item, sample['text'], terminator, tokenizer)
        if expected_candidates is not None:
            require(actual_candidates == expected_candidates, 'Candidate IDs differ across statements/stages')
        expected_candidates = actual_candidates
        if stage != 'dev':
            require(item.get('verdict') == verdict(item['margin'], run['threshold']), 'Prediction differs from frozen threshold')
    if stage == 'dev' and run['completed']:
        rows = development_rows(run, root)
        fitted = independent_fit(rows)
        compare_fit(run.get('development_fit'), fitted)
        for item in results:
            require(item.get('verdict') == verdict(item['margin'], fitted['threshold']), 'Development prediction differs from fitted threshold')
        diagnostics = {'zero_threshold': threshold_counts(rows, 0.), 'fitted_threshold': threshold_counts(rows, fitted['threshold']),
                       'independent_fit': fitted, 'interpretation': 'Both counts use the same development scores. Fitted performance is in-sample and optimistic, not holdout evidence.'}
        if calibration is not None:
            require(calibration_path is not None and calibration['contract'] == run['contract'] and
                    calibration['development_run_sha256'] == sha(root/calibration['development_run']), 'Calibration development binding differs')
            compare_fit(calibration['fit'], fitted)
            require(calibration['candidate_token_ids'] == expected_candidates and calibration['terminator_id'] == terminator,
                    'Calibration frozen candidate IDs differ')
            require(calibration['development_input_sha256'] == sha(input_path) and
                    calibration['development_gold_sha256'] == sha(root/'data/improvement/dev_dataset.json'), 'Calibration development data differs')
            require(datetime.fromisoformat(run['completed_at_utc']) <= datetime.fromisoformat(calibration['created_at_utc']),
                    'Calibration predates completed development scores')
    return {'passed': True, 'stage': stage, 'completed': run['completed'], 'samples_checked': len(results),
            'samples_expected': len(inputs), 'candidate_forward_calls_checked': 2*len(results),
            'tokenizer_replayed_samples': len(results) if tokenizer is not None else 0,
            'development_diagnostics': diagnostics,
            'limitations': ['Checks saved logits-derived scores arithmetically; does not independently recompute model logits.',
                            'Teacher-forced complete-candidate scores include EOS and are not actual generated-label confidence or truth probabilities.',
                            'Development fitting is in-sample; counter/old110 stages are retrospective, not unseen tests.',
                            'This independent audit was added after scoring inference was frozen; current audit code hashes are recorded separately.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--freeze', type=Path)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    run = minifact.read_json(args.run)
    freeze_path = args.freeze or args.run.with_suffix('.freeze.json')
    input_path = args.inputs or ROOT/f"data/improvement/{run['stage']}_inputs.json"
    calibration = minifact.read_json(args.calibration) if args.calibration else None
    if run.get('stage') == 'dev' and calibration is not None:
        require(calibration['development_run_sha256'] == sha(args.run),
                'Calibration points to a different development scoring artifact')
    tokenizer = None
    if args.cache_dir:
        from transformers import AutoTokenizer
        config = run['contract']['model']
        snapshot = args.cache_dir/('models--'+config['model_id'].replace('/', '--'))/'snapshots'/config['model_revision']
        require(snapshot.is_dir(), 'Pinned tokenizer snapshot unavailable')
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=True, local_files_only=True)
    result = audit(run, minifact.read_json(freeze_path), freeze_path, input_path, calibration,
                   args.calibration, tokenizer, args.allow_partial)
    result.update({'run_sha256': sha(args.run), 'freeze_sha256': sha(freeze_path), 'inputs_sha256': sha(input_path),
                   'audit_script_sha256': sha(Path(__file__)),
                   'audit_dependencies_sha256': {name: sha(ROOT/name) for name in AUDIT_DEPENDENCIES}})
    if args.calibration:
        result['calibration_sha256'] = sha(args.calibration)
    write(args.output, result)
    print('PASS candidate calibration audit:', args.output)


if __name__ == '__main__':
    main()
