"""Freeze first, label-free execution, per-arm checkpoints and offline audit."""
import math
import time
from collections import Counter
from pathlib import Path

from accuracy_calibration import predict
from next_eval.annotations import readiness, timestamp
from next_eval.backend import EvaluationBackend, ReplayBackend
from next_eval.common import (ARMS, ROOT, PROTOCOL, checkpoint, check_sources, now,
    object_sha, read, require, sha, source_contract, validate_inputs, write_new)
from next_eval.data import validate_bundle
from scripts.check_accuracy import load_backend
from scripts.check_calibrated_accuracy import validate_execution
from scripts.run_accuracy_calibration import check_calibration, code_contract

CALIBRATION = ROOT/'reports/accuracy_label_threshold_20261112.json'


def calibration():
    value = read(CALIBRATION)
    check_calibration(value, code_contract(), ROOT/'data/improvement/holdout_inputs.json')
    return value


def input_paths(stage, bundle):
    if stage == 'dev':
        return ROOT/'data/improvement/dev_evidence_inputs.json', ROOT/'data/improvement/dev_dataset.json'
    if stage == 'boundary':
        return bundle/'boundary_inputs.json', bundle/'boundary.json'
    require(stage == 'news', 'Unknown evaluation stage')
    return bundle/'inputs.json', bundle/'dataset.json'


def relative(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'Protocol artifacts must be inside the repository (large caches stay outside)')
    return path.relative_to(ROOT).as_posix()


def confidence(result, arm):
    if arm == 'A':
        return abs(result['candidate_scoring']['margin'] - result['threshold'])
    return result['generation'].get('mean_label_token_logprob')


def selective_policy(run):
    require(run.get('completed') is True and run['stage'] == 'dev' and len(run['results']) == 32, 'Need a complete 32-case development run')
    result = {}
    for arm in ARMS:
        values = sorted(c for r in run['results'] if r['arms'][arm]['verdict'] in ('SUPPORTED', 'REFUTED')
                        and (c := confidence(r['arms'][arm], arm)) is not None)
        result[arm] = {str(coverage): values[min(len(values)-1, math.floor((1-coverage)*len(values)))] if values else None
                       for coverage in (1.0, .9, .8)}
    return result


def freeze(stage, bundle, output, dev_run=None, limit=None):
    bundle = Path(bundle).resolve()
    require(not Path(output).exists(), 'Freeze exists; choose a new file')
    validate_bundle(bundle)
    reviews = readiness(bundle) if stage == 'news' else None
    inputs, gold = input_paths(stage, bundle)
    rows = read(inputs)
    validate_inputs(rows)
    require(limit is None or stage == 'dev' and type(limit) is int and 0 < limit <= 32, 'Only development smoke runs can be limited')
    if limit is not None: rows = rows[:limit]
    threshold = calibration()
    development, policies = None, None
    if stage == 'news':
        require(dev_run is not None, 'Freeze requires completed prospective development run')
        dev = read(dev_run)
        dev_freeze = ROOT/dev['freeze_file']
        audit(dev, read(dev_freeze), dev_freeze)
        policies = selective_policy(dev)
        development = {'file': relative(dev_run), 'sha256': sha(dev_run)}
    result = {'created_at_utc': now(), 'stage': stage, 'protocol': PROTOCOL,
              'files_sha256': source_contract(), 'bundle': relative(bundle),
              'manifest_sha256': sha(bundle/'manifest.json'), 'review': reviews,
              'inputs_file': relative(inputs), 'inputs_sha256': sha(inputs),
              'gold_file': relative(gold), 'gold_sha256': sha(gold),
              'ids': [r['id'] for r in rows], 'arms': list(('D', 'E', 'F') if stage == 'boundary' else ARMS),
              'model': read(ROOT/'config.json'), 'calibration_sha256': sha(CALIBRATION),
              'threshold': threshold['fit']['threshold'], 'development': development,
              'selective_thresholds': policies,
              'dev_limit': limit, 'formal_news_ready': stage == 'news'}
    if development:
        require(timestamp(read(dev_run)['completed_at_utc']) < timestamp(result['created_at_utc']), 'Development must complete before freeze')
    write_new(output, result)
    return result


def validate_freeze(frozen):
    check_sources(frozen['files_sha256'])
    require(frozen['protocol'] == PROTOCOL and frozen['model'] == read(ROOT/'config.json'), 'Protocol/model changed')
    bundle = ROOT/frozen['bundle']
    require(sha(bundle/'manifest.json') == frozen['manifest_sha256'], 'Bundle changed')
    # Execution checks gold/review bytes, never loads their contents. Deep data
    # and human-review validation was completed when constructing the freeze.
    manifest = read(bundle/'manifest.json')
    for name, digest in manifest['artifacts_sha256'].items():
        require(sha(bundle/name) == digest, f'Frozen data artifact changed: {name}')
    for name, digest in manifest['history_sha256'].items():
        require(sha(ROOT/name) == digest, 'Historical data bytes changed')
    inputs, gold = input_paths(frozen['stage'], bundle)
    require(relative(inputs) == frozen['inputs_file'] and relative(gold) == frozen['gold_file'], 'Unexpected input/gold path')
    require(sha(inputs) == frozen['inputs_sha256'] and sha(gold) == frozen['gold_sha256'], 'Input/gold bytes changed')
    rows = read(inputs)
    validate_inputs(rows)
    limit = frozen['dev_limit']
    require(limit is None or frozen['stage'] == 'dev' and type(limit) is int and 0 < limit <= 32, 'Invalid limit')
    if limit is not None: rows = rows[:limit]
    require(frozen['ids'] == [r['id'] for r in rows], 'Frozen ID/order differs')
    require(frozen['arms'] == list(('D','E','F') if frozen['stage'] == 'boundary' else ARMS), 'Frozen arms differ')
    require(sha(CALIBRATION) == frozen['calibration_sha256'] and frozen['threshold'] == calibration()['fit']['threshold'], 'Calibration changed')
    if frozen['stage'] == 'news':
        require(frozen['formal_news_ready'] is True and frozen['review'].get('ready') is True, 'Formal review missing')
        require(set(frozen['review']['review_files_sha256']) == {'event_review.json','reviewer_a.json','reviewer_b.json','adjudication.json'}, 'Missing review bindings')
        for name, digest in frozen['review']['review_files_sha256'].items():
            require(sha(bundle/name) == digest, 'Human review changed after freeze')
        require(timestamp(frozen['review']['completed_at_utc']) < timestamp(frozen['created_at_utc']), 'Review must precede freeze')
        dev = frozen['development']
        require(sha(ROOT/dev['file']) == dev['sha256'], 'Development bytes changed')
        devrun = read(ROOT/dev['file'])
        require(timestamp(devrun['completed_at_utc']) < timestamp(frozen['created_at_utc']), 'Development completed after freeze')
        require(frozen['selective_thresholds'] == selective_policy(devrun), 'Selective policy changed')
    return rows


def order(arms, index):
    start = index % len(arms)
    return arms[start:] + arms[:start]


def check_prefix(run, rows, arms, complete):
    results = run['results']
    require(len(results) <= len(rows), 'Too many checkpoint rows')
    if complete: require(len(results) == len(rows), 'Completed run missing samples')
    for i, r in enumerate(results):
        require(r['id'] == rows[i]['id'] and r['source'] == rows[i]['text'], 'Checkpoint is not the exact input prefix')
        expected = order(arms, i)
        keys = list(r['arms'])
        require(keys == expected[:len(keys)], 'Checkpoint arm order differs')
        if complete or i < len(results)-1:
            require(keys == expected, 'Missing arm in completed checkpoint row')


def execute(freeze_path, output, cache_dir, resume=False):
    frozen = read(freeze_path)
    rows = validate_freeze(frozen)  # before GPU load, no gold content used in prompts
    output = Path(output)
    if output.exists():
        require(resume, 'Run exists; use --resume with the exact freeze')
        run = read(output)
        require(run['freeze_sha256'] == sha(freeze_path) and not run['completed'], 'Cannot resume completed or changed protocol')
        check_prefix(run, rows, frozen['arms'], False)
        audit(run, frozen, freeze_path, allow_partial=True)
    else:
        require(not resume, 'No checkpoint to resume')
        run = {'stage': frozen['stage'], 'created_at_utc': now(), 'completed': False,
               'freeze_file': relative(freeze_path), 'freeze_sha256': sha(freeze_path),
               'results': [], 'interruptions': []}
        write_new(output, run)
    base, execution, torch = load_backend(frozen['model'], Path(cache_dir))
    cal = calibration()
    terminator = validate_execution(base, execution, cal)
    if 'execution' in run:
        require(run['execution'] == execution, 'Runtime differs on resume')
    run['execution'] = execution
    run['terminator_id'] = terminator
    model_eos = base.model.generation_config.eos_token_id
    run['eos_ids'] = model_eos if isinstance(model_eos, list) else [model_eos]
    checkpoint(output, run)
    backend = EvaluationBackend(base, cal, terminator)
    for i, row in enumerate(rows):
        if i == len(run['results']):
            run['results'].append({'id': row['id'], 'source': row['text'], 'arms': {}})
        target = run['results'][i]['arms']
        for arm in order(frozen['arms'], i):
            if arm in target: continue
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            try:
                result = backend.run(row, arm)
                torch.cuda.synchronize()
                result['elapsed_seconds'] = time.perf_counter()-start
                result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            except Exception as exc:
                run['interruptions'].append({'id': row['id'], 'arm': arm, 'at_utc': now(), 'exception_type': type(exc).__name__})
                checkpoint(output, run)
                raise  # preserve checkpoint; never retry silently or drop a difficult sample
            target[arm] = result
            checkpoint(output, run)
        print(f'{i+1}/{len(rows)} complete', flush=True)
    run['completed'], run['completed_at_utc'] = True, now()
    checkpoint(output, run)
    return run


def audit(run, frozen, freeze_path, tokenizer=None, allow_partial=False):
    from scripts.audit_accuracy import check_generation
    from scripts.audit_accuracy_calibration import check_scoring
    rows = validate_freeze(frozen)
    require(run['freeze_sha256'] == sha(freeze_path) and run['freeze_file'] == relative(freeze_path), 'Run/freeze binding differs')
    require(run['stage'] == frozen['stage'], 'Run stage differs')
    require(run.get('completed') is True or allow_partial, 'Run is incomplete')
    require(timestamp(run['created_at_utc']) >= timestamp(frozen['created_at_utc']), 'Run predates freeze')
    if run['completed']:
        require(timestamp(run['completed_at_utc']) >= timestamp(run['created_at_utc']), 'Run completion predates start')
    check_prefix(run, rows, frozen['arms'], run['completed'])
    stats = Counter()
    cal = calibration()
    execution = run.get('execution')
    if run['results']:
        require(execution is not None, 'Missing actual runtime')
        expected = {k: execution[k] for k in ('runtime','actual_dtype','actual_revision','attention_implementation','device_map')}
        expected['terminator_id'] = run['terminator_id']
        require(expected == cal['execution'], 'Actual runtime differs from fixed calibration')
        require(run['eos_ids'] == (execution['model_generation_config']['eos_token_id'] if isinstance(execution['model_generation_config']['eos_token_id'], list) else [execution['model_generation_config']['eos_token_id']]), 'EOS configuration differs')
    for i, row in enumerate(run['results']):
        for arm, result in row['arms'].items():
            require(type(result['elapsed_seconds']) in (int,float) and math.isfinite(result['elapsed_seconds']) and result['elapsed_seconds'] >= 0, 'Invalid elapsed time')
            require(type(result['peak_cuda_allocated_bytes']) is int and result['peak_cuda_allocated_bytes'] >= 0, 'Invalid peak memory')
            if arm == 'A':
                item = {**result, 'margin': result['candidate_scoring']['margin']}
                mapping = check_scoring(item, row['source'], run['terminator_id'], tokenizer)
                require(mapping == cal['candidate_token_ids'] and result['threshold'] == frozen['threshold'], 'Candidate/threshold differs')
                require(result['verdict'] == predict(item['margin'], frozen['threshold']), 'Threshold decision differs')
                stats['candidate_forward_calls'] += 2
            else:
                ReplayBackend(result).check(rows[i], arm)
                for call in result['calls']:
                    role = call['role']
                    check_generation(call['generation'], frozen['model'], run['eos_ids'],
                                     {'verify':'decision','feedback':'review'}.get(role, role), stats,
                                     labels=('TRUE','FALSE','UNKNOWN') if role in ('decision','verify') else None,
                                     tokenizer=tokenizer)
                for step in result['rounds']:
                    for claim in step['extraction']['accepted']:
                        require(row['source'][claim['start_idx']:claim['end_idx']] == claim['claim'], 'Accepted span differs')
                stats['feedback_transitions'] += sum(s['feedback_used'] for s in result['rounds'])
            stats['arm_results'] += 1
    return {'passed': True, 'completed': run['completed'], 'samples': len(run['results']),
            'counts': dict(stats), 'run_sha256': object_sha(run), 'freeze_sha256': sha(freeze_path),
            'tokenizer_replayed': tokenizer is not None,
            'limits': 'Audits stored values and tokenizer reconstruction, not a fresh model-logit forward pass. Human reviewer identities are attestations.'}
