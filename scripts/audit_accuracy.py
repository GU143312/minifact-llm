"""Audit accuracy experiments without model weights; optional CPU token replay."""
import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minifact
from accuracy_backend import (BINARY_PROMPT, CLOSED_CANDIDATES, EVIDENCE_PROMPT, EVIDENCE_REVIEW_PROMPT,
                              LABEL_MAP, NEUTRAL_PROMPT, REVIEW_PROMPT)
from phase3_experiment import SAMPLE_PROMPT, analysis_from_interaction
from scripts.audit_phase2_run import audit as audit_phase2
from scripts.run_accuracy import contract, inputs_for, validate_inputs
from scripts.run_phase3 import sha, write

AUDIT_FILES = ('scripts/audit_accuracy.py', 'scripts/audit_phase2_run.py',
               'scripts/summarize_accuracy.py', 'scripts/evaluate_phase3.py',
               'scripts/run_accuracy.py', 'scripts/run_phase3.py', 'accuracy_backend.py',
               'accuracy_claims.py', 'phase2_backend.py', 'phase2_pipeline.py',
               'phase3_experiment.py', 'phase3_metrics.py', 'minifact.py', 'config.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_mean(observed, values, message):
    require(observed is None if not values else finite(observed) and math.isclose(
        observed, sum(values) / len(values), rel_tol=1e-8, abs_tol=1e-8), message)


def check_messages(g, system, payload):
    require(g.get('messages') == [{'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
            'Model prompt/payload differs from the permitted information')


def check_generation(g, config, eos, role, stats, labels=None, tokenizer=None):
    """Independently check caps, raw-label token alignment and score means."""
    require(isinstance(g, dict), 'Missing generation')
    ids, prompt = g.get('generated_token_ids'), g.get('prompt_input_ids')
    for value in (ids, prompt):
        require(isinstance(value, list) and value and all(type(i) is int and i >= 0 for i in value),
                'Invalid or empty token IDs')
    params = {k: config[k] for k in ('max_new_tokens', 'do_sample', 'num_beams',
               'repetition_penalty', 'temperature', 'top_p', 'top_k', 'use_cache')}
    params['max_new_tokens'] = min(params['max_new_tokens'], {'decision': 16, 'review': 64, 'extract': 256}[role])
    require(g.get('generation_params') == params, 'Generation parameters differ')
    require(len(ids) <= params['max_new_tokens'], 'Generation exceeds cap')
    require(type(g.get('ended_with_eos')) is bool and g['ended_with_eos'] == (ids[-1] in eos),
            'EOS state differs')
    require(not any(i in eos for i in ids[:-1]), 'EOS before last output token')
    require(isinstance(g.get('raw_response'), str) and isinstance(g.get('raw_response_with_special_tokens'), str),
            'Missing decoded output')
    stats['generation_calls'] += 1
    stats['generations_without_eos'] += not g['ended_with_eos']
    if tokenizer is not None:
        require(tokenizer.apply_chat_template(g['messages'], add_generation_prompt=True, tokenize=True) == prompt,
                'Tokenizer prompt IDs differ')
        require(tokenizer.decode(ids, skip_special_tokens=True) == g['raw_response'] and
                tokenizer.decode(ids, skip_special_tokens=False) == g['raw_response_with_special_tokens'],
                'Tokenizer output decoding differs')
        stats['tokenizer_replayed_calls'] += 1
    if labels is None:
        return
    trace = g.get('token_trace')
    require(isinstance(trace, list) and len(trace) == len(ids), 'Decision trace length differs')
    pieces = []
    for index, (token, row) in enumerate(zip(ids, trace)):
        require(row.get('step') == index and row.get('token_id') == token, 'Trace token ID/step differs')
        require(type(row.get('special')) is bool and isinstance(row.get('decoded_piece'), str), 'Invalid trace token metadata')
        require(finite(row.get('raw_logprob')) and row['raw_logprob'] <= 1e-7, 'Invalid raw logprob')
        require(finite(row.get('full_vocab_entropy_nats')) and row['full_vocab_entropy_nats'] >= -1e-7,
                'Invalid full-vocabulary entropy')
        if token in eos:
            require(row['special'] and row['decoded_piece'] == '', 'EOS is not a special empty piece')
        if tokenizer is not None:
            require(row['special'] == (token in tokenizer.all_special_ids) and
                    row['decoded_piece'] == tokenizer.decode([token], skip_special_tokens=True),
                    'Tokenizer trace metadata differs')
        pieces.append(row['decoded_piece'])
    match = re.fullmatch(r'\s*(' + '|'.join(labels) + r')\s*', g['raw_response'])
    aligned = bool(match and ''.join(pieces) == g['raw_response'])
    require(g.get('label_token_alignment') == ('decoded_piece_overlap' if aligned else 'unavailable'),
            'Label alignment state differs')
    cursor = 0
    for row, piece in zip(trace, pieces):
        expected = (not row['special'] and cursor < match.end(1) and
                    cursor + len(piece) > match.start(1)) if aligned else None
        require(row.get('label_token') is expected, 'Raw-label token overlap differs')
        cursor += len(piece)
    check_mean(g.get('mean_label_token_logprob'), [r['raw_logprob'] for r in trace if r['label_token'] is True],
               'Label logprob mean differs')
    check_mean(g.get('mean_label_token_full_vocab_entropy_nats'),
               [r['full_vocab_entropy_nats'] for r in trace if r['label_token'] is True], 'Label entropy mean differs')
    check_mean(g.get('mean_nonspecial_generated_logprob'), [r['raw_logprob'] for r in trace if not r['special']],
               'Non-special score mean differs')
    stats['decision_traces'] += 1


def check_decision(result, legacy):
    g = result['generation']
    mapping = {s: s for s in LABEL_MAP.values()} if legacy else LABEL_MAP
    match = re.fullmatch(r'\s*(' + '|'.join(mapping) + r')\s*', g['raw_response'])
    raw = match.group(1) if match else None
    expected = mapping[raw] if raw and g['ended_with_eos'] else None
    require(result.get('verdict') == expected, 'Canonical verdict mapping differs')
    error = 'Expected one allowed label followed by EOS.' if legacy else 'Expected one exact allowed label followed by EOS.'
    require(result.get('parse_error') == (None if expected else error), 'Decision parse-error state differs')
    if not legacy:
        require(result.get('raw_label') == raw and result.get('label_map') == mapping, 'Raw label/map differs')


def check_arm(sample, arm, result, config, eos, stats, tokenizer=None):
    legacy = arm in ('legacy_direct', 'legacy_dual')
    roles = {'legacy_direct': ['decision'], 'legacy_dual': ['decision'], 'neutral': ['decision'],
             'binary': ['decision'], 'reasoned': ['review', 'decision'],
             'structured': ['extract', 'decision'], 'evidence': ['decision']}[arm]
    calls = result.get('calls')
    require(isinstance(calls, list) and [c.get('role') for c in calls] == roles, 'Arm call order differs')
    require(result.get('condition') == ('supplied_benchmark_evidence' if arm == 'evidence' else 'closed_book'),
            'Evidence condition mislabeled')
    require(finite(result.get('elapsed_seconds')) and result['elapsed_seconds'] >= 0, 'Invalid arm elapsed time')
    require(calls[-1]['generation'] == result['generation'], 'Final decision trace differs from calls')
    check_decision(result, legacy)
    if legacy:
        analysis = None if arm == 'legacy_direct' else analysis_from_interaction(result['interaction'])
        check_messages(result['generation'], SAMPLE_PROMPT, {'source': sample['text'], 'analysis': analysis})
    else:
        payload = {'statement': sample['text']}
        if arm == 'structured':
            from accuracy_claims import EXTRACTION_PROMPT, parse_extraction
            extract = calls[0]['generation']
            check_messages(extract, EXTRACTION_PROMPT, {'source': sample['text']})
            parsed = parse_extraction(sample['text'], extract['raw_response'])
            require(result.get('extraction') == parsed, 'Structured extraction parse differs')
            payload['claims'] = parsed
        else:
            require(result.get('extraction') is None, 'Unexpected structured extraction')
        if arm == 'evidence':
            payload['context'] = sample['context']
            payload['evidence'] = [{'id': f'E{i+1}', 'question': q['question'], 'answer': q['answer']}
                                   for i, q in enumerate(sample['evidence'])]
        if arm == 'reasoned':
            review = calls[-2]['generation']
            check_messages(review, EVIDENCE_REVIEW_PROMPT if arm == 'evidence' else REVIEW_PROMPT, payload)
            payload['review'] = review['raw_response'] if review['ended_with_eos'] else None
        prompt = EVIDENCE_PROMPT if arm == 'evidence' else BINARY_PROMPT if arm in ('binary', 'structured') else NEUTRAL_PROMPT
        check_messages(result['generation'], prompt, payload)
    for call in calls:
        check_generation(call['generation'], config, eos, call['role'], stats,
                         labels=({s: s for s in LABEL_MAP.values()} if legacy else LABEL_MAP)
                         if call['role'] == 'decision' else None, tokenizer=tokenizer)


def check_selection(run, selection_path, frozen_at=None):
    from scripts.summarize_accuracy import summarize
    selected = minifact.read_json(selection_path)
    require(sha(selection_path) == run['contract']['selection_sha256'], 'Selection hash differs')
    require(selected['selected_closed_book'] in CLOSED_CANDIDATES, 'Selection is not a new closed-book arm')
    development = ROOT / selected['development_run']
    require(development.resolve().is_relative_to(ROOT.resolve()), 'Development path leaves repository')
    require(sha(development) == selected['development_run_sha256'], 'Development run hash differs')
    dev = minifact.read_json(development)
    require(dev.get('completed') is True and dev['contract']['split'] == 'dev', 'Selection needs a completed development run')
    for name, expected in dev['contract']['files_sha256'].items():
        require(sha(ROOT / name) == expected, 'Development frozen file hash differs')
    scored = summarize(dev, minifact.read_json(ROOT / 'data/improvement/dev_dataset.json'))
    available = [a for a in CLOSED_CANDIDATES if a in scored['arms']]
    def key(a):
        item = scored['arms'][a]
        return (-item['metrics']['accuracy'], item['metrics']['false_positive_rate'],
                item['cost']['elapsed_seconds']/len(dev['results']), a)
    ranked = sorted(available, key=key)
    require(ranked and selected['candidate_ranking'] == ranked and selected['selected_closed_book'] == ranked[0],
            'Selection does not follow development-only ranking')
    require(selected['rule'] == dev['contract']['selection_rule'], 'Selection rule differs')
    require(datetime.fromisoformat(dev['completed_at_utc']) <= datetime.fromisoformat(run['created_at_utc']),
            'Holdout started before development completed')
    if frozen_at is not None:
        require(datetime.fromisoformat(dev['completed_at_utc']) <= datetime.fromisoformat(frozen_at),
                'Holdout freeze predates development completion')
    require(run['contract']['arms'] == ['legacy_direct', 'legacy_dual', ranked[0], 'evidence'], 'Holdout arms differ from selection')


def audit(run, freeze, freeze_path, tokenizer=None, allow_partial=False, selection_path=None):
    frozen = run['contract']
    if frozen['split'] == 'holdout':
        selection_path = selection_path or (ROOT / frozen['selection_file'] if frozen.get('selection_file') else None)
        require(selection_path is not None, 'Holdout audit requires its frozen selection file')
    require(frozen == contract(frozen['split'], frozen['arms'], selection_path), 'Current protocol hashes differ')
    require(freeze.get('contract') == frozen and run.get('freeze_sha256') == sha(freeze_path), 'Freeze differs')
    audit_hashes = freeze.get('audit_files_sha256', {})
    require(isinstance(audit_hashes, dict), 'Invalid audit-file binding')
    for name, expected in audit_hashes.items():
        require(name in AUDIT_FILES and sha(ROOT / name) == expected, 'Frozen audit file hash differs')
    require(datetime.fromisoformat(freeze['frozen_at_utc']) <= datetime.fromisoformat(run['created_at_utc']),
            'Run predates freeze')
    inputs = minifact.read_json(inputs_for(frozen['split']))
    validate_inputs(inputs)
    rows = run.get('results')
    require(type(run.get('completed')) is bool and isinstance(rows, list) and len(rows) <= len(inputs), 'Invalid run/prefix')
    require(allow_partial or run['completed'], 'Run incomplete')
    require(not run['completed'] or len(rows) == len(inputs), 'Completed run has missing samples')
    if run['completed']:
        require(datetime.fromisoformat(run['completed_at_utc']) >= datetime.fromisoformat(run['created_at_utc']),
                'Completion timestamp precedes start')
    execution = run['execution']
    config = frozen['model']
    require(execution.get('actual_revision') == config['model_revision'] and
            str(execution.get('actual_dtype', '')).replace('torch.', '') == config['dtype'] and
            execution.get('attention_implementation') == config['attention_implementation'], 'Actual model configuration differs')
    eos = execution['model_generation_config']['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    require(eos and all(type(i) is int and i >= 0 for i in eos), 'Invalid recorded EOS IDs')
    stats = Counter()
    for index, (sample, row) in enumerate(zip(inputs, rows)):
        require(row.get('id') == sample['id'] and row.get('source') == sample['text'], 'Input ID/source differs')
        arms = frozen['arms']
        require(row.get('arm_order') == arms[index % len(arms):] + arms[:index % len(arms)] and
                set(row['arms']) == set(arms), 'Arm set/order differs')
        for arm in arms:
            check_arm(sample, arm, row['arms'][arm], config, eos, stats, tokenizer)
    legacy = None
    if 'legacy_dual' in frozen['arms']:
        old_files = ('minifact.py', 'config.json', 'phase2_backend.py', 'phase2_pipeline.py', 'scripts/run_phase2.py')
        wrapper = {'completed': run['completed'], 'execution': execution,
                   'contract': {'protocol': 'phase2-closed-book-feedback-v1', 'input_kind': 'formal',
                                'input_sha256': sha(inputs_for(frozen['split'])), 'max_rounds': 2,
                                'files_sha256': {p: sha(ROOT/p) for p in old_files}, 'model': config},
                   'results': [{'id': r['id'], 'source': r['source'], 'interaction': r['arms']['legacy_dual']['interaction']}
                               for r in rows]}
        legacy = audit_phase2(wrapper, inputs, inputs_for(frozen['split']))
        if tokenizer is not None:
            from scripts.evaluate_phase3 import generations
            for row in rows:
                for _, g in generations(row['arms']['legacy_dual']['interaction']):
                    require(tokenizer.apply_chat_template(g['messages'], add_generation_prompt=True, tokenize=True) == g['prompt_input_ids'],
                            'Legacy tokenizer prompt differs')
                    require(tokenizer.decode(g['generated_token_ids'], skip_special_tokens=True) == g['raw_response'] and
                            tokenizer.decode(g['generated_token_ids'], skip_special_tokens=False) == g['raw_response_with_special_tokens'],
                            'Legacy tokenizer output differs')
                    stats['tokenizer_replayed_calls'] += 1
    if frozen['split'] == 'holdout':
        check_selection(run, selection_path, freeze['frozen_at_utc'])
    return {'passed': True, 'completed': run['completed'], 'samples_checked': len(rows),
            'samples_expected': len(inputs), 'counts': dict(stats), 'legacy_pipeline_audit': legacy,
            'limitations': ['Raw logits are not independently recomputed; recorded numerical consistency and optional tokenizer replay only.',
                            'Literal matching and extraction parser replay do not prove semantic completeness or atomicity.',
                            'Supplied benchmark evidence is not independent web retrieval. Timestamps are self-reported.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selection', type=Path)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    run = minifact.read_json(args.run)
    tokenizer = None
    if args.cache_dir:
        from transformers import AutoTokenizer
        config = run['contract']['model']
        snapshot = args.cache_dir / ('models--' + config['model_id'].replace('/', '--')) / 'snapshots' / config['model_revision']
        require(snapshot.is_dir(), 'Pinned local tokenizer unavailable')
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=True, local_files_only=True)
    result = audit(run, minifact.read_json(args.freeze), args.freeze, tokenizer, args.allow_partial, args.selection)
    result.update({'run_sha256': sha(args.run), 'freeze_sha256': sha(args.freeze), 'audit_script_sha256': sha(Path(__file__))})
    result['audit_dependency_files_sha256'] = {name: sha(ROOT / name) for name in AUDIT_FILES}
    result['audit_code_bound_in_freeze'] = 'scripts/audit_accuracy.py' in minifact.read_json(args.freeze).get('audit_files_sha256', {})
    write(args.output, result)
    print('PASS accuracy audit:', args.output)


if __name__ == '__main__':
    main()
