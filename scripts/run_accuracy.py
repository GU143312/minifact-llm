"""Run development candidates or a frozen, development-selected holdout plan."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minifact
from accuracy_backend import ARMS, CLOSED_CANDIDATES
from scripts.run_phase3 import sha, write

FILES = ('minifact.py', 'config.json', 'phase2_backend.py', 'phase2_pipeline.py',
         'phase3_experiment.py', 'phase3_metrics.py', 'accuracy_backend.py',
         'accuracy_claims.py', 'scripts/run_accuracy.py', 'scripts/summarize_accuracy.py')


def inputs_for(split):
    return ROOT / f'data/improvement/{split}_evidence_inputs.json'


def contract(split, arms, selection=None):
    files = [*FILES, f'data/improvement/{split}_dataset.json',
             f'data/improvement/{split}_inputs.json',
             f'data/improvement/{split}_evidence_inputs.json',
             'data/improvement/manifest.json']
    result = {'protocol': 'accuracy-candidates-v1', 'split': split, 'arms': arms,
              'files_sha256': {p: sha(ROOT / p) for p in files},
              'model': minifact.read_json(ROOT / 'config.json'),
              'input_sha256': sha(inputs_for(split)),
              'closed_book': 'Only source text is passed; no id, context, QA, labels or gold justification.',
              'evidence_condition': 'Supplied benchmark QA plus context; not automatic retrieval. URLs and gold labels/justifications excluded from model payload.',
              'role_caps': {'decision': 16, 'review': 64, 'extract': 256},
              'selection_rule': 'New closed-book candidate with highest development accuracy; ties lowest FPR, then mean elapsed seconds, then arm name. No holdout-dependent selection.',
              'order': 'Rotate arm order by sample index; one checkpoint per completed paired sample.'}
    if selection:
        result['selection_sha256'] = sha(selection)
        result['selection_file'] = selection.resolve().relative_to(ROOT).as_posix()
    return result


def validate_inputs(rows):
    if not isinstance(rows, list) or not rows or len({r.get('id') for r in rows}) != len(rows):
        raise ValueError('Invalid or duplicate inputs')
    for row in rows:
        if set(row) != {'id', 'text', 'context', 'evidence'}:
            raise ValueError('Input fields must be id/text/context/evidence only')
        if not isinstance(row['text'], str) or not row['text'].strip() or not isinstance(row['id'], str):
            raise ValueError('Missing input id/text')
        if set(row['context']) != {'claim_date', 'speaker', 'location_ISO_code'}:
            raise ValueError('Unexpected context fields')
        if not row['evidence'] or any(set(q) != {'question', 'answer', 'url'} for q in row['evidence']):
            raise ValueError('Unexpected evidence fields')


def check_holdout_selection(path, arms):
    selected = minifact.read_json(path)
    if selected['selected_closed_book'] not in CLOSED_CANDIDATES:
        raise ValueError('Invalid development-selected method')
    expected = ['legacy_direct', 'legacy_dual', selected['selected_closed_book'], 'evidence']
    if arms != expected:
        raise ValueError('Holdout arms must exactly follow development selection')
    dev = ROOT / selected['development_run']
    if sha(dev) != selected['development_run_sha256']:
        raise ValueError('Development run changed after selection')
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['dev', 'holdout'], required=True)
    p.add_argument('--arms', nargs='+', choices=ARMS, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--selection', type=Path)
    p.add_argument('--freeze-only', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if len(set(args.arms)) != len(args.arms):
        raise ValueError('Duplicate arms')
    if args.split == 'holdout':
        if args.selection is None:
            p.error('Holdout requires the immutable development selection')
        check_holdout_selection(args.selection, args.arms)
    current = contract(args.split, args.arms, args.selection)
    rows = minifact.read_json(inputs_for(args.split))
    validate_inputs(rows)
    if args.freeze_only:
        if args.freeze.exists():
            raise ValueError('Freeze exists; choose new version')
        write(args.freeze, {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'contract': current})
        return
    frozen = minifact.read_json(args.freeze)
    if frozen['contract'] != current:
        raise ValueError('Frozen code/data/config mismatch')
    if args.output is None or args.cache_dir is None:
        p.error('Execution needs --output and --cache-dir')
    if args.output.exists():
        if not args.resume:
            raise ValueError('Output exists; preserve historical run')
        run = minifact.read_json(args.output)
        if run['contract'] != current or run['freeze_sha256'] != sha(args.freeze) or run['completed']:
            raise ValueError('Cannot resume changed or completed run')
        if len(run['results']) > len(rows) or any(
                old['id'] != row['id'] or old['source'] != row['text']
                for old, row in zip(run['results'], rows)):
            raise ValueError('Resume prefix differs')
    else:
        if args.resume:
            raise ValueError('Resume output missing')
        run = {'contract': current, 'freeze_sha256': sha(args.freeze),
               'created_at_utc': datetime.now(timezone.utc).isoformat(), 'completed': False, 'results': []}
    config = current['model']
    snapshot = args.cache_dir / ('models--' + config['model_id'].replace('/', '--')) / 'snapshots' / config['model_revision']
    if not snapshot.is_dir():
        raise ValueError('Pinned local model snapshot missing')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accuracy_backend import AccuracyBackend
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), dtype=getattr(torch, config['dtype']), device_map='auto',
        attn_implementation=config['attention_implementation'],
        max_memory={0: '5GiB', 'cpu': '4GiB'}, local_files_only=True).eval()
    if (str(model.dtype).replace('torch.', '') != config['dtype'] or
            model.config._commit_hash != config['model_revision']):
        raise ValueError('Actual model dtype/revision differs')
    execution = {'runtime': minifact.runtime_info(torch), 'actual_dtype': str(model.dtype),
                 'actual_revision': model.config._commit_hash,
                 'attention_implementation': model.config._attn_implementation,
                 'device_map': {str(k): str(v) for k, v in model.hf_device_map.items()},
                 'model_generation_config': model.generation_config.to_dict()}
    if args.resume and run.get('execution') != execution:
        raise ValueError('Cannot resume changed runtime/device map')
    run['execution'] = execution
    backend = AccuracyBackend(model, tokenizer, config)
    for i in range(len(run['results']), len(rows)):
        sample = rows[i]
        order = args.arms[i % len(args.arms):] + args.arms[:i % len(args.arms)]
        record = {'id': sample['id'], 'source': sample['text'], 'arm_order': order, 'arms': {}}
        for arm in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = backend.run_arm(sample, arm)
            torch.cuda.synchronize()
            result['elapsed_seconds'] = time.perf_counter() - start
            record['arms'][arm] = result
        run['results'].append(record)
        write(args.output, run)
        print(f"{i+1}/{len(rows)} {sample['id']} " + ' '.join(f"{a}={record['arms'][a]['verdict']}" for a in args.arms), flush=True)
    run['completed'] = True
    run['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    write(args.output, run)


if __name__ == '__main__':
    main()
