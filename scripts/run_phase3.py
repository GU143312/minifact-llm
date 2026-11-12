"""Freeze then run the paired Week 6 closed-book benchmark comparison.

Only label-free id/text inputs are read during model execution. Gold labels are
read separately by the evaluator. Large model/cache files stay outside the repo.
"""
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
from scripts.run_phase2 import make_contract as phase2_contract

FILES = ('minifact.py', 'config.json', 'phase2_backend.py', 'phase2_pipeline.py',
         'scripts/run_phase2.py', 'phase3_experiment.py', 'phase3_metrics.py',
         'scripts/run_phase3.py', 'scripts/evaluate_phase3.py', 'scripts/audit_phase2_run.py',
         'data/phase3/dataset.json', 'data/phase3/curation.json')
THRESHOLDS = [-0.05, -0.1, -0.25, -0.5]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Atomic, explicitly LF serialization keeps frozen hashes cross-platform."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    temporary.replace(path)


def contract(inputs):
    return {'protocol': 'phase3-week6-paired-v1', 'input_sha256': sha(inputs),
            'files_sha256': {p: sha(ROOT / p) for p in FILES},
            'model': minifact.read_json(ROOT / 'config.json'), 'max_rounds': 2,
            'arms': ['direct', 'dual'], 'unit': 'whole_source_claim',
            'evidence': 'closed_book_no_retrieval_no_gold',
            'selective_logprob_thresholds': THRESHOLDS,
            'arm_order': 'Alternate direct-first and dual-first by input row index.',
            'decision': 'Identical sample judge prompt and generation parameters; direct analysis=null; dual analysis=all bounded Phase2 rounds with untrusted scores.',
            'metrics': 'REFUTED is error-positive. Accuracy counts abstention/invalid as not correct. FNR counts SUPPORTED among gold REFUTED; missed-error rate also includes abstention/invalid. Report all separately.',
            'phase2_contract': phase2_contract(inputs, 'formal', 2)}


def validate_inputs(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected nonempty label-free list')
    if any(not isinstance(r, dict) or set(r) != {'id', 'text'} or not isinstance(r['id'], str) or not r['id'].strip() or
           not isinstance(r['text'], str) or not r['text'].strip() for r in rows):
        raise ValueError('Only id/text fields are permitted')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate input ID')


def prepare_output(path, frozen, rows, resume=False):
    if path.exists():
        if not resume:
            raise ValueError('Output exists; preserve it or use exact-protocol --resume')
        run = minifact.read_json(path)
        if run.get('contract') != frozen['contract'] or run.get('freeze_sha256') != sha(frozen['_path']):
            raise ValueError('Resume protocol/freeze mismatch')
        done = run['results']
        if run.get('completed') or len(done) > len(rows):
            raise ValueError('Run complete or invalid prefix')
        for old, case in zip(done, rows):
            if old['id'] != case['id'] or old['source'] != case['text']:
                raise ValueError('Resume ID/source prefix mismatch')
        return run
    if resume:
        raise ValueError('Resume file missing')
    return {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'contract': frozen['contract'], 'freeze_sha256': sha(frozen['_path']),
            'completed': False, 'results': []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'data/phase3/inputs.json')
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--gpu-memory', default='5GiB')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    rows = minifact.read_json(args.inputs)
    validate_inputs(rows)
    current = contract(args.inputs)
    if args.freeze_only:
        if args.freeze.exists():
            raise ValueError('Freeze exists; never overwrite a preregistration')
        write(args.freeze, {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'contract': current})
        print('Frozen protocol:', args.freeze)
        return
    if args.output is None or args.cache_dir is None:
        parser.error('--output and --cache-dir are required for model execution')
    frozen = minifact.read_json(args.freeze)
    if frozen['contract'] != current:
        raise ValueError('Code/input/config differs from frozen protocol')
    frozen['_path'] = args.freeze
    run = prepare_output(args.output, frozen, rows, args.resume)
    config = current['model']
    snapshot = args.cache_dir / ('models--' + config['model_id'].replace('/', '--')) / 'snapshots' / config['model_revision']
    if not snapshot.is_dir():
        raise ValueError('Pinned local snapshot missing')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from phase3_experiment import Phase3Backend, analysis_from_interaction
    from phase2_pipeline import run_sample
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
        max_memory={0: args.gpu_memory, 'cpu': '4GiB'}, local_files_only=True).eval()
    if (str(model.dtype).replace('torch.', '') != config['dtype'] or
            getattr(model.config, '_commit_hash', None) != config['model_revision']):
        raise ValueError('Actual dtype/revision mismatch')
    execution = {'runtime': minifact.runtime_info(torch), 'actual_dtype': str(model.dtype),
                 'actual_revision': model.config._commit_hash,
                 'attention_implementation': model.config._attn_implementation,
                 'device_map': {str(k): str(v) for k, v in model.hf_device_map.items()},
                 'model_generation_config': model.generation_config.to_dict()}
    if args.resume and run.get('execution') != execution:
        raise ValueError('Resume runtime/device map differs')
    run['execution'] = execution
    backend = Phase3Backend(model, tokenizer, config)
    for index in range(len(run['results']), len(rows)):
        case = rows[index]
        row = {'id': case['id'], 'source': case['text'],
               'arm_order': ['direct', 'dual'] if index % 2 == 0 else ['dual', 'direct']}
        for arm in row['arm_order']:
            torch.cuda.synchronize()
            started = time.perf_counter()
            if arm == 'direct':
                row[arm] = backend.judge_sample(case['text'])
            else:
                row['interaction'] = run_sample(case['text'], backend, max_rounds=2)
                row[arm] = backend.judge_sample(case['text'], analysis_from_interaction(row['interaction']))
            torch.cuda.synchronize()
            row[arm + '_elapsed_seconds'] = time.perf_counter() - started
        run['results'].append(row)
        write(args.output, run)
        print(f"{index+1}/{len(rows)} {case['id']}: direct={row['direct']['verdict']} dual={row['dual']['verdict']}", flush=True)
    run['completed'] = True
    run['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    write(args.output, run)


if __name__ == '__main__':
    main()
