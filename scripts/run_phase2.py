"""Run the bounded extractor/verifier loop with a pinned local Qwen snapshot.

The model receives only sample id-independent source text. Progress is written
after each sample; --resume accepts only the exact original protocol and prefix.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minifact


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def samples_from_file(path, kind):
    data = minifact.read_json(path)
    if kind == 'dev':
        data = data['samples']
        pairs = [{'id': row['id'], 'text': row['source']} for row in data]
    else:
        pairs = data
        if len(pairs) != 60 or any(set(row) != {'id', 'text'} for row in pairs):
            raise ValueError('Formal inputs must be the frozen 60-row label-free input')
        if digest(path) != '0de38bd1a9348ccdf75216e80f3eb679fddd94e6605f8ef9f5f8a753e6f5e434':
            raise ValueError('Frozen formal input hash mismatch')
    if not pairs or any(not isinstance(r.get('id'), str) or not isinstance(r.get('text'), str)
                        or not r['text'].strip() for r in pairs):
        raise ValueError('Every sample must have an id and non-empty text')
    if len({r['id'] for r in pairs}) != len(pairs):
        raise ValueError('Input IDs must be unique')
    return pairs


def make_contract(input_path, kind, max_rounds):
    files = ['minifact.py', 'config.json', 'phase2_backend.py', 'phase2_pipeline.py',
             'scripts/run_phase2.py']
    return {'protocol': 'phase2-closed-book-feedback-v1', 'input_kind': kind,
            'input_sha256': digest(input_path), 'max_rounds': max_rounds,
            'files_sha256': {p: digest(ROOT / p) for p in files},
            'model': minifact.read_json(ROOT / 'config.json'),
            'evidence': 'None. Source is context, not independent fact-checking evidence.',
            'roles': 'Two prompting roles of the same model, not independently trained models.',
            'verdict_scope': 'Each extracted claim; not inherited from a whole-sample label.'}


def prepare_output(path, contract, samples, resume):
    if path.exists():
        if not resume:
            raise ValueError('Output exists; use --resume only for the exact same protocol')
        artifact = minifact.read_json(path)
        if artifact.get('contract') != contract:
            raise ValueError('Cannot resume: protocol or input hash changed')
        done = artifact['results']
        if len(done) > len(samples) or any(row['id'] != case['id'] for row, case in zip(done, samples)):
            raise ValueError('Cannot resume: completed samples do not match the input prefix')
        if artifact.get('completed'):
            raise ValueError('Run is already complete; preserve it and choose another output')
        return artifact
    if resume:
        raise ValueError('Cannot resume: output file does not exist')
    return {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'contract': contract,
            'completed': False, 'results': []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-kind', choices=['dev', 'formal'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--max-rounds', type=int, choices=[1, 2], default=2)
    parser.add_argument('--gpu-memory', default='5GiB')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    source_file = ROOT / ('data/phase2_dev.json' if args.input_kind == 'dev' else 'data/phase1/inputs.json')
    samples = samples_from_file(source_file, args.input_kind)
    contract = make_contract(source_file, args.input_kind, args.max_rounds)
    artifact = prepare_output(args.output, contract, samples, args.resume)
    config = contract['model']
    snapshot = args.cache_dir / ('models--' + config['model_id'].replace('/', '--')) / 'snapshots' / config['model_revision']
    if not snapshot.is_dir():
        raise ValueError('Pinned local model snapshot is unavailable')

    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from phase2_backend import Phase2Backend
    from phase2_pipeline import run_sample
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
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
        raise ValueError('Loaded model precision or revision differs from the frozen config')
    runtime = {'runtime': minifact.runtime_info(torch), 'actual_dtype': str(model.dtype),
               'actual_revision': model.config._commit_hash,
               'attention_implementation': model.config._attn_implementation,
               'device_map': {str(k): str(v) for k, v in model.hf_device_map.items()},
               'model_generation_config': model.generation_config.to_dict()}
    if args.resume and artifact.get('execution') != runtime:
        raise ValueError('Cannot resume on a changed runtime/device map/generation configuration')
    artifact['execution'] = runtime
    backend = Phase2Backend(model, tokenizer, config)
    for case in samples[len(artifact['results']):]:
        row = run_sample(case['text'], backend, max_rounds=args.max_rounds)
        artifact['results'].append({'id': case['id'], 'source': case['text'], 'interaction': row})
        minifact.write_json(args.output, artifact)
        print(f"{case['id']}: stop={row.get('stop_reason')} rounds={len(row.get('rounds', []))}", flush=True)
    artifact['completed'] = True
    artifact['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    minifact.write_json(args.output, artifact)
    print(f'Completed {len(artifact["results"])} samples: {args.output}', flush=True)


if __name__ == '__main__':
    main()
