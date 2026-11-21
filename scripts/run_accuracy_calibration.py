"""Score a development-fitted label-prior threshold; keep holdout gold unread."""
import argparse
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
from accuracy_backend import BINARY_PROMPT
from accuracy_calibration import encode_candidates, fit_threshold, predict, score_candidates, sha, validate_inputs, write

FILES = ('accuracy_calibration.py', 'scripts/run_accuracy_calibration.py',
         'accuracy_backend.py', 'config.json', 'minifact.py', 'scripts/check_accuracy.py')


def code_contract():
    return {'protocol': 'development-binary-label-threshold-v1',
            'files_sha256': {p: sha(ROOT/p) for p in FILES},
            'model': minifact.read_json(ROOT/'config.json'),
            'system_prompt': BINARY_PROMPT,
            'score': 'TRUE label+EOS sum loglikelihood minus FALSE label+EOS sum loglikelihood; no length normalization; no greedy generation.',
            'prediction': 'SUPPORTED iff margin >= frozen development threshold; otherwise REFUTED.',
            'information': 'Statement text only. No ID, context, evidence, gold labels or justification enters the model.'}


def check_calibration(calibration, current, input_path, root=ROOT):
    if calibration.get('contract') != current:
        raise ValueError('Calibration code/config/prompt changed')
    if calibration.get('holdout_input_sha256') != sha(input_path):
        raise ValueError('Holdout inputs changed after calibration')
    dev_path = root/calibration['development_run']
    if not dev_path.resolve().is_relative_to(root.resolve()) or sha(dev_path) != calibration['development_run_sha256']:
        raise ValueError('Development scoring run changed')
    dev = json.loads(dev_path.read_bytes())
    if (dev.get('completed') is not True or dev.get('stage') != 'dev' or
            dev.get('contract') != current or dev.get('development_fit') != calibration.get('fit')):
        raise ValueError('Calibration does not match the completed development fit')
    if sha(root/'data/improvement/dev_dataset.json') != calibration['development_gold_sha256']:
        raise ValueError('Development gold changed after fitting')
    # Validate the numeric threshold without reading any holdout label.
    predict(0., calibration['fit']['threshold'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['dev', 'holdout'], required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or (args.stage == 'dev' and args.calibration.exists()):
        raise ValueError('Output/calibration exists; preserve history and choose a new filename')
    input_path = ROOT/f'data/improvement/{args.stage}_inputs.json'
    rows = minifact.read_json(input_path)
    validate_inputs(rows)
    current = code_contract()
    calibration = None
    if args.stage == 'holdout':
        calibration = minifact.read_json(args.calibration)
        check_calibration(calibration, current, input_path)
    freeze_path = args.output.with_suffix('.freeze.json')
    if freeze_path.exists():
        raise ValueError('Run freeze exists; preserve it and choose a new output')
    frozen = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'contract': current,
              'stage': args.stage, 'inputs_sha256': sha(input_path),
              'calibration_sha256': sha(args.calibration) if calibration else None,
              'threshold': calibration['fit']['threshold'] if calibration else None}
    if args.stage == 'dev':
        frozen['development_gold_sha256'] = sha(ROOT/'data/improvement/dev_dataset.json')
        frozen['future_holdout_input_sha256'] = sha(ROOT/'data/improvement/holdout_inputs.json')
    write(freeze_path, frozen)

    config = current['model']
    snapshot = args.cache_dir/('models--'+config['model_id'].replace('/', '--'))/'snapshots'/config['model_revision']
    if not snapshot.is_dir():
        raise ValueError('Pinned local model snapshot missing')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), dtype=getattr(torch, config['dtype']),
        device_map='auto', attn_implementation=config['attention_implementation'],
        max_memory={0: '5GiB', 'cpu': '4GiB'}, local_files_only=True).eval()
    if (str(model.dtype).replace('torch.', '') != config['dtype'] or
            model.config._commit_hash != config['model_revision'] or
            model.config._attn_implementation != config['attention_implementation']):
        raise ValueError('Actual model precision/revision/attention differs')
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    terminator = tokenizer.eos_token_id if tokenizer.eos_token_id in eos else eos[0]
    run = {'contract': current, 'stage': args.stage, 'freeze_sha256': sha(freeze_path),
           'created_at_utc': datetime.now(timezone.utc).isoformat(), 'completed': False, 'results': [],
           'execution': {'runtime': minifact.runtime_info(torch), 'actual_dtype': str(model.dtype),
                         'actual_revision': model.config._commit_hash,
                         'attention_implementation': model.config._attn_implementation,
                         'device_map': {str(k): str(v) for k, v in model.hf_device_map.items()},
                         'terminator_id': terminator}}
    if calibration:
        if (calibration['terminator_id'] != terminator or
                calibration['execution'] != run['execution']):
            raise ValueError('Calibration tokenizer terminator/runtime/device map differs')
        run['calibration_sha256'] = sha(args.calibration)
        run['threshold'] = calibration['fit']['threshold']
    frozen_candidates = calibration['candidate_token_ids'] if calibration else None
    for index, row in enumerate(rows):
        messages, prompt, candidates = encode_candidates(tokenizer, BINARY_PROMPT, row['text'], terminator)
        if frozen_candidates is not None and candidates != frozen_candidates:
            raise ValueError('Candidate tokenization changed from frozen IDs')
        frozen_candidates = candidates
        torch.cuda.synchronize()
        started = time.perf_counter()
        scoring = score_candidates(model, prompt, candidates, terminator)
        torch.cuda.synchronize()
        scoring['messages'] = messages
        item = {'id': row['id'], 'source': row['text'], 'margin': scoring['margin'],
                'candidate_scoring': scoring, 'elapsed_seconds': time.perf_counter()-started}
        if calibration:
            item['verdict'] = predict(item['margin'], run['threshold'])
        run['results'].append(item)
        write(args.output, run)
        print(f"{index+1}/{len(rows)} {row['id']} margin={item['margin']:.6f}", flush=True)
    run['completed'] = True
    run['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    if args.stage == 'dev':
        # Gold is opened only here, after all development model forwards finish.
        gold_path = ROOT/'data/improvement/dev_dataset.json'
        if sha(gold_path) != frozen['development_gold_sha256']:
            raise ValueError('Development labels changed while scoring')
        dataset = minifact.read_json(gold_path)
        if [{'id': s['id'], 'text': s['text']} for s in dataset['samples']] != rows:
            raise ValueError('Development dataset/input projection differs')
        gold = {s['id']: s['gold']['verdict'] for s in dataset['samples']}
        fit = fit_threshold([{'id': r['id'], 'margin': r['margin'], 'gold': gold[r['id']]} for r in run['results']])
        run['development_fit'] = fit
        for item in run['results']:
            item['verdict'] = predict(item['margin'], fit['threshold'])
        write(args.output, run)
        relative = args.output.resolve().relative_to(ROOT).as_posix()
        write(args.calibration, {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'contract': current, 'fit': fit, 'development_run': relative,
              'development_run_sha256': sha(args.output), 'development_gold_sha256': sha(gold_path),
              'development_input_sha256': sha(input_path),
              'holdout_input_sha256': frozen['future_holdout_input_sha256'],
              'candidate_token_ids': frozen_candidates, 'terminator_id': terminator,
              'execution': run['execution'],
              'limitations': ['Development fit is in-sample and optimistic; evaluate once on frozen holdout.',
                              'Threshold fitting does not calibrate probabilities of truth.',
                              'Two complete labels may have different token lengths; sequence sums include EOS without normalization.']})
    else:
        write(args.output, run)
    print('Completed candidate scoring:', args.output)


if __name__ == '__main__':
    main()
