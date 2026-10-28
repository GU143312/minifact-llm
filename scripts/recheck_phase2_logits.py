"""Recompute one saved verdict trace with a pinned local model, offline.

Uses a single full-sequence teacher-forced forward pass. Differences from
incremental generation may arise from numeric precision or execution details.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minifact


def require(condition, message):
    if not condition:
        raise ValueError(message)


def first_trace(run):
    require(run.get('completed') is True, 'Run must be complete before rechecking')
    for sample in run.get('results', []):
        for step in sample.get('interaction', {}).get('rounds', []):
            for verification in step.get('verifications', []):
                response = verification.get('response')
                generation = response.get('verdict_generation') if isinstance(response, dict) else None
                if isinstance(generation, dict) and generation.get('token_trace'):
                    return {'sample_id': sample['id'], 'round': step['round'],
                            'quote_index': verification['quote_index'],
                            'claim': verification['claim']}, generation
    raise ValueError('No verdict token trace in completed run')


def compare_observations(ids, recorded, recomputed):
    require(len(ids) == len(recorded) == len(recomputed) and len(ids) > 0,
            'Token IDs and trace lengths differ or are empty')
    rows = []
    for i, (token, saved, fresh) in enumerate(zip(ids, recorded, recomputed)):
        require(type(token) is int and token >= 0 and saved.get('step') == i and
                saved.get('token_id') == token, 'Saved token index or ID mismatch')
        require(fresh.get('token_id') == token, 'Recomputed token ID mismatch')
        old_lp, new_lp = saved.get('raw_logprob'), fresh.get('raw_logprob')
        old_h, new_h = saved.get('full_vocab_entropy_nats'), fresh.get('full_vocab_entropy_nats')
        require(all(type(x) in (int, float) and math.isfinite(x) for x in (old_lp, new_lp, old_h, new_h)),
                'Non-finite token statistic')
        require(old_lp <= 1e-7 and new_lp <= 1e-7 and old_h >= -1e-7 and new_h >= -1e-7,
                'Invalid log probability or entropy')
        rows.append({'step': i, 'token_id': token, 'special': saved.get('special'),
                     'decoded_piece': saved.get('decoded_piece'),
                     'saved_raw_logprob': old_lp, 'recomputed_raw_logprob': new_lp,
                     'absolute_logprob_error': abs(old_lp - new_lp),
                     'saved_full_vocab_entropy_nats': old_h, 'recomputed_full_vocab_entropy_nats': new_h,
                     'absolute_entropy_error': abs(old_h - new_h)})
    return {'tokens': rows,
            'max_absolute_logprob_error': max(r['absolute_logprob_error'] for r in rows),
            'max_absolute_entropy_error_nats': max(r['absolute_entropy_error'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-memory', default='5GiB')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output exists; choose a new file')
    raw = args.run.read_bytes()
    run = json.loads(raw)
    identity, generation = first_trace(run)
    config = run['contract']['model']
    require(config == minifact.read_json(ROOT / 'config.json'), 'Run and local model config differ')
    execution = run['execution']
    require(execution.get('actual_revision') == config['model_revision'] and
            str(execution.get('actual_dtype', '')).replace('torch.', '') == config['dtype'] and
            execution.get('attention_implementation') == config['attention_implementation'],
            'Recorded runtime does not match frozen model configuration')
    tracked = {'minifact.py', 'config.json', 'phase2_backend.py', 'phase2_pipeline.py', 'scripts/run_phase2.py'}
    require(set(run['contract']['files_sha256']) == tracked, 'Protocol file set differs from expected')
    for name, expected in run['contract']['files_sha256'].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
                f'Protocol source differs from run: {name}')
    snapshot = (args.cache_dir / ('models--' + config['model_id'].replace('/', '--')) /
                'snapshots' / config['model_revision'])
    require(snapshot.is_dir(), 'Pinned snapshot not found in local cache')
    prompt_ids = generation.get('prompt_input_ids')
    token_ids = generation.get('generated_token_ids')
    trace = generation.get('token_trace')
    require(isinstance(prompt_ids, list) and prompt_ids and
            all(type(i) is int and i >= 0 for i in prompt_ids), 'Invalid saved prompt IDs')
    require(isinstance(token_ids, list) and token_ids and
            all(type(i) is int and i >= 0 for i in token_ids), 'Invalid saved generated IDs')
    require(isinstance(trace, list) and len(trace) == len(token_ids), 'Saved trace length mismatch')

    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    require(torch.cuda.is_available(), 'CUDA unavailable')
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=True, local_files_only=True)
    actual_prompt = tokenizer.apply_chat_template(generation['messages'], add_generation_prompt=True, tokenize=True)
    if hasattr(actual_prompt, 'tolist'):
        actual_prompt = actual_prompt.tolist()
    if actual_prompt and isinstance(actual_prompt[0], list):
        actual_prompt = actual_prompt[0]
    require(actual_prompt == prompt_ids, 'Saved prompt IDs differ from tokenizer chat template')
    require(tokenizer.decode(token_ids, skip_special_tokens=True) == generation['raw_response'],
            'Generated IDs do not decode to saved verdict text')
    eos = execution['model_generation_config']['eos_token_id']
    eos_ids = eos if isinstance(eos, list) else [eos]
    require(bool(token_ids[-1] in eos_ids) == generation['ended_with_eos'], 'Saved EOS status mismatch')
    require(generation['ended_with_eos'], 'First verdict did not generate EOS; no EOS token is available to recheck')
    require(all(row.get('special') == (token in tokenizer.all_special_ids)
                for row, token in zip(trace, token_ids)), 'Saved special-token flags differ from tokenizer')
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), dtype=getattr(torch, config['dtype']), device_map='auto',
        attn_implementation=config['attention_implementation'],
        max_memory={0: args.gpu_memory, 'cpu': '4GiB'}, local_files_only=True).eval()
    require(str(model.dtype).replace('torch.', '') == config['dtype'] and
            getattr(model.config, '_commit_hash', None) == config['model_revision'] and
            model.config._attn_implementation == config['attention_implementation'],
            'Loaded model precision, revision or attention implementation mismatch')
    device = model.get_input_embeddings().weight.device
    require(device.type != 'meta', 'Model input embedding is not materialized')
    joined = prompt_ids + token_ids
    with torch.inference_mode():
        model_input = torch.tensor([joined[:-1]], dtype=torch.long, device=device)
        output = model(input_ids=model_input, attention_mask=torch.ones_like(model_input), use_cache=False)
        logits = output.logits[0, len(prompt_ids)-1:len(prompt_ids)-1+len(token_ids)].float()
        require(logits.shape[0] == len(token_ids) and max(token_ids) < logits.shape[-1], 'Unexpected output logits shape')
        logdist = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor(token_ids, dtype=torch.long, device=logits.device).unsqueeze(-1)
        logprobs = logdist.gather(-1, targets).squeeze(-1).cpu().tolist()
        entropies = (-(logdist.exp() * logdist).sum(dim=-1)).cpu().tolist()
    fresh = [{'token_id': i, 'raw_logprob': p, 'full_vocab_entropy_nats': h}
             for i, p, h in zip(token_ids, logprobs, entropies)]
    comparison = compare_observations(token_ids, trace, fresh)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'run_sha256': hashlib.sha256(raw).hexdigest(), 'selection': identity,
              'method': 'Full-sequence teacher-forced forward on saved chat prompt IDs and actual generated token IDs, including EOS.',
              'model_revision': config['model_revision'], 'dtype': config['dtype'],
              'attention_implementation': config['attention_implementation'],
              'device_map': {str(k): str(v) for k, v in model.hf_device_map.items()},
              'runtime': minifact.runtime_info(torch), 'prompt_token_count': len(prompt_ids),
              'generated_token_count': len(token_ids), 'ended_with_eos': generation['ended_with_eos'],
              'comparison': comparison,
              'limitations': ['One selected trace only; no inference about other claims or semantic accuracy.',
                              'The original run did not record its cache path; identical physical cache or weight bytes cannot be independently established from this record. The recheck uses the requested local snapshot and recorded revision.',
                              'Full-sequence teacher forcing and incremental KV-cached generation may differ through floating-point rounding or device placement; actual errors are reported without assuming equality.']}
    minifact.write_json(args.output, report)
    print(json.dumps({'selection': identity, 'max_absolute_logprob_error': comparison['max_absolute_logprob_error'],
                      'max_absolute_entropy_error_nats': comparison['max_absolute_entropy_error_nats']}, indent=2))


if __name__ == '__main__':
    main()
