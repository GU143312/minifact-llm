"""Run the frozen extractor on formal input; save literal and prompt-token alignment."""
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
from token_alignment import encode_prompt_offsets, locate_unique_source, map_char_span

EXPECTED_INPUT_SHA256 = '0de38bd1a9348ccdf75216e80f3eb679fddd94e6605f8ef9f5f8a753e6f5e434'


def align_to_prompt(prompt, source, encoded, claims):
    location = locate_unique_source(prompt, source)
    mapped = []
    for claim in claims:
        if location['status'] != 'ok':
            mapping = {'status': 'source_' + location['status']}
        else:
            start = location['start'] + claim['start_idx']
            end = location['start'] + claim['end_idx']
            mapping = map_char_span(prompt, start, end, encoded['offset_mapping'],
                                    encoded['special_tokens_mask'])
        mapped.append({'quote_index': claim['quote_index'], **mapping})
    return location, mapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--gpu-memory', default='5GiB')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; choose a new filename to preserve the previous run.')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config = minifact.read_json(ROOT / 'config.json')
    input_path = ROOT / 'data/phase1/inputs.json'
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != EXPECTED_INPUT_SHA256:
        raise ValueError('Formal input file differs from the frozen Days 1–2 release')
    inputs = minifact.read_json(input_path)
    if len(inputs) != 60 or any(set(row) != {'id', 'text'} for row in inputs):
        raise ValueError('Expected the frozen 60-row label-free formal inputs')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; this run uses the established FP16 GPU protocol')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    snapshot = (Path(args.cache_dir) / ('models--' + config['model_id'].replace('/', '--')) /
                'snapshots' / config['model_revision'])
    if not snapshot.is_dir():
        raise ValueError('Pinned local model snapshot is missing')
    common = dict(local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=True, **common)
    if not tokenizer.is_fast:
        raise ValueError('Fast tokenizer required for offset_mapping')
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), dtype=getattr(torch, config['dtype']), device_map='auto',
        attn_implementation=config['attention_implementation'],
        max_memory={0: args.gpu_memory, 'cpu': '4GiB'}, **common).eval()
    if (str(model.dtype).replace('torch.', '') != config['dtype'] or
            getattr(model.config, '_commit_hash', None) != config['model_revision']):
        raise ValueError('Loaded model dtype or revision differs from the frozen config')
    tracked = ['minifact.py', 'config.json', 'token_alignment.py', 'scripts/run_phase1_extraction.py',
               'data/phase1/inputs.json']
    artifact = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'formal-60 extraction baseline; not a fact-verification verdict',
        'file_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in tracked},
        'prompt': minifact.PROMPT, 'config': config, 'runtime': minifact.runtime_info(torch),
        'execution': {'actual_dtype': str(model.dtype), 'actual_revision': model.config._commit_hash,
                      'device_map': {str(k): str(v) for k, v in model.hf_device_map.items()},
                      'model_generation_config': model.generation_config.to_dict()},
        'score_definition': 'Mean natural-log probability of generated non-special JSON quote tokens; not truth or verifier confidence.',
        'completed': False, 'results': []}
    for case in inputs:
        source = case['text']
        messages = [{'role': 'system', 'content': minifact.PROMPT},
                    {'role': 'user', 'content': f'Input: {source}'}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        actual = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors='pt')
        expected_ids = actual['input_ids'][0].tolist()
        encoded = encode_prompt_offsets(tokenizer, prompt, expected_ids)
        row = minifact.generate_one(model, tokenizer, source, config)
        expected_hash = minifact.sha256_bytes(json.dumps(expected_ids).encode())
        if row['prompt_input_ids_sha256'] != expected_hash:
            raise ValueError('Generation used a different prompt token sequence')
        row['case_id'] = case['id']
        row['prompt_source_location'], row['token_alignment'] = align_to_prompt(
            prompt, source, encoded, row['aligned_claims'])
        artifact['results'].append(row)
        minifact.write_json(args.output, artifact)
        print(f"{case['id']}: {len(row['aligned_claims'])} accepted, "
              f"{len(row['alignment_rejections'])} rejected, "
              f"mean logp {row['mean_quote_json_logprob']:.4f}", flush=True)
    artifact['completed'] = True
    minifact.write_json(args.output, artifact)
    print(f'Completed {len(artifact["results"])} samples: {args.output}', flush=True)


if __name__ == '__main__':
    main()
