"""One extraction implementation for Windows and Colab; no private document input."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPT = (
    "Extract every independent atomic factual claim from the input. "
    "Return ONLY a JSON array of strings, each copied EXACTLY and contiguously "
    "from the input. Never rewrite, reorder words, change case, or add punctuation. "
    "Split different facts into separate quotes. Keep each fact's subject, "
    "numbers, units, dates and time expressions together. Do not omit temporal "
    "qualifiers. Retain pronouns as written; do not replace them with names. "
    "Exclude clause-joining conjunctions and optional terminal punctuation. "
    'Example input: In May, Cedar opened a pool, and the pool has six lanes. '
    'Example output: ["In May, Cedar opened a pool", "the pool has six lanes"]'
)
CONTROLLED_FILES = ('minifact.py', 'evaluate.py', 'config.json', 'data/cases.json')
RUNTIME_PACKAGES = ('torch', 'transformers', 'accelerate', 'tokenizers', 'safetensors')


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def occurrences(source, quote):
    result, cursor = [], 0
    while quote:
        start = source.find(quote, cursor)
        if start < 0:
            break
        result.append({'start_idx': start, 'end_idx': start + len(quote)})
        cursor = start + 1
    return result


def align_quotes(source, quotes):
    """Ambiguous, duplicated, rewritten and malformed quotes remain explicit rejections."""
    if not isinstance(quotes, list) or not quotes:
        return [], [{'index': None, 'reason': 'expected_nonempty_string_array'}]
    aligned, rejected, seen = [], [], set()
    for index, quote in enumerate(quotes):
        if not isinstance(quote, str) or not quote:
            rejected.append({'index': index, 'reason': 'invalid_quote_type_or_empty'})
            continue
        matches = occurrences(source, quote)
        reason = None
        if len(matches) != 1:
            reason = 'no_literal_match' if not matches else 'ambiguous_repeated_quote'
        elif quote in seen:
            reason = 'duplicate_prediction'
        if reason:
            rejected.append({'index': index, 'quote': quote, 'reason': reason, 'matches': matches})
        else:
            seen.add(quote)
            aligned.append(dict(claim=quote, quote_index=index, **matches[0]))
    return aligned, rejected


def exact_span(source, claim):
    start, end = claim.get('start_idx'), claim.get('end_idx')
    return (type(start) is int and type(end) is int and
            0 <= start < end <= len(source) and isinstance(claim.get('claim'), str) and
            source[start:end] == claim['claim'])


def contract(config, cases):
    file_hashes = {name: sha256_bytes((ROOT/name).read_bytes()) for name in CONTROLLED_FILES}
    value = {'schema_version': 1, 'files_sha256': file_hashes,
             'prompt': PROMPT, 'user_template': 'Input: {source}',
             'config': config, 'dataset_sha256': sha256_bytes(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()),
             'score_definition': 'Mean natural-log probability over generated non-special quote-array JSON tokens. Derived indices are not scored.'}
    value['contract_sha256'] = sha256_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())
    return value


def runtime_info(torch):
    return {'python': platform.python_version(), 'os': platform.system(),
            'packages': {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES},
            'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def generate_one(model, tokenizer, source, config):
    import torch
    messages = [{'role': 'system', 'content': PROMPT},
                {'role': 'user', 'content': f'Input: {source}'}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                         return_dict=True, return_tensors='pt').to(model.device)
    length = inputs['input_ids'].shape[-1]
    params = {key: config[key] for key in ('max_new_tokens', 'do_sample', 'num_beams',
              'repetition_penalty', 'temperature', 'top_p', 'top_k', 'use_cache')}
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **params, return_dict_in_generate=True,
                                output_scores=True, pad_token_id=tokenizer.eos_token_id)
        scores = model.compute_transition_scores(output.sequences, output.scores,
                                                normalize_logits=True)[0]
    ids = output.sequences[0, length:].tolist()
    values = scores.tolist()
    assert len(ids) == len(values)
    special = set(tokenizer.all_special_ids)
    retained = [float(score) for token, score in zip(ids, values) if token not in special]
    assert retained and all(math.isfinite(s) for s in retained)
    raw = tokenizer.decode(ids, skip_special_tokens=True).strip()
    error = None
    try:
        quotes = json.loads(raw)
    except json.JSONDecodeError as exc:
        quotes, error = None, str(exc)
    aligned, rejected = align_quotes(source, quotes)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    return {'source': source, 'raw_response': raw, 'raw_quotes': quotes,
            'json_error': error, 'aligned_claims': aligned, 'alignment_rejections': rejected,
            'generated_token_ids': ids, 'token_logprobs': retained,
            'generated_token_count': len(retained), 'prompt_token_count': length,
            'prompt_input_ids_sha256': sha256_bytes(json.dumps(inputs['input_ids'][0].tolist()).encode()),
            'mean_quote_json_logprob': sum(retained)/len(retained),
            'ended_with_eos': bool(ids and ids[-1] in eos),
            'seconds': time.perf_counter() - started}


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config, cases = read_json(ROOT/'config.json'), read_json(ROOT/'data/cases.json')
    if not torch.cuda.is_available():
        raise RuntimeError('This GPU comparison requires a CUDA runtime on both sides.')
    # A dry manifest is available without model allocation; complete runs always cover all cases.
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    common = dict(revision=config['model_revision'], cache_dir=args.cache_dir,
                  local_files_only=not args.download)
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], **common)
    load = dict(dtype=getattr(torch, config['dtype']), device_map='auto',
                attn_implementation=config['attention_implementation'])
    if torch.cuda.is_available():
        load['max_memory'] = {0: args.gpu_memory, 'cpu': '4GiB'}
    model = AutoModelForCausalLM.from_pretrained(config['model_id'], **load, **common).eval()
    actual_dtype = str(model.dtype).replace('torch.', '')
    actual_revision = getattr(model.config, '_commit_hash', None)
    if actual_dtype != config['dtype'] or actual_revision != config['model_revision']:
        raise RuntimeError('Loaded model precision/revision differs from controlled configuration.')
    artifact = {'label': args.label, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'contract': contract(config, cases), 'runtime': runtime_info(torch),
                'execution': {'actual_dtype': actual_dtype, 'actual_revision': actual_revision,
                              'attention_implementation': model.config._attn_implementation,
                              'model_generation_config': model.generation_config.to_dict(),
                              'device_map': {str(k): str(v) for k,v in model.hf_device_map.items()}},
                'completed': False, 'results': []}
    for case in cases['cases']:
        row = generate_one(model, tokenizer, case['source'], config)
        row.update(case_id=case['id'], category=case['category'])
        artifact['results'].append(row)
        write_json(args.output, artifact)
        print(f"{case['id']}: {row['raw_response']} | rejected={len(row['alignment_rejections'])} | logp={row['mean_quote_json_logprob']:.6f}", flush=True)
    artifact['completed'] = True
    write_json(args.output, artifact)
    print(f'Completed {len(artifact["results"])} cases: {args.output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True, choices=['local', 'colab'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', default=os.environ.get('MINIFACT_MODEL_CACHE'))
    parser.add_argument('--gpu-memory', default='5GiB', help='Placement only; does not change precision or generation.')
    parser.add_argument('--download', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
