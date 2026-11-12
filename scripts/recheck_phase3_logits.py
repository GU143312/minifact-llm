"""Recompute the first sample's final direct and dual verdict logits offline.

Uses one pinned model load and two full-sequence teacher-forced forward passes,
without generating new text. These two traces do not validate semantic accuracy.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minifact
from phase3_experiment import SAMPLE_PROMPT, analysis_from_interaction
from scripts.recheck_phase2_logits import compare_observations, require
from scripts.run_phase3 import FILES, sha, write


def first_sample_traces(run):
    """Select both final traces from row zero, without looking at gold/correctness."""
    require(run.get('completed') is True, 'Run must be complete before rechecking')
    require(isinstance(run.get('results'), list) and run['results'], 'Completed run has no samples')
    sample = run['results'][0]
    require(isinstance(sample.get('id'), str) and isinstance(sample.get('source'), str),
            'First sample lacks its ID or source')
    selected = []
    for arm in ('direct', 'dual'):
        response = sample.get(arm)
        generation = response.get('generation') if isinstance(response, dict) else None
        require(isinstance(generation, dict) and generation.get('token_trace'),
                f'First sample final {arm} has no saved token trace; do not select another sample')
        analysis = None if arm == 'direct' else analysis_from_interaction(sample['interaction'])
        expected_messages = [
            {'role': 'system', 'content': SAMPLE_PROMPT},
            {'role': 'user', 'content': json.dumps(
                {'source': sample['source'], 'analysis': analysis}, ensure_ascii=False)}]
        require(generation.get('messages') == expected_messages,
                f'First sample final {arm} messages differ from expected source/analysis')
        selected.append(({'sample_id': sample['id'], 'sample_index': 0, 'arm': arm,
                          'saved_verdict': response.get('verdict')}, generation))
    return selected


def validate_tokenization(generation, tokenizer, eos_ids):
    """Check saved IDs, decoding, EOS and label flags using the pinned tokenizer."""
    prompt = generation.get('prompt_input_ids')
    ids = generation.get('generated_token_ids')
    trace = generation.get('token_trace')
    for values, name in ((prompt, 'prompt'), (ids, 'generated')):
        require(isinstance(values, list) and values and
                all(type(i) is int and i >= 0 for i in values), f'Invalid saved {name} token IDs')
    require(isinstance(trace, list) and len(trace) == len(ids), 'Saved trace length mismatch')
    actual_prompt = tokenizer.apply_chat_template(
        generation['messages'], add_generation_prompt=True, tokenize=True)
    if hasattr(actual_prompt, 'tolist'):
        actual_prompt = actual_prompt.tolist()
    if actual_prompt and isinstance(actual_prompt[0], list):
        actual_prompt = actual_prompt[0]
    require(actual_prompt == prompt, 'Saved prompt IDs differ from tokenizer chat template')
    text = tokenizer.decode(ids, skip_special_tokens=True)
    require(text == generation.get('raw_response'), 'Generated IDs do not decode to saved response')
    require(tokenizer.decode(ids, skip_special_tokens=False) ==
            generation.get('raw_response_with_special_tokens'), 'Special-token decoding mismatch')
    require(type(generation.get('ended_with_eos')) is bool and
            generation['ended_with_eos'] == (ids[-1] in eos_ids), 'Saved EOS status mismatch')
    pieces = [tokenizer.decode([token], skip_special_tokens=True) for token in ids]
    match = re.fullmatch(r'\s*(SUPPORTED|REFUTED|NOT_ENOUGH_INFO)\s*', text)
    span = match.span(1) if match and ''.join(pieces) == text else None
    require(generation.get('label_token_alignment') ==
            ('decoded_piece_overlap' if span else 'unavailable'), 'Label alignment status mismatch')
    specials = set(tokenizer.all_special_ids)
    cursor = 0
    for index, (token, piece, row) in enumerate(zip(ids, pieces, trace)):
        require(row.get('step') == index and row.get('token_id') == token,
                'Saved token index or ID mismatch')
        require(type(row.get('special')) is bool and row['special'] == (token in specials),
                'Saved special-token flag mismatch')
        require(row.get('decoded_piece') == piece, 'Saved decoded token piece mismatch')
        label_token = None if span is None else (
            token not in specials and cursor < span[1] and cursor + len(piece) > span[0])
        require(row.get('label_token') is label_token, 'Saved label-token flag mismatch')
        cursor += len(piece)
    return prompt, ids, trace


def recompute(model, torch, prompt, ids):
    """Recompute every generated token, including EOS if present, with no cache."""
    device = model.get_input_embeddings().weight.device
    require(device.type != 'meta', 'Model input embedding is not materialized')
    with torch.inference_mode():
        model_input = torch.tensor([prompt + ids[:-1]], dtype=torch.long, device=device)
        # Qwen2 in the pinned transformers version supports logits_to_keep.
        # All prompt/generated-prefix tokens pass through the transformer;
        # project only target positions to avoid a long-prompt vocabulary tensor.
        output = model(input_ids=model_input, attention_mask=torch.ones_like(model_input),
                       use_cache=False, logits_to_keep=len(ids))
        logits = output.logits[0].float()
        require(logits.shape[0] == len(ids) and max(ids) < logits.shape[-1],
                'Unexpected output logits shape')
        logdist = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor(ids, dtype=torch.long, device=logits.device).unsqueeze(-1)
        logprobs = logdist.gather(-1, targets).squeeze(-1).cpu().tolist()
        entropies = (-(logdist.exp() * logdist).sum(dim=-1)).cpu().tolist()
    return [{'token_id': token, 'raw_logprob': p, 'full_vocab_entropy_nats': h}
            for token, p, h in zip(ids, logprobs, entropies)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-memory', default='5GiB')
    args = parser.parse_args()
    require(not args.output.exists(), 'Output exists; preserve it and choose a new file')
    raw = args.run.read_bytes()
    run = json.loads(raw)
    selected = first_sample_traces(run)
    config, execution = run['contract']['model'], run['execution']
    require(config == minifact.read_json(ROOT / 'config.json'), 'Run/local model config mismatch')
    require(execution.get('actual_revision') == config['model_revision'] and
            str(execution.get('actual_dtype', '')).replace('torch.', '') == config['dtype'] and
            execution.get('attention_implementation') == config['attention_implementation'],
            'Recorded runtime differs from frozen model configuration')
    require(set(run['contract']['files_sha256']) == set(FILES), 'Unexpected protocol source-file set')
    for name, expected in run['contract']['files_sha256'].items():
        require(sha(ROOT / name) == expected, f'Protocol source differs from run: {name}')
    snapshot = (args.cache_dir / ('models--' + config['model_id'].replace('/', '--')) /
                'snapshots' / config['model_revision'])
    require(snapshot.is_dir(), 'Pinned local snapshot not found')

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
    eos = execution['model_generation_config']['eos_token_id']
    eos_ids = eos if isinstance(eos, list) else [eos]
    checked = [(identity, generation, validate_tokenization(generation, tokenizer, eos_ids))
               for identity, generation in selected]
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), dtype=getattr(torch, config['dtype']), device_map='auto',
        attn_implementation=config['attention_implementation'],
        max_memory={0: args.gpu_memory, 'cpu': '4GiB'}, local_files_only=True).eval()
    require(str(model.dtype).replace('torch.', '') == config['dtype'] and
            getattr(model.config, '_commit_hash', None) == config['model_revision'] and
            model.config._attn_implementation == config['attention_implementation'],
            'Loaded model precision, revision or attention implementation mismatch')
    results = []
    for identity, generation, (prompt, ids, recorded) in checked:
        fresh = recompute(model, torch, prompt, ids)
        comparison = compare_observations(ids, recorded, fresh)
        results.append({'selection': identity, 'prompt_token_count': len(prompt),
                        'generated_token_count': len(ids),
                        'ended_with_eos': generation['ended_with_eos'],
                        'tokenizer_checks': 'passed', 'comparison': comparison})
    inspected = (*FILES, 'scripts/recheck_phase3_logits.py', 'scripts/recheck_phase2_logits.py')
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'run_sha256': hashlib.sha256(raw).hexdigest(),
        'source_files_sha256': {name: sha(ROOT / name) for name in inspected},
        'selection_rule': 'Both final verdicts from first row of the completed run; no gold/correctness selection.',
        'scope': {'samples': 1, 'traces': 2, 'arms': ['direct', 'dual']},
        'method': 'Two full-sequence teacher-forced forward passes on saved chat prompt IDs and actual generated token IDs, including EOS when present. No text generation. Logits projection retains only the generated-token prediction positions.',
        'model_revision': config['model_revision'], 'dtype': config['dtype'],
        'attention_implementation': config['attention_implementation'],
        'device_map': {str(k): str(v) for k, v in model.hf_device_map.items()},
        'runtime': minifact.runtime_info(torch), 'results': results,
        'limitations': [
            'Two final verdict traces from one sample only; no claim about other traces or semantic accuracy.',
            'Recorded revision and requested local snapshot are checked; the original run does not prove identical physical cache or weight bytes.',
            'Full-sequence teacher forcing and incremental KV-cached generation may differ through floating-point rounding or device placement; observed numeric differences are reported without assuming equality.'],
    }
    write(args.output, report)
    print(json.dumps([{'selection': row['selection'],
                       'max_absolute_logprob_error': row['comparison']['max_absolute_logprob_error'],
                       'max_absolute_entropy_error_nats': row['comparison']['max_absolute_entropy_error_nats']}
                      for row in results], indent=2))


if __name__ == '__main__':
    main()
