"""Replay every saved Phase 3 prompt and output with the pinned local tokenizer.

This CPU-only check loads no model weights and does not recompute logits. With
--freeze it also runs the structural experiment auditor before token replay.
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minifact
from scripts.evaluate_phase3 import audit, generations
from scripts.run_phase3 import contract, sha, validate_inputs, write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _ids(value, description):
    require(isinstance(value, list) and value and
            all(type(token) is int and token >= 0 for token in value),
            f'{description}: missing or invalid token IDs')
    return value


def replay_generation(generation, tokenizer, eos_ids, context):
    """Validate one chat-template encoding and both saved output decodings."""
    require(isinstance(generation, dict), f'{context}: missing generation')
    prompt_ids = _ids(generation.get('prompt_input_ids'), f'{context} prompt')
    generated_ids = _ids(generation.get('generated_token_ids'), f'{context} output')
    messages = generation.get('messages')
    require(isinstance(messages, list) and messages and all(
        isinstance(message, dict) and isinstance(message.get('role'), str) and
        isinstance(message.get('content'), str) for message in messages),
        f'{context}: missing or invalid chat messages')
    fresh_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    require(fresh_prompt == prompt_ids, f'{context}: chat-template prompt token IDs differ')
    require(tokenizer.decode(generated_ids, skip_special_tokens=True) == generation.get('raw_response'),
            f'{context}: generated IDs differ from saved decoded output')
    require(tokenizer.decode(generated_ids, skip_special_tokens=False) ==
            generation.get('raw_response_with_special_tokens'),
            f'{context}: generated IDs differ from saved special-token output')
    require(type(generation.get('ended_with_eos')) is bool and
            generation['ended_with_eos'] == (generated_ids[-1] in eos_ids),
            f'{context}: EOS flag differs from generated token ID')
    require(not any(token in eos_ids for token in generated_ids[:-1]),
            f'{context}: EOS token appeared before final generated token')
    trace = generation.get('token_trace')
    if trace is not None:
        require(isinstance(trace, list) and len(trace) == len(generated_ids),
                f'{context}: token trace length differs')
        specials = set(tokenizer.all_special_ids)
        for index, (token, row) in enumerate(zip(generated_ids, trace)):
            require(isinstance(row, dict) and row.get('step') == index and row.get('token_id') == token,
                    f'{context}: trace token ID/step differs')
            require(type(row.get('special')) is bool and row['special'] == (token in specials),
                    f'{context}: trace special-token flag differs from tokenizer')
            require(row.get('decoded_piece') == tokenizer.decode([token], skip_special_tokens=True),
                    f'{context}: trace decoded token piece differs from tokenizer')
    return {'prompt_tokens': len(prompt_ids), 'generated_tokens': len(generated_ids),
            'trace_tokens': len(trace) if trace is not None else 0,
            'ended_with_eos': generation['ended_with_eos']}


def replay_run(run, tokenizer):
    """Replay direct/dual judges and every stored internal generation call."""
    require(run.get('completed') is True, 'Run must be complete before token replay')
    rows = run.get('results')
    require(isinstance(rows, list) and rows, 'Completed run has no result rows')
    eos = run.get('execution', {}).get('model_generation_config', {}).get('eos_token_id')
    eos_ids = eos if isinstance(eos, list) else [eos]
    require(eos_ids and all(type(token) is int and token >= 0 for token in eos_ids),
            'Recorded model EOS IDs are missing or invalid')
    require(all(token in tokenizer.all_special_ids for token in eos_ids),
            'Recorded EOS IDs are not special tokens in the local tokenizer')
    counts, prompt_tokens, output_tokens, trace_tokens = Counter(), Counter(), Counter(), Counter()
    missing_eos = Counter()
    for row in rows:
        sample_id = row.get('id', '<missing-id>')
        for arm in ('direct', 'dual'):
            calls = [('sample_judge', (row.get(arm) or {}).get('generation'))]
            if arm == 'dual':
                require(isinstance(row.get('interaction'), dict), f'{sample_id}: missing interaction')
                calls.extend(generations(row['interaction']))
            for role, generation in calls:
                role_name = f'{arm}/{role}'
                checked = replay_generation(generation, tokenizer, eos_ids, f'{sample_id} {role_name}')
                counts[role_name] += 1
                prompt_tokens[role_name] += checked['prompt_tokens']
                output_tokens[role_name] += checked['generated_tokens']
                trace_tokens[role_name] += checked['trace_tokens']
                missing_eos[role_name] += not checked['ended_with_eos']
    return {'passed': True, 'samples': len(rows), 'generation_calls_checked': sum(counts.values()),
            'calls_by_role': dict(sorted(counts.items())),
            'prompt_tokens_checked': sum(prompt_tokens.values()),
            'generated_tokens_checked_including_eos': sum(output_tokens.values()),
            'trace_token_pieces_checked': sum(trace_tokens.values()),
            'by_role': {role: {'calls': counts[role], 'prompt_tokens': prompt_tokens[role],
                               'generated_tokens_including_eos': output_tokens[role],
                               'trace_tokens': trace_tokens[role],
                               'generations_without_eos': missing_eos[role]}
                        for role in sorted(counts)},
            'scope': 'All stored direct/dual sample judges and every stored extractor, claim-verdict, feedback and source-logic generation. Chat-template IDs, output decodings, EOS and traced special-token metadata are replayed; no logits or semantic judgments are recomputed.'}


def validate_identity(run, inputs, input_path):
    """Bind the result to current frozen code/config/data and every input row."""
    validate_inputs(inputs)
    require(run.get('contract') == contract(input_path), 'Current protocol/code/input hashes differ from run')
    require(run.get('completed') is True and isinstance(run.get('results'), list) and
            len(run['results']) == len(inputs), 'Run is incomplete or has a different input count')
    for case, row in zip(inputs, run['results']):
        require(row.get('id') == case['id'] and row.get('source') == case['text'],
                'Run ID/source rows differ from frozen inputs')
    config = run['contract']['model']
    execution = run.get('execution', {})
    require(execution.get('actual_revision') == config['model_revision'] and
            str(execution.get('actual_dtype', '')).replace('torch.', '') == config['dtype'] and
            execution.get('attention_implementation') == config['attention_implementation'],
            'Recorded runtime differs from frozen model configuration')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'data/phase3/inputs.json')
    parser.add_argument('--freeze', type=Path, help='Also perform the full structural protocol/freeze audit')
    args = parser.parse_args()
    require(not args.output.exists(), 'Output exists; choose a new file')
    raw = args.run.read_bytes()
    run = json.loads(raw)
    inputs = minifact.read_json(args.inputs)
    validate_identity(run, inputs, args.inputs)
    structural = None
    if args.freeze:
        structural = audit(run, inputs, args.inputs, minifact.read_json(args.freeze), args.freeze)
    config = run['contract']['model']
    snapshot = (args.cache_dir / ('models--' + config['model_id'].replace('/', '--')) /
                'snapshots' / config['model_revision'])
    require(snapshot.is_dir(), 'Pinned local tokenizer snapshot is missing')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=True, local_files_only=True)
    require(tokenizer.is_fast, 'Expected the pinned fast tokenizer')
    recorded_commit = tokenizer.init_kwargs.get('_commit_hash')
    require(recorded_commit is None or recorded_commit == config['model_revision'],
            'Tokenizer commit metadata differs from frozen revision')
    checked = replay_run(run, tokenizer)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'run_sha256': hashlib.sha256(raw).hexdigest(), 'inputs_sha256': sha(args.inputs),
              'protocol_files_sha256': run['contract']['files_sha256'],
              'replay_script_sha256': sha(Path(__file__)),
              'model_id': config['model_id'], 'snapshot_directory_revision': config['model_revision'],
              'tokenizer_commit_metadata': recorded_commit,
              'tokenizer_class': type(tokenizer).__name__,
              'tokenizer_runtime': {package: importlib.metadata.version(package)
                                    for package in ('transformers', 'tokenizers')},
              'local_tokenizer_files_sha256': {path.name: sha(path) for path in sorted(snapshot.iterdir())
                                               if path.is_file() and path.name in {
                                                   'tokenizer_config.json', 'tokenizer.json', 'vocab.json',
                                                   'merges.txt', 'special_tokens_map.json', 'added_tokens.json',
                                                   'chat_template.jinja'}},
              'structural_audit': structural, 'replay': checked,
              'limitations': [
                  'CPU tokenization/decoding only; no model weights are loaded and no saved log-probability or entropy is independently recomputed.',
                  'The original run did not hash local tokenizer files; current snapshot hashes document this replay, not historical cache identity.',
                  'All stored generation calls are checked. Failed calls without generation records cannot be replayed; structural audit reports those separately.',
                  'Record consistency does not establish factual accuracy or independent evidence.']}
    if args.freeze:
        report['freeze_sha256'] = sha(args.freeze)
    write(args.output, report)
    print(json.dumps({'passed': True, 'samples': checked['samples'],
                      'generation_calls_checked': checked['generation_calls_checked'],
                      'calls_by_role': checked['calls_by_role']}, indent=2))


if __name__ == '__main__':
    main()
