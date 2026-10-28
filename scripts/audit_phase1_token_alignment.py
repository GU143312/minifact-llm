"""Replay prompt/token alignment from a saved Phase 1 run, without a model or network."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from token_alignment import encode_prompt_offsets, locate_unique_source, map_char_span


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'data/phase1/inputs.json')
    parser.add_argument('--snapshot', type=Path, required=True,
                        help='Local pinned Qwen tokenizer snapshot directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; choose a new audit filename.')
    raw = args.run.read_bytes()
    run = json.loads(raw)
    inputs_bytes = args.inputs.read_bytes()
    inputs = json.loads(inputs_bytes)
    if not run.get('completed') or len(run['results']) != 60 or len(inputs) != 60:
        raise ValueError('Expected a completed 60-sample run and the 60 frozen inputs')
    revision = run['config']['model_revision']
    if args.snapshot.name != revision or not args.snapshot.is_dir():
        raise ValueError('Tokenizer snapshot path must name the recorded model revision')
    if sha256(inputs_bytes) != run['file_sha256']['data/phase1/inputs.json']:
        raise ValueError('Formal inputs do not match the hash recorded in the run')
    versions = {name: version(name) for name in ('transformers', 'tokenizers')}
    if any(versions[k] != run['runtime']['packages'][k] for k in versions):
        raise ValueError('Tokenizer library versions differ from the recorded run')
    file_checks = {name: sha256((ROOT / name).read_bytes()) == recorded
                   for name, recorded in run['file_sha256'].items()}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot), use_fast=True, local_files_only=True)
    rows = []
    statuses = Counter()
    for case, saved in zip(inputs, run['results']):
        problems = []
        if case['id'] != saved['case_id'] or case['text'] != saved['source']:
            problems.append('input_identity_mismatch')
        messages = [{'role': 'system', 'content': run['prompt']},
                    {'role': 'user', 'content': f"Input: {case['text']}"}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        encoded = encode_prompt_offsets(tokenizer, prompt, ids)
        token_hash = sha256(json.dumps(ids).encode())
        if token_hash != saved['prompt_input_ids_sha256']:
            problems.append('prompt_token_hash_mismatch')
        if len(ids) != saved['prompt_token_count']:
            problems.append('prompt_token_count_mismatch')
        loc = locate_unique_source(prompt, case['text'])
        if loc != saved['prompt_source_location']:
            problems.append('source_location_mismatch')
        mappings = []
        for claim in saved['aligned_claims']:
            if case['text'][claim['start_idx']:claim['end_idx']] != claim['claim']:
                problems.append('accepted_character_slice_mismatch')
            if loc['status'] != 'ok':
                mapped = {'status': 'source_' + loc['status']}
            else:
                start, end = loc['start'] + claim['start_idx'], loc['start'] + claim['end_idx']
                mapped = map_char_span(prompt, start, end, encoded['offset_mapping'],
                                       encoded['special_tokens_mask'])
            mappings.append({'quote_index': claim['quote_index'], **mapped})
            statuses[mapped['status']] += 1
        if mappings != saved['token_alignment']:
            problems.append('token_alignment_mismatch')
        rows.append({'id': case['id'], 'accepted_claim_count': len(saved['aligned_claims']),
                     'prompt_token_hash_match': 'prompt_token_hash_mismatch' not in problems,
                     'source_location_match': 'source_location_mismatch' not in problems,
                     'token_alignment_match': 'token_alignment_mismatch' not in problems,
                     'problems': problems})
    passed = all(file_checks.values()) and all(not row['problems'] for row in rows)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Replay of complete prompt-token and accepted character alignment; not semantic correctness or verification.',
              'method': 'Local fast tokenizer, recorded chat prompt and revision; same mapping implementation as extraction.',
              'source_run_sha256': sha256(raw), 'model_revision': revision,
              'tokenizer_packages': versions, 'recorded_file_hashes_match': file_checks,
              'passed': passed, 'sample_count': len(rows),
              'samples_with_problems': sum(bool(row['problems']) for row in rows),
              'accepted_claim_count': sum(row['accepted_claim_count'] for row in rows),
              'alignment_status_counts': dict(sorted(statuses.items())), 'samples': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"{'PASS' if passed else 'FAIL'}: {len(rows)} prompts; "
          f"{report['accepted_claim_count']} accepted claims; {dict(statuses)}")
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
