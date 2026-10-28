"""Check the frozen Phase 2 protocol against dev and a formal run prefix.

This is a protocol consistency audit, not a truth or semantic evaluation.
Recorded timestamps are self-reported metadata, not trusted timestamps.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2_backend import FEEDBACK_MAX_NEW_TOKENS, VERDICT_MAX_NEW_TOKENS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp(value, label):
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{label}: invalid ISO timestamp') from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f'{label}: timezone required')
    return parsed.astimezone(timezone.utc)


def audit(freeze, dev, formal, dev_path, dev_input_path, formal_input_path, root=ROOT):
    files = {'minifact.py', 'config.json', 'phase2_backend.py', 'phase2_pipeline.py', 'scripts/run_phase2.py'}
    require(set(freeze.get('files_sha256', {})) == files, 'freeze file-hash set mismatch')
    require(freeze.get('developer_run_sha256') == digest(dev_path), 'developer run hash mismatch')
    with Path(dev_path).open(encoding='utf-8') as f:
        require(json.load(f) == dev, 'developer run data differs from hashed file')
    require(freeze.get('developer_input_sha256') == digest(dev_input_path), 'developer input hash mismatch')
    require(freeze.get('formal_input_sha256') == digest(formal_input_path), 'formal input hash mismatch')
    for name in sorted(files):
        require(freeze['files_sha256'][name] == digest(root / name), f'current file hash mismatch: {name}')
    with (root / 'config.json').open(encoding='utf-8') as f:
        require(json.load(f) == freeze['model_config'], 'frozen model configuration mismatch')
    config = freeze['model_config']
    limits = {'extraction': config['max_new_tokens'],
              'verdict': min(config['max_new_tokens'], VERDICT_MAX_NEW_TOKENS),
              'feedback': min(config['max_new_tokens'], FEEDBACK_MAX_NEW_TOKENS),
              'source_logic': min(config['max_new_tokens'], FEEDBACK_MAX_NEW_TOKENS)}
    require(freeze.get('generation_limits') == limits, 'frozen generation limits mismatch')
    for run, kind, input_sha in [(dev, 'dev', freeze['developer_input_sha256']),
                                 (formal, 'formal', freeze['formal_input_sha256'])]:
        contract = run['contract']
        require(contract.get('input_kind') == kind and contract.get('input_sha256') == input_sha,
                f'{kind}: input contract mismatch')
        for key, expected in [('protocol', freeze['protocol']), ('max_rounds', freeze['max_rounds']),
                              ('files_sha256', freeze['files_sha256']), ('model', config),
                              ('evidence', freeze['evidence'])]:
            require(contract.get(key) == expected, f'{kind}: {key} mismatch')
        execution = run.get('execution', {})
        require(execution.get('actual_revision') == config['model_revision'] and
                str(execution.get('actual_dtype', '')).replace('torch.', '') == config['dtype'] and
                execution.get('attention_implementation') == config['attention_implementation'],
                f'{kind}: actual runtime model configuration mismatch')
    dev_contract, formal_contract = dev['contract'], formal['contract']
    require({k: v for k, v in dev_contract.items() if k not in ('input_kind', 'input_sha256')} ==
            {k: v for k, v in formal_contract.items() if k not in ('input_kind', 'input_sha256')},
            'developer and formal protocols differ')
    frozen = timestamp(freeze.get('frozen_at_utc'), 'frozen_at_utc')
    started = timestamp(formal.get('created_at_utc'), 'formal.created_at_utc')
    require(frozen < started, 'frozen timestamp must be strictly earlier than formal run creation')
    require(dev.get('completed') is True, 'developer run is incomplete')
    dev_started = timestamp(dev.get('created_at_utc'), 'dev.created_at_utc')
    dev_done = timestamp(dev.get('completed_at_utc'), 'dev.completed_at_utc')
    require(dev_started <= dev_done <= frozen, 'developer completion must precede freeze')
    with Path(dev_input_path).open(encoding='utf-8') as f:
        dev_samples = [{'id': r['id'], 'text': r['source']} for r in json.load(f)['samples']]
    with Path(formal_input_path).open(encoding='utf-8') as f:
        formal_samples = json.load(f)
    for rows, samples, label, complete in [(dev['results'], dev_samples, 'dev', True),
                                           (formal['results'], formal_samples, 'formal', formal.get('completed'))]:
        require(type(complete) is bool, f'{label}: invalid completion flag')
        require(isinstance(rows, list) and len(rows) <= len(samples), f'{label}: invalid result count')
        require(not complete or len(rows) == len(samples), f'{label}: completed with missing results')
        require(all(row.get('id') == sample['id'] and row.get('source') == sample['text']
                    for row, sample in zip(rows, samples)), f'{label}: result prefix differs from input')
    return {'status': 'complete' if formal['completed'] else 'partial',
            'developer_samples_checked': len(dev['results']),
            'formal_samples_processed': len(formal['results']),
            'formal_samples_expected': len(formal_samples),
            'formal_samples_unprocessed': len(formal_samples) - len(formal['results']),
            'protocol': freeze['protocol'], 'max_rounds': freeze['max_rounds'],
            'frozen_at_utc': freeze['frozen_at_utc'], 'formal_created_at_utc': formal['created_at_utc'],
            'developer_run_sha256': freeze['developer_run_sha256'],
            'formal_input_sha256': freeze['formal_input_sha256'],
            'limitations': ['Recorded timestamps are self-reported, not independently timestamped proof.',
                            'Checks only protocol, files, chronology, and completed input prefix; not semantic correctness or scores.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True, type=Path)
    parser.add_argument('--dev-run', required=True, type=Path)
    parser.add_argument('--formal-run', required=True, type=Path)
    parser.add_argument('--dev-input', required=True, type=Path)
    parser.add_argument('--formal-input', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))
    report = audit(read(args.freeze), read(args.dev_run), read(args.formal_run), args.dev_run,
                   args.dev_input, args.formal_input)
    content = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        if args.output.exists():
            raise SystemExit('Output exists; choose a new filename')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')
    print(content, end='')


if __name__ == '__main__':
    main()
