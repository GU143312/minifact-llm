"""Check one statement using a development-selected local Qwen method.

Example: python scripts/check_accuracy.py --text "A statement to check" \
    --output runs/local_check.json

Evidence mode additionally requires --method evidence --evidence-file packet.json.
Packets contain context={claim_date,speaker,location_ISO_code} and a nonempty
evidence array of {question,answer,url}. This tool does not retrieve evidence or
establish truth; its output preserves unverified model judgments and scores.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minifact
from accuracy_backend import AccuracyBackend, CLOSED_CANDIDATES

METHODS = (*CLOSED_CANDIDATES, 'evidence')
DEFAULT_SELECTION = ROOT / 'reports/accuracy_selection_20261112.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repository_path(root, value):
    """Resolve data references, never executable paths, inside the repository."""
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        raise ValueError('Selection references must be repository-relative paths')
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Selection reference escapes repository')
    return resolved


def load_selection(path, root=ROOT):
    """Bind automatic method choice to its completed development run and code."""
    path = Path(path)
    selected = minifact.read_json(path)
    if not isinstance(selected, dict) or selected.get('selected_closed_book') not in CLOSED_CANDIDATES:
        raise ValueError('Selection must name a known closed-book candidate')
    reference = repository_path(root, selected.get('development_run'))
    if digest(reference) != selected.get('development_run_sha256'):
        raise ValueError('Selected development run hash differs')
    run = minifact.read_json(reference)
    contract = run.get('contract', {})
    method = selected['selected_closed_book']
    if (run.get('completed') is not True or contract.get('split') != 'dev' or
            method not in contract.get('arms', [])):
        raise ValueError('Selection must refer to a completed development candidate')
    hashes = contract.get('files_sha256')
    if (not isinstance(hashes, dict) or 'accuracy_backend.py' not in hashes or
            'config.json' not in hashes):
        raise ValueError('Development run lacks method/configuration source hashes')
    for name, expected in hashes.items():
        if digest(repository_path(root, name)) != expected:
            raise ValueError(f'Development-selected source changed: {name}')
    return method, {'kind': 'development_selection', 'selection_sha256': digest(path),
                    'development_run': selected['development_run'],
                    'development_run_sha256': digest(reference),
                    'files_sha256': hashes}


def validate_evidence_packet(packet):
    """Accept QA references, never a benchmark gold record or justification."""
    if not isinstance(packet, dict) or set(packet) != {'context', 'evidence'}:
        raise ValueError('Evidence packet must contain only context and evidence')
    context, evidence = packet['context'], packet['evidence']
    if (not isinstance(context, dict) or
            set(context) != {'claim_date', 'speaker', 'location_ISO_code'} or
            any(value is not None and not isinstance(value, str) for value in context.values())):
        raise ValueError('Context must contain claim_date, speaker, location_ISO_code as text or null')
    if context['claim_date']:
        try:
            date.fromisoformat(context['claim_date'])
        except ValueError as exc:
            raise ValueError('claim_date must be an ISO date or null') from exc
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Evidence mode requires at least one QA reference')
    for row in evidence:
        if not isinstance(row, dict) or set(row) != {'question', 'answer', 'url'}:
            raise ValueError('Each reference must contain only question, answer and url')
        if any(not isinstance(value, str) or not value.strip() for value in row.values()):
            raise ValueError('Reference fields must be nonempty strings')
        parsed = urlsplit(row['url'])
        if parsed.scheme not in ('https', 'http') or not parsed.netloc:
            raise ValueError('Reference URL must be HTTP(S)')
    return packet


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    statement = p.add_mutually_exclusive_group(required=True)
    statement.add_argument('--text', help='One statement; whitespace and punctuation are preserved')
    statement.add_argument('--input-file', type=Path, help='UTF-8 plain-text statement file')
    p.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    p.add_argument('--method', choices=METHODS,
                   help='Explicit override; binary/structured request a forced choice, otherwise use development selection')
    p.add_argument('--evidence-file', type=Path)
    p.add_argument('--cache-dir', type=Path,
                   default=Path(os.environ.get('MINIFACT_MODEL_CACHE', 'E:/MiniFact-LLM-runtime/hf-cache')))
    p.add_argument('--output', type=Path, required=True)
    return p


def prepare_request(args, root=ROOT):
    """Validate all user inputs and isolate closed-book payload before loading GPU."""
    if args.output.exists():
        raise ValueError('Output exists; preserve it and select a new file')
    if args.input_file is not None:
        with args.input_file.open('r', encoding='utf-8-sig', newline='') as handle:
            text = handle.read()
        input_info = {'kind': 'utf8_text_file', 'file_sha256': digest(args.input_file)}
    else:
        text = args.text
        input_info = {'kind': 'command_line_text'}
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Statement must be nonempty text')
    if args.method:
        if args.method not in METHODS:
            raise ValueError('Unknown inference method')
        method, choice = args.method, {'kind': 'explicit_cli_override_not_development_selection'}
    else:
        method, choice = load_selection(args.selection, root)
    row, evidence_info = {'id': 'local-statement', 'text': text}, None
    if method == 'evidence':
        if args.evidence_file is None:
            raise ValueError('Evidence method requires --evidence-file')
        packet = validate_evidence_packet(minifact.read_json(args.evidence_file))
        row.update(packet)
        evidence_info = {'kind': 'user_supplied_qa_packet', 'retrieval_performed': False,
                         'file_sha256': digest(args.evidence_file), 'packet': packet}
    elif args.evidence_file is not None:
        raise ValueError('--evidence-file is only permitted with --method evidence')
    return method, row, {'input': input_info, 'method_choice': choice,
                         'evidence_provenance': evidence_info}


def load_backend(config, cache_dir):
    """Load only the configured local revision, using the tested runtime settings."""
    snapshot = cache_dir / ('models--' + config['model_id'].replace('/', '--')) / 'snapshots' / config['model_revision']
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
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), dtype=getattr(torch, config['dtype']), device_map='auto',
        attn_implementation=config['attention_implementation'],
        max_memory={0: '5GiB', 'cpu': '4GiB'}, local_files_only=True).eval()
    if (str(model.dtype).replace('torch.', '') != config['dtype'] or
            model.config._commit_hash != config['model_revision'] or
            model.config._attn_implementation != config['attention_implementation']):
        raise ValueError('Loaded model dtype, revision or attention differs from configuration')
    execution = {'runtime': minifact.runtime_info(torch), 'actual_dtype': str(model.dtype),
                 'actual_revision': model.config._commit_hash,
                 'attention_implementation': model.config._attn_implementation,
                 'device_map': {str(k): str(v) for k, v in model.hf_device_map.items()},
                 'model_generation_config': model.generation_config.to_dict()}
    return AccuracyBackend(model, tokenizer, config), execution, torch


def write_new_output(path, value):
    """Exclusive creation also protects against another writer during inference."""
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(encoded)


def main(argv=None):
    args = parser().parse_args(argv)
    method, row, provenance = prepare_request(args)
    config = minifact.read_json(ROOT / 'config.json')
    created = datetime.now(timezone.utc).isoformat()
    backend, execution, torch = load_backend(config, args.cache_dir)
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = backend.run_arm(row, method)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = {'created_at_utc': created, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'completed': True, 'method': method,
              'condition': 'supplied_reference_evidence' if method == 'evidence' else 'closed_book',
              'decision_policy': 'forced_choice_prompt' if method in ('binary', 'structured') else 'allows_unknown',
              'source': row['text'], 'model': config, 'execution': execution,
              'inference_script_sha256': digest(Path(__file__)), 'provenance': provenance,
              'elapsed_seconds': elapsed, 'result': result,
              'interpretation': 'Unverified model verdict and generated-token scores, not an independent fact check or calibrated truth probability. Supplied QA is user-provided; no web retrieval is performed.'}
    write_new_output(args.output, output)
    print(json.dumps({'method': method, 'condition': output['condition'],
                      'verdict': result.get('verdict'), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
