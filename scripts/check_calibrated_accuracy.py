"""Use the frozen development threshold on complete TRUE/FALSE label scores.

This is closed-book teacher-forced scoring with two model forward passes. It
does not generate a greedy answer, retrieve evidence or output truth probabilities.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minifact
from accuracy_backend import BINARY_PROMPT
from accuracy_calibration import encode_candidates, predict, score_candidates
from scripts.check_accuracy import digest, load_backend, write_new_output
from scripts.run_accuracy_calibration import check_calibration, code_contract

DEFAULT_CALIBRATION = ROOT / 'reports/accuracy_label_threshold_20261112.json'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    text = p.add_mutually_exclusive_group(required=True)
    text.add_argument('--text', help='One statement, preserving its exact text')
    text.add_argument('--input-file', type=Path, help='UTF-8 plain-text statement file')
    p.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument('--cache-dir', type=Path,
                   default=Path(os.environ.get('MINIFACT_MODEL_CACHE', 'E:/MiniFact-LLM-runtime/hf-cache')))
    p.add_argument('--output', type=Path, required=True)
    return p


def prepare_request(args, root=ROOT):
    """Validate the fitted protocol, without equating user text to holdout inputs."""
    if args.output.exists():
        raise ValueError('Output exists; preserve it and select a new file')
    if args.input_file is None:
        statement = args.text
        input_info = {'kind': 'command_line_text'}
    else:
        with args.input_file.open('r', encoding='utf-8-sig', newline='') as handle:
            statement = handle.read()
        input_info = {'kind': 'utf8_text_file', 'file_sha256': digest(args.input_file)}
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError('Statement must be nonempty text')
    current = code_contract()
    calibration = minifact.read_json(args.calibration)
    # The holdout hash identifies the protocol on which this threshold is
    # evaluated. It is not a requirement that a new local statement be a member
    # of that benchmark, and no holdout gold is opened here.
    check_calibration(calibration, current,
                      root / 'data/improvement/holdout_inputs.json', root=root)
    return statement, calibration, current, {
        'input': input_info, 'calibration_sha256': digest(args.calibration),
        'development_run': calibration['development_run'],
        'development_run_sha256': calibration['development_run_sha256'],
        'holdout_input_sha256': calibration['holdout_input_sha256'],
        'holdout_hash_role': 'Protocol binding only; local user text is not looked up in benchmark inputs.',
        'threshold_fit_rule': calibration['fit']['rule']}


def validate_execution(backend, execution, calibration):
    """Use the same EOS and runtime boundary as the frozen scoring experiment."""
    eos = backend.model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    if not eos_ids or any(type(token) is not int or token < 0 for token in eos_ids):
        raise ValueError('Loaded model has invalid EOS token IDs')
    terminator = (backend.tokenizer.eos_token_id if backend.tokenizer.eos_token_id in eos_ids
                  else eos_ids[0])
    expected = {key: execution[key] for key in ('runtime', 'actual_dtype', 'actual_revision',
                                                'attention_implementation', 'device_map')}
    expected['terminator_id'] = terminator
    if (type(calibration.get('terminator_id')) is not int or
            calibration['terminator_id'] != terminator or calibration.get('execution') != expected):
        raise ValueError('Calibration terminator/runtime/device map differs from loaded model')
    return terminator


def main(argv=None):
    args = parser().parse_args(argv)
    statement, calibration, contract, provenance = prepare_request(args)
    created = datetime.now(timezone.utc).isoformat()
    backend, execution, torch = load_backend(contract['model'], args.cache_dir)
    terminator = validate_execution(backend, execution, calibration)
    messages, prompt_ids, candidates = encode_candidates(
        backend.tokenizer, BINARY_PROMPT, statement, terminator)
    if candidates != calibration['candidate_token_ids']:
        raise ValueError('Candidate token IDs differ from the frozen calibration')
    torch.cuda.synchronize()
    started = time.perf_counter()
    scoring = score_candidates(backend.model, prompt_ids, candidates, terminator)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    threshold = calibration['fit']['threshold']
    verdict = predict(scoring['margin'], threshold)
    scoring['messages'] = messages
    output = {'created_at_utc': created, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'completed': True, 'mode': 'binary_teacher_forced_closed_book',
              'source': statement, 'verdict': verdict, 'threshold': threshold,
              'prediction_rule': contract['prediction'], 'model': contract['model'],
              'contract': contract, 'execution': {**execution, 'terminator_id': terminator},
              'inference_script_sha256': digest(Path(__file__)), 'provenance': provenance,
              'candidate_scoring': scoring, 'elapsed_seconds': elapsed,
              'interpretation': 'Unverified binary model verdict from two complete teacher-forced label+EOS scores and a development-fitted decision threshold. No actual greedy generation, external evidence, web retrieval or calibrated truth probability.'}
    write_new_output(args.output, output)
    print(json.dumps({'mode': output['mode'], 'verdict': verdict,
                      'margin': scoring['margin'], 'threshold': threshold,
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
