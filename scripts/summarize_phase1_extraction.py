"""Summarize a formal extraction run without inferring fact-checking accuracy.

The run must correspond, in order, to the frozen input file. The optional
--allow-partial admits only a valid prefix and labels it as incomplete. Token
mapping checks validate the recorded fields' internal consistency; the runner
does not persist tokenizer offsets, so this script cannot independently verify
the reported token boundaries against the tokenizer.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'data/phase1/inputs.json'
EXPECTED_INPUT_SHA256 = '0de38bd1a9348ccdf75216e80f3eb679fddd94e6605f8ef9f5f8a753e6f5e434'


def require(test, message):
    if not test:
        raise ValueError(message)


def validate_mapping(mapping, claim, location, token_count):
    status = mapping.get('status')
    if location.get('status') != 'ok':
        require(status == 'source_' + str(location.get('status')), 'source-location status mismatch')
        require(set(mapping) == {'quote_index', 'status'}, 'unexpected fields for missing source')
        return status
    require(status in {'ok', 'partial_token', 'uncovered', 'non_contiguous'}, 'unknown token status')
    start = location['start'] + claim['start_idx']
    end = location['start'] + claim['end_idx']
    require(mapping.get('char_start') == start and mapping.get('char_end') == end, 'mapped character bounds mismatch')
    require(mapping.get('text') == claim['claim'], 'mapped text mismatch')
    hits = mapping.get('token_indices')
    require(isinstance(hits, list) and all(type(i) is int and 0 <= i < token_count for i in hits), 'invalid token indices')
    require(hits == sorted(set(hits)), 'token indices must be ordered and unique')
    a, b = mapping.get('token_start'), mapping.get('token_end')
    c, d = mapping.get('covered_char_start'), mapping.get('covered_char_end')
    if not hits:
        require(status == 'uncovered' and a is None and b is None and c is None and d is None, 'empty token mapping mismatch')
        return status
    require(a == hits[0] and b == hits[-1] + 1, 'token bounds do not enclose hits')
    require(type(c) is int and type(d) is int and 0 <= c < d, 'invalid covered character bounds')
    contiguous = hits == list(range(a, b))
    if status == 'non_contiguous':
        require(not contiguous, 'non-contiguous token status with contiguous indices')
    else:
        require(contiguous, 'token status contradicts non-contiguous indices')
    if status == 'ok':
        require((c, d) == (start, end), 'exact token coverage bounds mismatch')
    elif status == 'partial_token':
        require(c <= start and d >= end and (c, d) != (start, end), 'partial token coverage bounds mismatch')
    return status


def summarize(artifact, inputs, input_hash, allow_partial=False):
    require(input_hash == EXPECTED_INPUT_SHA256, 'formal input SHA256 differs from the frozen release')
    require(isinstance(inputs, list) and len(inputs) == 60, 'formal input must contain 60 cases')
    require(artifact.get('file_sha256', {}).get('data/phase1/inputs.json') == input_hash,
            'run input hash does not match frozen input')
    rows = artifact.get('results')
    require(isinstance(rows, list) and len(rows) <= len(inputs), 'invalid run results')
    completed = artifact.get('completed') is True
    require(completed or allow_partial, 'run incomplete; use --allow-partial to summarize the completed prefix')
    require(not completed or len(rows) == len(inputs), 'completed run does not contain all inputs')
    reasons, token_status = Counter(), Counter()
    details = []
    for n, (row, sample) in enumerate(zip(rows, inputs), 1):
        prefix = f'row {n} ({sample["id"]}): '
        require(row.get('case_id') == sample['id'] and row.get('source') == sample['text'], prefix + 'input mismatch')
        source = sample['text']
        location = row.get('prompt_source_location')
        require(isinstance(location, dict) and location.get('status') in {'ok', 'not_found', 'ambiguous'}, prefix + 'bad source location')
        if location['status'] == 'ok':
            start, end = location.get('start'), location.get('end')
            require(type(start) is int and type(end) is int and start >= 0 and end == start + len(source)
                    and location.get('occurrences') == [start], prefix + 'inconsistent source location')
        elif location['status'] == 'not_found':
            require(location.get('occurrences') == [], prefix + 'invalid source occurrences')
        else:
            occ = location.get('occurrences')
            require(isinstance(occ, list) and len(occ) >= 2 and all(type(i) is int and i >= 0 for i in occ)
                    and occ == sorted(set(occ)), prefix + 'invalid ambiguous source occurrences')
        claims, rejects, mappings = row.get('aligned_claims'), row.get('alignment_rejections'), row.get('token_alignment')
        require(all(isinstance(x, list) for x in [claims, rejects, mappings]), prefix + 'invalid alignment lists')
        require(len(claims) == len(mappings), prefix + 'token mappings count differs from accepted claims')
        count = row.get('prompt_token_count')
        require(type(count) is int and count > 0, prefix + 'invalid prompt token count')
        seen = set()
        for claim, mapping in zip(claims, mappings):
            a, b, text = claim.get('start_idx'), claim.get('end_idx'), claim.get('claim')
            idx = claim.get('quote_index')
            require(type(a) is int and type(b) is int and 0 <= a < b <= len(source)
                    and isinstance(text, str) and source[a:b] == text, prefix + 'source slice mismatch')
            require(type(idx) is int and idx >= 0 and idx not in seen and mapping.get('quote_index') == idx,
                    prefix + 'quote index mismatch')
            quotes = row.get('raw_quotes')
            require(isinstance(quotes, list) and idx < len(quotes) and quotes[idx] == text,
                    prefix + 'accepted quote differs from parsed model output')
            seen.add(idx)
            token_status[validate_mapping(mapping, claim, location, count)] += 1
        for rejection in rejects:
            reason = rejection.get('reason')
            require(isinstance(reason, str) and bool(reason), prefix + 'missing rejection reason')
            reasons[reason] += 1
        try:
            parsed = json.loads(row['raw_response'])
            parse_error = False
        except (json.JSONDecodeError, TypeError):
            parsed, parse_error = None, True
        require(parse_error == bool(row.get('json_error')) and parsed == row.get('raw_quotes'), prefix + 'JSON parsing record mismatch')
        values, token_n = row.get('token_logprobs'), row.get('generated_token_count')
        require(isinstance(values, list) and len(values) > 0 and type(token_n) is int and len(values) == token_n
                and all(type(v) in (int, float) and math.isfinite(v) for v in values), prefix + 'invalid token logprobs')
        mean = row.get('mean_quote_json_logprob')
        require(type(mean) in (int, float) and math.isfinite(mean) and math.isclose(mean, sum(values)/len(values), abs_tol=1e-6),
                prefix + 'mean logprob mismatch')
        require(type(row.get('ended_with_eos')) is bool, prefix + 'missing EOS flag')
        details.append({'id': sample['id'], 'accepted_quotes': len(claims), 'rejection_records': len(rejects),
                        'json_parse_error': parse_error, 'ended_with_eos': row['ended_with_eos'],
                        'possible_truncation_no_eos': not row['ended_with_eos'], 'mean_generated_json_logprob': mean})
    return {'completed': completed, 'partial': not completed, 'expected_samples': len(inputs), 'observed_samples': len(rows),
            'accepted_quotes': sum(r['accepted_quotes'] for r in details),
            'rejection_records': sum(r['rejection_records'] for r in details),
            'rejection_reasons': dict(sorted(reasons.items())), 'token_mapping_statuses': dict(sorted(token_status.items())),
            'json_parse_error_samples': sum(r['json_parse_error'] for r in details),
            'possible_truncation_no_eos_samples': sum(r['possible_truncation_no_eos'] for r in details),
            'token_alignment_check_scope': 'Internal fields only; tokenizer offsets were not saved and could not be independently verified.',
            'score_definition': 'Mean log-probability of generated non-special JSON quote tokens, not verifier confidence or truth probability.',
            'unassessed': ['factual coverage', 'qualifier retention', 'independent-claim quality', 'veracity accuracy'],
            'samples': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = summarize(json.loads(args.run.read_text(encoding='utf-8')), json.loads(raw),
                       hashlib.sha256(raw).hexdigest(), args.allow_partial)
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='\n') as target:
            target.write(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
