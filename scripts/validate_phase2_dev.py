"""Check Phase 2 development annotations against source strings, without a model."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(data):
    require(data['dataset_id'] == 'minifact-phase2-synthetic-dev-v1', 'Unexpected dataset')
    rows = data['samples']
    require(len(rows) == 10 and len({r['id'] for r in rows}) == len(rows), 'Sample count or IDs')
    count_facts = count_spans = count_qualifiers = count_logic = 0
    for row in rows:
        source = row['source']
        require(isinstance(source, str) and bool(source), f'{row["id"]}: empty source')
        fact_ids = [f['id'] for f in row['facts']]
        require(len(fact_ids) == len(set(fact_ids)) and fact_ids, f'{row["id"]}: fact IDs')
        for fact in row['facts']:
            count_facts += 1
            quote = fact['quote']
            require(bool(quote) and bool(fact['source_spans']), f'{row["id"]}: empty quote or spans')
            seen = set()
            for span in fact['source_spans']:
                a, b = span['start_idx'], span['end_idx']
                require(type(a) is int and type(b) is int and 0 <= a < b <= len(source),
                        f'{row["id"]}: invalid fact interval')
                require(source[a:b] == quote, f'{row["id"]}: fact slice mismatch')
                require((a, b) not in seen, f'{row["id"]}: duplicate span')
                seen.add((a, b))
                count_spans += 1
            for qual in fact['qualifiers']:
                count_qualifiers += 1
                require(len(qual['spans']) == len(fact['source_spans']), f'{row["id"]}: qualifier spans')
                for span, parent in zip(qual['spans'], fact['source_spans']):
                    a, b = span['start_idx'], span['end_idx']
                    require(type(a) is int and type(b) is int and parent['start_idx'] <= a < b <= parent['end_idx'],
                            f'{row["id"]}: qualifier outside quote')
                    require(source[a:b] == qual['text'], f'{row["id"]}: qualifier slice mismatch')
        for check in row['logical_checks']:
            ids = check['fact_ids']
            require(len(ids) == 2 and len(set(ids)) == 2 and all(i in fact_ids for i in ids),
                    f'{row["id"]}: logical pair references')
            require(check['relation'] in ('contradictory', 'no_contradiction_identified'),
                    f'{row["id"]}: logical relation')
            require(bool(check['rationale']), f'{row["id"]}: empty rationale')
            count_logic += 1
    return {'samples': len(rows), 'facts': count_facts, 'fact_spans': count_spans,
            'qualifiers': count_qualifiers, 'logical_pairs': count_logic}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/phase2_dev.json')
    args = parser.parse_args()
    counts = validate(json.loads(args.input.read_text(encoding='utf-8')))
    print('PASS: ' + ', '.join(f'{key}={value}' for key, value in counts.items()))


if __name__ == '__main__':
    main()
