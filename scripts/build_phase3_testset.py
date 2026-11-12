"""Build/check the frozen whole-sample Week 6 comparison set without model access.

With --source, verify the pinned upstream bytes and reproduce the selection.
Without --source, --check validates the shipped data and the preserved Phase1 set.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

try:
    from scripts.curate_phase1 import (canonical_article, encoded, normalized_text,
                                      record_hash, rejection, restore_upstream,
                                      sample_from_row)
except ModuleNotFoundError:
    from curate_phase1 import (canonical_article, encoded, normalized_text,
                              record_hash, rejection, restore_upstream,
                              sample_from_row)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/phase3'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_phase1(config):
    """Fail rather than silently use a modified original test set."""
    for name in ('dataset', 'inputs'):
        path = ROOT / f'data/phase1/{name}.json'
        if file_hash(path) != config['phase1'][f'{name}_sha256']:
            raise ValueError(f'Frozen Phase1 {name} SHA256 mismatch')
    return json.loads((ROOT / 'data/phase1/dataset.json').read_text(encoding='utf-8'))


def ranked_key(namespace, seed, *fields):
    value = '|'.join(str(v) for v in (namespace, seed, *fields))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def select_controls(rows, config, phase1):
    # Reuse the established structural criteria, not Phase1's Refuted-only label
    # requirement or its curated manual exclusions/claim-type quotas.
    filters = {'selection': config['selection'], 'source': {'required_label': 'Supported'},
               'manual_exclusions': config['manual_exclusions']}
    reasons = {i: rejection(i, row, filters) for i, row in enumerate(rows)}
    eligible = [i for i in reasons if reasons[i] is None]
    articles = {canonical_article(s['provenance']['fact_checking_article'])
                for s in phase1['samples']}
    texts = {normalized_text(s['text']) for s in phase1['samples']}
    rules = config['selection']
    ordered = sorted(eligible, key=lambda i: ranked_key(
        rules['rank_namespace'], rules['seed'], i, rows[i]['claim']))
    selected, duplicate_rejections = [], {}
    for index in ordered:
        row = rows[index]
        article = canonical_article(row['fact_checking_article'])
        text = normalized_text(row['claim'])
        if article in articles or text in texts:
            duplicate_rejections[str(index)] = ('repeated_factchecking_article'
                                               if article in articles else 'repeated_text')
            continue
        selected.append(index)
        articles.add(article)
        texts.add(text)
        if len(selected) == rules['max_controls']:
            break
    selected.sort()
    if selected != config['final_control_source_indices']:
        raise ValueError('Frozen control source indices mismatch')
    # The reused helper names its label rejection not_refuted. Relabel the audit
    # reason to avoid implying that controls were required to be Refuted.
    rejection_counts = Counter('not_supported' if v == 'not_refuted' else v
                               for v in reasons.values() if v is not None)
    return selected, {
        'source_records': len(rows),
        'source_labels': dict(sorted(Counter(r['label'] for r in rows).items())),
        'structurally_eligible_after_manual_exclusions': len(eligible),
        'filter_rejection_counts': dict(sorted(rejection_counts.items())),
        'duplicate_rejections_in_rank_order': duplicate_rejections,
        'selected_control_count': len(selected),
        'maximum_control_count': rules['max_controls'],
        'selection_limit_reached': len(selected) == rules['max_controls'],
    }


def ordered_samples(samples, config):
    rules = config['selection']
    return sorted(samples, key=lambda s: ranked_key(
        rules['input_order_namespace'], rules['seed'], s['id'], s['text']))


def input_rows(dataset):
    return [{'id': s['id'], 'text': s['text']} for s in dataset['samples']]


def statistics(samples):
    result = {'sample_count': len(samples),
              'verdicts': dict(sorted(Counter(s['gold']['verdict'] for s in samples).items())),
              'by_verdict': {}}
    for verdict in ('REFUTED', 'SUPPORTED'):
        group = [s for s in samples if s['gold']['verdict'] == verdict]
        lengths = [len(s['text'].split()) for s in group]
        domains = Counter(urlsplit(canonical_article(
            s['provenance']['fact_checking_article'])).hostname for s in group)
        result['by_verdict'][verdict] = {
            'sample_count': len(group),
            'primary_upstream_claim_types': dict(sorted(Counter(
                s['provenance']['claim_types'][0] for s in group).items())),
            'factchecker_domains': dict(sorted(domains.items())),
            'factchecker_domain_count': len(domains),
            'word_count': {'method': 'Python str.split', 'min': min(lengths),
                           'max': max(lengths), 'mean': round(sum(lengths) / len(lengths), 2)},
            'claim_date_min': min(s['claim_date'] for s in group),
            'claim_date_max': max(s['claim_date'] for s in group),
            'question_count': sum(len(s['gold']['questions']) for s in group),
            'answer_count': sum(len(q['answers']) for s in group for q in s['gold']['questions']),
        }
    return result


def build(source, config):
    if file_hash(source) != config['source']['sha256']:
        raise ValueError('Source SHA256 mismatch; refusing a changed upstream version')
    rows = json.loads(source.read_bytes())
    phase1 = load_phase1(config)
    indices, audit = select_controls(rows, config, phase1)
    samples = copy.deepcopy(phase1['samples'])
    for n, index in enumerate(indices, 1):
        sample = sample_from_row(index, rows[index], n)
        sample['id'] = f'MF-P3-C-{n:04d}'
        sample['gold']['verdict'] = 'SUPPORTED'
        sample['annotations']['status'] = 'not_annotated'
        samples.append(sample)
    samples = ordered_samples(samples, config)
    license_info = copy.deepcopy(phase1['license'])
    license_info['changes'] = (
        'Preserved the original 60 Phase1 Refuted samples without modification; '
        'selected 50 Supported dev controls, added project IDs and standardized '
        'dates/provenance, and deterministically interleaved sample order. '
        'Original claims, labels, justifications, QA and metadata are preserved.')
    return {
        '$schema': './dataset.schema.json', 'schema_version': '1.0.0',
        'dataset_id': config['dataset_id'], 'version': config['version'],
        'curation_date': config['curation_date'], 'language': 'en',
        'sample_unit': 'real_world_news_related_claim', 'project_split': 'test',
        'license': license_info, 'source': config['source'],
        'phase1': config['phase1'],
        'review': {
            'method': 'Agent screening of supplied claims/justifications and targeted QA; deterministic text/article deduplication; one cross-class story exclusion recorded in curation.json.',
            'independent_human_reverification': False,
            'live_source_retrieval_performed': False,
            'gold_origin': 'Inherited upstream benchmark annotations at their original claim date; not current truth or independently reverified labels.',
            'atomic_gold_status': 'No independent atomic-fact, qualifier or error-type annotations; whole-sample verdicts must not be assigned to extracted atoms.',
            'class_matching': config['selection']['class_matching'],
            'test_exposure': 'The original 60 Refuted inputs have prior Phase1/Phase2 model outputs; controls were selected before Week6 outputs. Public dev data may occur in pretraining.',
        },
        'statistics': statistics(samples), 'selection_audit': audit, 'samples': samples,
    }


def validate(dataset, inputs, config, schema=None):
    if schema is not None:
        from jsonschema import Draft202012Validator, FormatChecker
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(dataset)
    phase1 = load_phase1(config)
    samples = dataset['samples']
    if len(samples) != config['sample_count']:
        raise ValueError('Sample count mismatch')
    if any(set(row) != {'id', 'text'} for row in inputs):
        raise ValueError('Model inputs contain extra fields / potential gold leakage')
    if inputs != input_rows(dataset):
        raise ValueError('Model inputs mismatch')
    if samples != ordered_samples(samples, config):
        raise ValueError('Frozen sample order mismatch')
    by_id = {s['id']: s for s in samples}
    if len(by_id) != len(samples):
        raise ValueError('Duplicate sample ID')
    for original in phase1['samples']:
        if by_id.get(original['id']) != original:
            raise ValueError('Original Phase1 sample modified or missing')
    controls = sorted((s for s in samples if s['id'].startswith('MF-P3-C-')),
                      key=lambda s: s['id'])
    if len(controls) != config['control_count'] or [s['provenance']['source_record_index']
            for s in controls] != config['final_control_source_indices']:
        raise ValueError('Frozen control source indices mismatch')
    if [s['id'] for s in controls] != [f'MF-P3-C-{i:04d}' for i in range(1, len(controls) + 1)]:
        raise ValueError('Control IDs mismatch')
    texts, articles = set(), set()
    for sample in samples:
        text = normalized_text(sample['text'])
        article = canonical_article(sample['provenance']['fact_checking_article'])
        if text in texts:
            raise ValueError('Duplicate normalized text')
        if article in articles:
            raise ValueError('Duplicate fact-checking article')
        texts.add(text)
        articles.add(article)
        original = restore_upstream(sample)
        if record_hash(original) != sample['provenance']['source_record_sha256']:
            raise ValueError('Upstream record hash mismatch')
        expected_label = {'Refuted': 'REFUTED', 'Supported': 'SUPPORTED'}.get(original['label'])
        if expected_label != sample['gold']['verdict'] or sample['gold']['verdict_scope'] != 'whole_sample':
            raise ValueError('Whole-sample verdict mapping mismatch')
        if sample['claim_date'] != datetime.strptime(original['claim_date'], '%d-%m-%Y').date().isoformat():
            raise ValueError('Normalized date mismatch')
        if sample in controls and expected_label != 'SUPPORTED':
            raise ValueError('Control is not Supported')
    if dataset['source'] != config['source'] or dataset['phase1'] != config['phase1']:
        raise ValueError('Pinned provenance mismatch')
    if dataset['statistics'] != statistics(samples):
        raise ValueError('Dataset statistics mismatch')
    for name, value in (('dataset', dataset), ('inputs', inputs)):
        digest = hashlib.sha256(encoded(value)).hexdigest()
        if digest != config['artifacts'][f'{name}_sha256']:
            raise ValueError(f'Frozen Phase3 {name} SHA256 mismatch')
    return dataset['statistics']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Local pinned AVeriTeC dev.json; never downloaded automatically')
    parser.add_argument('--check', action='store_true', help='Validate delivered data; with --source, also rebuild byte-for-byte')
    args = parser.parse_args()
    config = json.loads((DATA / 'curation.json').read_text(encoding='utf-8'))
    if not args.source and not args.check:
        parser.error('Rebuilding requires --source; use --check to validate delivered data')
    if args.source:
        dataset = build(args.source, config)
        for name, content in {'dataset.json': encoded(dataset),
                              'inputs.json': encoded(input_rows(dataset))}.items():
            path = DATA / name
            if args.check:
                if not path.exists() or path.read_bytes() != content:
                    raise SystemExit(f'Stale generated dataset: {path}')
            else:
                path.write_bytes(content)
    dataset = json.loads((DATA / 'dataset.json').read_text(encoding='utf-8'))
    inputs = json.loads((DATA / 'inputs.json').read_text(encoding='utf-8'))
    schema = json.loads((DATA / 'dataset.schema.json').read_text(encoding='utf-8'))
    stats = validate(dataset, inputs, config, schema)
    for name in ('dataset', 'inputs'):
        if file_hash(DATA / f'{name}.json') != config['artifacts'][f'{name}_sha256']:
            raise ValueError(f'Frozen Phase3 {name} file bytes differ; preserve LF bytes')
    print(f'PASS: {stats["sample_count"]} whole-sample records; {stats["verdicts"]}; inputs contain only id/text.')


if __name__ == '__main__':
    main()
