"""Build reproducible project-development/holdout samples from pinned AVeriTeC train.

No model is run. Without --source, --check validates shipped bytes and provenance.
With --source, the complete frozen selection is reproduced byte-for-byte.
"""
import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.curate_phase1 import canonical_article, normalized_text, record_hash, rejection

DATA = ROOT / 'data/improvement'
SOURCE = {
    'benchmark': 'AVeriTeC', 'repository': 'MichSchli/AVeriTeC',
    'revision': '7c62d1ec8df3fb560d6efe2b85fa191135636f81',
    'file': 'data/train.json', 'original_split': 'train',
    'sha256': 'ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c',
    'git_blob_sha': '0f190e115cf2ee23416e8a539c8d6ac043d7cc83'}
RULES = {
    'seed': 42, 'rank_namespace': 'minifact-improvement-selection-v1',
    'order_namespace': 'minifact-improvement-order-v1',
    'dev_per_label': 16, 'holdout_per_label': 40,
    'min_words': 8, 'max_words': 65, 'max_evidence_items': 8,
    'max_answer_characters': 900, 'max_evidence_characters': 5000,
    'exclude_required_reannotation': True,
    'deduplication': 'Normalized claim and canonical fact-checking article, globally across old110, development and holdout.',
    'evidence_exclusions': 'Non-text/unanswerable answers; explicit verdict Q/A; answers sourced from the original fact-check article. URLs remain provenance only; model runners must omit URL strings.'}
LABELS = {'Supported': 'SUPPORTED', 'Refuted': 'REFUTED'}
LICENSE = {
    'id': 'CC-BY-NC-4.0', 'url': 'https://creativecommons.org/licenses/by-nc/4.0/',
    'attribution': 'Michael Sejr Schlichtkrull, Zhijiang Guo, and Andreas Vlachos (2023). AVeriTeC.',
    'citation_url': 'https://arxiv.org/abs/2305.13117',
    'changes': 'Selected and deterministically split train records; added project IDs/provenance. Claims and upstream records are unchanged. Separate evidence projections filter explicit verdicts/curator articles and truncate answers at fixed Unicode-character budgets.'}
ARTIFACTS = tuple(f'{split}_{kind}.json' for split in ('dev', 'holdout')
                  for kind in ('dataset', 'inputs', 'evidence_inputs'))

# Conservative lexical filtering does not guarantee that an answer contains no
# implicit verdict. Supplied benchmark QA remains privileged evidence, not retrieval.
VERDICT_PATTERNS = (
    r'\b(?:claim|statement|allegation|rumou?r|post|assertion)\b.{0,100}\b(?:true|false|misleading|incorrect|correct|fake|baseless|unfounded|debunked|refuted|supported)\b',
    r'\b(?:false|fake|misleading|baseless|unfounded|debunked|refuted)\s+(?:claim|statement|allegation|rumou?r|post|assertion)\b',
    r'\b(?:fact[ -]?check|verdict|rating|rated|pants on fire|mostly false|mostly true|half true)\b')


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rank(namespace, *values):
    return sha('|'.join(str(x) for x in (namespace, RULES['seed'], *values)).encode())


def explicit_verdict(text):
    return any(re.search(pattern, text, re.I | re.S) for pattern in VERDICT_PATTERNS)


def evidence_projection(row):
    """Deterministic supplied-QA projection; never reads label/justification."""
    items, budget = [], RULES['max_evidence_characters']
    fact_article = canonical_article(row.get('fact_checking_article', ''))
    for question in row.get('questions', []):
        q = question.get('question', '').strip()
        if not q or explicit_verdict(q):
            continue
        for answer in question.get('answers', []):
            a, url = answer.get('answer', '').strip(), answer.get('source_url', '').strip()
            if (answer.get('answer_type') == 'Unanswerable' or
                    answer.get('source_medium', '').casefold() not in ('web text', 'pdf') or
                    not a or not url.startswith(('https://', 'http://')) or
                    canonical_article(url) == fact_article or explicit_verdict(a)):
                continue
            text = a[:min(RULES['max_answer_characters'], budget)]
            if not text:
                return items
            items.append({'question': q, 'answer': text, 'url': url})
            budget -= len(text)
            if len(items) == RULES['max_evidence_items'] or not budget:
                return items
    return items


def exclusion(index, row):
    if row.get('label') not in LABELS:
        return 'not_binary_label'
    if row.get('required_reannotation') is not False:
        return 'requires_reannotation'
    config = {'selection': RULES, 'source': {'required_label': row['label']}, 'manual_exclusions': {}}
    reason = rejection(index, row, config)
    if reason:
        return reason
    if re.search(r'\bfact[ -]?check(?:ing)?\b', row['claim'], re.I):
        return 'editorial_claim_cue'
    try:
        datetime.strptime(row['claim_date'], '%d-%m-%Y')
    except (ValueError, TypeError):
        return 'invalid_date'
    return None if evidence_projection(row) else 'no_filtered_evidence'


def sample(index, row, split, sequence):
    context = {key: row.get(key) for key in ('speaker', 'location_ISO_code', 'reporting_source')}
    context['claim_date'] = datetime.strptime(row['claim_date'], '%d-%m-%Y').date().isoformat()
    return {'id': f'MFI-{"D" if split == "dev" else "H"}-{sequence:04d}',
            'text': row['claim'], 'context': context,
            'gold': {'verdict': LABELS[row['label']], 'original_label': row['label'],
                     'verdict_scope': 'whole_sample', 'questions': copy.deepcopy(row['questions']),
                     'justification': row['justification']},
            'provenance': {'original_split': 'train', 'source_record_index': index,
                           'source_record_sha256': record_hash(row)},
            'upstream_record': copy.deepcopy(row)}


def evidence_input(sample):
    return {'id': sample['id'], 'text': sample['text'],
            'context': {key: sample['context'][key] for key in ('claim_date', 'speaker', 'location_ISO_code')},
            'evidence': evidence_projection(sample['upstream_record'])}


def build(source_bytes, old_bytes):
    if sha(source_bytes) != SOURCE['sha256']:
        raise ValueError('Pinned upstream source SHA256 mismatch')
    rows, old = json.loads(source_bytes), json.loads(old_bytes)
    reasons = {i: exclusion(i, row) for i, row in enumerate(rows)}
    candidates = sorted((i for i in reasons if reasons[i] is None),
                        key=lambda i: rank(RULES['rank_namespace'], i, rows[i]['claim']))
    texts = {normalized_text(s['text']) for s in old['samples']}
    articles = {canonical_article(s['provenance']['fact_checking_article']) for s in old['samples']}
    outputs, selected, duplicate_counts = {}, {}, Counter()
    for split in ('dev', 'holdout'):
        indices, counts = [], Counter()
        for i in candidates:
            row = rows[i]
            if counts[row['label']] == RULES[split + '_per_label']:
                continue
            text, article = normalized_text(row['claim']), canonical_article(row['fact_checking_article'])
            if text in texts or article in articles:
                duplicate_counts[split] += 1
                continue
            indices.append(i)
            texts.add(text)
            articles.add(article)
            counts[row['label']] += 1
            if all(counts[label] == RULES[split + '_per_label'] for label in LABELS):
                break
        if not all(counts[label] == RULES[split + '_per_label'] for label in LABELS):
            raise ValueError(f'Insufficient disjoint {split} candidates: {counts}')
        selected[split] = sorted(indices)
        samples = [sample(i, rows[i], split, j) for j, i in enumerate(sorted(indices), 1)]
        samples.sort(key=lambda s: rank(RULES['order_namespace'], s['id'], s['text']))
        dataset = {'schema_version': '1.0.0', 'dataset_id': f'minifact-improvement-{split}-v1',
                   'project_split': split, 'source': SOURCE, 'license': LICENSE,
                   'review': {'selection': 'Deterministic structural filtering; no model-output selection.',
                              'gold_origin': 'Inherited upstream whole-sample historical labels.',
                              'independent_human_reverification': False,
                              'atomic_gold_available': False,
                              'unseen_boundary': 'Not previously used in this project; public upstream train may have occurred in pretraining. No exhaustive semantic event clustering.'},
                   'statistics': {'sample_count': len(samples), 'labels': dict(sorted(Counter(s['gold']['verdict'] for s in samples).items()))},
                   'samples': samples}
        outputs[f'{split}_dataset.json'] = dataset
        outputs[f'{split}_inputs.json'] = [{'id': s['id'], 'text': s['text']} for s in samples]
        outputs[f'{split}_evidence_inputs.json'] = [evidence_input(s) for s in samples]
    manifest = {'version': '1.0.0', 'curation_date': '2026-11-12', 'source': SOURCE,
                'old_comparison_dataset_sha256': sha(old_bytes), 'selection': RULES,
                'selected_train_indices': selected,
                'audit': {'source_records': len(rows), 'source_labels': dict(sorted(Counter(r['label'] for r in rows).items())),
                          'filter_rejection_counts': dict(sorted(Counter(r for r in reasons.values() if r).items())),
                          'eligible_counts': dict(sorted(Counter(rows[i]['label'] for i in candidates).items())),
                          'article_or_text_duplicates_skipped': dict(duplicate_counts)},
                'artifacts_sha256': {name: sha(encoded(value)) for name, value in outputs.items()}}
    return {**outputs, 'manifest.json': manifest}


def validate(directory=DATA, old_path=ROOT / 'data/phase3/dataset.json'):
    import jsonschema
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    if manifest['source'] != SOURCE or manifest['selection'] != RULES:
        raise ValueError('Frozen source/selection changed')
    old_bytes = old_path.read_bytes()
    if sha(old_bytes) != manifest['old_comparison_dataset_sha256']:
        raise ValueError('Original comparison dataset changed')
    old = json.loads(old_bytes)
    texts = {normalized_text(s['text']) for s in old['samples']}
    articles = {canonical_article(s['provenance']['fact_checking_article']) for s in old['samples']}
    all_ids, all_indices = set(), set()
    schema = json.loads((directory / 'dataset.schema.json').read_bytes())
    for name in ARTIFACTS:
        if sha((directory / name).read_bytes()) != manifest['artifacts_sha256'][name]:
            raise ValueError(f'Frozen artifact SHA256 mismatch: {name}')
    stats = {}
    for split in ('dev', 'holdout'):
        dataset = json.loads((directory / f'{split}_dataset.json').read_bytes())
        jsonschema.validate(dataset, schema)
        samples = dataset['samples']
        count = RULES[split + '_per_label']
        if dataset['source'] != SOURCE or dataset['license'] != LICENSE or len(samples) != 2 * count:
            raise ValueError('Dataset envelope/count mismatch')
        if Counter(s['gold']['verdict'] for s in samples) != Counter({'SUPPORTED': count, 'REFUTED': count}):
            raise ValueError('Balanced class counts changed')
        if sorted(s['provenance']['source_record_index'] for s in samples) != manifest['selected_train_indices'][split]:
            raise ValueError('Frozen selected source indices changed')
        if samples != sorted(samples, key=lambda s: rank(RULES['order_namespace'], s['id'], s['text'])):
            raise ValueError('Frozen sample order changed')
        for s in samples:
            row, p = s['upstream_record'], s['provenance']
            if p['source_record_sha256'] != record_hash(row) or exclusion(p['source_record_index'], row):
                raise ValueError('Source record hash/eligibility mismatch')
            expected = sample(p['source_record_index'], row, split, int(s['id'].rsplit('-', 1)[1]))
            if s != expected:
                raise ValueError('Sample differs from preserved upstream record')
            text, article = normalized_text(s['text']), canonical_article(row['fact_checking_article'])
            if text in texts or article in articles or s['id'] in all_ids or p['source_record_index'] in all_indices:
                raise ValueError('Cross-set claim/article/ID/index overlap')
            texts.add(text)
            articles.add(article)
            all_ids.add(s['id'])
            all_indices.add(p['source_record_index'])
        inputs = json.loads((directory / f'{split}_inputs.json').read_bytes())
        evidence = json.loads((directory / f'{split}_evidence_inputs.json').read_bytes())
        if inputs != [{'id': s['id'], 'text': s['text']} for s in samples]:
            raise ValueError('Label-free input projection mismatch or gold leakage')
        if evidence != [evidence_input(s) for s in samples]:
            raise ValueError('Evidence input projection mismatch or gold leakage')
        stats[split] = dataset['statistics']
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.source:
        artifacts = build(args.source.read_bytes(), (ROOT / 'data/phase3/dataset.json').read_bytes())
        DATA.mkdir(parents=True, exist_ok=True)
        for name, value in artifacts.items():
            raw = encoded(value)
            if args.check:
                if (DATA / name).read_bytes() != raw:
                    raise ValueError(f'Upstream rebuild differs: {name}')
            else:
                (DATA / name).write_bytes(raw)
    elif not args.check:
        parser.error('--source is required to build; --check is offline')
    stats = validate()
    print('PASS improvement data:', json.dumps(stats, sort_keys=True))


if __name__ == '__main__':
    main()
