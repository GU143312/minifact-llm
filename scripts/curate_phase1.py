"""Reproduce the frozen Days 1-2 dataset from a hash-pinned upstream file.

Only the Python standard library is needed. No model, network, or GPU is used.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/phase1'


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def record_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def canonical_article(url):
    match = re.search(r'/web/[^/]+/(https?://.*)', url)
    return (match.group(1) if match else url).split('#', 1)[0].rstrip('/')


def normalized_text(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()


def rejection(index, row, config):
    rules = config['selection']
    if row['label'] != config['source']['required_label']:
        return 'not_refuted'
    if str(index) in config['manual_exclusions']:
        return 'manual_exclusion'
    if any('image' in t.lower() or 'video' in t.lower() for t in row['claim_types']):
        return 'visual_type'
    text = row['claim']
    if re.search(r'\b(photo|photograph|image|picture|video|footage|meme)\b', text, re.I):
        return 'visual_text'
    if not rules['min_words'] <= len(text.split()) <= rules['max_words']:
        return 'length'
    if re.match(r'^(This|That|These|Those|It|They|He|She|You|I|We)\b', text, re.I):
        return 'unresolved_opening'
    if not row.get('justification', '').strip() or not row.get('fact_checking_article'):
        return 'missing_gold'
    answers = [a for q in row['questions'] for a in q['answers']
               if a.get('answer_type') != 'Unanswerable']
    if any(a.get('source_medium', '').lower() not in
           ['web text', 'pdf', 'web table', 'metadata'] for a in answers):
        return 'non_text_evidence'
    usable = [a for q in row['questions'] if q.get('question', '').strip()
              for a in q['answers'] if a.get('answer_type') != 'Unanswerable'
              and a.get('answer', '').strip()
              and a.get('source_url', '').startswith(('https://', 'http://'))
              and a.get('source_medium', '').lower() in ['web text', 'pdf']]
    return None if usable else 'no_text_evidence'


def select(rows, config):
    eligible = [i for i, row in enumerate(rows) if rejection(i, row, config) is None]
    rules = config['selection']
    def rank(index):
        text = f'{rules["rank_namespace"]}|{rules["seed"]}|{index}|{rows[index]["claim"]}'
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    selected, articles, texts = [], set(), set()
    domains, speakers = Counter(), Counter()
    for quota in rules['quotas_in_selection_order']:
        kind, count = quota['primary_claim_type'], quota['count']
        pool = sorted((i for i in eligible if rows[i]['claim_types'][0] == kind), key=rank)
        added = 0
        for index in pool:
            row = rows[index]
            article = canonical_article(row['fact_checking_article'])
            domain = urlsplit(article).hostname
            speaker = (row.get('speaker') or '').strip().casefold()
            text = normalized_text(row['claim'])
            if (article in articles or domains[domain] >= rules['max_per_factchecker_domain']
                    or (speaker and speakers[speaker] >= rules['max_per_named_speaker'])
                    or text in texts):
                continue
            selected.append(index)
            articles.add(article)
            texts.add(text)
            domains[domain] += 1
            if speaker:
                speakers[speaker] += 1
            added += 1
            if added == count:
                break
        if added != count:
            raise ValueError(f'Candidate quota not met: {kind}: {added}/{count}')
    if len(selected) != config['candidate_count']:
        raise ValueError('Candidate count mismatch')
    final = sorted(i for i in selected if str(i) not in config['final_review_exclusions'])
    if final != config['final_source_indices'] or len(final) != config['sample_count']:
        raise ValueError('Frozen final selection mismatch')
    audit = {'source_records': len(rows), 'source_labels': dict(sorted(Counter(
        r['label'] for r in rows).items())), 'eligible_before_candidate_selection': len(eligible),
        'filter_rejection_counts': dict(sorted(Counter(rejection(i, r, config)
        for i, r in enumerate(rows) if rejection(i, r, config)).items())),
        'candidate_count': len(selected), 'candidate_source_indices': sorted(selected),
        'final_review_excluded_count': len(selected) - len(final)}
    return final, audit


def sample_from_row(index, row, sequence):
    return {
        'id': f'MF-P1-{sequence:04d}', 'text': row['claim'], 'language': 'en',
        'claim_date': datetime.strptime(row['claim_date'], '%d-%m-%Y').date().isoformat(),
        'context': {k: row[k] for k in ['speaker', 'reporting_source', 'location_ISO_code']},
        'gold': {'verdict': 'REFUTED', 'original_label': row['label'],
                 'verdict_scope': 'whole_sample', 'justification': row['justification'],
                 'questions': row['questions']},
        'annotations': {'status': 'pending_days_3_4', 'atomic_claims': None,
                        'qualifiers': None, 'error_types': None},
        'provenance': {'source_record_index': index, 'source_record_sha256': record_hash(row),
                       **{k: row[k] for k in ['claim_date', 'original_claim_url',
                           'cached_original_claim_url', 'fact_checking_article',
                           'claim_types', 'fact_checking_strategies', 'required_reannotation'] if k in row}}
    }


def restore_upstream(sample):
    """Recover every original field for canonical record-hash verification."""
    p, gold = sample['provenance'], sample['gold']
    return {'claim': sample['text'], 'label': gold['original_label'],
            'justification': gold['justification'], 'questions': gold['questions'],
            **sample['context'], **{k: v for k, v in p.items()
            if k not in ['source_record_index', 'source_record_sha256']}}


def input_rows(dataset):
    # Only text and IDs: no verdicts, questions, evidence, article titles, or source IDs.
    return [{'id': s['id'], 'text': s['text']} for s in dataset['samples']]


def statistics(samples):
    counts = Counter(s['provenance']['claim_types'][0] for s in samples)
    domains = Counter(urlsplit(canonical_article(s['provenance']['fact_checking_article'])).hostname
                     for s in samples)
    words = [len(s['text'].split()) for s in samples]
    return {'sample_count': len(samples), 'verdicts': {'REFUTED': len(samples)},
            'primary_upstream_claim_types': dict(sorted(counts.items())),
            'factchecker_domains': dict(sorted(domains.items())),
            'factchecker_domain_count': len(domains),
            'locations': dict(sorted(Counter(s['context']['location_ISO_code'] or 'unknown'
                                             for s in samples).items())),
            'question_count': sum(len(s['gold']['questions']) for s in samples),
            'answer_count': sum(len(q['answers']) for s in samples for q in s['gold']['questions']),
            'word_count': {'method': 'Python str.split', 'min': min(words), 'max': max(words),
                           'mean': round(sum(words) / len(words), 2)},
            'claim_date_min': min(s['claim_date'] for s in samples),
            'claim_date_max': max(s['claim_date'] for s in samples)}


def build(source, config):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['source']['sha256']:
        raise ValueError('Source SHA256 mismatch; refusing a changed upstream version')
    rows = json.loads(raw)
    indices, audit = select(rows, config)
    samples = [sample_from_row(i, rows[i], n) for n, i in enumerate(indices, 1)]
    dataset = {
        '$schema': './dataset.schema.json', 'schema_version': '1.0.0',
        'dataset_id': config['dataset_id'], 'version': config['version'],
        'curation_date': config['curation_date'], 'language': 'en',
        'sample_unit': 'real_world_news_related_claim', 'project_split': 'test',
        'license': {'id': 'CC-BY-NC-4.0', 'url': 'https://creativecommons.org/licenses/by-nc/4.0/',
                    'attribution': 'Michael Sejr Schlichtkrull, Zhijiang Guo, and Andreas Vlachos (2023). AVeriTeC.',
                    'citation_url': 'https://openreview.net/forum?id=fKzSz0oyaI',
                    'changes': 'Selected and filtered 60 dev records; added IDs, normalized dates, provenance hashes, and a standardized envelope. Claim, justification, QA, and original metadata are preserved verbatim.'},
        'source': config['source'],
        'review': {'method': 'Agent screening of supplied claim and justification; targeted QA inspection; deterministic structural checks.',
                   'independent_human_reverification': False,
                   'live_source_retrieval_performed': False,
                   'gold_origin': 'Inherited upstream benchmark annotations at the original claim date; not current truth labels.',
                   'atomic_gold_status': 'Not annotated in Days 1-2; whole-sample verdict must not be assigned to every future atomic claim.'},
        'statistics': statistics(samples), 'selection_audit': audit, 'samples': samples}
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Local hash-pinned AVeriTeC dev.json')
    parser.add_argument('--check', action='store_true', help='Check generated files without writing')
    args = parser.parse_args()
    config = json.loads((DATA / 'curation.json').read_text(encoding='utf-8'))
    dataset = build(args.source, config)
    files = {'dataset.json': encoded(dataset), 'inputs.json': encoded(input_rows(dataset))}
    for name, content in files.items():
        target = DATA / name
        if args.check:
            if not target.exists() or target.read_bytes() != content:
                raise SystemExit(f'Stale generated dataset: {target}')
        else:
            target.write_bytes(content)
    print(f'{"Checked" if args.check else "Built"} {len(dataset["samples"])} frozen samples; source hash verified.')


if __name__ == '__main__':
    main()
