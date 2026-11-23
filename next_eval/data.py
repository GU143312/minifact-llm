"""Deterministic prospective candidates, blinded review packets and boundaries."""
import hashlib
from collections import Counter
from pathlib import Path

from next_eval.common import ROOT, PROTOCOL, encoded, object_sha, read, require, sha, write_new
from scripts.build_improvement_data import (SOURCE, LICENSE, RULES, exclusion,
    evidence_projection, canonical_article, normalized_text, sample, evidence_input)

HISTORY = ('data/phase3/dataset.json', 'data/improvement/dev_dataset.json',
           'data/improvement/holdout_dataset.json')


def rank(namespace, value):
    return hashlib.sha256(f'{namespace}|42|{value}'.encode()).hexdigest()


def history():
    records = []
    for name in HISTORY:
        for s in read(ROOT/name)['samples']:
            article = s.get('upstream_record', {}).get('fact_checking_article') or s['provenance'].get('fact_checking_article')
            records.append({'id': s['id'], 'text': s['text'], 'article': article})
    return records


def build(source, excluded=()):
    require(sha(source) == SOURCE['sha256'], 'Wrong pinned upstream bytes')
    rows = read(source)
    old = history()
    texts = {normalized_text(s['text']) for s in old}
    articles = {canonical_article(s['article']) for s in old}
    selected, counts = [], Counter()
    excluded = set(excluded)
    for i in sorted(range(len(rows)), key=lambda i: rank('minifact-next-news-v1', i)):
        r = rows[i]
        if i in excluded or exclusion(i, r) is not None or counts[r['label']] >= 100:
            continue
        text, article = normalized_text(r['claim']), canonical_article(r['fact_checking_article'])
        if text in texts or article in articles:
            continue
        item = sample(i, r, 'holdout', 1)
        item['id'] = f'MFN-{i:04d}'  # stable upstream index; no class cue
        selected.append(item)
        counts[r['label']] += 1
        texts.add(text)
        articles.add(article)
    require(counts == {'Supported': 100, 'Refuted': 100}, f'Insufficient disjoint candidates: {counts}')
    selected.sort(key=lambda s: rank('minifact-next-order-v1', s['id']))
    semantic_ids = []
    for label in ('SUPPORTED', 'REFUTED'):
        group = [s for s in selected if s['gold']['verdict'] == label]
        semantic_ids.extend(s['id'] for s in sorted(group, key=lambda s: rank('minifact-next-semantic-v1', s['id']))[:30])
    semantic_ids.sort(key=lambda s: rank('minifact-next-semantic-order-v1', s))
    dataset = {'schema_version': 1, 'status': 'candidate_pending_event_review_and_human_annotation',
               'source': SOURCE, 'license': LICENSE, 'samples': selected,
               'gold_scope': 'Inherited whole-sample benchmark labels, not independent truth or atomic labels.',
               'semantic_ids': semantic_ids,
               'history_sha256': {p: sha(ROOT/p) for p in HISTORY},
               'excluded_source_indices': sorted(excluded)}
    return dataset, [evidence_input(s) for s in selected], old


def spans(text, quote):
    starts, offset = [], 0
    while (index := text.find(quote, offset)) != -1:
        starts.append({'start_idx': index, 'end_idx': index + len(quote)})
        offset = index + 1
    return starts


def boundaries():
    """Fictional closed-world cases: four determinate and one insufficient per family."""
    cases = []
    for family in ('repeat', 'shared_subject', 'pronoun', 'negation', 'time', 'quantity_causality'):
        for j in range(5):
            name, year, n = f'Luma-{j+1}', 2018+j, 20+j
            if family == 'repeat':
                q = f'{name} Hall has {n} seats'
                text, facts = f'{q}. {q}.', [(q, q, [], [('quantity', str(n))])]
                evidence = f'{name} Hall has {n if j % 2 == 0 else n+5} seats.'
            elif family == 'shared_subject':
                q1, q2 = f'{name} Council opened a library', 'and closed a depot'
                text = f'{q1} {q2}.'
                facts = [(q1, q1, [], []), (q2, f'{name} Council closed a depot', [f'{name} Council'], [])]
                evidence = f'{name} Council opened a library and {"closed" if j % 2 == 0 else "kept open"} a depot.'
            elif family == 'pronoun':
                q1, q2 = f'Mira joined {name} Clinic', 'She managed its pharmacy'
                text = f'{q1}. {q2}.'
                facts = [(q1, q1, [], []), (q2, f'Mira managed the pharmacy of {name} Clinic', [q1], [])]
                evidence = f'Mira joined {name} Clinic. {"Mira" if j % 2 == 0 else "Only Leon"} managed its pharmacy.'
            elif family == 'negation':
                q = f'{name} School did not close in June {year}'
                text, facts = q+'.', [(q, q, [], [('negation', 'not'), ('time', f'June {year}')])]
                evidence = f'{name} School {"remained open throughout" if j % 2 == 0 else "closed during"} June {year}.'
            elif family == 'time':
                q = f'{name} Bridge was closed from 3 May {year} through 7 May {year}, inclusive'
                text, facts = q+'.', [(q, q, [], [('time', f'from 3 May {year} through 7 May {year}, inclusive')])]
                evidence = f'{name} Bridge closed on 3 May {year} and reopened on {8 if j % 2 == 0 else 6} May {year}.'
            else:
                q = f'A pump failure caused exactly {n} homes in {name} to lose water'
                text, facts = q+'.', [(q, q, [], [('quantity', f'exactly {n}'), ('causality', 'A pump failure caused')])]
                evidence = (f'A pump failure caused exactly {n} homes in {name} to lose water.' if j % 2 == 0 else
                            f'A planned shutdown, not a pump failure, caused exactly {n} homes in {name} to lose water.')
            if j == 4:
                evidence = f'{name} is listed in this fictional municipality register. No further event details are recorded.'
            atoms = []
            for k, (quote, proposition, contexts, qualifiers) in enumerate(facts):
                atoms.append({'fact_id': f'f{k+1}', 'proposition': proposition, 'quote': quote,
                              'occurrences': spans(text, quote),
                              'context_quotes': [{'quote': c, 'occurrences': spans(text, c)} for c in contexts],
                              'qualifiers': [{'qualifier_id': f'q{x+1}', 'kind': kind, 'text': value}
                                             for x, (kind, value) in enumerate(qualifiers)]})
            cases.append({'id': f'MNB-{family}-{j+1}', 'text': text,
                          'context': {'claim_date': f'{year}-12-31', 'speaker': None, 'location_ISO_code': None},
                          'evidence': [{'question': 'What does the fictional record establish?', 'answer': evidence,
                                        'url': 'https://example.org/minifact-fictional-record'}],
                          'family': family, 'gold': 'NOT_ENOUGH_INFO' if j == 4 else 'SUPPORTED' if j % 2 == 0 else 'REFUTED',
                          'facts': atoms})
    return {'schema_version': 1, 'status': 'authored_fixture_expectations_not_human_news_gold',
            'interpretation': 'Fictional, packet-relative judgments. Evaluate D/E/F separately from news.', 'samples': cases}


def prepare(source, output, excluded=()):
    output = Path(output)
    require(not output.exists(), 'Output directory exists; use a new version, never overwrite reviews')
    dataset, inputs, old = build(source, excluded)
    boundary = boundaries()
    blind = {s['id']: s for s in inputs}
    review_items = [{'id': i, 'text': blind[i]['text'], 'status': 'pending', 'facts': []}
                    for i in dataset['semantic_ids']]
    files = {'dataset.json': dataset, 'inputs.json': inputs, 'protocol.json': PROTOCOL,
             'boundary.json': boundary,
             'boundary_inputs.json': [{k: s[k] for k in ('id', 'text', 'context', 'evidence')} for s in boundary['samples']],
             'event_review.json': {'reviewer': None, 'reviewer_type': None, 'completed_at_utc': None,
                                  'historical_records': old,
                                  'items': [{'id': s['id'], 'text': s['text'], 'decision': 'pending',
                                             'event_key': None, 'overlaps_history': None, 'reason': ''} for s in inputs]},
             'reviewer_a.json': {'reviewer_id': None, 'reviewer_type': 'human', 'completed_at_utc': None, 'items': review_items},
             'reviewer_b.json': {'reviewer_id': None, 'reviewer_type': 'human', 'completed_at_utc': None, 'items': review_items},
             'adjudication.json': {'reviewer_ids': [], 'completed_at_utc': None, 'status': 'pending',
                                  'reviewer_a_sha256': None, 'reviewer_b_sha256': None, 'items': review_items}}
    for name, value in files.items():
        write_new(output/name, value)
    write_new(output/'manifest.json', {'source': SOURCE, 'selection': RULES,
        'protocol_sha256': object_sha(PROTOCOL), 'history_sha256': dataset['history_sha256'],
        'artifacts_sha256': {n: sha(output/n) for n in ('dataset.json', 'inputs.json', 'protocol.json', 'boundary.json', 'boundary_inputs.json')},
        'review_files_are_editable': True, 'formal_run_ready': False,
        'selection_rule': 'Seed 42 SHA ranking, 100/class, existing text/article exclusions. Event review pending; replace excluded cases only in a new bundle before annotation.'})
    validate_bundle(output)
    return {'news': 200, 'semantic_review': 60, 'boundary': 30, 'formal_run_ready': False}


def validate_bundle(directory):
    from next_eval.common import validate_inputs
    import jsonschema
    directory = Path(directory)
    manifest = read(directory/'manifest.json')
    require(manifest['source'] == SOURCE and manifest['selection'] == RULES, 'Source/filter contract changed')
    for name, digest in manifest['artifacts_sha256'].items():
        require(name in ('dataset.json', 'inputs.json', 'protocol.json', 'boundary.json', 'boundary_inputs.json'), 'Unknown artifact')
        require(sha(directory/name) == digest, f'Changed artifact: {name}')
    require(set(manifest['artifacts_sha256']) == {'dataset.json', 'inputs.json', 'protocol.json', 'boundary.json', 'boundary_inputs.json'}, 'Missing artifact hashes')
    require(read(directory/'protocol.json') == PROTOCOL and manifest['protocol_sha256'] == object_sha(PROTOCOL), 'Protocol differs')
    require(manifest['history_sha256'] == {p: sha(ROOT/p) for p in HISTORY}, 'Historical data changed')
    dataset, inputs = read(directory/'dataset.json'), read(directory/'inputs.json')
    jsonschema.validate(dataset, read(ROOT/'docs/next_evaluation.schema.json'))
    require(Counter(s['gold']['verdict'] for s in dataset['samples']) == {'SUPPORTED': 100, 'REFUTED': 100}, 'News class counts differ')
    require(inputs == [evidence_input(s) for s in dataset['samples']], 'Input projection differs')
    validate_inputs(inputs)
    old = history()
    seen_t = {normalized_text(r['text']) for r in old}
    seen_a = {canonical_article(r['article']) for r in old}
    for s in dataset['samples']:
        r = s['upstream_record']
        i = s['provenance']['source_record_index']
        require(exclusion(i, r) is None, 'Ineligible row')
        expected = sample(i, r, 'holdout', 1)
        expected['id'] = f'MFN-{i:04d}'
        require(s == expected, 'Sample/provenance differs')
        t, a = normalized_text(s['text']), canonical_article(r['fact_checking_article'])
        require(t not in seen_t and a not in seen_a, 'Historical or within-test duplicate')
        seen_t.add(t); seen_a.add(a)
    ids = dataset['semantic_ids']
    require(len(ids) == len(set(ids)) == 60, 'Expected 60 unique semantic IDs')
    require(Counter(s['gold']['verdict'] for s in dataset['samples'] if s['id'] in ids) == {'SUPPORTED': 30, 'REFUTED': 30}, 'Semantic class balance differs')
    require(read(directory/'boundary.json') == boundaries(), 'Boundary fixtures changed')
    require(read(directory/'boundary_inputs.json') == [{k:s[k] for k in ('id','text','context','evidence')} for s in boundaries()['samples']], 'Boundary projection differs')
    validate_inputs(read(directory/'boundary_inputs.json'))
    return dataset, inputs
