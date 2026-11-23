"""Human attestations and explicit semantic alignment; never auto-invent gold."""
from collections import Counter
from datetime import datetime

from next_eval.common import object_sha, read, require, sha
from next_eval.data import history, spans, validate_bundle


def timestamp(value):
    require(isinstance(value, str), 'Missing completion timestamp')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Completion timestamp needs a timezone')
    return result


def person(value):
    require(isinstance(value, str) and value.strip() and value.casefold() not in ('pending', 'todo', 'agent', 'model'), 'Use an actual reviewer pseudonym')
    return value.strip()


def indexed(items, expected):
    require(isinstance(items, list) and all(isinstance(s, dict) for s in items), 'Expected review items')
    require(len(items) == len(expected) and {s.get('id') for s in items} == set(expected), 'Review IDs missing/extra/duplicated')
    return {s['id']: s for s in items}


def check_facts(item, source):
    require(item.get('text') == source and item.get('status') == 'complete', 'Incomplete review or altered source')
    facts = item.get('facts')
    require(isinstance(facts, list) and bool(facts), 'Each reviewed news sample needs nonempty facts')
    fact_ids = []
    for fact in facts:
        require(isinstance(fact, dict), 'Invalid fact')
        fid = fact.get('fact_id')
        require(isinstance(fid, str) and fid.strip() and fid not in fact_ids, 'Duplicate/empty fact ID')
        fact_ids.append(fid)
        require(isinstance(fact.get('proposition'), str) and fact['proposition'].strip(), 'Missing complete proposition')
        quote = fact.get('quote')
        require(isinstance(quote, str) and quote and fact.get('occurrences') == spans(source, quote) and bool(fact['occurrences']), 'Reference occurrences must enumerate all exact Unicode slices')
        require(isinstance(fact.get('context_quotes'), list), 'Missing context_quotes array')
        for context in fact['context_quotes']:
            q = context.get('quote')
            require(isinstance(q, str) and q and context.get('occurrences') == spans(source, q) and bool(context['occurrences']), 'Invalid reference context spans')
        qualifiers = fact.get('qualifiers')
        require(isinstance(qualifiers, list), 'Missing qualifiers array')
        seen = set()
        for q in qualifiers:
            require(isinstance(q, dict) and isinstance(q.get('qualifier_id'), str) and q['qualifier_id'] and q['qualifier_id'] not in seen, 'Invalid qualifier ID')
            require(q.get('kind') in ('quantity', 'time', 'negation', 'condition', 'attribution', 'scope', 'causality', 'location'), 'Unknown qualifier kind')
            require(isinstance(q.get('text'), str) and bool(q['text'].strip()), 'Empty qualifier')
            seen.add(q['qualifier_id'])
    return {f['fact_id']: f for f in facts}


def readiness(directory):
    """Fail closed. Human identity is an attestation, not independently authenticated."""
    dataset, inputs = validate_bundle(directory)
    texts = {s['id']: s['text'] for s in inputs}
    events = read(directory/'event_review.json')
    person(events.get('reviewer'))
    require(events.get('reviewer_type') in ('human', 'agent_assisted_manual_review'), 'Event review needs a declared reviewer type')
    event_time = timestamp(events.get('completed_at_utc'))
    require(events.get('historical_records') == history(), 'Historical event review reference changed')
    event_items = indexed(events.get('items'), texts)
    keys = set()
    for sid, item in event_items.items():
        require(item.get('text') == texts[sid] and item.get('decision') == 'include' and item.get('overlaps_history') is False, 'Pending/excluded/overlapping event: rebuild a new bundle before annotation')
        key = item.get('event_key')
        require(isinstance(key, str) and key.strip() and key.strip().casefold() not in keys, 'Missing or duplicate event cluster')
        require(isinstance(item.get('reason'), str) and item['reason'].strip(), 'Event review needs a rationale')
        keys.add(key.strip().casefold())
    ids = dataset['semantic_ids']
    reviewers, mapped = [], []
    for file in ('reviewer_a.json', 'reviewer_b.json'):
        review = read(directory/file)
        reviewers.append(person(review.get('reviewer_id')))
        require(review.get('reviewer_type') == 'human', 'Formal semantic gold requires human reviewers')
        require(timestamp(review.get('completed_at_utc')) >= event_time, 'Event review must precede final semantic annotation')
        rows = indexed(review.get('items'), ids)
        mapped.append({sid: check_facts(item, texts[sid]) for sid, item in rows.items()})
    require(reviewers[0] != reviewers[1], 'Two distinct human reviewers are required')
    final = read(directory/'adjudication.json')
    require(final.get('status') == 'complete' and final.get('reviewer_ids') == reviewers, 'Incomplete joint adjudication')
    done = timestamp(final.get('completed_at_utc'))
    for letter, file in (('a', 'reviewer_a.json'), ('b', 'reviewer_b.json')):
        require(final.get(f'reviewer_{letter}_sha256') == sha(directory/file), 'Adjudication must bind exact independent reviews')
        require(done >= timestamp(read(directory/file)['completed_at_utc']), 'Adjudication precedes independent review')
    rows = indexed(final.get('items'), ids)
    agreements = Counter()
    for sid, item in rows.items():
        gold = check_facts(item, texts[sid])
        require(isinstance(item.get('resolution'), str) and item['resolution'].strip(), 'Document how differences were resolved')
        seen_a, seen_b = set(), set()
        for fact in gold.values():
            for name, seen, reference in (('reviewer_a_fact_ids', seen_a, mapped[0][sid]), ('reviewer_b_fact_ids', seen_b, mapped[1][sid])):
                links = fact.get(name)
                require(isinstance(links, list) and len(links) == len(set(links)) and set(links) <= set(reference) and not (set(links) & seen), 'Invalid adjudicated reviewer alignment')
                seen.update(links)
            a, b = fact['reviewer_a_fact_ids'], fact['reviewer_b_fact_ids']
            agreements['agreed_reference_facts'] += bool(a and b)
            agreements['adjudicated_facts'] += 1
        # Rejected reviewer proposals are explicitly listed, never silently lost.
        for name, seen, reference in (('rejected_reviewer_a_fact_ids', seen_a, mapped[0][sid]), ('rejected_reviewer_b_fact_ids', seen_b, mapped[1][sid])):
            rejected = item.get(name)
            require(isinstance(rejected, list) and len(rejected) == len(set(rejected)) and not (set(rejected)&seen) and set(rejected)|seen == set(reference), 'Every reviewer fact needs a resolution')
        agreements['same_fact_count_samples'] += len(mapped[0][sid]) == len(mapped[1][sid])
        agreements['samples'] += 1
    agreements['reviewer_a_facts'] = sum(map(len, mapped[0].values()))
    agreements['reviewer_b_facts'] = sum(map(len, mapped[1].values()))
    return {'ready': True, 'reviewer_attestation_only': True, 'agreement': dict(agreements),
            'agreement_definition': 'Adjudicator-aligned proposal overlap and count agreement; not independent kappa or exact-string semantic agreement.',
            'review_files_sha256': {f: sha(directory/f) for f in ('event_review.json', 'reviewer_a.json', 'reviewer_b.json', 'adjudication.json')},
            'completed_at_utc': final['completed_at_utc']}


def semantic_template(run, adjudication):
    ids = {s['id'] for s in adjudication['items']}
    items = []
    for row in run['results']:
        if row['id'] not in ids:
            continue
        for arm in ('E', 'F'):
            result = row['arms'][arm]
            extraction = result.get('extraction') or {'raw_items': None, 'accepted': []}
            raw = extraction.get('raw_items')
            predictions = []
            for i, value in enumerate(raw if isinstance(raw, list) else []):
                predictions.append({'prediction_index': i, 'raw_item': value,
                                    'accepted': any(c['quote_index'] == i for c in extraction['accepted']),
                                    'fact_ids': [], 'atomic': None, 'faithful': None,
                                    'retained_qualifiers': [], 'notes': ''})
            items.append({'id': row['id'], 'arm': arm, 'source': row['source'],
                          'status': 'pending', 'predictions': predictions,
                          'content_error_types': [], 'system_failure_types': [], 'cause_status': 'undetermined'})
    return {'run_sha256': object_sha(run), 'gold_sha256': object_sha(adjudication),
            'reviewer_ids': [], 'completed_at_utc': None, 'status': 'pending', 'items': items}


def semantic_metrics(run, adjudication, alignment):
    template = semantic_template(run, adjudication)
    require(alignment.get('run_sha256') == template['run_sha256'] and alignment.get('gold_sha256') == template['gold_sha256'], 'Semantic alignment is for a different run/gold')
    reviewers = alignment.get('reviewer_ids')
    require(isinstance(reviewers, list) and len(reviewers) == len(set(reviewers)) == 2, 'Two human semantic alignment reviewers required')
    for r in reviewers: person(r)
    require(alignment.get('status') == 'complete', 'Semantic alignment pending')
    require(timestamp(alignment.get('completed_at_utc')) >= timestamp(run['completed_at_utc']), 'Alignment predates model outputs')
    expected = {(r['id'], r['arm']): r for r in template['items']}
    observed = alignment.get('items', [])
    require(len(observed) == len(expected) and {(r['id'], r['arm']) for r in observed} == set(expected), 'Semantic alignment row mismatch')
    gold = {r['id']: {f['fact_id']: f for f in r['facts']} for r in adjudication['items']}
    totals = {a: Counter() for a in ('E', 'F')}
    for item in observed:
        key = item['id'], item['arm']
        t, facts = totals[item['arm']], gold[item['id']]
        require(item.get('source') == expected[key]['source'] and item.get('status') == 'complete', 'Semantic item incomplete')
        pred = item['predictions']
        require(len(pred) == len(expected[key]['predictions']), 'Prediction omitted from human alignment')
        covered, retained = set(), set()
        for p, ref in zip(pred, expected[key]['predictions']):
            require(all(p.get(k) == ref[k] for k in ('prediction_index', 'raw_item', 'accepted')), 'Prediction changed in human alignment')
            require(type(p.get('atomic')) is bool and type(p.get('faithful')) is bool, 'Unreviewed semantic prediction')
            require(isinstance(p.get('fact_ids'), list) and len(p['fact_ids']) == len(set(p['fact_ids'])) and set(p['fact_ids']) <= set(facts), 'Unknown/duplicate matched fact')
            require(isinstance(p.get('retained_qualifiers'), list), 'Missing qualifier matches')
            for pair in p['retained_qualifiers']:
                require(isinstance(pair, list) and len(pair) == 2 and pair[0] in p['fact_ids'] and pair[1] in {q['qualifier_id'] for q in facts[pair[0]]['qualifiers']}, 'Invalid qualifier match')
                if p['accepted']: retained.add(tuple(pair))
            t['raw_predictions'] += 1
            t['atomic_predictions'] += p['atomic']
            t['faithful_predictions'] += p['faithful']
            if p['accepted']: covered.update(p['fact_ids'])
        t['gold_facts'] += len(facts)
        t['covered_facts'] += len(covered)
        t['gold_qualifiers'] += sum(len(f['qualifiers']) for f in facts.values())
        t['matched_fact_qualifiers'] += sum(len(facts[f]['qualifiers']) for f in covered)
        t['retained_qualifiers'] += len(retained)
        t['samples'] += 1
    result = {}
    for arm, count in totals.items():
        def ratio(n, d): return count[n]/count[d] if count[d] else None
        result[arm] = {**count, 'fact_recall': ratio('covered_facts', 'gold_facts'),
                       'atomicity': ratio('atomic_predictions', 'raw_predictions'),
                       'faithfulness': ratio('faithful_predictions', 'raw_predictions'),
                       'qualifier_retention': ratio('retained_qualifiers', 'matched_fact_qualifiers'),
                       'end_to_end_qualifier_retention': ratio('retained_qualifiers', 'gold_qualifiers')}
    return result
