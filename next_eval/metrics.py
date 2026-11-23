"""Separate outcomes, semantic dimensions and predeclared acceptance gates."""
import math
import random
from collections import Counter

from phase3_metrics import evaluate, compare_paired
from next_eval.common import ARMS, PROTOCOL, ROOT, object_sha, read, require, sha
from next_eval.annotations import semantic_metrics
from next_eval.runner import audit, confidence, selective_policy


def wilson(k, n):
    if not n: return None
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-half), min(1.,center+half)]


def paired_interval(gold, baseline, candidate):
    b = {r['id']:r['verdict'] for r in baseline}
    c = {r['id']:r['verdict'] for r in candidate}
    require(set(b) == set(c) == {g['id'] for g in gold}, 'Paired bootstrap IDs differ')
    groups = [[int(c[g['id']] == label)-int(b[g['id']] == label)
               for g in gold if g['label'] == label] for label in ('SUPPORTED','REFUTED')]
    require(all(groups), 'Stratified bootstrap needs both gold classes')
    rng, values = random.Random(PROTOCOL['seed']), []
    for _ in range(PROTOCOL['bootstrap_repetitions']):
        total = sum(sum(rng.choices(group,k=len(group))) for group in groups)
        values.append(total/len(gold))
    values.sort()
    return {'method': 'class-stratified paired percentile bootstrap', 'seed': 42,
            'repetitions': len(values), 'ci95': [values[int(.025*len(values))], values[math.ceil(.975*len(values))-1]]}


def auroc(observations):
    correct = [s for s,c in observations if c]
    wrong = [s for s,c in observations if not c]
    return (sum((a>b)+.5*(a==b) for a in correct for b in wrong)/(len(correct)*len(wrong))
            if correct and wrong else None)


def predictions(run, arm):
    rows = []
    for row in run['results']:
        r = row['arms'][arm]
        g = r.get('generation') or {}
        rows.append({'id': row['id'], 'verdict': r['verdict'],
                     'mean_label_token_logprob': g.get('mean_label_token_logprob'),
                     'mean_label_token_full_vocab_entropy_nats': g.get('mean_label_token_full_vocab_entropy_nats')})
    return rows


def costs_and_spans(run, arm):
    count = Counter()
    times, peaks = [], []
    for row in run['results']:
        r = row['arms'][arm]
        times.append(r['elapsed_seconds'])
        peaks.append(r['peak_cuda_allocated_bytes'])
        for call in r['calls']:
            g = call['generation']
            count['generation_calls'] += 1
            count['prompt_tokens'] += len(g['prompt_input_ids'])
            count['generated_tokens_including_eos'] += len(g['generated_token_ids'])
            count['truncated_calls'] += not g['ended_with_eos']
        if arm == 'A':
            s = r['candidate_scoring']
            count['candidate_forward_calls'] += 2
            count['prompt_tokens'] += 2*len(s['prompt_token_ids'])
            count['scored_candidate_tokens_including_eos'] += sum(len(c['token_ids']) for c in s['candidates'])
        ex = r.get('extraction')
        if ex is not None:
            count['extraction_samples'] += 1
            count['samples_without_accepted_claim'] += not bool(ex['accepted'])
            count['top_level_extraction_errors'] += len(ex['errors'])
            count['rejected_items'] += len(ex['rejected'])
            count['ambiguous_items'] += sum('ambiguous' in c['reason'] for c in ex['rejected'])
            for claim in ex['accepted']:
                count['accepted_quotes'] += 1
                count['exact_quotes'] += row['source'][claim['start_idx']:claim['end_idx']] == claim['claim']
            count['feedback_transitions'] += sum(s['feedback_used'] for s in r['rounds'])
    times.sort()
    return {**count, 'total_elapsed_seconds': sum(times), 'mean_elapsed_seconds': sum(times)/len(times),
            'p95_elapsed_seconds': times[math.ceil(.95*len(times))-1], 'peak_cuda_allocated_bytes': max(peaks),
            'accepted_span_precision': count['exact_quotes']/count['accepted_quotes'] if count['accepted_quotes'] else None,
            'cost_scope': 'Observed execution excluding model loading and checkpoint I/O; two teacher-forced forwards differ from a generate call.'}


def uncertainties(run, arm, gold, policy):
    labels = {g['id']:g['label'] for g in gold}
    scores, entropies, retained = [], [], []
    for row in run['results']:
        r = row['arms'][arm]
        score = confidence(r, arm)
        correct = r['verdict'] == labels[row['id']]
        if score is not None:
            require(math.isfinite(score), 'Invalid confidence score')
            scores.append((score,correct))
        entropy = (r.get('generation') or {}).get('mean_label_token_full_vocab_entropy_nats')
        if entropy is not None: entropies.append((-entropy,correct))
    if policy is not None:
        for target, threshold in policy.items():
            chosen = [r for r in run['results'] if r['arms'][arm]['verdict'] in ('SUPPORTED','REFUTED')
                      and threshold is not None and confidence(r['arms'][arm],arm) is not None
                      and confidence(r['arms'][arm],arm) >= threshold]
            errors = sum(r['arms'][arm]['verdict'] != labels[r['id']] for r in chosen)
            retained.append({'development_target_coverage': float(target), 'frozen_threshold': threshold,
                             'retained': len(chosen), 'actual_coverage': len(chosen)/len(gold),
                             'errors': errors, 'conditional_error_rate': errors/len(chosen) if chosen else None})
    return {'score_kind': 'absolute_margin_from_frozen_threshold' if arm == 'A' else 'actual_generated_label_mean_logprob',
            'samples_with_scores': len(scores), 'samples_without_scores': len(gold)-len(scores),
            'correctness_auroc': auroc(scores), 'negative_entropy_correctness_auroc': auroc(entropies),
            'selective': retained, 'not_a_truth_probability': True}


def acceptance(metrics, semantics, paired, boundary_complete):
    d, f, gates = metrics['D'], metrics['F'], PROTOCOL['upgrade']
    delta = f['accuracy']-d['accuracy']
    checks = {'accuracy_at_least_80': f['accuracy'] >= gates['accuracy'],
              'gain_at_least_5pp': delta >= gates['accuracy_gain']-1e-12,
              'paired_lower_bound_positive': paired['bootstrap']['ci95'][0] > 0,
              'strict_fnr_at_most_10': f['false_negative_rate'] <= gates['strict_fnr'],
              'missed_at_most_15': f['missed_error_rate_including_abstentions'] <= gates['missed_error_rate'],
              'fpr_at_most_25': f['false_positive_rate'] <= gates['fpr'],
              'missed_risk_increase_at_most_2pp': f['missed_error_rate_including_abstentions']-d['missed_error_rate_including_abstentions'] <= gates['risk_increase']+1e-12,
              'fpr_increase_at_most_2pp': f['false_positive_rate']-d['false_positive_rate'] <= gates['risk_increase']+1e-12,
              'accepted_spans_exact': f['cost_and_extraction']['accepted_span_precision'] == 1.0}
    for key in ('fact_recall','atomicity','faithfulness','qualifier_retention'):
        value = (semantics or {}).get('F',{}).get(key)
        checks[key] = None if value is None else value >= gates[key]
    delivery = semantics is not None and boundary_complete
    return {'evaluation_complete': delivery, 'upgrade_default_method': delivery and all(v is True for v in checks.values()),
            'checks': checks, 'pending': ([] if semantics else ['human_semantic_alignment']) + ([] if boundary_complete else ['audited_boundary_run']),
            'interpretation': 'Research delivery may pass despite no improvement. Gates are sample-level engineering targets, not population risk guarantees.'}


def boundary_summary(run, frozen):
    samples = read(ROOT/frozen['gold_file'])['samples']
    labels = {s['id']:s for s in samples}
    result = {}
    for arm in frozen['arms']:
        confusion = {g:Counter() for g in ('SUPPORTED','REFUTED','NOT_ENOUGH_INFO')}
        families = {}
        for row in run['results']:
            g = labels[row['id']]
            p = row['arms'][arm]['verdict'] or 'INVALID'
            confusion[g['gold']][p] += 1
            stat = families.setdefault(g['family'], {'correct':0,'total':0})
            stat['total'] += 1; stat['correct'] += p == g['gold']
        result[arm] = {'confusion_matrix': {g:dict(c) for g,c in confusion.items()}, 'families':families,
                       'correct':sum(c[g] for g,c in confusion.items()), 'total':len(samples),
                       'insufficient_evidence_correct':confusion['NOT_ENOUGH_INFO']['NOT_ENOUGH_INFO'],
                       'insufficient_evidence_total':6, 'cost_and_extraction':costs_and_spans(run,arm)}
    return result


def summarize(run, frozen, freeze_path, alignment=None, boundary_run=None):
    audit_result = audit(run, frozen, freeze_path)
    require(run['completed'], 'Cannot summarize partial evaluation')
    output = {'stage':run['stage'], 'run_sha256':object_sha(run), 'freeze_sha256':sha(freeze_path), 'audit':audit_result,
              'primary_comparison':'F minus D', 'gold_sha256':frozen['gold_sha256']}
    if run['stage'] == 'boundary':
        output['boundary'] = boundary_summary(run, frozen)
        output['acceptance'] = {'evaluation_complete':False, 'upgrade_default_method':False, 'reason':'Boundary fixtures are not news test results.'}
        return output
    samples = read(ROOT/frozen['gold_file'])['samples']
    selected = set(frozen['ids'])
    gold = [{'id':s['id'],'label':s['gold']['verdict']} for s in samples if s['id'] in selected]
    policies = frozen['selective_thresholds']
    if run['stage'] == 'dev' and len(run['results']) == 32:
        policies = selective_policy(run)
    output['arms'] = {}
    for arm in ARMS:
        m = evaluate(gold, predictions(run,arm))
        # The historical evaluator's fixed score thresholds are replaced by this
        # round's development-frozen policy; raw score summaries remain unchanged.
        del m['selective_results']
        m['definitions']['selective_results'] = 'Development-only quantiles frozen before test; actual coverage may differ.'
        m['ci95'] = {'accuracy':wilson(m['correct_samples'],len(gold)),
                     'strict_fnr':wilson(m['false_negatives_supported'],m['gold_counts']['REFUTED']),
                     'missed_error_rate':wilson(m['missed_errors_including_abstentions'],m['gold_counts']['REFUTED']),
                     'fpr':wilson(m['false_positives'],m['gold_counts']['SUPPORTED'])}
        m['cost_and_extraction'] = costs_and_spans(run,arm)
        m['uncertainty'] = uncertainties(run,arm,gold,policies[arm] if policies else None)
        output['arms'][arm] = m
    output['paired'] = {}
    for baseline,candidate in (('D','F'),('B','C'),('C','D'),('D','E'),('E','F')):
        paired = compare_paired(gold,predictions(run,baseline),predictions(run,candidate))
        if baseline == 'D' and candidate == 'F':
            paired['bootstrap'] = paired_interval(gold,predictions(run,baseline),predictions(run,candidate))
        output['paired'][f'{candidate}_minus_{baseline}'] = paired
    output['semantics'], output['boundary'] = None, None
    labels = {g['id']:g['label'] for g in gold}
    human_notes = {(r['id'],r['arm']):r for r in (alignment or {}).get('items',[])}
    output['case_ledger'] = []
    for row in run['results']:
        record = {'id':row['id'],'source':row['source'],'whole_sample_gold':labels[row['id']],'arms':{}}
        for arm,result in row['arms'].items():
            g = result.get('generation') or {}
            ex = result.get('extraction') or {}
            note = human_notes.get((row['id'],arm),{})
            record['arms'][arm] = {'verdict':result['verdict'],'correct':result['verdict']==labels[row['id']],
                'accepted_claim_count':len(ex.get('accepted',[])), 'rejected_item_count':len(ex.get('rejected',[])),
                'extraction_errors':ex.get('errors',[]), 'mean_label_token_logprob':g.get('mean_label_token_logprob'),
                'candidate_margin':result.get('candidate_scoring',{}).get('margin'),
                'content_error_types':note.get('content_error_types',[]),
                'system_failure_types':note.get('system_failure_types',[]),
                'cause_status':note.get('cause_status','undetermined')}
        output['case_ledger'].append(record)
    if run['stage'] == 'news' and alignment is not None:
        output['semantics'] = semantic_metrics(run,read(ROOT/frozen['bundle']/'adjudication.json'),alignment)
    if boundary_run is not None:
        bf = ROOT/boundary_run['freeze_file']
        bfrozen = read(bf)
        require(boundary_run['stage'] == 'boundary' and bfrozen['bundle'] == frozen['bundle'], 'Boundary run must belong to the same bundle')
        audit(boundary_run,bfrozen,bf)
        output['boundary'] = boundary_summary(boundary_run,bfrozen)
        output['boundary_run_sha256'] = object_sha(boundary_run)
    output['acceptance'] = (acceptance(output['arms'],output['semantics'],output['paired']['F_minus_D'],output['boundary'] is not None)
                            if run['stage'] == 'news' else
                            {'evaluation_complete':False,'upgrade_default_method':False,'reason':'Development results cannot satisfy prospective test acceptance.'})
    return output
