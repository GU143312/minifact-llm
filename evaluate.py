"""Keep fact coverage, qualifier retention and span availability separate."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

from minifact import ROOT, exact_span, occurrences, read_json, write_json


def evaluate_case(case, row):
    assert row['source'] == case['source'], 'Source mismatch'
    raw = row.get('raw_quotes')
    quotes = raw if isinstance(raw, list) else []
    candidates = [(i,q) for i,q in enumerate(quotes) if isinstance(q,str) and q]
    literal = [(i,q) for i,q in candidates if occurrences(case['source'],q)]
    facts=[]
    for fact in case['facts']:
        supporting = [i for i,q in candidates if all(core in q for core in fact['core'])]
        qualifiers = [dict(q, retained=any(q['text'] in quotes[i] for i in supporting)) for q in fact['qualifiers']]
        facts.append({'fact_id':fact['id'], 'covered':bool(supporting),
                      'supporting_quote_indices':supporting, 'qualifiers':qualifiers,
                      'context_note':fact['context']})
    multi=[]
    for i,q in candidates:
        ids=[f['fact_id'] for f in facts if i in f['supporting_quote_indices']]
        if len(ids)>1:
            multi.append({'quote_index':i,'gold_fact_ids':ids})
    aligned=row.get('aligned_claims',[])
    counts={'gold_facts':len(facts), 'facts_covered':sum(f['covered'] for f in facts),
            'gold_qualifiers':sum(len(f['qualifiers']) for f in facts),
            'qualifiers_retained':sum(q['retained'] for f in facts for q in f['qualifiers']),
            'qualifiers_on_covered_facts':sum(len(f['qualifiers']) for f in facts if f['covered']),
            'predicted_quote_items':len(quotes), 'literal_quote_items':len(literal),
            'aligned_quote_items':len(aligned),
            'exact_accepted_spans':sum(exact_span(case['source'],x) for x in aligned),
            'rejected_quote_items':sum(x.get('index') is not None for x in row.get('alignment_rejections',[])),
            'schema_or_json_errors':int(not isinstance(raw,list) or not raw or any(not isinstance(q,str) or not q for q in quotes)),
            'truncated_generations':int(not row.get('ended_with_eos',False)),
            'multi_fact_quote_items':len(multi)}
    return {'case_id':case['id'],'category':case['category'],'counts':counts,
            'facts':facts,'multi_fact_quotes':multi,
            'alignment_rejections':row.get('alignment_rejections',[])}


def evaluate_run(run, dataset):
    expected=[c['id'] for c in dataset['cases']]
    actual=[r['case_id'] for r in run['results']]
    if not run.get('completed') or actual!=expected:
        raise ValueError('Incomplete/reordered/duplicated cases; refuse aggregate metrics.')
    results=[evaluate_case(c,r) for c,r in zip(dataset['cases'],run['results'])]
    totals=Counter()
    groups={}
    for result in results:
        totals.update(result['counts'])
        groups.setdefault(result['category'],Counter()).update(result['counts'])
    return {'label':run['label'],'contract_sha256':run['contract']['contract_sha256'],
            'metric_note':'No composite pass rate. Fact/qualifier evidence is judged from raw predicted strings independently of literal alignment and requires semantic review. Exact accepted spans are guaranteed by construction; always report rejected predictions and malformed/truncated outputs alongside them.',
            'totals':dict(totals),'by_category':{k:dict(v) for k,v in groups.items()},'cases':results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    metrics=evaluate_run(read_json(a.run),read_json(ROOT/'data/cases.json'))
    write_json(a.output,metrics)
    for key,value in metrics['totals'].items():
        print(f'{key}: {value}')


if __name__=='__main__':
    main()
