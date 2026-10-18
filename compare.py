"""Refuse uncontrolled comparisons; emit separate metrics for compatible runs."""
import argparse
import math
from pathlib import Path

from evaluate import evaluate_run
from minifact import ROOT, RUNTIME_PACKAGES, contract, read_json, write_json


def compatibility_errors(left, right):
    errors=[]
    if not left.get('completed') or not right.get('completed'):
        errors.append('Both runs must be complete.')
    if left.get('contract') != right.get('contract'):
        errors.append('Code, dataset, prompt, model revision, precision or generation contract differs.')
    for package in RUNTIME_PACKAGES:
        a=left.get('runtime',{}).get('packages',{}).get(package)
        b=right.get('runtime',{}).get('packages',{}).get(package)
        if not a or a!=b:
            errors.append(f'Package differs or missing: {package}.')
    for key in ('actual_dtype','actual_revision','attention_implementation'):
        a=left.get('execution',{}).get(key)
        b=right.get('execution',{}).get(key)
        if not a or a!=b:
            errors.append(f'Actual model setting differs or missing: {key}.')
    if left.get('execution',{}).get('model_generation_config') != right.get('execution',{}).get('model_generation_config'):
        errors.append('Loaded generation defaults differ.')
    ids_left=[r.get('case_id') for r in left.get('results',[])]
    ids_right=[r.get('case_id') for r in right.get('results',[])]
    if ids_left!=ids_right or len(set(ids_left))!=len(ids_left):
        errors.append('Case identifiers/order differ or are duplicated.')
    for a,b in zip(left.get('results',[]),right.get('results',[])):
        if a.get('source')!=b.get('source') or not a.get('prompt_input_ids_sha256') or a.get('prompt_input_ids_sha256')!=b.get('prompt_input_ids_sha256'):
            errors.append(f'Actual prompt tokenization differs: {a.get("case_id")}.')
    return errors


def compare(left,right,dataset):
    errors=compatibility_errors(left,right)
    if errors:
        return {'controlled_comparison':False,'errors':errors}
    if left['contract'] != contract(read_json(ROOT/'config.json'),dataset):
        return {'controlled_comparison':False,'errors':['Current evaluation/code/data differ from the recorded run contract.']}
    a,b=evaluate_run(left,dataset),evaluate_run(right,dataset)
    deltas=[]
    for x,y in zip(left['results'],right['results']):
        deltas.append({'case_id':x['case_id'], 'same_raw_response':x['raw_response']==y['raw_response'],
                       'local_mean_logprob':x['mean_quote_json_logprob'],
                       'colab_mean_logprob':y['mean_quote_json_logprob'],
                       'logprob_difference_colab_minus_local':y['mean_quote_json_logprob']-x['mean_quote_json_logprob']})
    return {'controlled_comparison':True,
            'residual_differences':'GPU architecture, operating system and placement may differ. Greedy FP16 outputs need not be bit-identical; speed is descriptive, not a hardware causal estimate.',
            'local':a, 'colab':b,'per_case_output_comparison':deltas,
            'same_raw_response_cases':sum(x['same_raw_response'] for x in deltas)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('local',type=Path)
    p.add_argument('colab',type=Path)
    p.add_argument('--output',type=Path,default=Path('reports/controlled_comparison.json'))
    a=p.parse_args()
    result=compare(read_json(a.local),read_json(a.colab),read_json(ROOT/'data/cases.json'))
    write_json(a.output,result)
    if not result['controlled_comparison']:
        raise SystemExit('Refused uncontrolled comparison: '+'; '.join(result['errors']))
    for label in ('local','colab'):
        print(label,result[label]['totals'])


if __name__=='__main__':
    main()
