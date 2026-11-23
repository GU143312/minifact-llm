"""Audit paired model records, then score inherited whole-sample labels offline."""
import argparse
import copy
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minifact
from phase2_backend import LABELS
from phase3_experiment import SAMPLE_PROMPT, analysis_from_interaction, prediction
from phase3_metrics import compare_paired, evaluate
from scripts.audit_phase2_run import audit as audit_phase2, audit_generation, require
from scripts.run_phase3 import contract, sha, validate_inputs, write


def audit(run, inputs, input_path, freeze, freeze_path):
    validate_inputs(inputs)
    require(run.get('contract') == contract(input_path), 'Current protocol/hash mismatch')
    require(freeze.get('contract') == run['contract'], 'Freeze contract mismatch')
    require(run.get('freeze_sha256') == sha(freeze_path), 'Freeze bytes mismatch')
    require(datetime.fromisoformat(freeze['frozen_at_utc']) <= datetime.fromisoformat(run['created_at_utc']),
            'Run predates self-reported freeze time')
    require(run.get('completed') is True and len(run['results']) == len(inputs), 'Run incomplete')
    require(datetime.fromisoformat(run['created_at_utc']) <= datetime.fromisoformat(run['completed_at_utc']),
            'Run timestamps out of order')
    # Reuse the independently tested Phase 2 auditor on every internal generation,
    # actual feedback transition and literal reference, with the new label-free input.
    wrapper = {'completed': True, 'contract': run['contract']['phase2_contract'],
               'execution': run['execution'],
               'results': [{'id': r['id'], 'source': r['source'], 'interaction': r['interaction']}
                           for r in run['results']]}
    phase2_audit = audit_phase2(wrapper, inputs, input_path)
    eos = run['execution']['model_generation_config']['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    counts = Counter()
    for index, (case, row) in enumerate(zip(inputs, run['results'])):
        require(row['id'] == case['id'] and row['source'] == case['text'], 'Source/ID mismatch')
        expected_order = ['direct', 'dual'] if index % 2 == 0 else ['dual', 'direct']
        require(row['arm_order'] == expected_order, 'Arm execution order mismatch')
        for arm in ('direct', 'dual'):
            result = row[arm]
            g = result['generation']
            audit_generation(g, eos, counts, run['contract']['model'], 'verdict', trace_required=True)
            analysis = None if arm == 'direct' else analysis_from_interaction(row['interaction'])
            messages = [{'role': 'system', 'content': SAMPLE_PROMPT},
                        {'role': 'user', 'content': json.dumps(
                            {'source': case['text'], 'analysis': analysis}, ensure_ascii=False)}]
            require(g['messages'] == messages, 'Sample judge prompt/analysis mismatch')
            parsed = g['raw_response'].strip()
            parsed = parsed if parsed in LABELS and g['ended_with_eos'] else None
            require(result['verdict'] == parsed, 'Sample label parse mismatch')
            require(result['parse_error'] == (None if parsed else 'Expected one allowed label followed by EOS.'),
                    'Parse-error state mismatch')
            elapsed = row[arm + '_elapsed_seconds']
            require(type(elapsed) in (float, int) and math.isfinite(elapsed) and elapsed >= 0,
                    'Invalid elapsed seconds')
    return {'passed': True, 'samples': len(inputs), 'phase2': phase2_audit,
            'sample_judges': dict(counts),
            'boundary': 'Structural/record checks; raw logits and tokenizer prompt IDs require model re-execution. Self-reported timestamps are not independent trusted timestamps.'}


def generations(interaction):
    for step in interaction['rounds']:
        if step.get('extract_response'):
            yield 'extract', step['extract_response']
        for check in step['verifications']:
            response = check.get('response') or {}
            for key in ('verdict_generation', 'feedback_generation'):
                if response.get(key):
                    yield key, response[key]
        logic = (step.get('source_logic_review') or {}).get('response') or {}
        if logic.get('generation'):
            yield 'source_logic', logic['generation']


def costs(run, arm):
    counts, prompt, generated, missing_eos = Counter(), 0, 0, 0
    for row in run['results']:
        calls = [('sample_judge', row[arm]['generation'])]
        if arm == 'dual':
            calls.extend(generations(row['interaction']))
        for role, g in calls:
            counts[role] += 1
            prompt += len(g['prompt_input_ids'])
            generated += len(g['generated_token_ids'])
            missing_eos += not g['ended_with_eos']
    return {'calls_by_role': dict(counts), 'total_calls': sum(counts.values()),
            'prompt_tokens_summed_over_calls': prompt, 'generated_tokens_including_eos': generated,
            'generations_without_eos': missing_eos,
            'elapsed_seconds': sum(r[arm + '_elapsed_seconds'] for r in run['results']),
            'timing_boundary': 'CUDA synchronized per arm; includes Python orchestration, excludes model load, checkpoints and scoring.'}


def summarize(run, dataset, audit_result):
    gold = [{'id': r['id'], 'label': r['gold']['verdict']} for r in dataset['samples']]
    direct = [prediction(r, 'direct') for r in run['results']]
    dual = [prediction(r, 'dual') for r in run['results']]
    groups = {'reused_refuted': {r['id'] for r in gold if r['label'] == 'REFUTED'},
              'new_supported_controls': {r['id'] for r in gold if r['label'] == 'SUPPORTED'}}
    return {'audit': audit_result, 'unit': 'whole_source_claim', 'gold_origin': 'Inherited AVeriTeC sample labels; no independent human re-verification.',
            'arms': {arm: {'metrics': evaluate(gold, preds), 'cost': costs(run, arm)}
                     for arm, preds in [('direct', direct), ('dual', dual)]},
            'paired': compare_paired(gold, direct, dual),
            'subsets': {name: {arm: evaluate([g for g in gold if g['id'] in ids],
                                            [p for p in preds if p['id'] in ids])
                               for arm, preds in [('direct', direct), ('dual', dual)]}
                        for name, ids in groups.items()},
            'limitations': ['Original 60 Refuted cases were already examined during Phase 2; this is a reused benchmark subset, not an unseen test.',
                            'New Supported controls are curated, not randomized population-matched controls; class balance and selection affect accuracy.',
                            'Both arms are closed-book and share weights. Analysis is generated model output, not retrieved evidence.',
                            'Dual arm uses additional calls/tokens and a new sample adjudicator; this system comparison cannot isolate the causal effect of feedback or logprob.',
                            'Selective confidence thresholds were fixed before execution; curves are descriptive, not calibrated truth probabilities or a selected best threshold.',
                            'No independent atomic gold: factual coverage, qualifier retention and atomicity remain unmeasured; literal location checks are reported separately.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--inputs', type=Path, default=ROOT / 'data/phase3/inputs.json')
    p.add_argument('--dataset', type=Path, default=ROOT / 'data/phase3/dataset.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run, inputs = minifact.read_json(args.run), minifact.read_json(args.inputs)
    frozen = minifact.read_json(args.freeze)
    checked = audit(run, inputs, args.inputs, frozen, args.freeze)
    require(sha(args.dataset) == run['contract']['files_sha256']['data/phase3/dataset.json'],
            'Scoring labels differ from frozen dataset bytes')
    dataset = minifact.read_json(args.dataset)
    require([{'id': r['id'], 'text': r['text']} for r in dataset['samples']] == inputs,
            'Dataset does not reproduce label-free inputs')
    result = summarize(run, dataset, checked)
    result['artifact_sha256'] = {'run': sha(args.run), 'freeze': sha(args.freeze),
                                 'dataset': sha(args.dataset), 'inputs': sha(args.inputs)}
    write(args.output, result)
    print('PASS paired audit + metrics:', args.output)


if __name__ == '__main__':
    main()
