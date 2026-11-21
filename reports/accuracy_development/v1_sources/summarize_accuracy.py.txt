"""Score runs offline and select new methods strictly from development results."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import minifact
from accuracy_backend import CLOSED_CANDIDATES
from phase3_metrics import evaluate, compare_paired
from scripts.evaluate_phase3 import generations
from scripts.run_phase3 import sha, write


def predictions(run, arm):
    result = []
    for sample in run['results']:
        row = sample['arms'][arm]
        g = row['generation']
        result.append({'id': sample['id'], 'verdict': row['verdict'],
                       'mean_label_token_logprob': g['mean_label_token_logprob'],
                       'mean_label_token_full_vocab_entropy_nats': g['mean_label_token_full_vocab_entropy_nats']})
    return result


def arm_calls(row):
    yield from ((c['role'], c['generation']) for c in row['calls'])
    if row.get('interaction'):
        yield from generations(row['interaction'])


def summarize(run, dataset):
    if not run['completed'] or len(run['results']) != len(dataset['samples']):
        raise ValueError('Cannot score incomplete run')
    gold = [{'id': row['id'], 'label': row['gold']['verdict']} for row in dataset['samples']]
    if [(s['id'], s['text']) for s in dataset['samples']] != [(r['id'], r['source']) for r in run['results']]:
        raise ValueError('Dataset/input projection mismatch')
    arms = {}
    for arm in run['contract']['arms']:
        calls = [g for row in run['results'] for _, g in arm_calls(row['arms'][arm])]
        arms[arm] = {'metrics': evaluate(gold, predictions(run, arm)),
                     'cost': {'calls': len(calls),
                              'prompt_tokens': sum(len(g['prompt_input_ids']) for g in calls),
                              'generated_tokens_including_eos': sum(len(g['generated_token_ids']) for g in calls),
                              'generations_without_eos': sum(not g['ended_with_eos'] for g in calls),
                              'elapsed_seconds': sum(row['arms'][arm]['elapsed_seconds'] for row in run['results'])}}
    paired = {arm: compare_paired(gold, predictions(run, 'legacy_dual'), predictions(run, arm))
              for arm in run['contract']['arms'] if arm != 'legacy_dual'} if 'legacy_dual' in arms else {}
    return {'split': run['contract']['split'], 'arms': arms, 'paired_vs_legacy_dual': paired,
            'limitations': ['Closed-book and supplied-evidence conditions must be reported separately.',
                            'Inherited whole-sample labels are not independently reverified atomic gold.',
                            'Generated-label confidence scores are not truth probabilities.',
                            'No claim of end-to-end web retrieval or unseen-pretraining data.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selection-output', type=Path)
    args = p.parse_args()
    run = minifact.read_json(args.run)
    split = run['contract']['split']
    path = ROOT / f'data/improvement/{split}_dataset.json'
    if sha(path) != run['contract']['files_sha256'][f'data/improvement/{split}_dataset.json']:
        raise ValueError('Scoring gold differs from frozen data')
    result = summarize(run, minifact.read_json(path))
    result['run_sha256'] = sha(args.run)
    write(args.output, result)
    if args.selection_output:
        if split != 'dev' or args.selection_output.exists():
            raise ValueError('Only a new development selection may be written')
        available = [a for a in CLOSED_CANDIDATES if a in result['arms']]
        def key(a):
            item = result['arms'][a]
            return (-item['metrics']['accuracy'], item['metrics']['false_positive_rate'],
                    item['cost']['elapsed_seconds']/len(run['results']), a)
        ranked = sorted(available, key=key)
        write(args.selection_output, {'selected_closed_book': ranked[0], 'candidate_ranking': ranked,
              'development_run': args.run.resolve().relative_to(ROOT).as_posix(),
              'development_run_sha256': sha(args.run), 'development_metrics_sha256': sha(args.output),
              'rule': run['contract']['selection_rule']})
    print('PASS metrics:', args.output)


if __name__ == '__main__':
    main()
