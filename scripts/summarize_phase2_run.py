"""Audit and summarize a Phase 2 run; do not infer ground-truth metrics."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_phase2_run import audit


def summarize(run, inputs, input_path):
    # The summary is only produced after the independently maintained audit passes.
    verified = audit(run, inputs, input_path)
    by_round = defaultdict(lambda: {'samples': 0, 'accepted_quote_occurrences': 0,
        'rejection_reasons': Counter(), 'verdicts': Counter(), 'verifier_statuses': Counter(),
        'source_logic_statuses': Counter(), 'feedback_entries_with_text': 0,
        'source_logic_feedbacks': 0, 'feedback_transitions_to_next_round': 0,
        'label_logprob_sum': 0., 'label_entropy_sum': 0., 'label_tokens': 0,
        'verdict_generations_missing_label_token_alignment': 0,
        'generations_missing': Counter(), 'generations_without_eos': Counter()})
    samples = []
    for row in run['results']:
        interaction = row['interaction']
        rounds = []
        for index, step in enumerate(interaction['rounds']):
            totals = by_round[step['round']]
            totals['samples'] += 1
            claims = step.get('accepted_claims', [])
            rejections = step.get('alignment_rejections', [])
            totals['accepted_quote_occurrences'] += len(claims)
            totals['rejection_reasons'].update(r['reason'] for r in rejections)
            missing, truncated = [], []
            def generation_state(role, generation):
                if not isinstance(generation, dict):
                    missing.append(role)
                    totals['generations_missing'][role] += 1
                    return 'missing'
                if not generation.get('ended_with_eos'):
                    truncated.append(role)
                    totals['generations_without_eos'][role] += 1
                    return 'no_eos_possible_truncation'
                return 'ended_with_eos'
            extract_state = generation_state('extract', step.get('extract_response'))
            verifications = []
            for check in step.get('verifications', []):
                status = check.get('status')
                totals['verifier_statuses'][str(status)] += 1
                response = check.get('response') if isinstance(check.get('response'), dict) else {}
                verdict = response.get('verdict')
                totals['verdicts'][verdict if verdict is not None else 'missing_or_invalid'] += 1
                vg, fg = response.get('verdict_generation'), response.get('feedback_generation')
                verdict_state = generation_state('verdict', vg)
                feedback_state = generation_state('feedback', fg)
                label_tokens = [t for t in (vg or {}).get('token_trace', []) if t.get('label_token') is True]
                count = len(label_tokens)
                if count:
                    totals['label_tokens'] += count
                    totals['label_logprob_sum'] += sum(t['raw_logprob'] for t in label_tokens)
                    totals['label_entropy_sum'] += sum(t['full_vocab_entropy_nats'] for t in label_tokens)
                else:
                    totals['verdict_generations_missing_label_token_alignment'] += 1
                note = response.get('feedback')
                if isinstance(note, str) and note.strip():
                    totals['feedback_entries_with_text'] += 1
                verifications.append({'quote_index': check.get('quote_index'), 'claim': check.get('claim'),
                    'status': status, 'verdict': verdict, 'verdict_generation_state': verdict_state,
                    'feedback_generation_state': feedback_state, 'feedback': note,
                    'label_token_count': count,
                    'mean_label_token_logprob': sum(t['raw_logprob'] for t in label_tokens)/count if count else None,
                    'mean_label_token_full_vocab_entropy_nats': sum(t['full_vocab_entropy_nats'] for t in label_tokens)/count if count else None})
            logic = step.get('source_logic_review')
            if logic is None:
                logic_status, logic_feedback, logic_state = 'missing', None, generation_state('source_logic', None)
            else:
                logic_status = logic.get('status')
                logic_response = logic.get('response') if isinstance(logic.get('response'), dict) else {}
                logic_feedback = logic_response.get('feedback')
                logic_state = generation_state('source_logic', logic_response.get('generation'))
                if isinstance(logic_feedback, str) and logic_feedback.strip():
                    totals['source_logic_feedbacks'] += 1
            totals['source_logic_statuses'][str(logic_status)] += 1
            next_feedback = step.get('next_feedback')
            transition = index + 1 < len(interaction['rounds'])
            totals['feedback_transitions_to_next_round'] += transition
            rounds.append({'round': step['round'], 'accepted_claims': claims, 'rejections': rejections,
                'extract_generation_state': extract_state, 'extract_error': step.get('extract_error'),
                'verifications': verifications, 'source_logic_status': logic_status,
                'source_logic_generation_state': logic_state, 'source_logic_feedback': logic_feedback,
                'feedback_for_next_round': next_feedback, 'feedback_passed_to_next_round': transition,
                'missing_generations': missing, 'generations_without_eos': truncated,
                'stop_reason': step.get('stop_reason')})
        samples.append({'id': row['id'], 'source': row['source'], 'round_count': len(rounds),
                        'stop_reason': interaction.get('stop_reason'), 'rounds': rounds})
    aggregates = []
    for number, entry in sorted(by_round.items()):
        count = entry.pop('label_tokens')
        lp = entry.pop('label_logprob_sum')
        ent = entry.pop('label_entropy_sum')
        aggregates.append({'round': number, **{key: dict(sorted(value.items())) if isinstance(value, Counter) else value
                                               for key, value in entry.items()},
                           'label_token_count': count,
                           'mean_label_token_logprob': lp / count if count else None,
                           'mean_label_token_full_vocab_entropy_nats': ent / count if count else None})
    return {'audit': verified, 'completed': run['completed'], 'samples_processed': len(samples),
            'round_statistics': aggregates, 'samples': samples,
            'limitations': ['Round totals count extracted quote occurrences, including repeated claims across rounds; they are not unique facts.',
                            'Verdicts, challenges and logic comments are closed-book model output, not independent evidence or gold labels.',
                            'Label-token means use actual generated verdict tokens and raw-logit-derived values; entropy is over the full vocabulary in nats. These are not calibrated probabilities of truth.',
                            'No fact coverage, qualifier retention, atomicity, veracity accuracy or false-negative rate is inferred.',
                            'No EOS indicates possible truncation; saved token scores and mapping are not independently recomputed from model logits.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run = json.loads(args.run.read_text(encoding='utf-8'))
    inputs = json.loads(args.input.read_text(encoding='utf-8'))
    result = summarize(run, inputs, args.input)
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
