"""Reject score, label and information-condition tampering without a model."""
import copy
import json
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from accuracy_backend import LABEL_MAP, NEUTRAL_PROMPT, REVIEW_PROMPT, EVIDENCE_PROMPT, EVIDENCE_REVIEW_PROMPT
from scripts.audit_accuracy import check_arm, check_decision, check_generation, check_selection

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text(encoding='utf-8'))


def generation(text='TRUE', role='decision', ended=True):
    params = {k: CONFIG[k] for k in ('max_new_tokens', 'do_sample', 'num_beams',
              'repetition_penalty', 'temperature', 'top_p', 'top_k', 'use_cache')}
    params['max_new_tokens'] = {'decision': 16, 'review': 128, 'extract': 256}[role]
    result = {'raw_response': text, 'raw_response_with_special_tokens': text + ('<eos>' if ended else ''),
              'generated_token_ids': [7, 9] if ended else [7], 'prompt_input_ids': [1, 2],
              'ended_with_eos': ended, 'generation_params': params, 'messages': []}
    if role == 'decision':
        result.update({'label_token_alignment': 'decoded_piece_overlap',
            'token_trace': [{'step': 0, 'token_id': 7, 'decoded_piece': text, 'special': False,
                            'label_token': True, 'raw_logprob': -.25, 'full_vocab_entropy_nats': .4}],
            'mean_label_token_logprob': -.25, 'mean_label_token_full_vocab_entropy_nats': .4,
            'mean_nonspecial_generated_logprob': -.25})
        if ended:
            result['token_trace'].append({'step': 1, 'token_id': 9, 'decoded_piece': '', 'special': True,
                                         'label_token': False, 'raw_logprob': -.01, 'full_vocab_entropy_nats': .1})
    return result


def messages(g, system, payload):
    g['messages'] = [{'role': 'system', 'content': system},
                     {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    return g


def result(g):
    return {'verdict': 'SUPPORTED', 'raw_label': 'TRUE', 'label_map': dict(LABEL_MAP),
            'generation': g, 'parse_error': None, 'extraction': None,
            'condition': 'closed_book', 'elapsed_seconds': 1.,
            'calls': [{'role': 'decision', 'generation': copy.deepcopy(g)}]}


class FakeTokenizer:
    all_special_ids = [9]

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join({7: 'TRUE', 9: '' if skip_special_tokens else '<eos>'}[i] for i in ids)


class AccuracyAuditTests(unittest.TestCase):
    def check(self, g, **kwargs):
        check_generation(g, CONFIG, [9], 'decision', Counter(), LABEL_MAP, **kwargs)

    def test_actual_alternate_label_scores_and_mapping_pass(self):
        g = generation()
        self.check(g, tokenizer=FakeTokenizer())
        check_decision(result(g), legacy=False)

    def test_corrupt_mean_cannot_masquerade_as_raw_label_confidence(self):
        for field in ('mean_label_token_logprob', 'mean_label_token_full_vocab_entropy_nats'):
            g = generation()
            g[field] = .9
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'mean differs'):
                self.check(g)

    def test_label_map_cannot_silently_invert_true_false(self):
        r = result(generation())
        r['verdict'] = 'REFUTED'
        with self.assertRaisesRegex(ValueError, 'Canonical verdict mapping'):
            check_decision(r, False)
        r = result(generation())
        r['label_map']['TRUE'] = 'REFUTED'
        with self.assertRaisesRegex(ValueError, 'Raw label/map'):
            check_decision(r, False)

    def test_incomplete_true_is_invalid_even_with_a_valid_token_score(self):
        g = generation(ended=False)
        self.check(g)
        r = result(g)
        with self.assertRaisesRegex(ValueError, 'Canonical verdict mapping'):
            check_decision(r, False)
        r.update(verdict=None, parse_error='Expected one exact allowed label followed by EOS.')
        check_decision(r, False)

    def test_wrong_eos_or_token_mean_flags_fail(self):
        for change in ('eos', 'label'):
            g = generation()
            if change == 'eos':
                g['ended_with_eos'] = False
            else:
                g['token_trace'][-1]['label_token'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.check(g)

    def test_prompt_ids_must_replay_with_tokenizer(self):
        g = generation()
        g['prompt_input_ids'] = [1, 999]
        with self.assertRaisesRegex(ValueError, 'Tokenizer prompt IDs'):
            self.check(g, tokenizer=FakeTokenizer())

    def test_generation_cap_cannot_drift(self):
        g = generation()
        g['generation_params']['max_new_tokens'] = 64
        with self.assertRaisesRegex(ValueError, 'parameters differ'):
            self.check(g)

    def test_closed_book_payload_cannot_receive_gold_context_or_qa(self):
        sample = {'text': 'The claim', 'context': {'claim_date': '2020-01-01'},
                  'evidence': [{'question': 'Q', 'answer': 'A', 'url': 'https://source.test'}]}
        for key in ('id', 'context', 'evidence', 'gold'):
            payload = {'statement': sample['text'], key: 'leaked'}
            g = messages(generation(), NEUTRAL_PROMPT, payload)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'permitted information'):
                check_arm(sample, 'neutral', result(g), CONFIG, [9], Counter())

    def test_truncated_review_must_not_enter_the_final_judge(self):
        sample = {'text': 'The claim'}
        review = messages(generation('Unfinished review', 'review', False), REVIEW_PROMPT,
                          {'statement': 'The claim'})
        g = messages(generation(), NEUTRAL_PROMPT, {'statement': 'The claim', 'review': 'Unfinished review'})
        r = result(g)
        r['calls'].insert(0, {'role': 'review', 'generation': review})
        with self.assertRaisesRegex(ValueError, 'permitted information'):
            check_arm(sample, 'reasoned', r, CONFIG, [9], Counter())

    def test_evidence_urls_are_provenance_not_model_payload(self):
        sample = {'text': 'The claim', 'context': {'claim_date': '2020-01-01'},
                  'evidence': [{'question': 'Q', 'answer': 'A', 'url': 'https://factcheck.test/false'}]}
        payload = {'statement': 'The claim', 'context': sample['context'],
                   'evidence': [{'id': 'E1', 'question': 'Q', 'answer': 'A'}]}
        final = payload
        g = messages(generation(), EVIDENCE_PROMPT, final)
        r = result(g)
        r.update(condition='supplied_benchmark_evidence')
        check_arm(sample, 'evidence', r, CONFIG, [9], Counter())
        leaked = copy.deepcopy(final)
        leaked['evidence'][0]['url'] = sample['evidence'][0]['url']
        messages(r['generation'], EVIDENCE_PROMPT, leaked)
        r['calls'][-1]['generation'] = copy.deepcopy(r['generation'])
        with self.assertRaisesRegex(ValueError, 'permitted information'):
            check_arm(sample, 'evidence', r, CONFIG, [9], Counter())

    def selection_fixture(self, winner='neutral', dev_completed=True):
        selected = {'selected_closed_book': winner, 'candidate_ranking': ['neutral', 'reasoned', 'structured'],
                    'development_run': 'reports/development_fixture.json', 'development_run_sha256': 'same',
                    'rule': 'development_only'}
        dev = {'completed': dev_completed, 'completed_at_utc': '2026-11-12T11:30:00+00:00',
               'contract': {'split': 'dev', 'files_sha256': {}, 'selection_rule': 'development_only'},
               'results': [{}]}
        run = {'created_at_utc': '2026-11-12T12:00:00+00:00',
               'contract': {'selection_sha256': 'same',
                            'arms': ['legacy_direct', 'legacy_dual', winner, 'evidence']}}
        scored = {'arms': {a: {'metrics': {'accuracy': value, 'false_positive_rate': .5},
                              'cost': {'elapsed_seconds': 1.}}
                           for a, value in [('neutral', .8), ('reasoned', .6), ('structured', .4)]}}
        return selected, dev, run, scored

    def invoke_selection(self, winner='neutral', dev_completed=True, frozen_at='2026-11-12T11:45:00+00:00'):
        selected, dev, run, scored = self.selection_fixture(winner, dev_completed)
        with patch('scripts.audit_accuracy.sha', return_value='same'), \
             patch('scripts.audit_accuracy.minifact.read_json', side_effect=[selected, dev, {}]), \
             patch('scripts.summarize_accuracy.summarize', return_value=scored):
            check_selection(run, Path('selection.json'), frozen_at)

    def test_holdout_winner_is_recomputed_from_completed_development(self):
        self.invoke_selection()
        with self.assertRaisesRegex(ValueError, 'development-only ranking'):
            self.invoke_selection(winner='reasoned')
        with self.assertRaisesRegex(ValueError, 'completed development'):
            self.invoke_selection(dev_completed=False)

    def test_development_must_finish_before_holdout_freeze_not_just_execution(self):
        with self.assertRaisesRegex(ValueError, 'freeze predates development completion'):
            self.invoke_selection(frozen_at='2026-11-12T11:00:00+00:00')


if __name__ == '__main__':
    unittest.main()
