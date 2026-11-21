"""Independent candidate arithmetic, frozen-threshold and stage-leak guards."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from accuracy_backend import BINARY_PROMPT
from accuracy_calibration import sha, write
from scripts.audit_accuracy_calibration import (audit, check_scoring, compare_fit,
                                                independent_fit, threshold_counts)


def scored_item(sample_id='a', source='Claim A', true_score=-.25, false_score=-2.):
    candidates = []
    for label, token, score in [('TRUE', 1, true_score), ('FALSE', 2, false_score)]:
        candidates.append({'label': label, 'token_ids': [token, 9], 'token_logprobs': [score, -.01],
                           'token_full_vocab_entropy_nats': [.4, .1], 'sequence_logprob': score-.01})
    margin = candidates[0]['sequence_logprob']-candidates[1]['sequence_logprob']
    return {'id': sample_id, 'source': source, 'margin': margin, 'elapsed_seconds': .5,
            'candidate_scoring': {'method': 'teacher_forced_complete_binary_candidates', 'forward_calls': 2,
                'messages': [{'role': 'system', 'content': BINARY_PROMPT},
                             {'role': 'user', 'content': json.dumps({'statement': source}, ensure_ascii=False)}],
                'prompt_token_ids': [0], 'terminator_id': 9, 'candidates': candidates, 'margin': margin}}


class Tokenizer:
    all_special_ids = [9]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [0] if tokenize else 'PROMPT'

    def encode(self, text, add_special_tokens):
        return {'PROMPT': [0], 'PROMPTTRUE': [0, 1], 'PROMPTFALSE': [0, 2]}[text]

    def decode(self, ids, skip_special_tokens):
        return ''.join({1: 'TRUE', 2: 'FALSE', 9: '' if skip_special_tokens else '<eos>'}[i] for i in ids)


class CandidateArithmeticTests(unittest.TestCase):
    def test_complete_sequence_and_tokenizer_replay_pass(self):
        mapping = check_scoring(scored_item(), 'Claim A', 9, Tokenizer())
        self.assertEqual(mapping, {'TRUE': [1, 9], 'FALSE': [2, 9]})

    def test_eos_cannot_be_excluded_from_sequence_sum(self):
        item = scored_item()
        item['candidate_scoring']['candidates'][0]['sequence_logprob'] = -.25
        with self.assertRaisesRegex(ValueError, 'sum all label and EOS'):
            check_scoring(item, 'Claim A', 9)

    def test_margin_must_be_true_minus_false_in_both_saved_locations(self):
        for field in ('outer', 'inner'):
            item = scored_item()
            target = item if field == 'outer' else item['candidate_scoring']
            target['margin'] = 999.
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'margin differs'):
                check_scoring(item, 'Claim A', 9)

    def test_early_eos_nonfinite_entropy_and_duplicate_candidates_fail(self):
        for damage in ('eos', 'entropy', 'duplicate'):
            item = scored_item()
            first, second = item['candidate_scoring']['candidates']
            if damage == 'eos':
                first['token_ids'] = [9, 9]
            elif damage == 'entropy':
                first['token_full_vocab_entropy_nats'][0] = float('nan')
            else:
                second['token_ids'] = list(first['token_ids'])
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                check_scoring(item, 'Claim A', 9)

    def test_gold_id_or_evidence_cannot_be_added_to_prompt(self):
        item = scored_item()
        item['candidate_scoring']['messages'][1]['content'] = json.dumps(
            {'statement': 'Claim A', 'gold': 'SUPPORTED'})
        with self.assertRaisesRegex(ValueError, 'extra information'):
            check_scoring(item, 'Claim A', 9)

    def test_retokenized_chat_prefix_fails(self):
        class WrongBoundary(Tokenizer):
            def encode(self, text, add_special_tokens):
                return [8, 1] if text == 'PROMPTTRUE' else super().encode(text, add_special_tokens)
        with self.assertRaisesRegex(ValueError, 'chat-prefix'):
            check_scoring(scored_item(), 'Claim A', 9, WrongBoundary())


class ThresholdAuditTests(unittest.TestCase):
    def test_independent_counts_and_fit_are_not_truth_probabilities(self):
        rows = [{'id': 's1', 'margin': -2., 'gold': 'SUPPORTED'},
                {'id': 's2', 'margin': -3., 'gold': 'SUPPORTED'},
                {'id': 'r1', 'margin': -5., 'gold': 'REFUTED'},
                {'id': 'r2', 'margin': -6., 'gold': 'REFUTED'}]
        zero, fitted = threshold_counts(rows, 0.), independent_fit(rows)
        self.assertEqual(zero['correct'], 2)
        self.assertEqual(zero['false_positive_rate'], 1.)
        self.assertEqual(fitted['threshold'], -3.)
        self.assertEqual(fitted['correct'], 4)
        with self.assertRaisesRegex(ValueError, 'fitted threshold differs'):
            compare_fit({**fitted, 'threshold': 0.}, fitted)

    def test_fit_tie_prefers_lower_fpr_then_closest_zero(self):
        rows = [{'id': 's', 'margin': -1., 'gold': 'SUPPORTED'},
                {'id': 'r', 'margin': -1., 'gold': 'REFUTED'}]
        fit = independent_fit(rows)
        self.assertEqual(fit['threshold'], -1.)
        self.assertEqual(fit['false_positive_rate'], 0.)


class StageAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = {'model_revision': 'rev', 'dtype': 'float16', 'attention_implementation': 'eager'}
        self.contract = {'model': self.config, 'protocol': 'fixture'}
        self.execution = {'actual_revision': 'rev', 'actual_dtype': 'torch.float16',
                          'attention_implementation': 'eager', 'terminator_id': 9}
        self.rows = [{'id': 'a', 'text': 'Claim A'}, {'id': 'b', 'text': 'Claim B'}]
        self.inputs = self.root/'data/improvement/dev_inputs.json'
        self.holdout = self.root/'data/improvement/holdout_inputs.json'
        write(self.inputs, self.rows)
        write(self.holdout, [{'id': 'h', 'text': 'Holdout claim'}])
        self.gold = self.root/'data/improvement/dev_dataset.json'
        write(self.gold, {'samples': [{**r, 'gold': {'verdict': g}} for r, g in
                                     zip(self.rows, ['SUPPORTED', 'REFUTED'])]})
        self.freeze_path = self.root/'dev.freeze.json'
        self.freeze = {'frozen_at_utc': '2026-11-12T10:00:00+00:00', 'stage': 'dev',
                       'contract': self.contract, 'inputs_sha256': sha(self.inputs),
                       'calibration_sha256': None, 'threshold': None,
                       'development_gold_sha256': sha(self.gold),
                       'future_holdout_input_sha256': sha(self.holdout)}
        write(self.freeze_path, self.freeze)
        a, b = scored_item(), scored_item('b', 'Claim B', -2., -.25)
        fitted = independent_fit([{'id': 'a', 'margin': a['margin'], 'gold': 'SUPPORTED'},
                                 {'id': 'b', 'margin': b['margin'], 'gold': 'REFUTED'}])
        a['verdict'], b['verdict'] = 'SUPPORTED', 'REFUTED'
        self.run = {'contract': self.contract, 'stage': 'dev', 'completed': True,
                    'freeze_sha256': sha(self.freeze_path), 'execution': self.execution,
                    'created_at_utc': '2026-11-12T10:00:01+00:00',
                    'completed_at_utc': '2026-11-12T10:01:00+00:00',
                    'results': [a, b], 'development_fit': fitted}
        self.dev_run = self.root/'dev.json'
        write(self.dev_run, self.run)
        self.calibration_path = self.root/'calibration.json'
        self.calibration = {'created_at_utc': '2026-11-12T10:02:00+00:00', 'contract': self.contract,
                            'development_run': 'dev.json', 'development_run_sha256': sha(self.dev_run),
                            'development_gold_sha256': sha(self.gold),
                            'development_input_sha256': sha(self.inputs),
                            'holdout_input_sha256': sha(self.holdout),
                            'candidate_token_ids': {'TRUE': [1, 9], 'FALSE': [2, 9]},
                            'terminator_id': 9, 'execution': self.execution, 'fit': fitted}
        write(self.calibration_path, self.calibration)

    def invoke(self, run=None, freeze=None, path=None, inputs=None, calibration=None):
        with patch('scripts.audit_accuracy_calibration.code_contract', return_value=self.contract):
            return audit(self.run if run is None else run, self.freeze if freeze is None else freeze,
                         path or self.freeze_path, inputs or self.inputs, calibration, self.calibration_path,
                         root=self.root)

    def test_complete_development_recomputes_fit_and_zero_reference(self):
        checked = self.invoke(calibration=self.calibration)
        self.assertTrue(checked['passed'])
        self.assertEqual(checked['development_diagnostics']['fitted_threshold']['correct'], 2)

    def test_prediction_source_and_threshold_tampering_fail(self):
        for damage in ('source', 'prediction', 'fit'):
            changed = copy.deepcopy(self.run)
            if damage == 'source':
                changed['results'][0]['source'] = 'Different claim'
            elif damage == 'prediction':
                changed['results'][0]['verdict'] = 'REFUTED'
            else:
                changed['development_fit']['threshold'] = 100.
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                self.invoke(run=changed)

    def holdout_fixture(self):
        freeze = {'frozen_at_utc': '2026-11-12T10:03:00+00:00', 'stage': 'holdout',
                  'contract': self.contract, 'inputs_sha256': sha(self.holdout),
                  'calibration_sha256': sha(self.calibration_path), 'threshold': self.calibration['fit']['threshold']}
        path = self.root/'holdout.freeze.json'
        write(path, freeze)
        item = scored_item('h', 'Holdout claim')
        item['verdict'] = 'SUPPORTED'
        run = {'contract': self.contract, 'stage': 'holdout', 'completed': True,
               'freeze_sha256': sha(path), 'execution': self.execution,
               'created_at_utc': '2026-11-12T10:03:01+00:00',
               'completed_at_utc': '2026-11-12T10:04:00+00:00', 'results': [item],
               'calibration_sha256': sha(self.calibration_path), 'threshold': self.calibration['fit']['threshold']}
        return run, freeze, path

    def test_holdout_audit_needs_no_holdout_gold_and_forbids_refitting(self):
        run, freeze, path = self.holdout_fixture()
        self.assertFalse((self.root/'data/improvement/holdout_dataset.json').exists())
        self.assertTrue(self.invoke(run, freeze, path, self.holdout, self.calibration)['passed'])
        run['development_fit'] = {'threshold': 1.}
        with self.assertRaisesRegex(ValueError, 'must not refit'):
            self.invoke(run, freeze, path, self.holdout, self.calibration)

    def test_calibration_must_exist_before_the_holdout_freeze(self):
        run, freeze, path = self.holdout_fixture()
        freeze['frozen_at_utc'] = '2026-11-12T10:01:30+00:00'
        write(path, freeze)
        run['freeze_sha256'] = sha(path)
        with self.assertRaisesRegex(ValueError, 'postdates evaluation freeze'):
            self.invoke(run, freeze, path, self.holdout, self.calibration)


if __name__ == '__main__':
    unittest.main()
