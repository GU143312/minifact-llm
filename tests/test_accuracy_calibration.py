"""Development-only threshold fitting and complete-label teacher forcing."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from accuracy_calibration import encode_candidates, fit_threshold, predict, score_candidates, sha, validate_inputs, write
from scripts.run_accuracy_calibration import check_calibration


class ThresholdTests(unittest.TestCase):
    def test_corrects_a_shifted_label_prior_using_development_only(self):
        rows = [{'id': 's1', 'margin': -2., 'gold': 'SUPPORTED'},
                {'id': 's2', 'margin': -3., 'gold': 'SUPPORTED'},
                {'id': 'r1', 'margin': -5., 'gold': 'REFUTED'},
                {'id': 'r2', 'margin': -6., 'gold': 'REFUTED'}]
        fitted = fit_threshold(rows)
        self.assertEqual(fitted['accuracy'], 1.)
        self.assertEqual(fitted['threshold'], -3.)
        self.assertEqual([predict(r['margin'], fitted['threshold']) for r in rows], [r['gold'] for r in rows])
        self.assertTrue(all(predict(r['margin'], 0.) == 'REFUTED' for r in rows))

    def test_tied_development_accuracy_prefers_lower_false_positive_rate(self):
        fit = fit_threshold([{'id': 's', 'margin': -1., 'gold': 'SUPPORTED'},
                             {'id': 'r', 'margin': -1., 'gold': 'REFUTED'}])
        self.assertEqual(fit['accuracy'], .5)
        self.assertEqual(fit['false_positive_rate'], 0.)
        self.assertEqual(fit['threshold'], -1.)

    def test_zero_is_preferred_when_it_separates_the_classes(self):
        fit = fit_threshold([{'id': 's', 'margin': .5, 'gold': 'SUPPORTED'},
                             {'id': 'r', 'margin': -.5, 'gold': 'REFUTED'}])
        self.assertEqual(fit['threshold'], 0.)

    def test_next_float_can_represent_a_strict_boundary(self):
        low = 1.
        high = math.nextafter(low, math.inf)
        fit = fit_threshold([{'id': 's', 'margin': high, 'gold': 'SUPPORTED'},
                             {'id': 'r', 'margin': low, 'gold': 'REFUTED'}])
        self.assertEqual(fit['accuracy'], 1.)
        self.assertEqual(fit['threshold'], high)

    def test_missing_class_duplicate_ids_and_nonfinite_scores_fail(self):
        valid = [{'id': 's', 'margin': 1., 'gold': 'SUPPORTED'},
                 {'id': 'r', 'margin': -1., 'gold': 'REFUTED'}]
        for rows in [[], valid[:1], [valid[0], valid[0]], [None],
                     [{**valid[0], 'margin': float('nan')}, valid[1]],
                     [{**valid[0], 'margin': True}, valid[1]]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                fit_threshold(rows)

    def test_holdout_predictor_accepts_no_gold_argument(self):
        self.assertEqual(predict(-1., -2.), 'SUPPORTED')
        self.assertEqual(predict(-3., -2.), 'REFUTED')
        with self.assertRaises(TypeError):
            predict(-1., -2., 'REFUTED')
        with self.assertRaises(ValueError):
            predict(float('inf'), 0.)

    def test_only_statement_projection_is_permitted(self):
        validate_inputs([{'id': 'a', 'text': 'Claim'}])
        for key in ('context', 'evidence', 'gold', 'label'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_inputs([{'id': 'a', 'text': 'Claim', key: 'leaked'}])

    def test_calibration_binds_unseen_inputs_and_development_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root/'holdout_inputs.json'
            write(input_path, [{'id': 'h', 'text': 'Holdout claim'}])
            current = {'protocol': 'test'}
            write(root/'dev_run.json', {'completed': True, 'stage': 'dev', 'contract': current,
                                        'development_fit': {'threshold': 0.}})
            write(root/'data/improvement/dev_dataset.json', {'development_only': True})
            calibration = {'contract': current, 'holdout_input_sha256': sha(input_path),
                'development_run': 'dev_run.json', 'development_run_sha256': sha(root/'dev_run.json'),
                'development_gold_sha256': sha(root/'data/improvement/dev_dataset.json'),
                'fit': {'threshold': 0.}}
            # There is deliberately no holdout gold file in this workspace.
            check_calibration(calibration, current, input_path, root)
            changed = copy.deepcopy(calibration)
            changed['contract']['protocol'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'code/config/prompt'):
                check_calibration(changed, current, input_path, root)
            changed = copy.deepcopy(calibration)
            changed['fit']['threshold'] = 1.
            with self.assertRaisesRegex(ValueError, 'completed development fit'):
                check_calibration(changed, current, input_path, root)
            write(input_path, [{'id': 'h', 'text': 'Changed holdout claim'}])
            with self.assertRaisesRegex(ValueError, 'Holdout inputs changed'):
                check_calibration(calibration, current, input_path, root)


class Tokenizer:
    all_special_ids = [9]

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return [0] if tokenize else 'PROMPT'

    def encode(self, text, add_special_tokens):
        return {'PROMPT': [0], 'PROMPTTRUE': [0, 1], 'PROMPTFALSE': [0, 2]}[text]

    def decode(self, ids, skip_special_tokens=False):
        return ''.join({1: 'TRUE', 2: 'FALSE', 9: '<EOS>'}[i] for i in ids)


class TokenBoundaryTests(unittest.TestCase):
    def test_complete_labels_have_exact_chat_prefix_and_shared_eos(self):
        messages, prompt, candidates = encode_candidates(Tokenizer(), 'System', 'Claim', 9)
        self.assertEqual(prompt, [0])
        self.assertEqual(candidates, {'TRUE': [1, 9], 'FALSE': [2, 9]})
        self.assertEqual(json.loads(messages[1]['content']), {'statement': 'Claim'})

    def test_retokenized_prompt_boundary_is_rejected(self):
        class Changed(Tokenizer):
            def encode(self, text, add_special_tokens):
                return [8, 1] if text == 'PROMPTTRUE' else super().encode(text, add_special_tokens)
        with self.assertRaisesRegex(ValueError, 'prefix boundary'):
            encode_candidates(Changed(), 'System', 'Claim', 9)


class ToyModel(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 1)
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        self.calls.append({'input': input_ids[0].tolist(), 'keep': logits_to_keep, 'cache': use_cache})
        self.assertions = torch.equal(attention_mask, torch.ones_like(input_ids))
        rows = []
        for token in input_ids[0].tolist():
            rows.append([0., math.log(2), math.log(3), math.log(4)] if token == 0 else
                        [0., 0., 0., 0.] if token == 1 else [0., 0., 0., math.log(3)])
        return SimpleNamespace(logits=torch.tensor([rows[-logits_to_keep:]], dtype=torch.float32))


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class TeacherForcedTests(unittest.TestCase):
    def test_candidate_and_eos_loglikelihoods_not_prompt_likelihood_are_summed(self):
        model = ToyModel().eval()
        scored = score_candidates(model, [0], {'TRUE': [1, 3], 'FALSE': [2, 3]}, 3)
        true, false = scored['candidates']
        self.assertAlmostEqual(true['sequence_logprob'], math.log(.2*.25), places=6)
        self.assertAlmostEqual(false['sequence_logprob'], math.log(.3*.5), places=6)
        self.assertAlmostEqual(scored['margin'], math.log(1/3), places=6)
        self.assertEqual(len(true['token_logprobs']), 2)
        self.assertEqual(model.calls, [{'input': [0, 1], 'keep': 2, 'cache': False},
                                       {'input': [0, 2], 'keep': 2, 'cache': False}])
        self.assertTrue(model.assertions)

    def test_unterminated_duplicate_candidates_and_training_mode_fail(self):
        model = ToyModel().eval()
        for candidates in [{'TRUE': [1, 3], 'FALSE': [2]},
                           {'TRUE': [1, 3], 'FALSE': [1, 3]},
                           {'TRUE': [3, 1, 3], 'FALSE': [2, 3]}]:
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                score_candidates(model, [0], candidates, 3)
        model.train()
        with self.assertRaisesRegex(ValueError, 'eval mode'):
            score_candidates(model, [0], {'TRUE': [1, 3], 'FALSE': [2, 3]}, 3)


if __name__ == '__main__':
    unittest.main()
