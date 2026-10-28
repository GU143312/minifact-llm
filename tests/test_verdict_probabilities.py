import math
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from verdict_probabilities import score_verdict_candidates


class PrefixSensitiveModel(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(4, 1)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids, attention_mask, use_cache):
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        assert use_cache is False
        scores = []
        for token in input_ids[0].tolist():
            if token == 0:
                scores.append([0, math.log(2), math.log(3), math.log(4)])
            elif token == 1:
                scores.append([0, 0, 0, 0])
            else:
                scores.append([0, 0, 0, math.log(3)])
        return SimpleNamespace(logits=torch.tensor([scores], dtype=torch.float32))


@unittest.skipIf(torch is None, 'PyTorch is not installed in this environment')
class VerdictProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.model = PrefixSensitiveModel().eval()

    def test_teacher_forcing_and_candidate_set_entropy(self):
        result = score_verdict_candidates(self.model, [0], {'A': [1, 3], 'B': [2, 3]}, 3)
        a, b = result['candidates']
        self.assertAlmostEqual(a['sequence_logprob'], math.log(.2 * .25), places=6)
        self.assertAlmostEqual(b['sequence_logprob'], math.log(.3 * .5), places=6)
        self.assertAlmostEqual(a['candidate_set_probability'], .25, places=6)
        self.assertAlmostEqual(b['candidate_set_probability'], .75, places=6)
        self.assertAlmostEqual(result['candidate_set_entropy'], -(.25*math.log(.25)+.75*math.log(.75)), places=6)
        self.assertAlmostEqual(a['token_full_vocab_entropy_nats'][1], math.log(4), places=6)
        self.assertAlmostEqual(b['token_full_vocab_entropy_nats'][1], -(.5*math.log(.5)+3/6*math.log(1/6)), places=6)

    def test_terminator_is_scored_not_silently_dropped(self):
        row = score_verdict_candidates(self.model, [0], {'A': [1, 3], 'B': [2, 3]}, 3)['candidates'][0]
        self.assertEqual(len(row['token_logprobs']), 2)
        self.assertAlmostEqual(row['token_logprobs'][1], math.log(.25), places=6)

    def test_duplicate_and_unterminated_candidates_rejected(self):
        with self.assertRaises(ValueError):
            score_verdict_candidates(self.model, [0], {'A': [1, 3], 'B': [1, 3]}, 3)
        with self.assertRaises(ValueError):
            score_verdict_candidates(self.model, [0], {'A': [1, 3], 'B': [2]}, 3)
        with self.assertRaises(ValueError):
            score_verdict_candidates(self.model, [0], {'A': [3, 1, 3], 'B': [2, 3]}, 3)

    def test_rejects_training_mode(self):
        self.model.train()
        with self.assertRaises(ValueError):
            score_verdict_candidates(self.model, [0], {'A': [1, 3], 'B': [2, 3]}, 3)


if __name__ == '__main__':
    unittest.main()
