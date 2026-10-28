"""Lightweight tests for verifier input isolation and token-boundary checks."""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('phase1_verifier', ROOT / 'scripts/run_phase1_verifier.py')
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class CharacterTokenizer:
    """Deterministic stand-in that exposes every prompt and continuation token."""
    eos_token = '~'
    eos_token_id = 999

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        prompt = 'PROMPT:'
        return [ord(c) for c in prompt] if tokenize else prompt

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [self.eos_token_id if c == self.eos_token else ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(i) for i in ids)


class ExtractionValidationTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [{'id': 'MF-P1-0001', 'text': 'A claim in context.'}]
        self.extraction = {'completed': True, 'results': [{
            'case_id': 'MF-P1-0001', 'source': 'A claim in context.',
            'aligned_claims': [{'claim': 'claim', 'start_idx': 2, 'end_idx': 7, 'quote_index': 0}]
        }]}

    def test_completed_literal_claim_is_accepted(self):
        verifier.validate_extraction(self.extraction, self.inputs)

    def test_incomplete_run_is_rejected(self):
        self.extraction['completed'] = False
        with self.assertRaisesRegex(ValueError, 'completed'):
            verifier.validate_extraction(self.extraction, self.inputs)

    def test_changed_source_is_rejected(self):
        self.extraction['results'][0]['source'] = 'A different claim.'
        with self.assertRaisesRegex(ValueError, 'differ'):
            verifier.validate_extraction(self.extraction, self.inputs)

    def test_mismatched_quote_and_offsets_are_rejected(self):
        self.extraction['results'][0]['aligned_claims'][0]['end_idx'] = 6
        with self.assertRaisesRegex(ValueError, 'non-literal'):
            verifier.validate_extraction(self.extraction, self.inputs)


class PromptAndCandidateTests(unittest.TestCase):
    def test_prompt_contains_only_provided_text_and_no_gold(self):
        tokenizer = CharacterTokenizer()
        source, claim = 'The context has a claim.', 'a claim'
        verifier.formatted_prompt(tokenizer, source, claim)
        self.assertEqual(len(tokenizer.messages), 2)
        self.assertEqual(tokenizer.messages[0]['role'], 'system')
        self.assertEqual(tokenizer.messages[1]['role'], 'user')
        payload = json.loads(tokenizer.messages[1]['content'])
        self.assertEqual(payload, {'source_context': source, 'claim_to_verify': claim})
        self.assertNotIn('gold', payload)
        self.assertNotIn('evidence', payload)
        self.assertNotIn('verdict', payload)

    def test_candidates_and_eos_are_checked_in_context(self):
        tokenizer = CharacterTokenizer()
        prompt, ids = verifier.formatted_prompt(tokenizer, 'source', 'claim')
        candidates, eos = verifier.candidate_tokens(tokenizer, prompt, ids)
        self.assertEqual(set(candidates), set(verifier.LABELS))
        for label, tokens in candidates.items():
            self.assertEqual(tokens[-1], eos)
            self.assertEqual(tokenizer.decode(tokens[:-1]), label)
            self.assertEqual(tokenizer(prompt + label + tokenizer.eos_token)['input_ids'], ids + tokens)

    def test_changed_prompt_boundary_is_rejected(self):
        class BoundaryMergeTokenizer(CharacterTokenizer):
            def __call__(self, text, add_special_tokens=False):
                result = super().__call__(text, add_special_tokens)
                if text.endswith('SUPPORTED'):
                    # A hypothetical subword merge changed the final prompt token.
                    result['input_ids'][len('PROMPT:') - 1] = 12345
                return result

        tokenizer = BoundaryMergeTokenizer()
        prompt, ids = verifier.formatted_prompt(tokenizer, 'source', 'claim')
        with self.assertRaisesRegex(ValueError, 'changed the prompt token prefix'):
            verifier.candidate_tokens(tokenizer, prompt, ids)

    def test_changed_eos_tokenization_is_rejected(self):
        class BadEosTokenizer(CharacterTokenizer):
            def __call__(self, text, add_special_tokens=False):
                result = super().__call__(text, add_special_tokens)
                if text.endswith(self.eos_token):
                    result['input_ids'][-1] = 998
                return result

        tokenizer = BadEosTokenizer()
        prompt, ids = verifier.formatted_prompt(tokenizer, 'source', 'claim')
        with self.assertRaisesRegex(ValueError, 'EOS'):
            verifier.candidate_tokens(tokenizer, prompt, ids)


if __name__ == '__main__':
    unittest.main()
