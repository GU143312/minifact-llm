"""Teacher-forced candidate verdict scores for an autoregressive causal LM.

Candidates are explicit token ID sequences ending in a shared terminator. Their
sum log-probabilities include that terminator; softmax is taken over those sums
without length normalization. The resulting distribution and its natural-log
entropy are conditional on this finite set of candidates, not calibrated truth
probabilities. Full-vocabulary per-token entropy is reported separately.

Token IDs define the experiment: tokenize any text candidates ahead of time,
freeze and record those IDs and the prompt IDs, and ensure the forced token
continuations decode as intended in context. This code does not conflate the
likelihood of extracted quotes with the likelihood of verifier verdicts.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def score_verdict_candidates(model, prompt_ids: Sequence[int], candidates: Mapping[str, Sequence[int]],
                             terminator_id: int):
    """Score complete, terminated label-token sequences using raw forward logits.

    ``prompt_ids`` already contains the formatted verifier prompt and assistant
    generation prefix. Each candidate includes the same final EOS or dedicated
    delimiter token and has no earlier terminator. Caller chooses model, evidence,
    formatting, and token IDs, which are all part of the experimental protocol.
    This function neither samples nor applies generation-time logit processors.
    """
    import torch

    def valid_id(value):
        return type(value) is int and value >= 0

    if not prompt_ids or any(not valid_id(t) for t in prompt_ids):
        raise ValueError('prompt_ids must be a nonempty sequence of nonnegative integers')
    if not valid_id(terminator_id):
        raise ValueError('terminator_id must be a nonnegative integer')
    if len(candidates) < 2 or any(not isinstance(k, str) or not k for k in candidates):
        raise ValueError('at least two named candidates are required')
    fixed = {}
    for name, ids in candidates.items():
        ids = list(ids)
        if len(ids) < 2 or any(not valid_id(t) for t in ids):
            raise ValueError(f'{name}: candidate needs label tokens and a terminator')
        if ids[-1] != terminator_id or terminator_id in ids[:-1]:
            raise ValueError(f'{name}: terminator must occur once, at the end')
        fixed[name] = ids
    if len({tuple(v) for v in fixed.values()}) != len(fixed):
        raise ValueError('different labels must have distinct token sequences')
    if getattr(model, 'training', False):
        raise ValueError('model must be in eval mode for reproducible scoring')

    device = model.get_input_embeddings().weight.device
    if device.type == 'meta':
        raise ValueError('input embedding device is meta; place inputs on the actual model device')
    rows = []
    with torch.inference_mode():
        for name, ids in fixed.items():
            joined = list(prompt_ids) + ids
            model_input = torch.tensor([joined[:-1]], dtype=torch.long, device=device)
            output = model(input_ids=model_input, attention_mask=torch.ones_like(model_input), use_cache=False)
            logits = output.logits[0, len(prompt_ids)-1:len(prompt_ids)-1+len(ids)].float()
            if logits.shape[0] != len(ids):
                raise ValueError('model returned an unexpected number of token positions')
            if max(ids) >= logits.shape[-1]:
                raise ValueError(f'{name}: candidate token ID outside model vocabulary')
            logdist = torch.log_softmax(logits, dim=-1)
            target = torch.tensor(ids, device=logits.device).unsqueeze(-1)
            token_logprobs = logdist.gather(-1, target).squeeze(-1)
            # For each teacher-forced prefix, H over the entire model vocabulary.
            token_entropy = -(logdist.exp() * logdist).sum(dim=-1)
            token_values = [float(x) for x in token_logprobs.cpu().tolist()]
            entropy_values = [float(x) for x in token_entropy.cpu().tolist()]
            if not all(math.isfinite(x) for x in token_values + entropy_values):
                raise ValueError('model returned non-finite scores')
            rows.append({'label': name, 'token_ids': ids, 'token_logprobs': token_values,
                         'token_full_vocab_entropy_nats': entropy_values,
                         'sequence_logprob': sum(token_values)})

    # Stable log-sum-exp; these are relative probabilities among complete choices.
    max_score = max(r['sequence_logprob'] for r in rows)
    weights = [math.exp(r['sequence_logprob'] - max_score) for r in rows]
    total = sum(weights)
    probs = [w / total for w in weights]
    for row, probability in zip(rows, probs):
        row['candidate_set_probability'] = probability
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return {'method': 'teacher_forced_raw_logits', 'score_scope': 'verifier_verdict_candidates',
            'terminator_id': terminator_id, 'prompt_token_ids': list(prompt_ids),
            'sequence_score': 'sum_token_logprob_including_terminator',
            'normalization': 'softmax_of_sequence_logprobs_over_supplied_candidates',
            'entropy_unit': 'nats', 'candidate_set_entropy': entropy, 'candidates': rows}
