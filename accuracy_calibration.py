"""Development-fitted binary threshold for complete teacher-forced label scores.

Scores include each candidate's label tokens and the common EOS. They are sums,
not length-normalized, not actual greedy-generation scores, and not truth
probabilities. Calibration here means fitting a binary decision threshold only.
"""
import hashlib
import json
import math
from pathlib import Path

CANONICAL = {'TRUE': 'SUPPORTED', 'FALSE': 'REFUTED'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    temporary.replace(path)


def validate_inputs(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected nonempty id/text inputs')
    if any(not isinstance(r, dict) or set(r) != {'id', 'text'} or
           not isinstance(r['id'], str) or not r['id'].strip() or
           not isinstance(r['text'], str) or not r['text'].strip() for r in rows):
        raise ValueError('Only nonempty id/text inputs are permitted')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate input ID')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def predict(margin, threshold):
    if not finite(margin) or not finite(threshold):
        raise ValueError('Margin and threshold must be finite')
    return 'SUPPORTED' if margin >= threshold else 'REFUTED'


def fit_threshold(rows):
    """Fit on development rows only, with a deterministic finite threshold set.

    Every possible binary partition is represented. Observed margins, their next
    representable float above, and zero also represent the closest-to-zero
    threshold within each partition, avoiding arbitrary midpoint tie choices.
    """
    if not rows or any(not isinstance(r, dict) or set(r) != {'id', 'margin', 'gold'} or not isinstance(r['id'], str)
                       or not finite(r['margin']) or r['gold'] not in CANONICAL.values() for r in rows):
        raise ValueError('Expected development id/margin/gold rows')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate development ID')
    counts = {g: sum(r['gold'] == g for r in rows) for g in CANONICAL.values()}
    if not all(counts.values()):
        raise ValueError('Development threshold fitting requires both classes')
    values = sorted({float(r['margin']) for r in rows})
    thresholds = {0., *values, *(math.nextafter(v, math.inf) for v in values)}
    thresholds = sorted(t for t in thresholds if math.isfinite(t))
    scored = []
    for threshold in thresholds:
        predictions = [predict(r['margin'], threshold) for r in rows]
        correct = sum(p == r['gold'] for r, p in zip(rows, predictions))
        fp = sum(r['gold'] == 'SUPPORTED' and p == 'REFUTED' for r, p in zip(rows, predictions))
        fn = sum(r['gold'] == 'REFUTED' and p == 'SUPPORTED' for r, p in zip(rows, predictions))
        fpr, fnr = fp/counts['SUPPORTED'], fn/counts['REFUTED']
        scored.append({'threshold': threshold, 'correct': correct, 'samples': len(rows),
                       'accuracy': correct/len(rows), 'false_positive_rate': fpr,
                       'false_negative_rate': fnr, 'balanced_error': (fpr+fnr)/2})
    best = min(scored, key=lambda r: (-r['correct'], r['false_positive_rate'],
               r['balanced_error'], abs(r['threshold']), r['threshold']))
    return {**best, 'candidate_thresholds_evaluated': len(scored),
            'rule': 'Maximize development accuracy; ties lower FPR, lower balanced error, lower absolute threshold, then lower threshold.',
            'prediction_rule': 'SUPPORTED iff TRUE-minus-FALSE complete-sequence loglikelihood margin >= threshold; otherwise REFUTED.'}


def encode_candidates(tokenizer, system, statement, terminator_id):
    """Require the chat prefix to survive concatenation with each literal label."""
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError('Statement must be nonempty')
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': json.dumps({'statement': statement}, ensure_ascii=False)}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if tokenizer.encode(text, add_special_tokens=False) != prompt_ids or not prompt_ids:
        raise ValueError('Chat-template text/token encodings differ')
    candidates = {}
    specials = set(tokenizer.all_special_ids)
    if type(terminator_id) is not int or terminator_id not in specials:
        raise ValueError('Terminator must be a tokenizer special token')
    for label in CANONICAL:
        joined = tokenizer.encode(text+label, add_special_tokens=False)
        if joined[:len(prompt_ids)] != prompt_ids:
            raise ValueError('Candidate retokenized the chat prefix boundary')
        suffix = joined[len(prompt_ids):]
        if not suffix or any(token in specials for token in suffix) or tokenizer.decode(suffix, skip_special_tokens=False) != label:
            raise ValueError('Candidate suffix does not decode to the complete label')
        candidates[label] = suffix + [terminator_id]
    return messages, prompt_ids, candidates


def score_candidates(model, prompt_ids, candidates, terminator_id):
    """Two forwards, projecting only the candidate prediction positions."""
    import torch
    if getattr(model, 'training', False):
        raise ValueError('Scoring requires eval mode')
    if not prompt_ids or any(type(i) is not int or i < 0 for i in prompt_ids):
        raise ValueError('Invalid prompt IDs')
    if set(candidates) != set(CANONICAL) or type(terminator_id) is not int or terminator_id < 0:
        raise ValueError('Expected TRUE/FALSE candidates and a valid terminator')
    sequences = {name: list(ids) for name, ids in candidates.items()}
    for ids in sequences.values():
        if (len(ids) < 2 or any(type(i) is not int or i < 0 for i in ids) or
                ids[-1] != terminator_id or terminator_id in ids[:-1]):
            raise ValueError('Candidate must have label tokens and exactly one final EOS')
    if sequences['TRUE'] == sequences['FALSE']:
        raise ValueError('Candidates must be distinct')
    device = model.get_input_embeddings().weight.device
    if device.type == 'meta':
        raise ValueError('Model embedding device is meta')
    rows = []
    with torch.inference_mode():
        for label in ('TRUE', 'FALSE'):
            ids = sequences[label]
            model_input = torch.tensor([list(prompt_ids)+ids[:-1]], dtype=torch.long, device=device)
            output = model(input_ids=model_input, attention_mask=torch.ones_like(model_input),
                           use_cache=False, logits_to_keep=len(ids))
            logits = output.logits[0].float()
            if logits.shape[0] != len(ids) or max(ids) >= logits.shape[-1]:
                raise ValueError('Unexpected candidate logits shape/vocabulary')
            logdist = torch.log_softmax(logits, dim=-1)
            targets = torch.tensor(ids, device=logits.device).unsqueeze(-1)
            logprobs = logdist.gather(-1, targets).squeeze(-1).cpu().tolist()
            entropies = (-(logdist.exp()*logdist).sum(dim=-1)).cpu().tolist()
            if not all(finite(v) for v in logprobs+entropies):
                raise ValueError('Non-finite candidate score')
            rows.append({'label': label, 'token_ids': ids, 'token_logprobs': logprobs,
                         'token_full_vocab_entropy_nats': entropies, 'sequence_logprob': sum(logprobs)})
    margin = rows[0]['sequence_logprob']-rows[1]['sequence_logprob']
    return {'method': 'teacher_forced_complete_binary_candidates', 'prompt_token_ids': list(prompt_ids),
            'terminator_id': terminator_id, 'candidates': rows, 'margin': margin,
            'score_definition': 'Sum of raw-logit token logprob for complete label plus EOS, with no length normalization. Not an actual greedy-generation score or a truth probability.',
            'forward_calls': 2, 'logits_projection': 'Candidate prediction positions only; full prompt still passes through transformer.'}
