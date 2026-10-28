"""Character-to-token alignment for an already-rendered, single chat prompt.

Offsets are Python string indices (Unicode code points), with exclusive ends.
Never mix offsets from independently tokenized source text with model input IDs.
"""


def locate_unique_source(prompt, source):
    """Locate source in the rendered prompt, rejecting absent or repeated text."""
    if not source:
        raise ValueError('source must be nonempty')
    positions = []
    cursor = 0
    while True:
        pos = prompt.find(source, cursor)
        if pos < 0:
            break
        positions.append(pos)
        cursor = pos + 1  # Include overlapping occurrences.
    if not positions:
        return {'status': 'not_found', 'occurrences': []}
    if len(positions) != 1:
        return {'status': 'ambiguous', 'occurrences': positions}
    return {'status': 'ok', 'start': positions[0], 'end': positions[0] + len(source),
            'occurrences': positions}


def encode_prompt_offsets(tokenizer, prompt, expected_input_ids):
    """Tokenize the rendered prompt and require exact identity with model input IDs."""
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('a Hugging Face fast tokenizer is required')
    actual = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True,
                       return_special_tokens_mask=True, truncation=False)
    expected = expected_input_ids.tolist() if hasattr(expected_input_ids, 'tolist') else expected_input_ids
    if len(expected) == 1 and isinstance(expected[0], (list, tuple)):
        expected = expected[0]
    if list(actual['input_ids']) != list(expected):
        raise ValueError('rendered-prompt token IDs differ from actual model input IDs')
    return {'input_ids': list(actual['input_ids']),
            'offset_mapping': [list(o) for o in actual['offset_mapping']],
            'special_tokens_mask': list(actual['special_tokens_mask'])}


def map_char_span(text, start, end, offset_mapping, special_tokens_mask=None):
    """Map a nonempty character span to a contiguous, half-open token interval.

    A result is `ok` only when offsets cover every requested character and both
    character boundaries coincide with token boundaries. Other outcomes retain
    diagnostic token positions but must not be treated as exact mappings.
    Special tokens and zero-width offsets are never selected.
    """
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
        raise ValueError('span must be a nonempty, in-bounds Python character interval')
    offsets = [tuple(pair) for pair in offset_mapping]
    if special_tokens_mask is None:
        special_tokens_mask = [0] * len(offsets)
    if len(special_tokens_mask) != len(offsets):
        raise ValueError('offset mapping and special-token mask lengths differ')
    for pair in offsets:
        if len(pair) != 2 or not all(isinstance(x, int) for x in pair):
            raise ValueError('offsets must be integer pairs')
        if not 0 <= pair[0] <= pair[1] <= len(text):
            raise ValueError('token offset outside the rendered text')
    hits = [i for i, ((a, b), special) in enumerate(zip(offsets, special_tokens_mask))
            if not special and a < b and a < end and b > start]
    result = {'status': None, 'char_start': start, 'char_end': end, 'text': text[start:end],
              'token_start': None, 'token_end': None, 'token_indices': hits,
              'covered_char_start': None, 'covered_char_end': None}
    if not hits:
        result['status'] = 'uncovered'
        return result
    result.update(token_start=hits[0], token_end=hits[-1] + 1,
                  covered_char_start=min(offsets[i][0] for i in hits),
                  covered_char_end=max(offsets[i][1] for i in hits))
    if hits != list(range(hits[0], hits[-1] + 1)):
        result['status'] = 'non_contiguous'
        return result
    # Offsets may overlap (including equal offsets for UTF-8 byte fallback tokens).
    covered_until = start
    for a, b in sorted(offsets[i] for i in hits):
        if a > covered_until:
            break
        covered_until = max(covered_until, b)
    if covered_until < end:
        result['status'] = 'uncovered'
    elif result['covered_char_start'] != start or result['covered_char_end'] != end:
        result['status'] = 'partial_token'
    else:
        result['status'] = 'ok'
    return result
