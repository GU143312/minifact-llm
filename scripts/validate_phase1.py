"""Validate the formal dataset and its label-free export, without a model."""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, FormatChecker

try:
    from .curate_phase1 import (DATA, canonical_article, input_rows, normalized_text,
                               record_hash, rejection, restore_upstream, statistics)
except ImportError:
    from curate_phase1 import (DATA, canonical_article, input_rows, normalized_text,
                              record_hash, rejection, restore_upstream, statistics)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(dataset, inputs, config, schema):
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(dataset)
    samples = dataset['samples']
    require(dataset['source'] == config['source'], 'Pinned upstream metadata mismatch')
    for key in ['dataset_id', 'version', 'curation_date']:
        require(dataset[key] == config[key], f'{key} mismatch')
    require(len(samples) == config['sample_count'], 'Sample count mismatch')
    require([s['id'] for s in samples] == [f'MF-P1-{i:04d}' for i in range(1, len(samples) + 1)],
            'IDs must be unique and sequential')
    require([s['provenance']['source_record_index'] for s in samples] == config['final_source_indices'],
            'Frozen source indices mismatch')
    require(len({normalized_text(s['text']) for s in samples}) == len(samples), 'Duplicate text')
    articles, domains, speakers = [], Counter(), Counter()
    for s in samples:
        p = s['provenance']
        row = restore_upstream(s)
        require(record_hash(row) == p['source_record_sha256'], f'{s["id"]}: upstream record hash mismatch')
        require(rejection(p['source_record_index'], row, config) is None, f'{s["id"]}: ineligible source')
        require(str(p['source_record_index']) not in config['final_review_exclusions'], 'Review exclusion selected')
        date = datetime.strptime(p['claim_date'], '%d-%m-%Y').date().isoformat()
        require(date == s['claim_date'], f'{s["id"]}: normalized date mismatch')
        article = canonical_article(p['fact_checking_article'])
        require(urlsplit(article).scheme in ['https', 'http'] and urlsplit(article).hostname,
                'Invalid fact-checking article URL')
        articles.append(article)
        domains[urlsplit(article).hostname] += 1
        speaker = (s['context']['speaker'] or '').strip().casefold()
        if speaker:
            speakers[speaker] += 1
    require(len(set(articles)) == len(articles), 'Duplicate fact-checking article')
    require(max(domains.values()) <= config['selection']['max_per_factchecker_domain'], 'Domain concentration')
    require(not speakers or max(speakers.values()) <= config['selection']['max_per_named_speaker'], 'Speaker concentration')
    require(inputs == input_rows(dataset), 'Label-free inputs mismatch or gold leakage')
    require(dataset['statistics'] == statistics(samples), 'Reported statistics mismatch')
    audit = dataset['selection_audit']
    require(audit['candidate_count'] == config['candidate_count'], 'Candidate count mismatch')
    candidates = audit['candidate_source_indices']
    require(len(set(candidates)) == len(candidates) == audit['candidate_count'], 'Candidate IDs mismatch')
    retained = sorted(i for i in candidates if str(i) not in config['final_review_exclusions'])
    require(retained == config['final_source_indices'], 'Candidate-to-final audit mismatch')
    require(audit['final_review_excluded_count'] == len(candidates) - len(samples), 'Review count mismatch')
    require(sum(audit['source_labels'].values()) == audit['source_records'] == 500, 'Source count mismatch')
    require(sum(audit['filter_rejection_counts'].values()) + audit['eligible_before_candidate_selection']
            == audit['source_records'], 'Filter audit mismatch')
    return dataset['statistics']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    args = parser.parse_args()
    def read(name):
        return json.loads((args.data_dir / name).read_text(encoding='utf-8'))
    stats = validate(read('dataset.json'), read('inputs.json'), read('curation.json'),
                     read('dataset.schema.json'))
    print(f'PASS: {stats["sample_count"]} samples; {stats["factchecker_domain_count"]} fact-checker domains; '
          f'{stats["question_count"]} questions; {stats["answer_count"]} answers; no input/gold leakage.')


if __name__ == '__main__':
    main()
