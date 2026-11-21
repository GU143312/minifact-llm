"""Versioned protocol, byte contracts and non-overwriting artifact helpers."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('A', 'B', 'C', 'D', 'E', 'F')
PROTOCOL = {
    'version': 'minifact-next-evaluation-v1', 'seed': 42,
    'news_count': 200, 'per_class': 100, 'semantic_count': 60,
    'boundary_count': 30, 'primary_comparison': ['D', 'F'],
    'bootstrap_repetitions': 10000,
    'arms': {'A': 'frozen_closed_book_threshold', 'B': 'text_direct',
             'C': 'context_direct', 'D': 'context_qa_direct',
             'E': 'context_qa_one_round', 'F': 'context_qa_up_to_two_rounds'},
    'caps': {'decision': 16, 'extract': 256, 'feedback': 64},
    'upgrade': {'accuracy': .80, 'accuracy_gain': .05, 'strict_fnr': .10,
                'missed_error_rate': .15, 'fpr': .25, 'risk_increase': .02,
                'fact_recall': .90, 'atomicity': .95, 'faithfulness': .95,
                'qualifier_retention': .95, 'accepted_span_precision': 1.0},
    'semantic_matching': 'Human alignment; distinct reference facts count once; omitted facts remain in recall denominator.',
    'uncertainty': 'Scores are not truth probabilities. Development quantiles only; never fit on news test outcomes.',
    'information': 'No model receives IDs, gold, curator justification, URL strings or reviewer annotations.',
}


def read(path):
    return json.loads(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(encoded(value))


def checkpoint(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as handle:
        handle.write(encoded(value))
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_contract():
    files = sorted([*ROOT.joinpath('next_eval').glob('*.py'), ROOT/'scripts/next_eval.py'])
    files += [ROOT/p for p in ('config.json', 'minifact.py', 'accuracy_backend.py',
        'accuracy_claims.py', 'accuracy_calibration.py', 'phase2_backend.py',
        'phase2_pipeline.py', 'phase3_experiment.py', 'phase3_metrics.py',
        'scripts/check_accuracy.py', 'scripts/check_calibrated_accuracy.py',
        'scripts/run_accuracy_calibration.py', 'scripts/build_improvement_data.py',
        'scripts/curate_phase1.py', 'scripts/audit_accuracy.py',
        'scripts/audit_phase2_run.py', 'scripts/run_phase2.py',
        'scripts/run_accuracy.py', 'scripts/summarize_accuracy.py',
        'scripts/run_phase3.py', 'scripts/evaluate_phase3.py',
        'scripts/audit_accuracy_calibration.py', 'docs/next_evaluation.schema.json')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def check_sources(hashes):
    require(hashes == source_contract(), 'Frozen source/configuration differs from current files')


def validate_inputs(rows):
    from scripts.check_accuracy import validate_evidence_packet
    require(isinstance(rows, list) and bool(rows), 'Expected nonempty input array')
    seen = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'id', 'text', 'context', 'evidence'}, 'Unexpected input fields')
        require(isinstance(row['id'], str) and row['id'].strip() and row['id'] not in seen, 'Invalid/duplicate ID')
        require(isinstance(row['text'], str) and row['text'].strip(), 'Empty statement')
        validate_evidence_packet({k: row[k] for k in ('context', 'evidence')})
        seen.add(row['id'])
