"""Export separate accuracy/error/cost charts from audited experiment metrics.

No model is loaded. Existing output directories are refused, including partially
written ones. Supplied-evidence arms are visibly separated from closed-book arms.
"""
import argparse
import hashlib
import json
import math
import textwrap
from pathlib import Path

GOLD = ('REFUTED', 'SUPPORTED')
PREDICTIONS = ('SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO', 'INVALID')
RATE_DEFINITIONS = {
    'accuracy': ('Whole-sample accuracy', 'Higher is better; abstentions and invalid outputs count as incorrect.'),
    'missed_error_rate_including_abstentions': (
        'Missed errors including abstentions',
        'Lower is better; denominator is all gold REFUTED samples. Includes NEI and invalid outputs.'),
    'false_positive_rate': ('False-positive rate', 'Lower is better; denominator is all gold SUPPORTED samples.')}
EVIDENCE_CAPTION = (
    'Purple / hatched: supplied curated evidence and context. Separate input condition; '
    'not end-to-end retrieval performance.')
LABELS = {
    'legacy_direct': 'Legacy direct', 'legacy_dual': 'Legacy dual',
    'neutral': 'Neutral judge', 'reasoned': 'Reasoned judge',
    'structured': 'Structured review', 'evidence': 'Supplied evidence',
    'calibrated': 'Dev-calibrated decision'}


def evidence_arm(name, value):
    condition = str(value.get('condition', value.get('input_condition', ''))).casefold()
    return 'evidence' in name.casefold() or 'evidence' in condition


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def label(name):
    return LABELS.get(name, name.replace('_', ' '))


def validate(summary):
    arms = summary.get('arms')
    if not isinstance(arms, dict) or not 1 <= len(arms) <= 5:
        raise ValueError('Expected between one and five named experiment arms')
    sample_counts, gold_counts = set(), set()
    for name, value in arms.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(value, dict):
            raise ValueError('Arms must be named objects')
        metrics = value.get('metrics', {})
        total = metrics.get('total_samples')
        if type(total) is not int or total < 1:
            raise ValueError(f'{name}: positive sample count required')
        sample_counts.add(total)
        for key in RATE_DEFINITIONS:
            rate = metrics.get(key)
            if key not in metrics or (rate is not None and (not finite(rate) or not 0 <= rate <= 1)):
                raise ValueError(f'{name}: invalid or missing {key}')
        matrix = metrics.get('confusion_matrix', {})
        if set(matrix) != set(GOLD):
            raise ValueError(f'{name}: expected two gold-label rows')
        counts = []
        for gold in GOLD:
            row = matrix[gold]
            if (set(row) != set(PREDICTIONS) or
                    any(type(v) is not int or v < 0 for v in row.values())):
                raise ValueError(f'{name}: invalid four-class confusion row')
            counts.append(sum(row.values()))
        if sum(counts) != total:
            raise ValueError(f'{name}: confusion matrix and sample count differ')
        gold_counts.add(tuple(counts))
        if metrics.get('gold_counts') != dict(zip(GOLD, counts)):
            raise ValueError(f'{name}: confusion matrix and gold counts differ')
        expected = {
            'accuracy': (matrix['REFUTED']['REFUTED'] + matrix['SUPPORTED']['SUPPORTED'], total),
            'missed_error_rate_including_abstentions': (
                counts[0] - matrix['REFUTED']['REFUTED'], counts[0]),
            'false_positive_rate': (matrix['SUPPORTED']['REFUTED'], counts[1])}
        for key, (numerator, denominator) in expected.items():
            rate = metrics[key]
            if (not denominator and rate is not None) or (denominator and
                    (rate is None or not math.isclose(rate, numerator / denominator, abs_tol=1e-10))):
                raise ValueError(f'{name}: {key} does not reproduce confusion counts')
        elapsed = value.get('cost', {}).get('elapsed_seconds')
        if elapsed is not None and (not finite(elapsed) or elapsed < 0):
            raise ValueError(f'{name}: invalid elapsed time')
    if len(sample_counts) != 1 or len(gold_counts) != 1:
        raise ValueError('Arms must use the same sample count and class denominators')
    return arms


def plot(summary, output_dir, source_bytes, source_name):
    arms = validate(summary)
    if output_dir.exists():
        raise ValueError('Output directory exists; preserve prior plots and choose a new directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'minifact-accuracy-comparison-v1'})
    output_dir.mkdir(parents=True, exist_ok=False)
    names = list(arms)
    has_evidence = any(evidence_arm(name, arms[name]) for name in names)
    total = arms[names[0]]['metrics']['total_samples']
    files = []

    def save(fig, stem):
        for extension in ('png', 'svg'):
            path = output_dir / f'{stem}.{extension}'
            metadata = {'Software': 'MiniFact-LLM'} if extension == 'png' else {
                'Date': None, 'Creator': 'MiniFact-LLM'}
            fig.savefig(path, dpi=180, bbox_inches='tight', metadata=metadata)
            files.append(path.name)
        plt.close(fig)

    def finish(fig, explanation):
        caption = explanation + ('\n' + EVIDENCE_CAPTION if has_evidence else '')
        fig.text(0.02, 0.015, caption, fontsize=8, va='bottom')
        fig.tight_layout(rect=(0, 0.12 if has_evidence else 0.085, 1, 1))

    def color(name, error=False):
        if evidence_arm(name, arms[name]):
            return '#8054a3'
        if name.startswith('legacy'):
            return ('#a1a8b0' if 'direct' in name else '#66717c') if not error else '#d0a0a0'
        if 'calibrat' in name and not error:
            return '#238573'
        return '#c43d43' if error else '#2471a3'

    for key, (title, explanation) in RATE_DEFINITIONS.items():
        fig, ax = plt.subplots(figsize=(max(7.7, 1.65 * len(names)), 4.7))
        values = [arms[name]['metrics'][key] for name in names]
        bars = ax.bar(range(len(names)), [v if v is not None else 0 for v in values],
                      color=[color(name, error=key != 'accuracy') for name in names],
                      edgecolor='#333333', linewidth=0.65, width=0.66)
        for name, bar, value in zip(names, bars, values):
            if evidence_arm(name, arms[name]):
                bar.set_hatch('///')
            if value is None:
                bar.set_facecolor('none')
                ax.text(bar.get_x() + bar.get_width()/2, 0.02, 'undefined', ha='center')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, min(value + 0.02, 0.985),
                        f'{value:.1%}', ha='center', va='bottom' if value < 0.96 else 'top')
        ax.set_xticks(range(len(names)), ['\n'.join(textwrap.wrap(label(n), width=19)) for n in names])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel('Rate')
        ax.set_title(f'{title} (N={total})', pad=14)
        ax.yaxis.grid(True, alpha=0.22)
        ax.set_axisbelow(True)
        finish(fig, explanation)
        save(fig, key)

    columns = min(2, len(names))
    rows = math.ceil(len(names)/columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.3*columns, 3.15*rows), squeeze=False)
    maximum = max(arms[n]['metrics']['confusion_matrix'][g][p]
                  for n in names for g in GOLD for p in PREDICTIONS)
    for index, (name, ax) in enumerate(zip(names, axes.flat)):
        matrix = arms[name]['metrics']['confusion_matrix']
        array = [[matrix[g][p] for p in PREDICTIONS] for g in GOLD]
        ax.imshow(array, cmap='Purples' if evidence_arm(name, arms[name]) else 'Blues',
                  norm=Normalize(0, max(maximum, 1)), aspect='auto')
        for i, row in enumerate(array):
            for j, count in enumerate(row):
                ax.text(j, i, str(count), ha='center', va='center',
                        color='white' if count > maximum*0.55 else '#222222')
        ax.set_xticks(range(4), ['SUPPORTED', 'REFUTED', 'NEI', 'INVALID'], fontsize=8)
        ax.set_yticks(range(2), GOLD, fontsize=8)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('Upstream gold')
        condition = ' [supplied evidence]' if evidence_arm(name, arms[name]) else ''
        ax.set_title(label(name) + condition, fontsize=11)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.suptitle(f'Whole-sample confusion counts (N={total}; shared count scale)', fontsize=13)
    finish(fig, 'Counts, not row-normalized percentages. NEI is abstention; INVALID is an invalid output.')
    save(fig, 'confusion_matrices')

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    missing_cost, elapsed_values = [], []
    for name in names:
        elapsed = arms[name].get('cost', {}).get('elapsed_seconds')
        accuracy = arms[name]['metrics']['accuracy']
        if elapsed is None or accuracy is None:
            missing_cost.append(name)
            continue
        evidence = evidence_arm(name, arms[name])
        elapsed_values.append(elapsed)
        ax.scatter(elapsed, accuracy, marker='D' if evidence else 'o',
                   s=95, color=color(name), edgecolor='#333333', linewidth=0.6,
                   label=f'{label(name)}: {elapsed:.1f}s, {accuracy:.1%}' +
                   (' [supplied evidence]' if evidence else ''))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(elapsed_values, default=1)*1.12 or 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('Recorded total inference time (seconds)')
    ax.set_ylabel('Whole-sample accuracy')
    ax.set_title(f'Observed cost and accuracy (N={total})', pad=13)
    ax.grid(alpha=0.22)
    if ax.collections:
        ax.legend(loc='lower right', fontsize=8)
    explanation = 'Elapsed time is a recorded run observation, not an isolated hardware benchmark.'
    if missing_cost:
        explanation += '\nMissing time/accuracy; omitted from scatter: ' + ', '.join(missing_cost)
    finish(fig, explanation)
    save(fig, 'cost_vs_accuracy')

    # Preserve the values actually plotted, without personal absolute input paths.
    chart_data = {
        'metrics_file': source_name,
        'metrics_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'matplotlib_version': matplotlib.__version__, 'sample_count': total,
        'arm_order': names,
        'arms': {name: {'label': label(name), 'supplied_evidence': evidence_arm(name, arms[name]),
                        'metrics': arms[name]['metrics'], 'cost': arms[name].get('cost', {})}
                 for name in names},
        'rate_definitions': {key: description for key, (_, description) in RATE_DEFINITIONS.items()},
        'input_condition_note': EVIDENCE_CAPTION if has_evidence else 'All named arms use the conditions recorded in source metrics.',
        'missing_cost_or_accuracy_for_scatter': missing_cost,
        'exports': files,
        'export_sha256': {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in files}}
    (output_dir / 'chart_data.json').write_bytes(
        (json.dumps(chart_data, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    return chart_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    raw = args.metrics.read_bytes()
    output = plot(json.loads(raw), args.output_dir, raw, args.metrics.name)
    print(json.dumps({'files': output['exports'], 'input_sha256': output['metrics_sha256'],
                      'arms': output['arm_order']}, indent=2))


if __name__ == '__main__':
    main()
