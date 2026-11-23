"""Publish a separate boundary report after complete, tokenizer-audited execution."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from next_eval.common import read, require, sha, write_new
from next_eval.metrics import summarize


def publish(run_path, metrics_path, audit_path, output):
    require(not output.exists(), 'Choose a new output directory')
    run, metrics, audited = read(run_path), read(metrics_path), read(audit_path)
    require(run['stage'] == 'boundary' and run['completed'], 'Need complete boundary execution')
    freeze_path = ROOT / run['freeze_file']
    require(audited.get('passed') is True and audited.get('completed') is True
            and audited.get('tokenizer_replayed') is True, 'Need full tokenizer audit')
    require(audited['run_sha256'] == sha(run_path)
            and audited['freeze_sha256'] == sha(freeze_path), 'Audit does not bind this run')
    require(metrics == summarize(run, read(freeze_path), freeze_path), 'Metrics differ from audited reconstruction')

    # This presentation tool is separate from the already-frozen inference
    # protocol. Its own bytes and every exported artifact are bound below.
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import matplotlib
    import matplotlib.pyplot as plt
    output.mkdir(parents=True)
    arms = ('D', 'E', 'F')
    families = list(metrics['boundary']['D']['families'])
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for i, (arm, color) in enumerate(zip(arms, ('#8855aa', '#aa77cc', '#552277'))):
        data = metrics['boundary'][arm]['families']
        y = [j + (i-1)*.23 for j in range(len(families))]
        values = [data[f]['correct']/data[f]['total']*100 for f in families]
        ax.barh(y, values, height=.22, label=arm, color=color)
        for pos, f, value in zip(y, families, values):
            ax.text(min(94, value+1), pos, f"{data[f]['correct']}/{data[f]['total']}",
                    va='center', fontsize=8)
    ax.set_yticks(range(len(families)), families)
    ax.set(xlim=(0, 106), xlabel='Agreement with authored packet-relative expectations (%)',
           title='Fictional boundary checks — 5 cases per family, identical supplied QA')
    ax.legend(title='Condition')
    artifacts = []
    for ext in ('png', 'svg'):
        name = 'boundary_families.' + ext
        fig.savefig(output/name, dpi=160)
        artifacts.append(name)
    plt.close(fig)

    lines = ['# 30 条合成边界检查', '',
             '这些是按虚构证据包制定的预期，不是新闻金标准；不并入正式新闻准确率。D/E/F 使用相同证据。', '',
             '| 条件 | 正确/30 | 证据不足正确/6 | 接受引用 | 拒绝项 | 无接受项样本 | 平均秒 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        m = metrics['boundary'][arm]
        c = m['cost_and_extraction']
        values = [str(c.get(k, 0)) if arm != 'D' else '不适用'
                  for k in ('accepted_quotes', 'rejected_items', 'samples_without_accepted_claim')]
        lines.append(f"| {arm} | {m['correct']}/30 | {m['insufficient_evidence_correct']}/6 | "
                     + ' | '.join(values) + f" | {c['mean_elapsed_seconds']:.3f} |")
    lines += ['', '![按句型分项](boundary_families.png)', '',
              '| 句型 | D 正确/5 | E 正确/5 | F 正确/5 |', '|---|---:|---:|---:|']
    for family in families:
        lines.append('| '+family+' | '+' | '.join(
            str(metrics['boundary'][a]['families'][family]['correct'])+'/5' for a in arms)+' |')
    lines += ['', '每类只有 5 条，不能据此估计真实新闻上的稳定泛化表现。定位、解析拒绝及证据不足应分别解释。', '',
              '## 逐条结果', '', '| ID | 作者预期 | D | E | F |', '|---|---|---|---|---|']
    expected = {s['id']: s['gold'] for s in read(ROOT/read(freeze_path)['gold_file'])['samples']}
    for row in run['results']:
        lines.append('| '+row['id']+' | '+expected[row['id']]+' | '+' | '.join(
            row['arms'][a]['verdict'] or 'INVALID' for a in arms)+' |')
    lines += ['', '本报告没有填写人工语义召回、原子性或限定保留分数。接受引用的逐字匹配不能替代这些指标。', '',
              f'指标 SHA256：`{sha(metrics_path)}`。原始运行 SHA256：`{sha(run_path)}`。',
              '完整提示、原始输出、解析错误和计时保存在对应运行 JSON；审计包含 tokenizer 重放，未重新计算全部模型 logits。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    artifacts.append('REPORT.md')
    write_new(output/'chart_data.json', {
        'run_sha256': sha(run_path), 'metrics_sha256': sha(metrics_path),
        'audit_sha256': sha(audit_path), 'reporter_sha256': sha(__file__),
        'matplotlib_version': matplotlib.__version__, 'data': metrics['boundary'],
        'artifacts_sha256': {name: sha(output/name) for name in artifacts}})
    return output/'REPORT.md'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'metrics', 'audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(publish(args.run, args.metrics, args.audit, args.output))


if __name__ == '__main__':
    main()
