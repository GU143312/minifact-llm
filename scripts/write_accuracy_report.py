"""Render the completed, hash-bound accuracy comparison as a Chinese report.

This script reads saved records only. It neither runs models nor selects a new
method from holdout outcomes. An existing report is never overwritten.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ARTIFACTS = {
    'run': 'reports/accuracy_holdout_run_20261112.json',
    'main_metrics': 'reports/accuracy_holdout_metrics_20261112.json',
    'calibration_run': 'reports/accuracy_calibration_holdout_20261112.json',
    'calibration': 'reports/accuracy_label_threshold_20261112.json',
    'holdout_gold': 'data/improvement/holdout_dataset.json',
    'summary_script': 'scripts/summarize_accuracy_comparison.py',
}
SUPPLEMENTARY = (
    'reports/accuracy_dev_freeze_v2_20261112.json',
    'reports/accuracy_dev_run_v2_20261112.json',
    'reports/accuracy_dev_metrics_v2_20261112.json',
    'reports/accuracy_dev_audit_v2_20261112.json',
    'reports/accuracy_selection_20261112.json',
    'reports/accuracy_holdout_freeze_20261112.json',
    'reports/accuracy_holdout_audit_20261112.json',
    'reports/accuracy_calibration_dev_20261112.freeze.json',
    'reports/accuracy_calibration_dev_20261112.json',
    'reports/accuracy_calibration_dev_audit_20261112.json',
    'reports/accuracy_calibration_holdout_20261112.freeze.json',
    'reports/accuracy_calibration_holdout_audit_20261112.json',
)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percent(value):
    return '未定义' if value is None else f'{100*value:.2f}%'


def number(value, digits=2):
    return '未定义' if value is None else f'{value:.{digits}f}'


def check_metrics(metrics, samples, count_per_class):
    require(metrics.get('total_samples') == samples, 'Metric sample count differs from completed split')
    require(metrics.get('gold_counts') == {'SUPPORTED': count_per_class, 'REFUTED': count_per_class},
            'Expected the frozen balanced sample classes')
    confusion = metrics['confusion_matrix']
    require(all(sum(confusion[label].values()) == count_per_class for label in ('SUPPORTED', 'REFUTED')),
            'Confusion matrix denominator mismatch')
    correct = sum(confusion[label][label] for label in ('SUPPORTED', 'REFUTED'))
    require(correct == metrics['correct_samples'] and math.isclose(correct/samples, metrics['accuracy']),
            'Correct-count/accuracy mismatch')


def load_bundle(metrics_path, root=ROOT):
    """Refuse a development-only, incomplete or substituted holdout summary."""
    metrics = read(metrics_path)
    require(metrics.get('split') == 'holdout', 'A completed holdout comparison is required')
    artifacts = metrics.get('artifact_sha256', {})
    for key, relative in CANONICAL_ARTIFACTS.items():
        require(sha(root / relative) == artifacts.get(key), f'Artifact bytes differ: {relative}')
    run = read(root / CANONICAL_ARTIFACTS['run'])
    scored = read(root / CANONICAL_ARTIFACTS['calibration_run'])
    calibration = read(root / CANONICAL_ARTIFACTS['calibration'])
    require(run.get('completed') is True and run['contract']['split'] == 'holdout' and
            scored.get('completed') is True and scored.get('stage') == 'holdout',
            'Holdout model records are incomplete')
    require(len(run['results']) == len(scored['results']) == 80,
            'Both holdout conditions must contain the frozen 80 cases')
    require([(r['id'], r['source']) for r in run['results']] ==
            [(r['id'], r['source']) for r in scored['results']],
            'The two holdout runs do not share identical ordered input text')
    require(scored.get('calibration_sha256') == sha(root / CANONICAL_ARTIFACTS['calibration']),
            'Candidate-scoring calibration binding differs')
    require(scored.get('threshold') == calibration['fit']['threshold'] == metrics['calibration']['threshold'],
            'The report threshold differs from its saved fit')
    selection_path = root / 'reports/accuracy_selection_20261112.json'
    selection = read(selection_path)
    selected = selection['selected_closed_book']
    require(run['contract'].get('selection_sha256') == sha(selection_path),
            'Development selection differs from holdout protocol')
    expected_arms = {'legacy_direct', 'legacy_dual', selected, 'evidence', 'calibrated'}
    require(set(metrics['arms']) == expected_arms, 'Unexpected final holdout conditions')
    for item in metrics['arms'].values():
        check_metrics(item['metrics'], 80, 40)
    check_metrics(metrics['uncalibrated_candidate_reference']['metrics'], 80, 40)
    require(metrics['uncalibrated_candidate_reference']['threshold'] == 0.,
            'Uncalibrated candidate reference must use zero margin')
    dev_path = root / 'reports/accuracy_dev_metrics_v2_20261112.json'
    development = read(dev_path)
    require(development.get('split') == 'dev' and len(development['arms']) == 6,
            'Expected all six completed development conditions')
    for item in development['arms'].values():
        check_metrics(item['metrics'], 32, 16)
    require(sha(dev_path) == selection['development_metrics_sha256'],
            'Development metric bytes changed after selection')
    require(development['run_sha256'] == selection['development_run_sha256'] ==
            sha(root / selection['development_run']), 'Development output selection binding differs')
    dev_score_path = root / calibration['development_run']
    require(dev_score_path.resolve().is_relative_to(root.resolve()) and
            sha(dev_score_path) == calibration['development_run_sha256'],
            'Calibration development scoring record differs')
    dev_scored = read(dev_score_path)
    require(dev_scored.get('completed') is True and dev_scored.get('stage') == 'dev' and
            dev_scored.get('development_fit') == calibration['fit'] and len(dev_scored['results']) == 32,
            'Calibration is not bound to a completed 32-case development fit')
    dev_gold_path = root / 'data/improvement/dev_dataset.json'
    require(sha(dev_gold_path) == calibration['development_gold_sha256'], 'Development labels changed')
    gold = {r['id']: r['gold']['verdict'] for r in read(dev_gold_path)['samples']}
    require(set(gold) == {r['id'] for r in dev_scored['results']}, 'Calibration development IDs differ')
    zero_correct = sum(('SUPPORTED' if r['margin'] >= 0 else 'REFUTED') == gold[r['id']]
                       for r in dev_scored['results'])
    return metrics, development, calibration, selected, zero_correct


def render(metrics, development, calibration, selected, zero_correct, output, metrics_path,
           tests_passed=None, root=ROOT):
    def link(relative, label=None):
        path = root / relative
        target = os.path.relpath(path, output.parent).replace(os.sep, '/')
        return f'[{label or path.name}]({target})'

    names = {'legacy_direct': '旧直接提示', 'legacy_dual': '旧双角色流程',
             'neutral': '中性三分类提示', 'binary': '强制二分类提示',
             'reasoned': '先分析后判决', 'structured': '关系/上下文抽取 + 二分类',
             'evidence': '提供参考 QA 与元数据', 'calibrated': '开发集拟合标签阈值'}
    chosen = metrics['arms'][selected]['metrics']
    calibrated = metrics['arms']['calibrated']['metrics']
    legacy = metrics['arms']['legacy_dual']['metrics']
    lines = [
        '# 准确率改进：开发集选择与 80 条留出评估', '',
        '模型仍为固定 revision 的 Qwen2.5-3B-Instruct，FP16、eager attention；没有更换权重或微调。原有 110 条 Week 6 数据、代码及运行结果保留。这里另用 32 条开发集（16/16）和 80 条项目未见留出样本（40/40）。', '',
        f"新闭卷候选由开发集选为 `{selected}`，其留出准确率为 {chosen['correct_samples']}/80（{percent(chosen['accuracy'])}），相对同一留出集上的旧双角色差 {100*(chosen['accuracy']-legacy['accuracy']):+.2f} 个百分点。单独拟合的标签阈值方法为 {calibrated['correct_samples']}/80（{percent(calibrated['accuracy'])}），对应差 {100*(calibrated['accuracy']-legacy['accuracy']):+.2f} 个百分点。全部预定条件均报告，不能根据留出分数再更换候选或阈值。", '',
        '本表与旧 110 条成绩的分布和样本不同，不能直接比较两个绝对百分比后声称提升。有效比较来自下面在**同一 80 条**上重新运行的条件。全部判为同一个类别的常数参考是 40/80（50%）；它是算术背景，不是额外模型分支。', '',
        '## 留出结果与代价', '',
        '| 条件 / 方法 | 判对 / 80 | 准确率 | 严格 FNR | 含弃答未检出率 | FPR | 二分类覆盖 | 累计秒 | 模型调用 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    order = ['legacy_direct', 'legacy_dual', selected, 'calibrated', 'evidence']
    for arm in order:
        value = metrics['arms'][arm]
        m, cost = value['metrics'], value['cost']
        condition = ' **[提供证据]**' if arm == 'evidence' else ' [闭卷]'
        lines.append(f"| {names.get(arm, arm)}{condition} | {m['correct_samples']} | {percent(m['accuracy'])} | {percent(m['false_negative_rate'])} | {percent(m['missed_error_rate_including_abstentions'])} | {percent(m['false_positive_rate'])} | {percent(m['binary_coverage'])} | {number(cost['elapsed_seconds'])} | {cost['calls']} |")
    lines += [
        '', '正类是错误主张（REFUTED）。严格 FNR 只计错误被判 SUPPORTED / 40；含弃答未检出率还计 NOT_ENOUGH_INFO 与无效输出。FPR 是正常主张被判 REFUTED / 40。所有弃答和无效输出均保留在 80 条准确率的分母中。强制二分类方法请求模型作答，其较高覆盖不能单独当作更强的事实判断能力。', '',
        '调用数中，旧流程与新提示的单位为 `generate` 调用，标签阈值方法的单位为完整 teacher-forced 前向调用；它每条做两次前向，不生成文本。累计耗时保留实际观测，排除模型加载和检查点写入，不是严格隔离的硬件性能基准。', '',
        '**提供证据条件单独解释：**它比闭卷条件多收到经筛选的上游 QA、主张日期和国家/说话人元数据；模型不读取金标签、裁决解释或证据 URL 字符串。证据由 benchmark 预先提供，不是本系统检索获得，不能把这行的差异归因于提示词或当作端到端网络核查成绩。QA 本身可能直接解答主张，筛选与截断也影响结果。', '',
        '## 开发过程与冻结选择', '',
        '| 开发条件（32 条） | 判对 / 32 | 准确率 | 严格 FNR | FPR | 二分类覆盖 | 累计秒 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for arm, value in development['arms'].items():
        m = value['metrics']
        lines.append(f"| {names.get(arm, arm)} | {m['correct_samples']} | {percent(m['accuracy'])} | {percent(m['false_negative_rate'])} | {percent(m['false_positive_rate'])} | {percent(m['binary_coverage'])} | {number(value['cost']['elapsed_seconds'])} |")
    fit = calibration['fit']
    lines += [
        '', f"闭卷候选按开发集准确率选定为 `{selected}`；并列时依次比较较低 FPR、较短平均耗时和方法名。开发集用于选择，成绩带有选择偏差，不能代替留出结果。提供 QA 条件没有参与闭卷候选排名。", '',
        f"标签阈值是另一个在留出执行前固定的诊断方法。使用相同开发集，零阈值候选分数判对 {zero_correct}/32（{percent(zero_correct/32)}）；拟合后判对 {fit['correct']}/32（{percent(fit['accuracy'])}），阈值为 `{fit['threshold']:.12g}`。拟合时 FPR 为 {percent(fit['false_positive_rate'])}，严格 FNR 为 {percent(fit['false_negative_rate'])}。这显示降低正常类误报可能伴随大量错误被判真的代价，开发集收益并不保证泛化。", '',
        '阈值拟合只校正二分类决策边界：分别计算完整 `TRUE + EOS` 与 `FALSE + EOS` 的原始 token logprob 总和，再用两者之差与开发集阈值比较。分数不作长度归一化；这与实际 greedy 生成的标签 token 均值不同，也不是校准后的真假概率。没有为提高测试分数而在留出标签上拟合阈值。', '',
        '## 阈值零值消融与配对变化', '',
        '| 同一留出候选分数 | 准确率 | 严格 FNR | FPR |',
        '| --- | ---: | ---: | ---: |',
    ]
    zero = metrics['uncalibrated_candidate_reference']['metrics']
    for label, item in [('阈值 0（无新增推理）', zero), ('固定开发阈值', calibrated)]:
        lines.append(f"| {label} | {percent(item['accuracy'])} | {percent(item['false_negative_rate'])} | {percent(item['false_positive_rate'])} |")
    lines += ['', '零阈值参照使用同一批保存的候选分数，规则在首次留出运行前声明；不是看过留出结果后挑选的另一个最佳阈值。', '',
              '| 相对旧双角色 | 双方判对 | 双方未判对 | 仅旧流程判对 | 仅新条件判对 | 准确率差（百分点） |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm, paired in metrics['paired_vs_legacy_dual'].items():
        lines.append(f"| {names.get(arm, arm)} | {paired['both_correct']} | {paired['both_incorrect']} | {paired['baseline_only_correct']} | {paired['dual_only_correct']} | {100*paired['accuracy_difference_dual_minus_baseline']:+.2f} |")
    lines += ['', '配对数量完整保存在组合 JSON 中，描述同一批样本上的判对变化；不能隔离额外计算、原子抽取、反馈或分数的独立因果贡献，也不能使提供证据与闭卷成为同信息条件。', '',
              '## 抽取维度单独保留', '']
    extraction = metrics['structured_extraction']
    lines += [f"留出结构化条件记录 {extraction['accepted_quote_occurrences']} 次接受引用、{extraction['rejected_records']} 条项目拒绝记录。组合汇总逐一检查了接受引用的 `source[start_idx:end_idx] == claim`。跨项目/调用出现次数不是独立事实数，项目拒绝也不能概括所有顶层 JSON 解析错误。", '',
              '| 维度 | 可支持的结论 |', '| --- | --- |',
              '| 原文位置 | 接受的引用逐字匹配；重复歧义、改写等拒绝保留 |',
              '| 事实是否提全、是否独立 | 缺少独立原子金标准，未报告正式覆盖率或原子性准确率 |',
              '| 限定条件保留 | 未建立独立限定参考标注，未报告正式保留率 |',
              '| 主语/谓语补全 | 与逐字引用分字段；重构命题及启发式警报都未被视为事实证据 |', '',
              '警报计数：`' + json.dumps(extraction['guard_flags'], ensure_ascii=False, sort_keys=True) + '`。', '',
              '拒绝原因：`' + json.dumps(extraction['rejection_reasons'], ensure_ascii=False, sort_keys=True) + '`。', '',
              '启发式警报减少或全部位置匹配，都不能证明语义抽取更准确。原文始终提供给最终裁决器；已补写的命题没有冒充原文跨度。', '',
              '## 使用与复现', '',
              '需要原固定模型快照和 CUDA 环境，模型/缓存留在仓库外。以下均使用不同输出名，工具拒绝覆盖已有文件。第一个入口使用开发集选定的新提示方法；第二个是单独的候选分数阈值方法。', '',
              '```powershell',
              'python scripts/check_accuracy.py --text "<待核查主张>" --output runs/local_check.json',
              'python scripts/check_calibrated_accuracy.py --text "<待核查主张>" --output runs/local_threshold.json',
              'python scripts/check_accuracy.py --method evidence --input-file statement.txt --evidence-file packet.json --output runs/local_evidence.json',
              '```', '',
              '`packet.json` 只含 `context`（claim_date、speaker、location_ISO_code）和 `evidence`（question、answer、url）；不要加入 gold、verdict 或 justification。来源 QA 需用户提供，工具不执行网络检索；闭卷方法拒绝证据文件参数。输出的标签、生成分数或阈值判决均为未证实的模型意见。', '',
              '方法说明：' + link('docs/ACCURACY_IMPROVEMENT.md') + '；抽取字段：' + link('docs/ACCURACY_EXTRACTION.md') + '；数据及来源：' + link('data/improvement/README.md') + '；许可与署名：' + link('data/improvement/NOTICE.md') + '；开发历史：' + link('reports/ACCURACY_DEVELOPMENT_HISTORY.md') + '。', '',
              '## 原始记录与核验', '',
              f"本报告输入组合指标 SHA256：`{sha(metrics_path)}`。报告生成器 SHA256：`{sha(Path(__file__))}`。生成器只读取已完成记录和绑定的字节哈希，不运行模型、不执行阈值拟合、不据留出成绩重新选方法。", '']
    if tests_passed is not None:
        lines += [f'调用者记录的本地测试通过数：**{tests_passed} 项**。该数由 `--tests-passed` 传入；本生成器不执行测试，也不把它当作模型正确性证明。', '']
    for key, relative in CANONICAL_ARTIFACTS.items():
        lines.append(f"- {link(relative)}：`{metrics['artifact_sha256'][key]}`。")
    for relative in SUPPLEMENTARY:
        if (root / relative).is_file():
            lines.append('- ' + link(relative) + '。')
    figures = sorted({*list((root / 'reports/accuracy_figures').glob('*.png')),
                      *list((root / 'reports/figures').glob('accuracy*.png'))})
    if figures:
        lines += ['', '## 结果图表', '']
        for path in figures:
            target = os.path.relpath(path, output.parent).replace(os.sep, '/')
            lines += [f'![{path.stem}]({target})', '']
    lines += ['', '## 解释边界', '',
              '新开发/留出数据来自固定版本 AVeriTeC 的上游 train 分区，与旧 110 条上游 dev 来源不同；“项目未见”只表示此前项目运行未使用，不意味着模型预训练未见。两组新数据均要求有可用 QA、排除需重标注记录，并做文本/核查文章去重，属于有质量筛选的平衡子集；没有证明事件完全独立或代表现实新闻比例。', '',
              '上游整条标签没有独立人工重新核实，不能继承为每个原子主张的真伪。提供证据条件有额外 QA 和元数据，存在先验整理的优势；没有完成自动检索。闭卷方法可能重复同一知识错误；阈值是开发集拟合的决策规则，不是概率校准。', '',
              '留出执行完成后，这 80 条可以用于解释失败，但任何据此修改方法后的泛化提升都需要新的未触碰评估数据。当前报告保留全部条件、失败和代价，不承诺未来样本上的准确率收益。', '']
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', type=Path, default=ROOT / 'reports/accuracy_comparison_20261112.json')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/ACCURACY_RESULTS.md')
    p.add_argument('--tests-passed', type=int)
    args = p.parse_args(argv)
    require(not args.output.exists(), 'Report exists; preserve it and choose a new file')
    require(args.tests_passed is None or args.tests_passed >= 0, 'Tests-passed count must be nonnegative')
    metrics, development, calibration, selected, zero_correct = load_bundle(args.metrics)
    content = render(metrics, development, calibration, selected, zero_correct,
                     args.output, args.metrics, args.tests_passed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        handle.write(content.encode('utf-8'))
    print('Wrote completed accuracy report:', args.output)


if __name__ == '__main__':
    main()
