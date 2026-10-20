"""Build/check readable dataset documentation and exact reference occurrences.

Never rewrite cases.json or the inference/evaluation protocol.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/cases.json'
CATEGORIES = {
    'baseline': ('基础复合句', '检查并列事实拆分，以及时间、数量是否保留。'),
    'repeated_quote': ('重复引文', '同一事实重复出现只计一个命题；引用存在多个位置时不得猜测。'),
    'shared_subject': ('共享主语', '第二个谓语沿用前文主语；检查是否擅自补写主语而破坏逐字引用。'),
    'pronoun': ('指代', '保留 she、its、that wing 等原文表达，并注明其上下文指向。'),
    'negation': ('否定', '区分事实核心覆盖与否定条件保留；检查 did not 和 No。'),
    'complex_time': ('复杂时间', '覆盖时间区间、钟点、相对日期、未来模态和适用范围。'),
}
KINDS = {'time': '时间', 'quantity': '数量', 'negation': '否定',
         'modality': '模态', 'restriction': '限制条件'}
CONTEXTS = {
    ('base_01', 'seats'): 'the building 指前文新图书馆。',
    ('base_02', 'emergency'): 'its 指 Riverdale Hospital。',
    ('base_03', 'prices'): '票价所属线路依赖整句语境；不自行给第二分句补写年份。',
    ('repeat_01', 'hall'): '同一命题有两处原文提及，事实分母为 1；两处位置均保留。',
    ('repeat_02', 'clinic'): '两段相同引文表达同一命题，不能按出现次数重复计数。',
    ('repeat_02', 'pharmacy'): 'its 指前文诊所。',
    ('shared_01', 'ward'): '沿用 Meadow Clinic 作为主语；不能在原文引用中补写该主语。',
    ('shared_02', 'road'): '沿用 Eastport Council 作为主语；同时保留 did not。',
    ('pronoun_01', 'bakers'): 'she 指 Lina。',
    ('pronoun_02', 'reopening'): 'that wing 指前文 west wing。',
    ('time_01', 'cafe'): 'its 指博物馆；that period 回指 3 月 3 日至 3 月 8 日。',
    ('time_02', 'tolls'): 'two days later 以桥梁重开时间为参照；不擅自换算绝对日期。',
    ('time_03', 'freeze'): 'its 指 River Labs。',
}


def matches(source, quote):
    spans, cursor = [], 0
    while True:
        start = source.find(quote, cursor)
        if start < 0:
            return spans
        end = start + len(quote)
        assert source[start:end] == quote
        spans.append({'start_idx': start, 'end_idx': end})
        cursor = start + 1


def build():
    raw = SOURCE.read_bytes()
    dataset = json.loads(raw)
    cases = dataset['cases']
    seen_cases, seen_qualifiers, by_category, kinds = set(), set(), {}, Counter()
    references = []
    total_facts = total_qualifiers = 0
    for case in cases:
        case_id, category, source = case['id'], case['category'], case['source']
        assert case_id not in seen_cases and category in CATEGORIES and source
        seen_cases.add(case_id)
        stats = by_category.setdefault(category, Counter())
        stats['cases'] += 1
        stats['facts'] += len(case['facts'])
        seen_facts, facts = set(), []
        for fact in case['facts']:
            fact_id, quote = fact['id'], fact['reference_quote']
            assert fact_id not in seen_facts and quote and fact['core']
            seen_facts.add(fact_id)
            spans = matches(source, quote)
            assert spans and all(core in quote for core in fact['core'])
            for qualifier in fact['qualifiers']:
                key = (case_id, qualifier['id'])
                assert key not in seen_qualifiers and qualifier['kind'] in KINDS
                assert qualifier['text'] and qualifier['text'] in quote
                seen_qualifiers.add(key)
                kinds[qualifier['kind']] += 1
                stats['qualifiers'] += 1
            assert fact['context'] is None or (case_id, fact_id) in CONTEXTS
            facts.append({'fact_id': fact_id, 'reference_quote': quote,
                          'occurrences': spans, 'occurrence_count': len(spans),
                          'unique_reference_location': len(spans) == 1})
            total_facts += 1
            total_qualifiers += len(fact['qualifiers'])
        references.append({'case_id': case_id, 'category': category,
                           'source': source, 'facts': facts})

    digest = hashlib.sha256(raw).hexdigest()
    overview = ['# MiniFact 测试集说明', '',
        f'当前固定数据共 **{len(cases)} 句、{total_facts} 个不同事实、{total_qualifiers} 项限定条件**，覆盖 {len(by_category)} 类。', '',
        '这是已用于检查输出和跨环境复现的合成开发／回归挑战集，不是未见的独立测试集；当前不做训练集与测试集拆分，也不据此估计真实新闻上的泛化能力。', '',
        '## 文件入口', '',
        '- `cases.json`：现有实验使用的原始标注，字节保持不变。',
        '- `TEST_CASES.md`：按类别整理的 15 句原文、参考事实、核心证据、限定及上下文说明。',
        '- `reference_spans.json`：从原文机械派生的全部参考位置；不是模型预测。', '',
        '## 分类与规模', '',
        '| 类别 | 字段值 | 句数 | 不同事实 | 限定项 | 检查重点 |',
        '|---|---|---:|---:|---:|---|']
    for category, stats in by_category.items():
        title, focus = CATEGORIES[category]
        overview.append(f'| {title} | `{category}` | {stats["cases"]} | {stats["facts"]} | {stats["qualifiers"]} | {focus} |')
    overview += ['', '限定类型：' + '；'.join(f'{KINDS[k]} {v} 项' for k, v in kinds.items()) + '。', '',
        '## 标注字段', '',
        '| 字段 | 含义 |', '|---|---|',
        '| `id` | 用例编号；事实编号在各用例内唯一 |',
        '| `source` | 原始英文文本，保留大小写和标点 |',
        '| `reference_quote` | 一种参考引用边界，必须是原文连续子串 |',
        '| `core` | 核心证据片段列表；当前规则要求同一预测字符串含全部片段 |',
        '| `qualifiers` | 单独标注的时间、数量、否定、模态和适用限制 |',
        '| `context` | 指代、共享主语或重复提及的上下文说明 |', '',
        '## 三个维度分别评价', '',
        '1. 事实核心覆盖：从原始预测字符串检查核心证据，不预先排除不能定位的改写。',
        '2. 限定保留：在覆盖对应事实核心的同一预测中逐项检查限定。核心提到但时间改错，仍算限定错误。',
        '3. 位置：单独报告接受、拒绝及 `source[start_idx:end_idx] == claim`，不能只报告已接受引用的正确率。', '',
        '当前规则按区分大小写的文本片段匹配，需语义复核，不是通用事实裁判。`cases.json` 中关于连续子串的约定首先约束参考引用；实际评估候选时，核心与限定不依赖定位是否通过。', '',
        '## 边界与上下文规则', '',
        '- start_idx 为零基 Python 字符索引，end_idx 为右开区间；不是字节或 token 下标。',
        '- 重复出现的相同命题只计一个事实，所有参考位置均列出。当前唯一匹配定位器会拒绝多处匹配。',
        '- 共享主语与指代可依赖原文上下文；不能为了自包含而给逐字引用补写词语。',
        '- 参考引用给出一种边界，合法输出可含可选的末尾标点；位置正确不等于事实与限定完整。',
        '- 若后续需要规范化命题，应另设字段并保留原始引用，不覆盖原始文本。', '',
        '## 版本与维护', '',
        f'原始文件 SHA256：`{digest}`。该值与已完成的本地／Colab 比较一致。', '',
        '本次只补充说明及派生位置，不修改原句、标注、编号、顺序或推理参数。新增用例、调整金标准或改提示词应另建实验版本并重跑，不回改本轮结果。', '',
        '生成或检查派生文件：', '',
        '```text', 'python scripts/build_testset.py', 'python scripts/build_testset.py --check', '```', '']
    detail = ['# MiniFact 逐句测试集', '',
              '原文及标注来自固定的 `cases.json`。下列区间均为 `[start_idx, end_idx)`，重复位置完整保留；参考事实数按不同命题计数。', '']
    for category, (title, focus) in CATEGORIES.items():
        detail += [f'## {title}', '', focus, '']
        for case, ref in zip(cases, references):
            if case['category'] != category:
                continue
            detail += [f'### {case["id"]}', '', f'> {case["source"]}', '',
                       f'参考事实：{len(case["facts"])} 条；限定：{sum(len(f["qualifiers"]) for f in case["facts"])} 项。', '']
            for n, (fact, located) in enumerate(zip(case['facts'], ref['facts']), 1):
                spans = '、'.join(f'[{s["start_idx"]}, {s["end_idx"]})' for s in located['occurrences'])
                qualifiers = '；'.join(f'{KINDS[q["kind"]]} `{q["text"]}`' for q in fact['qualifiers']) or '无单独标注的限定'
                detail += [f'**事实 {n}　`{fact["id"]}`**', '',
                           f'- 参考引用：`{fact["reference_quote"]}`',
                           f'- 全部位置：{spans}（{len(located["occurrences"])} 处）',
                           '- 核心证据：' + '、'.join(f'`{core}`' for core in fact['core']),
                           f'- 限定：{qualifiers}']
                if fact['context']:
                    detail.append('- 上下文：' + CONTEXTS[(case['id'], fact['id'])])
                detail.append('')
    catalog = {'schema_version': 1, 'source_file': 'data/cases.json',
               'source_sha256': digest, 'span_convention': 'zero-based Python characters; end exclusive',
               'note': 'All literal occurrences of each gold reference quote; not model predictions or a requirement to select an ambiguous occurrence.',
               'case_count': len(cases), 'fact_count': total_facts,
               'qualifier_count': total_qualifiers, 'cases': references}
    return {'data/README.md': '\n'.join(overview),
            'data/TEST_CASES.md': '\n'.join(detail),
            'data/reference_spans.json': json.dumps(catalog, ensure_ascii=False, indent=2) + '\n'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate labels and reject missing/stale generated files.')
    args = parser.parse_args()
    outputs = build()
    for name, content in outputs.items():
        path = ROOT/name
        expected = content.encode('utf-8')
        if args.check:
            if not path.exists() or path.read_bytes() != expected:
                raise SystemExit(f'Missing/stale generated file: {name}; run this script without --check.')
        else:
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_bytes(expected)
    print(f'Dataset labels and {len(outputs)} derived files: PASS' if args.check else 'Wrote Chinese overview, case catalog and all reference spans.')


if __name__ == '__main__':
    main()
