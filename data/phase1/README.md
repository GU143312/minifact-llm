# Days 1–2：正式新闻错误测试集 v1.0.0

交付物为 **60 条英文真实新闻相关主张**的标准化 JSON。它们改编自 FEVER 系列公开基准 **AVeriTeC 的 dev 集**，均保留上游 `Refuted` 标签。样本单位是事实核查机构从新闻、公开发言及社交媒体中提取的短主张，**不是 60 篇完整新闻文章**。本版满足 50–100 条含事实或逻辑错误样本的数量要求；没有合成或随机篡改新闻来补足数量。

## 文件

| 文件 | 用途 |
| --- | --- |
| `dataset.json` | 正式主数据集：原文、上下文、整条主张裁决、裁决解释、证据 QA、逐条来源及哈希 |
| `inputs.json` | 模型输入：仅含 `id` 和 `text`，不含标签、解释、证据或核查文章链接 |
| `dataset.schema.json` | JSON Schema Draft 2020-12，固定字段、类型、枚举和 60 条样本数量 |
| `curation.json` | 固定来源版本、筛选规则、候选配额、剔除理由和最终来源行号 |
| `NOTICE.md` | 原始作者、论文引用、数据许可和修改说明 |

正式数据与已有的 `../cases.json` **15 条合成回归用例分开**保存。旧用例只用于检查抽取器和本地/Colab 协议，不计入本次 60 条。

## 来源与筛选

- 官方数据说明：[AVeriTeC](https://fever.ai/dataset/averitec.html)；作者仓库：[MichSchli/AVeriTeC](https://github.com/MichSchli/AVeriTeC)。
- 原始文件：`data/dev.json`，固定 revision `7c62d1ec8df3fb560d6efe2b85fa191135636f81`。
- 文件 SHA256：`499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300`。
- 原始 500 条中有 305 条 `Refuted`。仅保留该标签，其他三类均不纳入错误样本集。
- 候选要求：8–65 个空白分词；有裁决解释和核查文章；至少有一条非 Unanswerable、带 HTTP(S) 来源的 Web text/PDF 答案；排除明显依赖图片/视频、开头指代不明及非文本证据的记录。该规则检查的是提供的文本材料可用性，不保证它是最佳证据。
- 审阅主张与裁决解释，并对部分 QA 定向检查；排除明显的标注矛盾、解释不足、重复故事和不必要的精确住宅地址。理由按零基来源行号记录。
- 对 178 条合格候选按原始首个 `claim_types` 分层、SHA256 排序选出 80 条。排序字符串为 `minifact-phase1-v1|42|来源行号|原文`；使用 UTF-8。每个核查文章最多 1 条、核查域名最多 8 条、具名发言者最多 3 条。
- 最后剔除 20 条范围或解释仍不清晰的记录，**不补样**，冻结 60 条。最终清单是 `curation.json` 的 `final_source_indices`。它是经过质量筛选的小样本，不能宣称代表新闻总体分布。

这是 agent 对公开标注材料进行的筛选，**没有独立人工二次裁决，也没有逐条在线抓取复核来源**。证据、原始日期和裁决文本保持上游版本；并未将这些历史标签重新解释为当前时点的真伪结论。

## 数据概况

| 项目 | 数量/范围 |
| --- | --- |
| 正式样本 | 60 |
| 标签 | 全部 REFUTED，整条样本级别 |
| 核查来源域名 | 18 |
| 核查问题 / 答案 | 142 / 151 |
| 原始主张日期 | 2020-08-26 至 2020-10-29 |
| 空白分词长度 | 9–47，平均 19.65 |
| 上游首个主张类型 | Event/Property 29；Quote Verification 13；Causal 8；Numerical 8；Position 2 |

原始 `claim_types` 仅用于保留基准信息及分层，**不是错误类型标注**，其类型名称也不保证与文本表面结构完全对应。领域分布未另行推断。具体域名和地理代码统计在 `dataset.json.statistics` 中。

## 标准字段与使用边界

- `id`：本项目 ID，按固定来源行号排序分配。上游没有显式 claim ID；`source_record_index` 是固定文件中的零基数组下标，不冒充作者提供的 ID。
- `text`：逐字保存的原始主张，保留拼写、标点及可能的复合句。尚未拆成原子事实。
- `claim_date`：把上游日-月-年转换为 ISO 日期；原始字符串保存在 `provenance.claim_date`。日期是上游对首次出现时间的估计。
- `context`：原始发言者、发布来源和地理代码，缺失值保留为 `null`。
- `gold.verdict`：把上游 `Refuted` 统一映射为 `REFUTED`；`verdict_scope` 固定为 `whole_sample`。**一句话有错不等于其中每个原子事实都错。**
- `gold.justification/questions`：上游裁决理由与完整 QA。`Unanswerable`、`Metadata` 来源及缺失缓存链接均原样保留；它们不能冒充可抓取网页证据。可访问的 HTTP(S) Web text/PDF 答案只是筛选最低要求。
- `provenance`：原始核查文章、原始主张 URL、来源行号、上游类型/策略/复标记，以及恢复原始记录后的 SHA256。记录哈希使用 UTF-8 JSON，键排序，紧凑分隔符，`ensure_ascii=False`。
- `annotations`：原子事实、限定条件和错误类别为 `null`，状态 `pending_days_3_4`。Days 1–2 只完成数据策划；没有伪造尚未完成的逐原子真伪标签或字符位置。

默认只把 `inputs.json` 的 `text` 传给抽取器。历史上下文若用于验证阶段，应在实验协议中提前声明，并在各比较条件一致提供；核查文章标题/URL、gold 和 QA 均不能暗中混入闭卷输入。若未来提供 gold QA 作为证据，应明确称为给定金标准证据的实验，与自主检索实验分开。

全部为错误正例，因此可测错误检出覆盖，**不能估计误报率、特异度或正常新闻上的表现**；本版不含独立标注的纯逻辑错误子类。样本来自公开 dev 集，可能出现在模型训练材料中，不保证未见。后续不能反复根据此测试集调提示词；调参需要另设开发集。

旧 15 句上的事实核心、限定条件、字符位置及 log-probability 结果不能移作本集的实验结果。后续需要另行建立原子事实及限定的参考标注，并分别报告覆盖、限定保留和位置正确性，不合成一个通过率。数据集 v1 本身不包含这些标注。

## 校验与重建

仅验证交付数据，无需模型、网络或 GPU：

```powershell
python -m pip install -r requirements-data.txt
python scripts/validate_phase1.py
python -m unittest discover -s tests -v
```

下载[固定版本源文件](https://raw.githubusercontent.com/MichSchli/AVeriTeC/7c62d1ec8df3fb560d6efe2b85fa191135636f81/data/dev.json)到仓库外的缓存目录，再执行：

```powershell
python scripts/curate_phase1.py --source <缓存目录>/dev.json --check
# 去掉 --check 可重建 dataset.json 和 inputs.json。
```

原始下载与模型缓存可放在 E 盘，仓库只保留筛选后的数据。重建先验证整个源文件 SHA256，再验证候选与最终清单；校验器检查 schema、逐条记录哈希、重复文本/文章、集中度、日期、统计和输入泄漏。版本冻结后如需修改，应创建新版本并记录变更。
