# MiniFact 下一轮评估：执行协议

这是 2026-11-22 确认方案的实现。正式新闻评估尚未启动；当前使用 `data/next_eval_v2` 的 200 条候选。已完成明确披露的 agent 事件与来源初筛，排除 20 个历史/内部事件重叠或来源歧义候选；60 条双人语义标注尚待真实人工完成。不能将模板填充、模拟测试或 agent 初审当作双人金标准。

数据结构由 [JSON Schema](next_evaluation.schema.json) 和跨记录校验器共同验证；正式冻结绑定 Schema 字节。

## 样本与对照

固定 AVeriTeC revision `7c62d1ec8df3fb560d6efe2b85fa191135636f81` 的 train 分区；引用、过滤和许可沿用 [数据说明](../data/improvement/README.md)。旧 110 条、32 条开发集及已曝光的 80 条不再作为新测试。200 条候选按 seed 42 的 SHA 排序、100 Supported / 100 Refuted 选择，排除历史文本和核查文章重复。v2 的额外排除索引和理由保留在 `event_review.json`；替补依照同一排序产生，未依据新测试模型输出选择。文章去重不能代替事件审查，agent 初筛也不能保证事件完全独立。

60 条盲标子集在每类固定选 30 条。另有 30 条虚构的、按证据包判断的边界用例：重复引文、共享主语、指代、否定、时间、数量与因果范围各 5 条；每类 1 条证据不足。其成绩不合并到真实新闻准确率。它们的预期由作者制定，不是人工新闻真伪金标准。

| 组 | 输入 | 方法 |
|---|---|---|
| A | 原文 | 原有冻结二候选 log-likelihood 阈值，保留原提示和阈值 |
| B | 原文 | 统一三标签直接判断 |
| C | 原文＋日期、说话人、地区 | 与 B 相同最终提示 |
| D | C＋固定参考 QA | 与 B 相同最终提示 |
| E | 与 D 完全相同 | 抽取、逐项验证、一次反馈检查、整条裁决；仅一轮 |
| F | 与 D 完全相同 | E 流程，有完整非 NO_ISSUE 反馈时允许第二轮；最多两轮 |

B—F 最终判决提示相同。E/F 第一轮相同；E 保存反馈但不修订，F 可修订。最多 256 抽取 token、64 反馈 token、16 判决 token，仍使用 Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`、FP16/eager、greedy。模型缓存留在 E 盘。候选 A 是两次 teacher-forced 前向，调用不能与一次 generate 直接等价。

主要比较只有 **F−D**；B→C、C→D、D→E、E→F 是次要诊断。附加计算预算单独报告，不宣称由此隔离反馈或 logprob 的因果贡献。不进行网络检索、微调或模型替换。

## 人工工作与冻结顺序

1. 阅读 [人工标注指南](NEXT_EVALUATION_ANNOTATION.md)。当前 v2 已保留 agent 事件初筛记录及其局限；人工可在开始语义标注前复核。发现历史事件或候选内部事件重复时，通过 `prepare --exclude <upstream indices>` 写入新版本目录，再开始语义标注。不要直接删改数据或用模型判错作为排除理由。
2. 将 `reviewer_a.json`、`reviewer_b.json` 分别交给两名人工；相互隐藏初始标注，且不可展示模型输出。完成后共同填写 `adjudication.json`。身份使用稳定化名，公开仓库不需要真实姓名。
3. 开发仅使用原 32 条开发集。正式冻结要求这 32 条六组运行完整；不能用 200 条新测试选提示或阈值。
4. `freeze --stage news` 在加载 GPU 前检查事件复核、两名不同人工、60 条完成、分歧处置、精确引用、日期和哈希。缺失则拒绝生成正式 freeze。身份及独立性属于人工声明，程序不能认证真实身份或认知独立性。
5. 冻结后一次性运行。程序按样本轮转方法次序，每一组完成即保存检查点；中断保留原因类型，只有代码、数据、运行环境和历史前缀一致时才可续跑。硬件异常不会被悄悄当作模型预测或自动跳过。
6. 完整运行后生成语义对齐模板，由两名人工共同核对 E/F 的全部预测；再次绑定原始 run 与金标准哈希。模板没有默认“正确”选项。

程序的正式执行入口对金标签和人工标注仅检查文件字节哈希，不解析其内容；运行时模型只接收各组允许的字段。证据 URL 保留为来源记录，提示仅见 E1、E2 等引用编号；不传样本 ID、金标签、裁决解释、核查文章。

## 指标与验收

所有弃答和无效输出留在准确率分母。Refuted 是错误正类：严格 FNR 仅算错误被判 Supported；含弃答未检出率另计 UNKNOWN 和 INVALID；FPR 是 Supported 被判 Refuted。报告 2×4 混淆矩阵、二分类覆盖、解析与截断失败、Wilson 95% 区间。

F−D 使用 seed 42、10,000 次按金标签分层的配对 bootstrap；报告配对正确性四格表。不得根据区间挑选别的主要比较。区间跨零为证据不足，不是证明等效。

语义指标单列：

- 事实召回：送入后续验证器的已接受预测覆盖的不同参考事实 / 所有参考事实；漏提仍计入分母，重复不得多得分。
- 原子性和忠实性：对所有可解析 raw 项逐条人工判断，包括拒绝项。无 raw 项时不制造一个正确预测，漏提由召回反映；零分母为 null。
- 限定保留：在已覆盖核心事实中的保留比例；另给将漏提计入的端到端限定保留率。主语、关系、宾语核心匹配与限定是否保留分开判断。
- 定位：最终抽取所有接受引用是否逐字等于原文半开区间；同时报告顶层解析错误、拒绝、歧义和无接受项。位置匹配不等于语义正确。

记录实际生成标签 logprob 均值和熵；A 单独使用距冻结边界的绝对 margin。它们不是真假概率。AUROC 衡量分数区分正确/错误的能力；弃答若有可用标签分数也计入错误类。选择性预测阈值来自完整开发集有效二分类分数的 0/10/20% 分位位置，对应目标覆盖 100/90/80%；实际测试覆盖必须原样报告。没有足够分数或某一正确性类别时 AUROC 为 null。

成本记录生成调用、候选前向、输入 token、生成 token、候选评分 token、平均/P95 秒和峰值 CUDA 分配字节；排除模型加载及 checkpoint I/O。

**交付验收与方法升级分开：**新闻全部记录及审计、双人金标准和逐预测语义对齐、完整同版本边界测试都到齐，才标交付完成。即使实验无收益，也可以完成研究交付。

默认方法升级需同时满足：准确率 ≥80%，F−D ≥5 个百分点且配对 95% 下界 >0；严格 FNR ≤10%、含弃答未检出 ≤15%、FPR ≤25%；后两项相对 D 恶化均 ≤2 个百分点；事实召回 ≥90%，原子性、忠实性、条件限定保留 ≥95%，接受引用逐字匹配 100%。语义值缺失时不得判通过。阈值是预先约定的样本工程门槛，不是人群风险保证。

## 命令

从仓库根目录运行，`python` 应指向既有 E 盘环境。输出只允许新文件/新目录，唯一例外是显式 `run --resume`。

```powershell
python scripts/next_eval.py check --bundle data/next_eval_v2 --source E:/MiniFact-LLM-runtime/datasets/averitec/train.json
python scripts/next_eval.py readiness --bundle data/next_eval_v2

python scripts/next_eval.py freeze --stage dev --bundle data/next_eval_v2 --output reports/next_dev.freeze.json
python scripts/next_eval.py run --freeze reports/next_dev.freeze.json --output reports/next_dev.json
python scripts/next_eval.py audit --run reports/next_dev.json --freeze reports/next_dev.freeze.json --cache-dir E:/MiniFact-LLM-runtime/hf-cache --output reports/next_dev.audit.json

python scripts/next_eval.py freeze --stage boundary --bundle data/next_eval_v2 --output reports/next_boundary.freeze.json
python scripts/next_eval.py run --freeze reports/next_boundary.freeze.json --output reports/next_boundary.json
python scripts/next_eval.py audit --run reports/next_boundary.json --freeze reports/next_boundary.freeze.json --cache-dir E:/MiniFact-LLM-runtime/hf-cache --output reports/next_boundary.audit.json
python scripts/next_eval.py evaluate --run reports/next_boundary.json --freeze reports/next_boundary.freeze.json --output reports/next_boundary.metrics.json
python scripts/report_next_boundary.py --run reports/next_boundary.json --metrics reports/next_boundary.metrics.json --audit reports/next_boundary.audit.json --output reports/next_boundary_report

# Only after actual human review and complete development:
python scripts/next_eval.py freeze --stage news --bundle data/next_eval_v2 --dev-run reports/next_dev.json --output reports/next_news.freeze.json
python scripts/next_eval.py run --freeze reports/next_news.freeze.json --output reports/next_news.json
python scripts/next_eval.py audit --run reports/next_news.json --freeze reports/next_news.freeze.json --cache-dir E:/MiniFact-LLM-runtime/hf-cache --output reports/next_news.audit.json
python scripts/next_eval.py alignment-template --run reports/next_news.json --freeze reports/next_news.freeze.json --output reports/next_news.semantic_review.json
# Fill the semantic review jointly before passing --alignment:
python scripts/next_eval.py evaluate --run reports/next_news.json --freeze reports/next_news.freeze.json --alignment reports/next_news.semantic_review.json --boundary-run reports/next_boundary.json --output reports/next_news.metrics.json
python scripts/next_eval.py report --metrics reports/next_news.metrics.json --output reports/next_news_report
```

`evaluate` 不带 `--alignment` 时仍可输出整条指标，但交付和升级均不会通过。`--limit` 仅用于开发冒烟测试；限量开发结果不能用于正式冻结。所有图表和报告记录指标 SHA；无模型分数时不会生成假结果。

`report_next_boundary.py` 是单独的离线呈现工具，要求完整边界 run 与 tokenizer 审计，并重建核对指标后生成按句型图表及逐条结果；自身脚本和导出文件哈希另记在图表 manifest 中，不修改已冻结推理协议。
