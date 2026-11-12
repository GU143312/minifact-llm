# Week 6 / Days 11–12：整条主张的受控比较

本周完成完整测试集上的两个系统比较：直接提示与 Phase 2 抽取—验证循环加整条样本裁决。主要观察量是 benchmark 标签准确率和漏报；不把事实覆盖、限定条件保留、位置匹配与分类成绩合成一个通过率。下一周再开展系统性的错误类型归因和可视化。

## 数据和判定单位

判定单位是**一条完整输入主张**，与 AVeriTeC 的上游样本级标签一致。原 Phase 1 的 60 条 Refuted 输入、ID 和文本保持原样，另加从相同固定版本选出的 Supported 对照；最终数量、筛选限制、剔除原因、来源和许可见 [`data/phase3/README.md`](../data/phase3/README.md)。下文用 N 表示最终总数，用 R、S 表示 Refuted、Supported 数量，不假定两类必然平衡。

`data/phase3/inputs.json` 只含 ID 和文本。运行器只把文本送入模型；ID 用于记录和对齐结果，不进入模型提示。标签、核查解释和参考证据只供离线评估，不能进入任一分支。`data/phase3/dataset.json` 保留可追溯的上游标签和来源，但这些标签没有经过独立人工二次裁决；抽取出的每个原子主张不能继承整条样本的标签。

这是经过筛选的 benchmark 子集，不代表现实新闻类别比例。原 60 条已参加 Phase 1、Phase 2 运行，不能称为全新的未见测试集；新增支持类对照与原错误集应分开解释。公开旧样本也可能出现在模型预训练中。筛选时排除可识别的重复文本和共享核查文章，仍不能保证所有事件、实体或故事语义独立。此处比较上游标签一致性，不声称重新证明每条新闻的真伪。

## 两个分支

两个分支使用同一份 [`phase3_experiment.py`](../phase3_experiment.py) 中的 `SAMPLE_PROMPT`，并生成 `SUPPORTED`、`REFUTED` 或 `NOT_ENOUGH_INFO`。`REFUTED` 表示至少一项重要事实断言错误，`SUPPORTED` 要求整条主张的重要断言均可成立；无法建立判断时允许弃答。必须完整输出一个允许的标签并生成 EOS，额外解释、拼写变体和截断均记为无效输出，不自动修补。

| 分支 | 最终裁决器的输入 | 额外推理 |
| --- | --- | --- |
| `direct` | 原文，`analysis: null` | 一次整条样本判决 |
| `dual` | 同一原文，以及标为未证实的模型分析 | 原 Phase 2 循环，最多两轮，再作整条样本判决 |

双角色分支的分析包含每一轮可逐字定位的引用、拒绝记录、抽取错误、逐条判决和反馈、源内逻辑反馈及停止原因。保留所有轮次，避免第二轮为空时把第一轮的成功引用隐藏。逐条判决的实际生成 log-probability 和熵也进入分析上下文，但明确标为同模型分数，不能当成独立证据或事实为真的概率。没有外部检索，也没有把原文当作其自身真实性的证明。

增加整条样本裁决器是本周为对齐上游标签而加入的步骤；历史 Phase 2 的原子引用判决不能直接当作本周整条样本的成绩。这是**系统比较**，不是相同计算预算的消融实验。双角色分支通常多次调用同一组权重，耗费更多输入、输出 token 和运行时间；调用数、生成 token 和耗时应随结果报告。不能把差异单独归因于反馈、logprob 或某一个组件。

逐样本交替执行顺序：偶数索引先 `direct`，奇数索引先 `dual`。每次 `generate` 建立自己的输入和 KV cache，不跨样本共享对话。每条完成两个分支后写入检查点；运行中断保留已完成的前缀，复跑不能混用不同代码、输入、冻结记录或运行环境。

## 固定模型与生成设置

沿用 `config.json` 中的 `Qwen/Qwen2.5-3B-Instruct`、revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`、FP16、seed 42、eager attention、greedy 单 beam，以及相同 repetition penalty、KV cache 和确定性设置。两个最终裁决器均至多生成 16 个新 token；原 Phase 2 角色上限保持抽取 256、判决 16、逐条反馈 64、源内逻辑检查 64。配置若给出更低上限，实际使用更低值并留存。

正式运行前冻结输入、代码和配置哈希、两轮上限、分支顺序、指标定义及以下阈值。正式测试的提示词或阈值不能根据结果调整；若发现问题，保留原运行、说明原因，建立另一个版本，不能覆盖成一次成功实验。冻结文件中的本机时间不是独立可信时间戳。

模型从固定本地快照加载，避免 revision 漂移和额外下载；大模型及缓存放在 E 盘。运行记录保留实际 dtype、revision、库版本、设备映射、生成设置、完整响应和实际提示词 token。数值可能受设备和 FP16 计算影响，固定配置不意味着跨硬件逐位相同。

## 指标、弃答与分数

正类定义为**存在错误，即 REFUTED**。对两个分支使用同样的 N、R、S 和下列分母：

| 指标 | 定义 |
| --- | --- |
| 整体准确率 | 与上游标签相同的完整判决数 / N；弃答和无效输出均未判对 |
| 错误召回率 | 金标签 Refuted 且预测 REFUTED / R |
| 严格漏报率 FNR | 金标签 Refuted 却预测 SUPPORTED / R |
| 弃答的错误样本比例 | 金标签 Refuted 且预测 NOT_ENOUGH_INFO / R |
| 无效输出的错误样本比例 | 金标签 Refuted 且输出无效 / R |
| 含弃答的未检出率 | 金标签 Refuted 未预测 REFUTED / R，即漏报、弃答和无效输出之和 |
| 误报率 FPR | 金标签 Supported 却预测 REFUTED / S |
| 有效二分类覆盖率 | 预测 SUPPORTED 或 REFUTED 的样本数 / N |
| 有效二分类条件准确率 | 有效二分类中判对的数量 / 有效二分类数量；不能代替整体准确率 |

同时给出金标签为行、预测 `SUPPORTED`、`REFUTED`、`NOT_ENOUGH_INFO`、`INVALID` 为列的混淆矩阵。若模型总是弃答，严格 FNR 可以为零，但含弃答的未检出率为一；因此必须一起阅读。分母为零的指标留空，不伪造零成绩。配对比较保留两边均正确、仅直接分支正确、仅双角色分支正确、两边均错误的数量；小样本统计不证明一般人群效果。

最终裁决的分数取自该次**实际 greedy 生成**的原始 logits，不使用反馈文本、抽取 JSON 或另一个候选答案的分数替代。逐标签 token 计算 log-probability 与全词表熵，排除特殊 token 和 EOS；对齐不确定或分数不可得时保留缺失状态。输出均值的解释见 [Phase 2 文档](PHASE2.md)。样本均值的平均与 token 加权平均不是同一统计量，结果应注明聚合方式。

探索性选择性分析固定均值 logprob 下限为 `-0.05`、`-0.1`、`-0.25`、`-0.5`，两个分支全部报告，不挑选测试集上最好看的阈值。只有完整有效的二分类判决且分数达到下限时保留；NEI、无效、缺失分数及阈值拒绝均算未作二分类决定。分别统计覆盖、保留决定的准确率及对全部错误样本的未检出情况。标签长度、条件 token 分布会影响这些分数；它们没有经过校准，不能解释为真假概率或声称阈值可靠识别幻觉。

## 复现步骤

在仓库根目录执行。下列报告名为示例，历史文件存在时应保留并选择新名称。

```powershell
python -m unittest discover -s tests -q
python scripts/run_phase3.py --freeze-only --freeze reports/phase3_week6_frozen_protocol.json
E:/MiniFact-LLM-runtime/.venv/Scripts/python.exe scripts/run_phase3.py --freeze reports/phase3_week6_frozen_protocol.json --output reports/phase3_week6_run.json --cache-dir E:/MiniFact-LLM-runtime/hf-cache
```

中断后，在相同环境使用同一命令加 `--resume`；已经完成的运行拒绝续跑。完整结果再由 `scripts/evaluate_phase3.py` 离线对齐金标签、检查生成记录和冻结哈希并计算指标，参数见 `--help`。数据构建与核对参见数据集说明。自动结构审计不证明模型语义判断正确，逐字位置正确也不证明抽取提全或原子性成立。

本次正式交付使用 `phase3_week6_frozen_protocol_v2_20261112.json` 和 `phase3_week6_run_v2_20261112.json`；[运行历史](../reports/PHASE3_RUN_HISTORY.md)保留第一次清单哈希不一致的部分运行。实际复算命令：

```powershell
python scripts/evaluate_phase3.py --run reports/phase3_week6_run_v2_20261112.json --freeze reports/phase3_week6_frozen_protocol_v2_20261112.json --output runs/week6_metrics.json
python scripts/replay_phase3_prompts.py --run reports/phase3_week6_run_v2_20261112.json --freeze reports/phase3_week6_frozen_protocol_v2_20261112.json --cache-dir <模型缓存目录> --output runs/week6_prompt_replay.json
python scripts/recheck_phase3_logits.py --run reports/phase3_week6_run_v2_20261112.json --cache-dir <模型缓存目录> --output runs/week6_logits_recheck.json
```

本周交付包括标准化对照数据、冻结协议、成对完整模型输出、可复算评估代码和比较报告。事实覆盖、限定条件保留仍需各自的原子语义标注；没有相应标注时明确记为未评估，不能由分类准确率或位置审计反推。
