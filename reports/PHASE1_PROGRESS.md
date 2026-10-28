# 正式实验进度与运行记录

2026-10-29，使用本地 Qwen/Qwen2.5-3B-Instruct、固定 revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`、FP16 和既有 greedy 配置，对 Days 1–2 冻结的 60 条无答案输入完成一次抽取。代码保留提示词、参数、输入、模型与环境的哈希或版本，未读取参考标签和证据。

## 已完成

- Days 3–4 所需的 Hugging Face 抽取入口和字符到模型输入 token 映射已实现：`scripts/run_phase1_extraction.py`、`token_alignment.py`，方法和限制见 `docs/TOKEN_ALIGNMENT.md`。原先把人工原子标注称为 Days 3–4 必交项并不准确；它仍是后续可靠评估所需工作。
- 正式输入 60/60 已运行；见 [原始抽取结果](phase1_local_20261029.json) 和[逐句汇总](phase1_local_20261029_summary.json)。全部输出 JSON 可解析且均以 EOS 结束。
- 预测引用共 116 条：93 条与原文存在唯一逐字匹配，23 条被拒绝，原因均为没有逐字匹配。这是定位统计，不是事实完整率。7 条输入没有任何可接受的引用。
- 93 条接受的字符引用中，输入 token 区间有 3 条边界精确，90 条为 `partial_token`，即覆盖了引用之外的字符。离线重新计算复现了 60 条提示词 token ID 和 93 条映射；[审计记录](phase1_token_alignment_audit_20261029.json)复用同一映射实现，只能证明复现，不能独立证明算法正确。
- [前 10 条试阅](phase1_pilot_review.json)按输入顺序记录事实提及、独立性、限定条件、引用位置及每句生成 JSON 的平均 log-probability。它是 agent 对前 10 条的语义初审，非独立人工金标准；已发现补写、片段不独立及时间或活动背景遗漏等问题，不将该试阅泛化为 60 条的总分。

## 后续验证器原型

已实现 `verdict_probabilities.py` 和 `scripts/run_phase1_verifier.py`。同一个 Qwen 以闭卷验证角色，使用原句作为指代上下文，对 93 条可定位引用分别对固定 `SUPPORTED`、`REFUTED`、`NOT_ENOUGH_INFO` 完整候选序列进行 teacher forcing；log-probability 包含 EOS。原始结果见 [验证器运行记录](phase1_verifier_20261029.json)。

93 条候选最高分标签分布为 `SUPPORTED` 21、`REFUTED` 4、`NOT_ENOUGH_INFO` 68；候选集平均熵为 0.303107 nats。这只是候选中模型偏好的分布，不是真假准确率。候选分数按三个完整序列的 log-probability 求和后在该候选集内 softmax；不是经过校准的真值概率，也不能直接与抽取 JSON 的平均 log-probability 等同。七条没有可接受引用的样本没有验证器判决。

该运行没有外部检索证据，抽取器和验证器是同一模型的两个提示角色，尚无验证器反馈给抽取器的迭代循环；因此 Days 5–10 的双代理交互和有证据验证尚未完成。上游 Refuted 是整条样本标签，不能自动赋给每个原子引用。全错误样本不支持估计误报率或总体准确率，且 2020 年公开样本可能被模型见过。

## 验证

- `python -X utf8 -m unittest discover -s tests -v`：43 项运行，42 通过、1 项因未设置本地 tokenizer 路径而跳过。
- 在本地模型环境设置 `MINIFACT_TEST_TOKENIZER` 为固定快照后，运行 `python -X utf8 -m unittest discover -s tests -p test_token_alignment.py -v`：7/7 通过，包括真实 Qwen chat template。
- 固定 tokenizer 离线检查：60/60 条抽取提示词重编码与模型输入 ID 一致；核查验证器候选在 prompt 边界的编码和 EOS（先行抽样及每次验证调用均检查）。
- `python -X utf8 scripts/validate_phase1.py`：60 条数据及输入隔离校验通过。`python -X utf8 scripts/summarize_phase1_extraction.py runs/phase1_local_20261029.json --output reports/phase1_local_20261029_summary.json`：完整运行记录一致性检查通过。
- 验证器输出独立复算：60 条样本、93 条引用的对应关系、三候选概率归一、序列 log-probability 和及熵均匹配。

接下来需建立逐原子事实、限定与引用的独立参考标注；加入明确来源和可审计证据的验证流程、反馈循环，再设计受支持对照及基线比较。现有产物不能报告 60 条事实覆盖率、限定保留率或真假准确率。
