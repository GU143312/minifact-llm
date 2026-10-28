# Phase 2：有界抽取与验证交互

## 流程

`scripts/run_phase2.py` 运行固定模型的两个提示角色。抽取器从输入中输出原文引用；`minifact.align_quotes` 对每条检查是否为唯一的原文连续切片，拒绝改写、重复和不明确的位置。验证器对每条可定位引用输出 `SUPPORTED`、`REFUTED` 或 `NOT_ENOUGH_INFO`，另外生成具体的质疑建议。源内逻辑角色还检查已抽取命题是否相互矛盾。

如果有完整、可用的模型反馈，程序将反馈标为未证实内容，连同原文交给抽取器进行下一轮；最多两轮。没有有效引用、无可执行反馈、提取结果重复、模型异常以及达到轮次上限会留存停止原因或逐项错误。源内检查异常或截断时，最终停止原因仍可能是无可执行反馈；这不代表逻辑检查成功或确认无矛盾。每轮保留请求、响应、逐字匹配与拒绝、判决、反馈和下一轮实际输入。无法解析的输出不会被悄悄修补；`NO_ISSUE` 和 `NO_CONTRADICTION` 仅作模型输出的无反馈标记，附带单个句末句号或感叹号也可识别。输出额外解释时保留原文。

这是同一个模型的不同角色、闭卷且无外部检索证据。源文只用于提供上下文或检查源内矛盾，并非独立的事实证据。一个被标为 `REFUTED` 的整条样本不意味着其中每个原子主张都是假的；验证器的输出也不构成金标准。源内逻辑结果本身是模型意见，不保证判断正确。相同提取结果会终止循环，最后一轮即使产生了新反馈也不一定有后续轮次消费；查看实际 `extract_request` 才能确认反馈已传递。

## 分数和随机性

判决通过模型**实际 greedy 生成**；对每个生成 token 从模型未经过生成处理器修改的原始 logits 计算 token log-probability 与全词表熵，记录 token ID、特殊 token、判决标签 token、逐 token 分数和判决标签 token 均值。仅当解码片段可对齐且输出完整合法时，才报告判决标签 token 均值。EOS 与特殊 token 不计入标签均值。若一个 token 同时覆盖标签及前后空白，均值包含该完整 token。熵单位为 nats。

这些分数反映模型的生成分布，**不是事实为真的概率，也未经过校准**。它们与早期 `verdict_probabilities.py` 中把固定候选完整序列逐一 teacher-force 后做候选集 softmax 的分数及熵不同；两者不可混称。引用 JSON 的平均生成 log-probability 也不代表判决置信度。

模型、revision、FP16、seed、eager attention 和 greedy 设置来自 `config.json`。生成上限按角色固定：抽取 256、判决 16、逐条反馈 64、源内逻辑检查 64 个新 token；这些上限和实际参数逐次写入运行记录，适用于所有执行环境。模型未结束即达到上限时保留原文，未完成的判决和反馈不会当作有效结果。抽取阶段即便未遇 EOS，只要生成内容可解析为数组，仍会尝试定位其引用；因此必须单独报告未结束的抽取，不能称之为完整输出。抽取配置是冻结的上限；若 `config.json` 的上限更小，实际使用较小值。

## 数据与实验边界

`data/phase2_dev.json` 是独立合成开发集；预期事实、限定及逻辑关系只用于开发审阅，不传入模型。开发集由 agent 设计和自审，仍需独立人工核验，不代表泛化性能。`data/phase1/inputs.json` 是 60 条正式主张的无答案输入；上游 Refuted 标签、核查文章、证据和解释均不传入这一闭卷流水线。

当前正式样本全部为上游样本级 Refuted，且没有独立的原子事实金标准。可以报告原文切片与模型输出行为，不能据此报告总体准确率、误报率、事实覆盖率或限定保留率。开发集上调过提示词，开发结果不能当作独立测试成绩。正式输入的前三条曾在 v5 协议预运行；发现开发集的标点解析问题后立即中止并保留该部分记录，v6 修正来自开发集问题而非正式输出。正式运行开始前记录协议时间、模型、代码和输入哈希，见 `reports/phase2_frozen_protocol.json`；这不是独立可信时间戳。

## 运行与检查

Python 环境按 `requirements.txt` 安装；本地模型快照可保存在 E 盘。示例：

```powershell
python scripts/validate_phase2_dev.py
python scripts/run_phase2.py --input-kind dev --cache-dir E:/MiniFact-LLM-runtime/hf-cache --output runs/phase2_dev.json
python scripts/run_phase2.py --input-kind formal --cache-dir E:/MiniFact-LLM-runtime/hf-cache --output runs/phase2_formal.json
python scripts/audit_phase2_run.py --run runs/phase2_formal.json --input data/phase1/inputs.json --output runs/phase2_formal_audit.json
python scripts/summarize_phase2_run.py --run runs/phase2_formal.json --input data/phase1/inputs.json --output runs/phase2_formal_summary.json
```

运行逐样本写入检查点。程序默认拒绝覆盖已有文件；使用相同参数的 `--resume` 只允许在输入哈希、协议代码、配置、运行环境、设备映射及已有样本前缀均一致时续跑。模型缓存或 GPU 预算不一致可能导致续跑被拒绝，需保留旧文件并开始新运行。对正式结果与冻结记录的核对使用 `scripts/audit_phase2_freeze.py --help` 中列出的路径参数。

审计器可以检查记录内的生成 token 与分数、EOS、反馈传递、字符切片、代码和数据哈希的一致性。`completed: true` 只表示所有输入都已处理，不能据此推断每次抽取或验证成功。运行记录不保存完整词表原始 logits，因此离线审计不能仅凭记录独立重算 logits；需要原模型再次运行。记录固定模型 revision，但不含本地权重及 tokenizer 缓存的逐文件哈希，不能保证跨设备逐位一致。自动位置和记录检查不能替代语义人工复核。
