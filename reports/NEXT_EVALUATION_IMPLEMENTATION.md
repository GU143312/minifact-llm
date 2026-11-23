# 下一轮评估实现与交付状态

2026-11-22（UTC+8）。评估框架、数据筛选、32 条开发验证和 30 条边界验证已完成实现与运行审计；**正式 200 条新闻评估尚未启动，也没有正式方法升级结论。**两位真实人工的 60 条独立标注及共同仲裁仍待完成。完整协议见 [执行文档](../docs/NEXT_EVALUATION.md)；代码与实验记录纳入 [PR #2](https://github.com/GU143312/minifact-llm/pull/2)，不自动合并。

## 实现范围

- 当前数据为 [v2](../data/next_eval_v2/README.md)：200 条平衡新闻主张，嵌套 60 条语义盲标子集；30 条虚构边界句另计。文本、上游整条标签、证据及许可留有来源记录。
- agent 事件/来源初筛额外排除 20 个候选，理由和替补规则公开；不是双人语义金标准，也不保证所有事件独立。没有依据新新闻集模型输出筛选。
- A—F 六组共享固定模型 revision/FP16/greedy；B—F 统一最终提示，D/E/F 使用相同参考证据。F 最多两轮，仅消费完整的可执行反馈。
- SHA 冻结代码、数据、模型配置、阈值和人工记录；逐组检查点及严格续跑；推理输入不含标签、人工标注或裁决解释。
- 准确率、严格漏报、含弃答未检出、误报、覆盖、语义召回、原子性、忠实性、限定保留和位置分别计算。包含 Wilson 区间、预先指定的 F−D 配对 bootstrap、成本、逐案例账本和图表。未标注语义值保持缺失。
- 正式冻结在 GPU 加载前检查真实人工声明、独立初审文件、仲裁及哈希；程序验证记录一致性，无法认证人员身份或认知独立性。

## 32 条开发诊断

这批是既有的开发数据，可以检查实现和拟合预定覆盖阈值，不能用于证明新测试集泛化。详细分项与图表见 [开发报告](next_eval_dev_report_20261122/REPORT.md)，原始记录及审计均保留。

| 条件 | 正确/32 | 准确率 | 严格漏报 | 含弃答未检出 | 误报 |
|---|---:|---:|---:|---:|---:|
| A 冻结闭卷阈值 | 20 | 62.50% | 62.50% | 62.50% | 12.50% |
| B 仅原文 | 4 | 12.50% | 0% | 75.00% | 12.50% |
| C 加元数据 | 5 | 15.62% | 0% | 68.75% | 6.25% |
| D 加参考 QA 直接判断 | 19 | 59.38% | 6.25% | 37.50% | 18.75% |
| E 一轮交互 | 18 | 56.25% | 0% | 25.00% | 25.00% |
| F 最多两轮交互 | 18 | 56.25% | 0% | 25.00% | 25.00% |

主要差 F−D 为 −3.125 个百分点，配对 95% 区间为 [−15.625, 9.375] 个百分点，不能证明收益或等效。D/E/F 平均每条约 1.66 / 35.21 / 35.27 秒，排除加载及检查点 I/O；运行环境负载未严格隔离，不能当作专门硬件基准。

B/C 各有 26/32 弃答，不能因严格漏报为零就认为核查可靠。F 有 10/32 弃答，含弃答未检出为 4/16；其误报为 4/16。三分类提示允许拒答，与旧二分类实验的分母和选择条件不同。

E/F 的 32 次反馈分别全部是 `NO_ISSUE`，因此没有触发第二轮，最终标签完全相同。每组最终接受 29 次引用、29 次逐字匹配；5 条没有接受项、4 条有顶层抽取解析错误、1 条拒绝记录。29 个位置正确不等于 29 个独立事实，也不能替代召回和限定保留的人工标注。

两组平均标签 token logprob 均为 −0.13618，D 为 −0.23314；F 的分数更高却未取得更高准确率，再次说明这些值不是真假概率。完整 label/EOS、提示 token 和分数计算记录可审计。

## 30 条边界诊断

完整结果与逐条表见 [边界报告](next_eval_boundary_v2_report_20261122/REPORT.md)。D 为 20/30，E/F 均为 17/30；证据不足 6 条分别识别 6/6、5/6、5/6。六类句型各 5 条，时间范围组均仅 1/5；这些是作者制定的虚构证据预期，不能合并进新闻准确率。

E/F 各有 13 条 `invalid_json`、5 条 `ambiguous_repeated_quote`，18/30 条没有接受项；接受的 15 次引用均逐字匹配。共享主语样本 `MNB-shared_subject-1` 的原始输出在 JSON 数组后多出一个 `}`，因此被拒绝；重复引文保留全部候选位置而不擅自选择首处。反馈分别 30/30 次返回 `NO_ISSUE`，没有实际修订。上述是可观察的解析和反馈失败，不能单独确定各项对最终准确率的因果贡献。

D/E/F 的平均耗时为 1.74 / 36.89 / 37.05 秒。F 的结果没有超过 D，且成本更高；本次不升级默认方法。若后续改进 JSON 约束、重复引文处理或反馈触发，需要新开发版本和新冻结记录，不能覆盖当前失败结果或把再次调整后的边界结果当作未见测试。

## 验证记录

本地完整单测：`python -X utf8 -m unittest discover -s tests -q`，设置 `MINIFACT_TEST_TOKENIZER` 为固定 Qwen 快照后，**285 tests / OK**（33.428 秒）。未加载第二个 GPU 模型。

以下命令均退出 0：

```powershell
python scripts/build_testset.py --check
python scripts/validate_phase1.py
python scripts/build_phase3_testset.py --check
python scripts/build_improvement_data.py --check
python scripts/next_eval.py check --bundle data/next_eval_v1
python scripts/next_eval.py check --bundle data/next_eval_v2 --source E:/MiniFact-LLM-runtime/datasets/averitec/train.json
python scripts/next_eval.py audit --run reports/next_eval_dev_20261122.json --freeze reports/next_eval_dev_20261122.freeze.json --cache-dir E:/MiniFact-LLM-runtime/hf-cache --output reports/next_eval_dev_20261122.audit.json
python scripts/next_eval.py audit --run reports/next_eval_boundary_v2_20261122.json --freeze reports/next_eval_boundary_v2_20261122.freeze.json --cache-dir E:/MiniFact-LLM-runtime/hf-cache --output reports/next_eval_boundary_v2_20261122.audit.json
python scripts/next_eval.py evaluate --run reports/next_eval_boundary_v2_20261122.json --freeze reports/next_eval_boundary_v2_20261122.freeze.json --output reports/next_eval_boundary_v2_20261122.metrics.json
python scripts/report_next_boundary.py --run reports/next_eval_boundary_v2_20261122.json --metrics reports/next_eval_boundary_v2_20261122.metrics.json --audit reports/next_eval_boundary_v2_20261122.audit.json --output reports/next_eval_boundary_v2_report_20261122
```

开发审计：32/32 样本，192 组结果、346 次生成、64 次候选前向；346 次提示/token 重放通过，0 次生成缺少 EOS，0 次反馈转移。审计核对保存值和 tokenizer 重建，未重新运行全部模型 logits。

边界审计：30/30 样本，90 组结果、240 次生成及 tokenizer 重放通过，0 次生成缺少 EOS，0 次反馈转移。两次完整运行均无中断，记录的执行环境及 EOS 配置完全一致。边界图表已视觉检查，逐类计数、逐条预期和预测保留；图表内容不依赖模糊的综合通过率。

发布验证使用 E 盘的独立暂存字节副本：285 项单测再次通过（36.389 秒），完整开发、边界审计及既有 80 条主实验/候选阈值审计均通过。检查的是 Git 实际将提交的内容。初次格式检查发现 Matplotlib SVG 的生成空白和报告末尾空行；为保持图表 manifest 哈希，`.gitattributes` 保留 SVG 原始字节并对这些生成位置定向放宽空白检查。20 个 PNG/SVG/报告文件的暂存对象哈希与图表 manifest 一致，最终 `git diff --cached --check` 通过；没有修改已冻结推理代码或实验数值。

`readiness --bundle data/next_eval_v2` 与 `freeze --stage news --bundle data/next_eval_v2 --dev-run reports/next_eval_dev_20261122.json --output <new-file>` 均按预期退出 1，原因为缺少真实 reviewer pseudonym；正式 freeze 文件未生成。这是待人工工作的有效阻断，不是已经通过验收。

## 人工接续与验收

将 [A 模板](../data/next_eval_v2/reviewer_a.json) 和 [B 模板](../data/next_eval_v2/reviewer_b.json) 分别交给两位标注者，按 [指南](../docs/NEXT_EVALUATION_ANNOTATION.md) 独立完成后共同仲裁。协调者使用同版本 event_review/adjudication 文件。若发现需排除的来源歧义或事件重复，先建新版本并重新核对替补，不覆盖已有记录。

两份标注和仲裁就绪后，正式冻结并运行 200 条六组比较；完成后再由人工对齐 E/F 在 60 条上的全部 raw 预测，最后独立判断“研究交付完成”和“默认方法升级”。未达到升级门槛也应完整报告，不能删除失败样本或事后换主比较。

当前新闻标签继承公开基准，并非重新确认现实真伪。项目历史去重无法排除模型预训练污染；固定类别平衡也不代表真实新闻分布。200 条并不保证检测到 5 个百分点的差异，实际配对区间决定证据强度。正式结果和人工语义成绩均为待验证边界。
