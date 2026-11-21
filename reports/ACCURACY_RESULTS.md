# 准确率改进：开发集选择与 80 条留出评估

模型仍为固定 revision 的 Qwen2.5-3B-Instruct，FP16、eager attention；没有更换权重或微调。原有 110 条 Week 6 数据、代码及运行结果保留。这里另用 32 条开发集（16/16）和 80 条项目未见留出样本（40/40）。

新闭卷候选由开发集选为 `structured`，其留出准确率为 45/80（56.25%），相对同一留出集上的旧双角色差 +1.25 个百分点。单独拟合的标签阈值方法为 54/80（67.50%），对应差 +12.50 个百分点。全部预定条件均报告，不能根据留出分数再更换候选或阈值。

本表与旧 110 条成绩的分布和样本不同，不能直接比较两个绝对百分比后声称提升。有效比较来自下面在**同一 80 条**上重新运行的条件。全部判为同一个类别的常数参考是 40/80（50%）；它是算术背景，不是额外模型分支。

## 留出结果与代价

| 条件 / 方法 | 判对 / 80 | 准确率 | 严格 FNR | 含弃答未检出率 | FPR | 二分类覆盖 | 累计秒 | 模型调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧直接提示 [闭卷] | 35 | 43.75% | 0.00% | 12.50% | 60.00% | 73.75% | 86.46 | 80 |
| 旧双角色流程 [闭卷] | 44 | 55.00% | 0.00% | 10.00% | 70.00% | 90.00% | 895.55 | 425 |
| 关系/上下文抽取 + 二分类 [闭卷] | 45 | 56.25% | 0.00% | 0.00% | 87.50% | 100.00% | 754.19 | 160 |
| 开发集拟合标签阈值 [闭卷] | 54 | 67.50% | 30.00% | 30.00% | 35.00% | 100.00% | 35.22 | 160 |
| 提供参考 QA 与元数据 **[提供证据]** | 64 | 80.00% | 0.00% | 7.50% | 30.00% | 95.00% | 45.25 | 80 |

正类是错误主张（REFUTED）。严格 FNR 只计错误被判 SUPPORTED / 40；含弃答未检出率还计 NOT_ENOUGH_INFO 与无效输出。FPR 是正常主张被判 REFUTED / 40。所有弃答和无效输出均保留在 80 条准确率的分母中。强制二分类方法请求模型作答，其较高覆盖不能单独当作更强的事实判断能力。

调用数中，旧流程与新提示的单位为 `generate` 调用，标签阈值方法的单位为完整 teacher-forced 前向调用；它每条做两次前向，不生成文本。累计耗时保留实际观测，排除模型加载和检查点写入，不是严格隔离的硬件性能基准。

**提供证据条件单独解释：**它比闭卷条件多收到经筛选的上游 QA、主张日期和国家/说话人元数据；模型不读取金标签、裁决解释或证据 URL 字符串。证据由 benchmark 预先提供，不是本系统检索获得，不能把这行的差异归因于提示词或当作端到端网络核查成绩。QA 本身可能直接解答主张，筛选与截断也影响结果。

## 开发过程与冻结选择

| 开发条件（32 条） | 判对 / 32 | 准确率 | 严格 FNR | FPR | 二分类覆盖 | 累计秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧直接提示 | 8 | 25.00% | 0.00% | 37.50% | 43.75% | 47.21 |
| 旧双角色流程 | 15 | 46.88% | 6.25% | 62.50% | 81.25% | 467.81 |
| 中性三分类提示 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 36.99 |
| 强制二分类提示 | 14 | 43.75% | 12.50% | 68.75% | 84.38% | 42.15 |
| 关系/上下文抽取 + 二分类 | 16 | 50.00% | 12.50% | 87.50% | 100.00% | 355.51 |
| 提供参考 QA 与元数据 | 24 | 75.00% | 0.00% | 37.50% | 93.75% | 28.14 |

闭卷候选按开发集准确率选定为 `structured`；并列时依次比较较低 FPR、较短平均耗时和方法名。开发集用于选择，成绩带有选择偏差，不能代替留出结果。提供 QA 条件没有参与闭卷候选排名。

标签阈值是另一个在留出执行前固定的诊断方法。使用相同开发集，零阈值候选分数判对 15/32（46.88%）；拟合后判对 20/32（62.50%），阈值为 `-9.10630331975`。拟合时 FPR 为 12.50%，严格 FNR 为 62.50%。这显示降低正常类误报可能伴随大量错误被判真的代价，开发集收益并不保证泛化。

阈值拟合只校正二分类决策边界：分别计算完整 `TRUE + EOS` 与 `FALSE + EOS` 的原始 token logprob 总和，再用两者之差与开发集阈值比较。分数不作长度归一化；这与实际 greedy 生成的标签 token 均值不同，也不是校准后的真假概率。没有为提高测试分数而在留出标签上拟合阈值。

## 阈值零值消融与配对变化

| 同一留出候选分数 | 准确率 | 严格 FNR | FPR |
| --- | ---: | ---: | ---: |
| 阈值 0（无新增推理） | 56.25% | 0.00% | 87.50% |
| 固定开发阈值 | 67.50% | 30.00% | 35.00% |

零阈值参照使用同一批保存的候选分数，规则在首次留出运行前声明；不是看过留出结果后挑选的另一个最佳阈值。

| 相对旧双角色 | 双方判对 | 双方未判对 | 仅旧流程判对 | 仅新条件判对 | 准确率差（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧直接提示 | 32 | 33 | 12 | 3 | -11.25 |
| 关系/上下文抽取 + 二分类 | 38 | 29 | 6 | 7 | +1.25 |
| 提供参考 QA 与元数据 | 42 | 14 | 2 | 22 | +25.00 |
| 开发集拟合标签阈值 | 33 | 15 | 11 | 21 | +12.50 |

配对数量完整保存在组合 JSON 中，描述同一批样本上的判对变化；不能隔离额外计算、原子抽取、反馈或分数的独立因果贡献，也不能使提供证据与闭卷成为同信息条件。

## 抽取维度单独保留

留出结构化条件记录 77 次接受引用、1 条项目拒绝记录。组合汇总逐一检查了接受引用的 `source[start_idx:end_idx] == claim`。跨项目/调用出现次数不是独立事实数，项目拒绝也不能概括所有顶层 JSON 解析错误。

| 维度 | 可支持的结论 |
| --- | --- |
| 原文位置 | 接受的引用逐字匹配；重复歧义、改写等拒绝保留 |
| 事实是否提全、是否独立 | 缺少独立原子金标准，未报告正式覆盖率或原子性准确率 |
| 限定条件保留 | 未建立独立限定参考标注，未报告正式保留率 |
| 主语/谓语补全 | 与逐字引用分字段；重构命题及启发式警报都未被视为事实证据 |

警报计数：`{"declared_context_dependency": 3, "nonliteral_proposition_requires_review": 2, "possible_clause_fragment": 2, "possible_pronoun_dependency": 1}`。

拒绝原因：`{"no_literal_match": 1}`。

启发式警报减少或全部位置匹配，都不能证明语义抽取更准确。原文始终提供给最终裁决器；已补写的命题没有冒充原文跨度。

## 使用与复现

需要原固定模型快照和 CUDA 环境，模型/缓存留在仓库外。以下均使用不同输出名，工具拒绝覆盖已有文件。第一个入口使用开发集选定的新提示方法；第二个是单独的候选分数阈值方法。

```powershell
python scripts/check_accuracy.py --text "<待核查主张>" --output runs/local_check.json
python scripts/check_calibrated_accuracy.py --text "<待核查主张>" --output runs/local_threshold.json
python scripts/check_accuracy.py --method evidence --input-file statement.txt --evidence-file packet.json --output runs/local_evidence.json
```

`packet.json` 只含 `context`（claim_date、speaker、location_ISO_code）和 `evidence`（question、answer、url）；不要加入 gold、verdict 或 justification。来源 QA 需用户提供，工具不执行网络检索；闭卷方法拒绝证据文件参数。输出的标签、生成分数或阈值判决均为未证实的模型意见。

方法说明：[ACCURACY_IMPROVEMENT.md](../docs/ACCURACY_IMPROVEMENT.md)；抽取字段：[ACCURACY_EXTRACTION.md](../docs/ACCURACY_EXTRACTION.md)；数据及来源：[README.md](../data/improvement/README.md)；许可与署名：[NOTICE.md](../data/improvement/NOTICE.md)；开发历史：[ACCURACY_DEVELOPMENT_HISTORY.md](ACCURACY_DEVELOPMENT_HISTORY.md)。

## 原始记录与核验

本报告输入组合指标 SHA256：`2e3be7b253c2f0af23203b3ae405d4a347ac9fa1aa324e42b920c367c9d1fcc8`。报告生成器 SHA256：`18cb8544b997c7ef8fac4ede1b85e80f6a38ba8736beaffec8e48405c2601836`。生成器只读取已完成记录和绑定的字节哈希，不运行模型、不执行阈值拟合、不据留出成绩重新选方法。

调用者记录的本地测试通过数：**254 项**。该数由 `--tests-passed` 传入；本生成器不执行测试，也不把它当作模型正确性证明。

- [accuracy_holdout_run_20261112.json](accuracy_holdout_run_20261112.json)：`d3f5f4ee3813343b7b1e80b2981e557e5f9ed91848610c819f6ad54ac1411d6f`。
- [accuracy_holdout_metrics_20261112.json](accuracy_holdout_metrics_20261112.json)：`d8b7fdc793747767445d045621b435cb7264da03976d974c004bd7f447255fc1`。
- [accuracy_calibration_holdout_20261112.json](accuracy_calibration_holdout_20261112.json)：`237db0dc32c4c4aa058477d01140d0acda24e5633e0d33bde22bee8da66c3488`。
- [accuracy_label_threshold_20261112.json](accuracy_label_threshold_20261112.json)：`1b852c26587ec728dd1a0bcbf85500f7a2480f4166e163baedb82a6d443628b9`。
- [holdout_dataset.json](../data/improvement/holdout_dataset.json)：`4a8eb765c6522b6319e7ee82d086f5da7d7bc74b1aa724cb97f0a87b5052e4b9`。
- [summarize_accuracy_comparison.py](../scripts/summarize_accuracy_comparison.py)：`31d435e26e499a3e5e8274fbc9e9b52824a9c979d7b675ecf4c1b22602d082e5`。
- [accuracy_dev_freeze_v2_20261112.json](accuracy_dev_freeze_v2_20261112.json)。
- [accuracy_dev_run_v2_20261112.json](accuracy_dev_run_v2_20261112.json)。
- [accuracy_dev_metrics_v2_20261112.json](accuracy_dev_metrics_v2_20261112.json)。
- [accuracy_dev_audit_v2_20261112.json](accuracy_dev_audit_v2_20261112.json)。
- [accuracy_selection_20261112.json](accuracy_selection_20261112.json)。
- [accuracy_holdout_freeze_20261112.json](accuracy_holdout_freeze_20261112.json)。
- [accuracy_holdout_audit_20261112.json](accuracy_holdout_audit_20261112.json)。
- [accuracy_calibration_dev_20261112.freeze.json](accuracy_calibration_dev_20261112.freeze.json)。
- [accuracy_calibration_dev_20261112.json](accuracy_calibration_dev_20261112.json)。
- [accuracy_calibration_dev_audit_20261112.json](accuracy_calibration_dev_audit_20261112.json)。
- [accuracy_calibration_holdout_20261112.freeze.json](accuracy_calibration_holdout_20261112.freeze.json)。
- [accuracy_calibration_holdout_audit_20261112.json](accuracy_calibration_holdout_audit_20261112.json)。

## 解释边界

新开发/留出数据来自固定版本 AVeriTeC 的上游 train 分区，与旧 110 条上游 dev 来源不同；“项目未见”只表示此前项目运行未使用，不意味着模型预训练未见。两组新数据均要求有可用 QA、排除需重标注记录，并做文本/核查文章去重，属于有质量筛选的平衡子集；没有证明事件完全独立或代表现实新闻比例。

上游整条标签没有独立人工重新核实，不能继承为每个原子主张的真伪。提供证据条件有额外 QA 和元数据，存在先验整理的优势；没有完成自动检索。闭卷方法可能重复同一知识错误；阈值是开发集拟合的决策规则，不是概率校准。

留出执行完成后，这 80 条可以用于解释失败，但任何据此修改方法后的泛化提升都需要新的未触碰评估数据。当前报告保留全部条件、失败和代价，不承诺未来样本上的准确率收益。
