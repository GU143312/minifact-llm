# Week 6 / Days 11–12 比较结果

2026-11-12：完成整条新闻主张的闭卷系统比较。保留原 60 条 Refuted，新增 50 条 Supported，共 110 条；两分支全部运行并保留原始输出。金标签继承固定版本 AVeriTeC，衡量的是 benchmark 标签一致性，不是独立人工重新核验的真伪。

双角色系统的准确率提高 14.55 个百分点，含弃答的错误未检出率下降 8.33 个百分点；但严格漏报率从 0% 升到 6.67%，误报率从 60% 升到 70%，累计耗时约为直接提示的 13.14 倍。直接组从未输出 SUPPORTED，其低严格漏报率伴随大量弃答，不能解释为可靠检测。当前闭卷原型仍有明显的正常主张误报问题。

直接分支仅看原文；双角色分支先运行最多两轮 Phase 2 抽取—验证，再把同模型未证实的分析与分数交给相同的整条样本裁决器。模型、revision、FP16、最终判决提示词、16 token 上限及 greedy 参数相同。完整方法见 [实验协议](../docs/PHASE3_WEEK6.md)。双角色分支使用更多计算，不能把差异单独归因于反馈或 logprob。

| 指标 | 直接提示 | 双角色 + 整条裁决 |
| --- | ---: | ---: |
| 整条准确率（弃答也计入分母） | 41.82% | 56.36% |
| 错误召回率 | 76.67% | 85.00% |
| 严格 FNR：错误被判 SUPPORTED / 60 | 0.00% | 6.67% |
| 含弃答的错误未检出率 / 60 | 23.33% | 15.00% |
| 误报率：Supported 被判 REFUTED / 50 | 60.00% | 70.00% |
| 有效二分类覆盖率 | 69.09% | 91.82% |
| 有效二分类条件准确率 | 60.53% | 61.39% |
| 判对数量 | 46 | 62 |
| 错误样本中的弃答数 | 14 | 5 |
| 错误样本中的无效输出数 | 0 | 0 |

配对准确率差（双角色 − 直接）为 14.55 个百分点；仅直接判对 3 条，仅双角色判对 19 条，两边均判对 43 条，两边均未判对 45 条。确切双侧 McNemar p=0.000855，仅作描述；筛选样本和事件相关性限制了总体推断。

严格 FNR 只统计被明确判成 SUPPORTED 的错误；模型弃答也会造成错误未被检出，须同时读含弃答的未检出率，不能用低严格 FNR 掩盖大量弃答。原 60 条已经参加此前实验；新增正常对照没有按国家、类型、句长或域名统计匹配。

## 直接提示：混淆矩阵

| 上游标签 | SUPPORTED | REFUTED | NOT_ENOUGH_INFO | INVALID |
| --- | ---: | ---: | ---: | ---: |
| REFUTED | 0 | 46 | 14 | 0 |
| SUPPORTED | 0 | 30 | 20 | 0 |

## 双角色 + 整条裁决：混淆矩阵

| 上游标签 | SUPPORTED | REFUTED | NOT_ENOUGH_INFO | INVALID |
| --- | ---: | ---: | ---: | ---: |
| REFUTED | 4 | 51 | 5 | 0 |
| SUPPORTED | 11 | 35 | 4 | 0 |

## 分数与固定阈值

下列分数来自最终判决实际 greedy 生成 token 的原始 logits，EOS/特殊 token 不计入标签均值。此处报告每条样本标签 token 均值的平均，不是 token 加权均值；logprob 与全词表熵（nats）均不是校准后的真假概率。

| 分数 | 直接提示 | 双角色 + 整条裁决 |
| --- | ---: | ---: |
| 平均标签 logprob | -0.038370 | -0.046525 |
| 平均全词表熵（nats） | 0.072129 | 0.109856 |

| 分支 | logprob 下限 | 保留二分类 / 110 | 覆盖率 | 保留决定准确率 | 含弃答的错误未检出率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | -0.05 | 59 | 53.64% | 64.41% | 36.67% |
| direct | -0.1 | 67 | 60.91% | 64.18% | 28.33% |
| direct | -0.25 | 76 | 69.09% | 60.53% | 23.33% |
| direct | -0.5 | 76 | 69.09% | 60.53% | 23.33% |
| dual | -0.05 | 73 | 66.36% | 64.38% | 28.33% |
| dual | -0.1 | 87 | 79.09% | 62.07% | 20.00% |
| dual | -0.25 | 100 | 90.91% | 61.00% | 15.00% |
| dual | -0.5 | 101 | 91.82% | 61.39% | 15.00% |

所有阈值在正式运行前固定，完整列出，不按测试结果挑选最佳阈值。标签长度和条件分布影响 logprob；高分的错误判断仍可能出现。

## 成本与抽取检查

| 成本 | 直接提示 | 双角色 + 整条裁决 |
| --- | ---: | ---: |
| 模型调用数 | 110 | 686 |
| 各调用输入 token 总和 | 20041 | 163415 |
| 输出 token 总数（含 EOS） | 474 | 6258 |
| 未生成 EOS 的调用数 | 0 | 0 |
| 分支累计耗时（秒） | 102.31 | 1344.74 |

耗时在每个分支前后同步 CUDA，包含该分支 Python 编排，不含模型加载、检查点写入与离线评分。输入 token 在每次调用中重复计数，反映实际请求预算；不是唯一文本 token 数。耗时是本机这次运行的观测，期间有 CPU 校验任务，不作隔离的性能基准。

双角色内部共 123 轮、13 次实际反馈传递；165 个引用出现次数可唯一逐字定位，49 条拒绝记录。跨轮重复不算不同事实。拒绝原因：{'no_literal_match': 41, 'expected_nonempty_string_array': 7, 'ambiguous_repeated_quote': 1}。所有已接受引用均通过 `source[start_idx:end_idx] == claim`；不能因此推断事实提全或原子性正确。

| 抽取维度 | 本周结论 |
| --- | --- |
| 事实是否提全、是否独立 | 无独立原子金标准，未计算正式覆盖率或原子性准确率 |
| 限定条件是否保留 | 无独立限定条件金标准，未计算正式保留率 |
| 位置是否正确 | 对所有已接受引用逐字切片核验，并保留拒绝记录及原因 |

## 复核与实验边界

完整自动审计通过：代码、输入、标签数据及冻结哈希一致，两个最终裁决提示与实际模型分析一致，内部反馈传递、逐字索引、生成参数、EOS、标签 token 对齐和保存分数均值检查通过。自动审计只能检查记录一致性；不证明模型的语义判断正确。

CPU tokenizer 重放对完整运行中的全部 796 次生成调用核对实际提示词 token ID、输出解码与特殊 token；结果见 `phase3_week6_prompt_replay_20261112.json`。固定复核完成运行首条样本的两个最终判决，重新加载同 revision 模型作完整前向计算（含 EOS），没有按正确性挑选样本：

- direct：最大 token logprob 绝对差 0.00156221；全词表熵差 0.00626465 nats。
- dual：最大 token logprob 绝对差 0.00543500；全词表熵差 0.01158717 nats。

复核仅覆盖一条样本的两个 trace。FP16 完整前向与逐 token KV cache 生成可能有数值差异；记录实际差值，不宣称所有分数逐位相同。

首个冻结因数据清单并发补入两个校验字段而停止，已完成的 3 条部分结果完整保留且不计分；模型、提示词、代码、阈值、110 条输入和标签均未改变，第二次冻结后从头运行。详情见 [运行历史](PHASE3_RUN_HISTORY.md)。冻结时间是本机时间，不是独立可信时间戳。

两分支均闭卷，无独立检索证据；相同权重的模型角色可能重复同一知识错误。标签对应原 2020 年新闻日期，而模型只收到主张原文，没有额外日期/出处元数据；时间敏感或省略上下文的主张可能因此受影响。公共旧基准可能存在预训练暴露。新增对照经过 agent 筛选，没有独立人工重裁；文章去重不保证事件独立。结果不得外推为现实新闻的总体准确率。

## 逐条最终判决

| ID | 上游整条标签 | direct | dual | direct 均值 logprob | dual 均值 logprob |
| --- | --- | --- | --- | ---: | ---: |
| MF-P3-C-0026 | SUPPORTED | NOT_ENOUGH_INFO | REFUTED | -0.004424 | -0.052668 |
| MF-P1-0022 | REFUTED | REFUTED | REFUTED | -0.002405 | -0.000598 |
| MF-P1-0047 | REFUTED | REFUTED | REFUTED | -0.023347 | -0.006135 |
| MF-P3-C-0047 | SUPPORTED | REFUTED | SUPPORTED | -0.061690 | -0.015801 |
| MF-P1-0009 | REFUTED | REFUTED | REFUTED | -0.000016 | -0.000004 |
| MF-P3-C-0032 | SUPPORTED | REFUTED | REFUTED | -0.119435 | -0.039748 |
| MF-P1-0053 | REFUTED | REFUTED | SUPPORTED | -0.000118 | -0.017527 |
| MF-P1-0050 | REFUTED | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO | -0.000718 | -0.152518 |
| MF-P3-C-0005 | SUPPORTED | REFUTED | REFUTED | -0.000060 | -0.000107 |
| MF-P3-C-0016 | SUPPORTED | REFUTED | REFUTED | -0.122609 | -0.010750 |
| MF-P3-C-0037 | SUPPORTED | REFUTED | REFUTED | -0.013820 | -0.017041 |
| MF-P1-0038 | REFUTED | REFUTED | REFUTED | -0.000044 | -0.008898 |
| MF-P1-0027 | REFUTED | REFUTED | REFUTED | -0.000069 | -0.001104 |
| MF-P3-C-0034 | SUPPORTED | REFUTED | REFUTED | -0.010595 | -0.008531 |
| MF-P1-0010 | REFUTED | NOT_ENOUGH_INFO | SUPPORTED | -0.097148 | -0.061141 |
| MF-P3-C-0006 | SUPPORTED | REFUTED | REFUTED | -0.056868 | -0.026109 |
| MF-P3-C-0008 | SUPPORTED | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO | -0.125377 | -0.168735 |
| MF-P3-C-0039 | SUPPORTED | REFUTED | REFUTED | -0.001114 | -0.008824 |
| MF-P3-C-0011 | SUPPORTED | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO | -0.014733 | -0.035233 |
| MF-P3-C-0033 | SUPPORTED | REFUTED | REFUTED | -0.211031 | -0.197017 |
| MF-P1-0024 | REFUTED | REFUTED | REFUTED | -0.001868 | -0.030747 |
| MF-P1-0044 | REFUTED | REFUTED | REFUTED | -0.024838 | -0.000961 |
| MF-P3-C-0046 | SUPPORTED | NOT_ENOUGH_INFO | REFUTED | -0.072589 | -0.108910 |
| MF-P1-0013 | REFUTED | REFUTED | REFUTED | -0.000051 | -0.000021 |
| MF-P3-C-0002 | SUPPORTED | REFUTED | REFUTED | -0.206082 | -0.003265 |
| MF-P1-0017 | REFUTED | REFUTED | REFUTED | -0.000086 | -0.001824 |
| MF-P3-C-0048 | SUPPORTED | REFUTED | NOT_ENOUGH_INFO | -0.053455 | -0.165688 |
| MF-P3-C-0050 | SUPPORTED | NOT_ENOUGH_INFO | REFUTED | -0.010615 | -0.136004 |
| MF-P3-C-0023 | SUPPORTED | REFUTED | REFUTED | -0.024844 | -0.087256 |
| MF-P1-0015 | REFUTED | REFUTED | REFUTED | -0.000038 | -0.002331 |
| MF-P3-C-0049 | SUPPORTED | REFUTED | SUPPORTED | -0.113205 | -0.074322 |
| MF-P1-0054 | REFUTED | NOT_ENOUGH_INFO | REFUTED | -0.118932 | -0.012363 |
| MF-P1-0001 | REFUTED | REFUTED | REFUTED | -0.030631 | -0.003405 |
| MF-P1-0006 | REFUTED | REFUTED | REFUTED | -0.000161 | -0.000355 |
| MF-P3-C-0044 | SUPPORTED | NOT_ENOUGH_INFO | SUPPORTED | -0.000559 | -0.040482 |
| MF-P1-0037 | REFUTED | REFUTED | REFUTED | -0.001721 | -0.015991 |
| MF-P1-0045 | REFUTED | NOT_ENOUGH_INFO | REFUTED | -0.001541 | -0.031544 |
| MF-P1-0043 | REFUTED | REFUTED | NOT_ENOUGH_INFO | -0.051956 | -0.158397 |
| MF-P1-0005 | REFUTED | REFUTED | REFUTED | -0.034636 | -0.060978 |
| MF-P1-0007 | REFUTED | REFUTED | REFUTED | -0.000027 | -0.000133 |
| MF-P1-0060 | REFUTED | REFUTED | REFUTED | -0.049202 | -0.182516 |
| MF-P1-0030 | REFUTED | REFUTED | REFUTED | -0.002627 | -0.061630 |
| MF-P1-0023 | REFUTED | REFUTED | REFUTED | -0.162652 | -0.015325 |
| MF-P3-C-0012 | SUPPORTED | REFUTED | REFUTED | -0.000027 | -0.000048 |
| MF-P3-C-0025 | SUPPORTED | REFUTED | REFUTED | -0.006692 | -0.034504 |
| MF-P1-0004 | REFUTED | REFUTED | REFUTED | -0.000631 | -0.000082 |
| MF-P3-C-0019 | SUPPORTED | REFUTED | REFUTED | -0.003321 | -0.012660 |
| MF-P3-C-0043 | SUPPORTED | NOT_ENOUGH_INFO | SUPPORTED | -0.014199 | -0.175504 |
| MF-P1-0035 | REFUTED | NOT_ENOUGH_INFO | REFUTED | -0.151337 | -0.003244 |
| MF-P1-0011 | REFUTED | REFUTED | REFUTED | -0.000040 | -0.000504 |
| MF-P3-C-0040 | SUPPORTED | REFUTED | REFUTED | -0.007783 | -0.141628 |
| MF-P3-C-0031 | SUPPORTED | NOT_ENOUGH_INFO | SUPPORTED | -0.000519 | -0.213502 |
| MF-P1-0026 | REFUTED | NOT_ENOUGH_INFO | SUPPORTED | -0.158513 | -0.248805 |
| MF-P1-0051 | REFUTED | REFUTED | REFUTED | -0.086414 | -0.000998 |
| MF-P1-0034 | REFUTED | REFUTED | REFUTED | -0.000078 | -0.004634 |
| MF-P1-0003 | REFUTED | REFUTED | REFUTED | -0.000021 | -0.000735 |
| MF-P1-0028 | REFUTED | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO | -0.001667 | -0.068529 |
| MF-P3-C-0028 | SUPPORTED | NOT_ENOUGH_INFO | SUPPORTED | -0.015394 | -0.009769 |
| MF-P1-0041 | REFUTED | REFUTED | REFUTED | -0.016256 | -0.004214 |
| MF-P3-C-0018 | SUPPORTED | REFUTED | REFUTED | -0.000147 | -0.006092 |
| MF-P1-0036 | REFUTED | REFUTED | REFUTED | -0.077488 | -0.026521 |
| MF-P1-0029 | REFUTED | REFUTED | REFUTED | -0.000616 | -0.001302 |
| MF-P1-0033 | REFUTED | REFUTED | REFUTED | -0.002811 | -0.001244 |
| MF-P3-C-0042 | SUPPORTED | NOT_ENOUGH_INFO | REFUTED | -0.137631 | -0.086275 |
| MF-P1-0012 | REFUTED | REFUTED | REFUTED | -0.000101 | -0.000365 |
| MF-P1-0019 | REFUTED | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO | -0.020094 | -0.061576 |
| MF-P1-0016 | REFUTED | REFUTED | REFUTED | -0.079489 | -0.060061 |
| MF-P1-0039 | REFUTED | REFUTED | REFUTED | -0.013929 | -0.114841 |
| MF-P3-C-0004 | SUPPORTED | REFUTED | REFUTED | -0.002366 | -0.000324 |
| MF-P3-C-0021 | SUPPORTED | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO | -0.011835 | -0.001935 |
| MF-P3-C-0020 | SUPPORTED | REFUTED | REFUTED | -0.035556 | -0.001733 |
| MF-P1-0020 | REFUTED | REFUTED | REFUTED | -0.018963 | -0.010071 |
| MF-P3-C-0017 | SUPPORTED | REFUTED | REFUTED | -0.000104 | -0.000275 |
| MF-P1-0048 | REFUTED | REFUTED | REFUTED | -0.005874 | -0.019683 |
| MF-P3-C-0038 | SUPPORTED | NOT_ENOUGH_INFO | REFUTED | -0.000847 | -0.000184 |
| MF-P1-0055 | REFUTED | REFUTED | REFUTED | -0.009376 | -0.029946 |
| MF-P1-0049 | REFUTED | REFUTED | REFUTED | -0.004572 | -0.000378 |
| MF-P3-C-0010 | SUPPORTED | REFUTED | REFUTED | -0.000310 | -0.004730 |
| MF-P1-0052 | REFUTED | REFUTED | REFUTED | -0.002311 | -0.000206 |
| MF-P3-C-0007 | SUPPORTED | NOT_ENOUGH_INFO | REFUTED | -0.000626 | -0.008932 |
| MF-P1-0042 | REFUTED | REFUTED | REFUTED | -0.225904 | -0.012417 |
| MF-P3-C-0014 | SUPPORTED | NOT_ENOUGH_INFO | REFUTED | -0.001138 | -0.138832 |
| MF-P3-C-0045 | SUPPORTED | REFUTED | REFUTED | -0.003925 | -0.011945 |
| MF-P3-C-0003 | SUPPORTED | NOT_ENOUGH_INFO | SUPPORTED | -0.018504 | -0.460289 |
| MF-P1-0058 | REFUTED | REFUTED | REFUTED | -0.000472 | -0.018571 |
| MF-P3-C-0035 | SUPPORTED | NOT_ENOUGH_INFO | REFUTED | -0.107735 | -0.049005 |
| MF-P1-0057 | REFUTED | NOT_ENOUGH_INFO | SUPPORTED | -0.128243 | -0.074887 |
| MF-P1-0032 | REFUTED | REFUTED | REFUTED | -0.128971 | -0.068259 |
| MF-P3-C-0027 | SUPPORTED | REFUTED | REFUTED | -0.140558 | -0.019464 |
| MF-P1-0008 | REFUTED | NOT_ENOUGH_INFO | REFUTED | -0.040378 | -0.053511 |
| MF-P3-C-0013 | SUPPORTED | REFUTED | REFUTED | -0.008313 | -0.003299 |
| MF-P1-0025 | REFUTED | REFUTED | REFUTED | -0.000607 | -0.000355 |
| MF-P1-0002 | REFUTED | REFUTED | REFUTED | -0.002365 | -0.003620 |
| MF-P1-0014 | REFUTED | NOT_ENOUGH_INFO | REFUTED | -0.169818 | -0.001140 |
| MF-P3-C-0036 | SUPPORTED | NOT_ENOUGH_INFO | REFUTED | -0.000650 | -0.062386 |
| MF-P1-0040 | REFUTED | REFUTED | REFUTED | -0.000093 | -0.000483 |
| MF-P3-C-0024 | SUPPORTED | NOT_ENOUGH_INFO | REFUTED | -0.066859 | -0.025743 |
| MF-P3-C-0030 | SUPPORTED | REFUTED | SUPPORTED | -0.018080 | -0.177630 |
| MF-P1-0059 | REFUTED | REFUTED | REFUTED | -0.010932 | -0.014989 |
| MF-P3-C-0015 | SUPPORTED | REFUTED | SUPPORTED | -0.042345 | -0.136176 |
| MF-P1-0021 | REFUTED | REFUTED | NOT_ENOUGH_INFO | -0.088787 | -0.039060 |
| MF-P1-0056 | REFUTED | REFUTED | REFUTED | -0.000110 | -0.148007 |
| MF-P3-C-0041 | SUPPORTED | NOT_ENOUGH_INFO | SUPPORTED | -0.104855 | -0.064270 |
| MF-P3-C-0009 | SUPPORTED | REFUTED | REFUTED | -0.014359 | -0.000469 |
| MF-P3-C-0029 | SUPPORTED | REFUTED | REFUTED | -0.000063 | -0.003220 |
| MF-P1-0018 | REFUTED | NOT_ENOUGH_INFO | REFUTED | -0.000582 | -0.016363 |
| MF-P1-0046 | REFUTED | NOT_ENOUGH_INFO | REFUTED | -0.026233 | -0.017495 |
| MF-P1-0031 | REFUTED | NOT_ENOUGH_INFO | REFUTED | -0.020760 | -0.001061 |
| MF-P3-C-0001 | SUPPORTED | REFUTED | SUPPORTED | -0.002018 | -0.019408 |
| MF-P3-C-0022 | SUPPORTED | NOT_ENOUGH_INFO | REFUTED | -0.131167 | -0.096470 |

## 文件与下一步

本地验证：在 Python 3.11.9 的模型环境中设置 `MINIFACT_TEST_TOKENIZER` 指向固定快照，执行 `python -X utf8 -m unittest discover -s tests -q`，159 项全部通过、无跳过。`scripts/build_testset.py --check`、`scripts/validate_phase1.py` 和 `scripts/build_phase3_testset.py --source <固定源缓存>/dev.json --check` 均通过。暂存 Git 对象中的冻结文件逐个 SHA256 一致；从暂存对象导出的独立目录再次执行 Phase 3 数据检查及完整评估也通过。额外使用独立标准库计数复算两组混淆矩阵、分母、分数均值和计算成本，与保存指标一致。

首次调用系统默认 Python 3.13 时缺少 `jsonschema`，未计为通过；随后用 `uv pip install --python <模型环境Python> -r requirements-data.txt` 安装数据校验依赖，并在原 Python 3.11.9 / Torch 2.10.0+cu128 模型环境完成上述检查。模型运行依赖的关键版本保持冻结配置。

- 数据及许可：`data/phase3/`。
- 完整正式记录：`phase3_week6_run_v2_20261112.json`；冻结：`phase3_week6_frozen_protocol_v2_20261112.json`。
- 审计和完整指标：`phase3_week6_metrics_v2_20261112.json`；输入、数据、运行哈希记录在其 `artifact_sha256`。
- logits 复核及全部提示 token 重放：`phase3_week6_logits_recheck_20261112.json`、`phase3_week6_prompt_replay_20261112.json`。

Week 6 / Days 11–12 的数据、完整运行、比较指标与报告已交付。下一周 Days 13–14 按计划做错误类型分析及可视化，不在本周调参改写这一冻结结果。
