# Days 1–2 交付记录

日期：2026-10-21。状态：数据策划完成，正式模型实验尚未运行。

正式交付为 `data/phase1/dataset.json`：**60 条英文真实新闻相关错误主张**，均来自固定版本 AVeriTeC dev 的 Refuted 样本，原文逐字保存。覆盖 18 个核查来源域名，附 142 个核查问题与 151 条答案。它们是短主张，而非完整文章；包含复合主张时，Refuted 只适用于整条样本。

筛选流程：原始 500 条 → 305 条 Refuted → 178 条满足筛选规则 → 分层候选 80 条 → 排除 20 条解释或范围不清的记录 → 冻结 60 条。候选数量不是模型结果。规则、逐条排除理由、固定来源 revision 和最终行号保存在 `data/phase1/curation.json`；源文件 SHA256 在重建时强制校验。

主数据与模型输入已分开：`data/phase1/inputs.json` 仅含 ID、原文；schema、README 和来源/许可说明一并交付。数据使用 CC BY-NC 4.0，并保留作者、论文、原始核查文章和证据链接。

已执行的检查：

| 命令 | 结果 |
| --- | --- |
| `python -X utf8 scripts/validate_phase1.py` | PASS：60 条、18 个核查域名、142 个问题、151 条答案；schema、记录哈希、重复、日期、清单、统计与输入隔离通过 |
| `python -X utf8 scripts/curate_phase1.py --source <缓存目录>/dev.json --check` | PASS：原始 SHA256 匹配；重新生成的两份 JSON 与交付文件逐字节一致 |
| `python -X utf8 -m unittest discover -s tests -v` | 22/22 通过，包括 9 项正式数据保护测试与原有 13 项抽取/比较测试 |
| `python -X utf8 scripts/build_testset.py --check` | PASS：旧回归集及三个派生文件一致 |
| `git diff --check` | PASS |

旧 `data/cases.json` 的 SHA256 仍为 `cf3ca74c4028a490d579ada403c44113a9abe0ce105ed762bf24c965bd0188d6`。冻结的抽取代码、提示词、模型设置及前期结果未混入本次数据策划。

限制：裁决继承公开基准，筛选属于 agent 对所提供文本的审阅，没有独立人工二次核查或逐条在线取证。样本集中在 2020 年，可能被模型见过；全部为错误正例，不能据此测误报率。原始主张类型不是错误类型标注，本版没有单独标注的纯逻辑错误子集。

下一步：Days 3–4 标注原子主张、限定条件与原文引用区间，并冻结评估协议。此处这些字段明确为 `null`/pending，没有为 60 条样本报告未经执行的事实覆盖、位置或 log-probability 结果。
