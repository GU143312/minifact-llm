# 双人语义标注与事件复核指南

这些文件需要真实人工填写。程序不代签、不将 agent/模型意见登记为人工金标准。先独立完成两份，再共同裁决；请使用稳定化名，避免在公开文件加入个人信息。共享计划文档不进入标注包。

现可使用 [本地标注网页](ANNOTATION_WEB.md) 完成 A/B 独立初审：选择原文、填写表单、保存草稿并导出完成稿，不必直接编辑 JSON。下文仍定义最终数据的字段及仲裁规则。

## 1. 事件复核

当前包来自 `data/next_eval_v2`；不要使用旧 v1 或旧版标注压缩包。`event_review.json` 给出 200 条候选和此前 222 条历史文本，并已有明确标记为 `agent_assisted_manual_review` 的初筛记录。20 个额外排除候选及理由保留在同一文件；这不是人工真伪裁决。协调者可复核是否是同一事件的改写或细节变体，不能只看 URL。每条填写：

- `decision`: `include` 或 `exclude`。
- `event_key`: 同一事件使用相同的简短字符串，区分主体、事件、时间；不要机械复制样本 ID。
- `overlaps_history`: true/false；`reason` 写明与旧事件及其他候选比较后的理由。
- 顶层 `reviewer`、`reviewer_type`（`human` 或明确披露的 `agent_assisted_manual_review`）、`completed_at_utc`（含时区）。

发现应排除项时，读取其 `dataset.json` 中 `provenance.source_record_index`，用 `prepare --exclude` 在新的版本目录重建；把之前的排除索引全部保留。重新复核替补及新 60 条子集后再标注。不要覆盖已填标注或在看到模型预测后删样本。

## 2. 独立盲标

分别给两位标注者 `reviewer_a.json`、`reviewer_b.json`。文件不含金标签或证据。仅标注“原文作出了哪些断言”，不判断现实真假，不把传闻转为事实，也不要自行加入原文没说的年份或国家。

独立事实必须包含足够的主语、关系和宾语/补足语。“签署一项要求关闭学校的命令”不能弱化成“签过命令”和“关闭学校”。重复提及同一命题只计一个事实，并记录所有出现位置。共享主语、代词可在 `proposition` 中补全；补全文字不是逐字引用。

顶层填写不同的 `reviewer_id`、`completed_at_utc`；保持 `reviewer_type: human`。每个 item 保留原 `id` 和 `text`，完成后置 `status: complete`，填写非空 `facts`。

例如原文 `Mira opened a clinic in June.` 的一个事实：

```json
{
  "fact_id": "f1",
  "proposition": "Mira opened a clinic in June",
  "quote": "Mira opened a clinic in June",
  "occurrences": [{"start_idx": 0, "end_idx": 28}],
  "context_quotes": [],
  "qualifiers": [{"qualifier_id": "q1", "kind": "time", "text": "in June"}]
}
```

所有位置为 Python Unicode 字符索引，起点包含、终点不包含；不得按 UTF-8 字节或 JavaScript UTF-16 单元计数。`occurrences` 必须列出 quote 的所有逐字出现位置。上下文项为 `{"quote":"...","occurrences":[...]}`。限定类型允许 quantity、time、negation、condition、attribution、scope、causality、location。

可用离线 Python 确认一个引用的全部位置：

```python
from next_eval.data import spans
print(spans(source, quote))
```

不要为减少歧义而改写 quote；可以增加必要原文上下文。限定既包括显式数字/日期，也包括“仅”“至少”“据某人称”“如果”“因为”等范围和归因。

## 3. 共同裁决

两份完成后，计算字节 SHA256，填入 `adjudication.json` 的 `reviewer_a_sha256`、`reviewer_b_sha256`。`reviewer_ids` 按 A/B 顺序填写两人的化名。每项最终 `facts` 沿用相同字段，另外增加：

```json
"reviewer_a_fact_ids": ["a_f1"],
"reviewer_b_fact_ids": ["b_f2"]
```

如果一方漏提可以用空列表；如果其合并/拆分有误，要在 item 的 `resolution` 中解释。每项还需 `rejected_reviewer_a_fact_ids` 和 `rejected_reviewer_b_fact_ids`，用来完整处置未采纳的提案；没有则为空列表。每个原始提案只能被最终事实链接一次或被拒绝，不能悄悄消失。

所有 60 条解决后，将每项和顶层 `status` 置 `complete`，填写晚于两份初审的 `completed_at_utc`。尚有争议时保持 pending，不填假“完成”。程序输出事实数量一致率和裁决对齐后的提案重叠计数，不冒称独立 Cohen's kappa。

## 4. 模型运行后的语义对齐

`alignment-template` 只对完整正式新闻 run 生成模板，包含 E/F 的 60×2 条记录及全部可解析 raw 预测项。两名人工共同核对：

- `fact_ids`：核心主语—关系—宾语对应哪些最终参考事实；重复预测可指向同一事实，计分只算一次。
- `atomic`：是否只有一个完整独立事实；`faithful`：是否忠实于原文、没有新增或丢失重要含义。
- `retained_qualifiers`：正确保留的 `[fact_id, qualifier_id]` 对。
- `notes`：说明拆坏关系、丢限定、指代问题等。`content_error_types` 和 `system_failure_types` 分开填写；原因不明保持 `cause_status: undetermined`。

模板中的 raw 项、accepted 状态及位置不允许改写。程序使用已接受预测计算下游事实召回和限定保留；拒绝项仍进入 raw 预测的原子性/忠实性评阅。顶层无可解析项时，不能自行补一个预测。

顶层由两人签署 `reviewer_ids`、`completed_at_utc` 与 `status: complete`，每条 item 同样置 complete。报告保留人工对齐局限；模型输出中的理由或高 logprob 不能当作判对依据。
