# 保留关系与上下文的抽取

`accuracy_claims.py` 是新的实验组件，不修改既有冻结抽取器、定位器或 Phase 2 / Week 6 结果。它把可核对的逐字引用与模型重构的核查命题分开，避免把补写主语或省略宾语的文本当作原文切片。

模型输出格式：

```json
[
  {"quote": "Alice owns three boats", "proposition": null, "context_quotes": []},
  {"quote": "four cars", "proposition": "Alice owns four cars", "context_quotes": ["Alice owns"]}
]
```

例子的原文为 `Alice owns three boats and four cars.`。`quote` 和每个 `context_quotes` 字符串都必须是唯一可定位的连续原文。程序使用原有 `minifact.align_quotes` 计算半开字符区间，不接受模型给出的索引。改写、不明确的重复引文和重复主引用会保留拒绝记录；上下文在不同主张之间复用是允许的。如果上下文引用无法定位，整个预测项目拒绝，不能靠一个看似正确的重构命题绕过引用检查。

`proposition: null` 表示直接使用引用作为待核查命题，减少输出 token。非空 `proposition` 允许补全共享主语或谓语，但明确保存为 `unverified_model_reconstruction`，不会获得原文索引，也不能作为独立证据。所有已接受项目均标为 `semantic_status: not_independently_verified`；`null` 不代表已经确认原子性或真实性。

提示词要求保留主语、谓语、宾语或补足语、归因、否定、数量、条件和时间范围。尤其不能把“某人签署了要求做某事的行政令”弱化成“某人签过行政令”后，独立核查这个较弱命题。最终裁决器仍须接收完整原文，逐项分析只能作为未证实的模型意见。

`parse_extraction(source, raw_response)` 返回 `raw_items`、`accepted`、`rejected`、`errors`。`accepted` 中保留逐字 `claim`、原预测顺序 `quote_index`、程序计算的 `start_idx` / `end_idx`、`proposition`、`context_dependencies` 和 `guard_flags`。无法解析的输出不会被移除 Markdown 或补齐 JSON 后悄悄修复。

英文启发式只提供复核提示：可能的代词依赖、动词开头或很短的片段、泛化关系后的补足语遗漏，以及重构命题相对引用/上下文的数字、否定变化。数字写法变化、正确的同义改写和某些完整句也可能触发提示。这些提示**不自动拒绝预测、不判真伪、不证明原子性**；没有提示也不等于语义完整。英文规则不用于宣称跨语言语法检查能力。

字符位置、事实提全、限定保留、原子性和最终真伪仍应分别评估。该组件不是独立人工金标准；不能只凭位置检查通过或启发式提示数量下降就宣称抽取更准确。

离线测试：

```powershell
python -m unittest discover -s tests -p test_accuracy_claims.py -v
```
