# 字符到 token 的位置映射

`token_alignment.py` 是独立工具，不修改已有抽取和比较协议。字符位置采用 Python Unicode 字符索引，token 和字符区间均左闭右开。

1. 用实际推理所用的 chat template 渲染完整 `prompt`，保存推理时的 `input_ids`。`encode_prompt_offsets(tokenizer, prompt, input_ids)` 使用 fast tokenizer 对整个 prompt 编码，逐 token 检查 ID 与实际输入完全一致；不一致会报错，避免把孤立 source 的 token 下标误当作模型输入下标。若实际输入经过截断、padding 或其他处理，必须先明确并复现同一处理；当前辅助函数只支持未截断、未 padding 的单条输入。
2. `locate_unique_source(prompt, source)` 查找原文：重复或缺失返回 `ambiguous` / `not_found`，不猜位置。调用方也可由其他可靠方式确定源文本起点，但应校验 `prompt[source_start:source_start+len(source)] == source`。
3. 将原文引用 `[start_idx:end_idx]` 转换为完整 prompt 字符区间后，调用 `map_char_span(prompt, start, end, offsets, special_tokens_mask)`。`ok` 表示完整覆盖且边界与 token 边界相符；`partial_token` 表示 token 包含引用外的字符；`uncovered` 表示缺少字符映射；`non_contiguous` 表示所选 token 被特殊 token 等隔开。除 `ok` 外只可作为诊断，不能声称精确对齐。

特殊 token 和零宽 offset 不计入匹配；重叠 offset 允许存在。这个工具不提供 token 的概率，也不保证引用本身是独立事实。字符区间必须先验证为原文的精确切片。生成 token 的 log-probability 属于模型输出序列，不能未经额外方法直接分配给输入中的 token。

运行基础测试：`python -m unittest discover -s tests -p test_token_alignment.py -v`。若本地已有 Qwen fast tokenizer，在当前会话设置 `MINIFACT_TEST_TOKENIZER` 为缓存路径并使用含 transformers 的 Python 环境，可以额外运行真实 chat template 测试；不会下载模型。
