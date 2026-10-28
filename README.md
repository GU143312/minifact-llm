# MiniFact-LLM

小规模、可复现的事实抽取实验：用本地开源模型从复合句中抽取原子主张，保留原文引用，检查字符区间，并观察生成 token 的 log-probability。

项目比较 Windows 本地 GPU 与 Google Colab。两端使用同一份 `minifact.py`、固定提示词、模型 revision、FP16 精度及 greedy 生成参数。共享文档、个人项目安排和原始附件不属于公开仓库内容。

## 正式实验：Days 1–2 已完成

已策划并冻结 **60 条真实新闻相关错误主张**，改编自 AVeriTeC 公开基准的 Refuted 样本，附上游裁决、证据 QA、出处及固定来源版本。交付为[标准化 JSON](data/phase1/dataset.json)、[无答案输入](data/phase1/inputs.json)和 [JSON Schema](data/phase1/dataset.schema.json)；筛选、许可与复现方法见[正式数据说明](data/phase1/README.md)。样本是短主张，不是完整文章；全部为错误正例。

原计划 Days 3–4 的交付为抽取基线和字符到 token 对齐代码。正式数据的人工参考标注仍待完成；下文 15 句本地/Colab 结果属于前期合成回归实验，不能当作这 60 条上的正式结果。

## 正式数据的抽取和 token 对齐

`scripts/run_phase1_extraction.py` 使用既有冻结抽取提示词、revision、精度及生成设置，读取 `data/phase1/inputs.json` 中的 60 条无答案输入，并保存原始输出、引用和生成 token 的平均 log-probability。字符引用先验证为唯一的原文切片，再映射到完整聊天提示词的输入 token。映射细节和局限见 [token 对齐说明](docs/TOKEN_ALIGNMENT.md)。

```powershell
python scripts/run_phase1_extraction.py --cache-dir <模型缓存目录> --output runs/phase1_local.json
python scripts/summarize_phase1_extraction.py runs/phase1_local.json --output runs/phase1_summary.json
```

运行需要本地 CUDA、依赖与指定 revision 的缓存快照；已有输出不会被覆盖。只有 `ok` 是 token 边界精确对齐，`partial_token`、`uncovered` 和 `non_contiguous` 必须分别保留。字符切片精确并不意味着命题独立或事实没有遗漏；正式样本还没有经独立人工核验的参考标注，不能计算事实覆盖或限定条件保留率。

Phase 2 的有界抽取—验证交互见 [运行与方法说明](docs/PHASE2.md)：不同提示角色使用同一个模型，提供源内逻辑检查和反馈，并记录实际生成的判决 token 分数与熵。它是闭卷原型，没有外部证据检索；模型分数不是校准后的真值概率。早期 `scripts/run_phase1_verifier.py` 采用固定候选的 teacher-forced 评分，二者口径不同。

## 实验约定

- 模型：`Qwen/Qwen2.5-3B-Instruct`。
- Revision：`aa8e72537993ba99e69dfaafa59ed015b17504d1`。
- 精度：FP16，两端均可使用；attention 使用 eager，关闭 TF32，固定 seed=42。
- 生成：greedy、单 beam、最多 256 个新 token、repetition penalty=1；完整设置见 `config.json`。
- 模型只生成引用字符串数组；程序对唯一原文匹配计算 `start_idx` 和 `end_idx`。零基 Python 字符索引，右端不包含。
- 不修复改写，不猜重复引文的位置，不把被拒绝的引用从报告中删除。
- 提示词沿用先前本地改进版本。共享主语的省略、重复引用和复杂时间表达用来暴露它的限制，不针对本轮结果改提示词。

## 三个独立维度

`data/cases.json` 在运行前规定 15 个合成用例、29 个不同命题和 38 项限定，覆盖原三句、重复引文、共享主语、指代、否定及复杂时间。

测试集已按 6 类整理，见[数据说明](data/README.md)与[逐句参考答案](data/TEST_CASES.md)；全部参考字符位置保存在 `data/reference_spans.json`。原始 `cases.json` 与本轮实验所用文件保持一致。

1. **事实核心覆盖**：每个预定义命题的核心证据是否出现在同一条原始预测中。重复提到同一命题只算一个事实；不因为引用不能定位就把已经提到的事实算作漏提。
2. **限定保留**：逐项检查时间、数量、否定、模态和限制条件。报告保留数／全部限定数，并另列已覆盖事实所带的限定总数。事实核心提到但丢失时间，属于限定缺失。
3. **位置**：报告预测引用数、接受定位数、拒绝数及拒绝原因，并逐条检查 `source[start_idx:end_idx] == claim`。已接受引用的精确匹配由算法保证，必须连同被拒绝引用一起读，不能作为整体成功率。

另外单列一条预测覆盖多个金标准命题的情况，用于辅助原子性复核；没有合成的“总通过率”。核心证据规则是便于审计的操作性标注，不是通用语义裁判；结果需逐句人工复核。指代和共享主语可以依赖上下文，不强行把它们改写成不存在于原文的完整句。

## 本地运行

Python 3.11+，需要 CUDA GPU。创建独立环境后安装固定依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:MINIFACT_MODEL_CACHE = 'E:\MiniFact-LLM-runtime\hf-cache'
.\.venv\Scripts\python.exe minifact.py --label local --output reports/local_fp16.json --download
.\.venv\Scripts\python.exe evaluate.py reports/local_fp16.json --output reports/local_metrics.json
```

已有模型缓存时去掉 `--download`，即可仅用本地文件。大文件路径通过环境变量或 `--cache-dir` 设置，仓库不包含环境、缓存或模型权重。GPU 权重预算默认 5 GiB，其余层由 CPU 承载；两端使用相同预算，仍记录实际 device map。

## Colab 与受控比较

使用 `notebooks/controlled_comparison.ipynb`。它在独立工作目录 checkout 固定源码提交，创建独立 Python 环境，安装同版本依赖，然后调用同一个入口，而不是粘贴另一份改写脚本。不会覆盖旧 notebook 的历史输出。

拿到两端结果后：

```powershell
python compare.py reports/local_fp16.json reports/colab_fp16.json --output reports/controlled_comparison.json
```

比较器会检查完整用例、代码与数据哈希、提示词、模型 revision、实际精度、attention、关键库版本、实际提示词 token 哈希以及加载的生成默认值。不一致则拒绝标记为受控比较。GPU 架构、OS 和数值舍入仍可产生差异，不能把一次运行的速度差直接解释成硬件因果效应。

旧 Colab 与先前本地结果采用不同提示词、精度、输出结构、token 上限和依赖版本；`reports/legacy_configuration_audit.json` 只记录配置差异，不把它们当作这次受控实验。

## 检查与结果

本地与 Colab 的统一 FP16 比较已完成：[完整报告](reports/CONTROLLED_COMPARISON.md)。15 句原始响应全部一致；两端分别得到事实核心覆盖 29/29、限定保留 37/38、定位接受 24 条／拒绝 5 条。共同复现日期改写、补写共享主语和重复引用歧义，未将这些维度合成总通过率。

```powershell
python -m pip install -r requirements-data.txt
python -m unittest discover -s tests -v
python scripts/validate_phase1.py
```

测试不下载模型，验证重复引文拒绝、Unicode 索引、改写拒绝、指标独立性，以及配置／实际 token 不一致时拒绝比较。完整模型实验的原始响应、逐 token 分数、拒绝记录和逐句指标保存在 `reports/`；当前运行状态见 `reports/STATUS.md`。

log-probability 是模型生成的引用 JSON 的自然对数概率均值，包含 JSON 标点，排除特殊 token；派生索引没有模型分数。它不是主张为真的概率。上述前期模型实验只做抽取和定位，没有开展外部事实查证，也不从 15 个合成样本推断真实新闻上的泛化表现。

## 方向

正式阶段已完成 60 条测试样本策划和抽取／token 对齐代码；接下来需完成独立语义参考标注和证据验证，并将事实覆盖、限定条件和位置分开评估。可继续研究验证器与抽取器协作及错误类型归因；当前结果以已执行实验为准。

模型使用条件见 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。仓库不分发模型权重。
