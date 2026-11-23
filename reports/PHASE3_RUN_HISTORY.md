# Week 6 运行历史

2026-11-12：第一次冻结记录为 `phase3_week6_frozen_protocol_20261112.json`。在运行开始后发现数据整理任务为 `data/phase3/curation.json` 补入了 `artifacts.dataset_sha256` 与 `artifacts.inputs_sha256` 两个校验字段，使清单的字节哈希与冻结值不同。立即停止；`phase3_week6_run_20261112.json` 保留已完成的 3 条成对输出，`completed` 为 false，不纳入正式指标。

第二次冻结为 `phase3_week6_frozen_protocol_v2_20261112.json`，重新从头运行，输出 `phase3_week6_run_v2_20261112.json`。两份冻结协议之间仅 `data/phase3/curation.json` 的哈希不同；去掉新增 `artifacts` 字段可逐字重建第一次清单哈希。模型、提示词、代码、阈值、原始 110 条输入及金标签数据均未修改。此重启用于修复并发写入引起的冻结不一致，没有依据前三条结果调参；前三条结果已经被观察，不能声称第二次运行完全未接触其输出。

两次冻结均为本机记录，不是独立可信时间戳。保留部分运行及重启原因，避免把开发过程包装为一次无中断实验。
