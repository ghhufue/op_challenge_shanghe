# BatchMatmulMaxSum performance archive

本目录使用“实现版本 + 采集轮次”管理真机性能，避免把历史实现的测评误认为当前实现结果。

## 当前版本

- 当前性能版本：[v002](versions/v002/manifest.md)
- 机器可读指针：[CURRENT_VERSION](CURRENT_VERSION)
- 当前结论：[analysis.md](analysis.md)
- 完整测评命令：[benchmark_workflow.md](benchmark_workflow.md)

`CURRENT_VERSION` 是当前核心实现唯一有效的性能版本号。读取性能数据前必须先读取该文件和对应版本的 `manifest.md`。

## 目录结构

```text
docs/perf/
├── CURRENT_VERSION
├── README.md
├── analysis.md
├── benchmark_workflow.md
└── versions/
    ├── v001/
    │   ├── manifest.md
    │   ├── round_001/
    │   ├── round_002/
    │   └── round_003/
    └── legacy_unversioned/
        ├── README.md
        └── round_NNN/
```

- `vNNN`：一次核心实现快照。
- `manifest.md`：该版本的 commit、测量状态、环境和 round 映射，是判断数据有效性的入口。
- `round_NNN`：同一实现版本下的一次完整 `msprof` 标准采集。round 编号在每个版本内从 `001` 重新开始。
- `legacy_unversioned`：建立版本制度前产生的历史数据，只能用于追溯，不能作为当前基线。

## 强制版本规则

以下变更必须把 `CURRENT_VERSION` 递增到下一个未使用的 `vNNN`：

- kernel 算法、同步、流水、数据搬运或归约逻辑；
- tiling、策略阈值、block 数、split 方式或 workspace 布局；
- host dispatch、launch 参数、编译选项或 submission 实现；
- 任何可能改变精度、执行路径、device code 或性能的核心修改。

新版本建立时：

1. 先递增 `CURRENT_VERSION`。
2. 创建新版本目录和 `manifest.md`，状态写为 `UNMEASURED`。
3. 不得复制旧 round；没有新测试就必须明确写“当前版本暂无有效性能结论”。
4. 真机测量完成后，把新数据写入该版本自己的 `round_NNN`，再将状态改为 `MEASURED`。
5. 更新 [analysis.md](analysis.md) 的当前版本结论。

仅修改文档、测试或不影响编译产物的辅助工具时不递增。若测评方法变化但核心实现未变，在当前版本 manifest 中记录方法修订，不伪造新的实现版本。

仓库根目录 [AGENTS.md](../../../../AGENTS.md) 对未来 AI agent 强制执行上述规则。

## 数据读取原则

查看“当前性能”时必须同时满足：

- 目标目录等于 `versions/$(cat CURRENT_VERSION)`；
- manifest 状态为 `MEASURED`；
- manifest 的实现 commit 与待评估核心实现一致，或者其后的提交被明确证明不改变核心逻辑；
- shape、dtype、layout、key、CANN 和硬件环境可比。

任一条件不满足时，只能称为历史参考数据。

## 修订记录

| 日期 | 说明 |
|---|---|
| 2026-09-16 | 建立性能版本制度；当前实现登记为 `v001`，早期 round 迁入 `legacy_unversioned`。 |
