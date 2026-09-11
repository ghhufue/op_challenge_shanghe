# BatchMatmulMaxSum 高阶 Matmul API 基线重构方案

> 状态：已实现代码基线，并通过 CANN 9.1、`dav-2201` 编译、本地 Python 测试及 816 组 Host tiling 探针；尚待真机正确性与性能验证。
>
> 目标：先实现一版结构最简单、可正确计时的官方高阶 Matmul API 自动融合基线；保留现有测试、批量运行、强制 key、性能结果分析和策略生成能力，为以后加入手写 Flag 内核及更多 shape/key 候选提供稳定对照。

## 1. 决策摘要

本次重构不使用 `CopyL0C2UB`，也不继续维护当前 BM/BMN 手工分块和 GM ping-pong 实现。第一版生产候选只保留两条路径：

1. `VECTOR_REFERENCE`：仅作为极小 shape 的正确性与低启动开销路径。
2. `AUTO_MATMUL_FUSED`：唯一的通用路径，使用 `matmul::Matmul`、`REGIST_MATMUL_OBJ`、`Iterate` 和 `GetTensorC`，C 的逻辑位置为 `TPosition::VECIN`，由高阶 API 管理 AIC/AIV 隔离及同步。

项目级不再出现 `BM`、`BMN`、N-group、GM tile staging、ready/reusable Flag 等旧执行逻辑。`splitN` 仅作为候选目录中的保留字段存在，当前实现固定为 1。Matmul API 内部仍然会使用 `baseM/baseN/baseK` 分块，这是库内部 tiling，不等同于旧的项目级 BM/BMN 路径。

同时保留候选 key 和离线调优基础设施。第一版只有一个自动融合 key；以后可以增加：

- 不同 `baseM/baseN/baseK` 的自动融合 key；
- 同步与异步 `Iterate/GetTensorC` key；
- 手写 Flag 的 `MANUAL_FUSED_*` key；
- 针对特定 shape 区间的调度策略。

这样，“官方高阶 API 基线”会成为后续所有优化的固定性能对照，而不是在后续重构中被覆盖。

## 2. 本阶段范围

### 2.1 要完成的工作

- 新增一个使用官方高阶 Matmul API 的 MIX kernel。
- Matmul 输出直接由 `GetTensorC` 获取到 `VECIN`，不经过自建 GM tile staging。
- 在 AIV 上对每个 C tile 做 FP32 行最大值，并跨 N tile 在线更新。
- 将每个 M shard 的部分分数写入 workspace，再用一个很小的 Vector kernel 汇总为 `[B]`。
- 支持 FP16、BF16 和四种 transpose storage 组合。
- 保留 `run_kernel_forced(..., tilingKey)`，使每个候选可以独立做正确性和性能测试。
- 保留数据生成、golden、校验、批量 case、重复计时、结果分析和提交打包能力。
- 新基线通过后，删除旧 BM/BMN 内核和只服务于旧实现的结构字段、测试与文档。

### 2.2 明确不做

- 不使用 `CopyL0C2UB`。
- 不手写 `CrossCoreSetFlag/CrossCoreWaitFlag`。
- 不实现 N 方向多核分片；候选元数据保留 `splitN=1`，生产 kernel 不包含兼容分支。
- 不同时调试多个 tile 候选；第一版只使用一个保守、合法的自动融合配置。
- 不在第一版融合最终跨 shard 的 Sum；先使用第二个轻量 Vector kernel，保证实现简单、同步关系明确。
- 不以本次重构的性能作为最终结论；它只是后续优化的可重复基线。

## 3. 官方接口依据

本方案以官方自动融合模式为准：

- Kernel 使用 `__mix__(1, 2)`。
- C 类型声明为 `MatmulType<TPosition::VECIN, ...>`。
- 通过 `REGIST_MATMUL_OBJ` 注册 Matmul 对象。
- 使用 `Iterate` 触发 Cube 计算，使用 `GetTensorC(LocalTensor)` 获取结果块。
- 分离架构下由框架完成 AIC/AIV 代码隔离和核间同步。
- Host tiling 使用 `MultiCoreMatmulTiling`；本基线让两个 AIV 客户端分别提交一个完整 N 的 Matmul，因此每个客户端的 tiling 使用 `SetDim(1)`，不让 tiler 再次跨客户端切 N。
- 高阶 Matmul API 需要系统 workspace；异步模式还需要按接口要求提供用户 workspace。

参考资料：

- [CV 融合算子实现](https://asc.gitcode.com/guide/operator_practice/simd_operator_impl/fusion_operator_programming/cv_fusion/operator_impl.html)
- [官方 matmul_fused 示例](https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_fused)
- [CANN 9.1 异步 Iterate/GetTensorC 示例](https://gitcode.com/cann/asc-devkit/tree/9.1.0-beta.1/examples/01_simd_cpp_api/03_libraries/00_matrix/matmul_async_iterate)
- [GetTensorC API](https://asc.gitcode.com/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/GetTensorC.html)

实现时必须以评测机安装的 CANN 头文件和示例为最终依据。`master` 文档可能包含晚于本机 CANN 9.1 的接口变化。

## 4. 目标执行结构

```text
run_kernel
  ├─ 校验输入并解析 Shape / dtype / transpose
  ├─ SelectTilingKey
  │    ├─ VECTOR_REFERENCE        极小 shape
  │    └─ AUTO_MATMUL_FUSED       其余 shape
  ├─ 创建或复用 tiling + workspace
  ├─ 启动 AUTO_MATMUL_FUSED MIX kernel
  │    ├─ Matmul API：完整 K 累加
  │    ├─ GetTensorC：C tile -> VECIN
  │    ├─ AIV：沿 N 在线 ReduceMax
  │    └─ AIV：写 partialScore[b, mShard]
  └─ 启动 FinalReduce Vector kernel
       └─ y[b] = sum(partialScore[b, :])
```

主融合 kernel 只负责 `Matmul + Max(N) + shard 内 Sum(M)`。最终 Vector kernel 只读取很小的 `partialScore`，不读取完整 C，也不重复矩阵计算。

## 5. 工作划分原则

### 5.1 只保留 B/M shard，不做项目级 N 分片

一个逻辑任务拥有：

```text
(batchIndex, mBegin, mLength)
```

该任务必须遍历这个 M shard 对应的完整 N，因此任务结束时可以直接得到完整的行最大值，不需要跨 N-group 合并。

```text
taskCount = B * ceil(M / shardM)
taskId -> (batchIndex, mShardIndex)
```

这里的 `shardM` 只是任务所有权粒度，不再派生 `BM_*`、`BMN_*` 两套 kernel。第一版采用一个固定且保守的值；以后若要枚举，只增加 key 配置，不复制 kernel 实现。

这种策略的代价是 `M` 很小、`N` 很大时并行度可能不足。第一版接受这个限制，因为它的目的正是测量“官方自动融合、无手写 N 分片”的真实基线。以后如果手写 Flag 或重新引入 N 方向并行，提升可以直接与此基线比较。

### 5.2 Matmul API 内部分块

每个任务的逻辑矩阵为：

```text
A: [mLength, K]
B: [K, N]
C: [mLength, N]
```

Host 生成 `TCubeTiling`，Matmul API 自己处理 K 累加以及 `baseM/baseN/baseK` 迭代。项目代码不再手工计算 N-group、tileRound、ping-pong slot 和跨核 ready/reusable 序号。

第一版 tiling 必须满足：

- `C` 的位置是 `VECIN`、数据类型是 FP32、布局是 ND。
- 单个 `baseM * baseN * sizeof(float)` 加 row-max 和 reduce scratch 后能够放入 UB。
- 不启用 split-K。
- 不把同一逻辑行的不同 N 范围交给需要外部合并的独立任务。
- M/N/K 尾块由 Matmul tiling 和 Vector mask 共同处理。

若 tiling API 无法保证“任务拥有完整 N”，该配置必须在 Host 侧判为不可用，而不是悄悄恢复旧 BMN 合并逻辑。

## 6. Kernel 侧设计

### 6.1 类型与对象

概念结构如下，具体模板参数按本机 CANN 9.1 头文件调整：

```cpp
using AType = matmul::MatmulType<
    TPosition::GM, CubeFormat::ND, InputT, TransposeX1>;
using BType = matmul::MatmulType<
    TPosition::GM, CubeFormat::ND, InputT, TransposeX2>;
using CType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, float>;
using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

matmul::Matmul<AType, BType, CType, BiasType> matmulObj;
```

正式代码中不保留上面伪代码里的占位符。四种 transpose 组合继续通过模板实例化表达，避免在热点循环中动态判断布局。

### 6.2 初始化顺序

主 kernel 使用以下固定顺序：

1. 创建 `TPipe` 和包含公开 `matmul::Matmul` 对象的 kernel 对象。
2. `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &cubeTiling)`；AIC 在宏内进入 Matmul server，后续初始化仅由 AIV 执行。
3. 解析统一的 `FusedTilingData`，初始化 C 输入队列、row-max buffer 和 reduce scratch。
4. 对当前任务设置 A/B 的 batch 与 M shard 起始地址。
5. 禁用 Bias。
6. 若采用异步模式，调用 `SetWorkspace(userWorkspace)`。
7. 调用 `Iterate`，随后逐块调用 `GetTensorC`。
8. 完成所有 C tile 消费后调用 `End()`。

第一版建议先用同步 `Iterate<true>/GetTensorC<true>` 跑通全部正确性，再只改成异步版本作为同一个 key 的后续提交。若希望严格比较同步与异步，则将它们保留为两个 key，但不要同时引入其他结构变化。

### 6.3 C tile 消费

每个 AIV 维护其拥有行的 FP32 `rowMax`：

```text
rowMax[:] = -inf

for each C tile returned by GetTensorC:
    屏蔽 N 尾部无效列为 -inf
    tileMax = ReduceMax(C tile, axis=N)
    rowMax = Maximum(rowMax, tileMax)

partialScore[task] = Sum(rowMax[validRows])
```

必须满足：

- C tile 在 `FreeTensor` 前完成 Vector 消费。
- `rowMax` 在该任务遍历完整 N 期间一直保留在 UB。
- Max 初值为 `-inf`，不能用 0。
- M 尾块只参与真实行的 Sum。
- N 尾块不能让补零列参与全负数输入的 Max。
- 不假设 `Iterate` 的隐式遍历顺序；使用本机 API 提供的 offset 信息，或用官方样例验证后的 tiling 顺序映射 C tile 的 `(mOffset,nOffset)`。

### 6.4 两个 AIV 的所有权

`__mix__(1,2)` 下必须在实现前做一个最小探针，确认：

- `GetBlockIdx()` 和 sub-block index 在两个 AIV 上的含义；
- 两个 AIV 各自收到哪些 C tile；
- `SetDim` 与 `SetSimdNumBlocks` 对任务切分的实际结果；
- 是否能通过 tiling 保证同一行的完整 N 由同一 AIV 在线归约。

探针确认 CANN 9.1 / `dav-2201` 上 AIV 的 `GetBlockIdx()` 为 `AIC block * taskRatio + subBlockId`。实现按 `subBlockId` 将 64 行 shard 分成两个 32 行 lane，每条 lane 使用 `SetDim(1)` 的独立 Matmul 客户端遍历完整 N，并写自己的独占 partial 槽。禁止两条 lane 写同一个 `partialScore`；后续改变调度方式时必须重新验证该所有权关系。

## 7. Host tiling 与 workspace

### 7.1 精简后的 tiling 数据

建议使用一个结构承载必要信息：

```text
FusedTilingData
  shape: B, M, N, K
  shardM
  taskCount
  launchBlocks
  partialScoreOffset / partialScoreCount
  userWorkspaceOffset / userWorkspaceBytes
  systemWorkspaceBytes
  TCubeTiling cubeTiling
```

以下旧字段从生产执行分支删除；`splitM/splitN` 可继续作为结果记录与未来候选元数据，但当前 kernel 不按它们切换算法：

```text
splitM
splitN
nGroups
nTiles owned by group
stage buffer count/offset
ready flag id
reusable flag id
```

候选目录中可以继续保留 `base_m/base_n/base_k`，用于以后枚举高阶 API tiling；它们不再决定不同的 C++ kernel 文件。

### 7.2 核数设置

本基线在一个 `__mix__(1,2)` block 内由两个 AIV 客户端各自提交一个 Matmul：

```text
kernel launch blocks = 实际 AIC/AIV 组合数
每个客户端 SetDim  = 1
```

`SetDim(2)` 会让 `MultiCoreMatmulTiling` 在大 N shape 上把 N 再切成两份，破坏 lane 内完整 N 的在线最大值语义，因此本基线明确不用它。launch block 数仍必须受 `taskCount` 和物理 AIC 数约束，不能把调用方传入的通用 core 数、物理 AIC 数、AIV 数视为同一个量。

### 7.3 workspace 分区

workspace 只保留：

```text
[系统 workspace]
[异步 Matmul 用户 workspace，如接口要求]
[每个 task/AIV 独占的 FP32 partialScore]
[必要的对齐 padding]
```

明确删除完整 C tile 的 GM staging 区。

当前直调 ABI 没有外部 workspace 参数，因此仍需保留按 stream 管理设备内存的机制。第一版可沿用当前缓存思想，但 key 应收缩为：

```text
stream + dtype + transpose + shape + tilingKey + requiredBytes
```

不要在每次调用后同步并释放，否则测到的主要是 Host 分配和同步开销。多 stream 必须拥有互不覆盖的 in-flight workspace。

## 8. 候选 key 与调优流程

### 8.1 第一版候选

第一版候选目录只标记两个 implemented key：

| key | 建议符号 | 实现 | 用途 |
| --- | --- | --- | --- |
| 0 | `VECTOR_REFERENCE` | 直接计算 | 极小 shape 与正确性兜底 |
| 100 | `AUTO_MATMUL_FUSED` | 高阶 API 自动融合 | 固定性能基线 |

key 100 中 `tileM=64` 表示两个 AIV lane 合计的 shard 行数，`vecM=32` 表示单 lane 行数，`tileN=128` 表示 VECIN 队列允许的最大 C tile 宽度。Host 对小 N 使用 `baseN=min(128, align_up(N,16))`，实际 `baseM/baseN/baseK` 以 CANN 返回的 `TCubeTiling` 为准；因此目录值是该自动策略的控制参数，不应被误读为所有 shape 上的最终库内 tiling 值。

旧的 `BM_16X128X64`、`BM_32X128X64`、`BM_32X256X64`、`BM_64X128X128`、`BMN_*_P2/P4` 全部删除，避免旧 key 名义保留但实际落到同一 kernel 的情况。

### 8.2 为后续实验预留的扩展方式

以后每个实验必须分配新 key，不覆盖 key 100：

```text
100        AUTO_MATMUL_FUSED_BASELINE
110~119    AUTO_MATMUL_FUSED 不同 baseM/baseN/baseK
120~129    AUTO_MATMUL_FUSED 异步/队列深度实验
200~299    MANUAL_FUSED 手写 Flag 实验
```

key 区间只是建议，但一旦确定应保持稳定。结果文件中至少记录：

- git commit；
- key 与配置；
- CANN/驱动/设备；
- dtype、transpose、B/M/N/K；
- correctness 状态；
- warmup、重复次数、p50/p95；
- workspace 字节数；
- 是否同步或异步。

策略树只能从全部通过正确性验证的 key 中选择。基线 key 100 永久可强制运行，以便判断新优化是否真正提升。

## 9. 文件整理方案

清理必须发生在新高阶 API 路径通过 smoke、correctness 和四布局测试之后，不能先删除唯一可运行实现。

### 9.1 目标目录

```text
batchmatmulmaxsum/
├─ CMakeLists.txt
├─ data_utils.h
├─ main.asc
├─ kernel.asc
├─ run.sh
├─ host/
│  ├─ acl_runtime.h
│  ├─ input_validation.h
│  └─ fused_tiling.h
├─ kernels/
│  ├─ auto_matmul_fused.asc
│  ├─ final_reduce.asc
│  ├─ reference.asc
│  └─ kernel_dispatch.asc
├─ configs/
│  └─ tiling_candidates.json
├─ tiling/
│  ├─ tiling_types.h
│  ├─ tiling_catalog.h
│  ├─ tiling_catalog_generated.h
│  ├─ tiling_policy.h
│  └─ submission_policy.h
├─ tuning/
│  └─ ... 保留候选加载、枚举、拟合策略能力
├─ scripts/
│  └─ ... 保留生成数据、运行、校验、计时、分析和打包能力
├─ tests/
│  ├─ test_tuning.py
│  └─ test_source_structure.py
└─ submission/
   └─ kernel.asc
```

这里的“最精简”指生产实现只有一个自动融合主 kernel 和一个最终归约 kernel，不是删除实验基础设施。

### 9.2 删除或替换

| 当前文件/内容 | 处理 |
| --- | --- |
| `kernels/bm.asc` | 删除；这是旧的分离 BM 实现 |
| `kernels/bmn.asc` | 删除；不再做项目级 N 分片 |
| `kernels/mixed.asc` | 由 `kernels/auto_matmul_fused.asc` 替换 |
| `host/bm_tiling.h` | 由 `host/fused_tiling.h` 替换 |
| `KernelPath::BM/BMN` | 删除，改为 `REFERENCE/AUTO_FUSED`；以后再增加 `MANUAL_FUSED` |
| N-group、stage buffer 字段 | 从 Plan、缓存 key、dispatch 和测试中删除；`splitN` 暂以固定值 1 作为候选与结果元数据保留，生产 kernel 不读取它进行分支 |
| 手写 CrossCore Flag 常量和断言 | 删除 |
| `TUNING.md` 的旧架构说明 | 重写为基线 key 和后续实验流程 |
| `submission/kernel.asc` | 新代码通过后重新生成，禁止保留旧拼接产物 |
| `submission/submission_policy.h` | 仅在打包格式确实需要独立策略文件时保留 |

### 9.3 保留并调整

| 文件/能力 | 处理 |
| --- | --- |
| `main.asc` | 保留数据加载、warmup、计时和 `tiling_key` 参数；移除旧 key 输出字段 |
| `kernels/reference.asc` | 保留，限制为极小 shape/强制 key 使用 |
| `configs/tiling_candidates.json` | 保留，候选收缩为 reference + auto baseline |
| `tuning/*` | 保留流程，删除对 BMN 和 `split_n` 的硬编码 |
| `scripts/tune_cases.py` | 保留强制 key 批量计时能力 |
| `scripts/analyze_tuning_results.py` | 保留结果对比；确保 key 100 可作为固定 baseline |
| `scripts/run_cases.py` | 保留 correctness/fuzz/stress 与可选强制 key |
| `scripts/gen_data.py`、`BatchMatmulMaxSum.py`、`verify_result.py` | 原样保留数学语义和校验能力 |
| `scripts/bundle_submission.*` | 更新拼接清单，删除旧 BM/BMN fragment |
| `tests/test_tuning.py` | 保留通用 catalog/策略测试，删除旧 workspace/N-group 断言 |
| `preliminary/docs/test_data.py` | 保留全部数据规格，包括检查跨 N tile Max 的 case；case 名可不改 |

`c25_bmn_first_group` 等测试名虽然带有 BMN，但其验证的是数学语义——最大值可能位于不同 N 区间——不能因为删除 BMN 实现就删除这些用例。可在后续纯整理提交中重命名为 `c25_first_n_range` 等，以免误导。

### 9.4 不应纳入提交的文件

继续依靠 `.gitignore` 排除：

- `build*`；
- `.pytest_cache`；
- `__pycache__` 和 `*.pyc`；
- 本地生成的 `test_data`、`test_results`、大体积 benchmark JSON。

## 10. 测试迁移

### 10.1 必须保留的功能测试

- FP16/BF16。
- 四种 transpose storage 组合。
- 全负相似度，验证 Max 初值不是 0。
- 最大值位于首、中、尾 N tile。
- 不同行最大值位于不同 N tile。
- M/N/K 尾块。
- B 大于 1 的 batch 地址与输出隔离。
- 重复执行一致性。
- 多 stream workspace 隔离。
- 最大合法输入约束下不越界、不死锁。

### 10.2 删除的旧结构测试

以下测试不应迁移为新实现约束：

- BM/BMN key 数量和固定顺序。
- `splitN=2/4`。
- round-robin N tile 分配。
- GM stage workspace 字节公式。
- Flag 5/6/7 和 `PIPE_FIX/PIPE_MTE*` 源码字符串。
- `MatmulImpl::IterateAll(stageGM)`。
- 旧 scheduler 的 ping-pong wait/release 次数。

### 10.3 新增结构测试

新结构测试只检查必要架构事实：

- 主 kernel 包含 `__mix__(1, 2)`。
- 使用公开 `matmul::Matmul`，不使用 `matmul::MatmulImpl`。
- C 类型为 `TPosition::VECIN`。
- 使用 `REGIST_MATMUL_OBJ`、`Iterate`、`GetTensorC` 和 `End`。
- 生产代码不存在 `CrossCoreSetFlag`、`CrossCoreWaitFlag` 和 GM C-tile staging。
- catalog 中至少存在 reference 与固定 auto baseline key。
- `run_kernel_forced` 能分别运行两个 key。
- submission bundle 不包含 `BMN`、`nGroups`、`splitN>1` 执行分支或旧 Flag 协议；允许保留固定为 1 的 `splitN` 元数据。

结构测试不能代替 NPU 正确性测试；它只防止后续重构意外偏离基线定义。

## 11. 实施顺序与提交边界

建议每一步形成独立、可回退的提交。

### P0：API 能力探针

- 在目标机器编译运行官方 `matmul_fused` 或等价最小样例。
- 确认 CANN 9.1 上 `VECIN`、`REGIST_MATMUL_OBJ`、同步/异步接口签名。
- 打印并验证 1:2 模式下 block/sub-block 与 C tile 的分配。
- 确认系统 workspace 获取和传递方式。

退出条件：一个小矩阵可以从 Matmul 输出到 VECIN，并由 AIV 完成简单 Vector 操作。

### P1：并行加入自动融合基线

- 新增 `auto_matmul_fused.asc`、`final_reduce.asc`、`fused_tiling.h`。
- 暂时保留旧路径，仅通过新 key 100 强制选择新实现。
- 先实现同步 `Iterate/GetTensorC`。

退出条件：smoke、FP16/BF16、四布局全部通过。

### P2：补齐通用正确性

- 完成 M/N/K 尾块。
- 完成所有 B/M shard 和 partialScore 汇总。
- 跑 correctness、fuzz、重复执行、多 stream。

退出条件：全部自建正确性测试通过，无超时和死锁。

### P3：建立性能基线

- 固定 key 100 的实现和配置。
- 记录同步版本逐 case p50/p95、workspace 和 profiler 指标。
- 若异步接口可用，单独提交异步改动并重新测量；决定 key 100 固定为最终基线版本，或保留同步/异步两个 key。

退出条件：结果文件可用 commit + key 完整复现。

### P4：删除旧实现并整理文件

- 删除 `bm.asc`、`bmn.asc`、旧 `mixed.asc` 和 `bm_tiling.h`。
- 收缩 catalog，更新 tuning 模型、结构测试、文档和 bundler。
- 重新生成 submission，运行全量测试。

退出条件：仓库只剩 reference + auto baseline 两个 implemented 候选，工具链功能不丢失。

### P5：后续实验

- 高阶 API tile/key 枚举从 110 段增加。
- 手写 Flag 路径从 200 段增加。
- 所有实验先 forced-key 做全量正确性，再进入策略拟合。
- 任何优化都同时报告相对 key 100 和相对官方最优耗时的变化。

## 12. 验收标准

### 12.1 正确性

- smoke、correctness、fuzz 全通过。
- FP16/BF16 和四布局无遗漏。
- 结果有限且满足现有 checker 误差门槛。
- 多次运行一致；多 stream 无 workspace 覆盖。

### 12.2 架构

以下搜索在生产源码和生成 submission 中应无结果：

```text
KernelPath::BMN
nGroups
CrossCoreSetFlag
CrossCoreWaitFlag
MatmulImpl
BMN_16X
```

以下关键结构必须存在：

```text
__mix__(1, 2)
matmul::Matmul
TPosition::VECIN
REGIST_MATMUL_OBJ
Iterate
GetTensorC
```

### 12.3 工具链

- 可强制运行 reference 和 key 100。
- 可批量跑 case、校验结果、重复计时、生成分析报告。
- 可在新增候选后生成 catalog 和拟合 submission policy。
- bundler 输出可独立提交，且与模块化源码一致。

### 12.4 性能记录

第一版不设置必须超过当前实现的门槛，但必须建立可重复数据。至少分别报告：

- 极小 shape：reference 与 auto baseline；
- 中等 shape：当前旧 MIX 与 auto baseline；
- 大 M/N/K：当前旧 MIX 与 auto baseline；
- Matmul task duration、Vector duration、workspace 大小和 kernel launch 数。

只有完成上述记录后，才能判断高级 API 自动融合本身的收益，以及后续手写 Flag 是否值得继续。

## 13. 风险与停止条件

| 风险 | 应对 |
| --- | --- |
| 两个 AIV 按 N 而不是 M 分担结果，无法独立得到完整 row max | P0 先验证 tiling；不能满足时停止实现并记录限制，不暗中恢复 BMN |
| `baseM * baseN` 导致 UB 不足 | 减小库 tiling base 块，重新生成 tiling；不增加 GM staging |
| 异步 API 需要额外 workspace 或顺序与预期不同 | 同步版先成为基线，异步作为独立实验 |
| 极小 shape 被融合启动开销拖慢 | 保留 reference 路径，用实测阈值选择 |
| 最终归约第二次 launch 占比明显 | 基线记录后再评估单 kernel 归约，不能与第一版同时引入 |
| 清理旧文件导致调优工具失效 | P4 中逐项迁移，先跑 Python 测试和 bundle check 再删除旧文件 |

本方案最重要的边界是：先获得一个简单、稳定、可强制选择的官方高阶 API 基线，再做手动同步和 shape 特化。基线 key、测试数据与计时方法一旦建立，后续优化不得静默修改它们。
