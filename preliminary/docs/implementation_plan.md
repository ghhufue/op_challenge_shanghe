# BatchMatmulMaxSum 实现与验证计划

本文基于本仓库初赛题目和当前直接调用模板，给出从正确性实现到融合优化的执行流程。

**状态：设计与测试工具已提供，NPU 算子尚未实现；本文中的性能参数均为待测候选，不是已验证的最优配置。** 官方完整 15 个测试点未在仓库中提供。本文附带的是自建回归集，不能声称等同于官方测试数据，也不能证明覆盖所有合法输入。

相关文件：

- [原始题目](problem.md)
- [待实现的 kernel 和 Host 启动入口](../batchmatmulmaxsum_template/kernel.asc)
- [本地 Host 测试程序](../batchmatmulmaxsum_template/main.asc)
- [现有 FP64 参考实现](../batchmatmulmaxsum_template/scripts/BatchMatmulMaxSum.py)
- [可执行测试数据规格、生成器和严格校验工具](test_data.py)

## 1. 交付目标与边界

完成一个支持 FP16/BF16 输入、四种存储布局、所有题目合法维度的 Ascend C 算子。输出严格为 `[B]` FP32，先保证正确性和重复执行一致性，再优化总执行耗时。

实施交付包括：

1. CPU 侧 shape/布局解析、tiling、路径选择、workspace 管理和核函数启动。
2. NPU 侧小规模正确性路径、通用 Cube/Vector 融合路径、最终归约路径。
3. 完整本地用例读取与执行流程，逐 case 校验、重复执行、多 stream 验证。
4. 按 shape 分类的性能报告、实际设备和 CANN 版本记录。

当前任务只落地本文和测试数据工具，不提前修改参赛 kernel 或原测试入口。

### 1.1 开始编码前确认的环境信息

在实际昇腾机器上记录设备型号、CANN/ASC 编译器版本、编译目标、AIC/AIV 数量、L1/L0/UB 容量、支持的 Matmul 输出布局与融合模式。当前 CMake 默认 `dav-2201` 只是模板配置，不能据此推断实际评测设备、片上容量或 AIC:AIV 比例。

首先编译与安装版本匹配的最小矩阵乘/融合样例，确认 BF16、转置、尾块、workspace 和混合核启动方法。模板注释中的 `__cube__` 仅是占位示例，不能据此假定纯 Cube kernel 可以直接承担全部 Vector 归约。官方提供了 CV 融合范式，但入口写法与能力必须匹配目标工具链。[CV 融合官方说明](https://asc.gitcode.com/guide/operator_practice/simd_operator_impl/fusion_operator_programming/cv_fusion/basics.html)

本地测试代码是 Bash/CANN/AscendCL 流程；仅在 Windows 上完成文档和 Python 自检不等于完成 NPU 编译验证。

## 2. 固定数学语义

逻辑输入 `X1[B,M,K]`、`X2[B,K,N]`：

```text
A[b,m,n] = sum_k X1[b,m,k] * X2[b,k,n]
R[b,m]   = max_n A[b,m,n]
y[b]     = sum_m R[b,m]

[B,M,K] × [B,K,N] → [B,M,N] → Max(N) → [B,M] → Sum(M) → [B]
```

`b` 之间独立。无 batch broadcast，无 mask，无额外归一化，无需输出 argmax；多个 query token 可以匹配同一个 document token。

三个维度分块的合并规则：

| 分块维度 | 部分结果 | 正确合并方式 |
| --- | --- | --- |
| K，embedding | 同一个点积的部分和 | 先累加完 K，才允许进入 Max |
| N，document token | 同一 query 行在各 document 分片内的最大值 | 逐行 Max，全部 N 完成后才允许 Sum(M) |
| M，query token | 每个 query 分片完成 MaxSim 后的分数 | Sum |
| B，batch | 不同文本对的分数 | 各自写入 y[b]，不互相归约 |

例如两行的相似度为 `[10,0]`、`[0,10]`，正确结果是 `10+10=20`。先沿 M 求和再 Max 得到 10，违反题意。

### 2.1 维度与精度

- `1≤B≤64`，`1≤M,N≤8192`，`32≤K≤8192`，`K%8=0`。
- `B*M*K≤2^26` 且 `B*N*K≤2^26`；没有同样大小的相似度矩阵约束。
- 输入连续 ND，两输入同为 FP16 或同为 BF16；点积、Max 中间值、最终 Sum 采用 FP32 或等效精度。
- 最大值初始化为负无穷；所有实际输入有限，最终输出必须有限。
- golden 从实际 FP16/BF16 存储值转成 FP64 后计算，再转 FP32；不能拿量化前的随机浮点数生成 golden。
- 题目误差表与模板需要分别记录：当前 FP32 输出脚本使用 `abs(out-ref) <= 1e-4 + 1e-4*abs(ref)` 的近似比较。本文以此作为本地门槛，所有输出元素必须通过，并额外严格检查长度和有限性。若取得官方 checker，以官方实际实现为准；不将此公式说成相对误差和绝对误差必须同时各小于 `1e-4`。

## 3. 工程结构和 Host 侧主要实现

```text
main.asc / main()                         CPU 测试驱动
  └─ run_kernel(...)                      CPU 调度入口，位于 kernel.asc
       ├─ 解析 shape、dtype、transpose
       ├─ 生成 Plan / Matmul tiling
       ├─ 申请或借用 workspace
       ├─ 启动主计算 kernel               NPU
       └─ 按需启动最终归约 kernel          NPU，同一 stream
main() 同步、复制输出、检查结果、释放资源
```

保留现有 `run_kernel` 签名。`kernel.asc` 被 `main.asc` include，不在它里面增加 main 或 include guard。CPU 的 shape 指针不能直接交给设备侧解引用，应转换为标量参数或设备可访问的 tiling 数据。

### 3.1 解析逻辑 shape 和地址

```cpp
B = info_x1.tensors[0].shape[0];
M = transposeX1 ? shape1[2] : shape1[1];
K = transposeX1 ? shape1[1] : shape1[2];
N = transposeX2 ? shape2[1] : shape2[2];
```

调试阶段核对另一输入的 B/K、输出 shape 和类型。FP16 对应模板 dtype=1，BF16 对应 2，FP32 输出对应 0。乘法、元素偏移和字节大小使用 64 位运算。

以元素为单位的地址公式：

```text
x1 false: b*M*K + m*K + k
x1 true : b*M*K + k*M + m
x2 false: b*K*N + k*N + n
x2 true : b*K*N + n*K + k
```

先用此公式验证小规模设备实现，再映射到 Matmul API 的布局参数和 stride。对一个逻辑子块，原矩阵 leading dimension 仍来自原始 storage shape，不能直接用 `tileM/tileN/tileK` 代替。

同一逻辑数据生成四种连续物理布局：

| TX1/TX2 | X1 storage | X2 storage |
| --- | --- | --- |
| 00 | `[B,M,K]` | `[B,K,N]` |
| 01 | `[B,M,K]` | `[B,N,K]` |
| 10 | `[B,K,M]` | `[B,K,N]` |
| 11 | `[B,K,M]` | `[B,N,K]` |

可以按 dtype × TX1 × TX2 做 8 个编译期实例并在 Host 分发，避免最内层反复判断。首版也可动态判断，先保证正确。

### 3.2 建议的 Plan 信息

| 字段 | 用途 |
| --- | --- |
| B/M/N/K、dtype、TX1/TX2 | 逻辑参数 |
| tileM/tileN/tileK | 分块；区分外部任务块和 Matmul API 内部 base 块 |
| mTiles、nPartitions、taskCount | 任务编号和最终归约组织 |
| launchBlocks、path | 启动核数和小矩阵/通用/N 分片路径 |
| strides、有效尾块尺寸 | 按原始物理布局访问 |
| workspace 区域偏移与大小 | 系统区、部分分数或部分最大值、可选 staging |
| Matmul tiling/config | 安装版本支持的矩阵乘配置 |

分块配置只依赖 shape、dtype、布局和硬件；不使用随机 seed、case ID、golden 或读取输入值来识别测试点。

## 4. 主实现：B/M 多核划分 + N 流式归约

### 4.1 为什么先按 B/M 分核

取 `Tm=tileM`，`Q=ceil(M/Tm)`，任务总数 `T=B*Q`。一个逻辑任务处理某个 batch 中的一组 query 行，并独立遍历完整 N、K。

```text
taskId ∈ [0, B*Q)
b  = taskId / Q
mi = taskId % Q
m0 = mi * Tm
mv = min(Tm, M-m0)

taskId = workerId; taskId < T; taskId += workerCount
```

这使不同任务之间无需合并 N 维最大值，只需在结尾合并 M 维部分和。逻辑 worker 到 AIC/AIV 的映射按实际混合执行模式实现，不能直接假定各核的 block index 含义相同。

### 4.2 主循环伪代码

以下是算法伪代码，不是可直接编译的 Ascend C API 调用：

```python
for task in assigned_tasks:
    b, m0, mv = decode(task)
    best = full(Tm, -inf, float32)

    for n0 in range(0, N, Tn):
        nv = min(Tn, N - n0)

        # 产出这一 M×N 子矩阵的完整 K 点积。
        # 首版优先让 Matmul API 内部完成 K 分块、FP32 累加。
        C = matmul_complete_k(x1[b, m0:m0+mv, :],
                              x2[b, :, n0:n0+nv])

        # 依照 C 的实际 ND/NZ 布局获取逻辑行；屏蔽无效列。
        C[:, nv:] = -inf
        tile_best = row_max(C)
        best[:mv] = maximum(best[:mv], tile_best[:mv])

    partial_score[task] = fixed_order_sum(best[:mv])
```

如果混合模式把一个 Cube 输出拆给多个 AIV，必须明确每个 AIV 对应哪些逻辑行、谁写哪一份结果；不能让两个 AIV 都重复写整个 `partial_score[task]`。可先各自写独占槽位，再在最终阶段固定顺序求和。

### 4.3 最终归约

主 kernel 写 `partial_score[B,Q]`，随后在同一 stream 启动 Vector 归约 kernel：

```text
y[b] = sum(partial_score[b,0:Q])
```

首版使用一个小 Vector 工作单元按固定顺序处理 B 个结果，安全写回精确的 `4*B` 字节。B 最大为 64，没必要一开始就为每个四字节输出创建独立核任务。

部分结果在 GM 中的槽位按实际写回粒度分配独占 padding，例如确认采用 32 字节粒度后每个标量占 8 个 float。消费者只取有效标量。不能让相邻核通过宽写回覆盖对方结果。最终 y 没有额外空间保证，必须使用目标版本支持的精确尾写方式；禁止把输出容量默认向上取整后越界写入。

不使用浮点 atomicAdd 聚合 y，以免引入不确定的累加顺序。第一版用两个 kernel 的 stream 顺序建立依赖，不在一个 kernel 中自制全局自旋 barrier。

当每个 batch 只有一个完整归约任务且写回安全时，可以走单 kernel 路径。使用第二个小归约 kernel 仍然保留主计算与 Max 的融合收益，不要求为追求一个 launch 而引入复杂全局同步。

### 4.4 两种搬运落地路线

**优先路线：Cube 输出分片进入可用的 Vector 本地缓冲，直接归约。** 要先确认输出格式、valid shape、API 迭代顺序、跨核通信和 workspace；获取一个结果块后按其真实 `(mOffset,nOffset)` 更新 best，不能假定 Iterate 一定先遍历 N。

**兼容过渡路线：每个 worker 使用有限大小的 GM staging 槽，Cube 写块、Vector 读块并归约。** 槽位拥有者和同步必须明确。若现有环境无法安全实现块内生产消费，先做仅用于调试的分离矩阵乘+归约基线，再按分批任务限制 staging 容量。不要把完整 `[B,M,N]` 常驻矩阵作为所有合法形状的保底实现。

同步首版验证后再引入异步。特定 CANN 版本对异步 `Iterate/GetTensorC` 的 workspace、输出位置和 ND/NZ 有限制，不能把同步/异步切换当成单纯修改一个 bool。[官方异步 Matmul 说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_10015.html)

## 5. 补充分支：小 M、大 N 时沿 N 多核切分

如果 `B*ceil(M/Tm)` 远小于可用计算 worker 数，先尝试减小 Tm。仍无法提高并行度，而且 N 足够大时，才启用 N 分片。

设每组 query 行分给 `P` 个 N 分片，任务为 `(b,mi,p)`，任务数 `B*Q*P`。每个分片覆盖完整且互不重叠的 N 范围，并只处理自己范围内的相似度。

主 kernel 输出：

```text
partial_max[b, mi, p, row] = max over n in partition p (dot over all K)
```

最终 kernel 对每行先沿 P 取 Max，再沿 M 求和：

```text
best_row = max_p partial_max[b, mi, p, row]
y[b] = sum over all valid mi,row (best_row)
```

**N 分片不能先把各自的行最大值求和后只保存一个标量。** 那会丢失逐行信息，无法合并。

候选 `P∈{1,2,4,8}`，上限同时受 worker 数、N 的有效分块数和 workspace 预算约束。不创建空分片，划分边界尽量对齐 Tn。启用条件通过实测确定；它会增加 query 重读、部分最大值存储和最终归约成本。

不把跨核 split-K 纳入首版。若以后使用 split-K，必须先合并相同 `(b,m,n)` 的完整 FP32 点积，再执行 Max，通信和同步成本通常更高。

## 6. 分块大小、缓存和融合流水

### 6.1 初始候选与容量核算

试验入口可从 `Tm∈{16,32,64}`、`Tn∈{64,128,256}` 开始；K 内部分块候选 `{64,128,256}` 必须经 Matmul tiling/API 校验。它们只是逻辑候选，不能宣称所有硬件都支持这些组合。

估算一组逻辑 tile 的有效载荷：

```text
A 输入：2 * Tm * Tk bytes
B 输入：2 * Tk * Tn bytes
FP32 C：4 * Tm * Tn bytes
FP32 best：4 * Tm bytes
```

例如 `(Tm,Tn,Tk)=(32,128,128)` 时，A=8 KiB，B=32 KiB，C=16 KiB。输入双缓冲有效载荷为 80 KiB，C 双缓冲为 32 KiB。

这些数字不能直接相加后判断“放得下”：A/B 的 L1 与 L0、C 的 L0C 与 Vector UB、格式转换区、API 内部缓冲、reduce scratch、对齐浪费都要分别核算。对本地 UB 可以先用 `2*C_bytes + best_bytes + reduce_scratch + format_scratch` 估算，再加入实际 API 占用。禁止用未经确认的固定 UB 容量做调参依据。

### 6.2 数据复用

- 一个 M 块遍历 N 时会重复使用同一 query 数据；K 足够小时可尝试在允许的片上层级保留它。
- K 大时不要假定整块 `[Tm,K]` 放得下，使用受控的 K 流水和输入分块。
- 相同 N 块被多个 M 任务使用，可研究调度局部性，但不依赖未保证的跨核缓存共享。
- 转置布局采用相应 stride/Matmul 转置能力；如需重排，优先局部分块 packing，并把 packing 成本计入总耗时。
- K 分块累加保持 FP32，不能每处理一个 K 块就把累加结果降回 FP16/BF16。

### 6.3 尾块

| 维度 | 处理 |
| --- | --- |
| K 尾部 | 只读取合法元素；内部补零使点积不变 |
| N 尾部 | 内部可补零参与矩阵乘，但归约前无效列必须屏蔽为 `-inf` |
| M 尾部 | 只累计真实 query 行，不让补齐行进入 Sum |
| GM 读写尾部 | 搬运长度、stride、对齐都显式处理，不跨过输入输出分配边界 |

尤其要测 `K=40/72/8184`，它们合法但可能不是底层矩阵块宽的整数倍。

### 6.4 双缓冲推进顺序

1. 单缓冲同步：一块完整 C 出来后立即做 Max，完成后复用槽位。
2. 给输入搬运增加 ping-pong，确认读取完成后才能覆写。
3. 再重叠 Cube 下一块计算和 Vector 当前块归约。
4. 用明确的 ready/consumed 事件管理每个缓冲槽；最后一块必须 drain。

```text
时间 →     0           1           2           3
Cube       计算 C0     计算 C1     计算 C2     ...
Vector                 归约 C0     归约 C1     归约 C2
```

是否真正重叠以 profiler 为准。临时缓冲不能在消费者读完之前复用，同一 best 向量的更新也必须有顺序。

## 7. Workspace 生命周期是必须先解决的工程问题

现有 `run_kernel` 没有外部 workspace 参数。直接调用模式下，使用高阶 API 时还可能需要系统 workspace，不能只计算 `partial_score` 的大小。[官方 workspace 说明](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0092.html)

建议分成两个阶段：

1. **正确性阶段**：在 run_kernel 内申请所需空间，启动主 kernel 与最终 kernel，显式等待完成后释放。它有同步和分配开销，但生命周期可验证。内部同步会改变返回时机，因此在性能报告中注明。
2. **性能阶段**：确认评测调用协议后，在保持入口 ABI 的前提下做按 device/stream 隔离的 workspace 管理；容量不足时安全扩容，同 stream 复用依赖流内顺序，不同 stream 使用独立 in-flight 区域。为本地驱动提供同步后的清理入口，不能在 device reset 后才释放旧地址。若评测不支持可靠清理/缓存生命周期，保留同步安全路径并计入真实开销，不假设静态指针永远有效。

还需保证 tiling 参数的传递生命周期：按值传递使用编译器支持的方式；若异步复制 tiling 到设备，Host 源内存在复制完成前保持有效。

通用路径部分分数有效载荷约 `4*B*Q` 字节，N 分片部分最大值有效载荷约 `4*B*Q*P*Tm` 字节。实际大小加入槽位对齐、AIV 子分片、系统区和 staging。所有工作区区域互不重叠，每轮有效项必须先写后读。

严禁用输出 y 或输入 x1/x2 的尾部当隐含 workspace；模板只承诺精确的输入输出容量。

## 8. 按阶段实施与退出条件

| 阶段 | 主要工作 | 完成标准 |
| --- | --- | --- |
| P0 环境和协议 | 最小 ASC 编译；确认设备、API、评测 ABI、workspace、计时边界 | FP16/BF16 基本算例可执行，有版本记录 |
| P1 测试驱动 | main 根据 metadata 动态加载；参数化 dtype/shape/属性/文件路径 | 可运行整个 smoke 集，故意错误能被 checker 拒绝 |
| P2 正确性 kernel | 小 shape 用简单 FP32 点积/Vector 路径，四布局地址明确 | smoke 和结构化测试全部通过；该路径仅用于小数据与调试 |
| P3 通用融合 | B/M 任务；同步 Matmul 完整 K；逐 N 块 Max；独立 Sum kernel | correctness+fuzz 全通过，尾块和重复执行通过 |
| P4 大尺寸 | 容量核算、完整 stress、workspace 生命周期和多 stream | 无越界、泄漏、死锁，stress 全通过 |
| P5 性能优化 | tile 搜索、输入复用、双缓冲、可选 N 分片、小 shape 特化 | 每次改动有正确性回归和逐 case 性能比较 |
| P6 收敛 | 简化分发、清理调试代码、正式 15 case（取得后） | 所有正式 case 正确，最终完整计时和可复现报告 |

不把标量四重循环作为大矩阵的通用交付路径。新增 N 分片或切换 buffer/布局策略后，重新跑所有受影响测试；纯文档改动无需重跑 NPU 性能。

## 9. 完整本地测试数据规格

### 9.1 用例展开规则

[test_data.py](test_data.py) 是可执行规格，内含每个基础 case 的 `(B,M,N,K)`、生成模式和确定性 seed 规则。

每个基础 case 展开为 **2 种 dtype × 4 种布局 = 8 个具体 case**。同一基础 case 的四种布局必须来自同一份逻辑数据，不能分别随机生成后假称验证了布局等价。case ID 形如 `c03_last_n_bfloat16_10`，末尾两位为 TX1/TX2。

| 集合 | 基础数 | 展开后 | 用途 |
| --- | --- | --- | --- |
| smoke | 5 | 40 | 快速验证调用和最基本语义 |
| correctness | 24 | 192 | 边界、数值、布局、非对齐和常见规模 |
| stress | 8 | 64 | 维度/输入容量/相似度容量边界，手动长测 |
| fuzz | 64 | 512 | 固定 seed 的小尺寸随机组合 |
| 总计 | 101 | 808 | 自建本地回归集，非官方 15 点 |

所有形状在枚举阶段校验题目约束。普通随机输入使用 `U[-1,1]`，先生成 FP32，再量化到输入 dtype；其数值未必已归一化，符合模板现有生成方式，算子不能依赖范数为 1。

### 9.2 smoke：解析结果明确的最小集合

| 基础 ID | B,M,N,K | 生成模式与期望 |
| --- | --- | --- |
| s00_dot | 1,1,1,32 | 随机点积；形状与原 case0 相同，随机数序列不冒充原 case0 |
| s01_known | 1,2,3,32 | 前两维为题目手算示例，其余补零；y=[2] |
| s02_negative | 1,2,3,32 | query=e0，document 第一维为 -1,-2,-3；y=[-2] |
| s03_order | 1,2,2,32 | 相似度 `[[10,0],[0,10]]`；y=[20] |
| s04_batch | 3,3,5,40 | query 第一维=b+1，document 第一维=(b+1)/8；y[b]=3*(b+1)^2/8 |

已知值示例一律补到合法 K，不使用原说明中的 K=2 或参考脚本演示中的 K=4 作为正式范围测试。

### 9.3 correctness：固定形状与数值模式

| 基础 ID | B,M,N,K | 覆盖目标 |
| --- | --- | --- |
| c00_zero | 2,17,33,40 | 全零，y=0 |
| c01_one | 2,17,33,40 | 全一，y=M*K |
| c02_negative_tail | 2,17,33,40 | 非对齐且全负；y=-M |
| c03_last_n | 2,17,257,40 | 最后一列唯一最大；其余为 -1，最后为 2；y=2M |
| c04_tie | 2,17,65,40 | 所有候选相等；y=M |
| c05_near_tie | 2,17,65,40 | 最后一列 1+2^-7，其余 1；两输入精度都能区分 |
| c06_k_cancel | 1,2,2,8192 | query 全 1；doc0 前半 +1 后半 -1，doc1 仅首维 1；y=M，检查不能对部分 K 做 Max |
| c07_sum_cancel | 2,33,17,40 | query 前 32 行 +1/-1 交替，末行 2^-10；document=e0；y=2^-10 |
| c08_range | 2,17,33,40 | K 维交替乘 2^-8、1、2^8；另一输入反向缩放，检验动态范围 |
| c09_m1 | 1,1,1025,72 | M=1，大 N 与尾部 |
| c10_n1 | 1,1025,1,72 | N=1，多 query 求和 |
| c11_b64 | 64,3,5,40 | B 上限，batch 可区分，y[b]=3*(b+1)^2/8 |
| c12_align16 | 2,16,16,32 | 小对齐块 |
| c13_tail15 | 2,15,15,40 | 16 前一位 |
| c14_tail17 | 2,17,17,72 | 16 后一位 |
| c15_tail31 | 2,31,33,120 | 32 两侧 |
| c16_tail63 | 2,63,65,128 | 64 两侧 |
| c17_tail127 | 2,127,129,264 | 128 两侧与 K 尾部 |
| c18_tail255 | 1,255,257,520 | 256 两侧、多块和 K 尾部 |
| c19_rect_m | 1,513,33,256 | 瘦长矩阵，M 并行 |
| c20_rect_n | 1,33,513,256 | 宽矩阵，N 分片候选 |
| c21_square | 4,128,128,128 | FP32 随机生成后按 token 归一化，再量化 |
| c22_large_k | 1,17,19,8192 | K 上限随机长点积 |
| c23_large_k_tail | 1,17,19,8184 | 接近上限且有底层块尾部 |

### 9.4 stress：按资源预算手动运行

| 基础 ID | B,M,N,K | 覆盖目标 |
| --- | --- | --- |
| t00_square | 1,2048,2048,128 | 大矩阵吞吐 |
| t01_max_mn | 1,8192,8192,32 | M/N 上限、小 K |
| t02_max_mk | 1,8192,1,8192 | X1 元素数正好 2^26，N=1 |
| t03_max_nk | 1,1,8192,8192 | X2 元素数正好 2^26，M=1 |
| t04_batch_cap | 64,128,128,8192 | 两输入均到元素上限，B/K 上限 |
| t05_all_max | 1,8192,8192,8192 | M/N/K 同时上限，约 5.50e11 MAC |
| t06_max_mn_tail | 1,8191,8191,40 | 大规模尾块 |
| t07_similarity_cap | 64,8192,8192,128 | 两输入均到元素上限；完整 FP32 相似度达 16 GiB |

t05/t07 的 FP64 golden 计算量很大，安排手动长测并缓存结果；不放入每次提交的默认快速回归。生成器默认只生成 smoke，不默认生成 stress。单个大 case 的输入文件合计最多 256 MiB，生成时 FP32 临时数组和物理重排副本会增加 CPU 内存峰值，建议预留数 GiB；逐 case 生成和运行，避免同时加载整个集合。

### 9.5 fuzz：精确的随机规格

使用 `np.random.default_rng(20260907)` 依次生成 64 个基础形状：

```text
B ∈ {1,2,3,7}
M,N ∈ {1,2,3,7,8,15,16,17,31,32,33,63,64,65,127,129}
K ∈ {32,40,56,64,72,120,128,264}
```

数据 seed 是基础 ID 的 SHA256 前四字节按 little-endian 转成整数，不使用进程随机化的 Python hash。导出的 meta 记录 NumPy 版本、seed、实际物理 shape 和每个二进制文件的 SHA256；跨环境复现以保存的二进制与 hash 为准。

### 9.6 golden 实现和磁盘格式

生成器先生成逻辑 query `[B,M,K]` 和 document `[B,N,K]`，量化后用 FP64 分 M/N 块计算 golden；每次点积覆盖完整 K，每行全局 Max 完成后用 FP64 Sum，最后转 FP32。CPU 分块只控制相似度临时内存，不模拟 NPU 的误差模式。大 case 不一次性构造完整 FP64 `[B,M,N]`。

```text
preliminary/test_data/<case_id>/
  meta.json        shape、dtype、属性、seed、字节数、hash、误差门槛
  x1.bin           连续物理布局，FP16/BF16，little-endian
  x2.bin           连续物理布局，FP16/BF16，little-endian
  golden_y.bin     B 个 FP32
  y.bin            待 NPU 测试驱动写入
```

输入精度两者都是每元素 2 字节，但 BF16 必须按 BF16 位模式读取，不能用 FP16 reinterpret。输出文件长度必须恰好 `4*B` 字节。

### 9.7 现在可以执行的命令

从项目根目录执行；Python 需要 NumPy、ml_dtypes，脚本使用 Python 3.11+ 的文件摘要接口：

```bash
python preliminary/docs/test_data.py list --suite all
python preliminary/docs/test_data.py self-check
python preliminary/docs/test_data.py generate --suite smoke --out preliminary/test_data
python preliminary/docs/test_data.py generate --suite correctness --out preliminary/test_data
python preliminary/docs/test_data.py generate --suite fuzz --out preliminary/test_data

# 大测试按一个具体 dtype/布局生成，避免默认启动全部长测。
python preliminary/docs/test_data.py generate --suite stress --case-id t05_all_max_float16_00 --out preliminary/test_data

# NPU 驱动生成 y.bin 后才执行：
python preliminary/docs/test_data.py verify --case-dir preliminary/test_data/s01_known_float16_00
```

生成器拒绝覆盖已存在 case 目录，复测直接复用旧数据，或指定新的输出根目录。`self-check` 检查全部 case 的合法性，并用小 M/N 分块对照仓库参考实现验证结构化数据、两种精度和四种布局；不编译或执行 NPU kernel。

## 10. 测试驱动改造和验证流程

现有 main 固定形状、内存大小和属性，现有 run.sh 只拷贝 case0。不能只替换输入文件就声称测到了其他 shape。

P1 将本地驱动改成可接收 case 目录，或由 Python runner 解析 meta 后以明确 CLI 参数传入 B/M/N/K、dtype、TX1/TX2 和文件路径。CLI 名称届时定义；本文不虚构当前 executable 已支持这些参数。

驱动需要：

1. 用 meta 的实际 storage shape 构造 TensorInfo，按 `2*B*M*K`、`2*B*N*K`、`4*B` 动态分配。
2. 校验输入文件精确长度，读取数据并 H2D；保持元信息在 Host 调用期间有效。
3. 为调试输出填充 NaN 哨兵，运行后确保每个 y 元素都被写入且有限。
4. 同步后写入对应 case 目录的 y.bin，再调用严格 verifier；失败记录完整 case ID 和误差。
5. 输入运行前后做设备回读 hash 比较，确保算子未修改输入；调试时增加带保护区的独立分配验证越界。
6. 每个小 case 连续执行至少 20 次，检查 y 的逐位一致性；穿插不同 shape/dtype/布局后重跑原 case，发现旧 workspace 污染。
7. 支持时添加两条 stream 的独立输入并发测试；对需要 workspace 缓存的版本额外检查分配/释放、扩容和设备切换。

附加变换测试在 runner 完成后增加：document token 排列不应改变结果；query token 排列保持数学结果不变但允许 FP32 求和误差；复制 document 列不改变数学结果；复制 query 行应使分数翻倍；batch 置换应对应置换输出。比较使用适当误差门槛，不要求不同算法/分块顺序逐位相同。

启用 N 分片时，强制运行 P=1/2/4 的同一逻辑数据，包括不同 query 行的最大值位于不同分片、最后一列唯一最大、全负值三类，和普通路径比较。这些是路径正确性测试，不能仅依赖自动分发恰好命中。

严格 verifier 自身用正确输出、错误值、少一个元素、多一个元素、额外尾字节、NaN、Inf 进行正反校验。原 checker 的 `equal_nan=True` 和允许截断的逻辑不应作为题目有限性与长度约束的替代；原代码保留供对照。

## 11. 性能测量和调参顺序

### 11.1 记录真实成本

- 首先固定输入，预热至少 10 次，再计时至少 100 次；特别昂贵的 stress 可减少次数并记录实际数量。
- 用设备事件或目标环境推荐方法测量所有计算 kernel 的总时间；分别记录主 kernel、最终归约和必要 packing/staging。
- 单独记录 CPU tiling、workspace 分配、内部同步和端到端调用延迟。不能只报告主 kernel 耗时，隐藏必要的第二个 kernel 或分配开销。
- 数据生成、FP64 golden、文件 I/O 不计入 kernel 时间；正式评分计时边界由评测协议决定。
- 普通计时与 msprof 采样分开运行，记录 median/P95，并观察设备负载、频率和首次执行影响。

使用 msprof 采集题目要求的 Task Duration、Cube/Vector 利用率、MTE 带宽相关指标；具体可用计数器和命令以安装版本为准，不保证所有设备都暴露同名指标。

### 11.2 调参优先级

| 现象 | 首先尝试 |
| --- | --- |
| 小 shape 启动成本占主导 | 小矩阵专用路径，减少 launch，避免反复分配 |
| B/M 任务太少 | 减小 Tm；再比较 N 分片的净收益 |
| Cube 算力利用不足 | 检查 tiling、尾块比例、布局、K 累加配置和输入供给 |
| Vector 成为瓶颈 | 检查 C 格式转换、逐行归约方式、Tn 和输出拆分 |
| GM 搬运占比大 | 去掉完整 C 落盘、复用 query、减少重排和 staging |
| 大 K 精度不够 | 排查降精度中间值与累加设置，再考虑稳定归约路径 |
| 随机错误或重复结果不一致 | 回退同步，检查槽位覆盖、事件、有效长度、workspace 生命周期 |

一次只改变一个主要因素。调优完成后按 shape/dtype/布局建立简洁分发规则，记录每条规则的依据和回退方案。不同路径在接近阈值的 shape 两侧都要测试。

### 11.3 性能报告格式与评分

建议逐 case 记录：`case_id, B,M,N,K, dtype,TX1,TX2, path,Tm,Tn,Tk,P, workers, workspace_bytes, max_abs_error, repeat_equal, main_us, reduce_us, total_device_us, end_to_end_us, baseline_us, speedup`。

题目一处描述融合相对拆分基线的加速收益，正式分数公式则使用最优耗时 T：

```text
Score_case = 100 / (1 + log_(1.5)(t/T))
final_score = mean(15 official case scores)
```

报告中把本地拆分基线加速比和官方评分分开；不知道 T 时不能推算官方得分。所有正式 case 精度通过是计分前提。

## 12. 最终验收清单

- [ ] FP16/BF16 × 四种布局全部通过；布局间使用相同逻辑数据验证。
- [ ] N 先 Max、M 后 Sum；K 完整累加；负值、并列值和尾部最大值正确。
- [ ] smoke/correctness/fuzz 全量通过，stress 在 NPU 上全部执行通过。
- [ ] 输出精确 B 个 FP32、有限，输入不变，无越界。
- [ ] 相同配置重复执行逐位一致；跨 shape 和多 stream 没有 workspace 污染。
- [ ] 容量与 buffer 预算经过目标硬件验证，不依赖完整 `[B,M,N]` 落盘。
- [ ] 双缓冲、N 分片和分发边界都有强制路径回归。
- [ ] 报告包含所有必要 kernel、packing 和分配/同步成本。
- [ ] 取得正式 15 个 case 后逐一复测；自建 808 个用例不能替代官方验收。

## 13. 依据和版本注意事项

计算语义、输入范围和评分以[仓库题目](problem.md)为依据，当前调用 ABI 以[模板](../batchmatmulmaxsum_template/kernel.asc)为依据。本文的任务划分、N 分片、缓冲和回归集是针对本题的设计建议。

硬件 API 参考已分别链接在融合、异步和 workspace 章节。所引用资料存在不同发布版本，CV 页面是持续开发版本；具体入口声明、空间获取方法、Matmul 输出格式和支持型号，在 P0 按目标安装版本核验后再落代码。

## 14. 本次文档交付的实际验证记录

以下仅是配套 Python 工具的本地验证结果，不代表 NPU 算子已实现或通过测试：

- 环境：Python 3.13.10、NumPy 2.3.5、ml_dtypes 0.5.4。
- 全部 808 个用例定义通过 shape 范围、输入容量约束和 ID 唯一性检查。
- 128 个 CPU golden/布局检查通过，与仓库 FP64 参考实现一致；结构化输入同时校验了解析期望值。
- 40 个 smoke 用例在临时目录完成实际二进制生成、按 dtype/shape 回读、golden 比较、文件长度和 SHA256 检查，正确输出全部被 verifier 接受。
- 错误数值、空输出、过长输出、额外尾字节、NaN、Inf 六类错误输出均被拒绝；重复生成不会覆盖旧目录。
- Markdown 本地链接与代码围栏已检查。

验证产生的临时数据已清理。交付的是可复现生成器和完整用例规格；大体积输入二进制按需生成，不在 docs 中存放数 GiB 测试文件。尚未进行 NPU 编译、设备正确性测试、stress golden 全量生成或性能测量。
