# BatchMatmulMaxSum 真机性能测评流程

本文给出从无上下文状态开始，对 `BatchMatmulMaxSum` 当前代码执行真机性能验收、候选策略对比、`msprof` 深度采集和回归判断的完整命令。开始前必须先读 [README.md](README.md) 和 [CURRENT_VERSION](CURRENT_VERSION)；当前性能结论维护在 [analysis.md](analysis.md)。

## 适用范围和固定口径

- 工程类型：Ascend C Kernel 直调，单进程、单设备，不是 MC2 多 rank 算子。
- 当前项目约定使用 `msprof`，不使用 `msprof op`。
- 不使用 `msprof_profile_run.sh --compare/--quick`：这两个模式要求标准的 `model.py`、`model_new_ascendc.py` 和 JSONL 对比目录，本工程采用自己的 `run_cases.py`/`tune_cases.py` 直调测试入口。这里先用工程脚本获得稳态 Host 延迟，再用 `msprof_profile_run.sh` 标准模式采集设备指标。
- 直调程序内部固定调用 `aclrtSetDevice(0)`；不要在同一个逻辑 Device 上并行跑多个性能任务，也不要根据 `npu-smi` 的物理 ID 猜测 ACL 逻辑 ID。
- Host 延迟来自可执行文件内部计时，范围是一次 kernel launch 加 `aclrtSynchronizeStream`；不含输入 H2D、输出 D2H、测试数据生成和进程启动。
- 稳态 Host 测量默认预热 10 次、采样 100 次；关键回归对比使用 200 次。
- Kernel 深度数据使用 `ops-profiling` 的标准模式：预热 3 次，依次采集 7 组 `aic-metrics`，再采集一次 sample-based 逐核数据。
- 不把 Host p50 和 `msprof` Task Duration 当成同一个指标；两者分别反映端到端调用成本和设备 kernel 本体成本。
- 当前机器有 20 个 Cube Core。其他机器运行时必须重新记录硬件、CANN 和频率状态，不能直接横向比较绝对耗时。

## 0. 初始化路径和环境

以下命令均从仓库根目录执行：

```bash
cd /home/developer/workspace/op_challenge_shanghe

REPO_ROOT=$(git rev-parse --show-toplevel)
OP_DIR="$REPO_ROOT/preliminary/batchmatmulmaxsum"
CANN_ROOT=/home/developer/Ascend/cann-9.0.0
BUILD_DIR="$OP_DIR/build_perf_latest"
PERF_DOC_DIR="$OP_DIR/docs/perf"
PERF_VERSION=$(tr -d '\r\n' < "$PERF_DOC_DIR/CURRENT_VERSION")
VERSION_DIR="$PERF_DOC_DIR/versions/$PERF_VERSION"
RESULT_DIR="$OP_DIR/tuning_results/$PERF_VERSION"
PROF_SKILL="$REPO_ROOT/.agents/skills/ops-profiling"

export ASCEND_HOME_PATH="$CANN_ROOT"
source "$CANN_ROOT/set_env.sh"
[[ "$PERF_VERSION" =~ ^v[0-9]{3}$ ]]
test -f "$VERSION_DIR/manifest.md"
mkdir -p "$RESULT_DIR"
```

确认关键路径存在：

```bash
test -f "$OP_DIR/CMakeLists.txt"
test -f "$OP_DIR/scripts/run_cases.py"
test -f "$PROF_SKILL/scripts/msprof_profile_run.sh"
test -f "$PROF_SKILL/scripts/msprof_perf_summary.py"
command -v npu-smi
command -v msprof
```

## 1. 固化性能版本、代码和设备上下文

首先读取当前版本及其状态：

```bash
printf 'PERF_VERSION=%s\n' "$PERF_VERSION"
sed -n '1,120p' "$VERSION_DIR/manifest.md"
```

版本门禁：

- 若本轮只是重测完全相同的核心实现，继续在当前版本下增加 round。
- 若 `kernel.asc`、`main.asc`、`kernels/**`、`host/**`、`tiling/**`、submission、tiling 配置或影响运行行为的编译/生成逻辑发生变化，必须先把 `CURRENT_VERSION` 加一，并创建状态为 `UNMEASURED` 的新 manifest。
- 新版本不得复制旧版本 round。测量完成前，`analysis.md` 必须明确说明当前版本暂无有效性能数据。
- 根目录 [AGENTS.md](../../../../AGENTS.md) 对 AI agent 强制执行该门禁。

先记录测试对象，避免把旧二进制、脏工作树或其他进程的影响归因到当前代码：

```bash
git status --short --branch
git log -1 --format=fuller
npu-smi info
```

验收条件：

- 明确记录 commit ID；若要求测试远端最新代码，确认当前分支与远端跟踪分支一致。
- 业务源码没有未说明的本地修改。
- NPU `Health` 为 `OK`。
- 正式测试前没有其他 NPU 进程。若有竞争进程，停止测量并记录原因，不要继续生成可比较结论。

当前工程的参考环境见 [../../../docs/env.md](../../../docs/env.md)。

## 2. 静态检查和从当前源码全新构建

不要复用仓库中历史 `build_*` 二进制。使用独立的、已被 `.gitignore` 排除的构建目录：

```bash
cd "$OP_DIR"

python3 scripts/generate_tiling_catalog.py --check
pytest -q tests

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build "$BUILD_DIR" -j4

test -x "$BUILD_DIR/batch_matmul_max_sum_custom"
```

若构建失败，性能测评到此终止。不要退回到旧二进制继续出报告。

## 3. 代表性正确性和稳态 Host 延迟

一次覆盖六类关键路径、两种 dtype 和四种输入布局，共 48 组：

```bash
cd "$OP_DIR"

python3 scripts/run_cases.py \
  --exe "$BUILD_DIR/batch_matmul_max_sum_custom" \
  --cann-root "$CANN_ROOT" \
  --suite all \
  --case-id s00_dot \
  --case-id c16_tail63 \
  --case-id c20_rect_n \
  --case-id c21_square \
  --case-id c22_large_k \
  --case-id t03_max_nk \
  --warmups 10 \
  --repeats 100 \
  --timeout 600 \
  --report "$RESULT_DIR/latest_default_perf.json"
```

用例用途：

| Case | Shape `(B,M,N,K)` | 主要观察点 |
|---|---|---|
| `s00_dot` | `(1,1,1,32)` | tiny/reference 路径和固定启动开销 |
| `c16_tail63` | `(2,63,65,128)` | M/N 非对齐尾块 |
| `c20_rect_n` | `(1,33,513,256)` | 小 M、大 N、单核欠占用 |
| `c21_square` | `(4,128,128,128)` | 常规多核 async double-buffer 路径 |
| `c22_large_k` | `(1,17,19,8192)` | 大 K 和布局敏感性 |
| `t03_max_nk` | `(1,1,8192,8192)` | 最大 N/K、split-N 和全卡并行 |

最低验收要求：

- 48/48 全部通过。
- `repeat_within_tolerance=true`。
- 非 bitwise equal 可以接受，但必须记录 `max_repeat_abs_error` 并确认低于精度门限。
- 报告中的 `tiling_key`、`launch_blocks` 和 `split_n` 与预期策略一致。

快速查看结果：

```bash
jq -r '.[] | [
  .id,
  .tiling_key,
  .launch_blocks,
  .split_n,
  (.p50_call_ms * 1000),
  (.p95_call_ms * 1000),
  .repeat_bitwise_equal,
  .max_repeat_abs_error
] | @tsv' "$RESULT_DIR/latest_default_perf.json"
```

## 4. 候选策略的同条件对照

这一步回答性能变化来自哪条策略，而不仅是默认策略最终有多快。每条命令都固定 FP16、非转置布局并采样 200 次。

### 4.1 常规 8 核路径：同步与异步双缓冲

```bash
python3 scripts/tune_cases.py \
  --exe "$BUILD_DIR/batch_matmul_max_sum_custom" \
  --cann-root "$CANN_ROOT" \
  --suite correctness \
  --case-id c21_square_float16_00 \
  --key 100 --key 121 \
  --warmups 10 --repeats 200 \
  --output "$RESULT_DIR/latest_c21_candidates.json"
```

### 4.2 单核欠占用路径：同步、异步和不应启用的 split-N

```bash
python3 scripts/tune_cases.py \
  --exe "$BUILD_DIR/batch_matmul_max_sum_custom" \
  --cann-root "$CANN_ROOT" \
  --suite correctness \
  --case-id c20_rect_n_float16_00 \
  --key 100 --key 121 --key 130 \
  --warmups 10 --repeats 200 \
  --output "$RESULT_DIR/latest_c20_candidates.json"
```

### 4.3 最大 N/K 路径：单核与 20-way split-N

```bash
python3 scripts/tune_cases.py \
  --exe "$BUILD_DIR/batch_matmul_max_sum_custom" \
  --cann-root "$CANN_ROOT" \
  --suite stress \
  --case-id t03_max_nk_float16_00 \
  --key 121 --key 130 \
  --warmups 10 --repeats 200 \
  --output "$RESULT_DIR/latest_t03_candidates.json"
```

汇总候选数据：

```bash
for PERF_JSON in \
  "$RESULT_DIR/latest_c21_candidates.json" \
  "$RESULT_DIR/latest_c20_candidates.json" \
  "$RESULT_DIR/latest_t03_candidates.json"
do
  jq -r '.[] | [
    .id,
    .candidate_key,
    .candidate_name,
    .launch_blocks,
    .split_n,
    (.p50_call_ms * 1000),
    (.p95_call_ms * 1000)
  ] | @tsv' "$PERF_JSON"
done
```

加速比统一使用 `baseline_p50 / candidate_p50`。不要用 mean 与 p50 混算。

## 5. `msprof` 深度采集

先完成第 3 步，因为以下命令复用已经生成的 case 数据。每个标准采集约执行一分钟，会产生 7 组 task-based 指标和一组 sample-based 数据。

```bash
RUN_TAG=$(date +%Y%m%d_%H%M%S)
PROF_ROOT="$RESULT_DIR/prof_raw/$RUN_TAG"
mkdir -p "$PROF_ROOT"
```

### 5.1 常规 8 核 async double-buffer 路径

```bash
bash "$PROF_SKILL/scripts/msprof_profile_run.sh" \
  --warm-up=3 \
  --output="$PROF_ROOT/c21" \
  -- "$BUILD_DIR/batch_matmul_max_sum_custom" \
  4 128 128 128 float16 0 0 \
  "$REPO_ROOT/preliminary/test_data/c21_square_float16_00" 1
```

### 5.2 单核欠占用路径

```bash
bash "$PROF_SKILL/scripts/msprof_profile_run.sh" \
  --warm-up=3 \
  --output="$PROF_ROOT/c20" \
  -- "$BUILD_DIR/batch_matmul_max_sum_custom" \
  1 33 513 256 float16 0 0 \
  "$REPO_ROOT/preliminary/test_data/c20_rect_n_float16_00" 1
```

### 5.3 20-way split-N 路径

```bash
bash "$PROF_SKILL/scripts/msprof_profile_run.sh" \
  --warm-up=3 \
  --output="$PROF_ROOT/t03" \
  -- "$BUILD_DIR/batch_matmul_max_sum_custom" \
  1 1 8192 8192 float16 0 0 \
  "$REPO_ROOT/preliminary/test_data/t03_max_nk_float16_00" 1
```

### 5.4 解析并归档到当前版本的 `round_NNN`

```bash
C21_GROUP=$(find "$PROF_ROOT/c21" -maxdepth 1 -type d \
  -name 'PROF_GROUP_*' | sort | tail -n1)
C20_GROUP=$(find "$PROF_ROOT/c20" -maxdepth 1 -type d \
  -name 'PROF_GROUP_*' | sort | tail -n1)
T03_GROUP=$(find "$PROF_ROOT/t03" -maxdepth 1 -type d \
  -name 'PROF_GROUP_*' | sort | tail -n1)

archive_prof_group() {
  PROF_GROUP_PATH=$1
  ARCHIVE_STAGE=$(mktemp -d /tmp/bmms_prof_archive_XXXXXX)

  # The shared parser always writes <ops_dir>/docs/perf/round_NNN, so parse in
  # an isolated staging root and then place the result under this version.
  python3 "$PROF_SKILL/scripts/msprof_perf_summary.py" \
    "$PROF_GROUP_PATH" "$ARCHIVE_STAGE"

  LAST_ROUND=$(find "$VERSION_DIR" -maxdepth 1 -type d \
    -name 'round_[0-9][0-9][0-9]' -printf '%f\n' \
    | sed 's/^round_//' | sort -n | tail -n1)
  if [ -n "$LAST_ROUND" ]; then
    NEXT_ROUND=$((10#$LAST_ROUND + 1))
  else
    NEXT_ROUND=1
  fi
  ROUND_NAME=$(printf 'round_%03d' "$NEXT_ROUND")
  test ! -e "$VERSION_DIR/$ROUND_NAME"

  mv "$ARCHIVE_STAGE/docs/perf/round_001" "$VERSION_DIR/$ROUND_NAME"
  sed -i \
    "s|$ARCHIVE_STAGE/docs/perf/round_001/|$VERSION_DIR/$ROUND_NAME/|" \
    "$VERSION_DIR/$ROUND_NAME/summary.txt"
  find "$VERSION_DIR/$ROUND_NAME" -type f -name '*.csv' \
    -exec sed -i 's/[[:space:]]*$//' {} +
  printf 'Archived %s\n' "$VERSION_DIR/$ROUND_NAME"
}

archive_prof_group "$C21_GROUP"
archive_prof_group "$C20_GROUP"
archive_prof_group "$T03_GROUP"
```

上述包装流程会在当前 `VERSION_DIR` 中创建连续的 `round_NNN`。归档完成后，立即在该版本的 `manifest.md` 中记录 commit、shape、dtype、布局、tiling key 和 round 映射；全部验收完成后将状态从 `UNMEASURED` 改为 `MEASURED`，并更新 [analysis.md](analysis.md)。

## 6. 深度指标判定规则

从每个 round 的 `summary.txt` 读取以下指标：

- `Task Duration`：设备 kernel 时延。
- `BlockDim`、`cube_utilization`：全卡并行覆盖。
- AIC/AIV pipe busy：核内主导流水。
- `aicore_time`、`aiv_time`：最长侧耗时。
- `(max-min)/max`：逐核负载不均衡度。
- 头开销：Task Duration 与最长核时间的差值。

主 Bound 使用 `ops-profiling` 的严格优先级，从上到下匹配第一条：

1. MTE2 busy 大于 80%，或 MTE2 为最大项且大于 70%。
2. Cube busy 大于 80%，或 Cube 为最大项且大于 70%。
3. Vector PUSHQ busy 大于 80%。
4. FIXP busy 大于 80%。
5. MTE3 busy 大于 80%。
6. Scalar 或 Scalar LD/ST busy 大于 80%。
7. 均不满足时写“无严格 bound”，不能把最大百分比直接命名为 bound。

额外判断：

- `launch_blocks / 20` 与全卡 Cube 利用率接近时，低利用率首先是任务并行度问题。
- Host p50 远大于 Task Duration 时，端到端受 launch/sync 固定成本主导。
- 全卡利用率接近 100% 但 MAC busy 很低时，不代表有效 Cube 计算效率高；还要结合 padding 和 `cube_fops` 判断无效计算。
- `transposeX1/transposeX2` 布局必须分开统计。当前 policy 不按布局选 key，布局差异可能远大于 tiling key 差异。

## 7. 与上一提交做同机回归对比

绝对耗时会受 DVFS、设备温度和其他系统负载影响。判断最新提交是否回退时，应在同一时段编译上一提交并交替测量，而不是直接与历史文档数字比较。

```bash
# 从上一性能版本的 manifest 读取 Core implementation commit。
# v001 建立时使用 5e38cba；后续版本不要简单假设 HEAD^ 是核心基线。
BASE_COMMIT=5e38cba
PREV_DIR=$(mktemp -d /tmp/bmms_prev_XXXXXX)

git archive "$BASE_COMMIT" | tar -x -C "$PREV_DIR"
cmake \
  -S "$PREV_DIR/preliminary/batchmatmulmaxsum" \
  -B "$PREV_DIR/build" \
  -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build "$PREV_DIR/build" -j4

python3 "$OP_DIR/scripts/run_cases.py" \
  --exe "$PREV_DIR/build/batch_matmul_max_sum_custom" \
  --cann-root "$CANN_ROOT" \
  --suite all \
  --case-id c20_rect_n_float16_00 \
  --case-id c21_square_float16_00 \
  --case-id t03_max_nk_float16_00 \
  --warmups 10 --repeats 200 --timeout 600 \
  --report "$RESULT_DIR/previous_commit_comparison.json"
```

回归结论至少报告三项：p50 相对变化、p95 相对变化、正确性是否一致。约 1% 级变化通常需要重复交替测试后才能归因给代码。

## 8. 结束检查和交付清单

```bash
git diff --check
git status --short --branch
npu-smi info
```

一次完整测评应留下：

- commit、CANN、SoC、Cube Core 数和设备空闲状态。
- `CURRENT_VERSION`、版本 manifest 状态和核心 commit。
- 静态测试与构建结果。
- `latest_default_perf.json` 的通过数和代表性 p50/p95。
- 候选 key 的同条件加速比。
- 当前版本目录下三个新 `round_NNN` 的映射和瓶颈判定。
- 与上一提交的同机回归结论。
- 当前未测内容，例如官方拆分算子基线或官方隐藏测试点。
- 更新后的 [analysis.md](analysis.md)。

不要提交 `build_perf_latest`、`preliminary/test_data`、`tuning_results` 或原始 `PROF_GROUP_*`；它们均是可再生临时产物。需要长期保留的 profiler 摘要和筛选后的 CSV 只能归档到 `docs/perf/versions/$PERF_VERSION/round_NNN`。

## 修订记录

| 日期 | 基线提交 | 说明 |
|---|---|---|
| 2026-09-16 | `v001` | 引入性能实现版本门禁、版本 manifest 和版本内 round 归档流程。 |
| 2026-09-16 | `7caae52` | 根据实际真机流程固化环境检查、全新构建、48 组稳态测量、候选对照、三类 `msprof` 深度采集和同机回归方法。 |
