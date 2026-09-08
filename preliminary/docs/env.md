# Ascend NPU 开发环境

本文记录当前项目使用的昇腾 NPU 环境，供算子实现、切分策略、内存规划和性能测试参考。

## 云服务器 CANN 环境

| 参数 | 值 |
| --- | --- |
| 环境 | 云服务器 |
| CANN 版本 | `9.0.0` |
| `ASCEND_HOME_PATH` | `/home/developer/Ascend/cann-9.0.0` |
| 环境初始化脚本 | `/home/developer/Ascend/cann-9.0.0/set_env.sh` |

登录云服务器后使用以下命令加载与当前编译、运行一致的 CANN 环境：

```bash
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
source "$ASCEND_HOME_PATH/set_env.sh"
```

## SoC 与计算资源

| 参数 | 值 | 说明 |
| --- | ---: | --- |
| `soc_version` | `Ascend910_9362` | 当前运行环境的 SoC 版本 |
| `cube_core_num` | 20 | 每个逻辑 Device 的 Cube Core（AIC）数量 |
| `vector_core_num` | 40 | 每个逻辑 Device 的 Vector Core（AIV）数量 |

## 片上存储资源

以下容量均为单个对应计算核心可使用的片上存储容量，不是整卡容量。

| 参数 | Bytes | KiB | MiB | 所属计算侧 |
| --- | ---: | ---: | ---: | --- |
| `l1_size` | 524288 | 512 | 0.500 | Cube/AIC |
| `l0a_size` | 65536 | 64 | 0.0625 | Cube/AIC |
| `l0b_size` | 65536 | 64 | 0.0625 | Cube/AIC |
| `l0c_size` | 131072 | 128 | 0.125 | Cube/AIC |
| `ub_size` | 196608 | 192 | 0.1875 | Vector/AIV |

## 当前设备可见性

驱动管理工具版本：

- `npu-smi`：25.5.5
- `npu-smi` 顶部 `Version` 字段：25.5.5

`npu-smi info` 查询到 NPU 2 上的两个芯片：

| NPU ID | Chip ID | Physical ID | PCIe Bus ID | 名称 | 健康状态 | HBM |
| ---: | ---: | ---: | --- | --- | --- | ---: |
| 2 | 0 | 4 | `0000:0B:00.0` | `Ascend910` | OK | 65536 MiB |
| 2 | 1 | 5 | `0000:0A:00.0` | `Ascend910` | OK | 65536 MiB |

查询时的状态快照：

| Chip ID | 温度 | 功耗 | AI Core 利用率 | HBM 使用量 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 44 °C | 167.8 W | 0% | 2909 / 65536 MiB |
| 1 | 45 °C | 未提供 | 0% | 2870 / 65536 MiB |

当时没有 NPU 进程运行。功耗、温度、利用率及 HBM 使用量属于动态数据，不应作为固定硬件规格使用。

ACL 查询结果：

```text
ACL logical device count = 2
ret = 0
```

因此，当前容器内有 2 个 ACL 逻辑 Device 可用。ACL 逻辑 Device ID 通常从 `0` 到 `count - 1`，但仅凭上述输出不能确定它们与 `npu-smi` 的 NPU ID、Chip ID 和 Physical ID 之间的逐项映射；需要映射时应在当前运行环境中单独查询确认。

## 常用查询命令

加载 CANN 环境：

```bash
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
source "$ASCEND_HOME_PATH/set_env.sh"
```

查看 NPU 状态和设备编号：

```bash
npu-smi info
npu-smi info -l
npu-smi info -m
```

查询 ACL 可见的逻辑 Device 数量：

```bash
python3 - <<'PY'
import acl

ret = acl.init()
if ret != 0:
    raise RuntimeError(f"acl.init failed: {ret}")

try:
    count, ret = acl.rt.get_device_count()
    print("ACL logical device count =", count)
    print("ret =", ret)
finally:
    acl.finalize()
PY
```

## 开发注意事项

- 算子启动核数不得超过对应逻辑 Device 的 20 个 Cube Core 或 40 个 Vector Core。
- L1、L0A、L0B、L0C 和 UB 是核内局部存储，进行 tiling 时应按单核容量规划，并预留队列、双缓冲及对齐所需空间。
- 当前记录尚不包含 GM/HBM 实测带宽和指定数据类型下的 Cube 吞吐量。后续得到测试结果时，应同时记录测试工具、数据类型、单位和测试方法，避免混淆理论峰值与实测值。
