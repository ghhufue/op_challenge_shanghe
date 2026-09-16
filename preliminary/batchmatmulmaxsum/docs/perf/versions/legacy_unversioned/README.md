# Legacy unversioned performance data

本目录保存性能版本制度建立前的历史采集。不同 round 可能对应不同 commit、不同候选 key 或开发阶段，因此整个目录不能视为一个一致的实现版本，也不能代表当前性能。

| Historical round | Known meaning |
|---|---|
| [round_001](round_001/summary.txt) | `(4,128,128,128)`, key 100 synchronous baseline |
| [round_002](round_002/summary.txt) | `(4,128,128,128)`, key 110 vector reduction |
| [round_003](round_003/summary.txt) | `(1,33,513,256)`, key 100 synchronous baseline |
| [round_004](round_004/summary.txt) | `(4,128,128,128)`, key 121 async double buffer |
| [round_005](round_005/summary.txt) | `(1,33,513,256)`, key 120 async path |
| [round_006](round_006/summary.txt) | `(1,33,513,256)`, historical key 130 split-N path |
| [round_007](round_007/summary.txt) | `(1,33,513,256)`, historical key 121 comparison |

这些数据仅用于理解优化演进。当前数据必须从 [../../CURRENT_VERSION](../../CURRENT_VERSION) 指向的版本目录读取。

## Revision record

| Date | Change |
|---|---|
| 2026-09-16 | Moved the former root-level rounds 001-007 here and marked them non-current. |
