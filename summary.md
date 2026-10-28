## Lot cases.txt

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 10 | 1 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 10.0 |
| subgroups | 10 | 1 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 10.0 |

### Worst groups (Fail + Crash + Timeout + InternalError + other)

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subgroups.arithmetic | 6 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| subgroups.ballot_broadcast | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| subgroups.vote | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0.0 |

### Most frequent failure messages

| count | status | message | example |
|---:|---|---|---|
| 8 | Fail | `Failed!` | dEQP-VK.subgroups.ballot_broadcast.graphics.subgroupbroadcastfirst_bvec3 |
| 1 | DeviceLost | `vk.waitForFences(device, #u, &amp;fence, VK_TRUE, timeoutNanos): VK_ERROR_DEVICE_LOST at vkCmdUtil.cpp:#` | dEQP-VK.subgroups.vote.graphics.subgroupallequal_bvec2 |

### Most frequent NotSupported reasons

| count | message | example |
|---:|---|---|
