## Lot cases.txt

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0.0 |
| subgroups | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0.0 |

### Worst groups (Fail + Crash + Timeout + InternalError + other)

| group | total | Pass | Fail | NotSupported | QualityWarning | CompatibilityWarning | Crash | Timeout | InternalError | NotRun | Other | pass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subgroups.basic | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0.0 |

### Most frequent failure messages

| count | status | message | example |
|---:|---|---|---|
| 1 | DeviceLost | `vk.waitForFences(device, #u, &amp;fence, VK_TRUE, timeoutNanos): VK_ERROR_DEVICE_LOST at vkCmdUtil.cpp:#` | dEQP-VK.subgroups.basic.graphics.subgroupelect |

### Most frequent NotSupported reasons

| count | message | example |
|---:|---|---|
