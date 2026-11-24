# 代码排查参考

## op_plugin 执行框架

### EXEC_NPU_CMD 宏的执行流程

`EXEC_NPU_CMD` 是 op_plugin 中所有 opapi 算子调用的核心宏。理解其执行流程对回归分析至关重要。

```
EXEC_NPU_CMD(aclnnApi, ...args)
│
├── 1. 获取函数地址
│   ├── getWorkspaceSizeFuncAddr = GetOpApiFuncAddr("aclnnApiGetWorkspaceSize")
│   └── opApiFuncAddr = GetOpApiFuncAddr("aclnnApi")
│
├── 2. 获取 NPU stream
│   └── acl_stream = c10_npu::getCurrentNPUStream().stream(false)
│
├── 3. [新版本] 捕获 deterministic 快照
│   └── snapshot = c10_npu::CaptureDeterministicSnapshot()
│
├── 4. 查询 PTA cache
│   ├── [新版本] ApplyDeterministicSnapshot(snapshot, true)  // 先设置 deterministic
│   ├── 构建哈希键: stream + deterministic + op_name + tensor_args + device
│   ├── PTAGetExecCache(hashId, &workspace_size)
│   └── 如果命中: 直接用缓存的 executor 执行 opApiFunc -> return true
│
├── 5. [仅旧版本] 设置 deterministic
│   └── at_npu::native::SetDeterministic()
│
├── 6. GetWorkspaceSize
│   ├── ConvertTypes(...args) -> 转换参数为 CANN 类型
│   ├── getWorkspaceSizeFunc(converted_params) -> 获取 workspace 大小
│   └── NPU_CHECK_ERROR(workspace_status) -> 失败则抛异常
│
├── 7. 分配 workspace
│   └── if (workspace_size != 0): unsafe_empty_workspace(workspace_size)
│
└── 8. 执行算子
    ├── opApiFunc(workspace_addr, workspace_size, executor, acl_stream)
    └── OpCommand::RunOpApiV2(op_name, acl_call) -> 放入 task queue 异步执行
```

### PTA Cache 哈希键组成

```cpp
// 哈希缓冲区构建顺序:
add_param_to_buf(aic_num);           // AIC 核心数 (uint32_t)
add_param_to_buf(aiv_num);           // AIV 核心数 (uint32_t)
add_param_to_buf(deterministic);     // deterministic 状态
                                      //   旧版本: bool (1 byte)
                                      //   新版本: uint32_t (4 bytes)
add_param_to_buf(op_name);           // 算子名 (string)
add_param_to_buf(tensor_args...);    // 各 tensor 参数 (shape + dtype + strides + ...)
add_param_to_buf(device);            // 设备 ID (int8_t)
add_param_to_buf(stream_ptr);        // stream 指针 (uintptr_t)
```

`add_param_to_buf` 对不同类型的处理：
- `bool`: `MEMCPY_TO_BUF(&value, 1)` — 1 字节
- `uint32_t`: `MEMCPY_TO_BUF(&value, 4)` — 4 字节
- `at::Tensor`: shape + dtype + strides + storage_offset + storage_shape (变长)
- `at::Scalar`: 按 ScalarType 写入对应大小的值
- `string`: 字符串数据 + 长度

### deterministic 机制

```
CaptureDeterministicSnapshot()
│
├── deterministic_algorithms_enabled = at::globalContext().deterministicAlgorithms()
├── requested_level = GetDeterministicLevel()
├── effective_level = deterministic_algorithms_enabled
│                     ? (requested_level == 0 ? 1 : requested_level)
│                     : 0
└── backend = IsSupportDeterministicLevel3() ? V2 : Legacy
    └── V2 需要 CANN >= 9.2.0

ApplyDeterministicSnapshot(snapshot, isOpapi)
│
├── if backend == V2:
│   └── AclrtCtxSetSysParamOpt(ACL_OPT_DETERMINISTIC, effective_level)
│       // 直接设置 deterministic level (0/1/2/3)
│
└── if backend == Legacy:
    └── ApplyLegacyDeterministicSnapshotLocked()
        ├── AclrtCtxSetSysParamOpt(ACL_OPT_DETERMINISTIC, deterministic ? 1 : 0)
        ├── AclrtCtxSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, strong ? 1 : 0)
        └── AclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, deterministic ? 1 : 0)
```

## 内存分配器

### apply_tensor_without_format 调用链

```
OpPreparation::apply_tensor_without_format(src, sizes)
│
└── apply_tensor_use_empty(sizes, src.options())
    │
    └── NPUNativeFunctions::empty(sizes, dtype, layout, device, pin_memory, MemoryFormat::Contiguous)
        │
        ├── 分配 NPU 内存 (不初始化)
        ├── empty_tensor_restride(MemoryFormat::Contiguous)
        ├── StorageDescHelper::SetDesc(tensor, size, strides)
        └── [新版本] if should_fill_empty_deterministic():
            └── fill_empty_deterministic_(tensor)  // 仅在 deterministic 开启时填零
```

### should_fill_empty_deterministic 条件

```cpp
inline bool should_fill_empty_deterministic() {
    return at::globalContext().deterministicAlgorithms() &&
           at::globalContext().deterministicFillUninitializedMemory() &&
           at_npu::native::env::CheckFillUninitializedMemory();
}
```

三个条件同时为 true 时才填零：
1. `deterministicAlgorithms()` — 全局 deterministic 开关
2. `deterministicFillUninitializedMemory()` — 填充未初始化内存开关
3. `CheckFillUninitializedMemory()` — NPU 侧环境变量 `TORCH_NPU_FILL_UNINITIALIZED_MEMORY`

CI 环境中这三个条件默认都为 false，因此 `empty()` 不填零。

## 常见回归模式

### 模式 1：opapi 路径不填零 + 算子不写入全部输出

```
apply_tensor_without_format() → empty() → 内存未初始化
         ↓
EXEC_NPU_CMD(aclnnXxx, ...) → 算子对某些输入（空输入/边界场景）不写入全部输出
         ↓
输出包含未初始化内存内容
         ↓
assertEqual 失败
```

对比 aclop 路径通常有显式填零：
```cpp
if (indices.numel() == 0) {
    output = npu_preparation::apply_tensor(weight);
    acl_op::fill_(output, 0);  // 显式填零
}
```

### 模式 2：PTA cache 哈希键变化

```
代码变更改变了哈希键的某个字段（如 bool → uint32_t）
         ↓
所有旧缓存条目失效（哈希值不匹配）
         ↓
算子需要重新执行 GetWorkspaceSize 创建新 executor
         ↓
执行路径/内存状态发生变化
         ↓
暴露潜在缺陷
```

### 模式 3：deterministic 设置时序变化

```
代码变更改变了 SetDeterministic/ApplyDeterministicSnapshot 的调用位置
         ↓
旧版本: 缓存命中时不调用 SetDeterministic
新版本: 无论缓存命中与否都调用 ApplyDeterministicSnapshot
         ↓
CANN 运行时状态设置频率变化
         ↓
影响算子编译/执行缓存
```

### 模式 4：内存分配器重构

```
NPUCachingAllocator 代码重构（如 mempool 支持）
         ↓
内存块的缓存、复用、释放模式变化
         ↓
output_tensor 分配到的内存块内容不同
         ↓
暴露未初始化内存缺陷
```

## 排查命令速查

```bash
# 对比两个 op_plugin 版本中特定算子文件
curl -s "https://raw.githubusercontent.com/Ascend/op-plugin/<OLD_SHA>/op_plugin/ops/opapi/XxxOpApi.cpp" > old.cpp
curl -s "https://raw.githubusercontent.com/Ascend/op-plugin/<NEW_SHA>/op_plugin/ops/opapi/XxxOpApi.cpp" > new.cpp
diff old.cpp new.cpp

# 用本地 git 仓库做带 -w 的 diff（忽略空白变化）
git diff <OLD_SHA>..<NEW_SHA> -w -- torch_npu/csrc/aten/common/TensorFactories.cpp

# 统计实际逻辑变更行数（排除格式化）
git diff <OLD_SHA>..<NEW_SHA> -w -- <file> | grep "^[-+]" | grep -v "^[-+][-+]" | wc -l

# 查看 commit 的完整 diff
curl -s "https://api.github.com/repos/Ascend/pytorch/commits/<SHA>" | python3 -c "
import json, sys; d = json.load(sys.stdin)
for f in d.get('files', []):
    print(f'=== {f[\"filename\"]} ===')
    print(f.get('patch', '')[:2000])
"
```
