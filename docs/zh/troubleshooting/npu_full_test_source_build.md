# npu-full-test 源码构建 torch_npu 的一次复盘

## 背景

`npu-full-test.yml` 从直接安装预编译 `torch_npu` wheel，改成了在 `build_torch_npu` 作业里从源码构建 wheel，再分发给各个 test shard。

第一次改造后，GitHub Actions 中的源码构建失败，失败 job 为：

- 当前失败 job：<https://github.com/Ascend/pytorch/actions/runs/24230124335/job/70739705506>
- 参考 job：<https://github.com/computing-infra/pytorch-infra/actions/runs/23893095476/job/69671038225>

## 现象

失败发生在 `setup.py build bdist_wheel` 的 CMake configure 阶段，而不是编译链接阶段。

关键报错为：

```text
CMake Error at third_party/Tensorpipe/third_party/libuv/CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

随后 Python 构建层抛出：

```text
subprocess.CalledProcessError: Command '['cmake', ..., '-DBUILD_TENSORPIPE=on', ...]' returned non-zero exit status 1.
```

这说明失败点是 Tensorpipe/libuv 的 CMake 兼容性，而不是 wheel 打包、artifact 上传或测试分片逻辑。

## 根因

当前仓库的 `setup.py` 默认会在未显式关闭 RPC 的情况下启用 Tensorpipe：

- `DISABLE_RPC_FRAMEWORK` 未设置时，默认值是 `FALSE`
- 进入 CMake 配置时会附加 `-DBUILD_TENSORPIPE=on`

而 Tensorpipe 依赖的 `third_party/Tensorpipe/third_party/libuv/CMakeLists.txt` 仍然声明了较老的 `cmake_minimum_required`。在当前失败环境里，这个旧声明已经从“告警”升级成了“错误”，导致配置直接失败。

## 为什么参考流水线没撞到

参考流水线并没有关闭 RPC/Tensorpipe。它的日志明确显示：

- 执行的是 `git clone --depth=1 --recurse-submodules https://gitcode.com/Ascend/pytorch.git ascend_pytorch`
- 构建时也走到了 Tensorpipe/libuv
- `BUILD_TENSORPIPE` 在 CMake configure 阶段出现过提示
- `third_party/Tensorpipe/third_party/libuv/CMakeLists.txt` 在参考 job 中只产生了 `CMake Deprecation Warning`，没有失败

也就是说，参考流水线确实走到了同一条 Tensorpipe 路径，但当时没有因为该 `cmake_minimum_required` 而中断。

从现有证据看，最合理的解释不是“参考流水线绕开了 Tensorpipe”，而是“参考流水线使用的有效 CMake/toolchain 状态与当前失败时不同”。

已确认的事实：

- 当前仓库与 upstream `master` 的 Tensorpipe 子模块指针一致
- 参考 job 和当前 workflow 都显示使用 `swr.cn-north-4.myhuaweicloud.com/frameworkptadapter/pytorch_2.11.0_a2_aarch64_builder:20260331`
- 但参考 job 中 libuv 是 deprecation warning，当前 job 中 libuv 是 hard error

因此，高概率是以下之一：

1. 同名容器 tag 对应的实际内容发生了漂移。
2. runner 或容器内的 `cmake` 有额外更新，导致旧兼容语义从 warning 升级为 error。

这次改造后，workflow 已额外把 `cmake --version` 和 `gcc --version` 写入 build summary，便于后续确认是否存在 toolchain 漂移。

## 为什么这次会踩坑

这次问题本质上是一个迁移假设错误：

1. 复制了参考流水线的高层结构和主构建命令，但没有先验证 full-test 场景是否真的需要 RPC/Tensorpipe。
2. 假设“参考流水线成功”就意味着“当前 PR workflow 可直接照搬”，没有把源码、子模块、toolchain、运行时间差异作为显式前提检查。
3. 看到固定容器 tag 后，默认认为 toolchain 是稳定且可复现的，这是不够严谨的。

## 修复策略

本次 workflow 采用的修复是：

- 在 `build_torch_npu` 作业中显式设置 `DISABLE_RPC_FRAMEWORK=TRUE`
- 仅为 full-test 构建测试所需的 `torch_npu` wheel，不再额外构建 RPC/Tensorpipe 组件

这样做的原因是：full-test 分片执行依赖 `torch` + `torch_npu` 的 Python 包能力，不依赖 RPC/Tensorpipe。既然该能力不是本次测试必需项，就不应让它成为构建链路的脆弱点。

## 后续准则

后续再做类似 CI 迁移或“参考别处流水线”时，至少先核对以下几点：

1. 参考流水线构建的是不是当前 PR/当前仓库这份源代码，而不是外部 clone 的另一份快照。
2. 参考流水线成功时，实际启用了哪些可选组件；不要默认全部都是必需的。
3. 对容器 tag 不要默认视为强不可变输入；关键 toolchain 版本要落到日志或 summary。
4. 如果某个组件不是当前测试目标的必要依赖，应优先在 CI 中关闭，而不是把它一起带进主路径。

## 本次沉淀

这次经验可以归纳成一句话：

> 迁移 CI 时，不要只复制“命令长得像”的方案；要先验证“依赖边界”和“toolchain 假设”是否仍然成立。