# TorchNPU 开发镜像

本目录提供 Dockerfile 及构建脚本，用于生成 TorchNPU 的构建与开发镜像：`builder` 镜像提供 torch_npu 编译环境（不含 CANN），`dev` 镜像在此基础上叠加 CANN 运行环境。

## 1 镜像介绍

Dockerfile 采用多阶段构建，分为三个阶段：

```Text
base        manylinux + Python 软链接 + pip 源 + 基础系统包（curl/wget/git/libgomp...）
  └── builder   base + gcc-toolset + cmake + PyTorch CPU 依赖（torch/numpy/pyyaml）  ← 编译环境
        └── dev     builder + CANN（Toolkit + Ops + NNAL 可选）  ← 运行环境
```

- **builder**（默认）：用于编译 torch_npu wheel，不含 CANN
- **dev**：基于 builder，叠加 CANN 运行环境；继承全部编译工具链，可在容器内直接重新编译

> Driver 不包含在镜像中，用户需在宿主机自行安装。镜像仅提供 CANN 编译环境，运行时需宿主机已安装匹配的 NPU 驱动。
> 镜像中预装的 PyTorch CPU 版本默认为 v2.9.0 分支对应的版本（`torch==2.9.0`）。可通过构建参数 `TORCH_VERSION` 指定其他版本（如 `2.7.1`），详见 [构建参数参考](#构建参数参考)。。

## 2 镜像构建

### 2.1 docker build 构建

#### 构建参数参考

| ARG | 默认值 | 说明 | 适用阶段 |
|-----|--------|------|---------|
| `PY_VERSION` | `3.10` | Python 版本，仅安装对应版本依赖 | all |
| `TORCH_VERSION` | `2.9.0` | PyTorch 版本，格式 `x.x.x`（如 `2.9.0`） | all |
| `DEVTOOLSET_VERSION` | `13` | GCC toolset 版本 | builder |
| `CANN_VERSION` | `9.1.0_beta.1` | CANN 版本号 | dev |
| `CANN_PRODUCT` | `910b` | CANN 算子包产品类型 | dev |
| `INSTALL_NNAL` | `0` | 是否安装 NNAL 神经网络加速库 | dev |
| `CANN_RELEASE_TRAIN` | - | CANN 发布版本号，仅当`CANN_VERSION`与默认值不同时需手动指定 | dev |

#### 使用示例

```Shell
cd pytorch/docker/builder/X86    # 或 ARM
```

##### 场景一：构建 builder 镜像

```Shell
docker build -t manylinux-builder:v1 \
    --target builder \
    --build-arg PY_VERSION=3.10 \
    .
```

##### 场景二：构建 dev 镜像

```Shell
docker build -t manylinux-builder:v1 \
    --target dev \
    --build-arg PY_VERSION=3.10 \
    --build-arg CANN_VERSION=9.1.0_beta.1 \
    --build-arg CANN_PRODUCT=910b \
    .
```

> dev 阶段基于 builder，继承全部编译工具和 Python 依赖，再叠加 CANN。
> 指定 CANN 版本时需同时给出 `CANN_RELEASE_TRAIN`，详见 [构建参数参考](#构建参数参考)。
> **关于 CANN 版本与 release train**：OBS 上的 CANN `.run` 包按 release train 分目录存放（如 `CANN%209.1.T1`、`CANN%209.0.0`），目录名无法由版本号自动推导。因此当 `CANN_VERSION` 与默认值 `9.1.0_beta.1` 不同时，必须同时通过 `CANN_RELEASE_TRAIN` 指定对应的发布版本号，否则构建会报错退出。请到 [昇腾官网下载页](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/softwareinst/instg/instg_0008.html?OS=openEuler&InstallType=local) 查找目标版本对应的目录名。

#### CANN 产品映射

| 产品代码 | 对应产品 |
|----------|----------|
| `910b` | Atlas A2 系列 |
| `910` | Atlas 训练系列 |
| `310p` | Atlas 推理系列 |
| `310b` | Atlas 200I/500 A2 推理 |
| `A3` | Atlas A3 系列 |
| `950` | Atlas 350 加速卡 |

### 2.2 脚本构建

> `builder.sh` 为 **Linux 构建脚本**，自动检测架构（X86/ARM）、构建镜像并启动容器。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `-p, --python VERSION` | Python 版本：3.9、3.10、3.11、3.12、3.13（默认：3.10） |
| `--torch-version VER` | PyTorch 版本，格式 `x.x.x`（如 `2.9.0`） |
| `--no-cache` | 构建镜像时不使用 Docker 缓存 |
| `--cann` | 构建 dev 镜像（含 CANN 运行环境），默认构建 builder 镜像 |
| `--cann-version VER` | CANN 版本（默认：9.1.0_beta.1） |
| `--cann-product PROD` | CANN 产品类型：950、A3、910b、910、310p、310b（默认：910b） |
| `--cann-release-train VER` | CANN 发布版本号，如 `CANN%209.1.T1`。当 `--cann-version` 与默认值不同时**必填** |
| `--nnal` | 安装 CANN NNAL 神经网络加速库（需配合 `--cann`） |
| `-h, --help` | 显示帮助信息 |

#### 使用示例

```Shell
bash builder.sh                                          # 默认：Python 3.10，不含 CANN
bash builder.sh -p 3.11                                  # 使用 Python 3.11
bash builder.sh --cann                                   # 含 CANN（默认 910b ops）
bash builder.sh --cann --cann-product A3                 # 含 CANN for Atlas A3
bash builder.sh --cann --cann-version 9.0.0 \
    --cann-release-train CANN%209.0.0 \
                                                          # 指定 CANN 版本（须同时给出对应release train）
bash builder.sh --cann --nnal                            # 含 CANN + NNAL
bash builder.sh --cann --no-cache                        # 含 CANN 且不使用缓存
```

## 3 启动容器

> torch_npu **编译**无需 CANN/驱动；CANN 与 NPU 驱动仅在**运行时**（`import torch_npu`、调用 NPU 算子）需要。因此根据镜像类型选择不同的启动方式。
>
> 若已通过 `builder.sh` 构建，脚本会自动构建镜像并启动容器，可直接执行下方的 `docker exec` 命令进入容器；本节其余命令仅适用于手动执行 `docker build` 后需自行启动容器的场景。

### 场景一：builder 镜像

仅需挂载源码，无需 NPU 驱动透传。以下命令须在 `pytorch/docker/builder/X86`（或 `ARM`）目录下执行，与手动构建步骤的工作目录一致：

```Shell
docker rm -f torch-npu-builder 2>/dev/null

docker run -d --rm \
    --name torch-npu-builder \
    -v $(pwd)/../../..:/home/pytorch \
    -e PY_VERSION=3.10 \
    manylinux-builder:v1 \
    tail -f /dev/null
```

### 场景二：dev 镜像

需透传 NPU 驱动、设备节点与 npu-smi 工具：

```Shell
docker rm -f torch-npu-builder 2>/dev/null

docker run -d --rm \
    --name torch-npu-builder \
    --privileged \
    -v /dev:/dev \
    -v $(pwd)/../../..:/home/pytorch \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi \
    -v /var/log/npu:/usr/slog \
    -e PY_VERSION=3.10 \
    -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/base:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver \
    manylinux-builder:v1 \
    tail -f /dev/null
```

参数说明：

| 参数 | 用途 |
|------|------|
| `--privileged -v /dev:/dev` | 透传全部 NPU 字符设备（davinci*/davinci_manager/devmm_svm/hisi_hdc），避免逐个 `--device` |
| `-v /usr/local/Ascend/driver` | 宿主机驱动库与 firmware，CANN 运行时依赖 |
| `-v /usr/local/Ascend/add-ons` | 驱动附加组件（如 profiler） |
| `-v /usr/local/sbin/npu-smi` | NPU 状态查看工具（依赖 driver/lib64 下的 so） |
| `-v /var/log/npu` | NPU 日志输出目录 |
| `-e LD_LIBRARY_PATH=...` | 让 npu-smi 与 CANN 找到 `libc_sec.so`/`libdrvdsmi_host.so` 等依赖 |

验证容器内 NPU 可见：

```Shell
docker exec torch-npu-builder npu-smi info
```

若输出 NPU 卡列表则挂载成功。

> `--privileged` 仅用于简化设备透传。如需最小权限，可改为显式 `--device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc`（每张卡需单独 `--device`）。

进入容器：

```Shell
docker exec -it torch-npu-builder bash
```

## 附录：Windows 环境参考

> 本附录仅供 Windows 用户参考。Windows 上 `builder.sh` 依赖的 bash 通常不可用（WSL 入口 `bash.exe` 默认指向 `docker-desktop` 分发，非完整 Linux），且 MSYS 路径转换会导致 `docker run -v` 挂载失败，建议直接使用 PowerShell 等价命令执行前文所述流程。

Windows 下命令格式与 Linux 的主要差异：

- **工作目录**：需进入对应架构子目录，如 `pytorch\docker\builder\X86`（ARM 机器用 `ARM` 子目录）。
- **路径分隔符**：使用反斜杠 `\`，而非正斜杠 `/`。
- **续行符**：PowerShell 使用反引号 `` ` ``，而非反斜杠 `\`。
- **挂载路径**：使用 Windows 风格（`E:\...`），Docker Desktop for Windows 会自动转换；切勿使用 MSYS 风格（`/e/...`）。
- **错误重定向**：删除已存在容器时使用 `2>$null` 而非 `2>/dev/null` 屏蔽错误输出。

请参照前文 Linux 流程，将相关命令按上述规则改写为 PowerShell 等价命令执行。
