# TorchNPU开发镜像

本目录提供Dockerfile及构建脚本，用于生成TorchNPU的构建与开发镜像：`builder` 镜像提供TorchNPU编译环境（不含CANN），`dev` 镜像在此基础上叠加CANN运行环境。使用该文档前，可通过 `npu-smi info` 检查宿主机是否已安装NPU驱动（driver），并以此判断是否可以使用 `dev` 镜像进行开发。

## 1 镜像介绍

Dockerfile采用多阶段构建，分为三个阶段：

```Text
base        manylinux + Python 软链接 + pip 源 + 基础系统包（curl/wget/git/libgomp...）
  └── builder   base + gcc-toolset + cmake + PyTorch CPU 依赖（torch/numpy/pyyaml）  ← 编译环境
        └── dev     builder + CANN（Toolkit + Ops + NNAL 可选）  ← 运行环境
```

- **builder**（默认）：用于编译TorchNPU wheel，不含CANN
- **dev**：基于builder，叠加CANN运行环境；继承全部编译工具链，可在容器内直接重新编译

> Dockerfile会自动根据当前架构（ARM/X86）拉取对应镜像。
> Driver不包含在镜像中，用户需在宿主机自行安装。
> TorchNPU的源码编译需要提前装好cpu版本的Pytorch。镜像中预装的PyTorch CPU版本默认为v2.13.0分支对应的版本（`torch==2.13.0`）。可通过构建参数 `TORCH_VERSION` 指定稳定版本（如 `2.7.1`），详见 [构建参数参考](#构建参数参考)。

## 2 镜像构建

```Shell
cd pytorch/docker/devel
export DOCKER_BUILDKIT=1
```

### 2.1 docker build构建

#### 构建参数参考

| ARG | 默认值 | 说明 | 适用阶段 |
|-----|--------|------|---------|
| `PY_VERSION` | `3.10` | Python版本，仅安装对应版本依赖 | all |
| `TORCH_VERSION` | `2.13.0` | PyTorch版本，格式 `x.x.x`（如 `2.13.0`）或dev版本（如 `2.13.0.dev20260610`）；dev版本从nightly源安装 | all |
| `DEVTOOLSET_VERSION` | `13` | GCC toolset版本 | builder |
| `CANN_VERSION` | `9.1.0` | CANN版本号 | dev |
| `CANN_PRODUCT` | `910b` | CANN算子包产品类型 | dev |
| `INSTALL_NNAL` | `0` | 是否安装NNAL神经网络加速库 | dev |
| `CANN_RELEASE_TRAIN` | - | CANN发布版本号，仅当`CANN_VERSION`与默认值不同时需手动指定 | dev |

#### 使用示例

##### 场景一：构建builder镜像

```Shell
docker build -t manylinux-builder:v1 \
    --target builder \
    --build-arg PY_VERSION=3.10 \
    .
```

##### 场景二：构建dev镜像

```Shell
docker build -t manylinux-builder:v1 \
    --target dev \
    --build-arg TORCH_VERSION=2.13.0 \
    --build-arg PY_VERSION=3.10 \
    --build-arg CANN_VERSION=9.1.0 \
    --build-arg CANN_PRODUCT=910b \
    .
```

> dev阶段基于builder，继承全部编译工具和Python依赖，再叠加CANN。
> 指定CANN版本时需同时给出 `CANN_RELEASE_TRAIN`，详见 [构建参数参考](#构建参数参考)。
> **关于CANN版本与release train**：OBS上的CANN `.run` 包按release train分目录存放（如 `CANN%209.1.T1`、`CANN%209.1.0`），目录名无法由版本号自动推导。因此当 `CANN_VERSION` 与默认值 `9.1.0` 不同时，必须同时通过 `CANN_RELEASE_TRAIN` 指定对应的发布版本号，否则构建会报错退出。请到 [昇腾官网下载页](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/softwareinst/instg/instg_0008.html?OS=openEuler&InstallType=local) 查找目标版本对应的目录名。

#### CANN产品映射

| 产品代码 | 对应产品 |
|----------|----------|
| `910b` | Atlas A2系列 |
| `910` | Atlas训练系列 |
| `310p` | Atlas推理系列 |
| `310b` | Atlas 200I/500 A2推理 |
| `A3` | Atlas A3系列 |
| `950` | Atlas 350加速卡 |

### 2.2 脚本构建

> `builder.sh` 为 **Linux构建脚本**，自动构建镜像并启动容器。

#### 参数说明

| 参数 | 说明 |
|------|------|
| `-p, --python VERSION` | Python版本：3.10、3.11、3.12、3.13（默认：3.10） |
| `--torch-version VER` | PyTorch版本，格式 `x.x.x`（如 `2.7.1`）或dev版本（如 `2.13.0.dev20260610`，默认：2.13.0） |
| `--no-cache` | 构建镜像时不使用Docker缓存 |
| `--cann` | 构建dev镜像（含CANN运行环境），默认构建builder镜像 |
| `--cann-version VER` | CANN版本（默认：9.1.0_beta.1） |
| `--cann-product PROD` | CANN产品类型：950、A3、910b、910、310p、310b（默认：910b） |
| `--cann-release-train VER` | CANN发布版本号，如 `CANN%209.1.0`。当 `--cann-version` 与默认值不同时**必填** |
| `--nnal` | 安装CANN NNAL神经网络加速库（需配合 `--cann`） |
| `-h, --help` | 显示帮助信息 |

#### 使用示例

```Shell
bash builder.sh                                          # 默认：Python 3.10，不含CANN
bash builder.sh -p 3.11                                  # 使用Python 3.11
bash builder.sh --cann                                   # 含CANN（默认910b ops）
bash builder.sh --cann --cann-product A3                 # 含CANN for Atlas A3
bash builder.sh --cann --cann-version 9.0.0 \
    --cann-release-train CANN%209.0.0 \
                                                         # 指定CANN版本（须同时给出对应release train）
bash builder.sh --cann --nnal                            # 含CANN + NNAL
bash builder.sh --cann --no-cache                        # 含CANN且不使用缓存
```

## 3 启动容器

> TorchNPU **编译**无需CANN/驱动；CANN与NPU驱动仅在**运行时**（`import torch_npu`、调用NPU算子）需要。因此根据镜像类型选择不同的启动方式。
>
> 若已通过 `builder.sh` 构建，脚本会自动构建镜像并启动容器，可直接执行下方的 `docker exec` 命令进入容器；本节其余命令仅适用于手动执行 `docker build` 后需自行启动容器的场景。

### 场景一：builder镜像

仅需挂载源码，无需NPU驱动透传。以下命令须在 `pytorch/docker/devel`目录下执行，与手动构建步骤的工作目录一致：

```Shell
docker rm -f torch-npu-builder 2>/dev/null

docker run -d --rm \
    --name torch-npu-builder \
    -v $(pwd)/../..:/home/pytorch \
    -e PY_VERSION=3.10 \
    manylinux-builder:v1 \
    tail -f /dev/null
```

### 场景二：dev镜像

需透传NPU驱动、设备节点与npu-smi工具：

```Shell
docker rm -f torch-npu-builder 2>/dev/null

docker run -d --rm \
    --name torch-npu-builder \
    --privileged \
    -v /dev:/dev \
    -v $(pwd)/../..:/home/pytorch \
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
| `--privileged -v /dev:/dev` | 透传全部NPU字符设备（davinci*/davinci_manager/devmm_svm/hisi_hdc），避免逐个 `--device` |
| `-v /usr/local/Ascend/driver` | 宿主机驱动库与firmware，CANN运行时依赖 |
| `-v /usr/local/Ascend/add-ons` | 驱动附加组件（如profiler） |
| `-v /usr/local/sbin/npu-smi` | NPU状态查看工具（依赖driver/lib64下的so） |
| `-v /var/log/npu` | NPU日志输出目录 |
| `-e LD_LIBRARY_PATH=...` | 让npu-smi与CANN找到 `libc_sec.so`/`libdrvdsmi_host.so` 等依赖 |

验证容器内NPU可见：

```Shell
docker exec torch-npu-builder npu-smi info
```

若输出NPU卡列表则挂载成功。

> `--privileged` 仅用于简化设备透传。如需最小权限，可改为显式 `--device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc`（每张卡需单独 `--device`）。

进入容器：

```Shell
docker exec -it torch-npu-builder bash
```

## 附录：Windows环境参考

> 本附录仅供Windows用户参考。Windows上 `builder.sh` 依赖的 bash 通常不可用（WSL入口 `bash.exe` 默认指向 `docker-desktop` 分发，非完整Linux），且MSYS路径转换会导致 `docker run -v` 挂载失败，建议直接使用PowerShell等价命令执行前文所述流程。

Windows下命令格式与Linux的主要差异：

- **工作目录**：需进入 `pytorch\docker\devel` 目录。
- **路径分隔符**：使用反斜杠 `\`，而非正斜杠 `/`。
- **续行符**：PowerShell使用反引号 `` ` ``，而非反斜杠 `\`。
- **挂载路径**：使用Windows风格（`E:\...`），Docker Desktop for Windows会自动转换；切勿使用MSYS风格（`/e/...`）。
- **错误重定向**：删除已存在容器时使用 `2>$null` 而非 `2>/dev/null` 屏蔽错误输出。

请参照前文Linux流程，将相关命令按上述规则改写为PowerShell等价命令执行。
