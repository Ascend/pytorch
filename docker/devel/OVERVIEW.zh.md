# TorchNPU devel

> [English](./OVERVIEW.md) | 中文

## 快速参考

- TorchNPU 由 [Ascend for PyTorch community](https://www.hiascend.com/developer/software/ai-frameworks/pytorch) 维护

- 从哪里获取帮助

   - [AscendHub 镜像仓库](https://www.hiascend.com/developer/ascendhub)
   - [TorchNPU 文档](https://www.hiascend.com/document/detail/zh/Pytorch/730/index/index.html)
   - [昇腾开发者社区](https://www.hiascend.com/developer)
   - [问题反馈](https://gitcode.com/Ascend/pytorch/issues)

---

## TorchNPU devel

`torch-npu-devel` 镜像是专为编译TorchNPU而设计的开发镜像。它在 `builder` 镜像（manylinux + Python + gcc-toolset + cmake + PyTorch CPU 依赖）基础上，集成了CANN工具包（Toolkit + Ops，可选 NNAL）。用户可在容器内完成TorchNPU wheel编译，并直接在NPU上运行。

> 镜像中不含 Driver，需在宿主机自行安装。

关于 Dockerfile 及构建脚本的更多细节，详见 [README](./README.md)。

---

## 支持的 Tags 及 Dockerfile 使用方法

### Tag 规范

Tag 遵循以下格式：

```text
<PyTorch版本号>-<CANN版本>-<硬件信息（芯片）>-<操作系统>
```

| 字段            | 值                          | 说明                      |
|-----------------|----------------------------|---------------------------|
| PyTorch 版本号 | 2.13.0                     | 镜像内预装的PyTorch版本 |
| CANN 版本       | 9.1.0                      | 镜像内预装的CANN版本 |
| 硬件信息（芯片） | 310p / 910 / 910b / a3 / 950 | 镜像适用的昇腾芯片型号        |
| 操作系统        | manylinux_2_28              | 基础镜像所使用的操作系统发行版 |

> Python 版本（默认 `3.10`）通过 `PY_VERSION` 构建参数指定，不体现在 tag 中。

### Tag(预装 PyTorch 2.13.0)

- `2.13.0-cann9.1.0-310p-manylinux_2_28`
- `2.13.0-cann9.1.0-910-manylinux_2_28`
- `2.13.0-cann9.1.0-910b-manylinux_2_28`
- `2.13.0-cann9.1.0-a3-manylinux_2_28`
- `2.13.0-cann9.1.0-950-manylinux_2_28`

### Dockerfile 构建参数

dockerfile 详见：[dockerfile](https://gitcode.com/Ascend/pytorch/blob/master/docker/devel/Dockerfile)

| 参数               | 说明                                                                | 必填 | 参考来源            | 参数取值                       |
|--------------------|------------------------------------------------------------------|----|-------------------|----------------------------|
| PY_VERSION         | Python 版本，仅安装对应版本依赖                                          | 是  | manylinux 镜像      | 3.10                        |
| TORCH_VERSION      | PyTorch 版本，格式 `x.x.x`（如 `2.13.0`）或 dev 版本（如 `2.13.0.dev20260610`） | 是  | TorchNPU 仓库发行版  | 2.13.0                      |
| DEVTOOLSET_VERSION | GCC toolset 版本                                              | 否  | Dockerfile 默认值   | 13                          |
| CANN_VERSION       | 昇腾 CANN 工具包版本                                               | 是  | CANN 基础镜像仓库   | 9.1.0                       |
| CANN_PRODUCT       | CANN 算子包产品类型                                               | 是  | CANN 产品映射       | 910b                        |
| INSTALL_NNAL       | 是否安装 NNAL 神经网络加速库                                         | 否  | Dockerfile 默认值   | 0                           |
| CANN_RELEASE_TRAIN | CANN 发布版本号，当 `CANN_VERSION` 与默认值不同时需手动指定                  | 否  | CANN 下载目录       | CANN%209.1.0                |

### CANN 产品映射

| 产品代码   | 对应产品              |
|------------|-----------------------|
| `910b`     | Atlas A2 系列         |
| `910`      | Atlas 训练系列        |
| `310p`     | Atlas 推理系列        |
| `A3`       | Atlas A3 系列         |
| `950`      | Atlas 350 加速卡      |

### 参数来源

1. OBS 上的 CANN `.run` 包按 release train 分目录存放（如 `CANN%209.1.T1`、`CANN%209.1.0`），目录名无法由版本号自动推导。因此当 `CANN_VERSION` 与默认值 `9.1.0` 不同时，必须同时通过 `CANN_RELEASE_TRAIN` 指定对应的发布版本号，否则构建会报错退出。请到 [昇腾官网下载页](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/softwareinst/instg/instg_0008.html?OS=openEuler&InstallType=local) 查找目标版本对应的目录名。

## 快速开始

### 构建 TorchNPU devel 镜像

以构建2.13.0-cann9.1.0-910b-manylinux_2_28为例：

```bash
docker build \
  --target dev \
  --build-arg PY_VERSION=3.10 \
  --build-arg TORCH_VERSION=2.13.0 \
  --build-arg CANN_VERSION=9.1.0 \
  --build-arg CANN_PRODUCT=910b \
  --build-arg CANN_RELEASE_TRAIN="CANN%209.1.0" \
  -t image_name:tag \
  -f Dockerfile .
```

所有 tag 的构建命令如下（若推送多架构镜像，如x86和arm，则需分别制作后合并）：

| 镜像 Tag                                         | 构建命令                                                                                                                                                                                                                                                                                                                                                                            |
|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `2.13.0-cann9.1.0-310p-manylinux_2_28`            | `docker build --target dev --build-arg PY_VERSION=3.10 --build-arg TORCH_VERSION=2.13.0 --build-arg CANN_VERSION=9.1.0 --build-arg CANN_PRODUCT=310p --build-arg CANN_RELEASE_TRAIN="CANN%209.1.0" -t *your-registry*/torch-npu-devel:2.13.0-cann9.1.0-310p-manylinux_2_28 --push .`                                                                |
| `2.13.0-cann9.1.0-910-manylinux_2_28`             | `docker build --target dev --build-arg PY_VERSION=3.10 --build-arg TORCH_VERSION=2.13.0 --build-arg CANN_VERSION=9.1.0 --build-arg CANN_PRODUCT=910 --build-arg CANN_RELEASE_TRAIN="CANN%209.1.0" -t *your-registry*/torch-npu-devel:2.13.0-cann9.1.0-910-manylinux_2_28 --push .`                                                                |
| `2.13.0-cann9.1.0-910b-manylinux_2_28`            | `docker build --target dev --build-arg PY_VERSION=3.10 --build-arg TORCH_VERSION=2.13.0 --build-arg CANN_VERSION=9.1.0 --build-arg CANN_PRODUCT=910b --build-arg CANN_RELEASE_TRAIN="CANN%209.1.0" -t *your-registry*/torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28 --push .`                                                                |
| `2.13.0-cann9.1.0-a3-manylinux_2_28`              | `docker build --target dev --build-arg PY_VERSION=3.10 --build-arg TORCH_VERSION=2.13.0 --build-arg CANN_VERSION=9.1.0 --build-arg CANN_PRODUCT=A3 --build-arg CANN_RELEASE_TRAIN="CANN%209.1.0" -t *your-registry*/torch-npu-devel:2.13.0-cann9.1.0-a3-manylinux_2_28 --push .`                                                                  |
| `2.13.0-cann9.1.0-950-manylinux_2_28`             | `docker build --target dev --build-arg PY_VERSION=3.10 --build-arg TORCH_VERSION=2.13.0 --build-arg CANN_VERSION=9.1.0 --build-arg CANN_PRODUCT=950 --build-arg CANN_RELEASE_TRAIN="CANN%209.1.0" -t *your-registry*/torch-npu-devel:2.13.0-cann9.1.0-950-manylinux_2_28 --push .`                                                                |

**注意**：若构建环境需要配置代理，需通过 `--build-arg` 传入代理变量，例如：

```bash
docker build \
  --build-arg HTTP_PROXY=http://proxy.example.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.example.com:8080 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  ... \
  --push .
```

代理地址和端口请替换为实际环境的值。

### 运行 TorchNPU devel 容器

dev 镜像需透传 NPU 驱动、设备节点与 npu-smi 工具：

```bash
docker run -d --rm \
    --name torch-npu-devel \
    --privileged \
    -v /dev:/dev \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi \
    -v /var/log/npu:/usr/slog \
    -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/base:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver \
    -it torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28 \
    tail -f /dev/null
```

验证容器内 NPU 可见：

```bash
docker exec torch-npu-devel npu-smi info
```

若输出 NPU 卡列表则挂载成功。

> `--privileged` 仅用于简化设备透传。如需最小权限，可改为显式 `--device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc`（每张卡需单独 `--device`）。

### 如何二次开发

`torch-npu-devel` 镜像内置于 `pytorch/docker/devel` 目录下，提供 `Dockerfile` 与 `builder.sh`，覆盖编译与运行两类场景。请根据自身需求选择目标镜像：

| 场景 | 目标镜像 | 说明 |
|------|----------|------|
| 仅编译 TorchNPU wheel | `builder` | manylinux + Python + gcc-toolset + cmake + PyTorch CPU 依赖，不含 CANN；无需 NPU 驱动 |
| 编译并运行于 NPU | `dev` | 在 `builder` 基础上叠加 CANN（Toolkit + Ops，可选 NNAL）；需宿主机已安装 NPU 驱动 |

构建流程、参数与脚本使用方式详见镜像内的 [README.md](https://gitcode.com/Ascend/pytorch/blob/master/docker/devel/README.md)。下面给出基于 `dev` 镜像叠加用户软件的示例：

```bash
# 以 TorchNPU devel 镜像为基础镜像，叠加用户软件
FROM *your-registry*/torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28

RUN yum install -y gcc ...

...
```

#### 切换 Python 版本与 torch CPU whl 说明

> **支持的 Python 版本范围**：支持 **`3.10 / 3.11 / 3.12 / 3.13 / 3.14`**，默认 `3.10`。
>
> **切换 Python 版本**：通过把 `python`/`pip` 软链接重新指向对应版本的 `/opt/python/cpXY-cpXY` 解释器，后续阶段（torch CPU 安装、TorchNPU 编译、CANN 环境）统一使用该解释器。若更换成非默认的 python 版本，需要手动重新下载相应的编译与开发依赖。
>
> **torch CPU whl 选择**：从 PyTorch 官方索引安装 CPU 版 torch，`pip` 会根据当前解释器自动选择对应 ABI 标签的 whl（`cp310`/`cp311`/`cp312`/`cp313`/`cp314`），无需手动指定：
>
> - 稳定版本（如 `2.13.0`）：`pip${PY_VERSION} install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu`
> - dev/nightly 版本（匹配 `*.dev*`，如 `2.13.0.dev20260610`）：`pip${PY_VERSION} install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/nightly/cpu`
> - 其他详细的编译与开发依赖可见对应仓库目录下的 [requirements(以2.13.0版本为例)](https://gitcode.com/Ascend/pytorch/blob/master/requirements_2.13.txt)。

---

## 支持的硬件

| 产品示例                       | 架构             |
|--------------------------------|----------------|
| Atlas 800T A2、Atlas 900 A2 PoD | ARM64 / x86_64 |
| Atlas 800T A3                  | ARM64 / x86_64 |
| Atlas 300I Pro、Atlas 300V Pro  | ARM64 / x86_64 |

---

## 许可证/免责声明

查看这些镜像中包含的 TorchNPU 的[许可证信息](https://gitcode.com/Ascend/pytorch/blob/master/LICENSE)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。

发布的昇腾软件镜像均是社区版本，不对商业负责、仅作为生产实践的参考。

责任说明在镜像启动信息和昇腾镜像平台展示。
