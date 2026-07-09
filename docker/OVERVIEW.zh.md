# TorchNPU

> [English](./OVERVIEW.md) | 中文

## 快速参考

- TorchNPU 由 [Ascend PyTorch community](https://www.hiascend.com/developer/software/ai-frameworks/pytorch) 维护

- 从哪里获取帮助

   - [AscendHub 镜像仓库](https://www.hiascend.com/developer/ascendhub)
   - [TorchNPU 文档](https://www.hiascend.com/document/detail/zh/Pytorch/730/index/index.html)
   - [昇腾开发者社区](https://www.hiascend.com/developer)
   - [问题反馈](https://gitcode.com/Ascend/pytorch/issues)

---

## TorchNPU

TorchNPU插件是基于昇腾的深度学习适配框架，使昇腾NPU可以支持PyTorch框架，为PyTorch框架的使用者提供昇腾AI处理器的超强算力。

---

## 支持的Tags 及 Dockerfile 使用方法

### Tag规范

Tag 遵循以下格式：

```text
<TorchNPU版本号>-<硬件信息（芯片）>-<操作系统>-<Python版本>
```

| 字段           | 值                            | 说明               |
|--------------|------------------------------|------------------|
| TorchNPU 版本号 | 2.10.0                       | 详见readme中版本说明部分  |
| 硬件信息（芯片）     | 910b / 310p / a3             | 昇腾芯片型号标识         |
| 操作系统         | ubuntu22.04 / openeuler24.03 | 基础镜像所使用的操作系统发行版  |
| Python 版本    | py3.11                       | 镜像内置 Python 大版本号 |

### 构建参数选择参考地址

### Tag

- `2.10.0-310p-ubuntu22.04-py3.11`
- `2.10.0-310p-openeuler24.03-py3.11`
- `2.10.0-910b-ubuntu22.04-py3.11`
- `2.10.0-910b-openeuler24.03-py3.11`
- `2.10.0-a3-ubuntu22.04-py3.11`
- `2.10.0-a3-openeuler24.03-py3.11`
- `2.9.0.post2-310p-ubuntu22.04-py3.11`
- `2.9.0.post2-310p-openeuler24.03-py3.11`
- `2.9.0.post2-910b-ubuntu22.04-py3.11`
- `2.9.0.post2-910b-openeuler24.03-py3.11`
- `2.9.0.post2-a3-ubuntu22.04-py3.11`
- `2.9.0.post2-a3-openeuler24.03-py3.11`
- `2.8.0.post4-310p-ubuntu22.04-py3.11`
- `2.8.0.post4-310p-openeuler24.03-py3.11`
- `2.8.0.post4-910b-ubuntu22.04-py3.11`
- `2.8.0.post4-910b-openeuler24.03-py3.11`
- `2.8.0.post4-a3-ubuntu22.04-py3.11`
- `2.8.0.post4-a3-openeuler24.03-py3.11`
- `2.7.1.post4-310p-ubuntu22.04-py3.11`
- `2.7.1.post4-310p-openeuler24.03-py3.11`
- `2.7.1.post4-910b-ubuntu22.04-py3.11`
- `2.7.1.post4-910b-openeuler24.03-py3.11`
- `2.7.1.post4-a3-ubuntu22.04-py3.11`
- `2.7.1.post4-a3-openeuler24.03-py3.11`

### Dockerfile构建参数

dockerfile详见：[dockerfile](https://gitcode.com/Ascend/pytorch/blob/master/docker/Dockerfile)

| 最新参数                      | 说明                              | 必填 | 参考来源              | 参数取值                                     |
|---------------------------|---------------------------------|----|-------------------|------------------------------------------|
| CHIP_ARCH                 | 昇腾芯片架构标识                        | 是  | CANN 镜像标签规则       | 910b / 310p / a3                         |
| OS                        | 基础镜像操作系统                        | 是  | CANN 镜像标签规则       | ubuntu / openeuler                       |
| OS_VERSION                | 操作系统版本                          | 是  | CANN 镜像标签规则       | 22.04 / 24.03                            |
| PY_VERSION                | 基础镜像内置 Python 版本                | 是  | CANN 镜像标签规则       | 3.11                                     |
| CANN_VERSION              | 昇腾 CANN 工具包版本                   | 是  | CANN 基础镜像仓库       | 9.0.0                                    |
| ARCH                      | 宿主机硬件架构                         | 是  | 环境硬件              | arm / x86                                |
| PY_TAG                    | Python 包 ABI 标签（cp + 版本号）       | 是  | 与 PY_VERSION 严格匹配 | cp311                                    |
| TORCH_NPU_RELEASE_VERSION | TorchNPU 官方发布 Tag（含 pytorch 版本） | 是  | TorchNPU 仓库发行版    | v26.0.0-pytorch2.10.0                    |
| TORCH_NPU_PATCH_TAG       | TorchNPU 官方发布包名里的版本号          | 是  | TorchNPU 仓库发行版    | 2.10.0                    |
| MANYLINUX_VER             | PyPI 包兼容系统版本                    | 否  | torch 官方 whl 规范   | manylinux_2_28                           |
| PIP_MIRROR_URL            | pip 安装源地址（默认清华源）                | 否  | PyPI 镜像源          | https://pypi.tuna.tsinghua.edu.cn/simple |

### 参数来源

1. 镜像标签、操作系统及其版本查询：[CANN 基础镜像仓库](https://quay.io/repository/ascend/cann?tab=tags)的tag。

2. TORCH_NPU_RELEASE_VERSION、TORCH_NPU_PATCH_TAG 参数的取值来自 [TorchNPU 官方发布版本](https://gitcode.com/Ascend/pytorch/releases)。以 whl 包下载地址为例：

https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.10.0/torch_npu-2.10.0-cp310-cp310-manylinux_2_28_aarch64.whl

- TORCH_NPU_RELEASE_VERSION 取 `download/` 与 `/torch_npu-` 之间的部分，如 `v26.0.0-pytorch2.10.0`。
- TORCH_NPU_PATCH_TAG 取 `torch_npu-` 与 `-cp310` 之间的部分，如 `2.10.0`。

## 快速开始

### 构建 TorchNPU 镜像

以构建2.10.0-a3-ubuntu22.04-py3.11镜像为例：

```bash
docker build \
  --build-arg CHIP_ARCH=a3 \
  --build-arg OS=ubuntu \
  --build-arg OS_VERSION=22.04 \
  --build-arg PY_VERSION=3.11 \
  --build-arg CANN_VERSION=9.0.0 \
  --build-arg ARCH=arm \
  --build-arg PY_TAG=cp311 \
  --build-arg TORCH_NPU_RELEASE_TAG=v26.0.0-pytorch2.10.0 \
  --build-arg TORCH_NPU_PATCH_TAG=2.10.0 \
  -t image_name:tag \
  -f Dockerfile .
```

**注意**：若构建环境需要配置代理，需通过 `--build-arg` 传入代理变量，例如：

```bash
docker build \
  --build-arg HTTP_PROXY=http://proxy.example.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.example.com:8080 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  ... \
  -f Dockerfile .
```

代理地址和端口请替换为实际环境的值。

### 运行 TorchNPU 容器

```bash
docker run \
    --name pta_container \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it ascend/pta:tag bash
```

### 如何二次开发

```bash
# 以 TorchNPU 镜像为基础镜像，叠加用户软件
FROM quay.io/ascend/torch-npu:2.10.0-910b-ubuntu22.04-py3.11

RUN apt update -y && \
    apt install gcc ...

...
```

---

## 支持的硬件

| 芯片系列    | 产品示例                           | 架构             |
|---------|--------------------------------|----------------|
| 昇腾 910B | Atlas 800T A2、Atlas 900 A2 PoD | ARM64 / x86_64 |
| 昇腾 A3   | Atlas 800T A3                  | ARM64 / x86_64 |
| 昇腾 310P | Atlas 300I Pro、Atlas 300V Pro  | ARM64 / x86_64 |

---

## 许可证

查看这些镜像中包含的 TorchNPU 的[许可证信息](https://gitcode.com/Ascend/pytorch/blob/master/LICENSE)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
