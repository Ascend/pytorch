# Ascend PyTorch

> [English](./OVERVIEW.md) | 中文

## 快速参考

- PTA 由 [Ascend PyTorch community](https://www.hiascend.com/developer/software/ai-frameworks/pytorch) 维护

- 从哪里获取帮助

   - [AscendHub 镜像仓库](https://www.hiascend.com/developer/ascendhub)
   - [PTA 文档](https://www.hiascend.com/document/detail/zh/Pytorch/730/index/index.html)
   - [昇腾开发者社区](https://www.hiascend.com/developer)
   - [问题反馈](https://gitcode.com/Ascend/pytorch/issues)

---

## Ascend PyTorch

Ascend Extension for PyTorch插件是基于昇腾的深度学习适配框架，使昇腾NPU可以支持PyTorch框架，为PyTorch框架的使用者提供昇腾AI处理器的超强算力。

---

## 支持的Tags 及 Dockerfile 使用方法

### Tag规范

Tag 遵循以下格式：

```
<PTA版本号>-<硬件信息（芯片）>-<操作系统>-<Python版本>
```

| 字段        | 示例值                        | 说明                           |
|-----------|----------------------------|------------------------------|
| PTA 版本号   | v26.0.0-beta.1-torch2.10.0 | 对应 torch_npu 官方发布 Tag 中的版本标识 |
| 硬件信息（芯片）  | 910b / 310p / a3           | 昇腾芯片型号标识                     |
| 操作系统      | ubuntu / openeuler         | 基础镜像所使用的操作系统发行版              |
| Python 版本 | py3.11                     | 镜像内置 Python 大版本号             |
| 系统架构      | arm / x86                  | 宿主机及镜像运行的硬件架构                |
| CANN 版本   | 9.0.0-beta.2               | 昇腾 CANN 工具包版本号               |

### 构建参数选择参考地址

1. torch_npu 官方发布版本 / 补丁版本查询

    https://gitcode.com/Ascend/pytorch/releases

2. CANN 基础镜像仓库（镜像标签、版本、系统查询）

    https://quay.io/repository/ascend/cann?tab=tags

### 构建参数

| 参数                        | 说明                               | 必填 | 参考来源              | 示例值                                      |
|---------------------------|----------------------------------|----|-------------------|------------------------------------------|
| CANN_VERSION              | 昇腾 CANN 工具包版本                    | 是  | CANN 基础镜像仓库       | 9.0.0-beta.2                             |
| CHIP_ARCH                 | 昇腾芯片架构标识                         | 是  | CANN 镜像标签规则       | 910b / 310p / a3                         |
| OS                        | 基础镜像操作系统                         | 是  | CANN 镜像标签规则       | ubuntu / openeuler                       |
| OS_VERSION                | 操作系统版本                           | 是  | CANN 镜像标签规则       | 22.04 / 24.03                            |
| PY_VERSION                | 基础镜像内置 Python 版本                 | 是  | CANN 镜像标签规则       | 3.11                                     |
| ARCH                      | 宿主机硬件架构                          | 是  | 环境硬件              | arm / x86                                |
| PY_TAG                    | Python 包 ABI 标签（cp + 版本号）        | 是  | 与 PY_VERSION 严格匹配 | cp311 (PY3.11)                           |
| TORCH_NPU_RELEASE_VERSION | torch_npu 官方发布 Tag（含 pytorch 版本） | 是  | PTA 仓库发行版         | v26.0.0-beta.1-pytorch2.10.0             |
| TORCH_VERSION             | torch_npu 完整版本号                  | 是  | PTA 仓库发行版         | 2.10.0rc3                                |
| MANYLINUX_VER             | PyPI 包兼容系统版本                     | 否  | torch 官方 whl 规范   | manylinux_2_28                           |
| PIP_MIRROR_URL            | pip 安装源地址（默认清华源）                 | 否  | PyPI 镜像源          | https://pypi.tuna.tsinghua.edu.cn/simple |

> Tips: 完整whl包下载示例链接为 
> 
> https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-beta.1-pytorch2.10.0/torch_npu-2.10.0rc3-cp310-cp310-manylinux_2_28_aarch64.whl
>
> TORCH_VERSION 取 torch_npu-{}-cp310 之间所有内容

## 快速开始

### 构建 PTA 镜像

```bash
docker build \
  --build-arg CANN_VERSION=xxx \
  --build-arg CHIP_ARCH=xxx \
  --build-arg OS=xxx \
  --build-arg OS_VERSION=xxx \
  --build-arg PY_VERSION=xxx \
  --build-arg TORCH_VERSION=xxx \
  --build-arg ARCH=xxx \
  --build-arg PY_TAG=xxx \
  --build-arg TORCH_NPU_RELEASE_TAG=xxx \
  --build-arg TORCH_NPU_PATCH_TAG=xxx \
  -t 镜像名:标签 \
  -f Dockerfile .
```

### 运行 PTA 容器

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

### 如何本地构建

```bash
docker buildx build -t {your_repo}/pta:latest -f Dockerfile .
```

### 如何二次开发

```bash
# 以 PTA 镜像为基础镜像，叠加用户软件
FROM quay.io/ascend/ascend-pytorch:v26.0.0-beta.1-torch2.10.0-910b-ubuntu-py3.11 # 暂未发布，仅示例地址，仍需修改。

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

查看这些镜像中包含的 PTA 的[许可证信息](https://gitcode.com/Ascend/pytorch/blob/master/LICENSE)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。