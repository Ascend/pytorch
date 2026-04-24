# Ascend PyTorch

> English | [中文](./OVERVIEW.zh.md)

## Quick Reference

- PTA is maintained by the [Ascend PyTorch community](https://www.hiascend.com/developer/software/ai-frameworks/pytorch)

- Where to get help

   - [AscendHub Image Repository](https://www.hiascend.com/developer/ascendhub)
   - [PTA Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/730/index/index.html)
   - [Ascend Developer Community](https://www.hiascend.com/developer)
   - [Issue Feedback](https://gitcode.com/Ascend/pytorch/issues)

---

## Ascend PyTorch

The Ascend Extension for PyTorch plugin is a deep learning adaptation framework based on Ascend, enabling Ascend NPUs to support the PyTorch framework and providing users of the PyTorch framework with the powerful computing power of Ascend AI processors.

---

## Supported Tags and Dockerfile Usage

### Tag Specification

Tags follow the format:

```
<PTA_version>-<chip>-<os>-<python_version>
```

| Field          | Example Value              | Description                                              |
|----------------|----------------------------|----------------------------------------------------------|
| PTA Version    | v26.0.0-beta.1-torch2.10.0 | Version identifier in the official torch_npu release tag |
| Chip           | 910b / 310p / a3           | Ascend chip model identifier                             |
| OS             | ubuntu / openeuler         | OS distribution used for the base image                  |
| Python Version | py3.11                     | Major Python version pre-installed in the image          |
| System Arch    | arm / x86                  | Host and image runtime hardware architecture             |
| CANN Version   | 9.0.0-beta.2               | Ascend CANN toolkit version                              |

### Build Argument Reference Links

1. Check torch_npu official releases / patch releases

    https://gitcode.com/Ascend/pytorch/releases

2. CANN base image repository (image tags, versions, OS query)

    https://quay.io/repository/ascend/cann?tab=tags

### Build Arguments

| Argument                  | Description                                                | Required | Reference Source          | Example Value                            |
|---------------------------|------------------------------------------------------------|----------|---------------------------|------------------------------------------|
| CANN_VERSION              | Ascend CANN toolkit version                                | Yes      | CANN base image repo      | 9.0.0-beta.2                             |
| CHIP_ARCH                 | Ascend chip architecture identifier                        | Yes      | CANN image tag rules      | 910b / 310p / a3                         |
| OS                        | Base image operating system                                | Yes      | CANN image tag rules      | ubuntu / openeuler                       |
| OS_VERSION                | Operating system version                                   | Yes      | CANN image tag rules      | 22.04 / 24.03                            |
| PY_VERSION                | Python version pre-installed in base image                 | Yes      | CANN image tag rules      | 3.11                                     |
| ARCH                      | Host hardware architecture                                 | Yes      | Environment hardware      | arm / x86                                |
| PY_TAG                    | Python package ABI tag (cp + version number)               | Yes      | Strictly match PY_VERSION | cp311 (PY3.11)                           |
| TORCH_NPU_RELEASE_VERSION | Official torch_npu release tag (including PyTorch version) | Yes      | PTA repo releases         | v26.0.0-beta.1-pytorch2.10.0             |
| TORCH_VERSION             | Full torch_npu version number                              | Yes      | PTA repo releases         | 2.10.0rc3                                |
| MANYLINUX_VER             | PyPI package compatible system version                     | No       | torch official wheel spec | manylinux_2_28                           |
| PIP_MIRROR_URL            | pip installation source URL (Tsinghua mirror by default)   | No       | PyPI mirror sources       | https://pypi.tuna.tsinghua.edu.cn/simple |

> Tips: Example full whl package download URL
> 
> https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-beta.1-pytorch2.10.0/torch_npu-2.10.0rc3-cp310-cp310-manylinux_2_28_aarch64.whl
>
> TORCH_VERSION takes the content between `torch_npu-` and `-cp310`

## Quick Start

### Build PTA Image

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
  -t image_name:tag \
  -f Dockerfile .
```

### Run PTA Container

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

### Local Build

```bash
docker buildx build -t {your_repo}/pta:latest -f Dockerfile .
```

### Secondary Development

```bash
# Use PTA image as base image and add user software
FROM quay.io/ascend/ascend-pytorch:v26.0.0-beta.1-torch2.10.0-910b-ubuntu-py3.11-arm # Not yet published, example only, subject to change.

RUN apt update -y && \
    apt install gcc ...

...
```

---

## Supported Hardware

| Chip Series | Product Examples                | Architecture   |
|-------------|---------------------------------|----------------|
| Ascend 910B | Atlas 800T A2, Atlas 900 A2 PoD | ARM64 / x86_64 |
| Ascend A3   | Atlas 800T A3                   | ARM64 / x86_64 |
| Ascend 310P | Atlas 300I Pro, Atlas 300V Pro  | ARM64 / x86_64 |

---

## License

See the [license information](https://gitcode.com/Ascend/pytorch/blob/master/LICENSE) for PTA included in these images.

Like all container images, pre-installed software packages (Python, system libraries, etc.) may be subject to their own licenses.