# TorchNPU

> English | [中文](./OVERVIEW.zh.md)

## Quick Reference

- TorchNPU is maintained by the [Atlas PyTorch community](https://www.hiascend.com/developer/software/ai-frameworks/pytorch)

- Where to get help

   - [Image Repository](https://www.hiascend.com/developer/ascendhub)
   - [TorchNPU Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/730/index/index.html)
   - [Developer Community](https://www.hiascend.com/developer)
   - [Issue Feedback](https://gitcode.com/Ascend/pytorch/issues)

---

## TorchNPU

The plugin is a deep learning adaptation framework based on Atlas, enabling Atlas NPUs to support the PyTorch framework and providing users of the PyTorch framework with the powerful computing power of Atlas AI processors.

---

## Supported Tags and Dockerfile Usage

### Tag Specification

Tags follow the format:

```text
<TorchNPU_version>-<chip>-<os>-<python_version>
```

| Field            | value                                                   | Description                                       |
|------------------|---------------------------------------------------------|---------------------------------------------------|
| TorchNPU Version | 2.10.0                                                  | For details, see the version notes in the readme. |
| Chip             | Specific example values can be found in the CANN mirror | chip model identifier                             |
| OS               | ubuntu22.04 / openeuler24.03                            | OS distribution used for the base image           |
| Python Version   | py3.11                                                  | Major Python version pre-installed in the image   |

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

### Dockerfile build parameters
  
See dockerfile:[dockerfile](https://gitcode.com/Ascend/pytorch/blob/master/docker/Dockerfile)

| Latest parameters         | Description                                               | Required | Reference Source          | Value                                                   |
|---------------------------|-----------------------------------------------------------|----------|---------------------------|---------------------------------------------------------|
| TORCH_VERSION             | Full TorchNPU version number                              | Yes      | TorchNPU repo releases    | 2.10.0                                                  |
| CHIP_ARCH                 | chip architecture identifier                              | Yes      | CANN image tag rules      | Specific example values can be found in the CANN mirror |
| OS                        | Base image operating system                               | Yes      | CANN image tag rules      | ubuntu / openeuler                                      |
| OS_VERSION                | Operating system version                                  | Yes      | CANN image tag rules      | 22.04 / 24.03                                           |
| PY_VERSION                | Python version pre-installed in base image                | Yes      | CANN image tag rules      | 3.11                                                    |
| CANN_VERSION              | CANN toolkit version                                      | Yes      | CANN base image repo      | 9.0.0                                                   |
| ARCH                      | Host hardware architecture                                | Yes      | Environment hardware      | arm / x86                                               |
| PY_TAG                    | Python package ABI tag (cp + version number)              | Yes      | Strictly match PY_VERSION | cp311                                                   |
| TORCH_NPU_RELEASE_VERSION | Official TorchNPU release tag (including PyTorch version) | Yes      | TorchNPU repo releases    | v26.0.0-pytorch2.10.0                                   |
| TORCH_NPU_PATCH_TAG       | TorchNPU version number in the release package name       | Yes      | TorchNPU repo releases    | 2.10.0                                                  |
| MANYLINUX_VER             | PyPI package compatible system version                    | No       | torch official wheel spec | manylinux_2_28                                          |
| PIP_MIRROR_URL            | pip installation source URL (Tsinghua mirror by default)  | No       | PyPI mirror sources       | https://pypi.tuna.tsinghua.edu.cn/simple                |

### Parameter Sources

1. Image tags, operating systems, and version information:[CANN Base Image Repository](https://quay.io/repository/ascend/cann?tab=tags) tag

2. TORCH_NPU_RELEASE_VERSION and TORCH_NPU_PATCH_TAG parameter values are from [TorchNPU Official Release Versions](https://gitcode.com/Ascend/pytorch/releases). Example with a wheel download URL:

https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.10.0/torch_npu-2.10.0-cp310-cp310-manylinux_2_28_aarch64.whl

- TORCH_NPU_RELEASE_VERSION is the part between `download/` and `/torch_npu-`, e.g. `v26.0.0-pytorch2.10.0`.
- TORCH_NPU_PATCH_TAG is the part between `torch_npu-` and `-cp310`, e.g. `2.10.0`.

## Quick Start

### Build TorchNPU Image

Taking the construction of the 2.10.0-a3-ubuntu22.04-py3.11 image as an example:

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

**Note**: If your build environment requires a proxy, pass proxy variables via `--build-arg`, for example:

```bash
docker build \
  --build-arg HTTP_PROXY=http://proxy.example.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.example.com:8080 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  ... \
  -f Dockerfile .
```

Replace the proxy address and port with your actual environment values.

### Run TorchNPU Container

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

### Secondary Development

```bash
# Use TorchNPU image as base image and add user software
FROM quay.io/ascend/torch-npu:2.10.0-910b-ubuntu22.04-py3.11 

RUN apt update -y && \
    apt install gcc ...

...
```

---

## Supported Hardware

| Product Examples                | Architecture   |
|---------------------------------|----------------|
| Atlas 800T A2, Atlas 900 A2 PoD | ARM64 / x86_64 |
| Atlas 800T A3                   | ARM64 / x86_64 |
| Atlas 300I Pro, Atlas 300V Pro  | ARM64 / x86_64 |

---

## License

See the [license information](https://gitcode.com/Ascend/pytorch/blob/master/LICENSE) for TorchNPU included in these images.

Like all container images, pre-installed software packages (Python, system libraries, etc.) may be subject to their own licenses.
