# TorchNPU devel

> English | [中文](./OVERVIEW.zh.md)

## Quick Reference

- TorchNPU is maintained by the [Atlas PyTorch community](https://www.hiascend.com/developer/software/ai-frameworks/pytorch)

- Where to get help

   - [Image Repository](https://www.hiascend.com/developer/ascendhub)
   - [TorchNPU Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/730/index/index.html)
   - [Developer Community](https://www.hiascend.com/developer)
   - [Issue Feedback](https://gitcode.com/Ascend/pytorch/issues)

---

## TorchNPU devel

The `torch-npu-devel` image is a development image designed for compiling TorchNPU. It is built on top of the `builder` image (manylinux + Python + gcc-toolset + cmake + PyTorch CPU dependencies) and integrates the CANN toolkit (Toolkit + Ops, NNAL optional). Users can compile the TorchNPU wheel inside the container and run it directly on NPU.

> Driver is not included in the image and must be installed on the host in advance.

For more details on the Dockerfile and build scripts, see [README](https://gitcode.com/Ascend/pytorch/blob/master/docker/devel/README.md).

---

## Supported Tags and Dockerfile Usage

### Tag Specification

Tags follow the format:

```text
<PyTorch_version>-<CANN_version>-<chip>-<os>
```

| Field            | value                              | Description                                       |
|------------------|------------------------------------|---------------------------------------------------|
| PyTorch Version | 2.13.0                             | Pre-installed PyTorch version |
| CANN_version     | 9.1.0                              | Pre-installed CANN version |
| Chip             | 310p / 910 / 910b / a3 / 950       | The Ascend chip models supported by the image     |
| OS               | manylinux_2_28                     | Base image OS distribution used                   |

> The Python version (default `3.10`) is specified via the `PY_VERSION` build arg and is not part of the tag.

### Latest Version (Pre-installed PyTorch 2.13.0)

The following table lists all images of the latest released version with pre-installed PyTorch 2.13.0. For tags associated with historical versions, please refer to [Supported Tags](https://gitcode.com/Ascend/pytorch/blob/master/docker/devel/supported_tags.md).

| Tag | Dockerfile | Image Contents |
|---|---|---|
| `2.13.0-cann9.1.0-310p-manylinux_2_28` | [Dockerfile](https://gitcode.com/Ascend/pytorch/blob/master/docker/devel/Dockerfile) | torch-cpu/CANN |
| `2.13.0-cann9.1.0-910-manylinux_2_28` | [Dockerfile](https://gitcode.com/Ascend/pytorch/blob/master/docker/devel/Dockerfile) | torch-cpu/CANN |
| `2.13.0-cann9.1.0-910b-manylinux_2_28` | [Dockerfile](https://gitcode.com/Ascend/pytorch/blob/master/docker/devel/Dockerfile) | torch-cpu/CANN |
| `2.13.0-cann9.1.0-a3-manylinux_2_28` | [Dockerfile](https://gitcode.com/Ascend/pytorch/blob/master/docker/devel/Dockerfile) | torch-cpu/CANN |
| `2.13.0-cann9.1.0-950-manylinux_2_28` | [Dockerfile](https://gitcode.com/Ascend/pytorch/blob/master/docker/devel/Dockerfile) | torch-cpu/CANN |

### Dockerfile build parameters

See dockerfile: [dockerfile](https://gitcode.com/Ascend/pytorch/blob/master/docker/devel/Dockerfile)

| parameters         | Description                                                                                   | Required | Reference Source        | Value                       |
|--------------------|--------------------------------------------------------------------------------------------|----------|-------------------------|-----------------------------|
| PY_VERSION         | Python version, only dependencies for the corresponding version are installed                | Yes      | manylinux image         | 3.10                        |
| TORCH_VERSION      | PyTorch version, format `x.x.x` (e.g. `2.13.0`) or dev version (e.g. `2.13.0.dev20260610`)   | Yes      | TorchNPU repo releases  | 2.13.0                      |
| DEVTOOLSET_VERSION | GCC toolset version                                                                          | No       | Dockerfile default      | 13                          |
| CANN_VERSION       | CANN toolkit version                                                                        | Yes      | CANN base image repo    | 9.1.0                       |
| CANN_PRODUCT       | CANN ops package product type                                                                | Yes      | CANN product mapping     | 910b                        |
| INSTALL_NNAL       | Whether to install NNAL neural network acceleration library                                  | No       | Dockerfile default      | 0                           |
| CANN_RELEASE_TRAIN | CANN release train; must be specified manually when `CANN_VERSION` differs from the default   | No       | CANN download directory  | CANN%209.1.0                |

### CANN Product Mapping

| Product Code | Corresponding Product     |
|--------------|---------------------------|
| `910b`       | Atlas A2 series           |
| `910`        | Atlas training series     |
| `310p`       | Atlas inference series    |
| `A3`         | Atlas A3 series           |
| `950`        | Atlas 350 accelerator card |

### Parameter Sources

1. CANN `.run` packages are stored in directories by release train on OBS (e.g. `CANN%209.1.T1`, `CANN%209.1.0`). The directory name cannot be derived from the version number, so when `CANN_VERSION` differs from the default `9.1.0`, you must specify `CANN_RELEASE_TRAIN` at the same time, otherwise the build will fail. Find the directory name corresponding to the target version on the [Ascend download page](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/softwareinst/instg/instg_0008.html?OS=openEuler&InstallType=local).

## Quick Start

### Build TorchNPU devel Image

Taking the construction of the 2.13.0-cann9.1.0-910b-manylinux_2_28 image as an example:

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

The build commands for all tags are as follows (if pushing multi-architecture images, such as x86 and ARM, you need to build each separately and then merge them):

| Image Tag                                         | Build Command                                                                                                                                                                                                                                                                                                                                                                            |
|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `2.13.0-cann9.1.0-310p-manylinux_2_28`            | `docker build --target dev --build-arg PY_VERSION=3.10 --build-arg TORCH_VERSION=2.13.0 --build-arg CANN_VERSION=9.1.0 --build-arg CANN_PRODUCT=310p --build-arg CANN_RELEASE_TRAIN="CANN%209.1.0" -t *your-registry*/torch-npu-devel:2.13.0-cann9.1.0-310p-manylinux_2_28 --push .`                                                                |
| `2.13.0-cann9.1.0-910-manylinux_2_28`             | `docker build --target dev --build-arg PY_VERSION=3.10 --build-arg TORCH_VERSION=2.13.0 --build-arg CANN_VERSION=9.1.0 --build-arg CANN_PRODUCT=910 --build-arg CANN_RELEASE_TRAIN="CANN%209.1.0" -t *your-registry*/torch-npu-devel:2.13.0-cann9.1.0-910-manylinux_2_28 --push .`                                                                |
| `2.13.0-cann9.1.0-910b-manylinux_2_28`            | `docker build --target dev --build-arg PY_VERSION=3.10 --build-arg TORCH_VERSION=2.13.0 --build-arg CANN_VERSION=9.1.0 --build-arg CANN_PRODUCT=910b --build-arg CANN_RELEASE_TRAIN="CANN%209.1.0" -t *your-registry*/torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28 --push .`                                                                |
| `2.13.0-cann9.1.0-a3-manylinux_2_28`              | `docker build --target dev --build-arg PY_VERSION=3.10 --build-arg TORCH_VERSION=2.13.0 --build-arg CANN_VERSION=9.1.0 --build-arg CANN_PRODUCT=A3 --build-arg CANN_RELEASE_TRAIN="CANN%209.1.0" -t *your-registry*/torch-npu-devel:2.13.0-cann9.1.0-a3-manylinux_2_28 --push .`                                                                  |
| `2.13.0-cann9.1.0-950-manylinux_2_28`             | `docker build --target dev --build-arg PY_VERSION=3.10 --build-arg TORCH_VERSION=2.13.0 --build-arg CANN_VERSION=9.1.0 --build-arg CANN_PRODUCT=950 --build-arg CANN_RELEASE_TRAIN="CANN%209.1.0" -t *your-registry*/torch-npu-devel:2.13.0-cann9.1.0-950-manylinux_2_28 --push .`                                                                |

**Note**: If your build environment requires a proxy, pass proxy variables via `--build-arg`, for example:

```bash
docker build \
  --build-arg HTTP_PROXY=http://proxy.example.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.example.com:8080 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  ... \
  --push .
```

Replace the proxy address and port with your actual environment values.

### Run TorchNPU devel Container

The dev image needs to pass through the NPU driver, device nodes, and npu-smi tool:

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

Verify that the NPU is visible inside the container:

```bash
docker exec torch-npu-devel npu-smi info
```

If the NPU card list is displayed, the mount is successful.

> `--privileged` is only used to simplify device passthrough. For minimum privileges, use explicit `--device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc` (each card requires a separate `--device`).

### Secondary Development

The `torch-npu-devel` image is built from the `pytorch/docker/devel` directory, which provides `Dockerfile` and `builder.sh` covering both compilation and runtime scenarios. Choose the target image according to your needs:

| Scenario | Target Image | Description |
|----------|--------------|-------------|
| Compile TorchNPU wheel only | `builder` | manylinux + Python + gcc-toolset + cmake + PyTorch CPU dependencies, no CANN; no NPU driver required |
| Compile and run on NPU | `dev` | Adds CANN (Toolkit + Ops, optional NNAL) on top of `builder`; requires NPU driver installed on the host |

For the build process, parameters, and script usage, see the [README.md](https://gitcode.com/Ascend/pytorch/blob/master/docker/devel/README.md) in the image. The following example adds user software on top of the `dev` image:

```bash
# Use the TorchNPU devel image as the base image and add user software
FROM *your-registry*/torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28

RUN yum install -y gcc ...

...
```

#### Switching Python Version and torch CPU Wheel Selection

> **Supported Python version range**: Supports **`3.10 / 3.11 / 3.12 / 3.13 / 3.14`**, default `3.10`.
>
> **Switching Python version**: Re-points the `python`/`pip` symlinks to the corresponding `/opt/python/cpXY-cpXY` interpreter, so all subsequent stages (torch CPU install, TorchNPU build, CANN env) use the new interpreter consistently. When switching to a non-default Python version, the corresponding compilation and development dependencies must be reinstalled manually.
>
> **torch CPU wheel selection**: Installs CPU torch from the official PyTorch index. `pip` auto-selects the correct ABI-tagged wheel (`cp310`/`cp311`/`cp312`/`cp313`/`cp314`) based on the active interpreter, so no manual ABI tag selection is needed:
>
> - Stable version (e.g. `2.13.0`): `pip${PY_VERSION} install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu`
> - Dev/nightly version (matching `*.dev*`, e.g. `2.13.0.dev20260610`): `pip${PY_VERSION} install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/nightly/cpu`
> - For other detailed compilation and development dependencies, see [requirements (taking version 2.13.0 as an example)](https://gitcode.com/Ascend/pytorch/blob/master/requirements_2.13.txt) in the corresponding repository directory.

---

## License/Disclaimer

See the [license information](https://gitcode.com/Ascend/pytorch/blob/master/LICENSE) for TorchNPU included in these images.

Like all container images, pre-installed software packages (Python, system libraries, etc.) may be subject to their own licenses.

The released Ascend software images are community versions; they are not intended for commercial use and serve solely as references for production practices.

Liability disclaimers are displayed in the image startup information and on the Ascend image platform.
