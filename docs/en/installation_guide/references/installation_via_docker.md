# Method 3: Installing with Docker

Use a Docker image to quickly pull and run an environment with the pre-integrated PyTorch framework and TorchNPU plugin, without manually installing dependencies.

## Pulling and Running the Docker Image

|PyTorch Version<!-- class: installation_torch_npu -->|TorchNPU Plugin Version|Python Version|Chip Model|System Architecture|Operating System|CANN Version|Installation Method| Installation Command|
|--|--|--|--|--|--|--|--|--|
|2.10.0|2.10.0|Python 3.11|Ascend 310P|X86_64|ubuntu|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.10.0-310p-ubuntu22.04-py3.11</copy> |
|2.10.0|2.10.0|Python 3.11|Ascend 310P|AArch64|openeuler|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.10.0-310p-openeuler24.03-py3.11</copy> |
|2.10.0|2.10.0|Python 3.11|Ascend 910B|X86_64|ubuntu|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.10.0-910b-ubuntu22.04-py3.11</copy> |
|2.10.0|2.10.0|Python 3.11|Ascend 910B|AArch64|openeuler|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.10.0-910b-openeuler24.03-py3.11</copy> |
|2.10.0|2.10.0|Python 3.11|Ascend A3|X86_64|ubuntu|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.10.0-a3-ubuntu22.04-py3.11</copy> |
|2.10.0|2.10.0|Python 3.11|Ascend A3|AArch64|openeuler|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.10.0-a3-openeuler24.03-py3.11</copy> |
|2.9.0|2.9.0.post2|Python 3.11|Ascend 310P|X86_64|ubuntu|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.9.0.post2-310p-ubuntu22.04-py3.11</copy> |
|2.9.0|2.9.0.post2|Python 3.11|Ascend 310P|AArch64|openeuler|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.9.0.post2-310p-openeuler24.03-py3.11</copy> |
|2.9.0|2.9.0.post2|Python 3.11|Ascend 910B|X86_64|ubuntu|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.9.0.post2-910b-ubuntu22.04-py3.11</copy> |
|2.9.0|2.9.0.post2|Python 3.11|Ascend 910B|AArch64|openeuler|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.9.0.post2-910b-openeuler24.03-py3.11</copy> |
|2.9.0|2.9.0.post2|Python 3.11|Ascend A3|X86_64|ubuntu|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.9.0.post2-a3-ubuntu22.04-py3.11</copy> |
|2.9.0|2.9.0.post2|Python 3.11|Ascend A3|AArch64|openeuler|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.9.0.post2-a3-openeuler24.03-py3.11</copy> |
|2.8.0|2.8.0.post4|Python 3.11|Ascend 310P|X86_64|ubuntu|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.8.0.post4-310p-ubuntu22.04-py3.11</copy> |
|2.8.0|2.8.0.post4|Python 3.11|Ascend 310P|AArch64|openeuler|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.8.0.post4-310p-openeuler24.03-py3.11</copy> |
|2.8.0|2.8.0.post4|Python 3.11|Ascend 910B|X86_64|ubuntu|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.8.0.post4-910b-ubuntu22.04-py3.11</copy> |
|2.8.0|2.8.0.post4|Python 3.11|Ascend 910B|AArch64|openeuler|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.8.0.post4-910b-openeuler24.03-py3.11</copy> |
|2.8.0|2.8.0.post4|Python 3.11|Ascend A3|X86_64|ubuntu|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.8.0.post4-a3-ubuntu22.04-py3.11</copy> |
|2.8.0|2.8.0.post4|Python 3.11|Ascend A3|AArch64|openeuler|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.8.0.post4-a3-openeuler24.03-py3.11</copy> |
|2.7.1|2.7.1.post4|Python 3.11|Ascend 310P|X86_64|ubuntu|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.7.1.post4-310p-ubuntu22.04-py3.11</copy> |
|2.7.1|2.7.1.post4|Python 3.11|Ascend 310P|AArch64|openeuler|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.7.1.post4-310p-openeuler24.03-py3.11</copy> |
|2.7.1|2.7.1.post4|Python 3.11|Ascend 910B|X86_64|ubuntu|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11</copy> |
|2.7.1|2.7.1.post4|Python 3.11|Ascend 910B|AArch64|openeuler|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.7.1.post4-910b-openeuler24.03-py3.11</copy> |
|2.7.1|2.7.1.post4|Python 3.11|Ascend A3|X86_64|ubuntu|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.7.1.post4-a3-ubuntu22.04-py3.11</copy> |
|2.7.1|2.7.1.post4|Python 3.11|Ascend A3|AArch64|openeuler|9.0.0|Installing with Docker| <copy>docker pull quay.io/ascend/torch-npu:2.7.1.post4-a3-openeuler24.03-py3.11</copy> |

> [!NOTE]
>
> - For more PyTorch versions, see [Links for torch](https://download.pytorch.org/whl/torch/).
> - For more TorchNPU plugin versions, see [PyTorch Release](https://gitcode.com/Ascend/pytorch/releases).
> - The Triton-Ascend plugin is used to support the graph-mode Inductor backend and only supports PyTorch 2.7.1 and 2.9.0.

## Running the Docker Container

1. After the image is pulled, run the following command to start the container.

    ```bash
    docker run -d --rm \
        --name torch-npu \
        --privileged \
        -v /dev:/dev \
        -v $(pwd):/home/pytorch \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
        -v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi \
        -v /var/log/npu:/usr/slog \
        -e PY_VERSION=3.11 \
        -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/base:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver \
        quay.io/ascend/torch-npu:<image tag> \
        tail -f /dev/null
    ```

2. After the container is started, run the following command to enter the container.

    ```bash
    docker exec -it torch-npu bash
    ```

> [!NOTE]
>
> - `LD_LIBRARY_PATH` specifies the Ascend driver library path through the `-e` parameter. Note that `-e` overrides the original `LD_LIBRARY_PATH` of the container image. If you need an additional library path, append it to the end of the `LD_LIBRARY_PATH` value in the `docker run` command, for example, `:/your/extra/path`.
> - Replace `<image tag>` with the actual image tag in the preceding table, for example `2.10.0-310p-ubuntu22.04-py3.11`.
> - Modify `PY_VERSION` based on the Python version corresponding to the image.

## Post-Installation Verification

Run the following commands to check whether the PyTorch framework and the TorchNPU plugin are installed successfully.

```Python
python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
```

Output similar to the following indicates a successful installation.

```text
tensor([[-0.6066,  6.3385,  0.0379,  3.3356],
        [ 2.9243,  3.3134, -1.5465,  0.1916],
        [-2.1807,  0.2008, -1.1431,  2.1523]], device='npu:0')
```

To view the versions of the installed Python, PyTorch, and TorchNPU installation packages in the current environment, see [Querying Versions](../references/check_installed_versions.md).
