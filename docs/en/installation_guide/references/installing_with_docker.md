# Installing with Docker

The Docker image comes with the PyTorch framework, the TorchNPU plugin, and the matching CANN software pre-installed. You do not need to manually install the preceding software; you only need to install the NPU driver and firmware. It is suitable for scenarios where you need to quickly build an out-of-the-box runtime environment, avoid manual dependency configuration, or isolate the runtime environment.

## Starting the Container

1. Pull the Docker image. For specific operations, refer to [Quick Installation](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=180&ids=89dda9ba9de741349efa03687a487678%2C204%2C200%2C1%2C6%2C177%2C).
2. Run the following command to start the container.

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

    **Table 1**  Container startup parameter description

    |Parameter|Description|
    |--|--|
    |`-d`|Runs the container in the background|
    |`--rm`|Automatically deletes the container after it exits|
    |`--name torch-npu`|Specifies the container name as `torch-npu`|
    |`--privileged`|Privileged mode, giving the container access to host devices|
    |`-v /dev:/dev`|Mounts the host device directory so that the container can access NPU devices|
    |`-v /usr/local/Ascend/driver:/usr/local/Ascend/driver`|Mounts the Ascend driver directory|
    |`-v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons`|Mounts the Ascend add-on components directory|
    |`-v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi`|Mounts the NPU management tool `npu-smi`|
    |`-v /var/log/npu:/usr/slog`|Mounts the NPU log directory|
    |`-e PY_VERSION=3.11`|Specifies the Python version in the container. Modify it based on the actual Python version in the image|
    |`-e LD_LIBRARY_PATH=...`|Specifies the Ascend driver library path|
    |`quay.io/ascend/torch-npu:<image tag>`|Specifies the image to use. Replace `<image tag>` with the actual tag, for example, `2.10.0-310p-ubuntu22.04-py3.11`|
    |`tail -f /dev/null`|Foreground placeholder command that keeps the container running|

3. After the container is started, run the following command to enter the container.

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
