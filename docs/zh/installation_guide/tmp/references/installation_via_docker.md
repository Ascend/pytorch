# 方式三：Docker安装

通过Docker镜像快速拉取并运行已预集成PyTorch框架和TorchNPU插件的环境，无需用户手动安装依赖。

## 拉取并运行Docker镜像

|PyTorch版本<!-- class: installation_torch_npu -->|TorchNPU插件版本|Python版本|芯片型号|系统架构|操作系统|CANN版本|安装方式| 安装命令|
|--|--|--|--|--|--|--|--|--|
|2.10.0|2.10.0|Python 3.11|Ascend 310P|X86_64|ubuntu|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.10.0-310p-ubuntu22.04-py3.11</copy> |
|2.10.0|2.10.0|Python 3.11|Ascend 310P|AArch64|openeuler|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.10.0-310p-openeuler24.03-py3.11</copy> |
|2.10.0|2.10.0|Python 3.11|Ascend 910B|X86_64|ubuntu|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.10.0-910b-ubuntu22.04-py3.11</copy> |
|2.10.0|2.10.0|Python 3.11|Ascend 910B|AArch64|openeuler|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.10.0-910b-openeuler24.03-py3.11</copy> |
|2.10.0|2.10.0|Python 3.11|Ascend A3|X86_64|ubuntu|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.10.0-a3-ubuntu22.04-py3.11</copy> |
|2.10.0|2.10.0|Python 3.11|Ascend A3|AArch64|openeuler|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.10.0-a3-openeuler24.03-py3.11</copy> |
|2.9.0|2.9.0.post2|Python 3.11|Ascend 310P|X86_64|ubuntu|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.9.0.post2-310p-ubuntu22.04-py3.11</copy> |
|2.9.0|2.9.0.post2|Python 3.11|Ascend 310P|AArch64|openeuler|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.9.0.post2-310p-openeuler24.03-py3.11</copy> |
|2.9.0|2.9.0.post2|Python 3.11|Ascend 910B|X86_64|ubuntu|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.9.0.post2-910b-ubuntu22.04-py3.11</copy> |
|2.9.0|2.9.0.post2|Python 3.11|Ascend 910B|AArch64|openeuler|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.9.0.post2-910b-openeuler24.03-py3.11</copy> |
|2.9.0|2.9.0.post2|Python 3.11|Ascend A3|X86_64|ubuntu|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.9.0.post2-a3-ubuntu22.04-py3.11</copy> |
|2.9.0|2.9.0.post2|Python 3.11|Ascend A3|AArch64|openeuler|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.9.0.post2-a3-openeuler24.03-py3.11</copy> |
|2.8.0|2.8.0.post4|Python 3.11|Ascend 310P|X86_64|ubuntu|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.8.0.post4-310p-ubuntu22.04-py3.11</copy> |
|2.8.0|2.8.0.post4|Python 3.11|Ascend 310P|AArch64|openeuler|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.8.0.post4-310p-openeuler24.03-py3.11</copy> |
|2.8.0|2.8.0.post4|Python 3.11|Ascend 910B|X86_64|ubuntu|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.8.0.post4-910b-ubuntu22.04-py3.11</copy> |
|2.8.0|2.8.0.post4|Python 3.11|Ascend 910B|AArch64|openeuler|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.8.0.post4-910b-openeuler24.03-py3.11</copy> |
|2.8.0|2.8.0.post4|Python 3.11|Ascend A3|X86_64|ubuntu|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.8.0.post4-a3-ubuntu22.04-py3.11</copy> |
|2.8.0|2.8.0.post4|Python 3.11|Ascend A3|AArch64|openeuler|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.8.0.post4-a3-openeuler24.03-py3.11</copy> |
|2.7.1|2.7.1.post4|Python 3.11|Ascend 310P|X86_64|ubuntu|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.7.1.post4-310p-ubuntu22.04-py3.11</copy> |
|2.7.1|2.7.1.post4|Python 3.11|Ascend 310P|AArch64|openeuler|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.7.1.post4-310p-openeuler24.03-py3.11</copy> |
|2.7.1|2.7.1.post4|Python 3.11|Ascend 910B|X86_64|ubuntu|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11</copy> |
|2.7.1|2.7.1.post4|Python 3.11|Ascend 910B|AArch64|openeuler|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.7.1.post4-910b-openeuler24.03-py3.11</copy> |
|2.7.1|2.7.1.post4|Python 3.11|Ascend A3|X86_64|ubuntu|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.7.1.post4-a3-ubuntu22.04-py3.11</copy> |
|2.7.1|2.7.1.post4|Python 3.11|Ascend A3|AArch64|openeuler|9.0.0|Docker安装| <copy>docker pull quay.io/ascend/torch-npu:2.7.1.post4-a3-openeuler24.03-py3.11</copy> |

> [!NOTE]
>
> - 更多PyTorch版本请参考[Links for torch](https://download.pytorch.org/whl/torch/)。
> - 更多TorchNPU插件版本请参考[PyTorch Release](https://gitcode.com/Ascend/pytorch/releases)。
> - Triton-Ascend插件用于支持图模式Inductor后端，且仅支持PyTorch2.7.1和2.9.0版本。

## 运行Docker容器

1. 镜像拉取完成后，执行以下命令启动容器。

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
        quay.io/ascend/torch-npu:<镜像标签> \
        tail -f /dev/null
    ```

2. 容器启动后，执行以下命令进入容器。

    ```bash
    docker exec -it torch-npu bash
    ```

> [!NOTE]
>
> - `LD_LIBRARY_PATH` 通过 `-e` 参数指定 Ascend 驱动库路径。注意 `-e` 会覆盖容器镜像原有的 `LD_LIBRARY_PATH`，如需额外库路径，请在 `docker run` 命令的 `LD_LIBRARY_PATH` 值末尾追加，例如 `:/your/extra/path`。
> - `<镜像标签>` 请替换为上表中实际的镜像标签，例如 `2.10.0-310p-ubuntu22.04-py3.11`。
> - `PY_VERSION` 请根据镜像对应的 Python 版本修改。

## 安装后验证

执行以下命令可检查PyTorch框架和TorchNPU插件是否已成功安装。

```Python
python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
```

输出如下类似信息说明安装成功。

```text
tensor([[-0.6066,  6.3385,  0.0379,  3.3356],
        [ 2.9243,  3.3134, -1.5465,  0.1916],
        [-2.1807,  0.2008, -1.1431,  2.1523]], device='npu:0')
```

如需查看当前环境中已安装的Python、PyTorch和TorchNPU安装包版本，请参见[查询版本](../references/check_installed_versions.md)。
