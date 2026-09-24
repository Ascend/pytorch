# Docker安装

Docker镜像中已预装PyTorch框架、TorchNPU插件及配套的CANN软件，用户无需再手动安装上述软件，仅需安装NPU驱动及固件即可。适用于需要快速搭建开箱即用的运行环境、避免手动配置依赖或需隔离运行环境的场景。

## 启动容器

1. 拉取Docker镜像，具体操作请参考[快速安装](../quick_install.md)。
2. 执行以下命令，启动容器。

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

    **表 1**  启动容器参数说明

    |参数|说明|
    |--|--|
    |`-d`|后台运行容器|
    |`--rm`|容器退出后自动删除容器|
    |`--name torch-npu`|指定容器名称为`torch-npu`|
    |`--privileged`|特权模式，使容器拥有宿主机设备访问权限|
    |`-v /dev:/dev`|挂载宿主机设备目录，使容器可访问NPU设备|
    |`-v /usr/local/Ascend/driver:/usr/local/Ascend/driver`|挂载Ascend驱动目录|
    |`-v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons`|挂载Ascend附加组件目录|
    |`-v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi`|挂载NPU管理工具`npu-smi`|
    |`-v /var/log/npu:/usr/slog`|挂载NPU日志目录|
    |`-e PY_VERSION=3.11`|指定容器内Python版本，请根据镜像中的实际Python版本修改|
    |`-e LD_LIBRARY_PATH=...`|指定Ascend驱动库路径|
    |`quay.io/ascend/torch-npu:<镜像标签>`|指定使用的镜像，`<镜像标签>`需替换为实际标签，例如`2.10.0-310p-ubuntu22.04-py3.11`|
    |`tail -f /dev/null`|保持容器运行的前台占位命令|

3. 容器启动后，执行以下命令进入容器。

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
