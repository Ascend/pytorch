# 查询版本

执行以下命令可检查安装的Python、PyTorch框架和TorchNPU安装包版本。

- 查看已安装的Python版本。

    ```bash
    python --version
    ```

    输出如下Python版本。

    ```text
    Python 3.13.0
    ```

- 查看已安装的PyTorch框架和TorchNPU安装包版本。

    ```bash
    pip list | grep torch
    ```

    输出如下PyTorch框架和TorchNPU安装包版本。

    ```text
    torch     2.13.0+cpu
    torch_npu      2.13.0rc1
    ```

    > [!NOTE]
    >
    > 由于每个TorchNPU版本会配套多个PyTorch版本发布安装包，因此配套发布的安装包版本号和TorchNPU版本号采取不同命名规则。如果需要查询版本号对应关系，请单击[相关产品版本配套说明](../../../../../COMPATIBILITY.md)查看。
    