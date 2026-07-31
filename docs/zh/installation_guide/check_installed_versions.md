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
    torch     2.12.0+cpu
    torch_npu      2.12.0
    ```

    > [!NOTE]
    >
    > 由于每个TorchNPU版本会配套多个PyTorch版本发布安装包，因此配套发布的安装包版本号和TorchNPU版本号采取不同命名规则。如果需要查询版本号对应关系，请单击[相关产品版本配套说明](https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/release_notes.md#%E7%9B%B8%E5%85%B3%E4%BA%A7%E5%93%81%E7%89%88%E6%9C%AC%E9%85%8D%E5%A5%97%E8%AF%B4%E6%98%8E)查看。
    