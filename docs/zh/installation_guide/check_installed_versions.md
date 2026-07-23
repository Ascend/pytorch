# 查询版本

执行以下命令可查看安装的Python、PyTorch框架和TorchNPU插件版本。

- 查看已安装的Python版本。

    ```bash
    python --version
    ```

    输出如下Python版本。

    ```text
    Python 3.13.0
    ```

- 查看已安装的PyTorch框架和TorchNPU插件版本。

    ```bash
    pip list | grep torch
    ```

    输出如下PyTorch框架和TorchNPU插件版本。

    ```text
    torch     2.10.0
    torch_npu      2.10.0 
    ```

    > [!NOTE]
    >
    > 如果需要查询TorchNPU安装包版本，请单击[相关产品版本配套说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/zh/release_notes/release_notes.md#%E7%9B%B8%E5%85%B3%E4%BA%A7%E5%93%81%E7%89%88%E6%9C%AC%E9%85%8D%E5%A5%97%E8%AF%B4%E6%98%8E)查看。
