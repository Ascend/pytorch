# 方式三：二进制软件包安装（abi1版本）

本章节介绍如何获取并安装abi1版本的PyTorch框架和torch\_npu插件。

执行安装命令前，请参见[安装前准备](preparing_installation.md)完成环境变量配置及其他环境准备。

> [!NOTE]   
> 以下操作仅适用于PyTorch 2.6.0。

## 安装PyTorch框架

使用以下命令可以直接从PyTorch官方获取abi1版本的安装包。

```
# 下载软件包
wget https://download.pytorch.org/whl/cpu-cxx11-abi/torch-<version>%2Bcpu.cxx11.abi-cp3x-cp3x-linux_x86_64.whl

# 安装命令
pip3 install torch-<version>+cpu.cxx11.abi-cp3x-cp3x-linux_x86_64.whl
```

示例命令如下：

```
wget https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.6.0%2Bcpu.cxx11.abi-cp39-cp39-linux_x86_64.whl
pip3 install torch-2.6.0+cpu.cxx11.abi-cp39-cp39-linux_x86_64.whl
```

> [!NOTE]   
> -   **x**表示9、10、11或12，即当前支持Python 3.9、Python 3.10、Python 3.11和Python 3.12。
> -   _<version\>_ 表示PyTorch框架版本，当前指2.6.0。
> -   当前暂不支持AArch64架构的abi1版本安装。

## 安装torch\_npu插件

如下操作以PyTorch 2.6.0版本为例，介绍如何获取abi1版本的安装包并安装torch\_npu插件。

1.  下载abi1版本安装包。

    ```
    wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post4_cxx11.abi_x86_64.zip
    ```

    以v2.6.0-7.3.0为例，下载对应的Ascend Extension for PyTorch安装包，其他分支请参见《版本说明》中的“[相关产品版本配套说明](../release_notes/related_product_version_compatibility_notes.md)”章节。

2.  解压缩安装包。

    ```
    unzip -o torch_npu-2.6.0.post4_cxx11.abi_x86_64.zip
    ```

3.  安装abi1版本torch\_npu插件。

    ```
    # 选择对应Python版本的安装包，如Python 3.9
    pip3 install torch_npu-2.6.0.post4+cxx11.abi-cp39-cp39-manylinux_2_28_x86_64.whl
    ```

> [!NOTE]  
> -   PyTorch框架版本为2.6.0，Python版本支持3.9\~3.12。
> -   当前暂不支持AArch64架构的abi1版本安装。

## 安装后验证

执行以下命令可检查PyTorch框架和torch\_npu插件是否已成功安装。

-   方法一

    ```Python
    python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
    ```

    输出如下类似信息说明安装成功。

    ```ColdFusion
    tensor([[-0.6066,  6.3385,  0.0379,  3.3356],
            [ 2.9243,  3.3134, -1.5465,  0.1916],
            [-2.1807,  0.2008, -1.1431,  2.1523]], device='npu:0')
    ```

-   方法二

    ```Python
    import torch
    import torch_npu
    
    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    z = x.mm(y)
    
    print(z)
    ```

    输出如下类似信息说明安装成功。

    ```ColdFusion
    tensor([[-0.0515,  0.3664],
            [-0.1258, -0.5425]], device='npu:0')
    ```

