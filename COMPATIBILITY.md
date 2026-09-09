# 版本配套

TorchNPU需与特定版本的PyTorch、CANN、Python及固件与驱动配合使用。各版本间的匹配关系由昇腾社区统一维护，建议优先使用推荐版本组合以确保最佳兼容性和性能。固件与驱动的版本配套与所使用的昇腾硬件及CANN版本相关，具体选择请参考 [固件与驱动安装页面](https://www.hiascend.com/hardware/firmware-drivers/commercial)。

## 推荐版本组合

|    组件    |               推荐版本               |
|:--------:|:--------------------------------:|
| TorchNPU |              2.12.0（安装包版本）             |
| PyTorch  |              2.12.0              |
|   CANN   |              9.1.0               |
|  Python  | 3.10 / 3.11 / 3.12 / 3.13 / 3.14 |

## TorchNPU 版本配套表

下表中的TorchNPU版本与`torch_npu`安装包版本采用不同的命名规则：

- **TorchNPU版本**：标识产品的发布版本，采用独立于PyTorch的版本编号，如`26.1.0`、`26.1.1`。正式版本通常使用三段数字，历史预发布版本包含`RC`标识，如`6.0.RC3`。同一个TorchNPU产品版本可以配套多个PyTorch版本，因此会发布多个不同版本号的安装包。
- **TorchNPU安装包版本**：以配套的PyTorch版本号为基础，通常采用`{PyTorch版本}`或`{PyTorch版本}.postN`的形式。`.postN`用于区分同一PyTorch版本下安装包的后续发布，`N`为发布序号，不能直接换算为TorchNPU产品版本。历史预发布安装包还可能包含`rc`标识。使用`pip`安装或查询`torch_npu`时，应使用安装包版本号。

例如，TorchNPU `26.1.0`配套PyTorch `2.7.1`时，安装包版本为`2.7.1.post8`；配套PyTorch `2.12.0`时，安装包版本为`2.12.0`。TorchNPU产品版本、安装包版本及依赖版本的对应关系以下表为准。

以下为当前活跃版本的匹配关系：

> 26.1.1的安装包版本及CANN `9.1.X`配套范围依据[发行版说明](https://gitcode.com/Ascend/pytorch/releases)。

| TorchNPU版本 | TorchNPU安装包版本 | PyTorch版本 | CANN版本 |
|:---:|:---:|:---:|:---:|
| 26.1.1 | 2.12.0.post2 | 2.12.0 | CANN 9.1.X |
| 26.1.1 | 2.11.0.post2 | 2.11.0 | CANN 9.1.X |
| 26.1.1 | 2.10.0.post6 | 2.10.0 | CANN 9.1.X |
| 26.1.1 | 2.9.0.post8 | 2.9.0 | CANN 9.1.X |
| 26.1.1 | 2.7.1.post10 | 2.7.1 | CANN 9.1.X |
| 26.1.0 | 2.12.0 | 2.12.0 | CANN 9.1.0 |
| 26.1.0 | 2.11.0 | 2.11.0 | CANN 9.1.0 |
| 26.1.0 | 2.10.0.post4 | 2.10.0 | CANN 9.1.0 |
| 26.1.0 | 2.9.0.post6 | 2.9.0 | CANN 9.1.0 |
| 26.1.0 | 2.7.1.post8 | 2.7.1 | CANN 9.1.0 |
| 26.0.0 | 2.10.0 | 2.10.0 | CANN 9.0.0 |
| 26.0.0 | 2.9.0.post2 | 2.9.0 | CANN 9.0.0 |
| 26.0.0 | 2.8.0.post4 | 2.8.0 | CANN 9.0.0 |
| 26.0.0 | 2.7.1.post4 | 2.7.1 | CANN 9.0.0 |
| 7.3.0 | 2.9.0 | 2.9.0 | CANN 8.5.0 |
| 7.3.0 | 2.8.0.post2 | 2.8.0 | CANN 8.5.0 |
| 7.3.0 | 2.7.1.post2 | 2.7.1 | CANN 8.5.0 |
| 7.3.0 | 2.6.0.post5 | 2.6.0 | CANN 8.5.0 |
| 7.2.0 | 2.8.0 | 2.8.0 | CANN 8.3.RC1 |
| 7.2.0 | 2.7.1 | 2.7.1 | CANN 8.3.RC1 |
| 7.2.0 | 2.6.0.post3 | 2.6.0 | CANN 8.3.RC1 |
| 7.2.0 | 2.1.0.post17 | 2.1.0 | CANN 8.3.RC1 |
| 7.1.0 | 2.6.0 | 2.6.0 | CANN 8.2.RC1 |
| 7.1.0 | 2.5.1.post1 | 2.5.1 | CANN 8.2.RC1 |
| 7.1.0 | 2.1.0.post13 | 2.1.0 | CANN 8.2.RC1 |
| 7.0.0 | 2.5.1 | 2.5.1 | CANN 8.1.RC1 |
| 7.0.0 | 2.4.0.post4 | 2.4.0 | CANN 8.1.RC1 |
| 7.0.0 | 2.3.1.post6 | 2.3.1 | CANN 8.1.RC1 |
| 7.0.0 | 2.1.0.post12 | 2.1.0 | CANN 8.1.RC1 |
| 6.0.0 | 2.4.0.post2 | 2.4.0 | CANN 8.0.0 |
| 6.0.0 | 2.3.1.post4 | 2.3.1 | CANN 8.0.0 |
| 6.0.0 | 2.1.0.post10 | 2.1.0 | CANN 8.0.0 |

<details>
<summary>点击展开历史版本（含 EOL）</summary>

| TorchNPU版本 | TorchNPU安装包版本 | PyTorch版本 | CANN版本 |
|:---:|:---:|:---:|:---:|
| 6.0.RC3 | 2.4.0 | 2.4.0 | CANN 8.0.RC3 |
| 6.0.RC3 | 2.3.1.post2 | 2.3.1 | CANN 8.0.RC3 |
| 6.0.RC3 | 2.1.0.post8 | 2.1.0 | CANN 8.0.RC3 |
| 6.0.RC2 | 2.3.1 | 2.3.1 | CANN 8.0.RC2 |
| 6.0.RC2 | 2.2.0.post2 | 2.2.0 | CANN 8.0.RC2 |
| 6.0.RC2 | 2.1.0.post6 | 2.1.0 | CANN 8.0.RC2 |
| 6.0.RC2 | 1.11.0.post14 | 1.11.0 | CANN 8.0.RC2 |
| 6.0.RC1 | 2.2.0 | 2.2.0 | CANN 8.0.RC1 |
| 6.0.RC1 | 2.1.0.post4 | 2.1.0 | CANN 8.0.RC1 |
| 6.0.RC1 | 1.11.0.post11 | 1.11.0 | CANN 8.0.RC1 |
| 5.0.0 | 2.1.0 | 2.1.0 | CANN 7.0.0 |
| 5.0.0 | 2.0.1.post1 | 2.0.1 | CANN 7.0.0 |
| 5.0.0 | 1.11.0.post8 | 1.11.0 | CANN 7.0.0 |
| 5.0.RC3 | 2.1.0.rc1 | 2.1.0 | CANN 7.0.RC1 |
| 5.0.RC3 | 2.0.1 | 2.0.1 | CANN 7.0.RC1 |
| 5.0.RC3 | 1.11.0.post4 | 1.11.0 | CANN 7.0.RC1 |
| 5.0.RC2.2 | 1.11.0.post3 | 1.11.0 | CANN 6.3.RC3.1 |
| 5.0.RC2.1 | 1.11.0.post2 | 1.11.0 | CANN 6.3.RC3 |
| 5.0.RC2 | 2.0.1.rc1 | 2.0.1 | CANN 6.3.RC2 |
| 5.0.RC2 | 1.11.0.post1 | 1.11.0 | CANN 6.3.RC2 |
| 5.0.RC2 | 1.8.1.post2 | 1.8.1 | CANN 6.3.RC2 |
| 5.0.RC1 | 1.11.0 | 1.11.0 | CANN 6.3.RC1 |
| 5.0.RC1 | 1.8.1.post1 | 1.8.1 | CANN 6.3.RC1 |
| 3.0.0 | 1.5.0.post8 | 1.5.0 | CANN 6.0.1 |
| 3.0.0 | 1.8.1 | 1.8.1 | CANN 6.0.1 |
| 3.0.0 | 1.11.0.rc2（beta） | 1.11.0 | CANN 6.0.1 |
| 3.0.RC3 | 1.5.0.post7 | 1.5.0 | CANN 6.0.RC1 |
| 3.0.RC3 | 1.8.1.rc3 | 1.8.1 | CANN 6.0.RC1 |
| 3.0.RC3 | 1.11.0.rc1（beta） | 1.11.0 | CANN 6.0.RC1 |
| 3.0.RC2 | 1.5.0.post6 | 1.5.0 | CANN 5.1.RC2 |
| 3.0.RC2 | 1.8.1.rc2 | 1.8.1 | CANN 5.1.RC2 |
| 3.0.RC1 | 1.5.0.post5 | 1.5.0 | CANN 5.1.RC1 |
| 3.0.RC1 | 1.8.1.rc1 | 1.8.1 | CANN 5.1.RC1 |
| 2.0.4 | 1.5.0.post4 | 1.5.0 | CANN 5.0.4 |
| 2.0.3 | 1.5.0.post3 | 1.8.1 | CANN 5.0.3 |
| 2.0.2 | 1.5.0.post2 | 1.5.0 | CANN 5.0.2 |

</details>

## PyTorch与Python版本配套表

|   PyTorch版本   |                           Python版本                           |
|:--------------:|:-------------------------------------------------------------:|
| PyTorch 2.13.0 | Python3.10, Python3.11, Python 3.12, Python 3.13, Python 3.14 |
| PyTorch 2.12.0 | Python3.10, Python3.11, Python 3.12, Python 3.13, Python 3.14 |
| PyTorch 2.11.0 | Python3.10, Python3.11, Python 3.12, Python 3.13, Python 3.14 |
| PyTorch 2.10.0 |       Python3.10, Python3.11, Python 3.12, Python 3.13        |
| PyTorch 2.9.0  |       Python3.10, Python3.11, Python 3.12, Python 3.13        |
| PyTorch 2.8.0  | Python3.9, Python3.10, Python 3.11, Python 3.12, Python 3.13  |
| PyTorch 2.7.1  | Python3.9, Python3.10, Python 3.11, Python 3.12, Python 3.13  |
| PyTorch 2.6.0  |              Python3.9, Python3.10, Python 3.11               |
| PyTorch 2.5.1  |              Python3.9, Python3.10, Python 3.11               |
| PyTorch 2.4.0  |         Python3.8, Python3.9, Python3.10, Python 3.11         |
| PyTorch 2.3.1  |         Python3.8, Python3.9, Python3.10, Python 3.11         |
| PyTorch 2.2.0  |               Python3.8, Python3.9, Python3.10                |
| PyTorch 2.1.0  |         Python3.8, Python3.9, Python3.10, Python 3.11         |
| PyTorch 1.11.0 |     Python3.7(>=3.7.5), Python3.8, Python3.9, Python3.10      |

## 硬件支持

TorchNPU支持如下昇腾产品系列：

|     产品系列      |
|:-------------:|
| 昇腾 950DT 系列产品 |
|  昇腾 A3 系列产品   |
|  昇腾 A2 系列产品   |
|  昇腾 910 系列产品  |
| 昇腾 310P 系列产品  |
| 昇腾 310B 系列产品  |
|   昇腾310系列产品   |
