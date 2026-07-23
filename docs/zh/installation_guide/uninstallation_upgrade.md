# 卸载与升级

## 卸载TorchNPU

- 卸载PyTorch框架需执行如下命令：

    ```bash
    pip3 uninstall torch
    ```

- 卸载TorchNPU插件需执行如下命令：

    ```bash
    pip3 uninstall torch_npu
    ```

> [!NOTE]
>
> 如需要保存卸载日志，可在pip3 uninstall命令后面加上参数--log <path\>，并对您指定的目录<path\>做好权限管控。

## 卸载CANN软件与驱动固件

当需要切换CANN版本、重新安装Ascend运行环境，或因环境配置异常需要清理现有组件时，可前往[CANN官网](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netconda)，选择对应版本，参考官方指导卸载CANN软件、驱动或固件。

## 升级TorchNPU

1. 需要先手动卸载旧版本，具体可参见上述卸载操作。
2. 安装新版本，具体可参见[快速安装](quick_install.md)进行操作。

> [!NOTE]
>
> - 在驱动、固件和CANN软件配套场景下，可单独升级PyTorch框架与TorchNPU插件。
> - 如果也要升级固件、驱动和CANN软件，请按照“固件-\>驱动-\>CANN软件”的顺序先进行升级，再升级PyTorch框架与TorchNPU插件。
