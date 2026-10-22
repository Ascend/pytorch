# Uninstallation and Upgrade

## Uninstalling TorchNPU

- To uninstall the PyTorch framework, run the following command:

    ```bash
    pip3 uninstall torch
    ```

- To uninstall the TorchNPU plugin, run the following command:

    ```bash
    pip3 uninstall torch_npu
    ```

> [!NOTE]
>
> To save the uninstallation log, add the parameter --log <path\> after the pip3 uninstall command, and control the permissions of the directory <path\>.

## Uninstalling the CANN Software, Driver, and Firmware

When you need to switch the CANN version, reinstall the Ascend runtime environment, or clean up existing components due to abnormal environment configurations, go to the [CANN official website](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netconda), select the corresponding version, and refer to the official guidance to uninstall the CANN software, driver, or firmware.

## Upgrading TorchNPU

1. Manually uninstall the old version first. For details, see the uninstallation operations above.
2. Install the new version. For details, see [Quick Installation](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=182&ids=89dda9ba9de741349efa03687a487678%2C202%2C106%2C1%2C6%2C177%2C).

> [!NOTE]
>
> - In scenarios where the driver, firmware, and CANN software form a matching set, the PyTorch framework and TorchNPU plugin can be upgraded separately.
> - To upgrade the firmware, driver, and CANN software as well, upgrade them in the order "firmware -\> driver -\> CANN software" first, and then upgrade the PyTorch framework and TorchNPU plugin.
