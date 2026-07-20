# Uninstallation

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-08T10:23:08.991Z pushedAt=2026-07-08T10:47:16.877Z -->

## Uninstalling Ascend Extension for PyTorch

- To uninstall the PyTorch framework, run the following command:

    ```bash
    pip3 uninstall torch
    ```

- To uninstall the torch_npu plugin, run the following command:

    ```bash
    pip3 uninstall torch_npu
    ```

- To uninstall the APEX module, run the following command:

    ```bash
    pip3 uninstall apex
    ```

> [!NOTE]
>
> To save the uninstallation log, add the parameter `--log <path>` after the `pip3 uninstall` command, and ensure proper access control on the directory `<path>` you specify.

## Uninstalling CANN Software

After installing CANN software offline using the _xxx_.run format, you can uninstall it by referring to the following method.

> [!NOTICE]
>
> - After ops is installed along with Toolkit, ops will be automatically uninstalled when Toolkit is uninstalled.
> - If you only uninstall CANN software packages (such as Toolkit), the uninstallation order does not matter. However, if you also need to uninstall the driver and firmware, you must uninstall other software packages before uninstalling the driver and firmware.

Navigate to the path where the uninstallation script is located and execute the uninstallation command.

- Toolkit

    ```bash
    cd /<path>/cann-<version>/{arch}-linux/script
    ./uninstall.sh
    ```

- Uninstall ops separately

    ```bash
    cd /<path>/cann-<version>/{arch}-linux/script
    ./ops_uninstall.sh
    ```

- NNAL

    ```bash
    cd /<path>/nnal/
    ./nnal_uninstall.sh
    ```

Where _<path\>_ is the installation path of the software package, _<version\>_ is the software package version, and \{arch\}-linux is the CPU architecture. Replace them based on the actual situation.

After the uninstallation is complete, if the following information is displayed, the software has been uninstalled successfully:

```text
[INFO] xxx uninstall success
```

_xxx_ indicates the actual software package name to be uninstalled.

## Uninstalling the Driver and Firmware

There is no required order for uninstalling the driver and firmware. The procedure is as follows:

1. Log in to the server's OS command line using PuTTY.
2. Run the following command to switch to the root user.

    ```bash
    su - root
    ```

3. Run the following command in any path to uninstall the package.

    - Uninstall the firmware

        ```bash
        <install_path>/firmware/script/uninstall.sh
        ```

        If the following key information is displayed, the firmware has been uninstalled successfully.

        ```text
        Firmware package uninstalled successfully! 
        ```

    - Uninstall the driver

        ```bash
        <install_path>/driver/script/uninstall.sh
        ```

        If the following key information is displayed, the driver has been uninstalled successfully.

         ```text
        Driver package uninstalled successfully!
        ```

    > [!NOTE]
    >
    > _<install\_path\>_ indicates the software package installation path. You can run the **cat /etc/ascend_install.info** command to query the installation path. Replace it based on the actual situation.

4. Decide whether to restart the server based on the system prompt. If a system restart is required, run the following command; otherwise, skip this step.

    ```bash
    reboot
    ```
