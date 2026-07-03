# Uninstallation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T03:38:38.197Z pushedAt=2026-06-15T07:27:21.220Z -->

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
> If you need to save the uninstall log, add the parameter `--log <path>` after the `pip3 uninstall` command, and implement proper permission control for the directory you specify as `<path>`.

## Uninstalling CANN Software

After installing CANN software offline in the `_xxx_.run` format, you can uninstall it by referring to the following method.

> [!NOTICE]
>
> - After ops is installed along with Toolkit, ops will be automatically uninstalled when Toolkit is uninstalled.
> - If you only uninstall CANN software packages (such as Toolkit), the uninstall order does not matter. However, if you also need to uninstall the driver and firmware, you must uninstall other software packages before uninstalling the driver and firmware.

Go to the path where the uninstall script is located and run the uninstall command.

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

After the uninstallation is complete, if the following information is displayed, the software has been successfully uninstalled:

```text
[INFO] xxx uninstall success
```

_xxx_ indicates the actual software package name to be uninstalled.

## Uninstalling Driver and Firmware

There is no specific order requirement for uninstalling the driver and firmware. The steps are as follows:

1. Use PuTTY to log in to the server's OS command line.
2. Run the following command to switch to the root user.

    ```bash
    su - root
    ```

3. Run the following command in any path to uninstall the software package.

    - Uninstall the firmware.

        ```bash
        <install_path>/firmware/script/uninstall.sh
        ```

        If the following key echo information is displayed, the firmware is uninstalled successfully.

        ```text
        Firmware package uninstalled successfully! 
        ```

    - Uninstall the driver.

        ```bash
        <install_path>/driver/script/uninstall.sh
        ```

        If the following key information is output, the driver is uninstalled successfully.

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
