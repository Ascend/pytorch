# Pre-installation Preparation

If you are using a pre-trained model for data processing and analysis, that is, performing only offline inference, skip this chapter.

If you are performing training or online inference, complete the following pre-installation preparations.

> [!NOTICE]
>
> You are advised to use a non-root user to install and run programs, and to properly manage the permissions of the installation directory and files: set folder permissions to 750 and file permissions to 640. You can control the permissions of installed files by setting umask, for example, setting umask to 0027. For more security-related information, see the "File Permission Control" description for each component in the [Security Statement](../security_statement/security_statement.md).

- Install the matching versions of the NPU driver and firmware and CANN software (Toolkit, ops, and NNAL), and configure CANN environment variables. For details, refer to the CANN Software Installation.
<!-- [CANN Software Installation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) -->

    CANN software provides a process-level environment variable setting script. Before using the NPU to execute service code in training or inference scenarios, you must call this script. Otherwise, the service code cannot be executed.

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

    The preceding command uses the default installation path for the root user as an example. Replace it with the actual path of `set_env.sh`.

- In a container scenario, the source installation of the TorchNPU plugin involves obtaining community-provided base images, Python third-party libraries, and source code for compilation and use from external networks. For network issues such as proxy configuration, refer to the [Docker official documentation](https://docs.docker.com/engine/cli/proxy/).
- Before installing the dependencies required by different types of operating systems, check whether the source is available under the installation user. For example, to configure the Huawei mirror source, refer to the configuration method for the corresponding mirror source on the [Huawei Open Source Mirror Site](https://mirrors.huaweicloud.com/).
- Python 3.11 offers better scheduling (that is, dispatch) performance than Python 3.10. You are advised to use Python 3.11 or later.
