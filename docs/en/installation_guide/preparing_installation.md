# Preparing for Installation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T03:38:25.846Z pushedAt=2026-06-15T07:27:21.217Z -->

If you are using a pre-trained model for data processing and analysis, that is, performing only offline inference, skip this chapter.

If you are performing training or online inference, complete the following pre-installation preparations.

> [!NOTICE]
>
> It is recommended to use a non-root user for installation and running programs, and to properly manage permissions for the installation directory: set folder permissions to 750 and file permissions to 640. You can control the permissions of installed files by setting umask, for example, setting umask to 0027. For more security-related information, see the "File Permission Control" description for each component in the [Security Statement](../security_statement/security_statement.md).

- Install the matching versions of the NPU driver, firmware, and CANN software (Toolkit, ops, and NNAL) and configure CANN environment variables. For details, see the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/en/canncommercial/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) (commercial version) or [CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) (community version).

CANN provides a process-level environment variable setting script. This script must be called before using the NPU to execute service code in training or inference scenarios; otherwise, the service code will fail to run.

```bash
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

The above command uses the default installation path for the root user as an example. Replace it with the actual path of `set_env.sh`.

- When installing the torch_npu plugin from source in a container scenario, this involves obtaining community-provided base images and third-party Python libraries from external networks, as well as compiling and using source code. For network issues such as proxy configuration, refer to the [Docker official documentation](https://docs.docker.com/engine/cli/proxy/).
- Before installing the dependencies required for different types of operating systems, check whether the repository is available under the installation user. For example, to configure a Huawei mirror repository, refer to the configuration method for the corresponding mirror repository on the [Huawei Open Source Mirror Site](https://mirrors.huaweicloud.com/).
- Python 3.11 offers better scheduling (that is, dispatch) performance than Python 3.10. Python 3.11 or later is recommended.
- When installing the PyTorch framework and the torch_npu plugin from source code, you must install the following dependencies:

    ```bash
    pip3 install pyyaml
    pip3 install wheel
    pip3 install setuptools
    ```

    When you install them as a non-root user, add `--user` to the end of the command(for example, `pip3 install pyyaml ​​--user`).
