# Building from Source

Source-code installation applies to secondary development scenarios. For example, after developing custom operator adaptation, you can select the desired branch version and compile the PyTorch framework and the TorchNPU plugin yourself.

## Pre-installation Preparation

### Hardware Compatibility

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 inference products</term>: Not supported
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->
<!-- npu="310p" id7 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id7 -->
<!-- npu="310b" id8 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id8 -->

### Environment Preparation

> [!NOTICE]
>
> You are advised to use a non-root user to install and run programs, and to properly manage the permissions of the directories and files where the installer is located: set folder permissions to 750 and file permissions to 640. You can control the permissions of installed files by setting umask, for example, setting umask to 0027. For more security-related content, see the "File Permission Control" description for each component in *[Security Statement](../../reference/security_statement.md)*.

- Install the matching versions of the NPU driver and firmware and CANN software (Toolkit, ops, and NNAL), and configure CANN environment variables. For details, refer to *[CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)*.

        CANN software provides a process-level environment variable setting script. Before using the NPU to execute service code in training or inference scenarios, you must call this script. Otherwise, the service code cannot be executed.

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

        The preceding commands use the default path after installation by the root user as an example. Replace it with the actual path of set\_env.sh.

- Python 3.11 offers better scheduling (that is, dispatch) performance than Python 3.10. You are advised to use Python 3.11 or later.

## Installing the PyTorch Framework

For the specific steps, see the [PyTorch official website](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source).

## Installing the TorchNPU Plugin

To shorten the source build time, you can configure Ninja, Mold, and CCache by referring to [Build Acceleration](build_acceleration.md), or compile only the required targets.

In a container scenario, the source installation of the TorchNPU plugin involves obtaining community-provided base images, Python third-party libraries, and source code for compilation from external networks. For network issues such as proxy configuration, refer to the [Docker official documentation](https://docs.docker.com/engine/cli/proxy/).

Before installing the dependencies required by different types of operating systems, check whether the source is available under the installation user. For example, to configure the Huawei mirror source, refer to the configuration method for the corresponding mirror source on the [Huawei Open Source Mirror Site](https://mirrors.huaweicloud.com/).

The following steps use the installation of PyTorch 2.13.0 and Python 3.10.0 as an example.

- **Method 1 (Recommended): Container Scenario**
    
    1. Download the TorchNPU source code.

        ```bash
        git clone https://gitcode.com/Ascend/pytorch.git -b v2.13.0-26.2.0 --depth 1
        ```

    2. Build and start the container.

        There are two ways to obtain the development image: pull the pre-built image directly from the Ascend image repository, or build it yourself using the Dockerfile.

        - Pull the image directly

            We have provided a usable development image for you to compile and build TorchNPU. You can pull it directly from the Ascend image repository: [torch-npu-devel](https://www.hiascend.com/developer/ascendhub/detail/3b0ca76864884546acd07845f6153ee6)

            Taking Atlas A2 training products as an example, the command to pull the image is as follows:

            ```bash
            docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28
            ```

            After the image is pulled, use the following command to start and enter the Docker container, and mount the TorchNPU source code into the container:

            ```bash
            docker run -it -v /{code_path}/pytorch:/home/pytorch swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28 bash
            ```

        - Build the image yourself

            We also provide a usable Dockerfile that can automatically detect the architecture and pull the corresponding image. You can read the README file in the directory (pytorch/docker/devel) for more information and build your own development environment according to its guidance. Alternatively, you can directly use the devcontainer tool to build the development environment.

            ```bash
            cd pytorch/docker/devel
            export DOCKER_BUILDKIT=1
            docker build -t manylinux-builder:v1 .
            ```

            Or directly use the one-click container creation script builder.sh.

            ```bash
            cd pytorch/docker/devel
            export DOCKER_BUILDKIT=1
            bash builder.sh --cann
            ```

            > [!NOTE]
            >
            > - The Dockerfile automatically pulls the corresponding image based on the current architecture (ARM/X86).
            > - If you need to specify more specific build parameters, refer to the README in the directory (pytorch/docker/devel).
            > - Be careful not to omit the "." at the end of the command.

            After the image is built, use the following command to start and enter the Docker container, and mount the TorchNPU source code into the container:

            ```bash
            docker run -it -v /{code_path}/pytorch:/home/pytorch manylinux-builder:v1 bash
            ```

        _{code_path}_ indicates the path to the TorchNPU source code. Replace it based on the actual situation.

        > [!NOTE]
        >
        > - A directly started container can only be used to compile the TorchNPU plugin.
        > - To run TorchNPU in the NPU environment in the container, ensure that the driver already exists on the host (which can be confirmed using the `npu-smi` command), and mount the driver when starting the container. For specific operations, refer to the README in the directory (pytorch/docker/devel).

    3. Compile and generate the Whl installation package.

        Before compilation, pay attention to whether the torch version installed in the environment is consistent with the TorchNPU version to be compiled. To change the version, use the following command:

       ```bash
       pip3 install torch==2.13.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu # Install the CPU version of torch as needed
       ```

        After environment preparation is complete, use the following command to compile:

        ```bash
        cd /home/pytorch
        bash ci/build.sh --python=3.10
        ```

        To specify another Python version, use --python=3.10, --python=3.11, --python=3.12, --python=3.13, or --python=3.14.
        > [!NOTE]
        > 
        > The Release version is compiled by default. To compile the Debug version, set the environment variable `DEBUG=1` when executing the build command.
        
    4. Install the generated TorchNPU plugin package in the runtime environment. If you install as a non-root user, append `--user` to the command.

        ```bash
        pip3 install --upgrade dist/torch_npu-2.13.0rc1-cp310-cp310-linux_aarch64.whl
        ```

        Change the TorchNPU package name in the command based on the actual situation.

    5. Install the dependency file requirements_2.13.txt in the pytorch directory of the runtime environment.

        ```bash
        pip3 install -r requirements_2.13.txt
        ```

- **Method 2: Physical Machine and Virtual Machine Scenarios**

    1. Install the system dependencies.

        1. Based on the operating system type, select the corresponding command to install the required dependencies.

            - openEuler, CentOS, Kylin, BCLinux, UOS V20, AntOS, AliOS, CTyunOS, CULinux, Tlinux, MTOS, vesselOS:

                1. Install the dependencies (except gcc and cmake).

                    ```bash
                    yum install -y patch libjpeg-turbo-devel dos2unix openblas git
                    ```

                2. Install gcc and cmake.

                    Install the corresponding gcc and cmake versions based on the actual situation. For version information and installation guidance, see [Table 1](#gcc_cmake).

            - Debian, Ubuntu, veLinux:

                1. Install the dependencies (except gcc and cmake).

                    ```bash
                    apt-get install -y patch build-essential libbz2-dev libreadline-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev liblzma-dev m4 dos2unix libopenblas-dev git
                    ```

                2. Install gcc and cmake.

                    Install the corresponding gcc and cmake versions based on the actual situation. For version information and installation guidance, see [Table 1](#gcc_cmake).

                    **Table 1**  gcc and cmake version requirements<a id="gcc_cmake"></a>

                    |PyTorch Version|System Architecture|gcc Version|cmake Version|
                    |--|--|--|--|
                    |2.7.1|X86_64|11.2.0|3.18.4|
                    |2.7.1|AArch64|11.2.0|3.31.1|
                    |2.9.0|X86_64|13.3.0|3.18.4|
                    |2.9.0|AArch64|13.3.0|4.0.3|
                    |2.10.0|X86_64|13.3.0|3.18.4|
                    |2.10.0|AArch64|13.3.0|4.0.3|
                    |2.11.0|X86_64|13.3.0|3.18.4|
                    |2.11.0|AArch64|13.3.0|4.3.2|
                    |2.12.0|X86_64|13.3.0|3.18.4|
                    |2.12.0|AArch64|13.3.0|4.3.2|
                    |2.13.0|X86_64|13.3.0|3.18.4|
                    |2.13.0|AArch64|13.3.0|4.3.2|
                    |2.14.0|X86_64|13.3.0|3.18.4|
                    |2.14.0|AArch64|13.3.0|4.4.3|                            

                    > [!NOTE]
                    >
                    > For installation guidance, see [Installing gcc](https://ubuntu.com/developers/docs/howto/gcc-setup/) and [Installing cmake](https://cmake.org/download/).

    2. Compile and generate the Whl installation package of the TorchNPU plugin.
        1. Download the v2.13.0-26.2.0 branch code and enter the root directory of the TorchNPU plugin.

            ```bash
            git clone -b v2.13.0-26.2.0 https://gitcode.com/Ascend/pytorch.git
            cd pytorch
            ```

        2. Install the environment dependencies.

            ```bash
            pip3 install -r requirements.txt
            ```

            If you install as a non-root user, append `--user` to the command, for example: **pip3 install -r requirements.txt --user**.

        3. Compile and generate the Whl installation package.

            Before compilation, pay attention to whether the torch version installed in the environment is consistent with the TorchNPU version to be compiled. To change the version, use the following command:

            ```bash
            pip3 install torch==2.13.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu # Install the CPU version of torch as needed
            ```

            After environment preparation is complete, use the following command to compile:

            ```bash
            bash ci/build.sh --python=3.10
            ```

            To specify another Python version, use --python=3.10, --python=3.11, --python=3.12, --python=3.13, or --python=3.14.
            > [!NOTE]
            > 
            > The Release version is compiled by default. To compile the Debug version, set the environment variable `DEBUG=1` when executing the build command.

    3. Install the generated TorchNPU plugin package in the pytorch/dist directory. If you install as a non-root user, append `--user` to the command.

        ```bash
        pip3 install --upgrade dist/torch_npu-2.13.0rc1-cp310-cp310-linux_aarch64.whl
        ```

        Change the TorchNPU package name in the command based on the actual situation.

## Post-Installation Verification

Run the following commands to check whether the PyTorch framework and the TorchNPU plugin are installed successfully.

```Python
python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
```

Output similar to the following indicates a successful installation.

```text
tensor([[-0.6066,  6.3385,  0.0379,  3.3356],
        [ 2.9243,  3.3134, -1.5465,  0.1916],
        [-2.1807,  0.2008, -1.1431,  2.1523]], device='npu:0')
```

To view the versions of the installed Python, PyTorch, and TorchNPU installation packages in the current environment, see [Querying Versions](../references/check_installed_versions.md).
