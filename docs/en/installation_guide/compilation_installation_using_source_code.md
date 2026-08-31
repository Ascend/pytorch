# Method 2: Source-Code Installation

Source-code installation applies to secondary development scenarios. For example, after developing custom operator adaptation, you can select the desired branch version and compile the PyTorch framework and the TorchNPU plugin yourself.

Before running the installation commands, see [Pre-installation Preparation](preparing_installation.md) to complete the environment variable configuration and other environment preparations.

## Installing the PyTorch Framework

For the specific steps, see the [PyTorch official website](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source).

## Installing the TorchNPU Plugin

The following steps use PyTorch 2.7.1 as an example.

- **Method 1 (Recommended): Container Scenario**

    1. Download the TorchNPU source code.

        ```bash
        git clone https://gitcode.com/Ascend/pytorch.git -b v2.7.1-26.1.0 --depth 1
        ```

        Taking v2.7.1-26.1.0 as an example, download the branch code of the corresponding TorchNPU version. To download the branch code of other TorchNPU versions, see the "[Version Mapping](../release_notes/release_notes.md)" section in the *Release Notes*.

    2. Build the image.

        ```bash
        cd pytorch/docker/builder/{arch}
        docker build -t manylinux-builder:v1 .
        ```

        > [!NOTE]
        > - `{arch}` indicates the CPU architecture (x86 or ARM).
        > - Be careful not to omit the "." at the end of the command.

    3. Enter the Docker container and mount the TorchNPU source code into the container.

        ```bash
        docker run -it -v /{code_path}/pytorch:/home/pytorch manylinux-builder:v1 bash
        ```

        `{code_path}` indicates the path to the TorchNPU source code. Replace it based on the actual situation.

    4. Compile and generate the Whl installation package.

        ```bash
        cd /home/pytorch
        bash ci/build.sh --python=3.10
        ```

        To specify another Python version, use `--python=3.9`, `--python=3.11`, `--python=3.12`, or `--python=3.13`.

    5. Install the generated TorchNPU plugin package in the runtime environment. If you install as a non-root user, append `--user` to the command.

        ```bash
        pip3 install --upgrade dist/torch_npu-2.7.1.post2-cp310-cp310-linux_aarch64.whl
        ```

        Change the TorchNPU package name in the command based on the actual situation.

    6. Install the dependency file `requirements.txt` in the pytorch directory of the runtime environment.

        ```bash
        pip3 install -r requirements.txt
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
                    |2.11.0|X86_64|13.3.1|3.18.4|
                    |2.11.0|AArch64|13.3.1|4.3.2|
                    |2.12.0|X86_64|13.3.1|3.18.4|
                    |2.12.0|AArch64|13.3.1|4.3.2|

                    > [!NOTE]
                    >
                    > For installation guidance, see [Installing gcc 11.2.0](installing_gcc_11-2-0.md) and [Installing cmake 3.18.4](installing_cmake_3-18-4.md).

        2. Install the environment dependencies.
    
            ```bash
            pip install pyyaml
            pip install setuptools
            pip install auditwheel
            ```

            If you install as a non-root user, append `--user` to the command. For example: **pip3 install pyyaml --user**.

    2. Compile and generate the Whl installation package of the TorchNPU plugin.
        1. Taking v2.7.1-26.1.0 as an example, download the branch code of the corresponding TorchNPU version and enter the plugin root directory.

            ```bash
            git clone -b v2.7.1-26.1.0 https://gitcode.com/Ascend/pytorch.git
            cd pytorch
            ```

            Refer to the "[Version Mapping](../release_notes/release_notes.md)" section in the *Release Notes* to download the branch code of other TorchNPU versions.

        2. Compile and generate the Whl installation package.

            ```bash
            bash ci/build.sh --python=3.10
            ```

            To specify another Python version, use `--python=3.9`, `--python=3.11`, `--python=3.12`, or `--python=3.13`.

    3. Install the TorchNPU plugin package generated in the pytorch/dist directory. If you install as a non-root user, append `--user` to the command.

        ```bash
        pip3 install --upgrade dist/torch_npu-2.7.1.post2-cp310-cp310-linux_aarch64.whl
        ```

        Change the TorchNPU package name in the command based on the actual situation.

    4. Install the dependency file `requirements.txt` in the pytorch directory.

        ```bash
        pip3 install -r requirements.txt
        ```

## Querying Versions

Run the following commands to check the versions of the installed Python, PyTorch framework, and TorchNPU installation packages.

- Check the installed Python version.

    ```bash
    python --version
    ```

    The output shows the Python version as follows.

    ```text
    Python 3.13.0
    ```

- Check the versions of the installed PyTorch framework and TorchNPU installation packages.

    ```bash
    pip list | grep torch
    ```

    The output shows the versions of the PyTorch framework and TorchNPU installation packages as follows.

    ```text
    torch     2.12.0+cpu
    torch_npu      2.12.0
    ```

    > [!NOTE]
    >
    > Because each TorchNPU version publishes installation packages for multiple PyTorch versions, the version numbers of the released installation packages and the TorchNPU version numbers follow different naming rules. To query the mapping between version numbers, click [Version Mapping](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md).

## Post-Installation Verification

Run the following commands to check whether the PyTorch framework and the TorchNPU plugin are installed successfully.

- Method 1

    ```Python
    python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
    ```

    Output similar to the following indicates a successful installation.

    ```text
    tensor([[-0.6066,  6.3385,  0.0379,  3.3356],
            [ 2.9243,  3.3134, -1.5465,  0.1916],
            [-2.1807,  0.2008, -1.1431,  2.1523]], device='npu:0')
    ```

- Method 2

    ```Python
    import torch
    import torch_npu
    
    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    z = x.mm(y)
    
    print(z)
    ```

    Output similar to the following indicates a successful installation.

    ```text
    tensor([[-0.0515,  0.3664],
            [-0.1258, -0.5425]], device='npu:0')
    ```
