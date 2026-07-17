# Method 2: Installation from Source Code

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T03:37:27.350Z pushedAt=2026-06-15T07:27:21.197Z -->

Source-code-based installation is suitable for custom development scenarios. For example, after developing custom operator adaptations, you can select the desired branch to compile the PyTorch framework and the torch_npu plugin.

Before executing the installation commands, please refer to [Pre-installation Preparation](preparing_installation.md) to complete environment variable configuration and other environment preparations.

## Installing the PyTorch Framework

For detailed steps, refer to the [PyTorch official website](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source).

## Installing the torch\_npu Plugin

The following example is based on PyTorch 2.7.1.

- **Method 1 (Recommended): Container Scenario**

    1. Download the torch\_npu source code.

        ```bash
        git clone https://gitcode.com/Ascend/pytorch.git -b v2.7.1-26.0.0 --depth 1
        ```

        Taking v2.7.1-26.0.0 as an example, download the corresponding Ascend Extension for PyTorch branch code. For other versions of Ascend Extension for PyTorch branch code, refer to the "[Version Mapping](../release_notes/release_notes.md)" section in the *Release Notes*.

    2. Build the image.

        ```bash
        cd pytorch/docker/builder/{arch} 
        docker build -t manylinux-builder:v1 .
        ```

        > [!NOTE]
        > - `{arch}` indicates the CPU architecture (x86 or ARM).
        > - Be careful not to omit the "." at the end of the command.

    3. Enter the Docker container and mount the torch\_npu source code into the container.

        ```bash
        docker run -it -v /{code_path}/pytorch:/home/pytorch manylinux-builder:v1 bash
        ```

        `{code_path}` indicates the path to the torch\_npu source code. Replace it based on the actual situation.

    4. Compile and generate the Whl installation package.

        ```bash
        cd /home/pytorch
        bash ci/build.sh --python=3.10
        ```

        To specify another Python version, use --python=3.9, --python=3.11, --python=3.12, or --python=3.13.

    5. Install the generated torch_npu plugin package in the runtime environment. If installing as a non-root user, append `--user` to the command.

        ```bash
        pip3 install --upgrade dist/torch_npu-2.7.1.post2-cp310-cp310-linux_aarch64.whl
        ```

        Change the torch_npu package name in the command based on your actual situation.

- **Method 2: Physical Machine and Virtual Machine Scenarios**

    1. Install system dependencies.

        Based on the operating system type, select the corresponding command to install the required dependencies.

        - openEuler, CentOS, Kylin, BCLinux, UOS V20, AntOS, AliOS, CTyunOS, CULinux, Tlinux, MTOS, vesselOS:

            1. Install dependencies (excluding gcc and cmake).

                ```bash
                yum install -y patch libjpeg-turbo-devel dos2unix openblas git
                ```

            2. Install GCC and cmake.

                Install the corresponding gcc and cmake versions based on your actual situation. For version information and installation instructions, see [Table 1](#gcc_cmake).

        - Debian, Ubuntu, veLinux:

            1. Install dependencies (excluding GCC and cmake).

                ```bash
                apt-get install -y patch build-essential libbz2-dev libreadline-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev liblzma-dev m4 dos2unix libopenblas-dev git
                ```

            2. Install gcc and cmake.

                Install the corresponding gcc and cmake versions based on your actual situation. For version information and installation instructions, see [Table 1](#gcc_cmake).

                **Table 1**  GCC and CMake version requirements<a id="gcc_cmake"></a>

                |PyTorch Version|System Architecture|GCC Version|CMake Version|
                |--|--|--|--|
                |2.7.1|X86_64|11.2.0|3.18.4|
                |2.7.1|AArch64|11.2.0|3.31.1|
                |2.8.0|X86_64|13.3.0|3.18.4|
                |2.8.0|AArch64|13.3.0|4.0.3|
                |2.9.0|X86_64|13.3.0|3.18.4|
                |2.9.0|AArch64|13.3.0|4.0.3|
                |2.10.0|X86_64|13.3.0|3.18.4|
                |2.10.0|AArch64|13.3.0|4.0.3|

                > [!NOTE]
                >
                > For installation instructions, see [Installing gcc 11.2.0](installing_gcc_11-2-0.md) and [Installing cmake 3.18.4](installing_cmake_3-18-4.md).

    2. Compile and generate the Whl installation package for the torch_npu plugin.
        1. Taking v2.7.1-26.0.0 as an example, download the corresponding Ascend Extension for PyTorch branch code and enter the plugin root directory.

            ```bash
            git clone -b v2.7.1-26.0.0 https://gitcode.com/Ascend/pytorch.git 
            cd pytorch
            ```

            Refer to the "[Version Mapping](../release_notes/release_notes.md)" section in the *Release Notes* to download branch code for other versions of Ascend Extension for PyTorch.

        2. Compile and generate the Whl installation package.

            ```bash
            bash ci/build.sh --python=3.10
            ```

            If you need to specify another Python version, use --python=3.9, --python=3.11, --python=3.12, or --python=3.13.

    3. Install the torch_npu plugin package generated in the pytorch/dist directory. If installing as a non-root user, add `--user` to the command.

        ```bash
        pip3 install --upgrade dist/torch_npu-2.7.1.post2-cp310-cp310-linux_aarch64.whl
        ```

        Change the torch_npu package name in the command based on your actual situation.

## Post-Installation Verification

Run the following command to check whether the PyTorch framework and torch_npu plugin have been successfully installed.

- Method 1

    ```Python
    python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
    ```

    Output similar to the following indicates successful installation.

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

    Output similar to the following indicates successful installation.

    ```text
    tensor([[-0.0515,  0.3664],
            [-0.1258, -0.5425]], device='npu:0')
    ```
