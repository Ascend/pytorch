# 源码编译

源码安装适用于二次开发场景，如自定义算子适配开发后，用户可以选择需要的分支版本自行编译PyTorch框架和TorchNPU插件。

执行安装命令前，请参见[快速安装](quick_install.md)中的安装前准备章节完成环境变量配置及其他环境准备。

## 安装PyTorch框架

具体步骤请参见[PyTorch官网](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source)。

## 安装TorchNPU插件

容器场景下源码安装TorchNPU插件，涉及从外部网络获取社区提供基础镜像、Python第三方库以及编译使用源码，代理配置等相关网络问题请参考[Docker官方文档](https://docs.docker.com/engine/cli/proxy/)。

在安装不同类型操作系统所需依赖前，请在安装用户下检查源是否可用。以配置华为镜像源为例，可参考[华为开源镜像站](https://mirrors.huaweicloud.com/)中镜像源对应的配置方法操作。

以下操作步骤以安装PyTorch 2.7.1版本为例。

- **方式一（推荐）：容器场景**
    
    1. 下载TorchNPU源码。

        ```bash
        git clone https://gitcode.com/Ascend/pytorch.git -b v2.7.1-26.0.0 --depth 1
        ```

        以v2.7.1-26.0.0为例，下载对应的TorchNPU分支代码。请参见《版本说明》中的“[相关产品版本配套说明](../release_notes.md#相关产品版本配套说明)”章节下载TorchNPU其他版本的分支代码。

    2. 构建镜像。

        ```bash
        cd pytorch/docker/devel
        export DOCKER_BUILDKIT=1
        docker build -t manylinux-builder:v1 .
        ```

        > [!NOTE]
        > - Dockerfile会自动根据当前架构（ARM/X86）拉取对应镜像。
        > - 如果需要指定更具体的构建参数，可参考该目录（pytorch/docker/devel）下README。
        > - 注意不要遗漏命令结尾的“.”。

    3. 进入Docker容器，并将TorchNPU源代码挂载至容器内。

        ```bash
        docker run -it -v /{code_path}/pytorch:/home/pytorch manylinux-builder:v1 bash
        ```

        _{code_path}_ 表示TorchNPU源代码路径，请根据实际情况进行替换。

    4. 编译生成Whl安装包。

        ```bash
        cd /home/pytorch
        bash ci/build.sh --python=3.10
        ```

        如需指定其他Python版本，请使用--python=3.9、--python=3.11、--python=3.12或--python=3.13。
        > [!NOTE]
        > 
        > 默认编译Release版本。如需编译Debug版本，请在执行构建命令时设置环境变量`DEBUG=1`。
        
    5. 在运行环境中安装生成的TorchNPU插件包，如果使用非root用户进行安装，需要在命令后加`--user`。

        ```bash
        pip3 install --upgrade dist/torch_npu-2.7.1.post2-cp310-cp310-linux_aarch64.whl
        ```

        请用户根据实际情况更改命令中的TorchNPU包名。

    6. 在运行环境中安装pytorch目录下的依赖文件requirements.txt。

        ```bash
        pip3 install -r requirements.txt
        ```

- **方式二：物理机及虚拟机场景**

    1. 安装系统依赖

        1. 根据不同类型的操作系统，选择对应的命令安装所需依赖。

            - openEuler、CentOS、Kylin、BCLinux、UOS V20、AntOS、AliOS、CTyunOS、CULinux、Tlinux、MTOS、vesselOS：

                1. 安装依赖（除gcc和cmake以外）。

                    ```bash
                    yum install -y patch libjpeg-turbo-devel dos2unix openblas git
                    ```

                2. 安装gcc和cmake。

                    根据实际情况，安装对应gcc和cmake版本，版本信息及安装指导请参见[表1](#gcc_cmake)。

            - Debian、Ubuntu、veLinux：

                1. 安装依赖（除gcc和cmake以外）。

                    ```bash
                    apt-get install -y patch build-essential libbz2-dev libreadline-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev liblzma-dev m4 dos2unix libopenblas-dev git
                    ```

                2. 安装gcc和cmake。

                    根据实际情况，安装对应gcc和cmake版本，版本信息及安装指导请参见[表1](#gcc_cmake)。

                    **表 1**  gcc和cmake版本要求<a id="gcc_cmake"></a>

                    |PyTorch版本|系统架构|gcc版本|cmake版本|
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
                    > 安装指导可参见[安装11.2.0版本gcc](installing_gcc_11-2-0.md)和[安装3.18.4版本cmake](installing_cmake_3-18-4.md)。

        2. 安装环境依赖。
    
            ```bash
            pip install pyyaml
            pip install setuptools
            pip install auditwheel
            ```

            如果使用非root用户安装，需要在命令后加`--user`，例如：**pip3 install pyyaml --user**。

    2. 编译生成TorchNPU插件的Whl安装包。
        1. 以v2.7.1-26.0.0为例，下载对应的TorchNPU分支代码并进入插件根目录。

            ```bash
            git clone -b v2.7.1-26.0.0 https://gitcode.com/Ascend/pytorch.git
            cd pytorch
            ```

            请参见《版本说明》中的“[相关产品版本配套说明](../release_notes.md#相关产品版本配套说明)”章节下载TorchNPU其他版本的分支代码。

        2. 编译生成Whl安装包。

            ```bash
            bash ci/build.sh --python=3.10
            ```

            如需指定其他Python版本，请使用--python=3.9、--python=3.11、--python=3.12或--python=3.13。
            > [!NOTE]
            > 
            > 默认编译Release版本。如需编译Debug版本，请在执行构建命令时设置环境变量`DEBUG=1`。

    3. 安装pytorch/dist目录下生成的插件TorchNPU包，如果使用非root用户安装，需要在命令后加`--user`。

        ```bash
        pip3 install --upgrade dist/torch_npu-2.7.1.post2-cp310-cp310-linux_aarch64.whl
        ```

        请用户根据实际情况更改命令中的TorchNPU包名。

    4. 安装pytorch目录下的依赖文件requirements.txt。

        ```bash
        pip3 install -r requirements.txt
        ```

## 安装后验证

如需查看当前环境中已安装的Python、PyTorch和TorchNPU版本，请参见[查询版本](check_installed_versions.md)。

执行以下命令可检查PyTorch框架和TorchNPU插件是否已成功安装。

- 方法一

    ```Python
    python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
    ```

    输出如下类似信息说明安装成功。

    ```text
    tensor([[-0.6066,  6.3385,  0.0379,  3.3356],
            [ 2.9243,  3.3134, -1.5465,  0.1916],
            [-2.1807,  0.2008, -1.1431,  2.1523]], device='npu:0')
    ```

- 方法二

    ```Python
    import torch
    import torch_npu
    
    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    z = x.mm(y)
    
    print(z)
    ```

    输出如下类似信息说明安装成功。

    ```text
    tensor([[-0.0515,  0.3664],
            [-0.1258, -0.5425]], device='npu:0')
    ```
