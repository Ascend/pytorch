# 源码编译

源码安装适用于二次开发场景，如自定义算子适配开发后，用户可以选择需要的分支版本自行编译PyTorch框架和TorchNPU插件。

## 安装前准备

### 硬件配套

**表 1**  产品硬件支持列表

|产品|是否支持|
|--|:-:|
|<term>Ascend 950DT</term>|√|
|<term>Atlas A3 训练系列产品</term>|√|
|<term>Atlas A3 推理系列产品</term>|x|
|<term>Atlas A2 训练系列产品</term>|√|
|<term>Atlas A2 推理系列产品</term>|x|
|<term>Atlas 训练系列产品</term>|√|
|<term>Atlas 推理系列产品</term>|x|
|<term>Atlas 200I/500 A2 推理产品</term>|x|

> [!NOTE]
>
> 本节表格中“√”代表支持，“x”代表不支持。

### 环境准备

> [!NOTICE]
>
> - 安装运行程序建议使用非root用户，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。更多安全相关内容请参见《[安全声明](../../../reference/security_statement.md)》中各组件关于“文件权限控制”的说明。

- 安装配套版本的NPU驱动固件、CANN软件（Toolkit、ops和NNAL）并配置CANN环境变量，具体请参考《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)》。

    CANN软件提供进程级环境变量设置脚本，训练或推理场景下使用NPU执行业务代码前需要调用该脚本，否则业务代码将无法执行。

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

    以上命令以root用户安装后的默认路径为例，请用户根据set\_env.sh的实际路径进行替换。

Python3.11的调度（即下发）性能优于Python3.10，建议用Python3.11及以上。

## 安装PyTorch框架

具体步骤请参见[PyTorch官网](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source)。

## 安装TorchNPU插件

容器场景下源码安装TorchNPU插件，涉及从外部网络获取社区提供基础镜像、Python第三方库以及编译使用源码，代理配置等相关网络问题请参考[Docker官方文档](https://docs.docker.com/engine/cli/proxy/)。

在安装不同类型操作系统所需依赖前，请在安装用户下检查源是否可用。以配置华为镜像源为例，可参考[华为开源镜像站](https://mirrors.huaweicloud.com/)中镜像源对应的配置方法操作。

以下操作步骤以安装PyTorch 2.13.0版本、Python 2.10.0版本为例。

- **方式一（推荐）：容器场景**
    
    1. 下载TorchNPU源码。

        ```bash
        git clone https://gitcode.com/Ascend/pytorch.git -b master --depth 1
        ```

    2. 构建镜像。

        我们已提供了可用的Dockerfile，可以自动检测架构来拉取镜像。你可以阅览该目录（pytorch/docker/devel）下的README文件，获取更多信息，并根据其指导构建自己的开发环境。或者您可以直接通过devcontainer工具来构建开发环境。

        ```bash
        cd pytorch/docker/devel
        export DOCKER_BUILDKIT=1
        docker build -t manylinux-builder:v1 .
        ```

        > [!NOTE]
        >
        > - Dockerfile会自动根据当前架构（ARM/X86）拉取对应镜像。
        > - 如果需要指定更具体的构建参数，可参考该目录（pytorch/docker/devel）下README。
        > - 注意不要遗漏命令结尾的“.”。

    3. 启动并进入Docker容器，并将TorchNPU源代码挂载至容器内。

        ```bash
        docker run -it -v /{code_path}/pytorch:/home/pytorch manylinux-builder:v1 bash
        ```

        _{code_path}_ 表示TorchNPU源代码路径，请根据实际情况进行替换。
        > [!NOTE]
        >
        > - 如果您已通过 `npu-smi` 确保宿主机存在驱动（driver），也可以在启动容器时将其挂载。可参考该目录（pytorch/docker/devel）下README。

    4. 编译生成Whl安装包。

        ```bash
        cd /home/pytorch
        bash ci/build.sh --python=3.10
        ```

        如需指定其他Python版本，请使用--python=3.10、--python=3.11、--python=3.12、--python=3.13或--python=3.14。
        > [!NOTE]
        > 
        > 默认编译Release版本。如需编译Debug版本，请在执行构建命令时设置环境变量`DEBUG=1`。
        
    5. 在运行环境中安装生成的TorchNPU插件包，如果使用非root用户进行安装，需要在命令后加`--user`。

        ```bash
        pip3 install --upgrade dist/torch_npu-2.13.0rc1-cp310-cp310-linux_aarch64.whl
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
                    |2.13.0|X86_64|13.3.1|3.18.4|
                    |2.13.0|AArch64|13.3.1|4.3.2|

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
        1. 下载master分支代码并进入TorchNPU插件根目录。

            ```bash
            git clone -b master https://gitcode.com/Ascend/pytorch.git
            cd pytorch
            ```

        2. 编译生成Whl安装包。

            ```bash
            bash ci/build.sh --python=3.10
            ```

            如需指定其他Python版本，请使用--python=3.10、--python=3.11、--python=3.12、--python=3.13或--python=3.14。
            > [!NOTE]
            > 
            > 默认编译Release版本。如需编译Debug版本，请在执行构建命令时设置环境变量`DEBUG=1`。

    3. 安装pytorch/dist目录下生成的插件TorchNPU包，如果使用非root用户安装，需要在命令后加`--user`。

        ```bash
        pip3 install --upgrade dist/torch_npu-2.13.0rc1-cp310-cp310-linux_aarch64.whl
        ```

        请用户根据实际情况更改命令中的TorchNPU包名。

    4. 安装pytorch目录下的依赖文件requirements.txt。

        ```bash
        pip3 install -r requirements.txt
        ```

## 安装后验证

执行以下命令可检查PyTorch框架和TorchNPU插件是否已成功安装。

```Python
python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
```

输出如下类似信息说明安装成功。

```text
tensor([[-0.6066,  6.3385,  0.0379,  3.3356],
        [ 2.9243,  3.3134, -1.5465,  0.1916],
        [-2.1807,  0.2008, -1.1431,  2.1523]], device='npu:0')
```

如需查看当前环境中已安装的Python、PyTorch和TorchNPU安装包版本，请参见[查询版本](../references/check_installed_versions.md)。
