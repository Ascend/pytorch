# 方式二：源码编译安装

编译安装适用于二次开发场景，如自定义算子适配开发后，用户可以选择需要的分支版本自行编译PyTorch框架和torch\_npu插件。

执行安装命令前，请参见[安装前准备](preparing_installation.md)完成环境变量配置及其他环境准备。

## 安装PyTorch框架

具体步骤请参见[PyTorch官网](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source)。

## 安装torch\_npu插件

以下操作步骤以安装PyTorch 2.7.1版本为例。

-   **方式一（推荐）：容器场景**
    1.  下载torch\_npu源码。

        ```
        git clone https://gitcode.com/Ascend/pytorch.git -b v2.7.1-7.3.0 --depth 1  
        ```

        以v2.7.1-7.3.0为例，下载对应的Ascend Extension for PyTorch分支代码。请参见《版本说明》中的“[相关产品版本配套说明](../release_notes/related_product_version_compatibility_notes.md)”章节下载Ascend Extension for PyTorch其他版本的分支代码。

    2.  构建镜像。

        ```
        cd pytorch/ci/docker/{arch} 
        docker build -t manylinux-builder:v1 .
        ```

        > [!NOTE]
        > -   _{arch}_ 表示CPU架构（X86或ARM）。
        > -   注意不要遗漏命令结尾的“.”。

    3.  进入Docker容器，并将torch\_npu源代码挂载至容器内。

        ```
        docker run -it -v /{code_path}/pytorch:/home/pytorch manylinux-builder:v1 bash
        ```

        _{code_path}_ 表示torch\_npu源代码路径，请根据实际情况进行替换。

    4.  编译生成二进制安装包。

        ```
        cd /home/pytorch
        bash ci/build.sh --python=3.10
        ```

        如需指定其他Python版本请使用--python=3.9、--python=3.11或--python=3.12。

    5.  在运行环境中安装生成的torch\_npu插件包，如果使用非root用户进行安装，需要在命令后加`--user`。

        ```
        pip3 install --upgrade dist/torch_npu-2.7.1.post2-cp310-cp310-linux_aarch64.whl
        ```

        请用户根据实际情况更改命令中的torch\_npu包名。

-   **方式二：物理机及虚拟机场景**
    1.  安装依赖。

        选择编译安装方式安装时需要安装系统依赖，根据不同类型的操作系统，选择对应的命令安装所需依赖。

        -   openEuler、CentOS、Kylin、BCLinux、UOS V20、AntOS、AliOS、CTyunOS、CULinux、Tlinux、MTOS、vesselOS：
            1.  安装依赖（除gcc和cmake以外）。

                ```
                yum install -y patch libjpeg-turbo-devel dos2unix openblas git
                ```

            2.  安装gcc和cmake。

                根据实际情况，安装对应gcc和cmake版本，版本信息及安装指导请参见[表1](#gcc_cmake)。

        -   Debian、Ubuntu、veLinux：
            1.  安装依赖（除gcc和cmake以外）。

                ```
                apt-get install -y patch build-essential libbz2-dev libreadline-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev liblzma-dev m4 dos2unix libopenblas-dev git
                ```

            2.  安装gcc和cmake。

                根据实际情况，安装对应gcc和cmake版本，版本信息及安装指导请参见[表1](#gcc_cmake)。

        **表 1**  gcc和cmake版本要求<a id="gcc_cmake"></a>

        |PyTorch版本|系统架构|gcc版本|cmake版本|
        |--|--|--|--|
        |2.6.0|X86_64|9.4.0|3.18.0版本及以上|
        |2.6.0|AArch64|11.2.0|3.31.0版本及以上|
        |2.7.1|X86_64|11.2.0|3.18.0版本及以上|
        |2.7.1|AArch64|11.2.0|3.31.0版本及以上|
        |2.8.0|X86_64|13.3.0|3.18.0版本及以上|
        |2.8.0|AArch64|13.3.0|3.31.0版本及以上|
        |2.9.0|X86_64|13.3.0|3.18.0版本及以上|
        |2.9.0|AArch64|13.3.0|3.31.0版本及以上|

        > [!NOTE]  
        > 安装指导可参见[安装11.2.0版本gcc](installing_gcc_11-2-0.md)和[安装3.18.0版本cmake](installing_cmake_3-18-0.md)。


    2.  编译生成torch\_npu插件的二进制安装包。
        1.  以v2.7.1-7.3.0为例，下载对应的Ascend Extension for PyTorch分支代码并进入插件根目录。

            ```
            git clone -b v2.7.1-7.3.0 https://gitcode.com/Ascend/pytorch.git 
            cd pytorch
            ```

            请参见《版本说明》中的“[相关产品版本配套说明](../release_notes/related_product_version_compatibility_notes.md)”章节下载Ascend Extension for PyTorch其他版本的分支代码。

        2.  编译生成二进制安装包。

            ```
            bash ci/build.sh --python=3.10
            ```

            如需指定其他Python版本请使用--python=3.9、--python=3.11或--python=3.12。

    3.  安装pytorch/dist目录下生成的插件torch\_npu包，如果使用非root用户安装，需要在命令后加`--user`。

        ```
        pip3 install --upgrade dist/torch_npu-2.7.1.post2-cp310-cp310-linux_aarch64.whl
        ```

        请用户根据实际情况更改命令中的torch\_npu包名。

## 安装后验证

执行以下命令可检查PyTorch框架和torch\_npu插件是否已成功安装。

-   方法一

    ```Python
    python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
    ```

    输出如下类似信息说明安装成功。

    ```ColdFusion
    tensor([[-0.6066,  6.3385,  0.0379,  3.3356],
            [ 2.9243,  3.3134, -1.5465,  0.1916],
            [-2.1807,  0.2008, -1.1431,  2.1523]], device='npu:0')
    ```

-   方法二

    ```Python
    import torch
    import torch_npu
    
    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    z = x.mm(y)
    
    print(z)
    ```

    输出如下类似信息说明安装成功。

    ```ColdFusion
    tensor([[-0.0515,  0.3664],
            [-0.1258, -0.5425]], device='npu:0')
    ```

