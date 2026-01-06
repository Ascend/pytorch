# 安装前准备

若用户使用预先训练好的模型进行数据处理和分析，即仅进行离线推理，请跳过此章节。

若用户进行训练或者在线推理，请参考以下完成安装前准备。

-   安装配套版本的NPU驱动固件、CANN软件（Toolkit、ops和NNAL）并配置CANN环境变量，具体请参考《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html)》（商用版）或《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0000.html)》（社区版）。

    CANN软件提供进程级环境变量设置脚本，训练或推理场景下使用NPU执行业务代码前需要调用该脚本，否则业务代码将无法执行。

    ```
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

    以上命令以root用户安装后的默认路径为例，请用户根据set\_env.sh的实际路径进行替换。

-   容器场景下源码编译安装torch\_npu插件，涉及从外部网络获取社区提供基础镜像、Python第三方库以及编译使用源码，代理配置等相关网络问题请参考[Docker官方文档](https://docs.docker.com/engine/cli/proxy/)。
-   根据不同类型的操作系统安装所需依赖前，请在安装用户下检查源是否可用。以配置华为镜像源为例，可参考[华为开源镜像站](https://mirrors.huaweicloud.com/)中镜像源对应的配置方法操作。
-   Python3.11的调度（即下发）性能优于Python3.10，建议用Python3.11及以上。
-   Ascend Extension for PyTorch支持安装abi1版本的安装包，具体请参见[方式三：二进制软件包安装（abi1版本）](installation_via_binary_package_abi1.md)。

    > [!NOTE]
    > -   PyTorch 2.7.1及以上版本的whl包默认使用CXX11\_ABI=1，无需修改。
    > -   PyTorch 2.6.0可自行选择CXX11\_ABI=1或CXX11\_ABI=0。

    从GCC 5.1版本开始，libstdc++为了更好的实现C++11规范，更改了std::string和std::list的一些接口，导致新老版本ABI不兼容。在C++代码编译时使用-D\_GLIBCXX\_USE\_CXX11\_ABI宏选择新老版本ABI。

    -   当安装abi1版本时，使用新版本ABI。
    -   当安装abi0版本时，使用老版本ABI。

    用户可根据实际情况的ABI版本，选择合适ABI的安装包。

-   通过源码编译安装PyTorch框架和torch\_npu插件时，需安装如下环境依赖。

    ```
    pip3 install pyyaml
    pip3 install wheel
    pip3 install setuptools
    ```

    如果使用非root用户安装，需要在命令后加`--user`，例如：**pip3 install pyyaml --user**。

> [!NOTICE]  
> 建议使用非root用户安装运行torch\_npu，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。
> 更多安全相关内容请参见《[安全声明](../security_statement/security_statement.md)》中各组件关于“文件权限控制”的说明。

