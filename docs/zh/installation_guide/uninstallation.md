# 卸载

## 卸载Ascend Extension for PyTorch

-   卸载PyTorch框架需执行如下命令：

    ```
    pip3 uninstall torch
    ```

-   卸载torch\_npu插件需执行如下命令：

    ```
    pip3 uninstall torch_npu
    ```

-   卸载APEX模块需执行如下命令：

    ```
    pip3 uninstall apex
    ```

> [!NOTE]  
> 如需要保存卸载日志，可在pip3 uninstall命令后面加上参数--log <path\>，并对您指定的目录<path\>做好权限管控。

## 卸载CANN软件

如果用户只卸载CANN软件包（如Toolkit），则卸载没有先后顺序，如果还要卸载驱动和固件，则需要卸载其他软件包以后再卸载驱动和固件。进入卸载脚本所在路径，执行卸载命令。

-  Toolkit
    ```
    cd /<path>/cann-<version>/{arch}-linux/script
    ./uninstall.sh
    ```

-  ops
    ```
    cd /<path>/cann-<version>/{arch}-linux/script
    ./ops_uninstall.sh
    ```
-  NNAL
    ```
    cd /<path>/nnal/
    ./nnal_uninstall.sh
    ```
其中 _<path\>_ 为软件包的安装路径，_<version\>_ 为软件包版本，\{arch\}-linux为CPU架构，请用户根据实际情况替换。

卸载完成后，若显示如下信息，则说明软件卸载成功：

```
[INFO] xxx uninstall success
```

_xxx_ 表示卸载的实际软件包名。

## 卸载驱动固件

驱动和固件的卸载没有先后顺序要求，操作步骤如下：

1.  使用PuTTY登录服务器的OS命令行。
2.  执行如下命令，切换至root用户。

    ```
    su - root
    ```

3.  在任意路径执行如下命令卸载软件包。

    -   卸载固件

        ```
        <install_path>/firmware/script/uninstall.sh
        ```
        若出现如下关键回显信息，则表示固件卸载成功。

        ```ColdFusion
        Firmware package uninstalled successfully! 
        ```
    -   卸载驱动

        ```
        <install_path>/driver/script/uninstall.sh
        ```
        若出现如下关键回显信息，则表示驱动卸载成功。
        
         ```ColdFusion
        Driver package uninstalled successfully!
        ```

    > [!NOTE]  
    > _<install\_path\>_ 表示软件包安装路径，可以执行**cat /etc/ascend\_install.info**命令查询安装路径，请根据实际情况替换。

4.  根据系统提示信息决定是否重启服务器，若需要重启系统，请执行以下命令；否则，请跳过此步骤。

    ```
    reboot
    ```

