# 配置pip源

配置pip源，配置方法如下：

1.  使用pip安装软件包的用户，执行如下命令：

    ```
    cd ~/.pip
    ```

    如果提示目录不存在，则执行如下命令创建：

    ```
    mkdir ~/.pip 
    cd ~/.pip
    ```

2.  编辑pip.conf文件。

    使用**vi pip.conf**命令打开pip.conf文件，写入如下内容：

    ```
    [global]
    #以华为源为例，请根据实际情况进行替换。
    index-url = https://mirrors.huaweicloud.com/repository/pypi/simple
    trusted-host = mirrors.huaweicloud.com
    timeout = 120
    ```

3.  执行 **:wq!** 命令保存文件。

