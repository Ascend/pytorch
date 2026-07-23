# 配置pip源

配置pip源，配置方法如下：

1. 使用pip安装软件包的用户，执行如下命令：

    ```bash
    cd ~/.pip
    ```

    如果提示目录不存在，则执行如下命令创建：

    ```bash
    mkdir ~/.pip 
    cd ~/.pip
    ```

2. 编辑pip.conf文件。

    使用**vi pip.conf**命令打开pip.conf文件，按 **i** 键进入编辑模式，写入如下内容：

    ```ini
    [global]
    #以华为源为例，请根据实际情况进行替换。
    index-url = https://mirrors.huaweicloud.com/repository/pypi/simple
    trusted-host = mirrors.huaweicloud.com
    timeout = 120
    ```

3. 按 **Esc** 键退出编辑模式，输入 **:wq!** 并回车保存并退出文件。
