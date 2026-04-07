# 安装3.18.4版本cmake

以下步骤请在**root**用户下执行。

1. 在服务器任意目录下（如“/home”），执行如下命令获取cmake软件包。

    ```bash
    wget https://cmake.org/files/v3.18/cmake-3.18.4.tar.gz
    ```

2. 解压并进入软件包目录。

    ```bash
    tar -xf cmake-3.18.4.tar.gz
    cd cmake-3.18.4/
    ```

3. 执行配置、编译和安装命令。

    ```bash
    ./configure --prefix=/usr/local/cmake
    make && make install
    ```

4. 设置软链接。

    ```bash
    ln -s /usr/local/cmake/bin/cmake /usr/bin/cmake
    ```

    如果执行上述命令报“ln: failed to create symbolic link '/usr/bin/cmake': File exists”错误，说明已有软链接（'/usr/bin/cmake'），则执行如下命令删除软链接（软链接'/usr/bin/cmake'仅为示例，请用户以实际情况为准）：

    ```bash
    rm -rf /usr/bin/cmake
    ```

    再执行**ln -s /usr/local/cmake/bin/cmake /usr/bin/cmake**命令设置软链接。

5. 执行如下命令验证是否安装成功。

    ```bash
    cmake --version
    ```

    如打印“cmake version 3.18.4”则表示安装成功。
