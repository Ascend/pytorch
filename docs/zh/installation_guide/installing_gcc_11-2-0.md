# 安装11.2.0版本gcc

以root安装gcc 11.2.0为例演示编译安装操作，编译前请检查系统时间，确认与当前一致后再进行编译，避免编译安装gcc时陷入死循环。

1.  在服务器任意目录（如“/home”）下，执行如下命令获取gcc-11.2.0.tar.gz源码包。

    ```
    wget https://repo.huaweicloud.com/gnu/gcc/gcc-11.2.0/gcc-11.2.0.tar.gz
    ```

    若该命令报错，用户可单击[Link](https://repo.huaweicloud.com/gnu/gcc/gcc-11.2.0/gcc-11.2.0.tar.gz)手动下载并上传源码包。

2.  安装gcc时会占用大量临时空间，可先执行以下命令清空/tmp目录：

    ```
    rm -rf /tmp/*
    ```

3.  执行如下命令安装依赖。
    -   openEuler、CentOS、Kylin、BCLinux、UOS V20、AntOS、AliOS、CTyunOS、CULinux、Tlinux、MTOS、vesselOS：

        ```
        yum install bzip2    
        ```

    -   Debian、Ubuntu、veLinux：

        ```
        apt-get install bzip2    
        ```

4.  编译安装gcc。
    1.  进入gcc-11.2.0.tar.gz源码包所在目录，解压源码包，命令为：

        ```
        tar -zxvf gcc-11.2.0.tar.gz
        ```

    2.  进入解压后的文件夹，执行如下命令下载gcc依赖包：

        ```
        cd gcc-11.2.0
        ./contrib/download_prerequisites
        ```

        执行上述命令若打印类似如下报错：

        ```
        gmp-6.1.0.tar.bz2: FAILED
        sha512sum: WARNING: 1 computed checksum did NOT match
        error: Cannot verify integrity of possibly corrupted file gmp-6.1.0.tar.bz2
        ```

        表示可能因网络原因gmp包没有下载完全，可执行**rm -rf gmp-6.1.0.tar.bz2**命令删除gmp包，再执行如下命令在“gcc-11.2.0/“文件夹下手动下载依赖包：

        ```
        wget http://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.1.0.tar.bz2
        wget http://gcc.gnu.org/pub/gcc/infrastructure/mpfr-3.1.6.tar.bz2
        wget http://gcc.gnu.org/pub/gcc/infrastructure/mpc-1.0.3.tar.gz
        wget http://gcc.gnu.org/pub/gcc/infrastructure/isl-0.18.tar.bz2
        ```

        下载好上述依赖包后，重新执行以下命令：

        ```
        ./contrib/download_prerequisites
        ```

    3.  <a id="4.c"></a>
        执行配置、编译和安装命令：

        ```
        ./configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/gcc11.2.0
        make -j15    # 通过grep -w processor /proc/cpuinfo|wc -l查看cpu数，示例为15，用户可自行设置相应参数。
        make install    
        ```

        > [!NOTE]  
        > -   编译耗时1小时左右，请用户耐心等待。
        > -   其中“--prefix“参数用于指定gcc11.2.0安装路径，用户可自行配置，但注意不要配置为“/usr/local“及“/usr“，因为会与系统使用软件源默认安装的gcc相冲突，导致系统原始gcc编译环境被破坏。示例指定为“/usr/local/gcc11.2.0“。

5.  配置环境变量（请在实际需要时再进行配置）。

    例如用户在启动在线推理或训练进程前需执行如下命令配置环境变量。

    ```
    export LD_LIBRARY_PATH=/usr/local/gcc11.2.0/lib64:${LD_LIBRARY_PATH}
    export CC=/usr/local/gcc11.2.0/bin/gcc
    export CXX=/usr/local/gcc11.2.0/bin/g++
    export PATH=/usr/local/gcc11.2.0/bin:${PATH}
    ```

    其中“/usr/local/gcc11.2.0”为[4.c](#4.c)中配置的gcc11.2.0安装路径，请根据实际情况替换。

6.  执行**gcc --version**命令查看gcc版本，若返回如下版本信息，表示安装成功。

    ```
    gcc (GCC) 11.2.0
    ```

> [!NOTICE]  
> 如果用户明确需要gcc 11.2.0编译，且由于用户编译脚本等问题，无法通过环境变量控制gcc版本时，可执行如下操作，修改软链接。
> 1.  备份旧版本软链接。
>     ```
>     mv /usr/bin/gcc /usr/bin/gcc.bak
>     mv /usr/bin/g++ /usr/bin/g++.bak
>     mv /usr/bin/c++ /usr/bin/c++.bak
>     mv /usr/bin/cc /usr/bin/cc.bak
>     ```
> 2.  建立新版本软链接。
>     ```
>     ln -s /usr/local/gcc11.2.0/bin/gcc /usr/bin/gcc
>     ln -s /usr/local/gcc11.2.0/bin/g++ /usr/bin/g++
>     ln -s /usr/local/gcc11.2.0/bin/c++ /usr/bin/c++
>     ln -s /usr/local/gcc11.2.0/bin/gcc /usr/bin/cc
>     ```
> 3.  执行**gcc --version**命令查看gcc版本，若返回如下版本信息，表示配置成功。
>     ```
>     gcc (GCC) 11.2.0
>     ```

