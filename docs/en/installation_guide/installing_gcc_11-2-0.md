# Installing GCC 11.2.0

This example demonstrates how to compile and install GCC 11.2.0 as the root user. Before compiling, check the system time and confirm that it matches the current time before proceeding, to avoid infinite loops during the GCC compilation and installation.

1. In any server directory (for example, `/home`), run the following command to obtain the `gcc-11.2.0.tar.gz` source package.

    ```bash
    wget https://repo.huaweicloud.com/gnu/gcc/gcc-11.2.0/gcc-11.2.0.tar.gz
    ```

    If this command reports an error, you can click [gcc source package](https://repo.huaweicloud.com/gnu/gcc/gcc-11.2.0/gcc-11.2.0.tar.gz) to manually download and upload it.

2. GCC installation uses a large amount of temporary space. You can first run the following command to clear the `/tmp` directory:

    ```bash
    rm -rf /tmp/*
    ```

3. Run the following command to install dependencies.
    - openEuler, CentOS, Kylin, BCLinux, UOS V20, AntOS, AliOS, CTyunOS, CULinux, Tlinux, MTOS, vesselOS:

        ```bash
        yum install bzip2    
        ```

    - Debian, Ubuntu, veLinux:

        ```bash
        apt-get install bzip2    
        ```

4. Compile and install GCC.
    1. Navigate to the directory containing the `gcc-11.2.0.tar.gz` source package and extract it. The command is:

        ```bash
        tar -zxvf gcc-11.2.0.tar.gz
        ```

    2. Enter the extracted folder and run the following command to download the GCC dependency packages:

        ```bash
        cd gcc-11.2.0
        ./contrib/download_prerequisites
        ```

        If the command prints an error similar to the following:

        ```bash
        gmp-6.1.0.tar.bz2: FAILED
        sha512sum: WARNING: 1 computed checksum did NOT match
        error: Cannot verify integrity of possibly corrupted file gmp-6.1.0.tar.bz2
        ```

        This indicates that the gmp package may not have been fully downloaded due to network issues. You can run the `rm -rf gmp-6.1.0.tar.bz2` command to delete the gmp package, and then run the following command to manually download the dependency packages in the `gcc-11.2.0/` folder:

        ```bash
        wget http://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.1.0.tar.bz2
        wget http://gcc.gnu.org/pub/gcc/infrastructure/mpfr-3.1.6.tar.bz2
        wget http://gcc.gnu.org/pub/gcc/infrastructure/mpc-1.0.3.tar.gz
        wget http://gcc.gnu.org/pub/gcc/infrastructure/isl-0.18.tar.bz2
        ```

        After downloading the dependency packages, run the following command again:

        ```bash
        ./contrib/download_prerequisites
        ```

    3. <a id="4.c"></a>
        Run the configuration, compilation, and installation commands:

        ```bash
        ./configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/gcc11.2.0
        make -j15    # Check the number of CPUs using grep -w processor /proc/cpuinfo|wc -l. The example uses 15, and you can set the parameter accordingly
        make install    
        ```

        > [!NOTE]  
        >
        > - The compilation takes about one hour. Please wait patiently.
        > - The `--prefix` parameter specifies the installation path of GCC 11.2.0. You can configure it as needed, but do not set it to `/usr/local` or `/usr`, because this conflicts with the GCC that the system installs by default from the software repository and breaks the original GCC compilation environment of the system. The example specifies `/usr/local/gcc11.2.0`.

5. Configure environment variables (only when actually needed).

    For example, before starting an online inference or training process, you need to run the following commands to configure the environment variables.

    ```bash
    export LD_LIBRARY_PATH=/usr/local/gcc11.2.0/lib64:${LD_LIBRARY_PATH}
    export CC=/usr/local/gcc11.2.0/bin/gcc
    export CXX=/usr/local/gcc11.2.0/bin/g++
    export PATH=/usr/local/gcc11.2.0/bin:${PATH}
    ```

    `/usr/local/gcc11.2.0` is the GCC 11.2.0 installation path configured in [4.c](#4.c). Replace it according to the actual situation.

6. Run the `gcc --version` command to check the gcc version. If the following version information is returned, the installation is successful.

    ```bash
    gcc (GCC) 11.2.0
    ```

> [!NOTICE]
>
> If you explicitly need to use GCC 11.2.0 for compilation, and the GCC version cannot be controlled through environment variables due to issues with your compilation script, you can perform the following operations to modify the symbolic links.
>
> 1. Back up the symbolic links of the old version.
>
>     ```bash
>     mv /usr/bin/gcc /usr/bin/gcc.bak
>     mv /usr/bin/g++ /usr/bin/g++.bak
>     mv /usr/bin/c++ /usr/bin/c++.bak
>     mv /usr/bin/cc /usr/bin/cc.bak
>     ```
>
> 2. Create the symbolic links of the new version.
>
>     ```bash
>     ln -s /usr/local/gcc11.2.0/bin/gcc /usr/bin/gcc
>     ln -s /usr/local/gcc11.2.0/bin/g++ /usr/bin/g++
>     ln -s /usr/local/gcc11.2.0/bin/c++ /usr/bin/c++
>     ln -s /usr/local/gcc11.2.0/bin/gcc /usr/bin/cc
>     ```
>
> 3. Run the `gcc --version` command to check the gcc version. If the following version information is returned, the configuration is successful.
>
>     ```bash
>     gcc (GCC) 11.2.0
>     ```
