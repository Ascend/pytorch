# Installing GCC 11.2.0

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T03:37:57.978Z pushedAt=2026-06-15T07:27:21.205Z -->

This example demonstrates how to compile and install GCC 11.2.0 as the root user. Before compiling, check the system time and ensure it matches the current time to avoid infinite loops during the GCC compilation and installation.

1. In any server directory (for example, **/home**), run the following command to obtain the gcc-11.2.0.tar.gz source package.

    ```bash
    wget https://repo.huaweicloud.com/gnu/gcc/gcc-11.2.0/gcc-11.2.0.tar.gz
    ```

    If this command reports an error, you can click [gcc source package](https://repo.huaweicloud.com/gnu/gcc/gcc-11.2.0/gcc-11.2.0.tar.gz) to manually download and upload it.

2. Installing GCC consumes a large amount of temporary space. You can first run the following command to clear the **/tmp** directory:

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
    1. Navigate to the directory containing the gcc-11.2.0.tar.gz source package and extract it. The command is:

        ```bash
        tar -zxvf gcc-11.2.0.tar.gz
        ```

    2. Enter the extracted folder and run the following command to download the GCC dependency packages:

        ```bash
        cd gcc-11.2.0
        ./contrib/download_prerequisites
        ```

        If the above command prints an error similar to the following:

        ```bash
        gmp-6.1.0.tar.bz2: FAILED
        sha512sum: WARNING: 1 computed checksum did NOT match
        error: Cannot verify integrity of possibly corrupted file gmp-6.1.0.tar.bz2
        ```

        It indicates that the gmp package may not have been fully downloaded due to network issues. You can run the **rm -rf gmp-6.1.0.tar.bz2** command to delete the gmp package, and then run the following command to manually download the dependency packages in the "gcc-11.2.0/" folder:

        ```bash
        wget http://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.1.0.tar.bz2
        wget http://gcc.gnu.org/pub/gcc/infrastructure/mpfr-3.1.6.tar.bz2
        wget http://gcc.gnu.org/pub/gcc/infrastructure/mpc-1.0.3.tar.gz
        wget http://gcc.gnu.org/pub/gcc/infrastructure/isl-0.18.tar.bz2
        ```

        After downloading the above dependency packages, re-run the following command:

        ```bash
        ./contrib/download_prerequisites
        ```

    3. <a id="4.c"></a>
        Run the configure, compile, and install commands:

        ```bash
        ./configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/gcc11.2.0
        make -j15    # Check the number of CPUs using grep -w processor /proc/cpuinfo | wc -l. The example uses 15, and users can set this parameter as needed.
        make install    
        ```

        > [!NOTE]  
        >
        > - The compilation takes about one hour. Please be patient.
        > - The `--prefix` parameter specifies the installation path for GCC 11.2.0. Users can configure it as needed, but do not set it to `/usr/local` or `/usr`, as this will conflict with the system's default GCC installed via the package manager, potentially damaging the original GCC build environment. The example specifies `/usr/local/gcc11.2.0`.

5. Configure environment variables (only when actually needed).

    For example, before starting an online inference or training process, users need to run the following commands to configure the environment variables.

    ```bash
    export LD_LIBRARY_PATH=/usr/local/gcc11.2.0/lib64:${LD_LIBRARY_PATH}
    export CC=/usr/local/gcc11.2.0/bin/gcc
    export CXX=/usr/local/gcc11.2.0/bin/g++
    export PATH=/usr/local/gcc11.2.0/bin:${PATH}
    ```

    Where "/usr/local/gcc11.2.0" is the gcc 11.2.0 installation path configured in [4.c](#4.c). Please replace it according to the actual situation.

6. Run the **gcc --version** command to check the gcc version. If the following version information is returned, the installation is successful.

    ```bash
    gcc (GCC) 11.2.0
    ```

> [!NOTICE]  
> If you need to use GCC 11.2.0 for compilation, but due to issues with the your compilation scripts, the GCC version cannot be controlled via environment variables, the following operations can be performed to modify the symbolic links.
>
> 1. Back up the old version symbolic links.
>
>     ```bash
>     mv /usr/bin/gcc /usr/bin/gcc.bak
>     mv /usr/bin/g++ /usr/bin/g++.bak
>     mv /usr/bin/c++ /usr/bin/c++.bak
>     mv /usr/bin/cc /usr/bin/cc.bak
>     ```
>
> 2. Create new version symbolic links.
>
>     ```bash
>     ln -s /usr/local/gcc11.2.0/bin/gcc /usr/bin/gcc
>     ln -s /usr/local/gcc11.2.0/bin/g++ /usr/bin/g++
>     ln -s /usr/local/gcc11.2.0/bin/c++ /usr/bin/c++
>     ln -s /usr/local/gcc11.2.0/bin/gcc /usr/bin/cc
>     ```
>
> 3. Run the **gcc --version** command to check the gcc version. If the following version information is returned, the configuration is successful.
>
>     ```bash
>     gcc (GCC) 11.2.0
>     ```
