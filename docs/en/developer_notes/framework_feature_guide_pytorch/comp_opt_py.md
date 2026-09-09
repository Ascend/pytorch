# Compilation Optimization (Python)

Starting with Python 3.6, Link-Time Optimization (LTO) and Profile-Guided Optimization (PGO) are supported and can be enabled at compile time.

> [!NOTE]
>
> Ascend 950DT does not yet support compilation optimization (Python).

1. Dependency installation.

    On Unix-based systems, Python source compilation will attempt to use available system libraries. Optional components are built only when the relevant system header files are available. If the header files are not available at compile time, the compilation can still complete, but an error will occur if the component is invoked at runtime.

    - On Fedora, RHEL, CentOS, and other dnf-based systems:

        ```shell
        sudo dnf install gcc gcc-c++ gdb lzma glibc-devel libstdc++-devel openssl-devel \
        readline-devel zlib-devel libffi-devel bzip2-devel xz-devel \
        sqlite sqlite-devel sqlite-libs libuuid-devel gdbm-libs perf \
        expat expat-devel mpdecimal python3-pip
        ```

    - On Debian, Ubuntu, and other apt-based systems:

        ```shell
        sudo apt-get install build-essential gdb lcov pkg-config \
        libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
        libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
        lzma lzma-dev tk-dev uuid-dev zlib1g-dev libmpdec-dev
        ```

2. Obtain the source code.

    Download and extract the Python source code of the desired version from [https://www.python.org/downloads/source/](https://www.python.org/downloads/source/).

    Taking Python 3.8.17 as an example, extract the file and enter the corresponding directory.

    ```shell
    tar -xvf Python-3.8.17.tgz
    cd Python-3.8.17
    ```

3. Compile and install.<a id="li2673165610272"></a>
    - For details about the environment variable configuration of the Bisheng Compiler, see [Installing Bisheng Compiler](install_bisheng_comp.md). In addition, set the following environment variables:

        ```shell
        export CC=clang
        export CXX=clang++
        ```

    - Run the command `mkdir -p <directory where Python is to be installed (absolute path)>` to create the Python installation directory.

        If you need to use conda for environment management, refer to [4](#li10673155619274) to specify the installation directory.

    - Run the command `./configure --prefix=<directory where Python is to be installed (absolute path)> --with-lto --enable-optimizations`
    - Run the following command to compile.

        ```shell
        make -j
        ```

    - Run the following command to install.

        ```shell
        make install
        ```

4. Configure and use.<a id="li10673155619274"></a>
    - In the installation directory, use `./bin/python3` to open the installed Python executable and enter the Python CLI.
    - To manage the environment with conda, first run the command `conda create -n env_name --offline -y` to create an empty conda environment.

        When compiling in [3](#li2673165610272), directly specify the Python installation directory as the directory where the empty environment resides, that is, the environment location shown in the following figure. After the installation succeeds, you can directly run `conda activate env_name` to activate the environment. The bin directory of the current compiled Python contains `python3` and `pip3` for direct use. Go to the bin directory of this Python environment and run the commands `ln -s python3 python` and `ln -s pip3 pip` to create symbolic links, after which you can use `python` and `pip` in the current conda environment.

        ![figure](../figures/comp_opt_py_fig_01.png)
