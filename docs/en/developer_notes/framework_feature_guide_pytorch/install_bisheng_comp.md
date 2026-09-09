# Installing BiSheng Compiler

## Downloading BiSheng Compiler

BiSheng Compiler can be downloaded from the official Kunpeng Community website. Click <a href="https://kunpeng-repo.obs.cn-north-4.myhuaweicloud.com/BiSheng Enterprise/BiSheng Enterprise 203.0.0/BiShengCompiler-4.1.0-aarch64-linux.tar.gz">LINK</a> to download BiSheng Compiler 4.1.0.

## Installing BiSheng Compiler

1. Decompress the BiSheng package.

    ```shell
    tar -xvf BiShengCompiler-4.1.0-aarch64-linux.tar.gz
    ```

2. Configure environment variables.

    ```shell
    export PATH=$(pwd)/BiShengCompiler-4.1.0-aarch64-linux/bin:$PATH
    export LD_LIBRARY_PATH=$(pwd)/BiShengCompiler-4.1.0-aarch64-linux/lib:$LD_LIBRARY_PATH
    ```

3. Run the `clang -v` command to verify whether the installation is successful. The following output indicates a successful installation.

    <img src="../figures/install_bisheng_comp_fig_01.png" height="132.2818" width="492.1">
