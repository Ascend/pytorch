# Compilation Optimization (PyTorch)

Take PyTorch 2.7.1 as an example for compilation optimization.

> [!NOTE]
>
> Ascend 950DT does not yet support compilation optimization (PyTorch).

1. Install dependencies.

    You are advised to compile PyTorch in a container. For details, refer to [Source Code Installation](../installation_guide/compilation_installation_using_source_code.md).

    For details about the environment variable configuration of the BiSheng Compiler, see [Installing BiSheng Compiler](install_bisheng_comp.md).

2. Obtain the source code.
    - Git download:

        ```shell
        git clone -b v2.7.1 https://github.com/pytorch/pytorch.git pytorch-2.7.1
        cd pytorch-2.7.1
        git submodule sync
        git submodule update --init --recursive
        ```

    - Install requirements:

        ```shell
        pip install -r requirements.txt
        ```

3. Configure compilation parameters and compile according to the required optimization type. LTO and PGO optimizations can be used individually or in combination.
    - LTO optimization
        1. Configure compilation parameters and set environment variables.

            ```shell
            export CMAKE_C_FLAGS="-flto=thin -fuse-ld=lld"
            export CMAKE_CXX_FLAGS="-flto=thin -fuse-ld=lld"
            export CC=clang
            export CXX=clang++
            export USE_XNNPACK=0
            ```

            > [!NOTE]
            >
            > During compilation, you can also enable CPU acceleration libraries (oneDNN/ACL, BLAS) to improve CPU operator performance. For details, see [Enabling oneDNN/ACL Acceleration](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/PyTorch/kunpengpytorch_02_0012.html) and [Enabling BLAS Acceleration](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/PyTorch/kunpengpytorch_02_0013.html).

        2. Run the compilation command.

            ```shell
            cd pytorch-2.7.1
            git clean -dfx
            python3 setup.py bdist_wheel
            ```

        3. Run the `ls dist` command to view the successfully compiled whl package.
        4. Install the whl package.

            ```shell
            pip3 install /path/to/*.whl --force-reinstall --no-deps
            ```

    - LTO+PGO Optimization
        - First compilation (instrumented compilation)
            - Configure compilation parameters and set environment variables.

                ```shell
                export CMAKE_C_FLAGS="-flto=thin -fuse-ld=lld -fprofile-generate=/path/to/profile"
                export CMAKE_CXX_FLAGS="-flto=thin -fuse-ld=lld -fprofile-generate=/path/to/profile"
                export CC=clang
                export CXX=clang++
                export USE_XNNPACK=0 
                ```

                > [!NOTE]
                >
                > `/path/to/profile` refers to the path where the profile data file is stored when running PyTorch later.
                >
                > During compilation, you can also enable CPU acceleration libraries (oneDNN/ACL, BLAS) to improve CPU operator performance. For details, see [Enabling oneDNN/ACL Acceleration](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/PyTorch/kunpengpytorch_02_0012.html) and [Enabling BLAS Acceleration](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/PyTorch/kunpengpytorch_02_0013.html).

            - Run the compilation command.

                ```shell
                cd pytorch-2.7.1
                git clean -dfx
                python3 setup.py bdist_wheel
                ```

        - Run the model to be optimized and collect profile information.
            - Run the following command to set the `OMP_PROC_BIND` environment variable to false.

                ```shell
                export OMP_PROC_BIND=false
                ```

                > [!NOTE]
                >
                > `OMP_PROC_BIND` affects model runtime performance. Whether collecting profile data or performing optimization, ensure that this environment variable is set to false before running the model.

            - Install the PyTorch whl package from the first compilation by running the following command:

                ```shell
                pip3 install /path/to/*.whl --force-reinstall --no-deps
                ```

                Perform the operation according to the actual situation and run the model normally. The instrumented binary has lower performance than the normal binary at runtime, and you do not need to be concerned about it.

                > [!NOTE]
                > 
                > PyTorch compiled with the BiSheng Compiler must be used together with TorchNPU compiled with the BiSheng Compiler. For other compatibility details, refer to [Table 1](comp_opt_intro.md#introduction-to-compilation-optimization-solutions).

            - After the model finishes running and the program stops, a profraw format file is generated in the directory specified during the first compilation. You can also configure the `LLVM_PROFILE_FILE` environment variable on the running machine to specify where the profraw file is generated. In the reference command, `%m` allows online merging of profile data, and changing it to `%p` records data by PID.

                Reference command:

                ```shell
                export LLVM_PROFILE_FILE=/tmp/profile/default_%m.profraw
                ```

        - Profile data format conversion.

            Run the following command:

            ```shell
            llvm-profdata merge /path/to/profile -o default.profdata
            ```

            This command can merge all profraw files in the `/path/to/profile` directory. Profile data files are not affected by the machine environment and can be migrated to other machines for use.

        - Secondary compilation (using profile data)
            - Configure compilation parameters and set environment variables.

                ```shell
                export CMAKE_C_FLAGS="-flto=thin -fuse-ld=lld -fprofile-use=/path/to/profile/default.profdata"
                export CMAKE_CXX_FLAGS="-flto=thin -fuse-ld=lld -fprofile-use=/path/to/profile/default.profdata"
                ```

                > [!NOTE]
                >
                > During compilation, you can also enable CPU acceleration libraries (oneDNN/ACL, BLAS) to further improve performance. For details, see [Enabling oneDNN/ACL Acceleration](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/PyTorch/kunpengpytorch_02_0012.html) and [Enabling BLAS Acceleration](https://www.hikunpeng.com/document/detail/en/SRA/ecosystemEnable/PyTorch/kunpengpytorch_02_0013.html).

            - Run the compilation command.

                ```shell
                cd pytorch-2.7.1
                git clean -dfx
                python3 setup.py bdist_wheel
                ```

                The PyTorch whl package after the second compilation is the high-performance package for production use.

            - After installing the high-performance package and before running the model, verify again that the `OMP_PROC_BIND` environment variable is set to false.

                ```shell
                export OMP_PROC_BIND=false
                ```
