# Compilation Optimization (TorchNPU)

> [!NOTE]
>
> Ascend 950DT does not yet support compilation optimization (TorchNPU).

1. Install dependencies.

    Building TorchNPU depends on PyTorch. Currently, you need to build PyTorch with the Bisheng Compiler, as described in [Compilation Optimization (PyTorch)](pytorch_comp_opt.md), then reinstall PyTorch in the environment and compile TorchNPU.

    You are advised to compile TorchNPU in a container. For details, see [Source Installation](../installation_guide/compilation_installation_using_source_code.md). The following descriptions use TorchNPU v2.7.1 as an example.

    Refer to [Installing Bisheng Compiler](install_bisheng_comp.md) to configure the Bisheng Compiler environment.

2. Obtain the source code.

    Git download:

    ```shell
    git clone -b v2.7.1 https://gitcode.com/ascend/pytorch.git torch_npu
    ```

3. Configure the corresponding compilation parameters based on the required optimization type and proceed with compilation. LTO and PGO optimizations can be used individually or in combination. TorchNPU now supports compilation optimization options based on the Bisheng Compiler.

    > [!NOTE]
    > 
    > Before compiling TorchNPU with Bisheng, you need to first refer to [Compilation Optimization (PyTorch)](pytorch_comp_opt.md), recompile PyTorch with Bisheng and install it, and then compile TorchNPU.

    - LTO optimization
        1. Configure compilation parameters and set environment variables.

            ```shell
            export CC=clang
            export CXX=clang++
            ```

        2. Run the compilation command.

            ```shell
            cd torch_npu
            git clean -dfx
            bash ci/build.sh --python=3.8 --enable_lto
            ```

        3. Run the `ls dist` command to view the successfully compiled whl package.
        4. Install the whl package.

            ```shell
            pip install torch_npu-*.whl --force-reinstall --no-deps
            ```

    - LTO+PGO optimization
        - First compilation (instrumentation compilation)
            - Configure compilation parameters and set environment variables.

                ```shell
                export CC=clang
                export CXX=clang++
                ```

            - Run the compilation command.

                ```shell
                cd torch_npu
                git clean -dfx
                bash ci/build.sh --python=3.8 --enable_lto --enable_pgo=1
                ```

        - Install the whl package of TorchNPU after the first compilation by running the following command:

            ```shell
            pip3 install /path/to/*.whl --force-reinstall --no-deps
            ```

            - Configure the `LLVM_PROFILE_FILE` environment variable to specify the file for generating profraw data.

                Example command:

                ```shell
                export LLVM_PROFILE_FILE=/tmp/profile/default_%m.profraw
                ```

                In the example command, `%m` allows online merging of profile data. Change it to `%p` to record data by PID.

            - Proceed based on your actual scenario and run the model normally. The instrumentation-based binary will have lower performance than the normal binary at runtime, which can be ignored.

        - Run the model to be optimized and collect profile information.
        - After the model finishes running and the program stops, a profraw format file is generated at the file path specified in the preceding steps.
        - Profile data format conversion.

            Run the following command:

            ```shell
            llvm-profdata merge /path/to/profile -o default.profdata
            ```

            This command can merge all profraw files in the `/path/to/profile` directory. Profile data files are not affected by the machine environment and can be migrated to other machines for use.

        - Secondary compilation (using Profile data)

            Configure the profdata file: Copy the `default.profdata` file generated in the previous step to the TorchNPU directory. Name the profdata file `default.profdata`.

            Run the compilation command:

            ```shell
            cd torch_npu
            git clean -dfx
            bash ci/build.sh --python=3.8 --enable_lto --enable_pgo=2
            ```

            The TorchNPU whl package after the second compilation is the high-performance package for production use.
