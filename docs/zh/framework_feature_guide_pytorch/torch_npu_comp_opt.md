# 编译优化（torch\_npu）

1. 依赖安装。

    torch\_npu构建依赖PyTorch，目前需要参考[编译优化（PyTorch）](pytorch_comp_opt.md)用毕昇编译器构建PyTorch之后，在环境里重新安装PyTorch，然后再编译torch\_npu。

    torch\_npu推荐在容器里进行编译，参考[使用源代码进行安装](https://gitcode.com/Ascend/pytorch#%E4%BD%BF%E7%94%A8%E6%BA%90%E4%BB%A3%E7%A0%81%E8%BF%9B%E8%A1%8C%E5%AE%89%E8%A3%85)章节拉取镜像。以下描述均以torch\_npu v2.7.1版为例。

    参考[安装毕昇编译器](install_bisheng_comp.md)配置毕昇编译器环境。

2. 获取源码。

    Git下载：

    ```shell
    git clone -b v2.7.1 https://gitcode.com/ascend/pytorch.git torch_npu
    ```

3. 根据需要的优化类型进行相应编译参数设置并进行编译，LTO和PGO优化可以单独使用，也可以叠加一起使用，当前torch\_npu已支持基于毕昇编译器的编译优化选项。

    > [!NOTE]
    > 
    > 使用毕昇编译torch\_npu之前需要先参照[编译优化（PyTorch）](pytorch_comp_opt.md)，用毕昇重新编译PyTorch并安装，然后再编译torch\_npu。

    - LTO优化
        1. 配置编译参数，设置环境变量。

            ```shell
            export CC=clang
            export CXX=clang++
            ```

        2. 执行编译命令。

            ```shell
            cd torch_npu
            git clean -dfx
            bash ci/build.sh --python=3.8 --enable_lto
            ```

        3. 执行`ls dist`命令，查看编译成功的whl包。
        4. 安装whl包。

            ```shell
            pip install torch_npu-*.whl --force-reinstall --no-deps
            ```

    - LTO+PGO优化
        - 一次编译（插桩编译）
            - 配置编译参数，设置环境变量。

                ```shell
                export CC=clang
                export CXX=clang++
                ```

            - 执行编译命令。

                ```shell
                cd torch_npu
                git clean -dfx
                bash ci/build.sh --python=3.8 --enable_lto --enable_pgo=1
                ```

        - 安装一次编译后的torch\_npu的whl包，执行如下命令：

            ```shell
            pip3 install /path/to/*.whl --force-reinstall --no-deps
            ```

            - 配置环境变量LLVM\_PROFILE\_FILE指定profraw数据生成文件。

                参考命令：

                ```shell
                export LLVM_PROFILE_FILE=/tmp/profile/default_%m.profraw
                ```

                参考命令中%m允许在线合并profile数据，改为%p将按pid记录数据。

            - 根据实际情况操作，正常跑模型即可，基于插桩的二进制在运行时会比正常二进制性能差，无需关注。

        - 运行需要优化的模型，采集Profile信息。
        - 模型运行完成之后，程序停止执行，在上述文件路径有已生成的profraw格式文件。
        - Profile数据格式转换。

            执行如下命令：

            ```shell
            llvm-profdata merge /path/to/profile -o default.profdata
            ```

            该命令可以合并/path/to/profile目录下所有的profraw文件，profile数据文件不受机器环境影响，可以迁移到其他机器上使用。

        - 二次编译（使用Profile数据）

            配置profdata文件：将前一步骤中生成的default.profdata文件拷贝到torch\_npu目录下。请将profdata文件命名为default.profdata。

            执行编译命令：

            ```shell
            cd torch_npu
            git clean -dfx
            bash ci/build.sh --python=3.8 --enable_lto --enable_pgo=2
            ```

            二次编译后的torch\_npu的whl包为正式使用的高性能包。
