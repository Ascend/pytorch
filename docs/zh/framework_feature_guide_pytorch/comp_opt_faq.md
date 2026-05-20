# 编译优化常见问题

- 运行模型时出现.so或者某模块找不到的情况。

    参考[依赖安装](comp_opt_py.md)，确认依赖是否已完全安装。

- 编译完的Python是否可以在不同服务器直接迁移。

    可以迁移，注意glibc版本，低版本glibc上编译的Python可以迁移到高版本，反之不行。

- Error while loading shared libraries: libomp.so: cannot open shared object file: No such file or directory.

    PyTorch构建时默认使用了OpenMP。PyTorch运行环境需要安装毕昇编译器包，并设置LD\_LIBRARY\_PATH环境变量，以便可以正确找到动态库libomp.so。

- “fatal error: 'filesystem' file not found.”

    gcc版本过低，请执行命令gcc --version查询版本，并参考《[Ascend Extension for PyTorch 软件安装指南](../installation_guide/installation_description.md)》安装GCC 8以上版本。

- “Error while loading shared libraries: libomp.so: cannot open shared object file: No such file or directory.”

    torch\_npu构建时默认使用了OpenMP。torch\_npu运行环境需要安装毕昇编译器包，并设置LD\_LIBRARY\_PATH环境变量，以便可以正确找到动态库libomp.so。

- PyTorch和torch\_npu中profile生成路径是否可以相同？

    可以相同，并且可以合并使用同一个profdata，编译器会识别内部信息，互不干扰。

- 运行时报错未定义符号，其中包含如basic\_string等字样。

    可能是编译时C++11 abi不一致导致，建议查看PyTorch和torch\_npu的build目录下的compile\_commands.json文件，检查宏GLIBCXX\_USE\_CXX11\_ABI的值是否一致。若不一致，设置环境变量**export _GLIBCXX_USE_CXX11_ABI=0**，再重新编译PyTorch。
