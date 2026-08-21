# Compilation Optimization FAQs

- The `.so` file or a certain module is not found when running a model.

    Refer to [Dependency Installation](comp_opt_py.md) to confirm whether all dependencies have been installed.

- Whether the compiled Python can be directly migrated across different servers.

    Yes, it can be migrated. Pay attention to the glibc version. Python compiled on a lower glibc version can be migrated to a higher version, but the reverse is not possible.

- Error while loading shared libraries: libomp.so: cannot open shared object file: No such file or directory.

    PyTorch uses OpenMP by default during build. The PyTorch runtime environment requires installing the Bisheng compiler package and setting the `LD_LIBRARY_PATH` environment variable so that the dynamic library `libomp.so` can be correctly located.

- "fatal error: 'filesystem' file not found."

    The gcc version is too old. Run the command `gcc --version` to check the version, and refer to [Software Installation](../installation_guide/installation_description.md) to install GCC 8 or later.

- "Error while loading shared libraries: libomp.so: cannot open shared object file: No such file or directory."

    TorchNPU uses OpenMP by default during build. The TorchNPU runtime environment requires installing the Bisheng compiler package and setting the `LD_LIBRARY_PATH` environment variable so that the dynamic library `libomp.so` can be correctly located.

- Can PyTorch and TorchNPU share the same profile generation path?

    Yes, they can share the same path and can be merged to use the same `profdata`. The compiler recognizes internal information, and the two do not interfere with each other.

- A runtime error reports undefined symbols, including strings such as `basic_string`.

    This may be caused by inconsistent C++11 ABI settings during compilation. Check the `compile_commands.json` file in the build directories of PyTorch and TorchNPU to verify whether the macro `GLIBCXX_USE_CXX11_ABI` has the same value. If the values are inconsistent, set the environment variable `export _GLIBCXX_USE_CXX11_ABI=0` and recompile PyTorch.
