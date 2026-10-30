# Building with Clang

Before compilation, ensure that the compiler settings are correct:

```bash
export CC=clang
export CC=clang++
```

## Compiling PyTorch with Clang

After downloading the corresponding source code, you need to first install the requirements-build.txt of the corresponding repository, and then compile the CPU package:

```bash
pip install -r requirements-build.txt

export USE_CUDA=0
export USE_CUDNN=0

# If the ARM architecture is used, special linker optimization settings may be required
# export USE_PRIORITIZED_TEXT_FOR_LD=OFF

python setup.py build bdist_wheel 2>&1 | tee build.log
```

> [!NOTE]
>
> When compiling PyTorch 2.12 with Clang 20, there is a known compilation issue. For details, see the [related issue](https://github.com/pytorch/pytorch/issues/185618) in the upstream community. Modify the code by referring to the content in the [fixing PR](https://github.com/pytorch/pytorch/pull/185785) before compiling.

After compiling PyTorch, install the build artifacts in the dist directory.

## Compiling TorchNPU with Clang

When compiling TorchNPU with Clang, you need to first install the CPU package of PyTorch compiled with Clang. However, this package is not provided in the native community, so users need to compile it themselves.

After ensuring that PyTorch compiled with Clang is installed, enter the TorchNPU repository directory and start compiling TorchNPU. The compilation method is the same as that of normal TorchNPU compilation; you only need to ensure that the compiler settings are correct.
