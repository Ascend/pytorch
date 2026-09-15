# Building with Clang
To compile TorchNPU with Clang, you first need to install the CPU-only upstream torch package compiled with Clang. However, this package is not provided by the official PyTorch community, so you will need to build it yourself.

Before compiling, ensure that the compiler settings are correctly configured:

```bash
export CC=clang
export CXX=clang++
```

## Building upstream torch with Clang

After downloading the corresponding source code, you must first install the requirements-build.txt from the repository, and then proceed to build the CPU package:

```bash
pip install -r requirements-build.txt

export USE_CUDA=0
export USE_CUDNN=0

# For ARM architectures, you may need to enable special linker optimizations
# export USE_PRIORITIZED_TEXT_FOR_LD=OFF

python setup.py build bdist_wheel 2>&1 | tee build.log
```
After building torch, install the compiled wheel located in the dist directory.

## Building TorchNPU with Clang

After ensuring that the Clang‑compiled PyTorch is installed, navigate to the TorchNPU repository directory and begin compiling TorchNPU. The compilation process is the same as before; you only need to ensure that the compiler settings are correctly set.