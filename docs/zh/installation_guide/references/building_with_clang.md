# 使用Clang编译

若要使用Clang编译TorchNPU，需要先安装使用Clang编译PyTorch的CPU包，但原生社区中并未提供该包，需要用户自行编译。

编译前，保证编译器设置正确：

```bash
export CC=clang
export CC=clang++
```

## 使用Clang编译PyTorch

下载相应源码后，需先安装对应仓库的requirements-build.txt，而后开始编译CPU包：

```bash
pip install -r requirements-build.txt

export USE_CUDA=0
export USE_CUDNN=0

# 若为ARM架构，可能需要进行特殊链接器优化设置
# export USE_PRIOTIZED_TEXT_FOR_LD=OFF

python setup.py build bdist_hweel 2>&1 | tee build.log
```

在编译完PyTorch后，安装位于dist目录下的编译产物。

## 使用Clang编译TorchNPU

确保已安装Clang编译的PyTorch后，进入TorchNPU仓库目录下，即可开始编译TorchNPU，编译方式与正常编译TorchNPU保持一致，仅需保证编译器设置正确。
