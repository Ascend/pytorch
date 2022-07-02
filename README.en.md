# AscendPyTorch


# Project Introduction

This project develops the PyTorch Adapter plugin to adapt Ascend to the PyTorch framework so that developers who use the PyTorch framework can obtain powerful compute capabilities of Ascend AI Processors.

# Compilation/Execution Constraints

GCC version: 7.3.0 (required only in compilation scenarios)

CMake version: 3.12.0 or later (required only in compilation scenarios)

Python version: 3.7.5, 3.8.x, or 3.9.x (PyTorch1.5 does not support Python 3.9.x.)

# System Dependencies

## CentOS & EulerOS

yum install -y cmake zlib-devel libffi-devel openssl-devel libjpeg-turbo-devel gcc-c++ sqlite-devel dos2unix openblas

## Ubuntu

apt-get install -y gcc g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev m4 cmake dos2unix libopenblas-dev


# Ascend Auxiliary Software

| AscendPyTorch Version | CANN Version | Supported PyTorch Version | Gitee Branch   |
| :-------------------- | :----------- | :------------------------ | -------------- |
| 2.0.2                 | CANN 5.0.2   | 1.5.0.post2               | 2.0.2.tr5      |
| 2.0.3                 | CANN 5.0.3   | 1.5.0.post3               | 2.0.2.tr5      |
| 2.0.4                 | CANN 5.0.4   | 1.5.0.post4               | 2.0.2.tr5      |
| 3.0.rc1               | CANN 5.1.RC1 | 1.5.0.post5               | v1.5.0-3.0.rc1 |
| 3.0.rc1               | CANN 5.1.RC1 | 1.8.1.rc1                 | v1.8.1-3.0.rc1 |

# Installation Method

## Compile and install the PyTorch and Ascend plugin.

Download the plugin code.

```sh
git clone -b v1.8.1-3.0.rc1 https://gitee.com/ascend/pytorch.git
```

Code in the current repository is PyTorch 1.8.1. Obtain the native Pytorch source code from the root directory **/pytorch** of the current repository and rename it to **pytorch_v1.8.1**.

```sh
// Version 1.8.1

cd  pytorch  # Root directory of the plugin

git clone -b  v1.8.1 --depth=1 https://github.com/pytorch/pytorch.git  pytorch_v1.8.1
```

Run the following commands to go to the native PyTorch source code directory **pytorch_v1.8.1** and obtain the PyTorch passive dependency code:

```
cd  pytorch_v1.8.1
git submodule sync
git submodule update --init --recursive
```

After the preceding operations are complete and no error is reported, the PyTorch and the third-party code on which the PyTorch depends are generated. Then, the patch is added to the PyTorch source code and compiled.

```sh
cd ../patch
bash apply_patch.sh ../pytorch_v1.8.1
cd ../pytorch_v1.8.1
Specify the Python version packaging mode:
bash build.sh --python=3.7
or
bash build.sh --python=3.8
or
bash build.sh --python=3.9
```

Install the torch package generated in the **pytorch/pytorch_v1.8.1/dist** directory, and then compile and install the plugin.

```
cd dist
pip3 install --upgrade torch-1.8.1+ascend.rc1-cp37-cp37m-linux_{arch}.whl
```

Build and generate the binary installation package of the PyTorch plugin.

```
cd ../../ci    # Access the root directory of the plugin.
Specify the Python version packaging mode:
bash build.sh --python=3.7
or
bash build.sh --python=3.8
or
bash build.sh --python=3.9
```

Install the torch_npu package generated in the **pytorch/dist** directory.

```
cd ../dist
pip3 install --upgrade torch_npu-1.8.1rc1-cp37-cp37m-linux_{arch}.whl
```


# Running

## Execute environment variables.

Run the script for setting environment variables in the root directory of the current repository.

```
cd ../
source env.sh
```


## Customize environment variables.

The following are optional environment variables that may affect running models:

```
export COMBINED_ENABLE=1 # (Optional) Performance optimization of discontiguous tensors to contiguous tensors. To enable this function, set the value to be **1**. When operator AsStrided has been called many times inside the model, pytorch users could open this environment variable to obtain higher performance.
export ACL_DUMP_DATA=1 # (Optional) Operator data dump function, which is used for debugging. To enable this function, set the value to **1**.
```


## Run the unit test script.

Verify the execution. The output result is OK.

```shell
cd test/test_network_ops/
python3 test_div.py
```

# Documentation

For more details about the installation guide, model porting and training/inference tutorials, and API list, see [User Documents](https://gitee.com/ascend/pytorch/tree/master/docs/en).

# Suggestions and Communication

We sincerely welcome you to join discussions in the community and contribute your suggestions. We will reply to you as soon as possible.

# Branch Maintenance Policies

The version branches of AscendPyTorch have the following maintenance phases:

| **Status**        | **Duration**  | **Description**                                              |
| ----------------- | ------------- | ------------------------------------------------------------ |
| Planning          | 1 - 3 months  | Plan features.                                               |
| Development       | 3 months      | Develop features.                                            |
| Maintained        | 6 - 12 months | Allow the incorporation of all resolved issues and release the version. |
| Unmaintained      | 0 - 3 months  | Allow the incorporation of all resolved issues. No dedicated maintenance personnel are available. No version will be released. |
| End Of Life (EOL) | N/A           | Do not accept any modification to a branch.                  |

# Maintenance Status of Existing Branches

| **Branch Name** | **Status** | **Launch Date** | **Subsequent Status**                  | **EOL Date** |
| --------------- | ---------- | --------------- | -------------------------------------- | ------------ |
| **v2.0.2**      | Maintained | 2021-07-29      | Unmaintained <br> 2022-07-29 estimated |              |
| **v2.0.3**      | Maintained | 2021-10-15      | Unmaintained <br> 2022-10-15 estimated |              |
| **v2.0.4**      | Maintained | 2022-01-15      | Unmaintained <br> 2023-01-15 estimated |              |
| **v3.0.rc1**    | Maintained | 2022-04-10      | Unmaintained <br> 2023-04-10 estimated |              |


# FAQ

## The error message "no module named yaml/typing_extensions." is displayed when **bash build.sh** is run during compilation.

PyTorch compilation depends on the YAML and typing_extensions libraries, which need to be manually installed.

pip3 install pyyaml
pip3 install typing_extensions

After the installation is successful, run **make clean** and then **bash build.sh** to perform compilation. Otherwise, an unknown compilation error may occur due to the cache.

## TE cannot be found during running.

Development state:

cd /urs/local/Ascend/ascend-toolkit/latest/{arch}-linux/lib64

User state:

cd /urs/local/Ascend/nnae/latest/{arch}-linux/lib64

pip3 install --upgrade topi-0.4.0-py3-none-any.whl

pip3 install --upgrade te-0.4.0-py3-none-any.whl



## An error message is reported during CMake compilation, indicating that the version is too early.

Download the Linux version from the CMake official website and install it. (The current version is 3.18.0.)

1. Run the **yum install -y cmake==3.18.0** command to install it.

2. Download the **cmake sh** script and install it. (For details, see the CMake official website.)

   Recommended script in the x86_64 environment: **cmake-3.18.2-Linux-x86_64.sh**

   

## GCC version switch errors occur.

When the test environment is switched from GCC 4.8.5 to GCC 7.3.0, errors may occur and the PyTorch compilation fails. The following lists the libraries that require soft connections:

gcc, g++, c++ (The version must be 7.3.0.)

libstdc++->libstdc++.so.6.0.24(7.3.0)



## libblas.so cannot be found.

The OpenBLAS library is missing in the environment. You need to install it.

CentOS and EulerOS

```sh
yum -y install openblas
```

Ubuntu

```sh
apt install libopenblas-dev
```



## Warning of duplicate registration of _has_compatible_shallow_copy_type



As shown in the following figure, this warning raised from shallow copy of `Tensor.set_data`. As a judgement function of shallow copy, `_has_compatible_shallow_copy_type` does not support NPU Tensors. So we re-register this API in our PyTorch extension(torch_npu) to enable shallow copy.

Note that this warning has no negative impact on the accuracy or performance of  models, so PyTorch users can ignore it.

![输入图片说明](https://images.gitee.com/uploads/images/2022/0701/153621_2b5080c4_7902902.png)

# Release Notes

For details, see [Release Notes](https://gitee.com/ascend/pytorch/tree/master/docs/en/RELEASENOTE).