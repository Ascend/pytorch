# AscendPyTorch


# Project Introduction
This project develops the PyTorch Adapter plugin to adapt Ascend to the PyTorch framework so that developers who use the PyTorch framework can obtain powerful compute capabilities of Ascend AI Processors.

# Compilation/Execution Constraints

GCC version: 7.3.0 (required only in compilation scenarios)

CMake version: 3.12.0 or later (required only in compilation scenarios)

Python versions: 3.7.5 and 3.8.*x* (The compilation methods are different. For details, see the script compilation section.)


# System Dependencies

## CentOS & EulerOS

yum install -y cmake zlib-devel libffi-devel openssl-devel libjpeg-turbo-devel gcc-c++ sqlite-devel dos2unix openblas

## Ubuntu

apt-get install -y gcc g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev m4 cmake dos2unix libopenblas-dev


# Ascend Auxiliary Software

| AscendPyTorch Version| CANN Version| Supported PyTorch Version|
| :---------------- | :--------- | :------------------------------- |
| 2.0.2             | CANN 5.0.2 | 1.5.0                            |
| 2.0.3             | CANN 5.0.3 | 1.5.0 and 1.8.1 (Only the ResNet-50 model is supported.)|
| 2.0.4             | CANN 5.0.4 | 1.5.0 and 1.8.1 (Only the ResNet-50 model is supported.)|

# Method of Use - Full Code Generation and Compilation

## Obtain the native PyTorch source code and third-party code.

Currently, PyTorch 1.5.0 and 1.8.1 are supported. Obtain the native PyTorch source code from the root directory **pytorch/** in the current repository as required.

```sh
// Version 1.5.0
git clone -b v1.5.0 --depth=1 https://github.com/pytorch/pytorch.git
// Version 1.8.1
git clone -b v1.8.1 --depth=1 https://github.com/pytorch/pytorch.git
```

Go to the **pytorch/pytorch/** directory and obtain the passive dependency code of PyTorch. (It takes a long time to obtain the code.)

```sh
git submodule sync
git submodule update --init --recursive
```

If no error is reported after the preceding operations are complete, the PyTorch and third-party code on which PyTorch depends is generated.

## Generate the PyTorch code adapted to Ascend AI Processors.

Go to the **pytorch/scripts** directory and run the script based on the selected version. (Note: The downloaded native PyTorch source code must match the following version. Otherwise, an error may occur.)

```sh
// The default version is 1.5.0.
bash gen.sh
// For version 1.8.1, use the -v option to specify the version number.
bash gen.sh -v 1.8.1
```

The full code adapted to NPUs is generated in the **pytorch/pytorch/** directory.


## Install the Python dependency.

Go to the **pytorch/pytorch/** directory and install the Python dependency.

```python3
pip3 install -r requirements.txt
```


## Compile the binary package of Torch.

Go to the **pytorch/pytorch/** directory and run the following command:

```sh
# Python 3.7
bash build.sh
or
bash build.sh --python=3.7 (recommended)

# Python 3.8
bash build.sh --python=3.8
```

The generated binary package is in the **pytorch/pytorch/dist/** directory.

# Installation

### (Version 1.5.0 is used as an example. The same rule applies to version 1.8.1.)

**x86_64:**

**torch-1.5.0+ascend-cp37-cp37m-linux_x86_64.whl** (The actual name may contain the minor version number, for example, **torch-1.5.0.post2+ascend-cp37-cp37m-linux_x86_64.whl**.)

```shell
pip3 uninstall torch
pip3 install --upgrade torch-1.5.0+ascend-cp37-cp37m-linux_x86_64.whl
```

**arm:**

**torch-1.5.0+ascend-cp37-cp37m-linux_aarch64.whl** (The actual name may contain the minor version number, for example, **torch-1.5.0.post2+ascend-cp37-cp37m-linux_aarch64.whl**.)

```shell
pip3 uninstall torch
pip3 install --upgrade torch-1.5.0+ascend-cp37-cp37m-linux_aarch64.whl
```


# Running

## Execute environment variables.

Run the script for setting environment variables in the root directory of the current repository.

```
source pytorch/env.sh
```


## Customize environment variables.

The following environment variables are function classes used in NPU scenarios or environment variables that can improve performance:

```
export TASK_QUEUE_ENABLE=1 # Delivered by an asynchronous task to asynchronously call the ACL interface. You are advised to enable this environment variable and set its value to 1.
export PTCOPY_ENABLE=1 # Use the PTCopy operator mode to accelerate continuous rotation and copy. You are advised to enable this environment variable and set its value to 1.
```

The following are optional environment variables that may affect running models:

```
export DYNAMIC_COMPILE_ENABLE=1  # Dynamic shape feature. This environment variable is optional for shape change scenarios. To enable it, set its value to 1.
export COMBINED_ENABLE=1 # Optimization of scenarios where two inconsecutive operators are combined. This environment variable is optional. To enable it, set its value to 1.
export TRI_COMBINED_ENABLE=1 # Optimization of scenarios where three inconsecutive operators are combined. This environment variable is optional. To enable it, set its value to 1.
export ACL_DUMP_DATA=1 # Operator data dump function, which is used for debugging. This environment variable is optional. To enable it, set its value to 1.
export DYNAMIC_OP="ADD#MUL" # Operator implementation. The ADD and MUL operators have different performance in different scenarios. This environment variable is optional.
```


## Run the unit test script.

Verify the execution. The output result is OK.

```shell
// Select a test script that matches the preceding version. The following uses the 1.5.0 version as an example.
python3 pytorch1.5.0/test/test_npu/test_div.py
// The following uses the 1.8.1 version as an example.
python3 pytorch1.8.1/test/test_npu/test_div.py
```

# Documentation

For more details about the installation guide, model porting and training/inference tutorials, and API list, see [User Documents](https://gitee.com/ascend/pytorch/tree/master/docs/en).

# Suggestions and Communication

We sincerely welcome you to join discussions in the community and contribute your suggestions. We will reply to you as soon as possible.

# Branch Maintenance Policies

The version branches of AscendPyTorch have the following maintenance phases:

| **Status**| **Duration**| **Description**|
| ----------------- | ------------- | -------------------------------------------------- |
| Planning          | 1 - 3 months  | Plan features.|
| Development       | 3 months      | Develop features.|
| Maintained        | 6 - 12 months | Allow the incorporation of all resolved issues and release the version.|
| Unmaintained      | 0 - 3 months  | Allow the incorporation of all resolved issues. No dedicated maintenance personnel are available. No version will be released.|
| End Of Life (EOL) | N/A           | Do not accept any modifications to a branch. |

# Maintenance Status of Existing Branches

| **Branch Name**| **Status**| **Launch Date**| **Subsequent Status**| **EOL Date**|
| ---------- | ------------ | ------------ | -------------------------------------- | ------------ |
| **v2.0.2** | Maintained   | 2021-07-29   | Unmaintained <br> 2022-07-29 estimated |              |
| **v2.0.3** | Maintained   | 2021-10-15   | Unmaintained <br> 2022-10-15 estimated |              |
| **v2.0.4** | Maintained   | 2022-11-15   | Unmaintained <br> 2023-01-15 estimated |              |


# FAQs

## The error message "no module named yaml/typing_extensions." is displayed when **bash build.sh** is run during compilation.

PyTorch compilation depends on the YAML and typing_extensions libraries, which need to be manually installed.

```
pip3 install pyyaml
pip3 install typing_extensions
```

After the installation is successful, run **make clean** and then **bash build.sh** to perform compilation. Otherwise, an unknown compilation error may occur due to the cache.

## TE cannot be found during running

Development state:

```
cd /urs/local/ascend-toolkit/latest/fwkacllib/lib64
```

User state:

```
cd /urs/local/nnae/latest/fwkacllib/lib64

pip3 install --upgrade topi-0.4.0-py3-none-any.whl

pip3 install --upgrade te-0.4.0-py3-none-any.whl
```



## An error message is reported during CMake compilation, indicating that the version is too early.

Download the Linux version from the CMake official website and install it. (The current version is 3.18.0.)

1. Run the **yum install -y cmake==3.18.0** command to install it.

2. Download the **cmake sh** script and install it. (For details, see the CMake official website.)

   Recommended script in the x86_64 environment: **cmake-3.18.2-Linux-x86_64.sh**

   

## GCC version switch errors occur.

When the test environment is switched from GCC 4.8.5 to GCC 7.3.0, errors may occur and the PyTorch compilation may fail. The following lists the libraries that require soft connections:

gcc, g++, c++ (The version must be 7.3.0.)

libstdc++->libstdc++.so.6.0.24 (7.3.0)



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

# Release Notes

For details, see [Release Notes](https://gitee.com/ascend/pytorch/tree/master/docs/en/RELEASENOTE).
