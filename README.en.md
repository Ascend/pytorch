# AscendPyTorch
## Overview

This project develops the PyTorch Adapter plugin to adapt Ascend to the PyTorch framework so that developers who use the PyTorch framework can obtain powerful compute capabilities of Ascend AI Processors. When setting up the environments for PyTorch model development and operating, developers can manually compile related modules on servers.

## Prerequisites

- The development or operating environment of CANN has been installed. For details, see the *CANN Software Installation Guide*.
- Python 3.7.5, 3.8, and 3.9 are supported.

# System Dependencies

## CentOS & EulerOS

yum install -y patch zlib-devel libffi-devel openssl-devel libjpeg-turbo-devel gcc-c++ sqlite-devel dos2unix openblas git dos2unix

yum install -y gcc==7.3.0 cmake==3.12.0

## Ubuntu

apt-get install -y patch g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev m4 dos2unix libopenblas-dev git dos2unix

apt-get install -y gcc==7.3.0 cmake==3.12.0

>![](figures/icon-note.gif) **NOTE:** 
>If an error occurs during the installation of the GCC and CMake dependency commands, use the source code for installation. For details, see the FAQ.  
# Ascend Auxiliary Software
| AscendPyTorch Version| CANN Version| Supported PyTorch Version| Gitee Branch|
| :------------ | :----------- | :----------- | ------------- |
| 2.0.2 | CANN 5.0.2 | 1.5.0.post2 | 2.0.2.tr5 |
| 2.0.3 | CANN 5.0.3 | 1.5.0.post3 | 2.0.3.tr5 |
| 2.0.4 | CANN 5.0.4 | 1.5.0.post4 | 2.0.4.tr5 |
| 3.0.rc1 | CANN 5.1.RC1 | 1.5.0.post5 | v1.5.0-3.0.rc1 |
| 3.0.rc1 | CANN 5.1.RC1 | 1.8.1.rc1 | v1.8.1-3.0.rc1 |
| 3.0.rc2 | CANN 5.1.RC2 | 1.5.0.post6 | v1.5.0-3.0.rc2 |
| 3.0.rc2 | CANN 5.1.RC2 | 1.8.1.rc2 | v1.8.1-3.0.rc2 |

# Installation Methods

## Install the PyTorch environment dependencies.

If you install dependencies as a non-root user, add **--user** at the end of each command in this step, for example, **pip3 install pyyaml --user**.

```sh
pip3 install pyyaml
pip3 install wheel
```

## Compile and install the PyTorch and Ascend plugin.

Install the official torch package, and then compile and install the plugin.

```sh
#x86_64
pip3 install torch==1.8.1+cpu # If an error is reported when you run the pip command to install PyTorch of the CPU version, manually download the .whl package from https://download.pytorch.org/whl/torch.

#AArch64
#The community does not provide the CPU installation package of the ARM architecture. For details, see the first FAQ to compile and install PyTorch using the source code.
```

Compile and generate the binary installation package of the PyTorch plugin.
Download code of the corresponding branch and go to the root directory of the plugin.
```
# 
git clone -b v1.8.1-3.0.rc2 https://gitee.com/ascend/pytorch.git 
cd pytorch    
# Specify the Python version packaging mode:
bash ci/build.sh --python=3.7
# or
bash ci/build.sh --python=3.8
# or
bash ci/build.sh --python=3.9
```

Install the **torch_npu** package generated in the **pytorch/dist** directory. *{arch}* indicates the architecture name.

```
pip3 install --upgrade dist/torch_npu-1.8.1rc2-cp37-cp37m-linux_{arch}.whl
```
Download TorchVision.
```
pip3 install torchvision==0.9.1
```


# Running

## Execute environment variables.

Run the script for setting environment variables in the root directory of the current repository.

```
source env.sh
```


## Customize environment variables.

The following are optional environment variables that may affect running models:

```
export COMBINED_ENABLE=1 # (Optional) Discontinuous-to-continuous level-2 derivation optimization. To enable this function, set the value to 1. When a large number of time-consuming AsStrided operators are called in the model, you can enable this function to improve the device execution efficiency.
export ACL_DUMP_DATA=1 # (Optional) Operator data dump function, which is used for debugging. To enable this function, set the value to 1.
```

**Table 1** Description of environment variables
<a name="zh-cn_topic_0000001152616261_table42017516135"></a>

<table><thead align="left"><tr id="zh-cn_topic_0000001152616261_row16198951191317"><th class="cellrowborder" valign="top" width="55.48%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001152616261_p51981251161315"><a name="zh-cn_topic_0000001152616261_p51981251161315"></a><a name="zh-cn_topic_0000001152616261_p51981251161315"></a>Environment Variable</p>
</th>
<th class="cellrowborder" valign="top" width="44.519999999999996%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001152616261_p9198135114133"><a name="zh-cn_topic_0000001152616261_p9198135114133"></a><a name="zh-cn_topic_0000001152616261_p9198135114133"></a>Description</p>
</td>
</tr>
<tr id="row78312162301"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p1832171673019"><a name="p1832171673019"></a><a name="p1832171673019"></a>COMBINED_ENABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p583261643014"><a name="p583261643014"></a><a name="p583261643014"></a>(Optional) Discontinuous-to-continuous level-2 derivation optimization. To enable this function, set the value to 1. When a large number of time-consuming AsStrided operators are called in the model, you can enable this function to improve the device execution efficiency. However, the host delivery performance may deteriorate.</p>
</td>
</tr>
<tr id="row183041355123411"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p730435533415"><a name="p730435533415"></a><a name="p730435533415"></a>ACL_DUMP_DATA</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p16304105533412"><a name="p16304105533412"></a><a name="p16304105533412"></a>(Optional) Operator data dump function, which is used for debugging. To enable this function, set the value to 1.</p>
</td>
</tr>
</tbody>
</table>



## Run the unit test script.

Verify the running. The output is **OK**.

```shell
cd test/test_network_ops/
python3 test_div.py
```

# (Optional) Installing the Mixed Precision Module

AscendPyTorch 1.8.1 integrates the AMP module and can be used for training with mixed precision. The differences between the AMP module and the Apex module are as follows. You can select a desired module. For details about how to compile and install the Apex module, see the related [**README.en**](https://gitee.com/ascend/apex) file.

- AMP
  - Dynamic loss scale: The loss scale value is dynamically calculated to determine whether overflow occurs.
  - Tensor fusion is not supported.

- Apex

  - O1 configuration mode: Conv and Matmul use the float16 precision for computing, and Softmax and BN use float32.
  - O2 configuration mode: BN uses the float32 precision for computing, and others use float16.
  - Static loss scale: Parameters are statically set to ensure the convergence of training with mixed precision.
  - Dynamic loss scale: The loss scale value is dynamically calculated to determine whether overflow occurs.

  


# Documents

For more details about the installation guide, model porting and training/inference tutorials, and API list, see [User Documents](https://gitee.com/ascend/pytorch/tree/master/docs/en).

# Suggestions and Communication

We sincerely welcome you to join discussions in the community and contribute your suggestions. We will reply to you as soon as possible.

# Branch Maintenance Policies

The version branches of AscendPyTorch have the following maintenance phases:

| **Status**      | **Duration**   | **Description**                                         |
|-------------|---------------|--------------------------------------------------|
| Planning    | 1-3 months  | Plan features.                    |
| Development | 3 months      | Develop features.                 |
| Maintained  | 6-12 months | Allow the incorporation of all resolved issues and release the version.|
| Unmaintained| 0-3 months  | Allow the incorporation of all resolved issues. No dedicated maintenance personnel are available. No version will be released.                                                |
| End Of Life (EOL) |  N/A |  Do not accept any modification to a branch.   |

# Maintenance Status of Existing Branches

| **Branch Name**| **Status** | **Launch Date**         | **Subsequent Status**                          | **EOL Date**|
|------------|--------------|----------------------|----------------------------------------|------------|
| **v2.0.2**   | Maintained   | 2021-07-29           | Unmaintained <br> 2022-07-29 estimated |            |
| **v2.0.3**   | Maintained   | 2021-10-15           | Unmaintained <br> 2022-10-15 estimated |            |
| **v2.0.4**   | Maintained   | 2022-01-15           | Unmaintained <br> 2023-01-15 estimated |            |
| **v3.0.rc1**   | Maintained   | 2022-04-10           | Unmaintained <br> 2023-04-10 estimated |            |
| **v3.0.rc2**   | Maintained   | 2022-07-15           | Unmaintained <br> 2023-07-15 estimated |  
# FAQs

## When the CPU architecture is ARM, PyTorch 1.8.1 cannot be installed using the PIP3 command because the community does not provide the torch package for the ARM CPU architecture. In this case, use the source code to compile and install PyTorch 1.8.1.

Download the PyTorch 1.8.1 source package.

```
git clone -b v1.8.1 https://github.com/pytorch/pytorch.git --depth=1 pytorch_v1.8.1
```

Access the source package to obtain the passive dependency code.

```
cd pytorch_v1.8.1
git submodule sync
git submodule update --init --recursive 
```

Compile and install PyTorch 1.8.1.

```
python3 setup.py install
```

## When PIP is set to the Huawei source, a Python environment error occurs after the typing dependency in the **requirements.txt** file is installed.

When PIP is set to the Huawei source, open the **requirements.txt** file, delete the typing dependency, and then run the following command:

```
pip3 install -r requirments.txt
```

## The error message "no module named yaml/typing_extensions." is displayed when **bash build.sh** is run during compilation.

PyTorch compilation depends on the YAML and typing_extensions libraries, which need to be manually installed.

```
pip3 install pyyaml

pip3 install typing_extensions
```

After the installation is successful, run **make clean** and then **bash build.sh** to perform compilation. Otherwise, an unknown compilation error may occur due to the cache.

## TE cannot be found during running.

Development state:

```
cd /urs/local/Ascend/ascend-toolkit/latest/{arch}-linux/lib64 # {arch} is the architecture name.

pip3 install --upgrade topi-0.4.0-py3-none-any.whl

pip3 install --upgrade te-0.4.0-py3-none-any.whl
```

User state:

```
cd /urs/local/Ascend/nnae/latest/{arch}-linux/lib64 # {arch} is the architecture name.

pip3 install --upgrade topi-0.4.0-py3-none-any.whl

pip3 install --upgrade te-0.4.0-py3-none-any.whl
```

## During CMake installation through the CLI, an error is reported indicating that the package cannot be found. During CMake compilation, another error is reported indicating that the version is too early. In this cause, you can use the installation script or source code to compile and install CMake.

Method 1: Download the installation script and install CMake. (For details, see the CMake official website.)

​		x86_64 environment: cmake-3.12.0-Linux-x86_64.sh  
​		AArch64 environment: cmake-3.12.0-Linux-aarch64.sh  
1. Run the following command:

   ```
   ./cmake-3.12.0-Linux-{arch}.sh # {arch} indicates the architecture name.
   ```

2. Set the soft link.

   ```
   ln -s /usr/local/cmake/bin/cmake /usr/bin/cmake
   ```

3. Run the following command to check whether CMake has been installed:

   ```
   cmake --version
   ```

   If the message "cmake version 3.12.0" is displayed, the installation is successful.

Method 2: Use the source code to compile and install.

1. Obtain the CMake software package.

   ```
   wget https://cmake.org/files/v3.12/cmake-3.12.0.tar.gz --no-check-certificate
   ```

2. Decompress the package and go to the software package directory.

   ```
   tar -xf cmake-3.12.0.tar.gz
   cd cmake-3.12.0/
   ```

3. Run the configuration, compilation, and installation commands.

   ```
   ./configure --prefix=/usr/local/cmake
   make && make install
   ```

4. Set the soft link.

   ```
   ln -s /usr/local/cmake/bin/cmake /usr/bin/cmake
   ```

5. Run the following command to check whether CMake has been installed:

   ```
   cmake --version
   ```

   If the message "cmake version 3.12.0" is displayed, the installation is successful.

## During GCC installation through the CLI, an error is reported indicating that the package cannot be found. During GCC compilation, another error is reported.

When downloading GCC from some sources, you may be prompted by an error message indicating that the package cannot be found. In this case, use the source code to compile and install GCC.

Perform the following steps as the **root** user.

1. Download **gcc-7.3.0.tar.gz** from [https://mirrors.tuna.tsinghua.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz](https://mirrors.tuna.tsinghua.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz).

2. GCC installation requires adequate temporary space. Run the following command to clear the **/tmp** directory in advance:

   ```
   sudo rm -rf /tmp/*
   ```

3. Install the dependency package. (CentOS and Ubuntu are used as examples.)

   - For CentOS, run the following command:

     ```
     yum install bzip2    
     ```

   - For Ubuntu, run the following command:

     ```
     apt-get install bzip2    
     ```

4. Compile and install GCC.

   1. Go to the directory where the source package **gcc-7.3.0.tar.gz** is located and run the following command to decompress it:

      ```
      tar -zxvf gcc-7.3.0.tar.gz
      ```

   2. Go to the extracted directory and download the GCC dependency packages:

      ```
      cd gcc-7.3.0
      ./contrib/download_prerequisites
      ```

      If an error is reported during the command execution, run the following commands in the **gcc-7.3.0/** directory to download the dependency packages:

      ```
      wget http://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.1.0.tar.bz2
      wget http://gcc.gnu.org/pub/gcc/infrastructure/mpfr-3.1.4.tar.bz2
      wget http://gcc.gnu.org/pub/gcc/infrastructure/mpc-1.0.3.tar.gz
      wget http://gcc.gnu.org/pub/gcc/infrastructure/isl-0.16.1.tar.bz2
      ```

      After the preceding dependencies are downloaded, run the following command again:

      ```
      ./contrib/download_prerequisites
      ```

      If the verification fails, check whether there are duplicate dependency packages in the folder. If yes, delete duplicate ones.

   3. <a name="zh-cn_topic_0000001135347812_zh-cn_topic_0000001173199577_zh-cn_topic_0000001172534867_zh-cn_topic_0276688294_li1649343041310"></a>Run the configuration, compilation, and installation commands.

      ```
      ./configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/linux_gcc7.3.0
      make -j15    # Check the number of CPUs by running **grep -w processor /proc/cpuinfo|wc -l**. In this example, the number is 15. You can set the parameters as required.
      make install    
      ```

      >![](https://gitee.com/ascend/pytorch/raw/v1.8.1-3.0.rc2/docs/en/PyTorch%20Network%20Model%20Porting%20and%20Training%20Guide/public_sys-resources/icon-notice.gif) **NOTICE:** 
      >The **--prefix** option is used to specify the **linux\_gcc7.3.0** installation path, which is configurable. Do not set it to **/usr/local** or **/usr,** which is the default installation path for the GCC installed by using the software source. Otherwise, a conflict occurs and the original GCC compilation environment of the system is damaged. In this example, the installation path is set to **/usr/local/linux\_gcc7.3.0**.
   4. Change the soft links.

         ```
      ln -s ${install_path}/bin/gcc /usr/bin/gcc
      ln -s ${install_path}/bin/g++ /usr/bin/g++
      ln -s ${install_path}/bin/c++ /usr/bin/c++
      ```


5. Configure the environment variable.

   
   Training must be performed in the compilation environment with GCC upgraded. If you want to run training, configure the following environment variable in your training script:

   ```
   export LD_LIBRARY_PATH=${install_path}/lib64:${LD_LIBRARY_PATH}
   ```

   **${install_path}** indicates the GCC 7.3.0 installation path configured in [3](#zh-cn_topic_0000001135347812_zh-cn_topic_0000001173199577_zh-cn_topic_0000001172534867_zh-cn_topic_0276688294_li1649343041310). In this example, the GCC 7.3.0 installation path is **/usr/local/gcc7.3.0/**.

   >![](figures/icon-note.gif) **NOTE:** 
   >Skip this step if you do not need to use the compilation environment with GCC upgraded.

If the PyTorch compilation fails, check whether the soft link library is correct.

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

## No device is mounted to the container.

An NPU-related error is reported when a script is run in the container. The **device** parameter is not set before container instance startup. As a result, the instance cannot be started.

![](https://gitee.com/ascend/pytorch/raw/v1.8.1-3.0.rc2/figures/FAQ.png)

Run the following command to restart the container:

```sh
docker run -it --ipc=host \
--device=/dev/davinciX \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver \
-v /usr/local/dcmi \
-v /usr/local/bin/npu-smi \
${image name}:{tag} \
/bin/bash
```

Parameter description:

**/dev/davinci*X***: NPU device. *X* is the physical ID of the chip, for example, **davinci0**.

**/dev/davinci_manager**: management device

**/dev/devmm_svm**: management device

**/dev/hisi_hdc**: management device

**/usr/local/Ascend/driver**: driver directory

**/usr/local/dcmi**: DCMI directory

**/usr/local/bin/npu-smi**: npu-smi tool

**$*{image name}:{tag}***: image name and version

## An error is reported during installation of -torch--whl-, indicating that -torch-1-5-0xxxx- and -torchvision- versions do not match.

During installation of **torch-\*.whl**, the message "ERROR: torchvision 0.6.0 has requirement torch==1.5.0, but you'll have torch 1.5.0a0+1977093 which is incompatible" is displayed.  
![](https://gitee.com/ascend/pytorch/raw/v1.8.1-3.0.rc2/figures/zh-cn_image_0000001190081735.png)

When the PyTorch is installed, the version check is automatically triggered. The version of the torchvision installed in the environment is 0.6.0. During the check, it is found that the version of the **torch-\*.whl** is inconsistent with the required version 1.5.0. As a result, an error message is displayed, but the installation is successful.

This problem does not affect the actual result, and no action is required.


## When you run **import torch_npu**, if "_has_compatible_shallow_copy_type" is output, a warning about a repeated registration error is displayed.

The warning is triggered by the shallow copy operation of **Tensor.set_data**, as shown in the following figure. The main cause is that after the PyTorch plugin is decoupled, `_has_compatible_shallow_copy_type` cannot detect shallow copy of NPU tensors. You need to register `_has_compatible_shallow_copy_type` again.

This error can be ignored because it does not affect the model accuracy and performance.

It will be resolved after the NPU's device ID is incorporated into the community or the `_has_compatible_shallow_copy_type` registration mode is changed in a later PyTorch version.

![Input Image Description](https://images.gitee.com/uploads/images/2022/0701/153621_2b5080c4_7902902.png)

## An error is reported when torch_npu is referenced with Python commands in the compilation directory.

To verify the torch_npu reference, switch to another directory. If you perform the verification in the compilation directory, the following error message is displayed:

<img src="figures/FAQ torch_npu.png" style="zoom:150%;" />

## Multi card training is stuck in initialization stage until timeout

IPv6 address is used in the "init_process_group" function of multi card communication, for example ::1 (Note that the localhost may point to the IPv6 address)
Use IPv4 to avoid this problem

# Version Description

For details, see [Release Notes](https://gitee.com/ascend/pytorch/tree/master/docs/en/RELEASENOTE).
