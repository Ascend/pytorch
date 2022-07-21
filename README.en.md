# AscendPyTorch


<h1 id="introductionmd">Introduction</h1>
This project develops the PyTorch Adapter plugin to adapt Ascend to the PyTorch framework so that developers who use the PyTorch framework can obtain powerful compute capabilities of Ascend AI Processors. When setting up the environment for model development and running under PyTorch, you can manually build and install modules adapted to the PyTorch framework on a server.

<h3 id="prerequisitesmd">Prerequisites</h3>

- The development or operating environment of CANN has been installed. For details, see the *CANN Software Installation Guide*.
- Python version: 3.7.5 or 3.8

# System Dependencies

## CentOS & EulerOS

yum install -y patch cmake==3.12.0 zlib-devel libffi-devel openssl-devel libjpeg-turbo-devel gcc-c++ sqlite-devel dos2unix openblas git gcc==7.3.0 dos2unix

## Ubuntu

apt-get install -y patch g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev m4 dos2unix libopenblas-dev git dos2unix

apt-get install -y gcc==7.3.0 cmake==3.12.0



>![](figures/icon-note.gif) **NOTE:** 
>If an error occurs during the installation of the GCC and CMake dependency commands, use the source code for installation. For details, see the FAQ.



# Ascend Auxiliary Software

| AscendPyTorch Version| CANN Version| Supported PyTorch Version| Gitee Branch|
| :------------ | :----------- | :----------- | ------------ |
| 2.0.2 | CANN 5.0.2 | 1.5.0.post2 | 2.0.2.tr5 |
| 2.0.3 | CANN 5.0.3 | 1.5.0.post3 | 2.0.3.tr5 |
| 2.0.4 | CANN 5.0.4 | 1.5.0.post4 | 2.0.4.tr5 |
| 3.0.rc1 | CANN 5.1.RC1 | 1.5.0.post5 | v1.5.0-3.0.rc1 |
| 3.0.rc1 | CANN 5.1.RC1 | 1.8.1.rc1 | v1.8.1-3.0.rc1 |
| 3.0.rc2 | CANN 5.1.RC2 | 1.5.0.post6 | v1.5.0-3.0.rc2 |
| 3.0.rc2 | CANN 5.1.RC2 | 1.8.1.rc2 | v1.8.1-3.0.rc2 |

# Installation Methods

## Installing the PyTorch Environment Dependencies


Obtain the PyTorch source code (from the current repository) that adapts to Ascend AI Processors.

   ```
   git clone -b v1.5.0-3.0.rc2 https://gitee.com/ascend/pytorch.git
   ```

## Obtaining the Native PyTorch Source Code and Third-Party Code

Obtain the native PyTorch 1.5.0 source code from the root directory **/pytorch** of the current repository. Check whether there are fixed security-related issues in the Security and Issues categories in the PyTorch native repository. Update the native PyTorch code if there are any fixes.

```sh
cd pytorch
git clone -b v1.5.0 --depth=1 https://github.com/pytorch/pytorch.git
```

Go to the **pytorch/pytorch/** directory and obtain the passive dependency code of PyTorch. It takes a long time to obtain the code.

```sh
cd pytorch
git submodule sync
git submodule update --init --recursive
```

If no error is reported, the PyTorch and third-party code on which PyTorch depends is generated.

## Generating Full PyTorch Code That Adapts to Ascend AI Processors

Go to the **pytorch/scripts** directory and run the script of the selected version. (Note: The downloaded native PyTorch source code must match the following version. Otherwise, an error may occur.)

```sh
cd ../scripts/
bash gen.sh
```

The full code adapted to the NPU is generated in the **pytorch/pytorch/** directory.


## Installing the Python Dependency Library

Go to the **pytorch/pytorch/** directory and install the Python dependency library.

```python3
cd ../pytorch
pip3 install -r requirements.txt
```


## Compiling Torch Binary Package

In the **pytorch/pytorch/** directory, run the following command:

```sh
# Python 3.7
bash build.sh
or
bash build.sh --python=3.7 (recommended)

# Python 3.8
bash build.sh --python=3.8
```

The generated binary package is in the **pytorch/pytorch/dist/** directory.

## Installing PyTorch

**x86_64:**

**torch-1.5.0+ascend-cp37-cp37m-linux_x86_64.whl** (The actual name may contain the minor version number, for example, **torch-1.5.0.post2+ascend-cp37-cp37m-linux_x86_64.whl**.)

```shell
cd dist
pip3 uninstall torch
pip3 install --upgrade torch-1.5.0+ascend-cp37-cp37m-linux_x86_64.whl
```


**arm:**

**torch-1.5.0+ascend-cp37-cp37m-linux_aarch64.whl** (The actual name may contain the minor version number, for example, **torch-1.5.0.post2+ascend-cp37-cp37m-linux_aarch64.whl**.)

```shell
cd dist
pip3 uninstall torch
pip3 install --upgrade torch-1.5.0+ascend-cp37-cp37m-linux_aarch64.whl
```


# Running

## Executing Environment Variables

Run the script for setting environment variables in the **pytorch/pytorch/** directory.

```
cd ../
source env.sh
```


The following are optional environment variables that may affect running models:

```

export COMBINED_ENABLE=1 # (Optional) Discontinuous-to-continuous level-2 derivation optimization. To enable this function, set the value to **1**. When a large number of time-consuming AsStrided operators are called in the model, you can enable this function to improve the device execution efficiency.
export ACL_DUMP_DATA=1 # (Optional) Operator data dump function, which is used for debugging. To enable this function, set the value to **1**.
export TASK_QUEUE_ENABLE=1 # Delivered by an asynchronous task to asynchronously call the ACL interface. You are advised to enable this environment variable and set its value to **1**.

```
If the system is openEuler or its inherited OS, such as UOS, run the following command to cancel CPU core binding.

    ```
    # unset GOMP_CPU_AFFINITY
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
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p583261643014"><a name="p583261643014"></a><a name="p583261643014"></a>(Optional) Discontinuous-to-continuous level-2 derivation optimization. To enable this function, set the value to <b>1</b>. When a large number of time-consuming AsStrided operators are called in the model, you can enable this function to improve the device execution efficiency.</p>
</td>
</tr>
<tr id="row183041355123411"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p730435533415"><a name="p730435533415"></a><a name="p730435533415"></a>ACL_DUMP_DATA</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p16304105533412"><a name="p16304105533412"></a><a name="p16304105533412"></a>(Optional) Operator data dump function, which is used for debugging. To enable this function, set the value to <b>1</b>.</p>
</td>
</tr>
<tr id="row19173161510309"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001152616261_p16711563237"><a name="zh-cn_topic_0000001152616261_p16711563237"></a><a name="zh-cn_topic_0000001152616261_p16711563237"></a>TASK_QUEUE_ENABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001152616261_p0711356152317"><a name="zh-cn_topic_0000001152616261_p0711356152317"></a><a name="zh-cn_topic_0000001152616261_p0711356152317"></a>(Optional) Use asynchronous task delivery to asynchronously invoke the ACL interface. To enable this function, set it to <b>1</b> (recommended default setting).</p>
</td>
</tbody>
</table>

## Running the Unit Test Script

Verify the running. The output is **OK**.


```shell
// Select a test script that matches the preceding version. The following uses the 1.5.0 version as an example.
cd ../
python3 pytorch1.5.0/test/test_npu/test_network_ops/test_div.py
```
# (Optional) Installing the Mixed Precision Module

Select the APEX module based on the following functions. For details about how to build and install the APEX module, see the [README.en](https://gitee.com/ascend/apex/tree/v1.5.0-3.0.rc2).

- APEX
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
| Planning    | 1 - 3 months  | Plan features.                    |
| Development | 3 months      | Develop features.                 |
| Maintained  | 6 - 12 months | Allow the incorporation of all resolved issues and release the version.|
| Unmaintained| 0 - 3 months  | Allow the incorporation of all resolved issues. No dedicated maintenance personnel are available. No version will be released.                                                |
| End Of Life (EOL) |  N/A |  Do not accept any modification to a branch.   |

# Maintenance Status of Existing Branches

| **Branch Name**| **Status** | **Launch Date**         | **Subsequent Status**                          | **EOL Date**|
|------------|--------------|----------------------|----------------------------------------|------------|
| **v2.0.2**   | Maintained   | 2021-07-29           | Unmaintained <br> 2022-07-29 estimated |            |
| **v2.0.3**   | Maintained   | 2021-10-15           | Unmaintained <br> 2022-10-15 estimated |            |
| **v2.0.4**   | Maintained   | 2022-01-15           | Unmaintained <br> 2023-01-15 estimated |            |
| **v3.0.rc1**   | Maintained   | 2022-04-10           | Unmaintained <br> 2023-04-10 estimated |            |
| **v3.0.rc2**   | Maintained   | 2022-07-15           | Unmaintained <br> 2023-07-15 estimated |            |

# FAQ

## When PIP is set to the Huawei source, a Python environment error occurs after the typing dependency in the **requirements.txt** file is installed.

When PIP is set to the Huawei source, open the **requirements.txt** file, delete the typing dependency, and then run the following command:

```
pip3 install -r requirments.txt
```



## The libhccl.so cannot be found by import torch.

Environment variables are not configured. You need to configure them using the **env.sh** script.

```
source pytorch/pytorch1.5.0/src/env.sh
```



## The error message "no module named yaml/typing_extensions." is displayed when bash build.sh is run during compilation.

PyTorch compilation depends on the YAML and typing_extensions libraries, which need to be manually installed.

```
pip3 install pyyaml
pip3 install typing_extensions
```

After the installation is successful, run **make clean** and then **bash build.sh** to perform compilation. Otherwise, an unknown compilation error may occur due to the cache.



## TE cannot be found during running.

Development state:

```
cd /urs/local/Ascend/ascend-toolkit/latest/{arch}-linux/lib64 # {arch} indicates the architecture name.

pip3 install --upgrade topi-0.4.0-py3-none-any.whl

pip3 install --upgrade te-0.4.0-py3-none-any.whl
```

User state:

```
cd /urs/local/Ascend/nnae/latest/{arch}-linux/lib64 # {arch} indicates the architecture name.

pip3 install --upgrade topi-0.4.0-py3-none-any.whl

pip3 install --upgrade te-0.4.0-py3-none-any.whl
```



## During CMake installation through the CLI, an error is reported indicating that the package cannot be found. During CMake compilation, another error is reported indicating that the version is too early. In this cause, you can use the installation script or source code to compile and install CMake.

Method 1: Download the installation script and install CMake. (For details, see the CMake official website.)

​		x86_64 environment: cmake-3.12.0-Linux-x86_64.sh

​		AArch64 environment: cmake-3.12.0-Linux-aarch64.sh

1. Run the following commands:

   ```
   ./cmake-3.12.0-Linux-{arch}.sh #{arch} indicates the architecture name.
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

      > ![img](figures/icon-notice.gif) **NOTE:**
      >The **--prefix** option is used to specify the **linux_gcc7.3.0** installation path, which is configurable. Do not set it to **/usr/local** or **/usr,** which is the default installation path for the GCC installed by using the software source. Otherwise, a conflict occurs and the original GCC compilation environment of the system is damaged. In this example, the installation path is set to **/usr/local/linux_gcc7.3.0**.

   4. Change the soft links.

         ```
      ln -s ${install_path}/bin/gcc /usr/bin/gcc
      ln -s ${install_path}/bin/g++ /usr/bin/g++
      ln -s ${install_path}/bin/c++ /usr/bin/c++
      ```

   5. Configure environment variables.

   Training must be performed in the compilation environment with GCC upgraded. If you want to run training, configure the following environment variable in your training script:

   ```
   export LD_LIBRARY_PATH=${install_path}/lib64:${LD_LIBRARY_PATH}
   ```

   **${install_path}** indicates the GCC 7.3.0 installation path configured in [3](#zh-cn_topic_0000001135347812_zh-cn_topic_0000001173199577_zh-cn_topic_0000001172534867_zh-cn_topic_0276688294_li1649343041310). In this example, the GCC 7.3.0 installation path is **/usr/local/linux_gcc7.3.0/**.

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



## torchvision fails to be installed using pip in the Arm environment.

The source code can be used for installation. (You need to install the AscendPyTorch first and configure environment variables using **env.sh**.)

```
git clone -b v0.6.0 https://github.com/pytorch/vision.git 
cd vision
python setup.py install
```

Verify that the torchvision is successfully installed.

```
python -c "import torchvision"
```

If no error is reported, the installation is successful.



# Release Notes

For details, see [Release Notes](docs/en/RELEASENOTE).
