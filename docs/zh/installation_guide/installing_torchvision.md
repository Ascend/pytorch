# 安装torchvision

torchvision为PyTorch的扩展库，主要用于计算机视觉任务。它包括许多常用的模型、数据集，使得处理图像数据变得更加简单高效。

安装PyTorch框架和torch\_npu插件后，需安装对应框架版本的torchvision，具体请参见下表。

**表 1**  torchvision版本配套<a id="torchvision_version"></a>

|PyTorch版本|torchvision版本|
|--|--|
|2.6.0|0.21.0|
|2.7.1|0.22.1|
|2.8.0|0.23.0|
|2.9.0|0.24.0|


以0.22.1版本为例，介绍如何安装torchvision。

- 方式一
    ```
    pip3 install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
    ```

- 方式二
  1. 获取torchvision源码。
        ```
        git clone https://github.com/pytorch/vision.git -b v0.22.1 --depth 1
        ```
        > [!NOTE]  
        > torchvision版本请参考[表1](#torchvision_version)根据实际场景进行替换。
  2. 进入vision目录，编译安装torchvision。
        ```
        cd vision
        python3 setup.py bdist_wheel
        ```
        > [!NOTE]  
        > 安装完成后将在dist目录下生成torchvision*.whl文件。
  3. 安装torchvision。
        ```
        cd dist
        pip3 install torchvision-0.22.*.whl
        ```
  4. 安装后验证。
        ```Python
        python3
        import torchvision
        print(torchvision.__version__)
        ```
        如果输出版本号，则表示安装成功。
        ```Python
        Python 3.10.17 (main, Nov 4 2025, 17:12:04) [GCc 9.4.0] on linux
        Type "help", "copyright", "credits" or "license" for more information.
        >>> import torchvision
        >>> print(torchvision.__version__)
        0.22.1
        >>>
        ```
