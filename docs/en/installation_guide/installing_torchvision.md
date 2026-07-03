# Installing torchvision

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T03:38:10.053Z pushedAt=2026-06-15T07:27:21.206Z -->

torchvision is an extension library for PyTorch, primarily used for computer vision tasks. It includes many commonly used models and datasets, making image data processing simpler and more efficient.

After installing the PyTorch framework and the torch_npu plugin, you need to install the torchvision version corresponding to the framework version. For details, see the table below.

**Table 1**  torchvision version compatibility<a id="torchvision_version"></a>

|PyTorch Version|torchvision Version|
|--|--|
|2.7.1|0.22.1|
|2.8.0|0.23.0|
|2.9.0|0.24.0|
|2.10.0|0.25.0|

Take version 0.22.1 as an example to introduce how to install torchvision.

- Method 1

    ```bash
    pip3 install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
    ```

- Method 2
  1. Obtain the torchvision source code.

        ```bash
        git clone https://github.com/pytorch/vision.git -b v0.22.1 --depth 1
        ```

        > [!NOTE]  
        > For the torchvision version, refer to [Table 1](#torchvision_version) and replace it according to the actual scenario.
  2. Go to the vision directory and compile the torchvision installation.

        ```bash
        cd vision
        python3 setup.py bdist_wheel
        ```

        > [!NOTE]  
        > After installation, a torchvision*.whl file will be generated in the dist directory.
  3. Install torchvision.

        ```bash
        cd dist
        pip3 install torchvision-0.22.*.whl
        ```

  4. Verify the installation.

        ```Python
        python3
        import torchvision
        print(torchvision.__version__)
        ```

        If the version number is displayed, the installation is successful.

        ```Python
        Python 3.10.17 (main, Nov 4 2025, 17:12:04) [GCC 9.4.0] on linux
        Type "help", "copyright", "credits" or "license" for more information.
        >>> import torchvision
        >>> print(torchvision.__version__)
        0.22.1
        >>>
        ```
