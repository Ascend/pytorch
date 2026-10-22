# Querying Versions

Run the following commands to check the versions of the installed Python, PyTorch framework, and TorchNPU installation packages.

- Check the installed Python version.

    ```bash
    python --version
    ```

    The output shows the Python version as follows.

    ```text
    Python 3.13.0
    ```

- Check the versions of the installed PyTorch framework and TorchNPU installation packages.

    ```bash
    pip list | grep torch
    ```

    The output shows the versions of the PyTorch framework and TorchNPU installation packages as follows.

    ```text
    torch     2.13.0+cpu
    torch_npu      2.13.0rc1
    ```

    > [!NOTE]
    >
    > Because each TorchNPU version publishes installation packages for multiple PyTorch versions, the version numbers of the released installation packages and the TorchNPU version numbers follow different naming rules. To query the mapping between version numbers, click [Related Product Version Compatibility Description](../../../../COMPATIBILITY.md).
