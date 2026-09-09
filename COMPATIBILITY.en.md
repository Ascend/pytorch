# Version Matching

Torch NPUs need to be used with specific versions of PyTorch, CANN, Python, and firmware and drivers. The mapping between versions is maintained by the Ascend community. The recommended version combination is recommended to ensure the best compatibility and performance. The firmware and driver versions are related to the Ascend hardware and CANN versions. For details, see the [Firmware and driver installation page](https://www.hiascend.com/hardware/firmware-drivers/commercial).

## Recommended Version Combination

| Component |          Recommended Version          |
|:---------:|:-------------------------------------:|
| TorchNPU  | 2.12.0 (Installation Package Version) |
|  PyTorch  |                2.12.0                 |
|   CANN    |                 9.1.0                 |
|  Python   |   3.10 / 3.11 / 3.12 / 3.13 / 3.14    |

## TorchNPU version mapping table

The TorchNPU versions listed below and `torch_npu` package versions use different naming rules:

- **TorchNPU Version**: Identifies a product release and follows a version sequence independent of PyTorch, such as `26.1.0` and `26.1.1`. Final releases typically use three numeric components. Historical prereleases include an `RC` identifier, such as `6.0.RC3`. One TorchNPU product version can support multiple PyTorch versions and therefore provide packages with different version numbers.
- **TorchNPU Package Version**: Uses the matching PyTorch version as its base, typically in the form `{PyTorch version}` or `{PyTorch version}.postN`. The `.postN` suffix distinguishes subsequent package releases for the same PyTorch version. `N` is a release sequence number and cannot be directly converted to a TorchNPU product version. Historical prerelease packages may also include an `rc` identifier. Use the package version when installing or querying `torch_npu` with `pip`.

For example, TorchNPU `26.1.0` provides package version `2.7.1.post8` for PyTorch `2.7.1` and package version `2.12.0` for PyTorch `2.12.0`. Refer to the tables below for the mapping between TorchNPU product versions, package versions, and dependencies.

The following table lists the mappings for current active versions:

> The package versions and CANN `9.1.X` compatibility range for 26.1.1 follow the [release descriptions](https://gitcode.com/Ascend/pytorch/releases).

| TorchNPU Version | TorchNPU Package Version | PyTorch Version | CANN Version |
|:---:|:---:|:---:|:---:|
| 26.1.1 | 2.12.0.post2 | 2.12.0 | CANN 9.1.X |
| 26.1.1 | 2.11.0.post2 | 2.11.0 | CANN 9.1.X |
| 26.1.1 | 2.10.0.post6 | 2.10.0 | CANN 9.1.X |
| 26.1.1 | 2.9.0.post8 | 2.9.0 | CANN 9.1.X |
| 26.1.1 | 2.7.1.post10 | 2.7.1 | CANN 9.1.X |
| 26.1.0 | 2.12.0 | 2.12.0 | CANN 9.1.0 |
| 26.1.0 | 2.11.0 | 2.11.0 | CANN 9.1.0 |
| 26.1.0 | 2.10.0.post4 | 2.10.0 | CANN 9.1.0 |
| 26.1.0 | 2.9.0.post6 | 2.9.0 | CANN 9.1.0 |
| 26.1.0 | 2.7.1.post8 | 2.7.1 | CANN 9.1.0 |
| 26.0.0 | 2.10.0 | 2.10.0 | CANN 9.0.0 |
| 26.0.0 | 2.9.0.post2 | 2.9.0 | CANN 9.0.0 |
| 26.0.0 | 2.8.0.post4 | 2.8.0 | CANN 9.0.0 |
| 26.0.0 | 2.7.1.post4 | 2.7.1 | CANN 9.0.0 |
| 7.3.0 | 2.9.0 | 2.9.0 | CANN 8.5.0 |
| 7.3.0 | 2.8.0.post2 | 2.8.0 | CANN 8.5.0 |
| 7.3.0 | 2.7.1.post2 | 2.7.1 | CANN 8.5.0 |
| 7.3.0 | 2.6.0.post5 | 2.6.0 | CANN 8.5.0 |
| 7.2.0 | 2.8.0 | 2.8.0 | CANN 8.3.RC1 |
| 7.2.0 | 2.7.1 | 2.7.1 | CANN 8.3.RC1 |
| 7.2.0 | 2.6.0.post3 | 2.6.0 | CANN 8.3.RC1 |
| 7.2.0 | 2.1.0.post17 | 2.1.0 | CANN 8.3.RC1 |
| 7.1.0 | 2.6.0 | 2.6.0 | CANN 8.2.RC1 |
| 7.1.0 | 2.5.1.post1 | 2.5.1 | CANN 8.2.RC1 |
| 7.1.0 | 2.1.0.post13 | 2.1.0 | CANN 8.2.RC1 |
| 7.0.0 | 2.5.1 | 2.5.1 | CANN 8.1.RC1 |
| 7.0.0 | 2.4.0.post4 | 2.4.0 | CANN 8.1.RC1 |
| 7.0.0 | 2.3.1.post6 | 2.3.1 | CANN 8.1.RC1 |
| 7.0.0 | 2.1.0.post12 | 2.1.0 | CANN 8.1.RC1 |
| 6.0.0 | 2.4.0.post2 | 2.4.0 | CANN 8.0.0 |
| 6.0.0 | 2.3.1.post4 | 2.3.1 | CANN 8.0.0 |
| 6.0.0 | 2.1.0.post10 | 2.1.0 | CANN 8.0.0 |

<details>
<summary>Click to expand the historical version (including EOL).</summary>

| TorchNPU Version | TorchNPU Package Version | PyTorch Version | CANN Version |
|:---:|:---:|:---:|:---:|
| 6.0.RC3 | 2.4.0 | 2.4.0 | CANN 8.0.RC3 |
| 6.0.RC3 | 2.3.1.post2 | 2.3.1 | CANN 8.0.RC3 |
| 6.0.RC3 | 2.1.0.post8 | 2.1.0 | CANN 8.0.RC3 |
| 6.0.RC2 | 2.3.1 | 2.3.1 | CANN 8.0.RC2 |
| 6.0.RC2 | 2.2.0.post2 | 2.2.0 | CANN 8.0.RC2 |
| 6.0.RC2 | 2.1.0.post6 | 2.1.0 | CANN 8.0.RC2 |
| 6.0.RC2 | 1.11.0.post14 | 1.11.0 | CANN 8.0.RC2 |
| 6.0.RC1 | 2.2.0 | 2.2.0 | CANN 8.0.RC1 |
| 6.0.RC1 | 2.1.0.post4 | 2.1.0 | CANN 8.0.RC1 |
| 6.0.RC1 | 1.11.0.post11 | 1.11.0 | CANN 8.0.RC1 |
| 5.0.0 | 2.1.0 | 2.1.0 | CANN 7.0.0 |
| 5.0.0 | 2.0.1.post1 | 2.0.1 | CANN 7.0.0 |
| 5.0.0 | 1.11.0.post8 | 1.11.0 | CANN 7.0.0 |
| 5.0.RC3 | 2.1.0.rc1 | 2.1.0 | CANN 7.0.RC1 |
| 5.0.RC3 | 2.0.1 | 2.0.1 | CANN 7.0.RC1 |
| 5.0.RC3 | 1.11.0.post4 | 1.11.0 | CANN 7.0.RC1 |
| 5.0.RC2.2 | 1.11.0.post3 | 1.11.0 | CANN 6.3.RC3.1 |
| 5.0.RC2.1 | 1.11.0.post2 | 1.11.0 | CANN 6.3.RC3 |
| 5.0.RC2 | 2.0.1.rc1 | 2.0.1 | CANN 6.3.RC2 |
| 5.0.RC2 | 1.11.0.post1 | 1.11.0 | CANN 6.3.RC2 |
| 5.0.RC2 | 1.8.1.post2 | 1.8.1 | CANN 6.3.RC2 |
| 5.0.RC1 | 1.11.0 | 1.11.0 | CANN 6.3.RC1 |
| 5.0.RC1 | 1.8.1.post1 | 1.8.1 | CANN 6.3.RC1 |
| 3.0.0 | 1.5.0.post8 | 1.5.0 | CANN 6.0.1 |
| 3.0.0 | 1.8.1 | 1.8.1 | CANN 6.0.1 |
| 3.0.0 | 1.11.0.rc2 (beta) | 1.11.0 | CANN 6.0.1 |
| 3.0.RC3 | 1.5.0.post7 | 1.5.0 | CANN 6.0.RC1 |
| 3.0.RC3 | 1.8.1.rc3 | 1.8.1 | CANN 6.0.RC1 |
| 3.0.RC3 | 1.11.0.rc1 (beta) | 1.11.0 | CANN 6.0.RC1 |
| 3.0.RC2 | 1.5.0.post6 | 1.5.0 | CANN 5.1.RC2 |
| 3.0.RC2 | 1.8.1.rc2 | 1.8.1 | CANN 5.1.RC2 |
| 3.0.RC1 | 1.5.0.post5 | 1.5.0 | CANN 5.1.RC1 |
| 3.0.RC1 | 1.8.1.rc1 | 1.8.1 | CANN 5.1.RC1 |
| 2.0.4 | 1.5.0.post4 | 1.5.0 | CANN 5.0.4 |
| 2.0.3 | 1.5.0.post3 | 1.8.1 | CANN 5.0.3 |
| 2.0.2 | 1.5.0.post2 | 1.5.0 | CANN 5.0.2 |

</details>

## PyTorch and Python versions

| PyTorch Version |                        Python version                         |
|:---------------:|:-------------------------------------------------------------:|
| PyTorch 2.13.0  | Python3.10, Python3.11, Python 3.12, Python 3.13, Python 3.14 |
| PyTorch 2.12.0  | Python3.10, Python3.11, Python 3.12, Python 3.13, Python 3.14 |
| PyTorch 2.11.0  | Python3.10, Python3.11, Python 3.12, Python 3.13, Python 3.14 |
| PyTorch 2.10.0  |       Python3.10, Python3.11, Python 3.12, Python 3.13        |
|  PyTorch 2.9.0  |       Python3.10, Python3.11, Python 3.12, Python 3.13        |
|  PyTorch 2.8.0  | Python3.9, Python3.10, Python 3.11, Python 3.12, Python 3.13  |
|  PyTorch 2.7.1  | Python3.9, Python3.10, Python 3.11, Python 3.12, Python 3.13  |
|  PyTorch 2.6.0  |              Python3.9, Python3.10, Python 3.11               |
|  PyTorch 2.5.1  |              Python3.9, Python3.10, Python 3.11               |
|  PyTorch 2.4.0  |         Python3.8, Python3.9, Python3.10, Python 3.11         |
|  PyTorch 2.3.1  |         Python3.8, Python3.9, Python3.10, Python 3.11         |
|  PyTorch 2.2.0  |               Python3.8, Python3.9, Python3.10                |
|  PyTorch 2.1.0  |         Python3.8, Python3.9, Python3.10, Python 3.11         |
| PyTorch 1.11.0  |     Python3.7(>=3.7.5), Python3.8, Python3.9, Python3.10      |

## Hardware support

Torch NPU supports the following Ascend product families:

|        Product range         |
|----------------------------|
| Ascend 950DT Series Products |
| Ascend A3 Series Products   |
| Ascend A2 Series Products   |
| Ascend 910 Series Products  |
| Ascend 310 P Series      |
| Ascend 310 B Series      |
| Ascend 310 Series       |
