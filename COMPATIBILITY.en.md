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

**The branch name of the TorchNPU release version is**`{PyTorch版本}-{TorchNPU版本}`Naming rules. The former is the PyTorch version matching the TorchNPU. The following are the mapping relationships between the current active versions:

| TorchNPU installation package version | GitCode Branch | PyTorch Version | CANN version |
|:-------------------------------------:|:--------------:|:---------------:|:------------:|
|                2.12.0                 | v2.12.0-26.1.0 |     2.12.0      |  CANN 9.1.0  |
|                2.11.0                 | v2.11.0-26.1.0 |     2.11.0      |  CANN 9.1.0  |
|             2.10.0.post4              | v2.10.0-26.1.0 |     2.10.0      |  CANN 9.1.0  |
|              2.9.0.post6              | v2.9.0-26.1.0  |      2.9.0      |  CANN 9.1.0  |
|              2.7.1.post8              | v2.7.1-26.1.0  |      2.7.1      |  CANN 9.1.0  |
|              2.9.0.post2              | v2.9.0-26.0.0  |      2.9.0      |  CANN 9.0.0  |
|              2.8.0.post4              | v2.8.0-26.0.0  |      2.8.0      |  CANN 9.0.0  |
|              2.7.1.post4              | v2.7.1-26.0.0  |      2.7.1      |  CANN 9.0.0  |
|                 2.9.0                 |  v2.9.0-7.3.0  |      2.9.0      |  CANN 8.5.0  |
|              2.8.0.post2              |  v2.8.0-7.3.0  |      2.8.0      |  CANN 8.5.0  |
|              2.7.1.post2              |  v2.7.1-7.3.0  |      2.7.1      |  CANN 8.5.0  |
|              2.6.0.post5              |  v2.6.0-7.3.0  |      2.6.0      |  CANN 8.5.0  |
|                 2.8.0                 |  v2.8.0-7.2.0  |      2.8.0      | CANN 8.3.RC1 |
|                 2.7.1                 |  v2.7.1-7.2.0  |      2.7.1      | CANN 8.3.RC1 |
|              2.6.0.post3              |  v2.6.0-7.2.0  |      2.6.0      | CANN 8.3.RC1 |
|             2.1.0.post17              |  v2.1.0-7.2.0  |      2.1.0      | CANN 8.3.RC1 |
|                 2.6.0                 |  v2.6.0-7.1.0  |      2.6.0      | CANN 8.2.RC1 |
|              2.5.1.post1              |  v2.5.1-7.1.0  |      2.5.1      | CANN 8.2.RC1 |
|             2.1.0.post13              |  v2.1.0-7.1.0  |      2.1.0      | CANN 8.2.RC1 |
|                 2.5.1                 |  v2.5.1-7.0.0  |      2.5.1      | CANN 8.1.RC1 |
|              2.4.0.post4              |  v2.4.0-7.0.0  |      2.4.0      | CANN 8.1.RC1 |
|              2.3.1.post6              |  v2.3.1-7.0.0  |      2.3.1      | CANN 8.1.RC1 |
|             2.1.0.post12              |  v2.1.0-7.0.0  |      2.1.0      | CANN 8.1.RC1 |
|              2.4.0.post2              |  v2.4.0-6.0.0  |      2.4.0      |  CANN 8.0.0  |
|              2.3.1.post4              |  v2.3.1-6.0.0  |      2.3.1      |  CANN 8.0.0  |
|             2.1.0.post10              |  v2.1.0-6.0.0  |      2.1.0      |  CANN 8.0.0  |

<details>
<summary>Click to expand the historical version (including EOL).</summary>

| TorchNPU installation package version |  GitCode Branch   | PyTorch Version |  CANN version  |
|:-------------------------------------:|:-----------------:|:---------------:|:--------------:|
|                 2.4.0                 |  v2.4.0-6.0.rc3   |      2.4.0      |  CANN 8.0.RC3  |
|              2.3.1.post2              |  v2.3.1-6.0.rc3   |      2.3.1      |  CANN 8.0.RC3  |
|              2.1.0.post8              |  v2.1.0-6.0.rc3   |      2.1.0      |  CANN 8.0.RC3  |
|                 2.3.1                 |  v2.3.1-6.0.rc2   |      2.3.1      |  CANN 8.0.RC2  |
|              2.2.0.post2              |  v2.2.0-6.0.rc2   |      2.2.0      |  CANN 8.0.RC2  |
|              2.1.0.post6              |  v2.1.0-6.0.rc2   |      2.1.0      |  CANN 8.0.RC2  |
|             1.11.0.post14             |  v1.11.0-6.0.rc2  |     1.11.0      |  CANN 8.0.RC2  |
|                 2.2.0                 |  v2.2.0-6.0.rc1   |      2.2.0      |  CANN 8.0.RC1  |
|              2.1.0.post4              |  v2.1.0-6.0.rc1   |      2.1.0      |  CANN 8.0.RC1  |
|             1.11.0.post11             |  v1.11.0-6.0.rc1  |     1.11.0      |  CANN 8.0.RC1  |
|                 2.1.0                 |   v2.1.0-5.0.0    |      2.1.0      |   CANN 7.0.0   |
|              2.0.1.post1              |   v2.0.1-5.0.0    |      2.0.1      |   CANN 7.0.0   |
|             1.11.0.post8              |   v1.11.0-5.0.0   |     1.11.0      |   CANN 7.0.0   |
|               2.1.0.rc1               |  v2.1.0-5.0.rc3   |      2.1.0      |  CANN 7.0.RC1  |
|                 2.0.1                 |  v2.0.1-5.0.rc3   |      2.0.1      |  CANN 7.0.RC1  |
|             1.11.0.post4              |  v1.11.0-5.0.rc3  |     1.11.0      |  CANN 7.0.RC1  |
|             1.11.0.post3              | v1.11.0-5.0.rc2.2 |     1.11.0      | CANN 6.3.RC3.1 |
|             1.11.0.post2              | v1.11.0-5.0.rc2.1 |     1.11.0      |  CANN 6.3.RC3  |
|               2.0.1.rc1               |  v2.0.1-5.0.rc2   |      2.0.1      |  CANN 6.3.RC2  |
|             1.11.0.post1              |  v1.11.0-5.0.rc2  |     1.11.0      |  CANN 6.3.RC2  |
|              1.8.1.post2              |  v1.8.1-5.0.rc2   |      1.8.1      |  CANN 6.3.RC2  |
|                1.11.0                 |  v1.11.0-5.0.rc1  |     1.11.0      |  CANN 6.3.RC1  |
|              1.8.1.post1              |  v1.8.1-5.0.rc1   |      1.8.1      |  CANN 6.3.RC1  |
|              1.5.0.post8              |   v1.5.0-3.0.0    |      1.5.0      |   CANN 6.0.1   |
|                 1.8.1                 |   v1.8.1-3.0.0    |      1.8.1      |   CANN 6.0.1   |
|           1.11.0.rc2 (beta)           |   v1.11.0-3.0.0   |     1.11.0      |   CANN 6.0.1   |
|              1.5.0.post7              |  v1.5.0-3.0.rc3   |      1.5.0      |  CANN 6.0.RC1  |
|               1.8.1.rc3               |  v1.8.1-3.0.rc3   |      1.8.1      |  CANN 6.0.RC1  |
|           1.11.0.rc1 (beta)           |  v1.11.0-3.0.rc3  |     1.11.0      |  CANN 6.0.RC1  |
|              1.5.0.post6              |  v1.5.0-3.0.rc2   |      1.5.0      |  CANN 5.1.RC2  |
|               1.8.1.rc2               |  v1.8.1-3.0.rc2   |      1.8.1      |  CANN 5.1.RC2  |
|              1.5.0.post5              |  v1.5.0-3.0.rc1   |      1.5.0      |  CANN 5.1.RC1  |
|               1.8.1.rc1               |  v1.8.1-3.0.rc1   |      1.8.1      |  CANN 5.1.RC1  |
|              1.5.0.post4              |     2.0.4.tr5     |      1.5.0      |   CANN 5.0.4   |
|              1.5.0.post3              |     2.0.3.tr5     |      1.8.1      |   CANN 5.0.3   |
|              1.5.0.post2              |     2.0.2.tr5     |      1.5.0      |   CANN 5.0.2   |

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
