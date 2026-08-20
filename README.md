# Ascend Extension for PyTorch Plug-in

<p>
  English | <a href="./README.zh.md">简体中文</a>
</p>

## Brief Introduction

In this project, the Ascend Extension for PyTorch plug-in torch_npu is developed to enable the Ascend NPU to adapt to the PyTorch framework and provide the Ascend AI processor with the super computing power for developers using the PyTorch framework.

Ascend provides full-stack AI computing infrastructure for industry applications and services based on Huawei Ascend processors and software. You can visit the [Ascend Community](https://www.hiascend.com/en/) to learn more about Ascend.

## Directory structure

The key directories are as follows:

```text
├─ci                             #Continuous Integration Script Directory
├─cmake                          #CMake build configuration directory
├─torch_npu                      #Core Adaptation Directory
│  ├─csrc/                       #Bottom Core Directory
│  ├─npu/                        #NPU Interface Directory
│  ├─distributed/                #Distributed training adaptation directory
│  ├─asd/                        #Directory for storing the Ascend Debug tool.
├─docs                           #Project Document Directory
├─examples                       #Sample Directory
├─torchnpugen/                   #Code generation module directory.
└─test                           #Test Directory
```

## Version Description

The version description of the Ascend Extension for PyTorch includes version mapping, version compatibility, and updates. For details, see the [Release Notes for Ascend Extension for PyTorch](docs/en/release_notes/release_notes.md).

## Environment Deployment

For details about how to install the Ascend Extension for PyTorch plug-in, see the [Ascend Extension for PyTorch Software Installation](docs/en/installation_guide/menu_installation_guide.md).

## Quick Start

This section uses the CNN model as an example to describe how to migrate it to the Ascend NPU for training. For details, see the [Ascend Extension for PyTorch Quick Start](docs/en/quick_start/quick_start.md).

## Feature Description

The Ascend Extension for PyTorch plug-in provides a series of unique features in terms of memory resource optimization, communication performance optimization, computing performance optimization, and error locating assistance. For details, see the [PyTorch Framework Feature Guide](docs/en/framework_feature_guide_pytorch/menu_framework_feature.md).

## API Reference

- For details about the support of native PyTorch APIs on Ascend NPUs, see the [Support for PyTorch Native APIs](docs/en/native_apis/menu_pt_native_apis.md).
- The Ascend Extension for PyTorch plug-in provides some customized APIs. For details, see the [Ascend Extension for PyTorch Custom API](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/en/custom_APIs/menu_Pytorch_API.md).

## Branch Maintenance Policy

The maintenance phases for the Ascend Extension for PyTorch release branch are as follows:

| **Status**        | **Time**           | **Description**                                                                                                                                                                                                                               |
| ----------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Planned           | 1-3 months         | Schedule Features                                                                                                                                                                                                                             |
| development       | 6-12 months        | Develop new features, fix problems, and release new versions periodically. Different PyTorch versions have different strategies, with the development cycle of 6 months for the regular branch and 12 months for the long-term support branch |
| Maintenance       | 1 year / 3.5 years | Regular branch maintenance: 1 year; long-term support branch maintenance: 3.5 years. Fix major bugs and add new features. Release patch versions based on the impact of bugs.                                                                 |
| End of Life (EOL) | N/A                | The branch no longer accepts any modifications.                                                                                                                                                                                               |

## PyTorch Version Maintenance Policy

| **PyTorch Version** | **Maintenance Policy** | **Current status** | **Release time** | **Follow-up status**                                        | **EOL Date**   |
| ------------------- | ---------------------- | ------------------ | ---------------- | ----------------------------------------------------------- | -------------- |
| 2.10.0              | General branch         | development        | 2026 / 04 / 15   | Expected to enter maintenance from 2026/10/15               | -              |
| 2.9.0               | General branch         | development        | 2026 / 01 / 15   | Expected to enter maintenance from 2026/07/15               | -              |
| 2.8.0               | General branch         | development        | 2025 / 10 / 15   | Expected to enter maintenance from 2026/03/15               | -              |
| 2.7.1               | Long-term branching    | development        | 2025 / 10 / 15   | Expected to enter maintenance from 2026/10/15               | -              |
| 2.6.0               | General branch         | Maintenance        | 2025 / 07 / 25   | Expected to enter maintenance from 2026/01/25               | -              |
| 2.5.1               | General branch         | Maintenance        | 2024 / 11 / 08   | Expected to enter the non-maintenance state from 2026/08/08 | -              |
| 2.4.0               | General branch         | Maintenance        | 2024 / 10 / 15   | Expected to enter non-maintenance state from 2026/06/15     | -              |
| 2.3.1               | General branch         | Maintenance        | 2024 / 06 / 06   | Expected to enter non-maintenance state from 2026/06/07     |                |
| 2.2.0               | General branch         | EOL                | 2024 / 04 / 01   |                                                             | 2025 / 10 / 14 |
| 2.1.0               | Long-term support      | Maintenance        | 2023 / 10 / 15   | Expected to enter non-maintenance state from 2026/12/30     |                |
| 2.0.1               | General branch         | EOL                | 2023 / 7 / 19    |                                                             | 2024 / 3 / 14  |
| 1.11.0              | Long-term support      | EOL                | 2023 / 4 / 19    |                                                             | 2025 / 10 / 25 |
| 1.8.1               | Long-term support      | EOL                | 2022 / 4 / 10    |                                                             | 2023 / 4 / 10  |
| 1.5.0               | Long-term support      | EOL                | 2021 / 7 / 29    |                                                             | 2022 / 7 / 29  |

## Contribution guidance

Describes how to contribute code to the Ascend Extension for PyTorch plug-in library, as described in [Ascend Extension for PyTorch Plug-in Contribution Guide](docs/en/CONTRIBUTING.md).

## Contact us

You are welcome to contribute to the community. If you have any questions or suggestions, please submit [GitCode Issues](https://gitcode.com/Ascend/pytorch/issues) We'll get back to you as soon as we can. Thank you for your support.

## Safety Statement

For details about system security hardening, user suggestions, and file permission control for Ascend Extension for PyTorch, see the [Ascend Extension for PyTorch Plug-in Security Statement](docs/en/SECURITYNOTE.md).

## Disclaimer

To Ascend Extension for PyTorch Plug-in Users

- This plug-in is for debugging and development only. You must bear the risks and understand the following:
    
  - Data processing and deletion: The data generated during the use of this plug-in belongs to the user's responsibility. You are advised to delete related data in a timely manner after using the data to prevent information leakage.
  - Data confidentiality and dissemination: Users understand and agree not to send or disseminate the data generated through this plug-in at will. This plug-in and its developers are not responsible for any information leakage, data leakage, or other adverse consequences arising therefrom.
  - User input security: Users must ensure the security of the entered command lines and bear any security risks or losses caused by improper input. This plug-in and its developers are not responsible for any problems caused by improper command line input.
- Scope of Disclaimer: This disclaimer applies to all individuals or entities using this plug-in. By using this plug-in, you agree to and accept the content of this statement and are willing to bear the risks and responsibilities arising from the use of this function. If you have any objection, please stop using this plug-in.
- Read and understand the disclaimer before using this tool. For any questions or questions arising from the use of this plug-in, please contact the developer in time.

## License

License for the Ascend Extension for PyTorch plug-in. For details, seeLICENSEFile.

## Acknowledgment

Thank you for every PR from the community and welcome to contribute the Ascend Extension for PyTorch plug-in!
