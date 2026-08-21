# TorchNPU

<p>
    English | <a href="./README.zh.md">简体中文</a>
</p>

## Brief Introduction

As a core component of the Ascend for PyTorch community, TorchNPU is a deep learning adaptation plug-in developed by Ascend for PyTorch. It enables the PyTorch framework to directly invoke the Ascend NPU and provide developers with the powerful computing power of the Ascend AI processor.

Ascend provides full-stack AI computing infrastructure for industry applications and services based on Huawei Ascend processors and software. You can visit the [Ascend Community](https://www.hiascend.com/en/) to learn more about Ascend.

## Directory structure

The key directories are as follows:

```text
├─ci                             #Continuous integration script directory
├─cmake                          #CMake build configuration directory
├─torch_npu                      #Core Adaptation Directory
│  ├─csrc/                       #Bottom Core Directory
│  ├─npu/                        #NPU Interface Directory
│  ├─distributed/                #Distributed training adaptation directory
│  ├─asd/                        #Ascend debug tool directory
├─docs                           #Project Document Directory
├─examples                       #Sample Directory
├─torchnpugen/                   #Code generation module directory.
└─test                           #Test Directory
```

## Version Description

The version description of the TorchNPU includes version mapping, version compatibility, and updates. For details, see the[Version Description](docs/en/release_notes/release_notes.md).

## Environment Deployment

For details about how to install the TorchNPU plug-in, see the [Install the software.](docs/en/installation_guide/menu_installation_guide.md)".

## Quick Start

This section uses the CNN model as an example to describe how to migrate it to the Ascend NPU for training. For details, see [Quick Start](docs/en/quick_start/quick_start.md).

## Feature Description

The TorchNPU plug-in provides a series of unique features in terms of memory resource optimization, communication performance optimization, computing performance optimization, and error locating assistance. For details, see the [Framework Features](docs/en/framework_feature_guide_pytorch/menu_framework_feature.md).

## API Reference

- For details about the native PyTorch APIs supported by Ascend NPUs, see the Native API.
- The TorchNPU plug-in provides some customized APIs. For details, see the [Custom API](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/en/custom_APIs/menu_Pytorch_API.md).

## Branch Maintenance Policy

For details about the maintenance policies of the TorchNPU version, see the [Maintenance Intervals for the TorchNPU](https://gitcode.com/Ascend/pytorch/blob/master/SUPPORT.md#torchnpu%E7%BB%B4%E6%8A%A4%E5%91%A8%E6%9C%9F%E8%AF%B4%E6%98%8E).

## PyTorch Version Maintenance Policy

For details about the version maintenance policy of the TorchNPU, see the [Branch Support Matrix](https://gitcode.com/Ascend/pytorch/blob/master/SUPPORT.md#%E5%88%86%E6%94%AF%E6%94%AF%E6%8C%81%E7%9F%A9%E9%98%B5).

## Contribution guidance

Describes how to contribute code to the Torch NPU plug-in library, as described in [Contribution Guide](docs/en/CONTRIBUTING.md).

## Contact us

You are welcome to contribute to the community. If you have any questions or suggestions, please submit [GitCode Issues](https://gitcode.com/Ascend/pytorch/issues) We'll get back to you as soon as we can. Thank you for your support.

## Safety Statement

For details about system security hardening, user suggestions, and file permission control for the TorchNPU, see the [Safety Statement](docs/en/SECURITYNOTE.md).

## Disclaimer

To TorchNPU plug-in users

- This plug-in is for debugging and development only. You must bear the risks and understand the following:
    
  - Data processing and deletion: The data generated during the use of this plug-in is the user's responsibility. You are advised to delete related data in a timely manner after using the data to prevent information leakage.
  - Data confidentiality and dissemination: Users understand and agree not to send or disseminate the data generated through this plug-in at will. This plug-in and its developers are not responsible for any information leakage, data leakage, or other adverse consequences arising therefrom.
  - User input security: Users must ensure the security of the entered command lines and bear any security risks or losses caused by improper input. This plug-in and its developers are not responsible for any problems caused by improper command line input.
- Scope of Disclaimer: This disclaimer applies to all individuals or entities using this plug-in. By using this plug-in, you agree to and accept the content of this statement and are willing to bear the risks and responsibilities arising from the use of this function. If you have any objection, please stop using this plug-in.
- Read and understand the disclaimer before using this tool. For any questions or questions arising from the use of this plug-in, please contact the developer.

## License

License for the TorchNPU plug-in. For details, see.LICENSEFile.

## Acknowledgment

Thank you for every PR from the community, welcome to contribute TorchNPU plug-in!
