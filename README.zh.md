# TorchNPU插件

<p>
    简体中文 | <a href="./README.md">English</a>
</p>

## 简介

作为Ascend for PyTorch社区的核心组件，TorchNPU是昇腾专为PyTorch打造的深度学习适配插件，使PyTorch框架能够直接调用昇腾NPU，为开发者提供昇腾AI处理器的超强算力。

昇腾为基于华为昇腾处理器和软件的行业应用及服务提供全栈AI计算基础设施。您可以通过访问[昇腾社区](https://www.hiascend.com/zh/)，了解关于昇腾的更多信息。

## 目录结构

关键目录如下：

```text
├─ci                             # 持续集成脚本目录
├─cmake                          # CMake构建配置目录
├─torch_npu                      # 核心适配目录
│  ├─csrc/                       # 底层核心目录
│  ├─npu/                        # NPU接口目录
│  ├─distributed/                # 分布式训练适配目录
│  ├─asd/                        # Ascend Debug工具目录
├─docs                           # 项目文档目录
├─examples                       # 示例目录
├─torchnpugen/                   # 代码生成模块目录
└─test                           # 测试目录
```

## 版本说明

TorchNPU的版本说明包含版本配套说明、版本兼容性说明和更新说明等，具体请参见《[版本说明](docs/zh/release_notes/release_notes.md)》。

## 环境部署

TorchNPU插件的安装操作，具体请参见《[软件安装](docs/zh/installation_guide/menu_installation_guide.md)》。

## 快速入门

以CNN模型为例，介绍将其迁移至昇腾NPU上进行训练的方法，具体操作请参见《[快速入门](docs/zh/quick_start/quick_start.md)》。

## 特性介绍

TorchNPU插件从内存资源优化、通信性能优化、计算性能优化、辅助报错定位等方面精心打造了一系列独特的特性，具体特性指导请参见《[框架特性](docs/zh/framework_feature_guide_pytorch/menu_framework_feature.md)》。

## API参考

- 原生PyTorch API在昇腾NPU设备上的支持情况请参见《[原生API](docs/zh/native_apis/menu_pt_native_apis.md)》。
- TorchNPU插件提供了部分自定义API接口，具体使用请参见《[自定义API](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/zh/custom_APIs/menu_Pytorch_API.md)》。

## 分支维护策略

TorchNPU版本分支的维护策略请参考[TorchNPU维护周期说明](https://gitcode.com/Ascend/pytorch/blob/master/SUPPORT.md#torchnpu%E7%BB%B4%E6%8A%A4%E5%91%A8%E6%9C%9F%E8%AF%B4%E6%98%8E)。

## PyTorch版本维护策略

TorchNPU版本维护策略请参考[分支支持矩阵](https://gitcode.com/Ascend/pytorch/blob/master/SUPPORT.md#%E5%88%86%E6%94%AF%E6%94%AF%E6%8C%81%E7%9F%A9%E9%98%B5)。

## 贡献指导

介绍如何向TorchNPU插件库贡献代码，具体请参见[贡献指南](docs/zh/CONTRIBUTING.md)。

## 联系我们

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[GitCode Issues](https://gitcode.com/Ascend/pytorch/issues)，我们会尽快回复。感谢您的支持。

## 安全声明

TorchNPU的系统安全加固、运行用户建议和文件权限控制等内容，请参见[安全声明](docs/zh/SECURITYNOTE.md)。

## 免责声明

致TorchNPU插件使用者

- 本插件仅供调试和开发使用，使用者需自行承担使用风险，并理解以下内容：
    - 数据处理及删除：用户在使用本插件过程中产生的数据属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防信息泄露。
    - 数据保密与传播：使用者了解并同意不得将通过本插件产生的数据随意外发或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本插件及其开发者概不负责。
    - 用户输入安全性：用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于输入命令行不当所导致的问题，本插件及其开发者概不负责。
- 免责声明范围：本免责声明适用于所有使用本插件的个人或实体。使用本插件即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本插件。
- 在使用本工具之前，请谨慎阅读并理解以上免责声明的内容。对于使用本插件所产生的任何问题或疑问，请及时联系开发者。

## License

TorchNPU插件的使用许可证，具体请参见[LICENSE](LICENSE)文件。

## 致谢

感谢来自社区的每一个PR，欢迎贡献TorchNPU插件！
