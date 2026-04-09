# Ascend Extension for PyTorch插件

## 简介

本项目开发了名为**torch_npu**的**Ascend Extension for PyTorch**插件，使昇腾NPU可以适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。

昇腾为基于华为昇腾处理器和软件的行业应用及服务提供全栈AI计算基础设施。您可以通过访问[昇腾社区](https://www.hiascend.com/zh/)，了解关于昇腾的更多信息。

## 版本说明

### PyTorch与Python版本配套表

| PyTorch版本     | Python版本                                                     |
|---------------|:-------------------------------------------------------------|
| PyTorch1.11.0 | Python3.7.x(>=3.7.5), Python3.8.x, Python3.9.x, Python3.10.x |
| PyTorch2.1.0  | Python3.8.x, Python3.9.x, Python3.10.x, Python 3.11.x        |
| PyTorch2.2.0  | Python3.8.x, Python3.9.x, Python3.10.x                       |
| PyTorch2.3.1  | Python3.8.x, Python3.9.x, Python3.10.x, Python 3.11.x        |
| PyTorch2.4.0  | Python3.8.x, Python3.9.x, Python3.10.x, Python 3.11.x        |
| PyTorch2.5.1  | Python3.9.x, Python3.10.x, Python 3.11.x                     |
| PyTorch2.6.0  | Python3.9.x, Python3.10.x, Python 3.11.x                     |
| PyTorch2.7.1  | Python3.9.x, Python3.10.x, Python 3.11.x                     |
| PyTorch2.8.0  | Python3.9.x, Python3.10.x, Python 3.11.x                     |

### 昇腾辅助软件

**Ascend Extension for PyTorch**的分支名称采用`{PyTorch版本}-{昇腾版本}`命名规则，前者为**Ascend Extension for PyTorch**匹配的PyTorch版本，后者为**Ascend Extension for PyTorch**版本号，详细匹配如下：

| CANN版本                | 支持的PyTorch版本 | 支持的Extension版本   | GitCode分支         | 
|-----------------------|--------------|------------------|-------------------|
| CANN 8.5.0            | 2.9.0        | 2.9.0            | v2.9.0-7.3.0      |
|                       | 2.8.0        | 2.8.0.post2      | v2.8.0-7.3.0      |
|                       | 2.7.1        | 2.7.1.post2      | v2.7.1-7.3.0      |
|                       | 2.6.0        | 2.6.0.post5      | v2.6.0-7.3.0      |
| CANN 8.3.RC1          | 2.8.0        | 2.8.0            | v2.8.0-7.2.0      |
|                       | 2.7.1        | 2.7.1            | v2.7.1-7.2.0      |
|                       | 2.6.0        | 2.6.0.post3      | v2.6.0-7.2.0      |
|                       | 2.1.0        | 2.1.0.post17     | v2.1.0-7.2.0      |
| CANN 8.2.RC1          | 2.6.0        | 2.6.0            | v2.6.0-7.1.0      |
|                       | 2.5.1        | 2.5.1.post1      | v2.5.1-7.1.0      |
|                       | 2.1.0        | 2.1.0.post13     | v2.1.0-7.1.0      |
| CANN 8.1.RC1          | 2.5.1        | 2.5.1            | v2.5.1-7.0.0      |
|                       | 2.4.0        | 2.4.0.post4      | v2.4.0-7.0.0      |
|                       | 2.3.1        | 2.3.1.post6      | v2.3.1-7.0.0      |
|                       | 2.1.0        | 2.1.0.post12     | v2.1.0-7.0.0      |
| CANN 8.0.0            | 2.4.0        | 2.4.0.post2      | v2.4.0-6.0.0      |
|                       | 2.3.1        | 2.3.1.post4      | v2.3.1-6.0.0      |
|                       | 2.1.0        | 2.1.0.post10     | v2.1.0-6.0.0      |
| CANN 8.0.RC3          | 2.4.0        | 2.4.0            | v2.4.0-6.0.rc3    |
|                       | 2.3.1        | 2.3.1.post2      | v2.3.1-6.0.rc3    |
|                       | 2.1.0        | 2.1.0.post8      | v2.1.0-6.0.rc3    | 
| CANN 8.0.RC2          | 2.3.1        | 2.3.1            | v2.3.1-6.0.rc2    | 
|                       | 2.2.0        | 2.2.0.post2      | v2.2.0-6.0.rc2    |
|                       | 2.1.0        | 2.1.0.post6      | v2.1.0-6.0.rc2    |
|                       | 1.11.0       | 1.11.0.post14    | v1.11.0-6.0.rc2   |
| CANN 8.0.RC1          | 2.2.0        | 2.2.0            | v2.2.0-6.0.rc1    |
|                       | 2.1.0        | 2.1.0.post4      | v2.1.0-6.0.rc1    | 
|                       | 1.11.0       | 1.11.0.post11    | v1.11.0-6.0.rc1   | 
| CANN 7.0.0            | 2.1.0        | 2.1.0            | v2.1.0-5.0.0      |
|                       | 2.0.1        | 2.0.1.post1      | v2.0.1-5.0.0      | 
|                       | 1.11.0       | 1.11.0.post8     | v1.11.0-5.0.0     | 
| CANN 7.0.RC1          | 2.1.0        | 2.1.0.rc1        | v2.1.0-5.0.rc3    | 
|                       | 2.0.1        | 2.0.1            | v2.0.1-5.0.rc3    | 
|                       | 1.11.0       | 1.11.0.post4     | v1.11.0-5.0.rc3   | 
| CANN 6.3.RC3.1        | 1.11.0       | 1.11.0.post3     | v1.11.0-5.0.rc2.2 | 
| CANN 6.3.RC3          | 1.11.0       | 1.11.0.post2     | v1.11.0-5.0.rc2.1 | 
| CANN 6.3.RC2          | 2.0.1        | 2.0.1.rc1        | v2.0.1-5.0.rc2    | 
|                       | 1.11.0       | 1.11.0.post1     | v1.11.0-5.0.rc2   |
|                       | 1.8.1        | 1.8.1.post2      | v1.8.1-5.0.rc2    |
| CANN 6.3.RC1          | 1.11.0       | 1.11.0           | v1.11.0-5.0.rc1   | 
|                       | 1.8.1        | 1.8.1.post1      | v1.8.1-5.0.rc1    | 
| CANN 6.0.1            | 1.5.0        | 1.5.0.post8      | v1.5.0-3.0.0      |
|                       | 1.8.1        | 1.8.1            | v1.8.1-3.0.0      |
|                       | 1.11.0       | 1.11.0.rc2（beta) | v1.11.0-3.0.0     | 
| CANN 6.0.RC1          | 1.5.0        | 1.5.0.post7      | v1.5.0-3.0.rc3    |
|                       | 1.8.1        | 1.8.1.rc3        | v1.8.1-3.0.rc3    |
|                       | 1.11.0       | 1.11.0.rc1（beta) | v1.11.0-3.0.rc3   | 
| CANN 5.1.RC2          | 1.5.0        | 1.5.0.post6      | v1.5.0-3.0.rc2    |
|                       | 1.8.1        | 1.8.1.rc2        | v1.8.1-3.0.rc2    |
| CANN 5.1.RC1          | 1.5.0        | 1.5.0.post5      | v1.5.0-3.0.rc1    |
|                       | 1.8.1        | 1.8.1.rc1        | v1.8.1-3.0.rc1    | 
| CANN 5.0.4            | 1.5.0        | 1.5.0.post4      | 2.0.4.tr5         |
| CANN 5.0.3            | 1.8.1        | 1.5.0.post3      | 2.0.3.tr5         |
| CANN 5.0.2            | 1.5.0        | 1.5.0.post2      | 2.0.2.tr5         |


## 环境部署

Ascend Extension for PyTorch插件的安装操作，具体请参见《[Ascend Extension for PyTorch 软件安装](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/installation_guide/menu_installation_guide.md)》。

## 快速入门

以CNN模型为例，介绍将其迁移至昇腾NPU上进行训练的方法，具体操作请参见《[Ascend Extension for PyTorch 快速入门](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/quick_start/quick_start.md)》。

## 特性介绍
 	 
Ascend Extension for PyTorch插件从内存资源优化、通信性能优化、计算性能优化、辅助报错定位等方面精心打造了一系列独特的特性，具体特性指导请参见《[PyTorch 框架特性指南](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/framework_feature_guide_pytorch/menu_framework_feature.md)》。
 	 
## API参考

- 原生PyTorch API在昇腾NPU设备上的支持情况请参见《[PyTorch 原生API支持度](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/native_apis/menu_pt_native_apis.md)》。
- Ascend Extension for PyTorch插件提供了部分自定义API接口，具体使用请参见《[Ascend Extension for PyTorch自定义API](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/menu_Pytorch_API.md)》。

## 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[GitCode Issues](https://gitcode.com/Ascend/pytorch/issues)，我们会尽快回复。感谢您的支持。

## 分支维护策略

Ascend Extension for PyTorch版本分支的维护阶段如下：


| **状态**            | **时间** | **说明**                                         |
| ------------------- | -------- | ------------------------------------------------ |
| 计划                | 1—3 个月 | 计划特性                                         |
| 开发                | 6—12 个月   | 开发新特性并修复问题，定期发布新版本。针对不同的PyTorch版本采取不同的策略，常规分支的开发周期分别为6个月，长期支持分支的开发周期为12个月 |
| 维护                |  1年/3.5年 | 常规分支维护1年,长期支持分支维护3.5年。对重大BUG进行修复，不合入新特性，并视BUG的影响发布补丁版本 |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                             |

## PyTorch版本维护策略

| **PyTorch版本** | **维护策略** | **当前状态** | **发布时间**   | **后续状态**             | **EOL日期** |
|---------------|----------|----------|------------|----------------------|-----------|
| 2.9.0         | 常规分支     | 开发       | 2026/01/15| 预计2026/07/15起进入维护状态        | -          | 
| 2.8.0         | 常规分支     | 开发       | 2025/10/15| 预计2026/03/15起进入维护状态        | -          | 
| 2.7.1         | 长期分支     | 开发       |  2025/10/15| 预计2026/10/15起进入维护状态       | -          | 
| 2.6.0         | 常规分支     | 开发       | 2025/07/25 | 预计2026/01/25起进入维护状态       | -          | 
| 2.5.1         | 常规分支     | 维护       | 2024/11/08 | 预计2026/08/08起进入无维护状态     | -          | 
| 2.4.0         | 常规分支     | 维护       | 2024/10/15 | 预计2026/06/15起进入无维护状态     | -          | 
| 2.3.1         | 常规分支     | 维护       | 2024/06/06 | 预计2026/06/07起进入无维护状态     |            |
| 2.2.0         | 常规分支     | EOL        | 2024/04/01 |                                  | 2025/10/14 |
| 2.1.0         | 长期支持     | 维护       | 2023/10/15 | 预计2026/12/30起进入无维护状态     |            |
| 2.0.1         | 常规分支     | EOL        | 2023/7/19  |                                  | 2024/3/14  |
| 1.11.0        | 长期支持     | EOL        | 2023/4/19  |                                  | 2025/10/25 |
| 1.8.1         | 长期支持     | EOL        | 2022/4/10  |                                  | 2023/4/10 |
| 1.5.0         | 长期支持     | EOL        | 2021/7/29  |                                  | 2022/7/29 |

## 贡献指导
介绍如何向Ascend Extension for PyTorch插件库贡献代码，具体请参见[Ascend Extension for PyTorch插件 贡献指南](CONTRIBUTING.md)。

## 联系我们

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[GitCode Issues](https://gitcode.com/Ascend/pytorch/issues)，我们会尽快回复。感谢您的支持。

## 安全声明
Ascend Extension for PyTorch的系统安全加固、运行用户建议和文件权限控制等内容，请参见[Ascend Extension for PyTorch插件 安全声明](SECURITYNOTE.md)。

## 免责声明
致Ascend Extension for PyTorch插件使用者
- 本插件仅供调试和开发使用，使用者需自行承担使用风险，并理解以下内容：
    - 数据处理及删除：用户在使用本插件过程中产生的数据属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防信息泄露。
    - 数据保密与传播：使用者了解并同意不得将通过本插件产生的数据随意外发或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本插件及其开发者概不负责。
    - 用户输入安全性：用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于输入命令行不当所导致的问题，本插件及其开发者概不负责。
- 免责声明范围：本免责声明适用于所有使用本插件的个人或实体。使用本插件即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本插件。
- 在使用本工具之前，请谨慎阅读并理解以上免责声明的内容。对于使用本插件所产生的任何问题或疑问，请及时联系开发者。

## License

Ascend Extension for PyTorch插件的使用许可证，具体请参见[LICENSE](LICENSE)文件。

## 致谢

感谢来自社区的每一个PR，欢迎贡献Ascend Extension for PyTorch插件！
