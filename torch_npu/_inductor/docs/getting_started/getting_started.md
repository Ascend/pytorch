# 快速上手

## 简介

Inductor作为PyTorch编译模式的核心backend选项，可通过torch.compile(backend="inductor")直接调用。

**前置条件**：
- PyTorch 2.0+
- CANN 8.0+
- Ascend驱动 23.0+

在Ascend上使用PyTorch编译模式的Inductor后端，请按以下步骤操作：

1. **环境准备**：确保已安装Ascend驱动、CANN和pta对应版本的PyTorch
2. **环境变量配置**：设置必要的Ascend驱动和CANN环境变量
3. **验证安装**：运行简单的测试用例验证环境配置
4. **开始使用**：参考昇腾社区的PyTorch编译模式介绍手册 《[PyTorch编译模式](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/Frameworkfeatures/docs/zh/framework_feature_guide_pytorch/pytorch_compilation_mode.md)》
