# 适配说明

C++ extensions插件提供了将自定义算子映射到昇腾AI处理器的功能，为使用PyTorch框架的开发者提供了便捷的NPU算子库调用能力，基于PyTorch原生提供的自定义算子扩展功能，用户可以编译安装自定义算子wheel包并运行。本章节基于C++ extensions的方式介绍如何在昇腾NPU上完成自定义算子开发和适配。PyTorch官方的C++ extensions功能具体可参考[C++ extensions官方文档](https://pytorch.org/tutorials/advanced/cpp_extension.html)。

-   单算子API调用适配可参考[单算子API调用适配](./single_operator_adaptation.md)章节。
-   kernel直调算子适配可参考[kernel直调算子适配](./kernel_launch_operator_adaptation.md)章节。

