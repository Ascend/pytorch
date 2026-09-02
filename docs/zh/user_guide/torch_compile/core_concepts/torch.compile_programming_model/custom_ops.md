# 自定义算子

**概要：**

- 使用自定义算子可让`torch.compile`将函数视为不透明对象。`torch.compile`永远不会追踪函数内部，Inductor（后端）会按原样运行该函数。

在以下任一情况下，可能需要使用自定义算子：

- 代码调用了某些C/C++ 代码。Dynamo是一个Python字节码解释器，通常不知道如何处理对绑定到Python的C/C++ 函数的调用。
- Dynamo和非严格追踪难以追踪某个函数，而您希望`torch.compile`忽略该函数。

有关如何将Python函数封装成`torch.compile`能够识别的自定义算子，请参考[Python自定义算子教程](https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial)。

对于更高级的用例，可能需要使用C++ 自定义算子API；有关详细信息，请参考[此处](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)。
