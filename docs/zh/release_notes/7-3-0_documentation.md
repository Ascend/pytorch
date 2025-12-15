# 7.3.0版本配套文档

|文档名称|内容简介|更新说明|
|--|--|--|
|《Ascend Extension for PyTorch 软件安装指南》|提供在昇腾设备安装PyTorch框架训练环境，以及升级、卸载等操作。|- 新增适配PyTorch 2.9.0。<br>- 新增支持Python 3.12。|
|《PyTorch 训练模型迁移调优指南》|包含模型的迁移及调优、精度问题定位、性能问题解决等指导，并提供了常用模型案例库。|新增适配PyTorch 2.9.0。|
|《PyTorch 框架特性指南》|基于Ascend Extension for PyTorch提供昇腾AI处理器的超强算力，从内存优化、报错定位、高性能计算等方面打造一系列独有特性。|- 新增“Stream级TaskQueue并行下发”特性。<br>- 新增“PyTorch编译模式（torch.compile）”特性。|
|《PyTorch 图模式使用指南(TorchAir)》|作为昇腾Ascend Extension for PyTorch的图模式能力扩展库，提供昇腾设备亲和的torch.compile图模式后端，实现PyTorch网络在昇腾NPU上的图模式推理加速和优化。|- 增强基础功能，包括完整Debug信息Dump、自定义FX Pass等。<br>- 增强aclgraph功能，包括支持Stream级控、内存复用、FX pass配置等。<br>- 增强GE功能，包括算子不超时配置等。|
|《Ascend Extension for PyTorch 自定义API参考》|提供Ascend Extension for PyTorch自定义API的函数原型、功能说明、参数说明与调用示例等。|- 新增适配PyTorch 2.9.0。<br>- 具体接口变更请参考[接口变更说明](api_changes.md)。|
|《PyTorch 原生API支持度》|提供PyTorch 2.9.0/2.8.0/2.7.1/2.6.0版本原生API在昇腾设备上的支持情况。|新增PyTorch 2.9.0原生API支持清单。|
|《套件与三方库支持清单》|介绍昇腾设备支持的模型套件与加速库、昇腾已原生支持的第三方库和昇腾自研插件。|新增原生指定的第三方库ms-swift。|
|《环境变量参考》|在Ascend Extension for PyTorch训练和在线推理过程中可使用的环境变量。|- 新增“PER_STREAM_QUEUE”。<br>- 新增“MULTI_STREAM_MEMORY_REUSE”。|


