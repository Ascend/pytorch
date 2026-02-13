# 适配前准备
- 参考PyTorch原生[Aten IR定义](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme)，明确算子名称、入参/返回值、语义等信息。

- 选择算子适配方式：优先选择aclnn算子（存于op_plugin/ops/opapi），兼容需求可选aclop算子（存于op_plugin/ops/aclops）。

- OpPlugin算子适配前，需确保已完成如下环境准备。
  
   1. 安装PyTorch框架，具体请参见《[Ascend Extension for PyTorch 软件安装指南](../installation_guide/menu_installation_guide.md)》。

   2. （可选）当用户使用“二进制软件包安装”或“二进制软件包安装（abi1版本）”安装torch\_npu插件时，适配前需执行如下命令拉取torch\_npu仓对应分支的代码并进入OpPlugin目录，完成torch_npu源码下载。

      ```
      git clone https://gitcode.com/ascend/pytorch.git -b v2.7.1-7.3.0 --recursive
      cd pytorch/third_party/op-plugin
      ```
    
      -   *2.7.1*为PyTorch版本，用户需根据实际情况指定PyTorch版本。
      -   *7.3.0*为Ascend Extension for PyTorch软件版本。
   3. 在OpPlugin算子适配前，请先确保CANN已有相关算子实现，具体可查询[CANN 算子库接口](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/operatorlist_00001.html)。