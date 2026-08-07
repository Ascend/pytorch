# 适配前准备

- 参考PyTorch原生[ATen IR定义](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme)，明确算子名称、入参/返回值、语义等信息。

- 选择算子适配方式：优先选择aclnn算子（存于op_plugin/ops/opapi），兼容需求可选aclop算子（存于op_plugin/ops/aclops）。

- OpPlugin算子适配前，需确保已完成如下环境准备。
  
   1. 安装PyTorch和TorchNPU，具体请参见[源码编译](../../../installation_guide/references/building_from_source.md)。

   2. 适配前需执行如下命令拉取TorchNPU仓对应分支的代码并进入OpPlugin目录。

      ```bash
      git clone https://gitcode.com/ascend/pytorch.git -b master --recursive
      cd pytorch/third_party/op-plugin
      ```
    
   3. 在OpPlugin算子适配前，请先确保CANN已有相关算子实现，具体可查询[CANN 算子库](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/aolapi/operatorlist_00001.html)。
