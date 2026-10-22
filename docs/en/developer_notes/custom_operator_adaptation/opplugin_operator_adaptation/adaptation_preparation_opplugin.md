# Preparation Before Adaptation

- Refer to the native PyTorch [ATen IR Definition](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme) to clarify the operator name, input parameters/return values, semantics, and other information.

- Select the operator adaptation method: prioritize `aclnn` operators (stored in `op_plugin/ops/opapi`). For compatibility requirements, you can use `aclop` operators (stored in `op_plugin/ops/aclops`).

- Before OpPlugin operator adaptation, ensure that the following environment preparations have been completed.
  
   1. Install PyTorch and TorchNPU. For details, see [Source Compilation](../../../installation_guide/references/building_from_source.md).

   2. Before adaptation, run the following commands to pull the code of the corresponding branch from the TorchNPU repository and enter the OpPlugin directory.

      ```bash
      git clone https://gitcode.com/ascend/pytorch.git -b master --recursive
      cd pytorch/third_party/op-plugin
      ```
    
   3. Before OpPlugin operator adaptation, ensure that CANN already has the corresponding operator implementation. For details, see the [CANN Operator Library](https://gitcode.com/cann/docs/blob/9.2.0/docs/en/ops-lib/0_README.md).
