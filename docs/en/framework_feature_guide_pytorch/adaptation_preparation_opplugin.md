# Preparation Before Adaptation

- Refer to the PyTorch native [Aten IR definition](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme) to clarify information such as the operator name, input parameters/return values, and semantics.

- Select the operator adaptation method: prioritize `aclnn` operators (stored in `op_plugin/ops/opapi`). For compatibility requirements, you can use `aclop` operators (stored in `op_plugin/ops/aclops`).

- Before OpPlugin operator adaptation, ensure that the following environment preparations have been completed.
  
   1. Install the PyTorch framework. For details, see the[Software Installation](../installation_guide/menu_installation_guide.md).

   2. Before adaptation, run the following commands to pull the code of the corresponding branch from the TorchNPU repository and enter the OpPlugin directory.

      ```bash
      git clone https://gitcode.com/ascend/pytorch.git -b v2.7.1-26.1.0 --recursive
      cd pytorch/third_party/op-plugin
      ```
    
      - *2.7.1* is the PyTorch version. Specify the version based on the actual situation.
      - *26.1.0* is the TorchNPU software version.
   3. Before OpPlugin operator adaptation, ensure that CANN already has the corresponding operator implementation. For details, see the [CANN Operator Library](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/operatorlist_00001.html).
