# Preparation Before Adaptation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T07:48:38.441Z pushedAt=2026-06-15T12:00:44.057Z -->

- Refer to the PyTorch native [Aten IR definition](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme) to clarify information such as the operator name, input parameters/return values, and semantics.

- Select the operator adaptation method: prioritize aclnn operators (stored in op_plugin/ops/opapi); for compatibility requirements, aclop operators (stored in op_plugin/ops/aclops) can be used.

- Before OpPlugin operator adaptation, ensure that the following environment preparations have been completed.

   1. Install the PyTorch framework. For details, see the [Ascend Extension for PyTorch Software Installation Guide](../installation_guide/menu_installation_guide.md).

   2. (Optional) When you install the torch_npu plugin using "Binary Package Installation" or "Binary Package Installation (abi1 version)", before adaptation, run the following command to pull the code of the corresponding branch of the torch_npu repository and enter the OpPlugin directory to download the torch_npu source code.

      ```bash
      git clone https://gitcode.com/ascend/pytorch.git -b v2.7.1-26.0.0 --recursive
      cd pytorch/third_party/op-plugin
      ```

      - *2.7.1* is the PyTorch version. Users need to specify the PyTorch version based on the actual situation.
      - *26.0.0* is the Ascend Extension for PyTorch software version.
   3. Before OpPlugin operator adaptation, ensure that the corresponding operators are already implemented in CANN. For details, see [CANN Operator Library](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/aolapi/operatorlist_00001.html).
