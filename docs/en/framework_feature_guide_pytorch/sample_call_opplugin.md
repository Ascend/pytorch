# Calling Example

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-08T10:22:15.960Z pushedAt=2026-07-08T10:47:16.865Z -->

After completing the development of the adaptation plugin for the PyTorch framework, you can call Ascend C custom operators from the PyTorch framework. The following uses the custom npu_add_custom operator as an example to introduce the operator adaptation development process under the PyTorch 2.7.1 framework.

## Preparation Before Adaptation

1. Install the PyTorch framework. For details, see *[Ascend Extension for PyTorch Software Installation Guide](../installation_guide/menu_installation_guide.md)*.

2. (Optional) If you installed the torch_npu plugin using "Binary Package Installation" or "Binary Package Installation (abi1 version)", you need to run the following commands before adaptation to pull the code of the corresponding branch from the torch_npu repository and enter the OpPlugin directory to download the torch_npu source code.

    ```bash
    git clone https://gitcode.com/ascend/pytorch.git -b v2.7.1-26.0.0 --recursive
    cd pytorch/third_party/op-plugin
    ```

    - *2.7.1* is the PyTorch version. Users should specify the PyTorch version according to their actual environment.
    - *26.0.0* is the Ascend Extension for PyTorch software version.

3. Before adapting the framework operator, ensure that the corresponding operator has already been implemented in CANN. For details, refer to the [CANN Operator Library](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/aolapi/operatorlist_00001.html).
    > [!NOTE]  
    >
    > The corresponding CANN operator for this example is aclnnAdd. Refer to the [operator interface](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/aolapi/context/ops-math/aclnnAdd&aclnnInplaceAdd.md) in the [CANN Operator Library].

## Adaptation Development

1. Configure the YAML file for the custom operator.
   1. Run the following command to open the op_plugin_functions.yaml file for operator YAML configuration.

        ```bash
        vi op_plugin/config/op_plugin_functions.yaml
        ```

        Copy the following information to the custom node in op_plugin_functions.yaml.

        ```yaml
        - func: npu_add_custom(Tensor x, Tensor y, *, Scalar alpha=1) -> Tensor 
          op_api: v2.7
        - func: npu_add_custom_backward(Tensor grad) -> (Tensor, Tensor)
          op_api: v2.7
        ```

        The copied sample code is as follows:

        ```yaml
        custom: 
          - func: npu_add_custom(Tensor x, Tensor y, *, Scalar alpha=1) -> Tensor 
            op_api: v2.7
          - func: npu_add_custom_backward(Tensor grad) -> (Tensor, Tensor)
            op_api: v2.7
        ```

   2. Open the derivatives.yaml file to register and bind the forward and backward passes of the custom operator.

        ```yaml
        vi op_plugin/config/derivatives.yaml
        ```

        Copy the following information into the backward node of the derivatives.yaml file.

        ```yaml
        - name: npu_add_custom(Tensor x, Tensor y, *, Scalar alpha=1) -> Tensor
          x, y: npu_add_custom_backward(grad)
          version: v2.7
        ```

        The copied sample code is as follows:

        ```yaml
        backward: 
        - name: npu_add_custom(Tensor x, Tensor y, *, Scalar alpha=1) -> Tensor
          x, y: npu_add_custom_backward(grad)
          version: v2.7
        ```

2. The following two methods are provided for code adaptation of custom operators. Users may choose based on their actual situation.
   - Perform structured operator adaptation in op_plugin_functions.yaml.

      > [!NOTE]  
      > npu_add_custom is a forward operator interface. Since the corresponding adaptation code has a simple structure, structured adaptation can be used to automatically generate the adaptation code.

        ```yaml
        custom: 
          - func: npu_add_custom(Tensor x, Tensor y, *, Scalar alpha=1) -> Tensor 
            op_api: v2.7
            gen_opapi:
              out:
                size: x
                dtype: x
              exec: aclnnAdd, x, y, alpha, out
          - func: npu_add_custom_backward(Tensor grad) -> (Tensor, Tensor)
            op_api: v2.7
        ```

- General operator adaptation.

1. In the `op_plugin/ops/opapi` directory, create the `AddCustomKernelNpuOpApi.cpp` file and implement the main operator adaptation functions `npu_add_custom` and `npu_add_custom_backward`. The core logic is to call the `EXEC_NPU_CMD` interface to compute the output results. The first argument of `EXEC_NPU_CMD` follows the format `aclnn+Optype` (operator type), and the subsequent arguments are the inputs and outputs. Since the backward computation of the add operation is relatively simple, there is no need to call an operator for the computation.

    ```bash
    vi op_plugin/ops/opapi/AddCustomKernelNpuOpApi.cpp
    ```

2. After completing the operator adaptation, the full `AddCustomKernelNpuOpApi.cpp` file is as follows.

    ```cpp
    #include "op_plugin/OpApiInterface.h" 
    #include "op_plugin/utils/op_api_common.h" 
          
    namespace op_api { 
    using npu_preparation = at_npu::native::OpPreparation;
          
    // Forward interface
    at::Tensor npu_add_custom(const at::Tensor& x, const at::Tensor& y, const at::Scalar &alpha)
    { 
        // Construct output tensor
        at::Tensor result = npu_preparation::apply_tensor_without_format(x);
        // Compute the output result
        // Call the EXEC_NPU_CMD interface to complete the computation of the output result
        // The first argument follows the format aclnn+Optype, and the subsequent arguments are inputs and outputs respectively
        EXEC_NPU_CMD(aclnnAdd, x, y, alpha, result); 
        return result; 
    }
          
    // Backward interface
    std::tuple<at::Tensor, at::Tensor> npu_add_custom_backward(const at::Tensor& grad)
    {
        // Construct the output tensor
        at::Tensor result = npu_preparation::apply_tensor_without_format(grad);
        result.copy_(grad);
        // Compute the output result
        return {result, result};
    }
    }  // namespace op_api
    ```

3. Operator auxiliary adaptation implementation.

   1. Add the operator interface documentation in the following file.

      In codegen/templates/\_op\_plugin\_docs.py, add the following content:

      ```python
      _add_torch_npu_docstr(
          "npu_add_custom",
          """
      torch_npu.npu_add_custom(self, other, alpha=1) -> Tensor

      Description
      Performs a custom addition operation on two tensors, supporting scaling of the second input tensor by a coefficient before addition. Supports FakeTensor mode.

      Parameters
      self (Tensor) - The first input tensor.
      other (Tensor) - The second input tensor, must be broadcastable with self.
      alpha (float, default 1) - The scaling coefficient applied to other. The computation is self + alpha * other.

      Example
      >>> x = torch.tensor([1.0, 2.0, 3.0]).npu()
      >>> y = torch.tensor([4.0, 5.0, 6.0]).npu()
      >>> result = torch_npu.npu_add_custom(x, y, alpha=0.5)
      >>> result
      tensor([3.0, 4.5, 6.0], device='npu:0')
      """
      )
      ```

   2. The public operator interface must be configured in the following documentation.

      - Add the following entry in `test/allowlist_for_publicAPI.json`:

          ```json
          {
          "torch_npu": 
            {
              "v2.7": ["npu_add_custom", "npu_add_custom_backward"]
            }
          }
          ```

      - Add the following entry in `test/core_tests/torch_npu_OpApi_schema_all.json`:

          ```json
          {
            "op_api: torch_npu.npu_add_custom(*args, **kwargs)": {
              "version": ["v2.7"]
            },
            "func: npu_add_custom(Tensor self, Tensor other, float alpha=1) -> Tensor": {
              "version": ["v2.7"]
            }
          }
          ```

   3. Register the meta implementation of the operator interface in the following file.

      Add the following entry in `op_plugin/python/meta/_meta_registrations.py`:

        ```python
        @impl(m, "npu_add_custom")
        def npu_add_custom_meta(self, other, alpha=1):
            # Simulate the computation logic of the custom addition, used for FakeTensor mode
            output = self + alpha * other
            return torch.empty_like(output, dtype=self.dtype)
        ```

## Compilation and Verification

1. Compile and install the Ascend Extension for PyTorch plugin. It is recommended to compile in a container scenario. For details, refer to the "Method 1 (Recommended): Container Scenario" section in the "[Method 2: Source Code Compilation and Installation](../installation_guide/compilation_installation_using_source_code.md)" chapter of the *Ascend Extension for PyTorch Software Installation Guide*.

2. After completing the above development process, call the developer test script to verify whether the basic functionality is normal.

   1. In the test/test\_custom\_ops directory, create a new developer test file test\_npu\_add\_custom.py and add the following content:

      ```Python
      import torch
      import torch_npu
      from torch_npu.testing.testcase import TestCase, run_tests

      torch.npu.config.allow_internal_format = False
      torch.npu.set_compile_mode(jit_compile=False)

      class TestCustomAdd(TestCase):

          def test_add_custom(self):
              length = [8, 2048]
              x = torch.rand(length, device='cpu', dtype=torch.float16)
              y = torch.rand(length, device='cpu', dtype=torch.float16)

              output = torch_npu.npu_add_custom(x.npu(), y.npu()).cpu()
              self.assertRtolEqual(output, x + y)

          def test_add_custom_backward(self):
              length = [8, 2048]
              x = torch.rand(length, device='cpu', dtype=torch.float16, requires_grad=True)
              y = torch.rand(length, device='cpu', dtype=torch.float16, requires_grad=True)

              output = torch_npu.npu_add_custom(x.npu(), y.npu()).cpu()
              grad_output = torch.rand(length, device='cpu', dtype=torch.float16)

              x_grad, y_grad = torch_npu.npu_add_custom_backward(grad_output.npu())
              self.assertRtolEqual(x_grad.cpu(), grad_output)
              self.assertRtolEqual(y_grad.cpu(), grad_output)

      if __name__ == "__main__":
         run_tests()
       ```

   2. Run the following command to verify whether the new operator functions correctly:
      > [!CAUTION]  
      >
      > Do not run this script in the root directory of the torch_npu repository, otherwise an error indicating that torch_npu._C cannot be found may occur.

      ```bash
      python op-plugin/test/test_custom_ops/test_npu_add_custom.py -v
      ```

      The following output indicates successful execution:

      ```text
      test_add_custom (__main__.TestCustomAdd) ... ok
      test_add_custom_backward (__main__.TestCustomAdd) ... ok

      ----------------------------------------------------------------------
      Ran 2 test in 1.199s

      OK
      ```
