# 调用样例

完成了PyTorch框架的适配插件开发后，即可实现从PyTorch框架调用Ascend C自定义算子。下文以自定义npu\_add\_custom算子为例，介绍PyTorch 2.7.1框架下，算子适配开发过程。

## 适配前准备

1. 安装PyTorch框架，具体请参见《[Ascend Extension for PyTorch 软件安装指南](../installation_guide/menu_installation_guide.md)》。

2. （可选）当用户使用“二进制软件包安装”或“二进制软件包安装（abi1版本）”安装torch\_npu插件时，适配前需执行如下命令拉取torch\_npu仓对应分支的代码并进入OpPlugin目录，完成torch_npu源码下载。

    ```
    git clone https://gitcode.com/ascend/pytorch.git -b v2.7.1-7.3.0 --recursive
    cd pytorch/third_party/op-plugin
    ```

    -   *2.7.1*为PyTorch版本，用户需根据实际情况指定PyTorch版本。
    -   *7.3.0*为Ascend Extension for PyTorch软件版本。

3. 在框架算子适配前，请先确保CANN已有相关算子实现，具体可查询[CANN 算子库接口](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/operatorlist_00001.html)。
    > [!NOTE]  
    > 本示例对应CANN算子为aclnnAdd，可参考[CANN 算子库接口]中[NN算子接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/aolapi/context/aclnnAdd&aclnnInplaceAdd.md)。

## 适配开发
1. 对自定义算子进行yaml配置。
   1.  执行如下命令打开op\_plugin\_functions.yaml文件进行算子yaml配置。

        ```yaml
        vi op_plugin/config/op_plugin_functions.yaml
        ```

        将如下信息拷贝至op\_plugin\_functions.yaml中的custom节点下。

        ```yaml
        - func: npu_add_custom(Tensor x, Tensor y, *, Scalar alpha=1) -> Tensor 
          op_api: v2.7
        - func: npu_add_custom_backward(Tensor grad) -> (Tensor, Tensor)
          op_api: v2.7
        ```

        拷贝后的示意代码如下：

        ```yaml
        custom: 
          - func: npu_add_custom(Tensor x, Tensor y, *, Scalar alpha=1) -> Tensor 
            op_api: v2.7
          - func: npu_add_custom_backward(Tensor grad) -> (Tensor, Tensor)
            op_api: v2.7
        ```

   2.  打开derivatives.yaml文件，进行自定义算子的前反向注册绑定。

        ```yaml
        vi op_plugin/config/derivatives.yaml
        ```

        将如下信息拷贝至derivatives.yaml文件的backward节点中。

        ```yaml
        - name: npu_add_custom(Tensor x, Tensor y, *, Scalar alpha=1) -> Tensor
          x, y: npu_add_custom_backward(grad)
          version: v2.7
        ```

        拷贝后的示意代码如下：

        ```yaml
        backward: 
        - name: npu_add_custom(Tensor x, Tensor y, *, Scalar alpha=1) -> Tensor
          x, y: npu_add_custom_backward(grad)
          version: v2.7
        ```

2. 提供以下两种方法对自定义算子进行代码适配，用户可根据实际情况进行选择。
   - 在op\_plugin\_functions.yaml中进行结构化算子适配。

      > [!NOTE]  
      > npu\_add\_custom为正向接口，由于对应的适配代码结构简单，可用结构化适配自动生成适配代码。

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
   -  通用化算子适配。
      1.  在op_plugin/ops/opapi目录下，创建AddCustomKernelNpuOpApi.cpp文件并实现算子适配主体函数npu\_add\_custom和npu\_add\_custom\_backward。其核心逻辑为调用EXEC\_NPU\_CMD接口，完成输出结果的计算，EXEC\_NPU\_CMD第一个入参格式为aclnn+Optype（算子类型），之后的参数分别为输入输出。其中由于add操作的反向计算相对简单，因此不需要调用算子进行计算。

          ```
          vi op_plugin/ops/opapi/AddCustomKernelNpuOpApi.cpp
          ```

      2.  完成算子适配，完整的AddCustomKernelNpuOpApi.cpp文件如下。

          ```cpp
          #include "op_plugin/OpApiInterface.h" 
          #include "op_plugin/utils/op_api_common.h" 
          
          namespace op_api { 
          using npu_preparation = at_npu::native::OpPreparation;
          
          // 正向接口
          at::Tensor npu_add_custom(const at::Tensor& x, const at::Tensor& y, const at::Scalar &alpha)
          { 
              // 构造输出tensor 
              at::Tensor result = npu_preparation::apply_tensor_without_format(x);
              // 计算输出结果
              // 调用EXEC_NPU_CMD接口，完成输出结果的计算
              // 第一个入参格式为aclnn+Optype，之后的参数分别为输入输出
              EXEC_NPU_CMD(aclnnAdd, x, y, alpha, result); 
              return result; 
          }
          
          // 反向接口
          std::tuple<at::Tensor, at::Tensor> npu_add_custom_backward(const at::Tensor& grad)
          {
              // 构造输出tensor
              at::Tensor result = npu_preparation::apply_tensor_without_format(grad);
              result.copy_(grad);
              // 计算输出结果
              return {result, result};
          }
          }  // namespace op_api
          ```

3. 算子辅助适配实现。

   1. 在如下文件中补充算子接口说明文档。

      在codegen/templates/\_op\_plugin\_docs.py中补充如下内容：

      ```python
      _add_torch_npu_docstr(
          "npu_add_custom",
          """
      torch_npu.npu_add_custom(self, other, alpha=1) -> Tensor

      功能描述
      对两个张量执行自定义加法运算，支持对第二个输入张量进行系数缩放后再相加。支持FakeTensor模式。

      参数说明
      self (Tensor) - 第一个输入张量。
      other (Tensor) - 第二个输入张量，需与self的形状可广播。
      alpha (float，默认值为1) - 应用于other的缩放系数，计算方式为self + alpha * other。

      示例
      >>> x = torch.tensor([1.0, 2.0, 3.0]).npu()
      >>> y = torch.tensor([4.0, 5.0, 6.0]).npu()
      >>> result = torch_npu.npu_add_custom(x, y, alpha=0.5)
      >>> result
      tensor([3.0, 4.5, 6.0], device='npu:0')
      """
      )
      ```

   2. 对外公开接口需在如下文档中进行配置。

      - 在test/allowlist\_for\_publicAPI.json中新增：

          ```json
          {
          "torch_npu": 
            {
              "v2.7": ["npu_add_custom", "npu_add_custom_backward"]
            }
          }
          ```

      - 在test/core\_tests/torch\_npu\_OpApi\_schema\_all.json中新增：

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

   3. 在如下文件中注册算子接口的meta实现。

      在op\_plugin/python/meta/\_meta\_registrations.py中新增：

        ```python
        @impl(m, "npu_add_custom")
        def npu_add_custom_meta(self, other, alpha=1):
            # 模拟自定义加法的计算逻辑，用于FakeTensor模式
            output = self + alpha * other
            return torch.empty_like(output, dtype=self.dtype)
        ```


## 编译验证
1. 编译Ascend Extension for PyTorch插件并安装，推荐使用容器场景进行编译，具体操作可参考《AscendExtension for PyTorch 软件安装指南》中的“[方式二：源码编译安装](../installation_guide/compilation_installation_using_source_code.md)”章节的“方式一（推荐）：容器场景”。


2. 上述开发过程完成后，调用开发者测试脚本，验证基本功能是否正常。

   1. 在test/test\_custom\_ops目录下，新增开发者测试文件test\_npu\_add\_custom.py，新增如下内容：
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

   2. 执行命令如下验证新增算子功能是否正常：
      > [!CAUTION]  
      > 注意运行该脚本的时候不要在torch\_npu仓的根目录下，否则可能会出现找不到torch\_npu.\_C的报错。

      ```
      python op-plugin/test/test_custom_ops/test_npu_add_custom.py -v
      ```

      输出如下打印，说明执行正确：

      ```coldfusion
      test_add_custom (__main__.TestCustomAdd) ... ok
      test_add_custom_backward (__main__.TestCustomAdd) ... ok

      ----------------------------------------------------------------------
      Ran 2 test in 1.199s

      OK
      ```

