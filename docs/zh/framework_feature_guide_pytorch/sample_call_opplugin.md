# 调用样例

完成了PyTorch框架的适配插件开发后，即可实现从PyTorch框架调用Ascend C自定义算子。下文以自定义Add算子为例，介绍PyTorch 2.7.1框架下，注册算子开发过程以及算子适配开发过程。

1.  安装指定的gcc和cmake版本，具体可参见《Ascend Extension for PyTorch 软件安装指南》中的“[（可选）安装扩展模块](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0008.html)”章节。
2.  完成自定义算子工程创建、算子开发及编译部署流程，具体可参考[Samples](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/1_add_frameworklaunch)。
3.  进行Ascend Extension for PyTorch自定义算子配置。
    1.  拉取torch\_npu仓代码，并进入仓库根目录。

        ```yaml
        git clone https://gitcode.com/ascend/pytorch.git -b v2.7.1-7.3.0 --recursive
        cd pytorch/third_party/op-plugin
        ```

        -   _2.7.1_为PyTorch版本，用户需根据实际情况指定PyTorch版本。
        -   7.3.0为Ascend Extension for PyTorch软件版本。

    2.  执行如下命令打开op\_plugin\_functions.yaml文件进行算子yaml配置。

        ```yaml
        vi op_plugin/config/op_plugin_functions.yaml
        ```

        将如下信息拷贝至op\_plugin\_functions.yaml中的custom节点下。

        ```yaml
        - func: npu_add_custom(Tensor x, Tensor y) -> Tensor 
          op_api: v2.7
        - func: npu_add_custom_backward(Tensor grad) -> (Tensor, Tensor)
          op_api: v2.7
        ```

        拷贝后的示意代码如下：

        ```yaml
        custom: 
          - func: npu_add_custom(Tensor x, Tensor y) -> Tensor 
            op_api: v2.7
          - func: npu_add_custom_backward(Tensor grad) -> (Tensor, Tensor)
            op_api: v2.7
        ```

    3.  进行结构化适配。<a id="li1813012186257"></a>

        > [!NOTE]  
        >正向接口npu\_add\_custom由于对应的适配代码结构简单，可用结构化适配自动生成适配代码，通过结构化适配的接口则无需在[4.b](#li151781548557)进行适配。

        ```yaml
        custom: 
          - func: npu_add_custom(Tensor x, Tensor y) -> Tensor 
            op_api: v2.7
            gen_opapi:
              out:
                size: x
                dtype: x
              exec: aclnnAddCustom, x, y, out
          - func: npu_add_custom_backward(Tensor grad) -> (Tensor, Tensor)
            op_api: v2.7
        ```

    4.  打开derivatives.yaml文件，进行自定义算子的前反向注册绑定。

        ```yaml
        vi op_plugin/config/derivatives.yaml
        ```

        将如下信息拷贝至derivatives.yaml文件的backward节点中。

        ```yaml
        - name: npu_add_custom(Tensor x, Tensor y) -> Tensor
          x, y: npu_add_custom_backward(grad)
          version: v2.7
        ```

        拷贝后的示意代码如下：

        ```yaml
        backward: 
        - name: npu_add_custom(Tensor x, Tensor y) -> Tensor
          x, y: npu_add_custom_backward(grad)
          version: v2.7
        ```

4.  在op\_plugin/ops/opapi目录下，创建AddCustomKernelOpApi.cpp文件并实现算子适配主体函数npu\_add\_custom和npu\_add\_custom\_backward。其核心逻辑为调用EXEC\_NPU\_CMD接口，完成输出结果的计算，EXEC\_NPU\_CMD第一个入参格式为aclnn+Optype（算子类型），之后的参数分别为输入输出。其中由于add操作的反向计算相对简单，因此不需要调用算子进行计算。
    1.  创建AddCustomKernelOpApi.cpp文件。

        ```
        vi op_plugin/ops/opapi/AddCustomKernelOpApi.cpp
        ```

    2.  完成算子适配，完整的AddCustomKernelOpApi.cpp文件如下。<a id="li151781548557"></a>

        > [!NOTE]  
        >若正向接口npu\_add\_custom已在[3.c](#li1813012186257)中通过结构化适配，则无需在该文件中适配正向接口。

        ```cpp
        #include "op_plugin/OpApiInterface.h" 
        #include "op_plugin/utils/op_api_common.h" 
        
        namespace op_api { 
        using npu_preparation = at_npu::native::OpPreparation; 
        
        // 正向接口，可选操作，已结构化适配，可无需添加
        at::Tensor npu_add_custom(const at::Tensor& x, const at::Tensor& y)
        { 
            // 构造输出tensor 
            at::Tensor result = npu_preparation::apply_tensor_without_format(x); 
            // 计算输出结果
            // 调用EXEC_NPU_CMD接口，完成输出结果的计算
            // 第一个入参格式为aclnn+Optype，之后的参数分别为输入输出
            EXEC_NPU_CMD(aclnnAddCustom, x, y, result); 
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

5.  编译Ascend Extension for PyTorch插件并安装，推荐使用容器场景进行编译，具体操作可参考《AscendExtension for PyTorch 软件安装指南》中的“[方式二：源码编译安装](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0005.html)”章节的“方式一（推荐）：容器场景”。

上述开发过程完成后，您可以调用如下的脚本test\_ops\_custom.py，测试torch\_npu.npu\_add\_custom\(\)的功能，测试脚本如下：

> [!CAUTION]  
> 注意运行该脚本的时候不要在torch\_npu仓的根目录下，否则可能会出现找不到torch\_npu.\_C的报错。

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
        print(x, '\n', y)

        output = torch_npu.npu_add_custom(x.npu(), y.npu()).cpu()

        print(output)
        self.assertRtolEqual(output, x + y)


if __name__ == "__main__":
    run_tests()
```

执行命令如下：

```
python3 test_ops_custom.py
```

输出如下打印，说明执行正确：

```coldfusion
tensor([[0.1152, 0.9385, 0.7095,  ..., 0.7500, 0.3130, 0.0044],
        [0.2759, 0.1240, 0.3550,  ..., 0.7183, 0.3540, 0.5127],
        [0.6475, 0.8037, 0.6343,  ..., 0.0840, 0.3560, 0.8677],
        ...,
        [0.7900, 0.2070, 0.7319,  ..., 0.2363, 0.2803, 0.2510],
        [0.2993, 0.3140, 0.4355,  ..., 0.8130, 0.3618, 0.5693],
        [0.3540, 0.7471, 0.9448,  ..., 0.8877, 0.8691, 0.0869]],
       dtype=torch.float16)
 tensor([[0.6689, 0.2119, 0.3105,  ..., 0.6313, 0.9546, 0.7935],
        [0.0757, 0.8447, 0.2329,  ..., 0.7256, 0.9160, 0.3975],
        [0.1968, 0.6567, 0.5322,  ..., 0.3071, 0.8501, 0.0947],
        ...,
        [0.6748, 0.4189, 0.7202,  ..., 0.0103, 0.6133, 0.3706],
        [0.1079, 0.3457, 0.7505,  ..., 0.5947, 0.4390, 0.4434],
        [0.4102, 0.1792, 0.9648,  ..., 0.6333, 0.5381, 0.6646]],
       dtype=torch.float16)
tensor([[0.7842, 1.1504, 1.0195,  ..., 1.3809, 1.2676, 0.7979],
        [0.3516, 0.9688, 0.5879,  ..., 1.4434, 1.2695, 0.9102],
        [0.8442, 1.4609, 1.1660,  ..., 0.3911, 1.2061, 0.9624],
        ...,
        [1.4648, 0.6260, 1.4521,  ..., 0.2466, 0.8936, 0.6216],
        [0.4072, 0.6597, 1.1855,  ..., 1.4082, 0.8008, 1.0127],
        [0.7642, 0.9263, 1.9102,  ..., 1.5215, 1.4072, 0.7515]],
       dtype=torch.float16)
.
----------------------------------------------------------------------
Ran 1 test in 0.669s
OK
```

