# 常见参考

## PyTorch scheme规则<a id="section001">
官方schema(算子描述规范)指导可参见[LINK](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md)。

由于PTA 2.1版本使用官方的torchgen进行代码生成，因此要遵循官方的一些生成规范，未满足schema规范的算子会在编译时报错。当前有涉及到的有：

- 函数名已"new_"开头、"_like"结尾或入参中带有tensor_options但又没tensor入参的，需要有CompositeExplicitAutograd的dispatch。

    ```yaml
    - func: empty_with_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, int acl_format=2) -> Tensor
      dispatch:
        CompositeExplicitAutograd: empty_with_format
    ```
- 函数名中带有"rand"、带有"dropout"、或者入参带有generator的，需要有nondeterministic_seeded的tag。

    ```yaml
    - func: dropout_with_byte_mask(Tensor self, float p, bool train) -> Tensor
      tags: nondeterministic_seeded
    ```


## 算子适配常见API接口<a id="section002">

torch_npu算子操作基础接口可参见[LINK](https://gitcode.com/Ascend/pytorch/blob/7.3.0/torch_npu/csrc/framework/utils/OpPreparation.h)。常见接口如下：

- **`at_npu::native::OpPreparation::apply_tensor`**  
   - 功能：创建与输入张量属性（设备、数据类型、格式）一致的输出张量，适用于大多数算子的输出初始化。  
   - 示例：  
     ```cpp
     at::Tensor result = at_npu::native::OpPreparation::apply_tensor(rois, output_size);
     ```

- **`at_npu::native::OpPreparation::apply_tensor_without_format`**  
   - 功能：创建与输入张量尺寸和数据类型匹配的输出张量，但不指定格式（如通道顺序），适用于对格式无强制要求的场景。  
   - 示例：  
     ```cpp
     at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, input.options());
     ```

- **`at_npu::native::OpPreparation::check_tensor`**  
   - 功能：校验输出张量与输入张量的属性一致性（尺寸、数据类型等），若不匹配则调整，保障算子稳健性。  
   - 示例：  
     ```cpp
     at_npu::native::OpPreparation::check_tensor({self}, out, out, output_size);
     ```

- **`copy_scalar_to_device`**  
   - 功能：将主机（CPU）上的标量值复制到设备（NPU），确保标量数据能在NPU计算中正确使用，解决跨设备数据访问问题。  
   - 示例（参考逻辑）：  
     ```cpp
     at::Scalar scalar = 2.0f;
     at::Tensor device_scalar = copy_scalar_to_device(scalar, input.device());
     ```

- **`binary_op_check`**  
   - 功能：校验二元算子的两个输入张量是否满足运算条件（如设备一致性、数据类型兼容性等），提前发现不合法输入并抛出异常。  
   - 示例（参考逻辑）：  
     ```cpp
     binary_op_check(input1, input2, "add");
     // 确保input1和input2在设备、数据类型等方面符合add算子的要求
     ```


## 算子适配常见宏定义<a id="section003">

算子适配宏定义接口可参见[LINK](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/op_plugin/utils/op_api_common.h)。常见宏定义如下：

- **`DO_COMPATIBILITY`**  
   - 功能：用于算子兼容性处理，当NPU原生算子不可用时自动降级为备选实现（如CPU版本），确保不同环境下的功能兼容性。  
   - 示例：  
     ```cpp
     DO_COMPATIBILITY(aclnnForeachTan, at::native::foreach_tensor_tan_slow_(self));
     ```

- **`EXEC_NPU_CMD`**  
   - 功能：封装NPU底层算子调用逻辑，自动处理输入输出张量传递，简化与NPU硬件接口的交互，支持批量拆分执行。  
   - 示例：  
     ```cpp
     EXEC_NPU_CMD(aclnnForeachAddScalarV2, temp_tensors1, scalar_, temp_result);
     ```

- **`OPS_ERROR`**  
   - 功能：生成带错误码的异常信息，用于参数校验或运行时错误提示，提升错误定位精度。  
   - 示例：  
     ```cpp
     TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.", OPS_ERROR(ErrCode::VALUE));
     ```

- **`VERSION_BETWEEN`**  
   - 功能：根据NPU版本范围进行条件编译，仅在指定版本区间内生效，适配不同硬件版本特性差异（结合CMake版本判断逻辑）。  
   - 示例（逻辑参考）：  
     ```cpp
     #if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
     // 特定版本生效的代码
     #endif
     ```

- **`FLOP_COUNT`**  
   - 功能：统计算子的浮点运算次数（FLOPs），用于性能分析和优化。  
   - 示例（逻辑参考）：  
     ```cpp
     FLOP_COUNT(FlopCounter::mm_flop, input, weight_t);
     ```
