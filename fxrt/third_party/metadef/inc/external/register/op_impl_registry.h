/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_H_
#define INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_H_
#include <initializer_list>
#include <string>
#include <unordered_set>
#include "graph/types.h"
#include "graph/compiler_def.h"
#include "exe_graph/runtime/base_type.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "exe_graph/runtime/op_execute_launch_context.h"
#include "graph/infer_format_context.h"
#include "graph/ascend_string.h"
#include "exe_graph/runtime/exe_res_generation_context.h"

// InferSymbol 注册接口的跨版本弱符号兼容：注册库构建（opp_registry PRIVATE 定义
// GE_INFER_SYMBOL_API_EXPORTS）时为强符号；外部算子 so 对类内落表入口
// RegisterInferSymbolForSo 产生弱引用，老 GE 上置空判空跳过（注册静默降级），
// dlopen 不因缺符号失败；覆盖点为弱未定义引用的动态解析，so 侧 -Bsymbolic 不影响
#ifndef GE_INFER_SYMBOL_API
#if defined(GE_INFER_SYMBOL_API_EXPORTS) || !defined(__GNUC__)
#define GE_INFER_SYMBOL_API
#else
#define GE_INFER_SYMBOL_API __attribute__((weak))
#endif
#endif

namespace ge {
class AnyValue;
} // namespace ge

// 直接引入rts的头文件会引入额外的include path，导致用户侧代码编译失败，因此重新定义了rts头文件中的结构体
#ifdef __cplusplus
extern "C" {
#endif

typedef struct rtExceptionInfo aclrtExceptionInfo;

#ifdef __cplusplus
}
#endif

namespace gert {
enum TilingPlacement {
  TILING_ON_HOST = 0,
  TILING_ON_AICPU = 1,
};

class InferSymbolContext;
class OpImplRegisterV2Impl;

class OpImplRegisterV2 {
 public:
  explicit OpImplRegisterV2(const ge::char_t* op_type);
  OpImplRegisterV2(OpImplRegisterV2&& register_data) noexcept;
  OpImplRegisterV2(const OpImplRegisterV2& register_data);
  OpImplRegisterV2& operator=(const OpImplRegisterV2&) = delete;
  OpImplRegisterV2& operator=(OpImplRegisterV2&&) = delete;
  ~OpImplRegisterV2();

  using InferShapeKernelFunc = UINT32 (*)(InferShapeContext*);
  using InferShapeRangeKernelFunc = UINT32 (*)(InferShapeRangeContext*);
  using TilingKernelFunc = UINT32 (*)(TilingContext*);
  using InferDataTypeKernelFunc = UINT32 (*)(InferDataTypeContext*);
  using InferSymbolKernelFunc = UINT32 (*)(InferSymbolContext*);
  using GenSimplifiedKeyKernelFunc = UINT32 (*)(TilingContext*, ge::char_t*);
  // aclnn接口的原型，入参含义：
  // OpExecuteContext：保存算子的Input，Output，Attr信息
  using OpExecFunc = UINT32 (*)(OpExecuteContext*);
  using OpExecPrepareFunc = UINT32 (*)(OpExecutePrepareContext*);
  using OpExecLaunchFunc = UINT32 (*)(OpExecuteLaunchContext*);
  using OpType = ge::AscendString;
  using PrivateAttrList = std::vector<std::pair<ge::AscendString, ge::AnyValue>>;
  using PrivateAttrSet = std::unordered_set<ge::AscendString>;
  using CompileInfoCreatorFunc = void* (*)();
  using CompileInfoDeleterFunc = void (*)(void*);
  using KernelFunc = UINT32 (*)(KernelContext* context);
  using TilingParseFunc = UINT32 (*)(TilingParseContext* context);
  using InferFormatFunc = UINT32 (*)(InferFormatContext* context);
  using OpCalcParamKernelFunc = UINT32 (*)(ExeResGenerationContext* context);
  using OpGenTaskKernelFunc =
      UINT32 (*)(const ExeResGenerationContext* context, std::vector<std::vector<uint8_t>>& tasks);
  using OP_CHECK_FUNC_V2 = ge::graphStatus (*)(const OpCheckContext* context, ge::AscendString& result);
  using ExceptionDumpFunc = void (*)(aclrtExceptionInfo* exception_info, void* reserved);

 public:
  OpImplRegisterV2& InferShape(InferShapeKernelFunc infer_shape_func);
  OpImplRegisterV2& InferShapeRange(InferShapeRangeKernelFunc infer_shape_range_func);
  OpImplRegisterV2& InferDataType(InferDataTypeKernelFunc infer_datatype_func);
  // 类内 inline 转发器（带 body → stub 生成器跳过不生成 stub）；环境支持与否经
  // 落表入口弱引用的动态解析判定——老 GE 上置空跳过注册，行为等同未注册
  OpImplRegisterV2& InferSymbol(InferSymbolKernelFunc infer_symbol_func) {
#if defined(__GNUC__) && !defined(GE_INFER_SYMBOL_API_EXPORTS)
    if (RegisterInferSymbolForSo != nullptr) {
      RegisterInferSymbolForSo(*this, infer_symbol_func);
    }
#else
    RegisterInferSymbolForSo(*this, infer_symbol_func);
#endif
    return *this;
  }
  /*
   * 一种datatype推导规则，将第一个输入的datatype作为所有输出的datatype。
   * 使用方式：注册此规则，可以不用再注册自定义推导规则。若同时注册了InferDataType和InferOutDataTypeByFirstInput，将使能最后注册的规则。
   * 异常场景：若算子无输入或输入datatype为未定义，推导将报错。
   * */
  OpImplRegisterV2& InferOutDataTypeSameWithFirstInput();
  OpImplRegisterV2& Tiling(TilingKernelFunc tiling_func, size_t max_tiling_data_size = 2048);
  OpImplRegisterV2& GenSimplifiedKey(GenSimplifiedKeyKernelFunc gen_simplifiedkey_func);
  OpImplRegisterV2& PrivateAttr(const ge::char_t* private_attr);
  OpImplRegisterV2& PrivateAttr(const ge::char_t* private_attr, int64_t private_attr_val);
  OpImplRegisterV2& PrivateAttr(const ge::char_t* private_attr, const std::vector<int64_t>& private_attr_val);
  OpImplRegisterV2& PrivateAttr(const ge::char_t* private_attr, const ge::char_t* private_attr_val);
  OpImplRegisterV2& PrivateAttr(const ge::char_t* private_attr, ge::float32_t private_attr_val);
  OpImplRegisterV2& PrivateAttr(const ge::char_t* private_attr, bool private_attr_val);
  OpImplRegisterV2& PrivateAttr(const ge::char_t* private_attr, const std::vector<ge::float32_t>& private_attr_val);
  template <typename T>
  OpImplRegisterV2& TilingParse(KernelFunc const tiling_parse_func) {
    return TilingParse(tiling_parse_func, CreateCompileInfo<T>, DeleteCompileInfo<T>);
  }
  template <typename T>
  OpImplRegisterV2& TilingParse(TilingParseFunc const tiling_parse_func) {
    return TilingParse(reinterpret_cast<KernelFunc>(tiling_parse_func), CreateCompileInfo<T>, DeleteCompileInfo<T>);
  }
  OpImplRegisterV2& InputsDataDependency(std::initializer_list<int32_t> inputs);
  OpImplRegisterV2& OpExecuteFunc(OpExecFunc op_execute_func);
  OpImplRegisterV2& Op2StageExecuteFuncs(OpExecPrepareFunc prepare_func, OpExecLaunchFunc launch_func);
  OpImplRegisterV2& HostInputs(std::initializer_list<int32_t> inputs);
  OpImplRegisterV2& TilingInputsDataDependency(std::initializer_list<int32_t> inputs);
  OpImplRegisterV2& TilingInputsDataDependency(
      std::initializer_list<int32_t> inputs,
      std::initializer_list<TilingPlacement> placements);
  OpImplRegisterV2& OutputShapeDependOnCompute(std::initializer_list<int32_t> outputs);
  OpImplRegisterV2& InferFormat(InferFormatFunc infer_format_func);
  OpImplRegisterV2& CalcOpParam(OpCalcParamKernelFunc calc_op_param_func);
  OpImplRegisterV2& GenerateTask(OpGenTaskKernelFunc gen_task_func);
  OpImplRegisterV2& CheckSupport(OP_CHECK_FUNC_V2 check_support_func);
  OpImplRegisterV2& OpSelectFormat(OP_CHECK_FUNC_V2 op_select_format_func);
  OpImplRegisterV2& ExceptionDumpParseFunc(ExceptionDumpFunc exception_func);
  OpImplRegisterV2& NullableOutputs(std::initializer_list<int32_t> outputs);
  OpImplRegisterV2& SetSupportPcieThrough();

 private:
  OpImplRegisterV2& TilingParse(
      KernelFunc tiling_parse_func,
      CompileInfoCreatorFunc creator_func,
      CompileInfoDeleterFunc deleter_func);
  OpImplRegisterV2& PrivateAttr(const ge::char_t* private_attr, ge::AnyValue private_attr_av);

  template <typename T, typename std::enable_if<(!std::is_array<T>::value), int32_t>::type = 0>
  static void* CreateCompileInfo() {
    return new T();
  }
  template <typename T>
  static void DeleteCompileInfo(void* const obj) {
    delete reinterpret_cast<T*>(obj);
  }

 private:
  // 落表入口：libopp_registry 强定义，外部算子 so 弱引用，老 GE 上解析为空判空跳过。
  // 必须为静态成员：静态成员函数取地址得到普通函数指针，可表达弱符号是否缺失；
  // 非静态成员取地址得到成员指针，与符号解析无关，判空恒真。
  GE_INFER_SYMBOL_API static void RegisterInferSymbolForSo(
      OpImplRegisterV2& reg,
      InferSymbolKernelFunc infer_symbol_func);

  std::unique_ptr<OpImplRegisterV2Impl> impl_;
};

} // namespace gert

#define IMPL_OP_COUNTER(op_type, name, counter) \
  static gert::OpImplRegisterV2 VAR_UNUSED name##counter = gert::OpImplRegisterV2(#op_type)
#define IMPL_OP_COUNTER_NUMBER(op_type, name, counter) IMPL_OP_COUNTER(op_type, name, counter)
#define IMPL_OP(op_type) IMPL_OP_COUNTER_NUMBER(op_type, op_impl_register_##op_type, __COUNTER__)
#define IMPL_OP_DEFAULT() IMPL_OP(DefaultImpl)

#define IMPL_OP_INFERSHAPE(op_type) \
  gert::OpImplRegisterV2 VAR_UNUSED op_impl_register_infershape_##op_type = gert::OpImplRegisterV2(#op_type)
#define IMPL_OP_OPTILING(op_type) \
  gert::OpImplRegisterV2 VAR_UNUSED op_impl_register_optiling_##op_type = gert::OpImplRegisterV2(#op_type)
#define IMPL_OP_INFERSYMBOL(op_type) \
  gert::OpImplRegisterV2 VAR_UNUSED op_impl_register_infersymbol_##op_type = gert::OpImplRegisterV2(#op_type)

#endif // INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_H_
