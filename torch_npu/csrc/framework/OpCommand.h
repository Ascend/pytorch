// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef __PULGIN_NATIVE_UTILS_OP_COMMAND__
#define __PULGIN_NATIVE_UTILS_OP_COMMAND__

#include "torch_npu/csrc/framework/OpParamMaker.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/core/npu/NPURunMode.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/graph/construct/GraphConstructor.h"
#include "torch_npu/csrc/aten/mirror/NPUTensorIterator.h"

#define IF_GRAPH_MODE_THEN_RUN(...)            \
  do {                                         \
    if (c10_npu::NpuRunMode::IsGraphMode()) { \
      __VA_ARGS__;                             \
    }                                          \
  } while (false);

#define IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(...) \
  do {                                            \
    if (c10_npu::NpuRunMode::IsGraphMode()) {    \
      __VA_ARGS__;                                \
      return *this;                              \
    }                                             \
  } while (false);

namespace at_npu {
namespace native {

// get common dtype and shape from op adapter layer
struct UnifiedResult {
  c10::optional<at::ScalarType> common_type = c10::nullopt;
  c10::optional<c10::IntArrayRef> common_shape = c10::nullopt;
  // judge result tensor's dtype is defined or not.
  // if result's dtype is defined, result_type_defined is true and result's dtype remains unchanged.
  bool result_type_defined = false;
};

class OpCommand {
public:
  OpCommand() {
    IF_GRAPH_MODE_THEN_RUN(return;)
    aclCmds = OpCommandImpls::GetInstanceByTid(std::this_thread::get_id());
    aclCmds->Push(aclCmd);
    aclCmd->SetCustomHandler(nullptr);
  }
  ~OpCommand() {}

  OpCommand(const OpCommand &other) = delete;
  OpCommand(OpCommand &&other) = delete;
  OpCommand &operator=(const OpCommand &) = delete;
  OpCommand &operator=(OpCommand &&) = delete;

  OpCommand& Name(const string &name);

  OpCommand& SetCustomHandler(PROC_FUNC func);

  OpCommand& DynamicInputReg(
      DynamicInputRegFunc func,
      DyNumAndIndex num_and_index);

  OpCommand& Expect(UnifiedResult unified_result);

  // None Input
  OpCommand& Input();

  // Tensor Input which need contiguous
  OpCommand& Input(
      const at::Tensor &input,
      const string &descName = "",
      const c10::optional<aclFormat> &sensitive_format = c10::nullopt,
      const string &realData = "");

  // Tensor Input with stride info, only used in OutfeedEnqueueOpV2
  OpCommand& InputWithMetaInfo(const at::Tensor &input, const string &descName,
                               string &meta);

  // Tensor Input which no need contiguous
  OpCommand& InputWithoutContiguous(const at::Tensor &input,
                                    const string &descName = "",
                                    const string &realData = "");

  // ArrayRef Input, usually hostmemory input, we will do h2d in launch kernel
  template <typename T>
  OpCommand& Input(const c10::ArrayRef<T> &dimListRef, at::IntArrayRef realShape,
                   at::ScalarType toType,
                   CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT,
                   const string& realDtype = "",
                   const string& descName = "") {
    at::Tensor &cpuTensor = CreateHostTensor((void *) dimListRef.data(),
                                             realShape,
                                             c10::TensorOptions(at::kCPU).dtype(c10::CppTypeToScalarType<T>::value),
                                             toType);
    return AddHostTensorInput(cpuTensor, compileType, realDtype, descName);
  }
  
  // IntArrayRef/SmallVector Input, usually hostmemory input, we will do h2d in launch kernel
  OpCommand& Input(const c10::IntArrayRef &dimListRef,
                   at::ScalarType toType = at::kLong,
                   CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT,
                   const string& realDtype = "",
                   const string& descName = "");

  // DoubleArrayRef/SmallVector Input, usually hostmemory input, we will do h2d in launch kernel
  OpCommand& Input(const c10::ArrayRef<double> &dimListRef, at::IntArrayRef realShape,
                   at::ScalarType toType = at::kDouble,
                   CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT,
                   const string& realDtype = "");

  // Scalar Input, we will do h2d in launch kernel
  OpCommand& Input(const c10::Scalar &input, const at::ScalarType type,
                 CompileType compileType = CompileType::MEMORY_HOST_COMPILE_INDEPENDENT);

  // String Input Node in graph mode
  OpCommand& Input(const string &str);

  // A list of Tensor
  OpCommand& Inputs(const at::TensorList &inputs);

  OpCommand& InputScalarToNPUTensor(const c10::Scalar& input, const at::ScalarType type);

  // Output Tensor
  OpCommand& Output(
      at::Tensor &output,
      const string &descName = "",
      const c10::optional<aclFormat> &sensitive_format = c10::nullopt,
      const string &realType = "");

  // Attr
  template<typename dataType>
  OpCommand& Attr(const string &name, dataType value) {
    IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
        graphCmd.AddAttr<dataType>(name, value);
    )
    aclCmd->AddAttr(name, value);
    return *this;
  }

  // Attr depend on condition
  template<typename dataType>
  OpCommand& Attr(const string &name, dataType value, bool cond) {
    if (!cond) {
      return *this;
    }
    return Attr(name, value);
  }

  // Run a single op
  void Run();

  OpCommand& Sync(c10::SmallVector<int64_t, N> &sync_index);

  OpCommand& Sync();
private:
  OpCommand& AddTensorInput(at::Tensor &tensor,
                          at::ScalarType forceScaleType = at::ScalarType::Undefined,
                          const string &descName = "", const string &realData = "") ;
  
  OpCommand& AddTensorInput(const string &str);

  OpCommand& AddHostTensorInput(
      const at::Tensor &tensor,
      CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT,
      const string& realDtype = "",
      const string& descName = "");

  OpCommand& AddScalarInput(const c10::Scalar& input, at::ScalarType type);

  OpCommand& AddNoneTensor();

  OpCommand& AddOutput(at::Tensor &output, const string &realType = "");

  // 由于format_contiguous会生成新Tensor，为了保证其在生命周期内有效，故而放到对象中存储
  // 同下，CopyScalarToDevice也有同样问题
  at::Tensor& Contiguous(const at::Tensor &input);

  at::Tensor CopyHostToDevice(const c10::Scalar &scalar, at::ScalarType type);

  at::Tensor CopyHostToDevice(const at::Tensor &cpuTensor);

  at::Tensor& CreateHostTensor(void *data, at::IntArrayRef size,
                              const c10::TensorOptions &options, at::ScalarType toType);

  bool ScalarIsInLimits(const c10::Scalar &scalar, at::ScalarType type);

  at::Tensor& CreateScalarTensor(const c10::Scalar &scalar, at::ScalarType type);

  c10::SmallVector<at::Tensor, N> storage; // tensor's life cycle should maintain when Run() is called
  OpCommandImpls *aclCmds = nullptr; // owned
  OpCommandImpl *aclCmd = nullptr;
  GraphCommandImpl graphCmd;

  c10::optional<at::ScalarType> commonType = c10::nullopt;
  c10::optional<c10::IntArrayRef> commonShape = c10::nullopt;
  bool resultTypeDefined = false;
  bool sync = false;
  c10::SmallVector<int64_t, N> sync_index;
  c10::SmallVector<at::Tensor, N> outputTensor;
  c10::SmallVector<at::Tensor, N> inputTensor;
}; // class OpCommand
} // namespace native
} // namespace at_npu

#endif