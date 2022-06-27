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

#ifndef __NATIVE_NPU_UTILS_COMMAND_BASE__
#define __NATIVE_NPU_UTILS_COMMAND_BASE__

#include "ATen/native/npu/frame/OpCmdHelper.h"
#include "ATen/native/npu/frame/OpParamMaker.h"
#include "ATen/native/npu/frame/FormatHelper.h"
#include "ATen/native/npu/graph/construct/GraphConstructor.h"
#include "ATen/native/npu/mirror/NPUTensorIterator.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "THNPU/THNPUCachingHostAllocator.h"
#include "c10/npu/NPURunMode.h"
#include "c10/npu/interface/AsyncTaskQueueInterface.h"

#define IF_GRAPH_MODE_THEN_RUN(...)            \
  do {                                         \
    if (c10::npu::NpuRunMode::IsGraphMode()) { \
      __VA_ARGS__;                             \
    }                                          \
  } while (false);

#define IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(...) \
  do {                                            \
    if (c10::npu::NpuRunMode::IsGraphMode()) {    \
      __VA_ARGS__;                                \
      return static_cast<Derived&>(*this);        \
    }                                             \
  } while (false);

namespace at {
namespace native {
namespace npu {

// get common dtype and shape from op adapter layer
struct UnifiedResult {
  c10::optional<ScalarType> common_type = c10::nullopt;
  c10::optional<IntArrayRef> common_shape = c10::nullopt;
  // judge result tensor's dtype is defined or not.
  // if result's dtype is defined, result_type_defined is true and result's
  // dtype remains unchanged.
  bool result_type_defined = false;
};

template <class Derived>
class OpCommandBase {
 public:
  explicit OpCommandBase() {
    IF_GRAPH_MODE_THEN_RUN(return;)
    aclCmds = OpCommandImpls::GetInstance();
    aclCmds->Push(aclCmd);
  }
  virtual ~OpCommandBase() {}

  OpCommandBase(const OpCommandBase& other) = delete;
  OpCommandBase(OpCommandBase&& other) = delete;
  OpCommandBase& operator=(const OpCommandBase&) = delete;
  OpCommandBase& operator=(OpCommandBase&&) = delete;

  Derived& Name(const string& name) {
    IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(graphCmd.SetName(name);)
    aclCmd->SetName(name);
    return static_cast<Derived&>(*this);
  }

  Derived& DynamicInputReg(
      DynamicInputRegFunc func,
      DyNumAndIndex num_and_index) {
    IF_GRAPH_MODE_THEN_RUN(
        graphCmd.AddDynamicInputRegFunc(func, num_and_index);)
    return static_cast<Derived&>(*this);
  }

  Derived& Expect(UnifiedResult unified_result) {
    commonType = unified_result.common_type;
    resultTypeDefined = unified_result.result_type_defined;
    commonShape = unified_result.common_shape;
    return static_cast<Derived&>(*this);
  }

  template <typename dataType>
  Derived& Attr(const string& name, dataType value) {
    IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
        graphCmd.AddAttr<dataType>(name, value);
        )
    aclCmd->AddAttr(name, value);
    return static_cast<Derived&>(*this);
  }

  Derived& Input() {
    IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
        graphCmd.AddInput();
        )
    return AddNoneTensor();
  }

  Derived& Input(
      const Tensor& input,
      const string& descName = "",
      const optional<aclFormat>& sensitive_format = nullopt,
      const string& realData = "") {
    IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
        auto contiguous_input = Contiguous(input);
        if (commonType.has_value() &&
            commonType.value() != contiguous_input.scalar_type()) {
            contiguous_input = contiguous_input.npu_dtype_cast(commonType.value());
        }
        graphCmd.AddInput(contiguous_input, descName, realData, sensitive_format);
        )
    return AddTensorInput(
        Contiguous(input), ScalarType::Undefined, descName, realData);
  }

  Derived& InputWithoutContiguousGeneral(
      const Tensor& input,
      const string& descName = "",
      const optional<aclFormat>& sensitive_format = nullopt,
      const string& realData = "") {
    return AddTensorInput(const_cast<Tensor &>(input), ScalarType::Undefined, descName, realData);
  }

  Derived& InputWithoutContiguous(const Tensor& input,
                                  const string& descName = "",
                                  const string& realData = "") {
    IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
        graphCmd.AddInput(input, descName, realData);
        )
    if (input.storage_offset() != 0) {
      TORCH_WARN_ONCE(
          "[Check][offset] Check input storage_offset[%ld] = 0 failed, result is untrustworthy",
          input.storage_offset());
    }
    return AddTensorInput(const_cast<Tensor&>(input));
  }

  Derived& Input(
      const Tensor& cpuTensor,
      SmallVector<int64_t, N> dimList,
      const string& descName = "") {
    IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
        graphCmd.AddInput(dimList, cpuTensor.scalar_type());
        )
    Tensor npuTensor = CopyHostToDevice(cpuTensor);
    return AddTensorInput(
        npuTensor, ScalarType::Undefined, descName, "", cpuTensor);
  }

  Derived& Input(const IntArrayRef& dimListRef, ScalarType toType = at::kLong) {
    IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
        graphCmd.AddInput(dimListRef, toType);
        )
    Tensor& cpuTensor = CreateHostTensor(
        (void*)dimListRef.data(),
        dimListRef.size(),
        TensorOptions(kCPU).dtype(at::kLong),
        toType);
    return AddHostTensorInput(cpuTensor);
  }

  Derived& Input(
      const Scalar& input,
      const ScalarType type,
      CompileType compileType = CompileType::MEMORY_HOST_COMPILE_INDEPENDENT) {
    IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
        auto true_type = commonType.has_value() ? commonType.value() : type;
        graphCmd.AddInput(input, true_type, compileType);
        )
    auto scalarTensor = CreateScalarTensor(input, type);
    return AddHostTensorInput(scalarTensor, compileType);
  }

  Derived& InputScalarToNPUTensor(
      const Scalar& input,
      const ScalarType type) {
    return AddScalarInput(input, type);
  }

  Derived& Input(const string& str) {
    IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
      graphCmd.AddInput(str);
    )
    AT_ERROR("single op mode do not support string input temporarily");
    return static_cast<Derived&>(*this);
  }

  // TODO(ascend): 这个类型的参数应该是一个bug
  Derived& Output(
      Tensor& output,
      const string& descName = "",
      const optional<aclFormat>& sensitive_format = nullopt,
      const string& realType = "") {
    IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
        if (sensitive_format.has_value() &&
            FormatHelper::GetBaseFormat(output) != sensitive_format.value()) {
            output = output.npu_format_cast(sensitive_format.value());
        }
        graphCmd.AddOutput(output, descName, realType, sensitive_format);
        if (!resultTypeDefined && commonType.has_value() &&
            output.scalar_type() != commonType.value()) {
            output = output.npu_dtype_cast(commonType.value());
        }
        )
    output_sync_tensor.emplace_back(output);
    return AddOutput(output, realType);
  }

  Derived& Sync(SmallVector<int64_t, N> &output_sync_idx) {
    if (!output_sync_idx.empty()) {
      sync = true;
    }
    output_sync_index = output_sync_idx;
    return static_cast<Derived&>(*this);
  }

  void Run() {
    IF_GRAPH_MODE_THEN_RUN(
      graphCmd.Run();
      return;)
    if (c10::npu::OptionsManager::CheckQueueEnable() && !sync) {
      ExecuteParas execParams;
      aclCmd->ExportParams(execParams);
      c10::npu::queue::QueueParas params(
          c10::npu::queue::COMPILE_AND_EXECUTE,
          sizeof(ExecuteParas),
          &execParams);
      c10::npu::enCurrentNPUStream(&params);
      aclCmd->releaseSource(false);
    } else {
      aclCmd->Run(sync, output_sync_index, output_sync_tensor);
      aclCmd->releaseSource();
    } 
    aclCmds->Pop();
  }

 protected:
  Derived& AddTensorInput(
      Tensor& tensor,
      ScalarType forceScaleType = ScalarType::Undefined,
      const string& descName = "",
      const string& realData = "",
      c10::optional<Tensor> cpu_tensor = c10::nullopt) {
    std::tuple<aclTensorDesc*, aclDataBuffer*> res;
    if (commonType.has_value() && commonType.value() != tensor.scalar_type()) {
      tensor = tensor.npu_dtype_cast(commonType.value());
    }
    // 针对dim=0的场景，绝对不会有输入为uint16的情况，因为这个是TBE引入的，TBE没有dim=0的情况
    if (tensor.dim() == 0) {
      if (tensor.is_npu()) {
        res =
            OpCmdHelper::CovertNPUTensorWithZeroDimToAclInput(tensor, descName);
      } else {
        res = OpCmdHelper::CovertTensorWithZeroDimToAclInput(
            tensor, forceScaleType);
      }
    } else {
      res = OpCmdHelper::CovertTensorToAclInput(
          tensor, cpu_tensor, descName, realData);
    }
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res));
    return static_cast<Derived&>(*this);
  }

  Derived& AddHostTensorInput(const Tensor& tensor,
    CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT) {
    std::tuple<aclTensorDesc*, aclDataBuffer*> res;
    res = OpCmdHelper::CovertHostTensorToAclInput(tensor, tensor.scalar_type(), compileType);
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res), tensor);
    return static_cast<Derived&>(*this);
  }

  Derived& AddNoneTensor() {
    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED).Get();
    AclTensorBufferMaker buffer(nullptr, 0);
    aclCmd->AddInput(aclDesc, buffer.Get());
    return static_cast<Derived&>(*this);
  }

  Derived& AddScalarInput(const Scalar& input, ScalarType type) {
    ScalarType type_bk = type;
    if (commonType.has_value()) {
      type_bk = commonType.value();
    }
    Tensor aclInput = CopyHostToDevice(input, type_bk);
    auto res = OpCmdHelper::CovertScalarToAclInput(aclInput, type_bk);
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res));
    return static_cast<Derived&>(*this);
  }

  Derived& AddOutput(Tensor& output, const string& realType = "") {
    if (resultTypeDefined == false && commonType.has_value() &&
        commonType.value() != output.scalar_type()) {
      output = output.npu_dtype_cast(commonType.value());
    }
    auto res = OpCmdHelper::CovertToAclOutput(output, realType);
    aclCmd->AddOutput(std::get<0>(res), std::get<1>(res));
    return static_cast<Derived&>(*this);
  }


 protected:
  // 由于format_contiguous会生成新Tensor，为了保证其在生命周期内有效，故而放到对象中存储
  // 同下，CopyScalarToDevice也有同样问题
  Tensor& Contiguous(const Tensor& input) {
    storage.emplace_back(NpuUtils::format_contiguous_add_copy_optimize(input));
    return storage.back();
  }
  Tensor CopyHostToDevice(const Scalar& scalar, ScalarType type) {
    auto tensor = scalar_to_tensor(scalar).to(type);
    return CopyHostToDevice(tensor);
  }
  Tensor CopyHostToDevice(const Tensor& cpuTensor) {
    Tensor cpuPinMemTensor = cpuTensor.pin_memory();
    int deviceIndex = 0;
    AT_NPU_CHECK(aclrtGetDevice(&deviceIndex));
    auto tensor = cpuPinMemTensor.to(
        c10::Device(DeviceType::NPU, deviceIndex),
        cpuPinMemTensor.scalar_type(),
        true,
        true);
    storage.emplace_back(tensor);
    return storage.back();
  }

  Tensor& CreateHostTensor(
      void* data,
      size_t size,
      const TensorOptions& options,
      ScalarType toType) {
    AT_ASSERT(options.dtype() == at::kLong);
    auto cpuTensor = at::empty(size, options);
    AT_ASSERT(cpuTensor.is_contiguous());
    std::memcpy(
        cpuTensor.data_ptr(), data, sizeof(int64_t) * cpuTensor.numel());
    if (toType != at::kLong) {
      cpuTensor = cpuTensor.to(toType);
    }

    storage.emplace_back(std::move(cpuTensor));
    return storage.back();
  }
  Tensor CreateScalarTensor(const Scalar& scalar, ScalarType type) {
    if (commonType.has_value()) {
      type = commonType.value();
    }
    storage.emplace_back(scalar_to_tensor(scalar).to(type));
    return storage.back();
  }
  SmallVector<Tensor, N> storage; // tensor's life cycle should maintain when Run() is called

 protected:
  OpCommandImpls* aclCmds = nullptr; // owned
  OpCommandImpl* aclCmd = nullptr;
  GraphCommandImpl graphCmd;

 private:
  c10::optional<ScalarType> commonType = c10::nullopt;
  c10::optional<IntArrayRef> commonShape = c10::nullopt;
  bool resultTypeDefined = false;

  bool sync = false;
  SmallVector<int64_t, N> output_sync_index;
  SmallVector<Tensor, N> output_sync_tensor;
  
}; // class OpCommandBase

} // namespace npu
} // namespace native
} // namespace at

#endif