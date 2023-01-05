// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include <unordered_map>
#include <Python.h>
#include <c10/util/Exception.h>
#include <ATen/record_function.h>

#include "torch_npu/csrc/framework/graph/util/GraphModeGuard.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/aten/mirror/NPUMemoryOverlap.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/NpuFuzzyBlacklist.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/framework/contiguous/ReshapeOpt.h"
#include "torch_npu/csrc/core/NPUBridge.h"


#include"torch_npu/csrc/core/NPUStorageImpl.h"

namespace at_npu
{
  namespace native
  {

    namespace
    {
      const static aclDataType kUnknownAclDataType = static_cast<aclDataType>(100);
      const static aclFormat kUnKnownAclFormat = static_cast<aclFormat>(100);
      const static string kUnknownDataTypeName = "UNKNOWN";
      constexpr float EPSILON = 1e-6;

      static std::unordered_map<at::ScalarType, aclDataType> AT_SCALAR_TYPE_TO_ACL_TYPE_MAP = {
          {at::ScalarType::Byte, ACL_UINT8},
          {at::ScalarType::Char, ACL_INT8},
          {at::ScalarType::Short, ACL_INT16},
          {at::ScalarType::Int, ACL_INT32},
          {at::ScalarType::Half, ACL_FLOAT16},
          {at::ScalarType::Float, ACL_FLOAT},
          {at::ScalarType::Bool, ACL_BOOL},
          {at::ScalarType::Long, ACL_INT64},
          {at::ScalarType::Double, ACL_DOUBLE},
      };

      static std::unordered_map<const at::ScalarType, const std::string> AT_SCALAR_TYPE_NAME_MAP = {
          {at::ScalarType::Byte, "at::ScalarType::Byte"},
          {at::ScalarType::Char, "at::ScalarType::Char"},
          {at::ScalarType::Short, "at::ScalarType::Short"},
          {at::ScalarType::Int, "at::ScalarType::Int"},
          {at::ScalarType::Long, "at::ScalarType::Long"},
          {at::ScalarType::Half, "at::ScalarType::Half"},
          {at::ScalarType::Float, "at::ScalarType::Float"},
          {at::ScalarType::Double, "at::ScalarType::Double"},
      };

      static std::map<const string, const aclDataType>
          STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP = {
              {"uint16", ACL_UINT16},
              {"uint8", ACL_UINT8},
              {"uint64", ACL_UINT64},
              {"string", ACL_STRING}};
      
      static std::map<aclDataType, at::ScalarType> ACL_SCALAR_TYPE_TO_AT_TYPE_MAP = {
          {ACL_UINT8,   at::ScalarType::Byte},
          {ACL_INT8,    at::ScalarType::Char},
          {ACL_INT16,   at::ScalarType::Short},
          {ACL_INT32,   at::ScalarType::Int},
          {ACL_FLOAT16, at::ScalarType::Half},
          {ACL_FLOAT,   at::ScalarType::Float},
          {ACL_BOOL,    at::ScalarType::Bool},
          {ACL_INT64,   at::ScalarType::Long},
          {ACL_DOUBLE,  at::ScalarType::Double},
      };

      string GetAtScalarTypeName(const at::ScalarType data_type)
      {
        auto iter = AT_SCALAR_TYPE_NAME_MAP.find(data_type);
        if (iter == AT_SCALAR_TYPE_NAME_MAP.end())
        {
          return kUnknownDataTypeName;
        }

        return iter->second;
      }

      aclError AclrtMemcpyAsyncParamCheck(
          void *dst,
          size_t destMax,
          const void *src,
          size_t count,
          aclrtMemcpyKind kind,
          aclrtStream stream)
      {
        if (c10_npu::NpuRunMode::IsGraphMode())
        {
          if (dst == nullptr || src == nullptr)
          {
            AT_ERROR(
                "Dst ptr or Src ptr of aclrtMemcpyAsync is nullptr!",
                "Current run mode is graph mode, "
                "try to use torch.npu.disable_graph_mode() to fix this error.");
          }
        }

        auto ret = aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
        return ret;
      }
    
      aclError AclrtMemcpyParamCheck(
          void *dst,
          size_t destMax,
          const void *src,
          size_t count,
          aclrtMemcpyKind kind)
      {
        if (c10_npu::NpuRunMode::IsGraphMode())
        {
          if (dst == nullptr || src == nullptr)
          {
            AT_ERROR(
                "Dst ptr or Src ptr of aclrtMemcpy is nullptr!",
                "Current run mode is graph mode, "
                "try to use torch.npu.disable_graph_mode() to fix this error.");
          }
        }

        auto ret = aclrtMemcpy(dst, destMax, src, count, kind);
        return ret;
      }
    } // namespace
 
    aclDataType CalcuOpUtil::convert_to_acl_data_type(const at::ScalarType data_type)
    {
      const auto &iter = AT_SCALAR_TYPE_TO_ACL_TYPE_MAP.find(data_type);
      if (iter == AT_SCALAR_TYPE_TO_ACL_TYPE_MAP.end())
      {
        NPU_LOGE(
            "Unsupport data type: %s.", GetAtScalarTypeName(data_type).c_str());
        return kUnknownAclDataType;
      }

      return iter->second;
    }

    aclDataType CalcuOpUtil::convert_to_acl_data_type(
        const at::ScalarType data_type,
        const string &realDataType)
    {
      auto iter = AT_SCALAR_TYPE_TO_ACL_TYPE_MAP.find(data_type);
      if (iter == AT_SCALAR_TYPE_TO_ACL_TYPE_MAP.end())
      {
        NPU_LOGE(
            "Unsupport data type: %s.", GetAtScalarTypeName(data_type).c_str());
        return kUnknownAclDataType;
      }
      if (!realDataType.empty())
      {
        return STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP[realDataType];
      }

      return iter->second;
    }

    at::ScalarType CalcuOpUtil::convert_to_at_data_type(const aclDataType acl_type)
    {
      auto iter = ACL_SCALAR_TYPE_TO_AT_TYPE_MAP.find(acl_type);
      if (iter == ACL_SCALAR_TYPE_TO_AT_TYPE_MAP.end()) {
        NPU_LOGE(
            "Unsupport data type: %d.", static_cast<int32_t>(acl_type));
        return at::ScalarType::Undefined;
      }
      return iter->second;
    }

    c10::Scalar CalcuOpUtil::ConvertTensorToScalar(const at::Tensor &tensor)
    {
      c10::Scalar expScalar;
      const at::Tensor *aclInput = &tensor;
      if (aclInput->scalar_type() == at::ScalarType::Double)
      {
        double value = *(double *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
      }
      else if (aclInput->scalar_type() == at::ScalarType::Long)
      {
        int64_t value = *(int64_t *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
      }
      else if (aclInput->scalar_type() == at::ScalarType::Float)
      {
        float value = *(float *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
      }
      else if (aclInput->scalar_type() == at::ScalarType::Int)
      {
        int value = *(int *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
      }
      else if (aclInput->scalar_type() == at::ScalarType::Half)
      {
        c10::Half value = *(c10::Half *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
      }
      else
      {
        NPU_LOGE("unsupport scalar type! ");
        C10_NPU_CHECK(ACL_ERROR_UNSUPPORTED_DATA_TYPE);
      }

      return expScalar;
    }

    at::Tensor CalcuOpUtil::CopyScalarToDevice(
        const c10::Scalar &cpu_scalar,
        at::ScalarType scalar_data_type)
    {
      return CalcuOpUtil::copy_tensor_host_to_device(
          scalar_to_tensor(cpu_scalar).to(scalar_data_type));
    }

    at::Tensor CalcuOpUtil::copy_tensor_host_to_device(const at::Tensor &cpu_tensor)
    {
      at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
      int deviceIndex = 0;
      C10_NPU_CHECK(aclrtGetDevice(&deviceIndex));
      return cpuPinMemTensor.to(
          c10::Device(at_npu::key::NativeDeviceType, deviceIndex),
          cpuPinMemTensor.scalar_type(),
          true,
          true);
    }

    NPUStatus CalcuOpUtil::AclrtMemcpyAsync(
        const std::pair<at::Tensor, int64_t>& dst,
        size_t dst_size,
        const std::pair<at::Tensor, int64_t>& src,
        size_t src_size,
        aclrtMemcpyKind kind)
    {
      GraphModeGuard mode_guard(c10_npu::ModeKind::SINGLE_OP_MODE);
      void* dst_ptr = reinterpret_cast<uint8_t*>(dst.first.data_ptr()) +
            dst.second * dst.first.itemsize();
      void* src_ptr = reinterpret_cast<uint8_t*>(src.first.data_ptr()) +
            src.second * src.first.itemsize();
      C10_NPU_CHECK(c10_npu::queue::LaunchAsyncCopyTask(
          dst_ptr, dst_size, const_cast<void *>(src_ptr), src_size, kind));

      return "SUCCESS";
    }

    aclError CalcuOpUtil::AclrtMemcpyAsyncWithModeSwitch(
        const StorageAndOffsetMemSizePair& dst,
        size_t dstMax,
        const StorageAndOffsetMemSizePair& src,
        size_t count,
        aclrtMemcpyKind kind,
        aclrtStream stream) {
      if (c10_npu::NpuRunMode::IsGraphMode()) {
        GraphExecutor::GetInstance().ConstructAndExecuteGraph();
      }

      void* dst_ptr = static_cast<void*>(
          static_cast<uint8_t*>(dst.first->data()) + dst.second);
      void* src_ptr = static_cast<void*>(
          static_cast<uint8_t*>(src.first->data()) + src.second);
      return AclrtMemcpyAsyncParamCheck(
          dst_ptr, dstMax, const_cast<void*>(src_ptr), count, kind, stream);
    }

    aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(
        const StorageAndOffsetMemSizePair& dst,
        size_t dstMax,
        const StorageAndOffsetMemSizePair& src,
        size_t count,
        aclrtMemcpyKind kind) {
      if (c10_npu::NpuRunMode::IsGraphMode()) {
        GraphExecutor::GetInstance().ConstructAndExecuteGraph();
      }

      void* dst_ptr = static_cast<void*>(
          static_cast<uint8_t*>(dst.first->data()) + dst.second);
      void* src_ptr = static_cast<void*>(
          static_cast<uint8_t*>(src.first->data()) + src.second);
      return AclrtMemcpyParamCheck(
          dst_ptr, dstMax, const_cast<void*>(src_ptr), count, kind);
    }

    aclError CalcuOpUtil::AclrtMemcpyAsyncWithModeSwitch(
        const StorageAndOffsetMemSizePair& dst,
        size_t dstMax,
        const void* src,
        size_t count,
        aclrtMemcpyKind kind,
        aclrtStream stream) {
      if (c10_npu::NpuRunMode::IsGraphMode()) {
        GraphExecutor::GetInstance().ConstructAndExecuteGraph();
      }

      void* dst_ptr = static_cast<void*>(
          static_cast<uint8_t*>(dst.first->data()) + dst.second);
      return AclrtMemcpyAsyncParamCheck(
          dst_ptr, dstMax, src, count, kind, stream);
    }

    aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(
        const StorageAndOffsetMemSizePair& dst,
        size_t dstMax,
        const void* src,
        size_t count,
        aclrtMemcpyKind kind) {
      if (c10_npu::NpuRunMode::IsGraphMode()) {
        GraphExecutor::GetInstance().ConstructAndExecuteGraph();
      }

      void* dst_ptr = static_cast<void*>(
          static_cast<uint8_t*>(dst.first->data()) + dst.second);
      return AclrtMemcpyParamCheck(
          dst_ptr, dstMax, src, count, kind);
    }

    aclError CalcuOpUtil::AclrtMemcpyAsyncWithModeSwitch(
        void* dst,
        size_t dstMax,
        const StorageAndOffsetMemSizePair& src,
        size_t count,
        aclrtMemcpyKind kind,
        aclrtStream stream) {
      if (c10_npu::NpuRunMode::IsGraphMode()) {
        GraphExecutor::GetInstance().ConstructAndExecuteGraph();
      }

      void* src_ptr = static_cast<void*>(
          static_cast<uint8_t*>(src.first->data()) + src.second);
      return AclrtMemcpyAsyncParamCheck(
          dst, dstMax, const_cast<void*>(src_ptr), count, kind, stream);
    }

    aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(
        void* dst,
        size_t dstMax,
        const StorageAndOffsetMemSizePair& src,
        size_t count,
        aclrtMemcpyKind kind) {
      if (c10_npu::NpuRunMode::IsGraphMode()) {
        GraphExecutor::GetInstance().ConstructAndExecuteGraph();
      }

      void* src_ptr = static_cast<void*>(
          static_cast<uint8_t*>(src.first->data()) + src.second);
      return AclrtMemcpyParamCheck(
          dst, dstMax, const_cast<void*>(src_ptr), count, kind);
    }

    aclError CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(
        const at::Tensor& dst,
        size_t dstMax,
        const at::Tensor& src,
        size_t count,
        aclrtMemcpyKind kind) {
      if (c10_npu::NpuRunMode::IsGraphMode()) {
        GraphExecutor::GetInstance().ConstructAndExecuteGraph();
      }

      aclError ret = c10_npu::queue::LaunchAsyncCopyTask(
          dst.data_ptr(), dstMax, src.data_ptr(), count, kind);
      return ret;
    }

    aclError CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(
        const c10::StorageImpl& dst,
        size_t dstMax,
        void* src,
        size_t count,
        aclrtMemcpyKind kind) {
      if (c10_npu::NpuRunMode::IsGraphMode()) {
        GraphExecutor::GetInstance().ConstructAndExecuteGraph();
      }

      aclError ret = c10_npu::queue::LaunchAsyncCopyTask(
          dst.data(), dstMax, src, count, kind);
      return ret;
    }

    bool check_npu_format_unchanged_in_format_contiguous(
        const at::Tensor &tensor,
        const torch_npu::NPUStorageDesc &tensor_desc) {
      // There are 2 cases that we need apply new tensors with native formats.
      if (!tensor.is_contiguous()) {
        // case1-discontiguous & slice case(op supports inputs with native
        // formats)
        return TransContiguous::CanOptimize(tensor, {"slice"});
      }

      if (!StorageDescHelper::MetaDataAreMatch(&tensor)) {
        if (tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ) {
          // case2-unmatched & memory copy without transdata
          return CanUseMemcpyForOtherFormat(tensor);
        }
        return false;
      }

      bool isPadding = FormatHelper::IsPadded(&tensor);
      if (isPadding && (!StorageDescHelper::OffsetAreMatch(&tensor))) {
        return false;
      }
      return true;
    }

    int64_t CalcuOpUtil::get_tensor_npu_format(const at::Tensor &tensor) {
      TORCH_CHECK(tensor.device().type() == at_npu::key::NativeDeviceType,
          "Expected all tensors to be on the same device. "
          "Expected NPU tensor, please check whether the input tensor device is correct.");
      if (NpuUtils::check_match(&tensor) || NpuUtils::check_5d_5d_match(tensor))
      {
        const torch_npu::NPUStorageDesc &tensor_desc =
          torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
        return tensor_desc.npu_format_;
      }
      else
      {
        return InferFormat::GuessFormatWhenContiguous(tensor);
      }
    }

    int64_t CalcuOpUtil::judge_and_get_format_from_input(bool is_cast_weight,
                                                         const at::Tensor &input,
                                                         int64_t target_format) {
        return is_cast_weight ? target_format : get_tensor_npu_format(input);
    }
    
    void CalcuOpUtil::check_memory_over_laps(
        c10::SmallVector<at::Tensor, N> &inputs,
        c10::SmallVector<at::Tensor, N> &outputs)
    {
      for (int i = 0; i < outputs.size(); i++)
      {
        if (!outputs[i].defined())
          continue;

        assert_no_internal_overlap(outputs[i]);

        for (int j = 0; j < inputs.size(); j++)
        {
          assert_no_partial_overlap(outputs[i], inputs[j]);
        }
      }
    }

    string CalcuOpUtil::get_reduction_str(int64_t reduction)
    {
      string reductionStr;
      if (reduction == at::Reduction::None)
      {
        reductionStr = "none";
      }
      else if (reduction == at::Reduction::Mean)
      {
        reductionStr = "mean";
      }
      else
      {
        reductionStr = "sum";
      }

      return reductionStr;
    }

    int64_t CalcuOpUtil::make_wrap_dim(int64_t dim, int64_t dim_post_expr)
    {
      if (dim_post_expr <= 0)
      {
        dim_post_expr = 1; // this will make range [-1, 0]
      }

      int64_t min = -dim_post_expr;
      int64_t max = dim_post_expr - 1;
      if (dim < 0)
      {
        dim += dim_post_expr;
      }
      return dim;
    }

    bool CalcuOpUtil::is_transpose_last_two_dims(const at::Tensor &tensor)
    {
      if (tensor.dim() < 2 || tensor.dim() > 3)
      {
        return false;
      }
      int64_t numel = 1;
      auto storageSize = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->get_npu_desc().storage_sizes_;

      for (int i = 0; i < storageSize.size(); i++)
      {
        numel *= storageSize[i];
      }

      int64_t dim1 = tensor.dim() - 1;
      int64_t dim2 = tensor.dim() - 2;

      auto tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->get_npu_desc();
      if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2) &&
          tensor.size(dim1) == tensor_desc.base_sizes_[dim2] &&
          tensor.size(dim2) == tensor_desc.base_sizes_[dim1] &&
          tensor.numel() == numel &&
          tensor_desc.base_sizes_.size() == tensor.dim())
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    bool CalcuOpUtil::is_scalar_wrapped_to_tensor(const at::Tensor &tensor)
    {
      return tensor.unsafeGetTensorImpl()->is_wrapped_number() &&
             (!at_npu::key::isDeviceTensor(tensor));
    }

    bool CalcuOpUtil::is_scalar_one(const c10::Scalar &scalar)
    {
      if (scalar.isIntegral(false))
      {
        return scalar.toInt() == 1;
      }
      else if (scalar.isFloatingPoint())
      {
        return fabs(scalar.toFloat() - 1.0) < EPSILON;
      }
      else
      {
        return false;
      }
    }

    float CalcuOpUtil::get_scalar_float_value(const c10::Scalar &scalar)
    {
      float value;
      if (scalar.isFloatingPoint())
      {
        value = scalar.toFloat();
      }
      else
      {
        value = (float)scalar.toInt();
      }

      return value;
    }

    at::ScalarType CalcuOpUtil::GetNPUTensorDescScalarType(
        const NPUTensorDesc &npuTensorDesc)
    {
      at::ScalarType scalarDataType;

      if (npuTensorDesc.tensorDescType == NPUTensorDesc::TensorDescType::SCALAR ||
          (npuTensorDesc.tensorDescType ==
               NPUTensorDesc::TensorDescType::TENSOR_SCALAR &&
           (!at_npu::key::isDeviceTensor(npuTensorDesc.tensor)) &&
           is_scalar_wrapped_to_tensor(npuTensorDesc.tensor)))
      {
        scalarDataType = npuTensorDesc.scalarType;
      }
      else
      {
        scalarDataType = npuTensorDesc.tensor.scalar_type();
      }

      return scalarDataType;
    }

    c10::SmallVector<at::Tensor, N> CalcuOpUtil::ConvertTensorListToSmallVector(
        at::TensorList tensors)
    {
      c10::SmallVector<at::Tensor, N> tensorVec;
      for (int i = 0; i < tensors.size(); i++)
      {
        tensorVec.emplace_back(tensors[i]);
      }

      return tensorVec;
    }

    c10::SmallVector<int64_t, N> CalcuOpUtil::ConvertIntArrayRefToSmallVector(
        c10::IntArrayRef intArray)
    {
      c10::SmallVector<int64_t, N> intVec;
      for (int i = 0; i < intArray.size(); i++)
      {
        intVec.emplace_back(intArray[i]);
      }

      return intVec;
    }

    NPUStatus CalcuOpUtil::CreateAclTensorDescInfo(
        c10::SmallVector<NPUTensorDesc, N> &input,
        c10::SmallVector<NPUTensorDesc, N> &output,
        ACL_PARAMS &params,
        string opName,
        const c10::SmallVector<NPUAttrDesc, N> &attrs)
    {
      int inputNum = input.size();
      int outputNum = output.size();

      size_t inputTensorDescArrLen = inputNum * sizeof(uintptr_t);
      size_t inputDataBuffArrLen   = inputNum * sizeof(uintptr_t);

      size_t outputTensorDescArrLen = outputNum * sizeof(uintptr_t);
      size_t outputDataBuffArrLen   = outputNum * sizeof(uintptr_t);

      size_t totalMemLen =
        inputTensorDescArrLen + inputDataBuffArrLen +
        outputTensorDescArrLen + outputDataBuffArrLen;
      char* basePtr = static_cast<char* >(malloc(totalMemLen));
      AT_ASSERT(basePtr != nullptr);

      const aclTensorDesc** aclTensorInputDescArr = reinterpret_cast<const aclTensorDesc** >(basePtr);
      basePtr += inputTensorDescArrLen;
      const aclDataBuffer** aclDataInputBuffArr = reinterpret_cast<const aclDataBuffer** >(basePtr);
      basePtr += inputDataBuffArrLen;

      const aclTensorDesc** aclTensorOutputDescArr = reinterpret_cast<const aclTensorDesc** >(basePtr);
      basePtr += outputTensorDescArrLen;
      aclDataBuffer** aclDataOutputBuffArr = reinterpret_cast<aclDataBuffer** >(basePtr);

      for (int i = 0; i < inputNum; i++)
      {
        at::ScalarType scalarDataType =
            CalcuOpUtil::GetNPUTensorDescScalarType(input[i]);
        aclDataType aclDataType = CalcuOpUtil::convert_to_acl_data_type(
            scalarDataType, input[i].realDataType);

        size_t dimCnt = 0;
        int64_t shape[0];

        if (input[i].tensorDescType == NPUTensorDesc::TensorDescType::NONE_TENSOR)
        {
          // Create aclCreateTensorDesc and aclCreateDataBuffer of a NoneTensor.
          aclTensorInputDescArr[i] = aclCreateTensorDesc(
              ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
          aclDataInputBuffArr[i] = aclCreateDataBuffer(nullptr, 0);
        }
        else if (
            input[i].tensorDescType == NPUTensorDesc::TensorDescType::TENSOR)
        {
          at::Tensor *aclInput = &input[i].tensor;
          const auto& dims = torch_npu::NPUBridge::GetNpuStorageImpl(*aclInput)->get_npu_desc().base_sizes_;
          const auto& storageDims = torch_npu::NPUBridge::GetNpuStorageImpl(*aclInput)->get_npu_desc().storage_sizes_;
          int64_t numel = 1;
          for (int j = 0; j < storageDims.size(); j++)
          {
            numel *= storageDims[j];
          }

          aclTensorDesc *acl_tensor_desc = aclCreateTensorDesc(
              aclDataType,
              dims.size(),
              dims.data(),
              torch_npu::NPUBridge::GetNpuStorageImpl(*aclInput)->get_npu_desc().origin_format_);
          aclSetTensorFormat(
              acl_tensor_desc, torch_npu::NPUBridge::GetNpuStorageImpl(*aclInput)->get_npu_desc().npu_format_);
          aclSetTensorShape(
              acl_tensor_desc, storageDims.size(), storageDims.data());
          if (input[i].tensorDescName != "")
          {
            aclSetTensorDescName(acl_tensor_desc, input[i].tensorDescName.c_str());
          }
          aclTensorInputDescArr[i] = acl_tensor_desc;
          aclDataInputBuffArr[i] = aclCreateDataBuffer(
              (void *)(aclInput->data_ptr()), aclInput->itemsize() * numel);
        }
        else if (
            input[i].tensorDescType ==
                NPUTensorDesc::TensorDescType::TENSOR_SCALAR &&
            at_npu::key::isDeviceTensor(input[i].tensor))
        {
          at::Tensor *aclInput = &input[i].tensor;
          aclTensorInputDescArr[i] =
              aclCreateTensorDesc(aclDataType, dimCnt, shape, ACL_FORMAT_ND);
          aclDataInputBuffArr[i] = aclCreateDataBuffer(
              (void *)aclInput->data_ptr(), aclInput->itemsize());
        }
        else
        {
          c10::Scalar expScalar;
          if (input[i].tensorDescType == NPUTensorDesc::TensorDescType::SCALAR)
          {
            expScalar = input[i].scalar;
          }
          else
          {
            expScalar = ConvertTensorToScalar(input[i].tensor);
          }

          at::Tensor aclInput =
              CalcuOpUtil::CopyScalarToDevice(expScalar, scalarDataType);
          aclTensorInputDescArr[i] =
              aclCreateTensorDesc(aclDataType, dimCnt, shape, ACL_FORMAT_ND);
          aclDataInputBuffArr[i] =
              aclCreateDataBuffer((void *)aclInput.data_ptr(), aclInput.itemsize());
        }
      }

      for (int i = 0; i < outputNum; i++)
      {
        at::Tensor *aclOutput = &output[i].tensor;
        aclDataType aclDataType = CalcuOpUtil::convert_to_acl_data_type(
            aclOutput->scalar_type(), output[i].realDataType);

        auto dims = aclOutput->sizes();
        auto storageDims = torch_npu::NPUBridge::GetNpuStorageImpl(*aclOutput)->get_npu_desc().storage_sizes_;
        int64_t numel = 1;
        for (int j = 0; j < storageDims.size(); j++)
        {
          numel *= storageDims[j];
        }

        aclTensorDesc *acl_tensor_desc = aclCreateTensorDesc(
            aclDataType,
            dims.size(),
            dims.data(),
            torch_npu::NPUBridge::GetNpuStorageImpl(*aclOutput)->get_npu_desc().origin_format_);
        aclSetTensorFormat(
            acl_tensor_desc, torch_npu::NPUBridge::GetNpuStorageImpl(*aclOutput)->get_npu_desc().npu_format_);
        aclSetTensorShape(
            acl_tensor_desc, storageDims.size(), storageDims.data());
        aclTensorOutputDescArr[i] = acl_tensor_desc;
        aclDataOutputBuffArr[i] = aclCreateDataBuffer(
            (void *)aclOutput->data_ptr(), aclOutput->itemsize() * numel);
      }

      params.input_num = inputNum;
      params.input_desc = aclTensorInputDescArr;
      params.input_data_buf = aclDataInputBuffArr;

      params.output_num = outputNum;
      params.output_desc = aclTensorOutputDescArr;
      params.output_data_buf = aclDataOutputBuffArr;

      return "SUCCESS";
    }

    c10::SmallVector<NPUTensorDesc, N> CalcuOpUtil::create_npu_input_tensor_desc(
        const c10::SmallVector<at::Tensor, N> &inputTensor)
    {
      c10::SmallVector<NPUTensorDesc, N> inputs;

      for (int i = 0; i < inputTensor.size(); i++)
      {
        inputs.emplace_back(
            NPUTensorDesc(NpuUtils::format_contiguous(inputTensor[i])));

        if (inputTensor[i].dim() == 0)
        {
          inputs[i].tensorDescType = NPUTensorDesc::TensorDescType::TENSOR_SCALAR;
        }
      }

      return inputs;
    }

    c10::SmallVector<NPUTensorDesc, N> CalcuOpUtil::create_npu_input_tensor_desc(
        const c10::SmallVector<at::Tensor, N> &inputTensor,
        const c10::SmallVector<uint, N> &masks)
    {
      c10::SmallVector<NPUTensorDesc, N> inputs;

      for (int i = 0; i < inputTensor.size(); i++)
      {
        inputs.emplace_back(
            NPUTensorDesc(NpuUtils::format_contiguous(inputTensor[i])));

        if (inputTensor[i].dim() == 0)
        {
          inputs[i].tensorDescType = NPUTensorDesc::TensorDescType::TENSOR_SCALAR;
        }
      }

      // Set NPUTensorDesc.tensorDescType be NONE_TENSOR, which index in masks.
      for (int j = 0; j < masks.size(); j++)
      {
        inputs[masks[j]].tensorDescType =
            NPUTensorDesc::TensorDescType::NONE_TENSOR;
      }

      return inputs;
    }

    c10::SmallVector<NPUTensorDesc, N> CalcuOpUtil::create_npu_input_tensor_desc(
        const c10::SmallVector<c10::Scalar, N> &inputScalar,
        at::ScalarType scalar_type)
    {
      c10::SmallVector<NPUTensorDesc, N> inputs;

      for (int i = 0; i < inputScalar.size(); i++)
      {
        inputs.emplace_back(NPUTensorDesc(inputScalar[i], scalar_type));
      }

      return inputs;
    }

    c10::SmallVector<NPUTensorDesc, N> CalcuOpUtil::create_npu_output_tensor_desc(
        const c10::SmallVector<at::Tensor, N> &outputTensor)
    {
      c10::SmallVector<NPUTensorDesc, N> outputs;

      for (int i = 0; i < outputTensor.size(); i++)
      {
        outputs.emplace_back(NPUTensorDesc(outputTensor[i]));
      }

      return outputs;
    }

    c10::SmallVector<int64_t, N> CalcuOpUtil::get_dimlist_for_tensor(
        const at::Tensor &self)
    {
      c10::SmallVector<int64_t, N> dimList = {};
      for (int64_t i = 0; i < self.dim(); i++)
      {
        dimList.emplace_back(i);
      }
      return dimList;
    }

    aclopAttr* CalcuOpUtil::CreateNpuAttrDesc(
        const c10::SmallVector<NPUAttrDesc, N> &attrs)
    {
      aclopAttr *attr = aclopCreateAttr();

      for (NPUAttrDesc npuAttrDesc : attrs)
      {
        switch (npuAttrDesc.attrType)
        {
        case NPUAttrDesc::AttrDescType::BOOL_TYPE:
          aclopSetAttrBool(
              attr, npuAttrDesc.attrName.c_str(), npuAttrDesc.boolAttrValue);
          break;
        case NPUAttrDesc::AttrDescType::INT_TYPE:
          aclopSetAttrInt(
              attr, npuAttrDesc.attrName.c_str(), npuAttrDesc.intAttrValue);
          break;
        case NPUAttrDesc::AttrDescType::FLOAT_TYPE:
          aclopSetAttrFloat(
              attr, npuAttrDesc.attrName.c_str(), npuAttrDesc.floatAttrValue);
          break;
        case NPUAttrDesc::AttrDescType::STRING_TYPE:
          aclopSetAttrString(
              attr,
              npuAttrDesc.attrName.c_str(),
              npuAttrDesc.stringAttrValue.c_str());
          break;
        case NPUAttrDesc::AttrDescType::LIST_INT_TYPE:
          aclopSetAttrListInt(
              attr,
              npuAttrDesc.attrName.c_str(),
              npuAttrDesc.listIntAttrValue.size(),
              npuAttrDesc.listIntAttrValue.data());
          break;
        case NPUAttrDesc::AttrDescType::LIST_FLOAT_TYPE:
          aclopSetAttrListFloat(
              attr,
              npuAttrDesc.attrName.c_str(),
              npuAttrDesc.listFloatAttrValue.size(),
              npuAttrDesc.listFloatAttrValue.data());
          break;
        case NPUAttrDesc::AttrDescType::LIST_LIST_INT_TYPE:
          aclopSetAttrListListInt(
              attr,
              npuAttrDesc.attrName.c_str(),
              npuAttrDesc.listListIntAttrValue.size(),
              npuAttrDesc.listListIntAttrListIntNum.data(),
              npuAttrDesc.listListIntAttrValue.data());
          break;
        default:
          AT_ERROR("unsupport attr type", npuAttrDesc.attrType);
        }
      }

      return attr;
    }

    void CalcuOpUtil::execute_npu_operate(
        string opName,
        c10::SmallVector<NPUTensorDesc, N> &inputs,
        c10::SmallVector<NPUTensorDesc, N> &outputs,
        const c10::SmallVector<NPUAttrDesc, N> &attrs)
    {
      if (c10_npu::NpuRunMode::IsGraphMode()) {
        AT_ERROR(
            "In graph mode, can not use CalcuOpUtil::execute_npu_operate to execute op.",
            "Try to use eager mode, or fix operator ",
            opName,
            " with OpCommand to solve this problem.");
      }

      if (c10_npu::option::OptionsManager::CheckQueueEnable())
      {
        ExecuteParas cur_paras;
        opName.copy(cur_paras.opType, opName.length() + 1);
        CalcuOpUtil::CreateAclTensorDescInfo(
            inputs, outputs, cur_paras.paras, opName, attrs);
        auto attrRes = CalcuOpUtil::CreateNpuAttrDesc(attrs);
        cur_paras.attr = attrRes;
        if (!FuzzyCompileBlacklist::GetInstance().IsInBlacklist(cur_paras.opType) && env::CheckFuzzyEnable()) {
          cur_paras.isFuzzy = true;
        }
        c10_npu::queue::QueueParas params(c10_npu::queue::COMPILE_AND_EXECUTE, sizeof(ExecuteParas), &cur_paras);
        c10_npu::enCurrentNPUStream(&params);
        return;
      }

      ACL_PARAMS params;

      CalcuOpUtil::CreateAclTensorDescInfo(inputs, outputs, params, opName, attrs);
      aclopAttr *attr = CalcuOpUtil::CreateNpuAttrDesc(attrs);

      auto stream = c10_npu::getCurrentNPUStream();
      RECORD_FUNCTION(opName, std::vector<c10::IValue>({}));
      bool reset_flag = false;
      if (env::CheckFuzzyEnable() &&
          FuzzyCompileBlacklist::GetInstance().IsInBlacklist(opName))
      {
        AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_DEFAULT);
        reset_flag = true;
      }
      NPU_LOGD("Op %s aclopCompileAndExecute Run.", opName.c_str());
      if (PyGILState_Check())
      {
        Py_BEGIN_ALLOW_THREADS
            aclError ret;
        int index = 0;
        do
        {
          ret = aclopCompileAndExecute(
              opName.c_str(),
              params.input_num,
              params.input_desc,
              params.input_data_buf,
              params.output_num,
              params.output_desc,
              params.output_data_buf,
              attr,
              ACL_ENGINE_SYS,
              ACL_COMPILE_SYS,
              NULL,
              stream);
          ++index;
        } while (NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
        ACL_REQUIRE_OK_OP(ret, opName.c_str());
        Py_END_ALLOW_THREADS
      }
      else
      {
        aclError ret;
        int index = 0;
        do
        {
          ret = aclopCompileAndExecute(
              opName.c_str(),
              params.input_num,
              params.input_desc,
              params.input_data_buf,
              params.output_num,
              params.output_desc,
              params.output_data_buf,
              attr,
              ACL_ENGINE_SYS,
              ACL_COMPILE_SYS,
              NULL,
              stream);
          ++index;
        } while (NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
        ACL_REQUIRE_OK_OP(ret, opName.c_str());
      }
      if (reset_flag)
      {
        AclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
      }
      aclopDestroyAttr(attr);
      NPUStatus ret = DestroyAclParams(params);

      if (ret != "SUCCESS")
      {
        NPU_LOGE("DestroyAclParams fail, ret: %s", ret.c_str());
      }
    }

    int64_t CalcuOpUtil::completePad(
        int64_t s_size,
        int64_t p_size,
        int64_t k_size,
        int64_t stride)
    {
      int64_t needpads = 0;
      int64_t sizeP = s_size + p_size * 2;
      int64_t leftLen = sizeP - k_size;
      if (stride == 0)
      {
        AT_ERROR("completePad stride is zero!");
      }
      auto reminder = leftLen % stride;
      if (reminder != 0)
      {
        needpads = stride - reminder;
      }
      return needpads;
    }

    c10::SmallVector<int64_t, 3> CalcuOpUtil::compute_output_size(
        c10::IntArrayRef input_size, // Full input tensor size.
        c10::optional<c10::IntArrayRef> output_size,
        c10::optional<c10::ArrayRef<double>> scale_factors)
    {
      int spatial_dimensions = input_size.size() - 2;
      if (output_size)
      {
        TORCH_CHECK(!scale_factors, "Must specify exactly one of output_size and scale_factors");
        TORCH_CHECK(output_size->size() == spatial_dimensions);
        return {output_size->data(), output_size->data() + output_size->size()};
      }
      if (scale_factors)
      {
        TORCH_CHECK(!output_size, "Must specify exactly one of output_size and scale_factors");
        TORCH_CHECK(scale_factors->size() == spatial_dimensions);
        c10::SmallVector<int64_t, 3> ret;
        for (int i = 0; i < spatial_dimensions; ++i)
        {
          ret.push_back(static_cast<double>(input_size[i + 2]) * scale_factors.value()[i]);
        }
        return ret;
      }
      TORCH_CHECK(false, "Must specify exactly one of output_size and scale_factors");
    }

    c10::optional<double> CalcuOpUtil::get_scale_value(c10::optional<c10::ArrayRef<double>> scales, int idx)
    {
      if (!scales)
      {
        return c10::nullopt;
      }
      return scales->at(idx);
    }

  } // namespace native
} // namespace at_npu
