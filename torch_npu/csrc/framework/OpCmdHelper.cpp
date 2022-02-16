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

#include "torch_npu/csrc/framework/OpCmdHelper.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/OpParamMaker.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

namespace at_npu
{
  namespace native
  {

    std::tuple<aclTensorDesc *, aclDataBuffer *, int64_t, aclFormat> OpCmdHelper::CovertTensorToAclInput(
        const at::Tensor &tensor,
        const c10::optional<at::Tensor> &cpu_tensor,
        const string &descName,
        const string &forceDataType)
    {
      at::ScalarType scalarDataType = tensor.scalar_type();
      aclDataType aclDataType =
          CalcuOpUtil::convert_to_acl_data_type(scalarDataType, forceDataType);
      const auto &npuDesc = tensor.storage().get_npu_desc();
      auto &storageDims = npuDesc.storage_sizes_;
      AclTensorDescMaker desc;
      auto aclDesc = desc.Create(aclDataType, npuDesc)
                         .SetFormat(npuDesc.npu_format_)
                         .SetShape(storageDims)
                         .SetName(descName)
                         .SetConstAttr(cpu_tensor)
                         .Get();

      int64_t numel = at::prod_intlist(storageDims);
      AclTensorBufferMaker buffer(tensor, numel);
      auto aclBuff = buffer.Get();
      int64_t storageDim = storageDims.size();
      return std::tie(aclDesc, aclBuff, storageDim, npuDesc.npu_format_);
    }

    std::tuple<aclTensorDesc *, aclDataBuffer *, int64_t, aclFormat> OpCmdHelper::CovertTensorWithZeroDimToAclInput(
        const at::Tensor &tensor, at::ScalarType type)
    {
      // 针对在host侧的tensor，需要做大量处理
      at::ScalarType scalarDataType = type;
      if (!tensor.unsafeGetTensorImpl()->is_wrapped_number())
      {
        scalarDataType = tensor.scalar_type();
      }
      aclDataType aclDataType =
          CalcuOpUtil::convert_to_acl_data_type(scalarDataType);
      c10::Scalar expScalar = CalcuOpUtil::ConvertTensorToScalar(tensor);
      at::Tensor aclInput =
          CalcuOpUtil::CopyScalarToDevice(expScalar, scalarDataType);

      AclTensorDescMaker desc;
      auto aclDesc = desc.Create(aclDataType, ACL_FORMAT_ND).Get();
      AclTensorBufferMaker buffer(aclInput);
      auto aclBuff = buffer.Get();
      int64_t storageDim = 0;
      aclFormat stroageFormate = ACL_FORMAT_ND;
      return std::tie(aclDesc, aclBuff, storageDim, stroageFormate);
    }

    std::tuple<aclTensorDesc *, aclDataBuffer *, int64_t, aclFormat> OpCmdHelper::CovertNPUTensorWithZeroDimToAclInput(
        const at::Tensor &tensor, const string &descName)
    {
      aclDataType aclDataType =
          CalcuOpUtil::convert_to_acl_data_type(tensor.scalar_type());
      AclTensorDescMaker desc;
      auto aclDesc =
          desc.Create(aclDataType, ACL_FORMAT_ND).SetName(descName).Get();
      AclTensorBufferMaker buffer(tensor);
      auto aclBuff = buffer.Get();
      int64_t storageDim = 0;
      aclFormat stroageFormate = ACL_FORMAT_ND;
      return std::tie(aclDesc, aclBuff, storageDim, stroageFormate);
    }

    std::tuple<aclTensorDesc *, aclDataBuffer *, int64_t, aclFormat> OpCmdHelper::CovertScalarToAclInput(
        const at::Tensor &aclInput, at::ScalarType type)
    {
      aclDataType aclDataType = CalcuOpUtil::convert_to_acl_data_type(type);

      AclTensorDescMaker desc;
      auto aclDesc = desc.Create(aclDataType, ACL_FORMAT_ND).Get();
      AclTensorBufferMaker aclBuffer(aclInput);
      auto aclBuff = aclBuffer.Get();
      int64_t storageDim = 0;
      aclFormat stroageFormate = ACL_FORMAT_ND;
      return std::tie(aclDesc, aclBuff, storageDim, stroageFormate);
    }

    std::tuple<aclTensorDesc *, aclDataBuffer *, int64_t, aclFormat> OpCmdHelper::CovertHostTensorToAclInput(
        const at::Tensor &tensor, at::ScalarType type, CompileType compileType)
    {
      aclDataType aclDataType = CalcuOpUtil::convert_to_acl_data_type(type);

      const auto &dims = tensor.sizes();
      AclTensorDescMaker desc;
      aclFormat format = ACL_FORMAT_ND;
      auto aclDesc = desc.Create(aclDataType, dims, format)
                         .SetPlacement(static_cast<aclMemType>(compileType))
                         .Get();
      int64_t numel = at::prod_intlist(dims);
      AclTensorBufferMaker buffer(tensor, numel);
      auto aclBuff = buffer.Get();
      int64_t dim = dims.size();

      return std::tie(aclDesc, aclBuff, dim, format);
    }

    std::tuple<aclTensorDesc *, aclDataBuffer *, int64_t, aclFormat> OpCmdHelper::CovertToAclOutput(
        const at::Tensor *tensorPtr, const string &forceDataType)
    {
      aclDataType aclDataType = CalcuOpUtil::convert_to_acl_data_type(
          tensorPtr->scalar_type(), forceDataType);
      const auto &npuDesc = tensorPtr->storage().get_npu_desc();
      const auto &dims = tensorPtr->sizes();
      auto &storageDims = npuDesc.storage_sizes_;
      AclTensorDescMaker desc;
      auto aclDesc = desc.Create(aclDataType, dims, npuDesc.origin_format_)
                         .SetFormat(npuDesc.npu_format_)
                         .SetShape(storageDims)
                         .Get();
      auto numel = at::prod_intlist(storageDims);
      AclTensorBufferMaker aclBuffer(tensorPtr, numel);
      auto aclBuff = aclBuffer.Get();
      int64_t storageDim = storageDims.size();
      return std::tie(aclDesc, aclBuff, storageDim, npuDesc.npu_format_);
    }

    std::tuple<aclTensorDesc *, aclDataBuffer *, int64_t, aclFormat> OpCmdHelper::CovertTransDataTensorToAcl(
        const at::Tensor &tensor)
    {
      at::Tensor *tensorPtr = (at::Tensor *)&tensor;
      aclDataType aclDataType = CalcuOpUtil::convert_to_acl_data_type(tensorPtr->scalar_type());

      auto format = FormatHelper::GetFormat(tensor);
      auto dims = InferFormat::GuessStorageSizeWhenConvertFormat(tensor);
      AclTensorDescMaker desc;
      auto aclDesc = desc.Create(aclDataType, dims, format).Get();

      int nelements = at::prod_intlist(dims);
      AclTensorBufferMaker buffer(
          tensorPtr, tensorPtr->storage_offset(), nelements);
      auto aclBuffer = buffer.Get();
      int64_t storageDim = dims.size();
      return std::tie(aclDesc, aclBuffer, storageDim, format);
    }

  } // native
} // at
