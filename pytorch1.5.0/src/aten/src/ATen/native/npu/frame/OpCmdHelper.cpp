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

#include "OpCmdHelper.h"
#include "ATen/native/npu/frame/FormatHelper.h"
#include "ATen/native/npu/frame/OpParamMaker.h"
#include "ATen/native/npu/frame/InferFormat.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
namespace npu {

std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat>
OpCmdHelper::CovertTensorToAclInput(
    const Tensor& tensor,
    const c10::optional<Tensor>& cpu_tensor,
    const string& descName,
    const string& forceDataType) {
  ScalarType scalarDataType = tensor.scalar_type();
  aclDataType aclDataType =
      CalcuOpUtil::convert_to_acl_data_type(scalarDataType, forceDataType);
  const auto& npuDesc = tensor.storage().get_npu_desc();
  auto& storageDims = npuDesc.storage_sizes_;
  AclTensorDescMaker desc;
  auto aclDesc = desc.Create(aclDataType, npuDesc)
                      .SetFormat(npuDesc.npu_format_)
                      .SetShape(storageDims)
                      .SetName(descName)
                      .SetConstAttr(cpu_tensor)
                      .Get();

  int64_t numel = prod_intlist(storageDims);
  AclTensorBufferMaker buffer(tensor, numel);
  auto aclBuff = buffer.Get();
  int64_t storageDim = storageDims.size();
  return std::tie(aclDesc, aclBuff, storageDim, npuDesc.npu_format_);
}

std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat>
OpCmdHelper::CovertTensorWithZeroDimToAclInput(const Tensor& tensor, ScalarType type) {
  // 针对在host侧的tensor，需要做大量处理
  ScalarType scalarDataType = type;
  if (!tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    scalarDataType = tensor.scalar_type();
  }
  aclDataType aclDataType =
      CalcuOpUtil::convert_to_acl_data_type(scalarDataType);
  Scalar expScalar = CalcuOpUtil::ConvertTensorToScalar(tensor);
  Tensor aclInput = 
      CalcuOpUtil::CopyScalarToDevice(expScalar, scalarDataType);

  AclTensorDescMaker desc;
  auto aclDesc = desc.Create(aclDataType, ACL_FORMAT_ND).Get();
  AclTensorBufferMaker buffer(aclInput);
  auto aclBuff = buffer.Get();
  int64_t storageDim = 0;
  aclFormat stroageFormate = ACL_FORMAT_ND;
  return std::tie(aclDesc, aclBuff, storageDim, stroageFormate);
}

std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat>
OpCmdHelper::CovertNPUTensorWithZeroDimToAclInput(const Tensor& tensor, const string& descName) {
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

std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat>
OpCmdHelper::CovertScalarToAclInput(const Tensor& aclInput, ScalarType type) {
  aclDataType aclDataType = CalcuOpUtil::convert_to_acl_data_type(type);

  AclTensorDescMaker desc;
  auto aclDesc = desc.Create(aclDataType, ACL_FORMAT_ND).Get();
  AclTensorBufferMaker aclBuffer(aclInput);
  auto aclBuff = aclBuffer.Get();
  int64_t storageDim = 0;
  aclFormat stroageFormate = ACL_FORMAT_ND;
  return std::tie(aclDesc, aclBuff, storageDim, stroageFormate);
}

std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat>
OpCmdHelper::CovertHostTensorToAclInput(const Tensor& tensor, ScalarType type) {
  aclDataType aclDataType = CalcuOpUtil::convert_to_acl_data_type(type);

  const auto& dims = tensor.sizes();
  AclTensorDescMaker desc;
  aclFormat format = ACL_FORMAT_ND;
  auto aclDesc = desc.Create(aclDataType, dims, format)
                      .SetPlacement(aclMemType::ACL_MEMTYPE_HOST)
                      .Get();
  int64_t numel = prod_intlist(dims);
  AclTensorBufferMaker buffer(tensor, numel);
  auto aclBuff = buffer.Get();
  int64_t dim = dims.size();

  return std::tie(aclDesc, aclBuff, dim, format);
}

std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat>
OpCmdHelper::CovertToAclOutput(const Tensor* tensorPtr, const string& forceDataType) {
  aclDataType aclDataType = CalcuOpUtil::convert_to_acl_data_type(
      tensorPtr->scalar_type(), forceDataType);
  const auto& npuDesc = tensorPtr->storage().get_npu_desc();
  const auto& dims = tensorPtr->sizes();
  auto& storageDims = npuDesc.storage_sizes_;
  AclTensorDescMaker desc;
  auto aclDesc = desc.Create(aclDataType, dims, npuDesc.origin_format_)
                      .SetFormat(npuDesc.npu_format_)
                      .SetShape(storageDims)
                      .Get();
  auto numel = prod_intlist(storageDims);
  AclTensorBufferMaker aclBuffer(tensorPtr, numel);
  auto aclBuff = aclBuffer.Get();
  int64_t storageDim = storageDims.size();
  return std::tie(aclDesc, aclBuff, storageDim, npuDesc.npu_format_);
}

} // npu
} // native
} // at
