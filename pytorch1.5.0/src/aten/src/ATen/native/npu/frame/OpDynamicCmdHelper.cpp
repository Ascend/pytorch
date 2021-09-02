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

#include "OpDynamicCmdHelper.h"
#include "ATen/native/npu/frame/FormatHelper.h"
#include "ATen/native/npu/frame/OpParamMaker.h"
#include "ATen/native/npu/frame/InferFormat.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
namespace npu {

aclTensorDesc* OpDynamicCmdHelper::CovertToAclInputDynamicCompileDesc(Tensor tensor,
    c10::optional<Tensor> cpu_tensor,
    string& dynamicKey,
    string descName,
    string forceDataType,
    shapeStrage strage) {
  ScalarType scalarDataType = tensor.scalar_type();
  aclDataType aclDataType =
      CalcuOpUtil::convert_to_acl_data_type(scalarDataType, forceDataType);
  auto npuDesc = tensor.storage().get_npu_desc();
  auto res = CreateDynamicCompilelDims(npuDesc, strage, true);
  auto dims = std::get<0>(res);
  auto storageDims = std::get<1>(res);
  auto ranges = std::get<2>(res);
  dynamicKey += to_string(aclDataType) + "_" + to_string(npuDesc.origin_format_) + "_" + to_string(npuDesc.npu_format_) + "_";
  dynamicKey += CreateShapeKey(dims, storageDims);
  IntArrayRef defaultDims(dims.data(), dims.size());

  AclTensorDescMaker desc;
  auto aclDesc = desc.Create(aclDataType, defaultDims, npuDesc.origin_format_)
                      .SetFormat(npuDesc.npu_format_)
                      .SetShape(storageDims)
                      .SetRange(ranges)
                      .SetName(descName)
                      .SetConstAttr(cpu_tensor)
                      .Get();
  return aclDesc;
}

aclTensorDesc* OpDynamicCmdHelper::CovertToAclInputConstDynamicCompileDesc(Tensor tensor,
    c10::optional<Tensor> cpu_tensor,
    string& dynamicKey,
    string descName,
    string forceDataType,
    shapeStrage strage) {
  ScalarType scalarDataType = tensor.scalar_type();
  aclDataType aclDataType =
      CalcuOpUtil::convert_to_acl_data_type(scalarDataType, forceDataType);
  auto npuDesc = tensor.storage().get_npu_desc();
  auto dims = npuDesc.base_sizes_;
  auto storageDims = npuDesc.storage_sizes_;
  
  SmallVector<int64_t, N> dimList;
  for (int64_t i = 0; i < cpu_tensor.value().numel(); ++i) {
    dimList.emplace_back(cpu_tensor.value()[i].item().toInt());
  }
  dynamicKey += to_string(aclDataType) + "_2_2_";
  dynamicKey += CreateConstShapeKey(dimList, strage);
  AclTensorDescMaker desc;
  auto aclDesc = desc.Create(aclDataType, dims, ACL_FORMAT_ND)
                      .SetFormat(ACL_FORMAT_ND)
                      .SetShape(storageDims)
                      .SetName(descName)
                      .SetConstAttr(cpu_tensor)
                      .Get();
  return aclDesc;
}

aclTensorDesc* OpDynamicCmdHelper::CovertToAclOutputDynamicCompileDesc(const Tensor* tensorPtr, 
    string forceDataType,
    shapeStrage strage,
    bool isDimZeroToOne) {
  aclDataType aclDataType = CalcuOpUtil::convert_to_acl_data_type(
      tensorPtr->scalar_type(), forceDataType);
  auto npuDesc = tensorPtr->storage().get_npu_desc();
  AclTensorDescMaker desc;
  auto res = CreateDynamicCompilelDims(npuDesc, strage, isDimZeroToOne);
  auto dims = std::get<0>(res);
  auto storageDims = std::get<1>(res);
  auto ranges = std::get<2>(res);
  IntArrayRef defaultDims(dims.data(), dims.size());

  auto aclDesc = desc.Create(aclDataType, defaultDims, npuDesc.origin_format_)
                      .SetFormat(npuDesc.npu_format_)
                      .SetShape(storageDims)
                      .SetRange(ranges)
                      .Get();
  return aclDesc;
}

std::tuple<SmallVector<int64_t, N>, SmallVector<int64_t, N>, SmallVector<int64_t, N>>
OpDynamicCmdHelper::CreateDynamicCompilelDims(NPUStorageDesc npuDesc, shapeStrage strage, bool isDimZeroToOne) {
  size_t dimsSize = ((npuDesc.base_sizes_.size() == 0) && (isDimZeroToOne == true)) ? 1 : npuDesc.base_sizes_.size();
  size_t storageSize = ((npuDesc.base_sizes_.size() == 0) && (isDimZeroToOne == true)) ? 1 : npuDesc.storage_sizes_.size();

  SmallVector<int64_t, N> shape(dimsSize, -1);
  SmallVector<int64_t, N> storageShape(storageSize, -1);
  if (npuDesc.npu_format_ == ACL_FORMAT_NC1HWC0) {
    storageShape[storageSize - 1] = 16;
  }

  if (strage != FIXED_NONE) {
    ShapeStrageMaker(npuDesc, shape, storageShape, strage);
  }

  SmallVector<int64_t, N> range(dimsSize * 2);
  for (int64_t k = 0; k < dimsSize * 2; k += 2) {
    range[k] = 1;
    range[k + 1] = -1;
  }
  
  return std::tie(shape, storageShape, range);
}

void OpDynamicCmdHelper::ShapeStrageMaker(
    NPUStorageDesc npuDesc,
    SmallVector<int64_t, N>& shape,
    SmallVector<int64_t, N>& storageShape,
    shapeStrage strage) {
  auto dims = npuDesc.base_sizes_;
  auto storageDims = npuDesc.storage_sizes_;

  if (strage == FIXED_ALL) {
    shape = dims;
    storageShape = storageDims;
  }

  if (strage == FIXED_C) {
    if (npuDesc.origin_format_ == ACL_FORMAT_NCHW) {
      shape[1] = dims[1];
      storageShape[1] = storageDims[1];
    }
    if (npuDesc.npu_format_ == ACL_FORMAT_NC1HWC0) {
      storageShape[4] = storageDims[4];
    }
  }
}

string OpDynamicCmdHelper::CreateConstShapeKey(
    SmallVector<int64_t, N> dimList,
    shapeStrage strage) {
  string shapeKey = "";
  if (strage == FIXED_CONST_VALUE) {
    shapeKey = CreateShapeKey(dimList, dimList);
  } else if (strage == FIXED_CONST_DIM) {
    SmallVector<int64_t, N> dim = {dimList.size()};
    shapeKey = CreateShapeKey(dim, dim);
  } else {
    std::stringstream msg;
    msg << __func__ << ":" << __FILE__ << ":" << __LINE__;
    TORCH_CHECK(0, msg.str() + ": const input not support the shapeStrage=" + to_string(strage));
  }
  shapeKey += "_isConst;";
  return shapeKey;
}

string OpDynamicCmdHelper::CreateShapeKey(
    SmallVector<int64_t, N> shape,
    SmallVector<int64_t, N> storageShape) {
  string shapeKey = "";
  std::for_each(shape.begin(), shape.end(), [&](int64_t dim){shapeKey += to_string(dim) + ",";});
  shapeKey += "_";
  std::for_each(storageShape.begin(), storageShape.end(), [&](int64_t dim){shapeKey += to_string(dim) + ",";});
  shapeKey += ";";
  return shapeKey;
}

std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat, aclTensorDesc*>
OpDynamicCmdHelper::CovertNPUTensorWithZeroDimToDynamicAclInput(const Tensor& tensor, string descName) {
  
  aclDataType aclDataType =
      CalcuOpUtil::convert_to_acl_data_type(tensor.scalar_type());
      
  SmallVector<int64_t, 5> dims = {1};
  SmallVector<int64_t, 5> storageDims = {1};

  AclTensorDescMaker desc;
  auto aclDesc = desc.Create(aclDataType, dims, ACL_FORMAT_ND)
                      .SetFormat(ACL_FORMAT_ND)
                      .SetShape(storageDims)
                      .SetName(descName)
                      .Get();
  
  SmallVector<int64_t, 5> compileDims = {-1};
  SmallVector<int64_t, 5> compileStorageDims = {-1};
  SmallVector<int64_t, 5> compileRange = {1, -1};
  AclTensorDescMaker compileDesc;
  auto aclCompileDesc = compileDesc.Create(aclDataType, compileDims, ACL_FORMAT_ND)
                      .SetFormat(ACL_FORMAT_ND)
                      .SetShape(compileStorageDims)
                      .SetName(descName)
                      .SetRange(compileRange)
                      .Get();

  int64_t numel = prod_intlist(storageDims);
  AclTensorBufferMaker buffer(tensor, numel);

  auto aclBuff = buffer.Get();
  int64_t storageDim = storageDims.size();
  aclFormat storageFormate = ACL_FORMAT_ND;
  return std::tie(aclDesc, aclBuff, storageDim, storageFormate, aclCompileDesc);
}

std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat, aclTensorDesc*>
OpDynamicCmdHelper::CovertTensorWithZeroDimToDynamicAclInput(const Tensor& tensor, ScalarType type) {
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
  
  SmallVector<int64_t, 5> dims = {1};
  SmallVector<int64_t, 5> storageDims = {1};

  AclTensorDescMaker desc;
  auto aclDesc = desc.Create(aclDataType, dims, ACL_FORMAT_ND)
                      .SetFormat(ACL_FORMAT_ND)
                      .SetShape(storageDims)
                      .Get();
  
  SmallVector<int64_t, 5> compileDims = {-1};
  SmallVector<int64_t, 5> compileStorageDims = {-1};  
  SmallVector<int64_t, 5> compileRange = {1, -1};
  AclTensorDescMaker compileDesc;
  auto aclCompileDesc = compileDesc.Create(aclDataType, compileDims, ACL_FORMAT_ND)
                      .SetFormat(ACL_FORMAT_ND)
                      .SetShape(compileStorageDims)
                      .SetRange(compileRange)
                      .Get();

  AclTensorBufferMaker buffer(aclInput);
  auto aclBuff = buffer.Get();
  int64_t storageDim = 1;
  aclFormat storageFormate = ACL_FORMAT_ND;
  return std::tie(aclDesc, aclBuff, storageDim, storageFormate, aclCompileDesc);
}

const aclTensorDesc** OpDynamicCmdHelper::ConvertTensorWithZeroDimToOneDim(const aclTensorDesc** descs, int num) {
  for (int i = 0; i < num; i++) {
    aclTensorDesc* desc = const_cast<aclTensorDesc*>(descs[i]);
    int dims = (int)aclGetTensorDescNumDims(desc);
    if (dims == 0){
      aclDataType dtype = aclGetTensorDescType(desc);
      aclFormat format = aclGetTensorDescFormat(desc);
      dims = 1;
      int storageDims = 1;
      
      std::vector<int64_t> desc_dims(dims, 1);
      std::vector<int64_t> storage_desc_dims(storageDims, 1);
        
      aclTensorDesc* new_desc = aclCreateTensorDesc(dtype, dims, desc_dims.data(), format);
      aclSetTensorFormat(new_desc, format);
      aclSetTensorShape(new_desc, storageDims, storage_desc_dims.data());
        
      aclDestroyTensorDesc(descs[i]);
      descs[i] = new_desc;
    }
  }

  return descs;
}

std::tuple<string, int, const aclTensorDesc**, int, const aclTensorDesc**, const aclopAttr*>
OpDynamicCmdHelper::CreateDynamicCompileParams(ExecuteParas& params) {
  if (params.opDynamicType != "") {
    return std::tie(params.opDynamicType,
      params.dynamicParam.input_num,
      params.dynamicParam.compile_input_desc,
      params.dynamicParam.output_num,
      params.dynamicParam.compile_output_desc,
      params.dynamicCompileAttr);
  } else { 
    return std::tie(params.opType,
      params.paras.input_num,
      params.paras.input_desc,
      params.paras.output_num,
      params.paras.output_desc,
      params.attr);
  }
}

std::tuple<string, int, const aclTensorDesc**, const aclDataBuffer**, int, const aclTensorDesc**, aclDataBuffer**, const aclopAttr*>
OpDynamicCmdHelper::CreateDynamicRunParams(ExecuteParas& params) {
  if (params.opDynamicType != "") {
    return std::tie(params.opDynamicType,
      params.dynamicParam.input_num,
      params.dynamicParam.input_desc,
      params.dynamicParam.input_data_buf,
      params.dynamicParam.output_num,
      params.dynamicParam.output_desc,
      params.dynamicParam.output_data_buf,
      params.dynamicRunAttr);
  } else {
    params.paras.input_desc 
      = ConvertTensorWithZeroDimToOneDim(params.paras.input_desc, params.paras.input_num);
    params.paras.output_desc 
      = ConvertTensorWithZeroDimToOneDim(params.paras.output_desc, params.paras.output_num);

    return std::tie(params.opType,
      params.paras.input_num,
      params.paras.input_desc,
      params.paras.input_data_buf,
      params.paras.output_num,
      params.paras.output_desc,
      params.paras.output_data_buf,
      params.attr);
  }
}

} // npu
} // native
} // at
