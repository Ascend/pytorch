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

#include <mutex>
#include "NpuUtils.h"
#include "c10/npu/register/OptionRegister.h"
#include "c10/npu/interface/AsyncTaskQueueInterface.h"

#include "CalcuOpUtil.h"
#include "ATen/native/npu/frame/FormatHelper.h"
#include "ATen/native/npu/frame/StorageDescHelper.h"
#include "KernelNpuOutputSize.h"
#include <ATen/native/npu/contiguous/ContiguousOpt.h>
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/interface/EnvVariables.h"
#include <set>

namespace at {
namespace native {
namespace npu {

void NpuUtils::format_fresh_view(
    Tensor& x,
    const Tensor& y) {
  // x:NPU before inplace_op, y: NPU computed
  // now we fresh x according to y
  RECORD_HOST_FUNCTION("format_fresh_view", vector<c10::IValue>({x, y}));

  x.copy_(y);
}


// NOTE [Check Match for Npu Tensor]
// check_match is used to ensure that npu tensor satisfies the
// calculation requirements of npu operators.
// The rules are as follows:
// 1、tensor should be contiguous
// Not contiguous means the operator needs to read and write memory
// at intervals according to strides and sizes. Npu operators has
// no such ability for the time being
// 2、metadata should be match
// Resize_ a contiguous cpu tensor from [1,2,3,4] to [4,3,2,1] no
// need to change the physical memory. However, for a contiguous npu
// tensor whose npu_format_ is 5HD, storage shape should be change
// from [1,1,3,4,16] to [4,1,2,1,16]. So metadata not match often
// results in unexpected physical memory. format_contiguous will be
// called preparing correct memory of operand in these case.
bool NpuUtils::check_match(const Tensor* tensor) {
  // case1:uncontiguous tensor
  if (!tensor->is_contiguous()) {
    return false;
  }

  // case2:meta data not match, sizes or strides of presentation
  // layer is different from that of storage layer
  if (!StorageDescHelper::MetaDataAreMatch(tensor)) {
    return false;
  }

  // case3:meta data not match, storage_offset of presentation layer
  // is different from that of storage layer
  bool isPadding = FormatHelper::IsPadded(tensor);
  if (isPadding && (!StorageDescHelper::OffsetAreMatch(tensor))) {
    return false;
  }
  return true;
}

bool NpuUtils::check_5d_5d_match(const Tensor& tensor){
  // (1) NC1HWC0 format in storage, NCHW format in des.
  // (2) 4d format situation, only uncontiguous in Channel size
  // (3) size and start point must be 16*, make sure the memory be contiguous
  // std::cout<<"step in check5d5d Match."<<std::endl;
  const c10::Storage storage = tensor.storage();
  const c10::NPUStorageDesc npuDesc = storage.get_npu_desc();

  if (tensor.is_contiguous()) {
    return false;
  }

  if(npuDesc.npu_format_ != ACL_FORMAT_NC1HWC0){
      return false;
  }

  if(tensor.sizes().size() != 4){
      return false;
  }

  bool is_c_channel_slice = true;
  int64_t z = 1;
  for (int64_t d = tensor.dim() - 1; d >= 1; d--){
  if(tensor.size(d) != 1){
      if(tensor.stride(d) == z){
      z*=tensor.size(d);
      }else{
      is_c_channel_slice=false;
      break;
      }
  }
  }
  if(!is_c_channel_slice){
      return false;
  }

  int64_t contiguous_len = 16;
  int64_t c0_len = 16;
  for(int i = 2; i < npuDesc.base_sizes_.size(); i++){
      contiguous_len *= npuDesc.base_sizes_[i];
      }
  bool is_offset_match = (tensor.storage_offset() % contiguous_len==0);
  bool is_length_match = (tensor.size(1) % c0_len==0);

  return is_offset_match && is_length_match;
}
// src will be modified, can not use &
Tensor convert_continue_using_gatherv2_improve(Tensor& src){
  // ref: IndexSelectKernelNpu.cpp
  // std::cout<<"step in convert_continue_using_gatherv2."<<std::endl;
  RECORD_HOST_FUNCTION("continue_by_gatherv2_improve", vector<c10::IValue>({src}));
  // 1. get gatherv2 start index and end index
  int64_t start = src.storage_offset()/(src.size(2)*src.size(3))/16;
  int64_t end = start+src.size(1)/16;
  Tensor index = arange(start,end).to(at::kNPU);
  int64_t dim = 1;

  // 2. recovery the src tensor desc
  const c10::NPUStorageDesc src_npuDesc = src.storage().get_npu_desc();
  src.set_(src.storage(), 0, src_npuDesc.base_sizes_, src_npuDesc.base_strides_);
  Tensor src_tmp = src.reshape({src.size(0),src.size(1)/16,src.size(2),src.size(3)*16});
  src_tmp.storage().unsafeGetStorageImpl()->npu_desc_.base_sizes_ = src_tmp.sizes();
  src_tmp.storage().unsafeGetStorageImpl()->npu_desc_.base_strides_ = src_tmp.strides();
  src_tmp.storage().unsafeGetStorageImpl()->npu_desc_.storage_sizes_ = src_tmp.sizes();
  // std::cout << "src_tmp storage_offset(): " << src_tmp.storage_offset() << std::endl;
  // std::cout << "src_tmp sizes(): " << src_tmp.sizes() << std::endl;
  // std::cout << "src_tmp strides(): " << src_tmp.strides() << std::endl;
  // std::cout << "src_tmp data_recovery: " << src_tmp.to(at::kCPU) << std::endl;

  // 3. get output size
  auto outputSize = index_select_npu_output_size(src_tmp, dim, index);
  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(src_tmp);
  Tensor result = at::empty_with_format(outputSize, src_tmp.options(), npu_format);
  // std::cout << "npu_format: " << npu_format << std::endl;

  // 4. get input and output
  SmallVector<Tensor, N> inputTensor = {src_tmp, index};
  SmallVector<NPUTensorDesc, N> inputs;
  for (int i = 0; i < inputTensor.size(); i++) {
    inputs.emplace_back(
        NPUTensorDesc(inputTensor[i]));
    if (inputTensor[i].dim() == 0) {
      inputs[i].tensorDescType = NPUTensorDesc::TensorDescType::TENSOR_SCALAR;
    }
  }

  auto outputs = CalcuOpUtil::create_npu_input_tensor_desc({result});

  NPUAttrDesc npuAttrAxis = NPUAttrDesc("axis", dim);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrAxis};

  // 5. run
  CalcuOpUtil::execute_npu_operate("GatherV2D", inputs, outputs, attrs);
  return result;
}
void NpuUtils::RefreshFormat(const Tensor& tensor) {
  auto& tensor_desc = tensor.storage().unsafeGetStorageImpl()->npu_desc_;
  if (tensor_desc.storage_sizes_.size() == 4 && tensor_desc.npu_format_ == ACL_FORMAT_ND) {
    tensor_desc.npu_format_ = ACL_FORMAT_NCHW;
    tensor_desc.origin_format_ = ACL_FORMAT_NCHW;
  } else if (tensor_desc.storage_sizes_.size() != 4 && tensor_desc.npu_format_ == ACL_FORMAT_NCHW) {
    tensor_desc.npu_format_ = ACL_FORMAT_ND;
    tensor_desc.origin_format_ = ACL_FORMAT_ND;
  }
}

Tensor deal_with_5d_5d_match(const Tensor& src) {
    auto src_desc = src.storage().unsafeGetStorageImpl()->npu_desc_;
    Tensor src_new = at::empty_with_format(src_desc.base_sizes_, src.options(), ACL_FORMAT_NC1HWC0);
    int64_t numel = src_new.numel();
    aclError error = c10::npu::queue::LaunchAsyncCopyTask(
        src_new.data_ptr(),
        numel * src_new.element_size(),
        (uint8_t*)src.data_ptr() - src.storage_offset() * src.element_size(),
        numel * src.element_size(),
        ACL_MEMCPY_DEVICE_TO_DEVICE);
    src_new.set_(src_new.storage(), src.storage_offset(), src.sizes(), src.strides());

    src_new.storage().unsafeGetStorageImpl()->npu_desc_.npu_format_ = ACL_FORMAT_NCHW;
    Tensor ret = convert_continue_using_gatherv2_improve(src_new);
    // std::cout << "ret data_recovery: " << ret.to(at::kCPU) << std::endl;
    Tensor ret_tmp = ret.reshape({ret.size(0),ret.size(1)*16,ret.size(2),ret.size(3)/16});
    ret_tmp.storage().unsafeGetStorageImpl()->npu_desc_.base_sizes_ = ret_tmp.sizes();
    ret_tmp.storage().unsafeGetStorageImpl()->npu_desc_.base_strides_ = ret_tmp.strides();
    ret_tmp.storage().unsafeGetStorageImpl()->npu_desc_.storage_sizes_ = ret_tmp.sizes();
    ret_tmp.storage().unsafeGetStorageImpl()->npu_desc_.npu_format_ = ACL_FORMAT_NC1HWC0;
    return ret_tmp;
}

Tensor metadata_convert_match(const Tensor& src) {
  auto& src_desc = src.storage().unsafeGetStorageImpl()->npu_desc_;
  bool numelEq = (src.numel() == prod_intlist(src_desc.base_sizes_));
  // Only when a tensor monopolizes a storage can NpuStorageDesc be refreshed.
  // When the original format is not NCHW, the npu_format_cast to NCHW will generate
  // a temporary tensor, which always monopolizes its own storage.
  if (numelEq && (!FormatHelper::IsBaseFormatType(src))) {
    Tensor tempTensor = at::npu_format_cast(src, FormatHelper::GetBaseFormat(src));
    at::npu_reshape_out(tempTensor, tempTensor, tempTensor.sizes(), true);
    NpuUtils::RefreshFormat(tempTensor);
    return tempTensor;
  } else {
    Tensor contiguous_view = at::empty(src.sizes(), src.options());
    contiguous_view.copy_(src);
    NpuUtils::RefreshFormat(contiguous_view);
    return contiguous_view;
  }
}

Tensor metadata_with_offset_padding_convert_match(const Tensor& src) {
  Tensor contiguous_view = at::empty(src.sizes(), src.options());
  contiguous_view.copy_(src);
  NpuUtils::RefreshFormat(contiguous_view);
  return contiguous_view;
}

Tensor NpuUtils::format_contiguous(const Tensor& src) {
  // case1:tensor src is not contiguous
  if (!src.is_contiguous()) {
    RECORD_HOST_FUNCTION("format_contiguous", vector<c10::IValue>({src}));
    return src.contiguous();
  }
  // case2:meta data not match, sizes or strides of presentation
  // layer is different from that of storage layer
  if (!StorageDescHelper::MetaDataAreMatch(&src)) {
    // Fix not match case2, tensor should have matched metadata and NPUStorageDesc.
    RECORD_HOST_FUNCTION("format_contiguous", vector<c10::IValue>({src}));
    return metadata_convert_match(src);
  }

  // case3:meta data not match, storage_offset of presentation layer
  // is different from that of storage layer
  if (FormatHelper::IsPadded(&src) && (!StorageDescHelper::OffsetAreMatch(&src))) {
    // Fix not match case3, tensor with padding should not have storage-offset.
    RECORD_HOST_FUNCTION("format_contiguous", vector<c10::IValue>({src}));
    return metadata_with_offset_padding_convert_match(src);
  }

  return src;
}

Tensor NpuUtils::format_contiguous_add_copy_optimize(const Tensor& src) {
  // case1:tensor src is not contiguous
  if (!src.is_contiguous()) {
    RECORD_HOST_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
    return src.contiguous();
  }
  // case2:meta data not match, sizes or strides of presentation
  // layer is different from that of storage layer
  if (!StorageDescHelper::MetaDataAreMatch(&src)) {
    // Fix not match case2, tensor should have matched metadata and NPUStorageDesc.
    RECORD_HOST_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
    // copy optimize for reshape cases with 3 choices
    // [1] memory-repoint: base format or NZ[1. key dims keep matched; 2. no padding]
    // [2] d2dCopyAsync: base format or NZ[key dims keep matched]
    // [3] copy_: Universal method
    OptimizationCases optimizations_reshape{"reshapeV2"};
    auto reshapeTensor =
        TransContiguous::ContiguousOptimizeWithAnyFormat(src, optimizations_reshape);
    if (reshapeTensor.has_value()) {
      return reshapeTensor.value();
    }
    return metadata_convert_match(src);
  }

  // case3:meta data not match, storage_offset of presentation layer
  // is different from that of storage layer
  if (FormatHelper::IsPadded(&src) && (!StorageDescHelper::OffsetAreMatch(&src))) {
    // Fix not match case3, tensor with padding should not have storage-offset.
    RECORD_HOST_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
    return metadata_with_offset_padding_convert_match(src);
  }

  return src;
}

bool NpuUtils::IsOomError(aclError ret, int index)
{
  if (ret == ACL_ERROR_GE_DEVICE_MEMORY_ALLOCATION_FAILED) {
    int deviceId = 0;
    // free devcie cached memory when return value of the first op execution is oom
    if (index == 1) {
      C10_NPU_CHECK(aclrtGetDevice(&deviceId));
      c10::npu::NPUCachingAllocator::FreeDeviceCachedMemory(deviceId);
      return true;
    }
    AT_ERROR("NPU out of memory. device id: ", deviceId);
  }
  return false;
}

std::string NpuUtils::get_reduction_str(int64_t reduction) {
  string reductionStr;
  if (reduction == Reduction::Mean) {
    reductionStr = "mean";
  } else if (reduction == Reduction::Sum) {
    reductionStr = "sum";
  } else {
    reductionStr = "none";
  }
  return reductionStr;
}

} // namespace npu
} // namespace native
} // namespace at
