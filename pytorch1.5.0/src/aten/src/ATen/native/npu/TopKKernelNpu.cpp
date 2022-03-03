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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

namespace {
bool is_large_topk(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  // 当前aicore支持大规模topk的触发条件为：输入tensor为1维，dtype为fp16，size大于300000，k值大于7936
  if (self.dtype() == at::kHalf &&
      self.dim() == 1 &&
      self.size(0) > 300000 &&
      k > 7936) {
    return true;
  } else {
    return false;
  }
} // is_large_topk

SmallVector<int64_t, SIZE> segment_sort_npu_output_size(
    const Tensor& self,
    int64_t k) {
  int64_t core_num = 32;
  int64_t merge_channel_num = 4;
  int64_t tail_num =16;
  int64_t data_num = self.size(0);
  int64_t each_core_align_num = 1984;
  int64_t each_core_min_num = 7936;
  int64_t each_core_data_num = CeilDiv(data_num, core_num);
  int64_t each_core_proposal_num = 
      CeilDiv(each_core_data_num, each_core_align_num) * each_core_align_num > each_core_min_num?
      CeilDiv(each_core_data_num, each_core_align_num) * each_core_align_num : each_core_min_num;
  int64_t core_num_use = CeilDiv(data_num, each_core_proposal_num);
  if (core_num_use > merge_channel_num) {
    core_num_use = CeilDiv(core_num_use, merge_channel_num) * merge_channel_num;
  }
  each_core_proposal_num += tail_num;
  SmallVector<int64_t, SIZE> outputsize = {core_num_use, each_core_proposal_num, 8};
  return outputsize;
} // segment_sort_npu_output_size

SmallVector<int64_t, SIZE> multi_merge_npu_output_size(
    const Tensor& self,
    int64_t k) {
  int64_t merge_num = 4;
  int64_t merge_channel_num = 4;
  int64_t channel_num = self.size(0);
  int64_t proposal_repeat_num = 16;
  int64_t sorted_num_align = CeilDiv(k,proposal_repeat_num) * proposal_repeat_num;
  int64_t ai_core_num = CeilDiv(channel_num, merge_num);
  int64_t sorted_num = ((self.size(1) - proposal_repeat_num) * merge_num < sorted_num_align?
      (self.size(1) - proposal_repeat_num) * merge_num : sorted_num_align) + proposal_repeat_num;
  if (ai_core_num > merge_channel_num) {
    ai_core_num = CeilDiv(ai_core_num, merge_channel_num) * merge_channel_num;
  }
  SmallVector<int64_t, SIZE> outputsize = {ai_core_num, sorted_num, 8};
  return outputsize;
} // multi_merge_npu_output_size

} // namespace

tuple<Tensor&, Tensor&> topk_out_npu_no_transpose(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  SmallVector<int64_t, N> kVec = {k};
  Tensor kCpuTensor = from_blob((void*)kVec.data(), {1}, at::kLong).to(at::kInt);
  if (!c10::npu::OptionsManager::CheckDynamicEnable()){
    OpCommand cmd;
    cmd.Name("TopKV2")
      .Input(self)
      .Input(kCpuTensor, kVec, "k")
      .Output(values)
      .Output(indices)
      .Attr("dim", dim)
      .Attr("largest", largest)
      .Attr("sorted", sorted)
      .Run();
  } else{
    OpDynamicCommand cmd;
    // Although the value is fixed to false, only the value of sorted can be true.
    cmd.Name("TopKV2")
      .Input(self)
      .Input(kCpuTensor, kVec, "k")
      .Output(values)
      .Output(indices)
      .Attr("dim", dim)
      .Attr("largest", largest)
      .Attr("sorted", sorted);
    cmd.DynamicName("TopKV2")
        .DynamicInput(self)
        .DynamicInput(kVec, at::kLong, at::kInt, "k")
        .DynamicOutput(values)
        .DynamicOutput(indices)
        .DynamicAttr("dim", dim)
        .DynamicAttr("largest", largest)
        .DynamicAttr("sorted", sorted)
        .DynamicOpRun();
  }
  return tuple<Tensor&, Tensor&>(values, indices);
}

Tensor& segment_sort_out_npu(
    Tensor& result,
    const Tensor& self,
    int64_t k) {
  Tensor inputIndex = at::range(0, 2047, 1).to("npu").to(at::kHalf);
  OpCommand cmd;
  cmd.Name("SegmentSort")
      .Input(self)
      .Input(inputIndex)
      .Output(result)
      .Attr("k_num", k)
      .Run();
  return result;
}

tuple<Tensor&, Tensor&> multi_merge_out_npu(
    Tensor& result,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    bool include_index) {
  OpCommand cmd;
  cmd.Name("MultiMerge")
      .Input(self)
      .Output(result)
      .Output(indices)
      .Attr("k_num", k)
      .Attr("include_index", include_index)
      .Run();
  return std::tie(result, indices);
}

tuple<Tensor&, Tensor&> large_topk_out_npu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  auto outputsizeOfSegmentSort = segment_sort_npu_output_size(self, k);
  Tensor resultOfSegmentSort = OpPreparation::ApplyTensorWithFormat(
      outputsizeOfSegmentSort, self.options(), ACL_FORMAT_ND);
  segment_sort_out_npu(resultOfSegmentSort, self, k);

  auto outputsizeOfMultiMerge1 = multi_merge_npu_output_size(resultOfSegmentSort, k);
  Tensor resultOfMultiMerge1 = OpPreparation::ApplyTensorWithFormat(
      outputsizeOfMultiMerge1, resultOfSegmentSort.options(), ACL_FORMAT_ND);
  Tensor resultIndicesOfMultiMerge1 = OpPreparation::ApplyTensorWithFormat(
      {1}, resultOfSegmentSort.options().dtype(kInt), ACL_FORMAT_ND);
  multi_merge_out_npu(resultOfMultiMerge1, resultIndicesOfMultiMerge1, resultOfSegmentSort, k, false);

  auto outputsizeOfMultiMerge2 = multi_merge_npu_output_size(resultOfMultiMerge1, k);
  Tensor resultOfMultiMerge2 = OpPreparation::ApplyTensorWithFormat(
      outputsizeOfMultiMerge2, resultOfMultiMerge1.options(), ACL_FORMAT_ND);
  Tensor resultIndicesOfMultiMerge2 = OpPreparation::ApplyTensorWithFormat(
      {1}, resultOfSegmentSort.options().dtype(kInt), ACL_FORMAT_ND);
  multi_merge_out_npu(resultOfMultiMerge2, resultIndicesOfMultiMerge2, resultOfMultiMerge1, k, false);

  multi_merge_out_npu(values, indices, resultOfMultiMerge2, k, true);

  return std::tie(values, indices);
}

tuple<Tensor&, Tensor&> topk_out_npu_nocheck(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  // aicore support large topk scenario
  if (is_large_topk(self, k, dim, largest, sorted)) {
    large_topk_out_npu(values, indices, self, k, dim, largest, sorted);
    return std::tie(values, indices);
  }
  
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  int64_t lastDim = CalcuOpUtil::make_wrap_dim(-1, self.dim());

  if (dim != lastDim) {
    SmallVector<int64_t, SHAPE_SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[lastDim]);

    // construct the output tensor of the NPU
    Tensor transposeSelf = at::npu_transpose(self, perm);
    auto outputSize = transpose_npu_output_size(values, perm);
    Tensor transposeValue = at::empty_with_format(
        outputSize,
        values.options(),
        CalcuOpUtil::get_tensor_npu_format(values));
    Tensor transposeIndices = at::empty_with_format(
        outputSize,
        indices.options(),
        CalcuOpUtil::get_tensor_npu_format(indices));
    topk_out_npu_no_transpose(
        transposeValue,
        transposeIndices,
        transposeSelf,
        k,
        lastDim,
        largest,
        sorted);
    at::npu_transpose_out(values, transposeValue, perm);
    at::npu_transpose_out(indices, transposeIndices, perm);
  } else {
    topk_out_npu_no_transpose(
        values, indices, self, k, lastDim, largest, sorted);
  }

  return tuple<Tensor&, Tensor&>(values, indices);
}

tuple<Tensor&, Tensor&> topk_out_npu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  Tensor selfCp = OpPreparation::CastBackToOriFormat(self);

  // calculate the output size
  auto outputSize = topk_npu_output_size(selfCp, k, dim, largest, sorted);
  SmallVector<int64_t, SIZE> indicesSize = outputSize;

  // calculate the output result of the NPU
  auto func = [&selfCp, k, dim, largest, sorted](Tensor& values, Tensor& indices) {
    topk_out_npu_nocheck(values, indices, selfCp, k, dim, largest, sorted);
  };

  Tensor indices_tmp;
  OpPipeWithMultiOut<Tensor&, Tensor&> pipe(values, indices_tmp);
  return pipe.FixOutputSizeAndFormat<0>({selfCp}, selfCp, CalcuOpUtil::get_tensor_npu_format(selfCp), outputSize)
      .ApplyOutputWithSpecailParams<1>(indicesSize, selfCp.options().dtype(kInt), ACL_FORMAT_ND)
      .Call(func)
      .ReflushOutputDtype<1>(ScalarType::Long)
      .FixOutputExceptDtype<1>({selfCp}, ACL_FORMAT_ND, ScalarType::Long, indicesSize)
      .FixOutputWithReplace<1>(indices)
      .ReturnRef<Tensor&, Tensor&>();
}

tuple<Tensor, Tensor> topk_npu(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  Tensor selfCp = OpPreparation::CastBackToOriFormat(self);
  // calculate the output size
  auto outputSize = topk_npu_output_size(selfCp, k, dim, largest, sorted);
  // construct the output tensor of the NPU
  Tensor values = at::empty_with_format(
      outputSize, selfCp.options(), CalcuOpUtil::get_tensor_npu_format(selfCp));
  Tensor indices = at::empty_with_format(
      outputSize, selfCp.options().dtype(kInt), ACL_FORMAT_ND);

  // calculate the output result of the NPU
  topk_out_npu_nocheck(values, indices, selfCp, k, dim, largest, sorted);

  // indices dtype transform Int64
  indices = indices.to(at::kLong);

  return tuple<Tensor, Tensor>(values, indices);
}

} // namespace native
} // namespace at
