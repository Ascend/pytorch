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

#include <ATen/NamedTensorUtils.h>
#include <ATen/native/npu/contiguous/ContiguousOpt.h>
#include <ATen/native/npu/utils/KernelNpuOutputSize.h>
#include <ATen/quantized/QTensorImpl.h>
#include <map>

namespace at {
namespace native {
namespace npu {

class CombinedContiguousOpt : public ContiguousOpt {
 public:
  // Combined tensor == discontiguous tensor caused by combined view operators.
  bool Optimizer(const Tensor& src, Tensor& self) override {
    // Maximum combined operators suggested: Maxlen = 2
    // NOTE: n-cmobined(n>2) can also be supported
    int64_t maxLen = 2;

    // Setting for 3-combined cases: "TRI_COMBINED_ENABLE=1".
    if (c10::npu::OptionsManager::CheckTriCombinedOptimizerEnable()) {
      maxLen = 3;
    }

    // Stacks used for storing inferred infos about shape, stride, offset
    // "viewInfos": {{{shape1},{stride1}};{{shape2},{stride2}};...}
    // "viewOffsets": {storage_offset1, storage_offset2,...}
    SmallVector<SmallVector<FormatShape, 2>, 4> viewInfos;
    SmallVector<int64_t, 4> viewOffsets;

    if (can_use_combined(src, viewInfos, viewOffsets, maxLen)) {
      RECORD_FUNCTION("npuCombined", std::vector<c10::IValue>({src}));

      // Record src infos for recovering after trans-contiguous
      auto src_npu_desc = src.storage().get_npu_desc();

      // Construct base tensor(contiguous)
      Tensor base_tensor = at::empty(src_npu_desc.base_sizes_, src.options());
      base_tensor.set_(src.storage());

      // Reconstruct combined discontiguous tensor ==trans==> contiguous tensor
      bool contiguousOrNot =
          combined_to_contiguous(base_tensor, self, viewInfos, viewOffsets);

      // Recover modified tensor infos of src after trans-contiguous
      StorageDescHelper::CopyDesc(base_tensor, src_npu_desc);
      return contiguousOrNot;
    }
    return false;
  }

 private:
  bool cases_avoid(const Tensor& tensor) {
    for (auto i = 0; i < tensor.sizes().size(); i++) {
      // expand+x,x+expand
      if (tensor.stride(i) == 0) {
        return true;
      }
    }
    return false;
  }

  // Unmatched tensor ==refresh(no copy)==> macthed tensor
  bool reshape_without_copy_match(Tensor& tensor) {
    if (!tensor.is_contiguous()) {
      return false;
    }
    auto npu_desc = tensor.storage().get_npu_desc();

    if ((prod_intlist(tensor.sizes()) != prod_intlist(npu_desc.base_sizes_)) ||
        (tensor.storage_offset() != npu_desc.base_offset_)) {
      return false;
    }
    RECORD_FUNCTION("npuMatch", std::vector<c10::IValue>({tensor}));
    StorageDescHelper::SetDesc(
        tensor,
        array_to_small_vector(tensor.sizes()),
        array_to_small_vector(tensor.strides()));
    return true;
  }

  // Whether tensor can be optimized(no optimization).
  bool can_be_optimize_from_default_cases(const Tensor& tensor) {
    std::vector<string> optimizations{"reshape", "slice", "select"};
    return TransContiguous::CanOptimize(tensor, optimizations);
  }

  // Conduct trans-contiguous for given optimization cases.
  bool copy_optimize_contiguous_by_given_cases(
      const Tensor& tensor,
      Tensor& self,
      std::vector<string> optimizations) {
    // Set "OpenCombined = false" to avoid recursion.
    return TransContiguous::ContiguousOptimizeWithBaseFormat(
        self, tensor, optimizations, false);
  }

  // Weak constrains for transpose case
  bool maybe_permute(const Tensor& tensor) {
    // tensors with nonmonotonic strides will be taken into consideration
    // TODO: 对于特殊stride的情况例如：[*,*,1,1]这种，需要进一步分析影响
    for (auto i = 0; i < tensor.strides().size() - 1; i++) {
      if (tensor.stride(i) < tensor.stride(i + 1)) {
        return true;
      }
    }
    return false;
  }

  bool maybe_select(const Tensor& tensor) {
    for (auto i = tensor.dim() - 1; i > 0; i--) {
      if (tensor.strides()[i - 1] % (tensor.sizes()[i] * tensor.strides()[i]) != 0) {
        return false;
      }
      if (tensor.strides()[i - 1] / (tensor.sizes()[i] * tensor.strides()[i]) != 1) {
        if (tensor.storage_offset() %
                (tensor.sizes()[i] * tensor.strides()[i]) != 0) {
          return false;
        }
        // Avoid combined-cases such as squeeze+indexing at the first axis.
        if(tensor.strides()[0] != tensor.storage().get_npu_desc().base_strides_[0]){
          return false;
        }
      }
    }
    return true;
  }

  bool maybe_slice(const Tensor& tensor) {
    // tensors with reduced numel will be taken into consideration.
    if (prod_intlist(tensor.sizes()) <
        prod_intlist(tensor.storage().get_npu_desc().base_sizes_)) {
      for (auto i = 0; i < tensor.sizes().size() - 2; i++) {
        if (tensor.strides()[i] % tensor.strides()[i + 1] != 0) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  /*
  Kernel function of "Inference",
  Key inferred infos: infer_size,infer_stride and infer_offset,
  Inference order: permute, select, slice.
  */
  bool can_infer_view_tensor(
      const Tensor& src,
      Tensor& tensor,
      FormatShape& infer_size,
      FormatShape& infer_stride,
      int64_t& infer_offset) {
    auto base_sizes = src.storage().get_npu_desc().base_sizes_;
    auto base_strides = src.storage().get_npu_desc().base_strides_;
    auto view_sizes = array_to_small_vector(src.sizes());
    auto view_strides = array_to_small_vector(src.strides());

    if (maybe_permute(src)) {
      FormatShape& permute_size_sorted = infer_size;
      FormatShape& permute_stride_sorted = infer_stride;
      permute_size_sorted = view_sizes;
      permute_stride_sorted = view_strides;

      // Sort stride
      std::sort(permute_stride_sorted.rbegin(), permute_stride_sorted.rend());

      // Map stride to shape
      std::map<int64_t, int64_t> map_shape_stride;
      std::map<int64_t, int64_t> label_map_shape_stride;
      for (auto i = 0; i < view_sizes.size(); i++) {
        map_shape_stride[view_strides[i]] = view_sizes[i];
      }
      //除去第0维，其他维shape为1时，不记录对应的stride值，该stride的值会和其他维的stride有重复
      for (auto i = 0; i < view_sizes.size(); i++) {
        if (i == 0) {
          map_shape_stride[view_strides[0]] = view_sizes[0];
        } else if (i != 0 && view_sizes[i] != 1) {
          map_shape_stride[view_strides[i]] = view_sizes[i];
        }
      }
      // stride中有相等的情况，后面相等的stride对应的shape为1
      for (auto i = 0; i < view_sizes.size(); i++) {
        if (label_map_shape_stride[permute_stride_sorted[i]] != true) {
          permute_size_sorted[i] = map_shape_stride[permute_stride_sorted[i]];
          label_map_shape_stride[permute_stride_sorted[i]] = true;
        } else {
          permute_size_sorted[i] = 1;
        }
      }

      // Refresh tensor's base info to construct transposed tensor
      StorageDescHelper::SetDesc(
          tensor, permute_size_sorted, permute_stride_sorted);

      infer_offset = 0;
      // Whether the construted tensor is transposed?
      return maybe_permute(tensor);
    }

    if (maybe_select(src)) {
      FormatShape& select_size = infer_size;
      FormatShape& select_stride = infer_stride;
      // Infer base shape according to view shape and stride
      select_stride = view_strides;
      select_size = view_sizes;
      // select_size and stride should be one more than view_size
      select_size.emplace_back((int64_t)1);
      select_stride.emplace_back((int64_t)1);

      int64_t i = view_sizes.size() - 1;
      if (view_strides[i] == 1) {
        select_size[i + 1] = view_sizes[i];
        select_stride[i + 1] = 1;

        for (i = i - 1; i >= 0; i--) {
          if (view_strides[i] != view_strides[i + 1] * view_sizes[i + 1]) {
            select_size[i + 1] =
                view_strides[i] / (view_sizes[i + 1] * view_strides[i + 1]);
            select_stride[i + 1] = view_sizes[i + 1] * view_strides[i + 1];
            infer_offset = src.storage_offset() % view_strides[i];
            break;
          }
          select_size[i + 1] = view_sizes[i];
          select_stride[i + 1] = view_strides[i];
        }
      } else {
        select_size[i + 1] = view_strides[i];
        select_stride[i + 1] = 1;
        infer_offset = src.storage_offset() % view_strides[i];
      }
      for (i = i - 1; i >= 0; i--) {
        select_size[i + 1] = view_sizes[i + 1];
        select_stride[i + 1] = view_strides[i + 1];
      }

      select_size[0] = view_sizes[0];
      select_stride[0] = view_strides[0];

      // Refresh tensor's base info to construct selected tensor
      StorageDescHelper::SetDesc(tensor, select_size, select_stride);
      // Whether the construted tensor is selected?
      return maybe_select(tensor);
    }

    if (maybe_slice(src)) {
      FormatShape& slice_size = infer_size;
      FormatShape& slice_stride = infer_stride;

      slice_stride = view_strides;
      slice_size = view_sizes;
      // Infer base shape according to base stride
      for (auto i = slice_size.size() - 1; i > 0; i--) {
        // Strides is not divisible means this case cannot be inferred.
        if (view_strides[i] == 0 ||
            view_strides[i - 1] % view_strides[i] != 0) {
          return false;
        }
        slice_size[i] = (view_strides[i - 1] / view_strides[i]);
      }
      slice_size[0] = 1;
      slice_size[0] = (prod_intlist(base_sizes) / prod_intlist(slice_size));

      // Refresh tensor's base info and storage info to construct sliced tensor
      StorageDescHelper::SetDesc(tensor, slice_size, slice_stride);
      infer_offset = src.storage_offset();
      // Whether the construted tensor is sliced?
      return maybe_slice(tensor);
    }
    return false;
  }

  bool emplace_info(
      Tensor& tensor,
      SmallVector<SmallVector<FormatShape, 2>, 4>& view_infos,
      SmallVector<int64_t, 4>& view_offsets,
      int64_t infer_offset,
      int64_t max_len) {
    // Only max_len-combined Ops cases are taken into consideration
    if (view_infos.size() == max_len) {
      return false;
    }
    auto tensor_desc = tensor.storage().get_npu_desc();
    SmallVector<FormatShape, 2> view_info_part;
    view_info_part.emplace_back(array_to_small_vector(tensor.sizes()));
    view_info_part.emplace_back(array_to_small_vector(tensor.strides()));

    view_infos.emplace_back(view_info_part);
    view_offsets.emplace_back(infer_offset);
    return true;
  }

  // Conduct inferring
  bool can_use_combined(
      const Tensor& src,
      SmallVector<SmallVector<FormatShape, 2>, 4>& view_infos,
      SmallVector<int64_t, 4>& view_offsets,
      int64_t max_len) {
    // combined tensor should be discontiguous
    if (src.is_contiguous() || cases_avoid(src)) {
      return false;
    }

    auto combined_base_sizes = src.storage().get_npu_desc().base_sizes_;
    auto combined_base_strides = src.storage().get_npu_desc().base_strides_;

    // Key infos that should be inferred.
    FormatShape infer_size;
    FormatShape infer_stride;
    int64_t infer_offset = 0;

    // Reconstruct "the discontiguous combined tensor"
    // viewInfo = combined tensor(src)'s viewInfo
    // baseInfo = combined tensor(src)'s baseInfo
    Tensor temp_src =
        at::empty(IntArrayRef{combined_base_sizes}, src.options());
    temp_src.set_(
        src.storage(), src.storage_offset(), src.sizes(), src.strides());

    // Construct "the first inferred tensor" inside "can_infer_view_tensor()"
    // viewInfo = combined tensor(src)'s viewInfo
    // baseInfo = inferred info(infer_size, infer_stride, infer_offset)
    // If the first inferred tensor can be optimized, store its info.
    if (can_infer_view_tensor(
            src, temp_src, infer_size, infer_stride, infer_offset) &&
        emplace_info(
            temp_src, view_infos, view_offsets, infer_offset, max_len)) {
      // Construct "the second inferred tensor"
      // viewInfo = inferred info(infer_size, infer_stride, infer_offset)
      // baseInfo = combined tensor(src)'s baseInfo
      temp_src.set_(
          src.storage(),
          temp_src.storage_offset() - infer_offset,
          infer_size,
          infer_stride);
      StorageDescHelper::SetDesc(
          temp_src, combined_base_sizes, combined_base_strides);

      // The second inferred tensor can be optimized or not
      if (can_be_optimize_from_default_cases(temp_src) &&
          emplace_info(
              temp_src,
              view_infos,
              view_offsets,
              temp_src.storage_offset(),
              max_len)) {
        return true;
      } else if (view_infos.size() >= max_len) {
        return false;
      }
      // However, n-combined ops(n>2), also could be processed
      return can_use_combined(temp_src, view_infos, view_offsets, max_len);
    }
    // Constructed two inferred tensors can be optimized at the same time or not
    return false;
  }

  // Reconstructing discontiguous tensor at trans-contiguous procedure.
  bool reconstruct_tensor(
      Tensor& src,
      SmallVector<SmallVector<FormatShape, 2>, 4>& view_infos,
      SmallVector<int64_t, 4>& view_offsets) {
    auto view_info = view_infos.pop_back_val();
    auto view_offset = view_offsets.pop_back_val();
    // Set view info to make discontiguous tensor.
    // view_info[0]: stored shape infos in inferring procedure.
    // view_info[1]: stored stride infos in inferring procedure.
    src.set_(src.storage(), view_offset, view_info[0], view_info[1]);

    // If current tensor is sliced and the stack is still not empty:
    // stored infos in the stack should be modified.
    if (view_infos.size() >= 1 && maybe_slice(src)) {
      auto view_info_pre = view_infos.pop_back_val();

      std::map<int64_t, int64_t> map_stride_shape;
      auto computed_stride =
          StorageDescHelper::ComputeStrideFromShape(view_info[0]);
      // Adjust shape according to sorted stride
      for (auto i = 0; i < view_info_pre[0].size(); i++) {
        // "shape[i] == shape [j]" causes non-unique keys for
        // "map_stride_shape";
        // Temporarily, making size[i] * stride[i] to obtain unique keys;
        // TODO: explore unique keys for any cases when "shape[i] == shape [j]"
        map_stride_shape[view_info[0][i] * view_info[1][i]] =
            computed_stride[i];
      }

      for (auto i = 0; i < view_info_pre[0].size(); i++) {
        view_info_pre[1][i] =
            map_stride_shape[view_info_pre[0][i] * view_info_pre[1][i]];
      }
      // re-store modified infos
      view_infos.emplace_back(view_info_pre);
    }
    return true;
  }

  // Conduct trans-contiguous under strict constrains
  bool combined_to_contiguous(
      Tensor& src,
      Tensor& self,
      SmallVector<SmallVector<FormatShape, 2>, 4>& view_infos,
      SmallVector<int64_t, 4>& view_offsets) {
    // Base case: the last tensor to be processed.
    if (view_infos.size() == 1) {
      if (reconstruct_tensor(src, view_infos, view_offsets)) {
        std::vector<string> optimizations_last{"reshape", "permute", "slice", "select"};
        return copy_optimize_contiguous_by_given_cases(
            src, self, optimizations_last);
      }
      return false;
    }
    // Construct the first tensor and judge whether it can be optimized.
    if (reconstruct_tensor(src, view_infos, view_offsets)) {
      std::vector<string> optimizations_first{"reshape", "slice", "select"};
      if (reshape_without_copy_match(src)) {
        // case 1 : The first tensor is reshape-type, refresh its info is enough
        return combined_to_contiguous(src, self, view_infos, view_offsets);
      } else if (can_be_optimize_from_default_cases(src)) {
        // case 2: The first tensor is discontiguous-type,
        // conduct the standard optimization procedure.
        auto contiguous_src = at::empty_with_format(
            src.sizes(),
            src.options(),
            src.storage().get_npu_desc().npu_format_);
        return (
            copy_optimize_contiguous_by_given_cases(
                src, contiguous_src, optimizations_first) &&
            combined_to_contiguous(
                contiguous_src, self, view_infos, view_offsets));
      }
      // case3 ： The first tensor is contiguous or cannot be identified==>exit
      return false;
    }
    // If the first tensor cannnot be reconstructed==>exit
    return false;
  }

}; // class combinedContiguousOpt

REGISTER_COPY_OPT(combined, CombinedContiguousOpt)

} // namespace npu
} // namespace native
} // namespace at
