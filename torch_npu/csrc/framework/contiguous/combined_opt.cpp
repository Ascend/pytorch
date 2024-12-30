#include <ATen/NamedTensorUtils.h>
#include <ATen/quantized/QTensorImpl.h>

#include <map>

#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"

namespace at_npu {
namespace native {

class CombinedContiguousOpt : public ContiguousOpt {
public:
    // Combined tensor == discontiguous tensor caused by combined view operators.
    bool Optimizer(at::Tensor &self, const at::Tensor &src,
                   const ContiguousTensorDesc &src_desc) override {
        // Maximum combined operators suggested: combined_cases_num = 2
        // NOTE: n-cmobined(n>2) can also be supported
        int combined_cases_num = MaxCombinedCasesNum;

        ShapeStrideStack shape_stride_stacks;
        OffsetStack offset_stack;

        if (can_use_combined(shape_stride_stacks, offset_stack, src_desc,
                             combined_cases_num)) {
            RECORD_FUNCTION("contiguous_h_combined", std::vector<c10::IValue>({src}));
            return pre_combined_to_contiguous(self, src, shape_stride_stacks, offset_stack);
        }
        return false;
    }

    bool CachedOptimizer(at::Tensor &self, const at::Tensor &src,
                         const ContiguousTensorDesc &src_desc) override
    {
        ShapeStrideStack shape_stride_stacks;
        OffsetStack offset_stack;
        if (src_desc.cached_contiguous) {
            RECORD_FUNCTION("cached_contiguous_h_combined", std::vector<c10::IValue>({src}));

            CachedContiguousOpt cachedContiguousOpt = TransContiguous::cached_contiguous_opt[src_desc.hash_src_desc];
            shape_stride_stacks = cachedContiguousOpt.shape_stride_stack;
            offset_stack = cachedContiguousOpt.offset_stack;
            return pre_combined_to_contiguous(self, src, shape_stride_stacks, offset_stack);
        }

        int combined_cases_num = MaxCombinedCasesNum;
        if (can_use_combined(shape_stride_stacks, offset_stack, src_desc,
                             combined_cases_num)) {
            ShapeStrideStack cached_shape_stride_stacks = shape_stride_stacks;
            OffsetStack cached_offset_stack = offset_stack;
            RECORD_FUNCTION("contiguous_h_combined", std::vector<c10::IValue>({src}));

            bool contiguousOrNot = pre_combined_to_contiguous(self, src, shape_stride_stacks, offset_stack);
            if (contiguousOrNot) {
                CachedContiguousOpt cached_opt = CachedContiguousOpt{
                        "combined"
                };
                cached_opt.shape_stride_stack = cached_shape_stride_stacks;
                cached_opt.offset_stack = cached_offset_stack;
                cached_opt.contiguous_tensor_desc = src_desc;
                TransContiguous::cached_contiguous_opt[src_desc.hash_src_desc] = cached_opt;
            }
            return contiguousOrNot;
        }
        return false;
    }

private:

    bool pre_combined_to_contiguous(at::Tensor &self, const at::Tensor &src,
                                    ShapeStrideStack &shape_stride_stacks,
                                    OffsetStack &offset_stack)
    {
        // Record src infos for recovering after trans-contiguous
        auto src_storage_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->get_npu_desc();

        at::Tensor base_tensor =
                at::empty(src_storage_desc.base_sizes_, src.options());
        base_tensor.set_(src.storage());

        // Reconstruct combined discontiguous tensor ==trans==> contiguous tensor
        bool contiguousOrNot = combined_to_contiguous(self, base_tensor, shape_stride_stacks, offset_stack);
        // Recover modified tensor infos of src after trans-contiguous
        StorageDescHelper::CopyDesc(base_tensor, src_storage_desc);
        return contiguousOrNot;
    }

    bool cases_avoid(const ContiguousTensorDesc &tensor_desc)
    {
        for (const auto i : c10::irange(tensor_desc.sizes_.size())) {
            // expand+x,x+expand
            if (tensor_desc.strides_[i] == 0) {
                return true;
            }
        }
        return false;
    }

  // Unmatched tensor ==refresh(no copy)==> macthed tensor
  bool reshape_without_copy_match(at::Tensor &tensor) {
    if (!tensor.is_contiguous()) {
      return false;
    }
    auto npu_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->get_npu_desc();
    if ((c10::multiply_integers(tensor.sizes()) !=
         c10::multiply_integers(npu_desc.base_sizes_)) ||
        (tensor.storage_offset() != npu_desc.base_offset_)) {
      return false;
    }
    RECORD_FUNCTION("contiguous_h_match", std::vector<c10::IValue>({tensor}));
    StorageDescHelper::SetDesc(tensor, CalcuOpUtil::ConvertIntArrayRefToSmallVector(tensor.sizes()),
                               CalcuOpUtil::ConvertIntArrayRefToSmallVector(tensor.strides()));
    return true;
  }

  // Whether tensor can be optimized(no optimization).
  bool can_be_optimize_from_default_cases(ContiguousTensorDesc &tensor_desc) {
    OptimizationCases opt_cases{"reshape", "slice", "select"};
    tensor_desc.reset_optimization_cases(opt_cases);
    return TransContiguous::CanOptimize(tensor_desc);
  }

  // Conduct trans-contiguous for given optimization cases.
  bool
  copy_optimize_contiguous_by_given_cases(at::Tensor &self,
                                          const at::Tensor &tensor,
                                          OptimizationCases &optimizations) {
    // Set "OpenCombined = false" to avoid recursion.
    return TransContiguous::ContiguousOptimizeWithBaseFormat(
        self, tensor, optimizations, false);
  }

  // Weak constrains for transpose cases
  bool maybe_permute(const ContiguousTensorDesc &tensor_desc) {
    // tensors with nonmonotonic strides will be taken into consideration
    // (Ascend): 对于特殊stride的情况例如：[*,*,1,1]这种，需要进一步分析影响
    for (const auto i : c10::irange(tensor_desc.sizes_.size() - 1)) {
      if (tensor_desc.strides_[i] < tensor_desc.strides_[i + 1]) {
        return true;
      }
    }
    return false;
  }

    // Weak constrains for select cases
    bool maybe_select(const ContiguousTensorDesc &tensor_desc) {
        for (auto i = tensor_desc.sizes_.size() - 1; i > 0; i--) {
            if (tensor_desc.strides_[i] == 0) {
                return false;
            }
            if (tensor_desc.strides_[i - 1] %
                    (tensor_desc.sizes_[i] * tensor_desc.strides_[i]) !=
                0) {
                return false;
            }
            if (tensor_desc.strides_[i - 1] /
                    (tensor_desc.sizes_[i] * tensor_desc.strides_[i]) !=
                1) {
                if (tensor_desc.offset_ %
                        (tensor_desc.sizes_[i] * tensor_desc.strides_[i]) !=
                    0) {
                    return false;
                }
                // Avoid combined-cases such as squeeze+indexing at the first axis.
                if (tensor_desc.strides_[0] != tensor_desc.base_strides_[0]) {
                    return false;
                }
            }
        }
        return true;
    }

    // Weak constrains for slice cases
    bool maybe_slice(const ContiguousTensorDesc &tensor_desc) {
        // tensors with reduced numel will be taken into consideration.
        if (c10::multiply_integers(tensor_desc.sizes_) <
            c10::multiply_integers(tensor_desc.base_sizes_)) {
            for (const auto i : c10::irange(tensor_desc.sizes_.size() - 2)) {
                if (tensor_desc.strides_[i + 1] == 0 ||
                    tensor_desc.strides_[i] % tensor_desc.strides_[i + 1] != 0) {
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
  bool can_infer_view_tensor(ContiguousTensorDesc &tensor_desc,
                             FormatShape &infer_size, FormatShape &infer_stride,
                             int64_t &infer_offset) {
    const auto &view_sizes = tensor_desc.sizes_;
    const auto &view_strides = tensor_desc.strides_;

    if (maybe_permute(tensor_desc)) {
      FormatShape &permute_size_sorted = infer_size;
      FormatShape &permute_stride_sorted = infer_stride;
      permute_size_sorted = view_sizes;
      permute_stride_sorted = view_strides;

      // Sort stride
      std::sort(permute_stride_sorted.rbegin(), permute_stride_sorted.rend());

      // Map stride to shape
      std::map<int64_t, int64_t> map_shape_stride;
      std::map<int64_t, int64_t> label_map_shape_stride;
      for (const auto i : c10::irange(view_sizes.size())) {
        map_shape_stride[view_strides[i]] = view_sizes[i];
      }
      // 除去第0维，其他维shape为1时，不记录对应的stride值，该stride的值会和其他维的stride有重复
      for (const auto i : c10::irange(view_sizes.size())) {
        if (i == 0) {
          map_shape_stride[view_strides[0]] = view_sizes[0];
        } else if (i != 0 && view_sizes[i] != 1) {
          map_shape_stride[view_strides[i]] = view_sizes[i];
        }
      }
      // stride中有相等的情况，后面相等的stride对应的shape为1
      for (const auto i : c10::irange(view_sizes.size())) {
        if (label_map_shape_stride[permute_stride_sorted[i]] != true) {
          permute_size_sorted[i] = map_shape_stride[permute_stride_sorted[i]];
          label_map_shape_stride[permute_stride_sorted[i]] = true;
        } else {
          permute_size_sorted[i] = 1;
        }
      }
      infer_offset = 0;
      // Refresh tensor's base info to construct transposed tensor
      tensor_desc.base_sizes_ = permute_size_sorted;
      tensor_desc.base_strides_ = permute_stride_sorted;
      // double-checking of may_permute is not required, because view strides
      // does not changed.
      return true;
    }

    if (maybe_select(tensor_desc)) {
      FormatShape &select_size = infer_size;
      FormatShape &select_stride = infer_stride;
      // Infer base shape according to view shape and stride
      select_stride = view_strides;
      select_size = view_sizes;
      // select_size and stride should be one more than view_size
      select_size.emplace_back((int64_t)1);
      select_stride.emplace_back((int64_t)1);

      int64_t i = static_cast<int64_t>(view_sizes.size()) - 1;
      if (view_strides[i] == 1) {
        select_size[i + 1] = view_sizes[i];
        select_stride[i + 1] = 1;

        for (i = i - 1; i >= 0; i--) {
          if (view_strides[i] != view_strides[i + 1] * view_sizes[i + 1]) {
            select_size[i + 1] =
                view_strides[i] / (view_sizes[i + 1] * view_strides[i + 1]);
            select_stride[i + 1] = view_sizes[i + 1] * view_strides[i + 1];
            infer_offset = tensor_desc.offset_ % view_strides[i];
            break;
          }
          select_size[i + 1] = view_sizes[i];
          select_stride[i + 1] = view_strides[i];
        }
      } else {
        select_size[i + 1] = view_strides[i];
        select_stride[i + 1] = 1;
        infer_offset = tensor_desc.offset_ % view_strides[i];
      }
      for (i = i - 1; i >= 0; i--) {
        select_size[i + 1] = view_sizes[i + 1];
        select_stride[i + 1] = view_strides[i + 1];
      }

      select_size[0] = view_sizes[0];
      select_stride[0] = view_strides[0];

      // Refresh tensor's base info to construct selected tensor
      tensor_desc.base_sizes_ = select_size;
      tensor_desc.base_strides_ = select_stride;
      // Whether the construted tensor is selected?
      return maybe_select(tensor_desc);
    }

    if (maybe_slice(tensor_desc)) {
      FormatShape &slice_size = infer_size;
      FormatShape &slice_stride = infer_stride;

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
      slice_size[0] = (c10::multiply_integers(tensor_desc.base_sizes_) /
                       c10::multiply_integers(slice_size));
      infer_offset = tensor_desc.offset_;
      // Refresh tensor's base info and storage info to construct sliced tensor
      tensor_desc.base_sizes_ = slice_size;
      tensor_desc.base_strides_ = slice_stride;
      // Whether the construted tensor is sliced?
      return maybe_slice(tensor_desc);
    }
    return false;
  }

  bool stack_infer_info(ShapeStrideStack &shape_stride_stacks,
                        OffsetStack &offset_stacks, int64_t infer_offset,
                        int64_t combined_cases_num,
                        ContiguousTensorDesc &tensor_desc) {
    // Only combined_cases_num-combined Ops cases are taken into consideration
    if (static_cast<int16_t>(shape_stride_stacks.size()) == combined_cases_num) {
      return false;
    }

    c10::SmallVector<FormatShape, 2> stack_shape_stride_part;
    stack_shape_stride_part.emplace_back(
        CalcuOpUtil::ConvertIntArrayRefToSmallVector(tensor_desc.sizes_));
    stack_shape_stride_part.emplace_back(
        CalcuOpUtil::ConvertIntArrayRefToSmallVector(tensor_desc.strides_));

    shape_stride_stacks.emplace_back(stack_shape_stride_part);
    offset_stacks.emplace_back(infer_offset);
    return true;
  }

  // Conduct inferring
  bool can_use_combined(ShapeStrideStack &shape_stride_stacks,
                        OffsetStack &offset_stacks,
                        const ContiguousTensorDesc &src_desc,
                        int64_t combined_cases_num) {
    // combined tensor should be discontiguous
    if (src_desc.is_contiguous_ || cases_avoid(src_desc)) {
      return false;
    }

    // Key infos that should be inferred.
    FormatShape infer_size;
    FormatShape infer_stride;
    int64_t infer_offset = 0;

    // Reconstruct "the discontiguous combined tensor desc"
    // viewInfo = combined tensor(src)'s viewInfo
    // baseInfo = combined tensor(src)'s baseInfo
    // src's desc would be modified, so a local struct is created.
    ContiguousTensorDesc local_src_desc = src_desc;

    // Construct "the first inferred tensor" inside "can_infer_view_tensor()"
    // viewInfo = combined tensor(src)'s viewInfo
    // baseInfo = inferred info(infer_size, infer_stride, infer_offset)
    // If the first inferred tensor can be optimized, store its info.
    if (can_infer_view_tensor(local_src_desc, infer_size, infer_stride,
                              infer_offset) &&
        stack_infer_info(shape_stride_stacks, offset_stacks, infer_offset,
                         combined_cases_num, local_src_desc)) {
      // Construct "the second inferred tensor"
      // viewInfo = inferred info(infer_size, infer_stride, infer_offset)
      // baseInfo = combined tensor(src)'s baseInfo
      local_src_desc.sizes_ = infer_size;
      local_src_desc.strides_ = infer_stride;
      local_src_desc.offset_ -= infer_offset;
      local_src_desc.base_sizes_ = src_desc.base_sizes_;
      local_src_desc.base_strides_ = src_desc.base_strides_;
      local_src_desc.refresh_contiguous_using_size_and_stride();
      // The second inferred tensor can be optimized or not
      if (can_be_optimize_from_default_cases(local_src_desc) &&
          stack_infer_info(shape_stride_stacks, offset_stacks,
                           local_src_desc.offset_, combined_cases_num,
                           local_src_desc)) {
        return true;
      }
      // If the second pattern is not inferred successfully, retrun false
      return false;
    }
    // If the first pattern is not inferred successfully, retrun false
    return false;
  }

  // Reconstructing discontiguous tensor at trans-contiguous procedure.
  bool reconstruct_tensor(at::Tensor &src,
                          ShapeStrideStack &shape_stride_stacks,
                          OffsetStack &offset_stacks) {
    auto stack_shape_stride = shape_stride_stacks.pop_back_val();
    auto stack_offset = offset_stacks.pop_back_val();
    // Set view info to make discontiguous tensor.
    // stack_shape_stride[0]: stored shape infos in inferring procedure.
    // stack_shape_stride[1]: stored stride infos in inferring procedure.

    src.set_(src.storage(), stack_offset, stack_shape_stride[0],
             stack_shape_stride[1]);

    // If current tensor is sliced and the stack is still not empty:
    // stored infos in the stack should be modified.
    if (shape_stride_stacks.size() >= 1 &&
        maybe_slice(TransContiguous::GetTensorDescInfo(src))) {
      auto stack_shape_stride_pre = shape_stride_stacks.pop_back_val();

      std::map<int64_t, int64_t> map_stride_shape;
      auto computed_stride =
          StorageDescHelper::ComputeStrideFromShape(stack_shape_stride[0]);
      // Adjust shape according to sorted stride
      for (const auto i : c10::irange(stack_shape_stride_pre[0].size())) {
        // if shape_i equals to shape_j, non-unique keys for "map_stride_shape" would be made;
        // Temporarily, making size[i] * stride[i] to obtain unique keys;
        // (Ascend): explore unique keys for any cases when "shape[i] == shape [j]"
        map_stride_shape[stack_shape_stride[0][i] * stack_shape_stride[1][i]] =
            computed_stride[i];
      }

      for (const auto i : c10::irange(stack_shape_stride_pre[0].size())) {
        stack_shape_stride_pre[1][i] =
            map_stride_shape[stack_shape_stride_pre[0][i] *
                             stack_shape_stride_pre[1][i]];
      }
      // re-store modified infos
      shape_stride_stacks.emplace_back(stack_shape_stride_pre);
    }
    return true;
  }

  // Conduct trans-contiguous under strict constrains
  bool combined_to_contiguous(at::Tensor &self, at::Tensor &src,
                              ShapeStrideStack &shape_stride_stacks,
                              OffsetStack &offset_stacks) {
    // Base case: the last tensor to be processed.
    if (shape_stride_stacks.size() == 1) {
      if (reconstruct_tensor(src, shape_stride_stacks, offset_stacks)) {
        OptimizationCases opt_cases_last{"reshape", "permute", "slice",
                                         "select"};
        return copy_optimize_contiguous_by_given_cases(self, src,
                                                       opt_cases_last);
      }
      return false;
    }
    // Construct the first tensor and judge whether it can be optimized.
    if (reconstruct_tensor(src, shape_stride_stacks, offset_stacks)) {
      ContiguousTensorDesc src_desc_ = TransContiguous::GetTensorDescInfo(src);
      OptimizationCases opt_cases_first{"reshape", "slice", "select"};
      if (reshape_without_copy_match(src)) {
        // case 1 : The first tensor is reshape-type, refresh its info is enough
        return combined_to_contiguous(self, src, shape_stride_stacks,
                                      offset_stacks);
      } else if (can_be_optimize_from_default_cases(src_desc_)) {
        // case 2: The first tensor is discontiguous-type,
        // conduct the standard optimization procedure.
        auto transfer_tensor = OpPreparation::ApplyTensorWithFormat(
            src.sizes(), src.options(),
            torch_npu::NPUBridge::GetNpuStorageImpl(src)->get_npu_desc().npu_format_);
        return (copy_optimize_contiguous_by_given_cases(transfer_tensor, src,
                                                        opt_cases_first) &&
                combined_to_contiguous(self, transfer_tensor,
                                       shape_stride_stacks, offset_stacks));
      }
      // case3 ： The first tensor is contiguous or cannot be identified==>exit
      return false;
    }
    // If the first tensor cannnot be reconstructed==>exit
    return false;
  }
}; // class combinedContiguousOpt

REGISTER_COPY_OPT(combined, CombinedContiguousOpt)

} // namespace native
} // namespace at_npu
