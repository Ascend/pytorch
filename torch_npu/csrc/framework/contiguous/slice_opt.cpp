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

#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    class SliceContiguousOpt : public ContiguousOpt
    {
    public:
      bool Optimizer(const at::Tensor &src, at::Tensor &self) override
      {
        // Pattern slice. Current pattern should be used before PTcopy process.
        // Current pattern does not directly depend on other patterns.
        // The relative sequence of this pattern and other patterns is not important.
        c10::SmallVector<int64_t, SHAPE_SIZE> offsets;
        c10::SmallVector<int64_t, SHAPE_SIZE> size;
        if (can_use_slice(src, offsets, size))
        {
          RECORD_FUNCTION("narrow_npuSliceD", std::vector<c10::IValue>({src}));
          slice_to_contiguous(src, self, offsets, size);
          return true;
        }
        return false;
      }

      bool CanOptimizer(const at::Tensor &src) override
      {
        c10::SmallVector<int64_t, SHAPE_SIZE> offsets;
        c10::SmallVector<int64_t, SHAPE_SIZE> size;
        return can_use_slice(src, offsets, size);
      }

    private:
      // npu-slice pattern cover several view ops, including chunk, split, narrow and part of index.
      // Judgment logic is based on the implement of view ops in adapter layer.
      bool can_use_slice(const at::Tensor &src,
                         c10::SmallVector<int64_t, SHAPE_SIZE> &offsets,
                         c10::SmallVector<int64_t, SHAPE_SIZE> &size)
      {
        // After narrow, tensor must be discontiguous
        if (src.is_contiguous())
        {
          return false;
        }

        auto base_sizes = src.storage().get_npu_desc().base_sizes_;
        auto base_strides = src.storage().get_npu_desc().base_strides_;
        auto view_sizes = array_to_small_vector(src.sizes());
        auto view_strides = array_to_small_vector(src.strides());

        // narrow+select(select at last dim) ==> single narrow
        // 限制条件：1. 最后一轴stride非1==>最后一轴select；2. 基础格式；3.非最后一轴发生narrow（元素减少）
        // 最小化影响：仅限最后一轴的select，即tensor.select(-1, 1) == tensor[**,1:2],select过渡到narrow
        if (view_strides[view_strides.size() - 1] != 1 &&
            FormatHelper::IsBaseFormatType(src) &&
            view_strides.size() < base_strides.size() &&
            at::prod_intlist(view_sizes) <
                at::prod_intlist(base_sizes) / base_sizes[base_sizes.size() - 1])
        {
          view_sizes.emplace_back(1);
          view_strides.emplace_back(1);
        }

        // Strides must be the same.
        if (view_strides != base_strides)
        {
          return false;
        }

        // Only narrow dims are different.
        c10::SmallVector<int64_t, SHAPE_SIZE> narrow_dims;
        if (view_sizes.size() != base_sizes.size())
        {
          return false;
        }
        for (auto i = 0; i < view_sizes.size(); i++)
        {
          if (view_sizes[i] == base_sizes[i])
          {
            narrow_dims.emplace_back(0);
          }
          else if (view_sizes[i] < base_sizes[i])
          {
            narrow_dims.emplace_back(1);
          }
          else
          {
            return false;
          }
        }

        // Calculate npu slice param.
        size = view_sizes;
        offsets.clear();
        int64_t storage_offsets = src.storage_offset();
        // src.storage_offset() == start[narrow_dims[i]]*stride[narrow_dims[i]]
        for (auto i = 0; i < view_strides.size(); i++)
        {
          offsets.emplace_back(storage_offsets / view_strides[i]);
          storage_offsets = storage_offsets % view_strides[i];
        }
        if (storage_offsets != 0)
        {
          return false;
        }
        for (auto i = 0; i < offsets.size(); i++)
        {
          if ((offsets[i] + view_sizes[i]) > base_sizes[i])
          {
            // In narrow calculation, (start + length) <= cur_size
            return false;
          }
          if (offsets[i] != 0 && narrow_dims[i] == 0)
          {
            // narrow_dims[i] == 0 means dim i is not involved in narrow calculation.
            // offsets[i] != 0 means dim i has the start of narrow calculation.
            // Two conditions are contradictory.
            return false;
          }
        }
        return true;
      }

      void slice_to_contiguous(const at::Tensor &src, at::Tensor &self,
                               const c10::SmallVector<int64_t, SHAPE_SIZE> &offsets,
                               const c10::SmallVector<int64_t, SHAPE_SIZE> &size)
      {
        // create contiguous tensor for npu slice
        auto temp_tensor_size = src.storage().unsafeGetStorageImpl()->npu_desc_.base_sizes_;
        at::Tensor temp_src = at::empty(temp_tensor_size, src.options());
        temp_src.set_(src.storage(), temp_src.storage_offset(), temp_src.sizes(), temp_src.strides());

        // [临时解决方案，已讨论] sliceD算子当前对NCDWH格式入参不友好,框架层转ND格式
        if (temp_src.dim() == 5 && FormatHelper::GetFormat(temp_src) == ACL_FORMAT_NCDHW)
        {
          NPUNativeFunctions::npu_format_cast_(temp_src, ACL_FORMAT_ND);
        }

        NPUNativeFunctions::npu_slice_out(temp_src, offsets, size, self);
        return;
      }

    }; // class SliceContiguousOpt

    REGISTER_COPY_OPT(slice, SliceContiguousOpt)

  } // native
} // at