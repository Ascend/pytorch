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

namespace at_npu
{
  namespace native
  {

    class UnfoldContiguousOpt : public ContiguousOpt
    {
    public:
      bool Optimizer(const at::Tensor &src, at::Tensor &self) override
      {
        // `unfold`(*dimension*, *size*, *step*) → at::Tensor
        int64_t fold_dimension;
        int64_t fold_size;
        int64_t fold_step;

        if (can_use_unfold(src, fold_dimension, fold_size, fold_step))
        {
          RECORD_FUNCTION("unfold_npuTransposeD", std::vector<c10::IValue>({src}));
          unfold_to_contiguous(src, self, fold_dimension, fold_size, fold_step);
          return true;
        }
        return false;
      }

    private:
      bool can_use_unfold(const at::Tensor &src, int64_t &fold_dimension, int64_t &fold_size, int64_t &fold_step)
      {
        /*
    pattern infer:
    step1: discontiguous？Y：continue, N: return false
    step2：Dimensions of src is increased by 1?Y: continue; N:return false
    step3: Dim size differs at only one dim except for last dim？Y：continue ； N： return false
           ==> para: unfold dim:**dimension**
    step4: ==> para::**size** = src.size(-1) ; para::**step**=src.stride(**dimension**) // base.stride(**dimension**)
               slice step equals slcie size ?: Y:continue; N: return fasle.
    step5: Lastly, src.size(**dimension**) == (base.size(**dimension**) - **size**) // **step** + 1? Y :return true; N:return fasle
    */

        // step1 contiguous or not
        if (src.is_contiguous())
        {
          return false;
        }

        auto base_sizes = src.storage().get_npu_desc().base_sizes_;
        auto base_strides = src.storage().get_npu_desc().base_strides_;
        auto view_sizes = array_to_small_vector(src.sizes());
        auto view_strides = array_to_small_vector(src.strides());

        // step2 size match？ src.size() - base.size() == 1
        if (view_sizes.size() - base_sizes.size() != 1 ||
            view_strides.size() - base_strides.size() != 1)
        {
          return false;
        }

        int64_t unmatch_dim_nums = 0;
        int64_t temp_dimension;
        // step3 dim size differs at only one dim except for last dim
        for (auto i = 0; i < base_sizes.size(); i++)
        {
          if (base_sizes[i] != view_sizes[i])
          {
            unmatch_dim_nums++;
            temp_dimension = i;
          }
        }

        if (unmatch_dim_nums != 1)
        {
          return false;
        }
        fold_dimension = temp_dimension;

        // step4 size eqs step or not
        if (base_strides[fold_dimension] == 0)
        {
          NPU_LOGD("Base_strides at slice dim should not be 0!");
          return false;
        }
        fold_size = view_sizes.back();
        fold_step = view_strides[fold_dimension] / base_strides[fold_dimension];

        if (fold_size != fold_step)
        {
          NPU_LOGD("It cannot be optimized when size != step!");
          return false;
        }

        if (fold_step == 0)
        {
          NPU_LOGD("It cannot be optimized when size != step!");
          return false;
        }

        // step5 the last limitation
        if (view_sizes[fold_dimension] == (base_sizes[fold_dimension] - fold_size) / fold_step + 1)
        {
          return true;
        }
        return false;
      }

      void unfold_to_contiguous(const at::Tensor &src,
                                at::Tensor &self,
                                int64_t &fold_dimension,
                                int64_t &fold_size,
                                int64_t &fold_step)
      {

        auto base_sizes = src.storage().get_npu_desc().base_sizes_;

        TORCH_CHECK(fold_size != 0, "size should not be 0");
        int64_t split_nums = base_sizes[fold_dimension] / fold_size;

        // recover contiguous base tensor
        at::Tensor temp_src = at::empty(base_sizes, src.options());
        temp_src.set_(src.storage(), temp_src.storage_offset(),
                      temp_src.sizes(), temp_src.strides());

        // for dim size is not divisible ==> narrow
        if (base_sizes[fold_dimension] % fold_size != 0)
        {
          temp_src = temp_src.narrow(fold_dimension, 0, split_nums * fold_size).contiguous();
        }

        // Obtain reshape and permute info
        c10::SmallVector<int64_t, SHAPE_SIZE> reshape_index;
        c10::SmallVector<int64_t, SHAPE_SIZE> permute_index;

        for (int64_t i = 0; i < base_sizes.size(); i++)
        {
          reshape_index.emplace_back(base_sizes[i]);
        }
        reshape_index[fold_dimension] = split_nums;
        reshape_index.insert(reshape_index.begin() + fold_dimension + 1, fold_size);

        for (int64_t i = 0; i < base_sizes.size(); i++)
        {
          if (i > fold_dimension)
          {
            permute_index.emplace_back(i + 1);
          }
          else
          {
            permute_index.emplace_back(i);
          }
        }
        permute_index.emplace_back(fold_dimension + 1);

        // Obtain self tensor
        self = temp_src.reshape(reshape_index).clone().permute(permute_index).contiguous();
        return;
      }
    }; // class UnfoldContiguousOpt

    REGISTER_COPY_OPT(unfold, UnfoldContiguousOpt)

  } // native
} // at