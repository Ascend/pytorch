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

namespace at_npu
{
  namespace native
  {

    const std::vector<string> TransContiguous::optimizations_default = {};

    const std::vector<string> TransContiguous::optimizations_any_format = {
        "reshape",
        "slice"};

    std::vector<string> TransContiguous::FindMatchOptimizationsKeywords(
        const at::Tensor &tensor)
    {
      for (auto i = 0; i < tensor.sizes().size(); i++)
      {
        if (tensor.stride(i) == 0)
        {
          return {"broadcast"};
        }
      }

      for (auto i = 0; i < tensor.strides().size() - 1; i++)
      {
        if (tensor.stride(i) < tensor.stride(i + 1))
        {
          return {"permute", "unfold"};
        }
      }

      if (at::prod_intlist(tensor.sizes()) <
          at::prod_intlist(tensor.storage().get_npu_desc().base_sizes_))
      {
        return {"slice", "select", "indexing"};
      }

      return {};
    }

    bool TransContiguous::CheckClone(const at::Tensor &src, at::Tensor &self)
    {
      // self tensor may not be temporary constructed empty tensor from src, so:
      // 1. contiguous storage is needed:storage_offset and numels eq
      // 2. full memory copy: size match between src and self
      if (StorageDescHelper::OffsetAreMatch(&self) && self.is_contiguous() &&
          src.sizes().equals(self.sizes()) &&
          self.sizes().equals(self.storage().get_npu_desc().base_sizes_))
      {
        return true;
      }
      return false;
    }

    bool TransContiguous::CanOptimize(
        const at::Tensor &src,
        std::vector<string> optimizations)
    {
      for (auto opt : optimizations)
      {
        bool res =
            register_opt::CopyOptRegister::GetInstance()->CanOptimize(opt, src);
        if (res)
        {
          return true;
        }
      }
      return false;
    }

    bool TransContiguous::ContiguousOptimizeWithAnyFormat(
        at::Tensor &self,
        const at::Tensor &src,
        const std::vector<string> &optimizations)
    {
      if (!CheckClone(src, self))
      {
        return false;
      }
      for (auto &opt : optimizations)
      {
        bool res =
            register_opt::CopyOptRegister::GetInstance()->Run(opt, src, self);
        if (res)
        {
          return true;
        }
      }
      return false;
    }

    c10::optional<at::Tensor> TransContiguous::ContiguousOptimizeWithAnyFormat(
        const at::Tensor &src,
        const std::vector<string> &optimizations)
    {
      auto self = at::empty_with_format(
          src.sizes(),
          src.options(),
          src.storage().get_npu_desc().npu_format_);
      if (ContiguousOptimizeWithAnyFormat(self, src, optimizations))
      {
        return self;
      }
      return c10::nullopt;
    }

    bool TransContiguous::ContiguousOptimizeWithBaseFormat(
        at::Tensor &self,
        const at::Tensor &src,
        std::vector<string> optimizations,
        bool OpenCombined)
    {
      if (!FormatHelper::IsBaseFormatType(src))
      {
        // transadata???
        return false;
      }
      // In non-specific cases, classify the cases and simplify judgement.
      if (optimizations.size() == 0)
      {
        optimizations = FindMatchOptimizationsKeywords(src);
      }

      if (OpenCombined &&
          torch_npu::option::OptionsManager::CheckCombinedOptimizerEnable())
      {
        optimizations.emplace_back("combined");
      }
      return ContiguousOptimizeWithAnyFormat(self, src, optimizations);
    }

  } // namespace native
} // namespace at_npu