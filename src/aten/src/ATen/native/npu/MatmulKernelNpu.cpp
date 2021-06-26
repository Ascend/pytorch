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
#include <ATen/NamedTensorUtils.h>
#include "c10/npu/OptionsManager.h"
#include "ATen/native/npu/interface/EnvVariables.h"
#include "ATen/native/npu/common/InnerNpuNativeFunction.h"
namespace at {
namespace native {
Tensor matmul_npu(
    c10::optional<Tensor> out_opt,
    const Tensor& tensor1,
    const Tensor& tensor2) {
  NoNamesGuard guard;
  auto has_out = out_opt.has_value();
  Tensor out = out_opt.value_or(Tensor());
  if (tensor1.is_npu() && tensor2.is_npu() && 
    tensor1.scalar_type() == kHalf && tensor2.scalar_type() == kHalf && 
    npu::env::CheckBmmV2Enable()) {
      auto res = matmul_by_bmmV2(tensor1, tensor2);
      return has_out ? out.set_(res) : res;
  }
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return has_out ? at::native::dot_out(out, tensor1, tensor2) : tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return has_out ? at::mv_out(out, tensor1, tensor2) : tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return has_out ? at::mm_out(out, tensor1.unsqueeze(0), tensor2).squeeze_(0)
                   : tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return has_out ? at::mm_out(out, tensor1, tensor2) : tensor1.mm(tensor2);
  } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    // optimization: use mm instead of bmm by folding tensor1's batch into
    // its leading matrix dimension.

    Tensor t2 = dim_tensor2 == 1 ? tensor2.unsqueeze(-1) : tensor2;
    auto size1 = tensor1.sizes();
    auto size2 = t2.sizes();
    std::vector<int64_t> output_size;
    output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
    if (dim_tensor2 > 1) {
      output_size.push_back(size2[dim_tensor2 - 1]);
    }

    // fold the batch into the first dimension
    Tensor t1 = tensor1.contiguous().view({-1, size1[size1.size() - 1]});
    Tensor output = has_out ? at::_unsafe_view(at::mm_out(out, t1, t2), output_size)
                            : at::_unsafe_view(t1.mm(t2), output_size);
    return has_out ? out.set_(output) : output;
  } else if ((dim_tensor1 == 1 || dim_tensor1 == 2) && dim_tensor2 >= 3) {
    // optimization: transpose the inner dimensions of the arguments, call
    // matmul on the swapped arguments, then transpose the inner dimensions
    // of the result.
    const int64_t n = dim_tensor1 == 2 ? tensor1.size(-2) : 1;
    const int64_t m = tensor1.size(-1);
    const int64_t p = tensor2.size(-1);

    const Tensor t2_T = tensor2.transpose(-1, -2);
    const Tensor t1_T = dim_tensor1 == 2 ? tensor1.t() : tensor1.reshape({n, m}).t();
    const Tensor res_T = matmul_npu(out_opt, t2_T, t1_T);

    if (dim_tensor1 == 2) {
      Tensor res = res_T.transpose(-1, -2).contiguous();
      return has_out ? out.set_(res) : res;
    }
    else {
      std::vector<int64_t> shape = tensor2.sizes().slice(0, dim_tensor2 - 2).vec();
      shape.push_back(p);

      Tensor res = res_T.reshape(shape).contiguous();
      return has_out ? out.set_(res) : res;
    }
  } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    // We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
    // we track m1 vs m2 separately even though they must match for nicer error messages
    int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
    int64_t m1 = tensor1.size(-1);
    IntArrayRef batch_tensor1(tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));
    int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-2) : 1;
    int64_t p = tensor2.size(-1);
    IntArrayRef batch_tensor2(tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});

    int expand_batch_product = std::accumulate(expand_batch_portion.begin(), expand_batch_portion.end(),
                                               1, std::multiplies<int64_t>());

    std::vector<int64_t> tensor1_bmm_view({expand_batch_product});
    tensor1_bmm_view.insert(tensor1_bmm_view.end(), {n, m1});

    std::vector<int64_t> tensor2_bmm_view({expand_batch_product});
    tensor2_bmm_view.insert(tensor2_bmm_view.end(), {m2, p});

    // flatten expanded batches
    Tensor tensor1_expanded = tensor1.expand(tensor1_expand_size).contiguous().view(tensor1_bmm_view);
    Tensor tensor2_expanded = tensor2.expand(tensor2_expand_size).contiguous().view(tensor2_bmm_view);

    // reshape batches back into result
    std::vector<int64_t> output_shape(expand_batch_portion);
    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }

    Tensor output = has_out ? at::_unsafe_view(at::bmm_out(out, tensor1_expanded, tensor2_expanded), output_shape)
                            : at::_unsafe_view(tensor1_expanded.bmm(tensor2_expanded), output_shape);

    return has_out ? out.set_(output) : output;
  }

 AT_ERROR("both arguments to matmul need to be at least 1D, but they are ",
          dim_tensor1, "D and ", dim_tensor2, "D");
}

Tensor matmul_npu(const Tensor & tensor1, const Tensor & tensor2) {
  auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
  auto result = at::native::matmul_npu(c10::nullopt, tensor1, tensor2);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor& matmul_out_npu(Tensor &result, const Tensor & tensor1, const Tensor & tensor2) {
  auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
  at::native::matmul_npu(c10::optional<Tensor>(result), tensor1, tensor2);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}


} // namespace native
} // namespace at
