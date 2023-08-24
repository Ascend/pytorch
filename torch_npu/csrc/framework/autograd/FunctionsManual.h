// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#pragma once

// NB: Must be at the top of file to avoid including the deprecated "math.h".
// https://stackoverflow.com/questions/6563810/m-pi-works-with-math-h-but-not-with-cmath-in-visual-studio
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#endif

#include <ATen/ATen.h>

#include "torch_npu/csrc/aten/Functions.h"

namespace at_npu {
namespace autograd {
namespace generated {
namespace details {

// A simple way to imperatively compute index ranges for slots
// that have been flattened
struct IndexRangeGenerator {
  IndexRange range(size_t range_size) {
    i += range_size;
    return {i - range_size, i};
  }
  size_t size() { return i; }
  private:
    size_t i = 0;
};

Tensor toNonOptFwGrad(const c10::optional<Tensor>& t);
Tensor toNonOptPrimal(const c10::optional<Tensor>& t);
Tensor toNonOptTensor(const c10::optional<Tensor>& t);

Tensor apply_loss_reduction(const Tensor& unreduced, int64_t reduction);
bool any_variable_defined(const variable_list& variables);
void copy_range(variable_list& out, IndexRange range, const at::Tensor& t);
void copy_range(variable_list& out, IndexRange range, at::ArrayRef<at::Tensor> t);
at::Tensor not_implemented(const char* name, const char* reason="");
std::vector<Tensor> not_implemented_list(const char* name, const char* reason="");

at::Tensor maybe_multiply(const at::Tensor& t, const at::Scalar& s);

std::tuple<Tensor, Tensor, Tensor> rotary_mul_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& r1,
    const Tensor& r2);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
at::Tensor, at::Tensor, at::Tensor, at::Tensor> multi_head_attention_backward(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& query_weight, const at::Tensor& key_weight, const at::Tensor& value_weight,
    const at::Tensor& out_proj_weight, const c10::optional<at::Tensor>& query_bias_opt,
    const c10::optional<at::Tensor>& key_bias_opt, const c10::optional<at::Tensor>& value_bias_opt,
    const c10::optional<at::Tensor>& out_proj_bias_opt, const at::Tensor& query_res,
    const at::Tensor& key_res, const at::Tensor& value_res,
    const at::Tensor& attn_scores, const at::Tensor& attn_res, const at::Tensor& context,
    const at::Tensor& y_grad, const at::Tensor& dropout_mask,
    int64_t attn_head_num, int64_t attn_dim_per_head,
    int64_t src_len, int64_t tgt_len,
    double dropout_prob, bool softmax_use_float);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> gru_backward(
    const c10::optional<at::Tensor>& grady_opt,
    const c10::optional<at::Tensor>& gradh_opt,
    const at::Tensor& input,
    const at::Tensor& weight_input,
    const at::Tensor& weight_hidden,
    const at::Tensor& bias_input,
    const at::Tensor& bias_hidden,
    const at::Tensor& seq_length,
    const at::Tensor& init_h,
    const at::Tensor& output_y,
    const at::Tensor& output_h,
    const at::Tensor& output_updata,
    const at::Tensor& output_reset,
    const at::Tensor& output_new,
    const at::Tensor& hidden_new);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> lstm_backward(
    const c10::optional<at::Tensor>& grady_opt,
    const c10::optional<at::Tensor>& gradh_opt,
    const c10::optional<at::Tensor>& gradc_opt,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& init_h,
    const at::Tensor& init_c,
    const at::Tensor& y,
    const at::Tensor& h,
    const at::Tensor& c,
    const at::Tensor& i,
    const at::Tensor& j,
    const at::Tensor& f,
    const at::Tensor& o,
    const at::Tensor& tanhc);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> lstm_cell_backward(
    const c10::optional<at::Tensor>& grad_y_opt_,
    const c10::optional<at::Tensor>& grad_h_opt_,
    const c10::optional<at::Tensor>& grad_c_opt_,
    const at::Tensor& input,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const at::Tensor& h,
    const at::Tensor& c,
    const at::Tensor& y_output,
    const at::Tensor& h_output,
    const at::Tensor& c_output,
    const at::Tensor& i,
    const at::Tensor& j,
    const at::Tensor& f,
    const at::Tensor& o,
    const at::Tensor& tanhc);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> lstm_data_backward(
    const c10::optional<at::Tensor>& grady_opt,
    const c10::optional<at::Tensor>& gradh_opt,
    const c10::optional<at::Tensor>& gradc_opt,
    const at::Tensor& input,
    const at::Tensor& batch_sizes,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& init_h,
    const at::Tensor& init_c,
    const at::Tensor& y,
    const at::Tensor& h,
    const at::Tensor& c,
    const at::Tensor& i,
    const at::Tensor& j,
    const at::Tensor& f,
    const at::Tensor& o,
    const at::Tensor& tanhc,
    bool flag_direction);
} // namespace details
} // namespace generated
} // namespace autograd
} // namespace at_npu
