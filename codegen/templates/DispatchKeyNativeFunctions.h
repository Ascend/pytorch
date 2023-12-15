// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2022, Facebook CORPORATION.
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
#ifndef ${macro}
#define ${macro}


#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
${include_npu_native_functions}

${static_define}

// ${generated_comment}
namespace ${cpp_namespace} {

struct ${class_name} {

static at::Tensor argmax(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim);
static at::Tensor argmin(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim);
static at::Tensor _embedding_bag_dense_backward(
    const at::Tensor& grad,
    const at::Tensor& indices,
    const at::Tensor& offset2bag,
    const at::Tensor& bag_size,
    const at::Tensor & maximum_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const c10::optional<at::Tensor> & per_sample_weights,
    int64_t padding_idx);
static at::Tensor nan_to_num(
    const at::Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> posinf,
    c10::optional<double> neginf);
static at::Tensor& nan_to_num_(
    at::Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> posinf,
    c10::optional<double> neginf);
static at::Tensor& nan_to_num_out(
    const at::Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> posinf,
    c10::optional<double> neginf,
    at::Tensor& out);

${dispatch_declarations}

};
}  // namespace ${cpp_namespace}

#endif
