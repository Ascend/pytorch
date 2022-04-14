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

#pragma once

#include <ATen/ATen.h>
#include <c10/core/StorageImpl.h>
#include <c10/npu/NPUGraph.h>

namespace at {
namespace native {
namespace npu {

using c10::npu::graph::Value;
using c10::npu::hash_utils::hash_t;
class GraphUtils {
public:
  static Value& GetTensorIrValue(const at::Tensor& tensor);

  static hash_t GetTensorIrValueHash(const at::Tensor& tensor);

  static void SetTensorIrValue(StorageImpl* storage, const Value& value);
  static void SetTensorIrValue(const at::Tensor& tensor, const Value& value);

  static void SetDataOp(StorageImpl* storage);

  static void SetDataOp(const at::Tensor& tensor);

  static void ResetOp(StorageImpl* storage);
  static void ResetOp(at::Tensor& tensor);

  static bool IsDataTensor(const StorageImpl* storage);
  static bool IsDataTensor(const at::Tensor& tensor);

  static bool IsTensorWithoutNode(const StorageImpl* storage);
  static bool IsTensorWithoutNode(const at::Tensor& tensor);

  static void RetainGraphDataTensor(const at::Tensor& data_tensor,
                                    const c10::optional<int32_t>& device_index = c10::nullopt);

  // StorageImpl of cpu tensor does not have npu_graph_desc
  // we need to init it by this func
  static void InitGraphDescForCpuTensor(const at::Tensor& cpu_tensor);

  static void RetainNoneOutputNode(c10::npu::graph::NodePtr none_output_node);
};
} // namespace npu
} // namespace native
} // namespace at