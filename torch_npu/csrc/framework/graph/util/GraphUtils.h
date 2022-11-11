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
#include <torch_npu/csrc/framework/graph/util/NPUGraph.h>
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/NPUBridge.h"

namespace at_npu {
namespace native {

using namespace at_npu::native::hash_utils;

class GraphUtils {
public:
  static Value& GetTensorIrValue(const at::Tensor& tensor);

  static hash_t GetTensorIrValueHash(const at::Tensor& tensor);

  static size_t GetTensorCapacity(c10::StorageImpl* storage);

  static void SetTensorIrValue(c10::StorageImpl* storage, const Value& value);
  static void SetTensorIrValue(const at::Tensor& tensor, const Value& value);

  static void SetDataOp(c10::StorageImpl* storage);

  static void SetDataOp(const at::Tensor& tensor);

  static void SetDataPtrAndNbytes(c10::StorageImpl* storage, size_t nbytes);

  static void ResetOp(c10::StorageImpl* storage);
  static void ResetOp(at::Tensor& tensor);

  static bool IsDataTensor(const c10::StorageImpl* storage);
  static bool IsDataTensor(const at::Tensor& tensor);

  static bool IsTensorWithoutNode(const c10::StorageImpl* storage);
  static bool IsTensorWithoutNode(const at::Tensor& tensor);

  static void RetainGraphDataTensor(const at::Tensor& data_tensor);

  static void RetainNoneOutputNode(at_npu::native::NodePtr none_output_node);
};
} // namespace native
} // namespace at_npu

