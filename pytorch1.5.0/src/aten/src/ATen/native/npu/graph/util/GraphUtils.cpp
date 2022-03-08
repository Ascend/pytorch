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

#include "GraphUtils.h"

#include <c10/npu/NPUGraphContextManager.h>

namespace at {
namespace native {
namespace npu {
Value& GraphUtils::GetTensorIrValue(const at::Tensor& tensor) {
  auto storage = tensor.storage().unsafeGetStorageImpl();
  TORCH_CHECK(storage != nullptr, "Storage is null");
  return storage->get_mutable_npu_graph_desc().graph_value;
}

hash_t GraphUtils::GetTensorIrValueHash(const at::Tensor& tensor) {
  return GetTensorIrValue(tensor).GetValueHash();
}

void GraphUtils::SetTensorIrValue(StorageImpl* storage, const Value& value) {
  TORCH_CHECK(storage != nullptr, "Storage is null");
  auto& npu_graph_desc = storage->get_mutable_npu_graph_desc();
  npu_graph_desc.graph_value.UpdateFromOther(value);
  return;
}

void GraphUtils::SetTensorIrValue(
    const at::Tensor& tensor,
    const Value& value) {
  SetTensorIrValue(tensor.storage().unsafeGetStorageImpl(), value);
  return;
}

void GraphUtils::SetDataOp(StorageImpl* storage) {
  TORCH_CHECK(storage != nullptr, "Storage is null");
  auto data_node = std::make_shared<c10::npu::graph::Node>("Data");
  auto data_value = Value(data_node, data_node, 0);
  auto& npu_graph_desc = storage->get_mutable_npu_graph_desc();

  // Replace node directly, regardless of inplace op.
  // Use SetFromOther instead of UpdateFromOther.
  npu_graph_desc.graph_value.SetFromOther(data_value);
}

void GraphUtils::SetDataOp(const at::Tensor& tensor) {
  SetDataOp(tensor.storage().unsafeGetStorageImpl());
}

void GraphUtils::ResetOp(StorageImpl* storage) {
  TORCH_CHECK(storage != nullptr, "Storage is null");
  storage->get_mutable_npu_graph_desc().graph_value.ResetValue();
}
void GraphUtils::ResetOp(at::Tensor& tensor) {
  ResetOp(tensor.storage().unsafeGetStorageImpl());
}

bool GraphUtils::IsDataTensor(const StorageImpl* storage) {
  TORCH_CHECK(storage != nullptr, "Storage is null");
  auto& value = storage->get_mutable_npu_graph_desc().graph_value;
  auto cur_node = value.GetCurNode();
  TORCH_CHECK(cur_node != nullptr, "Cur storage does not have node");
  return (cur_node->GetOpType() == "Data");
}

bool GraphUtils::IsDataTensor(const at::Tensor& tensor) {
  return IsDataTensor(tensor.storage().unsafeGetStorageImpl());
}

bool GraphUtils::IsTensorWithoutNode(const StorageImpl* storage) {
  TORCH_CHECK(storage != nullptr, "Storage is null");
  return !storage->get_npu_graph_desc().graph_value.HashNode();
}

bool GraphUtils::IsTensorWithoutNode(const at::Tensor& tensor) {
  return IsTensorWithoutNode(tensor.storage().unsafeGetStorageImpl());
}

void GraphUtils::RetainGraphDataTensor(const at::Tensor& data_tensor) {
  auto storage = data_tensor.storage().unsafeGetStorageImpl();
  auto storage_ptr = c10::intrusive_ptr<StorageImpl>::reclaim(storage);
  c10::npu::graph::NpuGraphContextManager::GetInstance().AddInputStorage(
      storage_ptr);
  storage_ptr.release();
}
} // namespace npu
} // namespace native
} // namespace at
