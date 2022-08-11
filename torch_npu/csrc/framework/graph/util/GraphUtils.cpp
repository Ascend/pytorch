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

#include "torch_npu/csrc/framework/graph/util/GraphUtils.h"
#include "torch_npu/csrc/framework/graph/util/NPUGraphContextManager.h"

namespace at_npu {
namespace native {
Value& GraphUtils::GetTensorIrValue(const at::Tensor& tensor) {
  auto storage = torch_npu::NPUBridge::GetNpuStorageImpl(tensor);
  TORCH_CHECK(storage != nullptr, "Storage is null");
  return storage->get_mutable_npu_graph_desc().graph_value;
}

hash_t GraphUtils::GetTensorIrValueHash(const at::Tensor& tensor) {
  return GetTensorIrValue(tensor).GetValueHash();
}

size_t GraphUtils::GetTensorCapacity(c10::StorageImpl* storage) {
  auto npu_desc = torch_npu::NPUBridge::GetNpuStorageImpl(storage)->get_npu_desc();
  size_t nbytes = c10::multiply_integers(npu_desc.storage_sizes_) * npu_desc.data_type_.itemsize();
  return nbytes;
}

void GraphUtils::SetTensorIrValue(c10::StorageImpl* storage, const Value& value) {
  TORCH_CHECK(storage != nullptr, "Storage is null");
  auto& npu_graph_desc = torch_npu::NPUBridge::GetNpuStorageImpl(storage)->get_mutable_npu_graph_desc();
  npu_graph_desc.graph_value.UpdateFromOther(value);
  return;
}

void GraphUtils::SetTensorIrValue(
    const at::Tensor& tensor,
    const Value& value) {
  SetTensorIrValue(tensor.storage().unsafeGetStorageImpl(), value);
  return;
}

void GraphUtils::SetDataOp(c10::StorageImpl* storage) {
  TORCH_CHECK(storage != nullptr, "Storage is null");
  auto data_node = std::make_shared<Node>("Data");
  auto data_value = Value(data_node, data_node, 0);
  SetTensorIrValue(storage, data_value);
}

void GraphUtils::SetDataOp(const at::Tensor& tensor) {
  SetDataOp(tensor.storage().unsafeGetStorageImpl());
}

void GraphUtils::ResetOp(c10::StorageImpl* storage) {
  TORCH_CHECK(storage != nullptr, "Storage is null");
  torch_npu::NPUBridge::GetNpuStorageImpl(storage)->get_mutable_npu_graph_desc().graph_value.ResetValue();
}
void GraphUtils::ResetOp(at::Tensor& tensor) {
  ResetOp(tensor.storage().unsafeGetStorageImpl());
}

bool GraphUtils::IsDataTensor(const c10::StorageImpl* storage) {
  TORCH_CHECK(storage != nullptr, "Storage is null");
  auto& value = torch_npu::NPUBridge::GetNpuStorageImpl(const_cast<c10::StorageImpl*>(storage))->get_mutable_npu_graph_desc().graph_value;
  auto cur_node = value.GetCurNode();
  TORCH_CHECK(cur_node != nullptr, "Cur storage does not have node");
  return (cur_node->GetOpType() == "Data");
}

bool GraphUtils::IsDataTensor(const at::Tensor& tensor) {
  return IsDataTensor(tensor.storage().unsafeGetStorageImpl());
}

bool GraphUtils::IsTensorWithoutNode(const c10::StorageImpl* storage) {
  TORCH_CHECK(storage != nullptr, "Storage is null");
  return !torch_npu::NPUBridge::GetNpuStorageImpl(const_cast<c10::StorageImpl*>(storage))->get_npu_graph_desc().graph_value.HashNode();
}

bool GraphUtils::IsTensorWithoutNode(const at::Tensor& tensor) {
  return IsTensorWithoutNode(tensor.storage().unsafeGetStorageImpl());
}

void GraphUtils::RetainGraphDataTensor(const at::Tensor& data_tensor) {
  auto storage = data_tensor.storage().unsafeGetStorageImpl();
  auto storage_ptr = c10::intrusive_ptr<c10::StorageImpl>::reclaim(storage);
  NpuGraphContextManager::GetInstance().AddInputStorage(
      storage_ptr);
  storage_ptr.release();
}
} // namespace native
} // namespace at_npu