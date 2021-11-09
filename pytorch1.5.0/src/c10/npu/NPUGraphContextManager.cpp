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

#include "NPUGraphContextManager.h"

#include <c10/core/StorageImpl.h>

namespace c10 {
namespace npu {
namespace graph {

void InputContext::AddInput(const c10::intrusive_ptr<StorageImpl>& storage) {
  if (uid_of_input_in_ctx.find(storage.get()->get_npu_graph_desc().unique_id) !=
      uid_of_input_in_ctx.end()) {
    return;
  }
  uid_of_input_in_ctx.insert(storage.get()->get_npu_graph_desc().unique_id);
  input_storage_impls.emplace_back(storage);
  return;
}

void NpuGraphContextManager::AddOutputStorage(
    const c10::intrusive_ptr<StorageImpl> storage) {
  auto npu_ctx = GetDeviceContext<OutputContext>(
      storage.get()->device().index(), output_contexts_);
  std::lock_guard<std::mutex> lock(npu_ctx->ctx_lock);
  npu_ctx->output_storage_impl.emplace(
      storage.get()->get_npu_graph_desc().unique_id,
      c10::weak_intrusive_ptr<StorageImpl>(storage));
  return;
}

void NpuGraphContextManager::EraseOutputStorage(
    DeviceIndex device_idx,
    uint64_t storage_id) {
  auto npu_ctx = GetDeviceContext<OutputContext>(device_idx, output_contexts_);
  std::lock_guard<std::mutex> lock(npu_ctx->ctx_lock);
  npu_ctx->output_storage_impl.erase(storage_id);
}

std::vector<StorageImpl*> NpuGraphContextManager::GetAllStorageOfLiveTensors(
    DeviceIndex device_idx) {
  std::vector<StorageImpl*> storages;
  for (const auto& npu_ctx : output_contexts_) {
    std::lock_guard<std::mutex> lock(npu_ctx.second.get()->ctx_lock);
    for (auto& weak_storage : npu_ctx.second.get()->output_storage_impl) {
      auto storage_ptr = weak_storage.second.lock();
      if (storage_ptr) {
        storages.push_back(storage_ptr.get());
      }
    }
  }
  return storages;
}

void NpuGraphContextManager::AddInputStorage(
    const c10::intrusive_ptr<StorageImpl> storage) {
  auto npu_data_ctx = GetDeviceContext<InputContext>(
      storage.get()->device().index(), input_contexts_);
  std::lock_guard<std::mutex> lock(npu_data_ctx->ctx_lock);
  npu_data_ctx->AddInput(storage);
  return;
}

void NpuGraphContextManager::EraseInputStorage(DeviceIndex device_idx) {
  auto npu_data_ctx =
      GetDeviceContext<InputContext>(device_idx, input_contexts_);
  std::lock_guard<std::mutex> lock(npu_data_ctx->ctx_lock);
  npu_data_ctx->input_storage_impls.clear();
  npu_data_ctx->uid_of_input_in_ctx.clear();
}

std::vector<StorageImpl*> NpuGraphContextManager::GetAllInputStorages(
    DeviceIndex device_idx) {
  std::vector<StorageImpl*> data_storages;
  auto npu_data_ctx =
      GetDeviceContext<InputContext>(device_idx, input_contexts_);
  std::lock_guard<std::mutex> lock(npu_data_ctx->ctx_lock);
  for (auto& data_storage : npu_data_ctx->input_storage_impls) {
    data_storages.push_back(data_storage.get());
  }
  return data_storages;
}
} // namespace graph
} // namespace npu
} // namespace c10