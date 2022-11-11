#include "torch_npu/csrc/framework/graph/util/NPUGraphContextManager.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

namespace at_npu {
namespace native {

void InputContext::AddInput(const c10::intrusive_ptr<c10::StorageImpl>& storage) {
  if (uid_of_input_in_ctx.find(torch_npu::NPUBridge::GetNpuStorageImpl(storage.get())->get_npu_graph_desc().unique_id) !=
      uid_of_input_in_ctx.end()) {
    return;
  }
  uid_of_input_in_ctx.insert(torch_npu::NPUBridge::GetNpuStorageImpl(storage.get())->get_npu_graph_desc().unique_id);
  input_storage_impls.emplace_back(storage);
  return;
}

void NpuGraphContextManager::AddOutputStorage(
    const c10::intrusive_ptr<c10::StorageImpl> storage) {
  auto npu_ctx = GetDeviceContext<OutputContext>(
      storage.get()->device().index(), output_contexts_);
  std::lock_guard<std::mutex> lock(npu_ctx->ctx_lock);
  npu_ctx->output_storage_impl.emplace(
      torch_npu::NPUBridge::GetNpuStorageImpl(storage.get())->get_npu_graph_desc().unique_id,
      c10::weak_intrusive_ptr<c10::StorageImpl>(storage));
  return;
}

void NpuGraphContextManager::EraseOutputStorage(
    c10::DeviceIndex device_idx,
    uint64_t storage_id) {
  auto npu_ctx = GetDeviceContext<OutputContext>(device_idx, output_contexts_);
  std::lock_guard<std::mutex> lock(npu_ctx->ctx_lock);
  npu_ctx->output_storage_impl.erase(storage_id);
}

std::vector<c10::StorageImpl*> NpuGraphContextManager::GetAllStorageOfLiveTensors(
    c10::DeviceIndex device_idx) {
  std::vector<c10::StorageImpl*> storages;
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
    const c10::intrusive_ptr<c10::StorageImpl> storage) {
  auto npu_data_ctx = GetDeviceContext<InputContext>(
      storage.get()->device().index(), input_contexts_);
  std::lock_guard<std::mutex> lock(npu_data_ctx->ctx_lock);
  npu_data_ctx->AddInput(storage);
  return;
}

void NpuGraphContextManager::EraseInputStorage(c10::DeviceIndex device_idx) {
  auto npu_data_ctx =
      GetDeviceContext<InputContext>(device_idx, input_contexts_);
  std::lock_guard<std::mutex> lock(npu_data_ctx->ctx_lock);
  npu_data_ctx->input_storage_impls.clear();
  npu_data_ctx->uid_of_input_in_ctx.clear();
}

std::vector<c10::StorageImpl*> NpuGraphContextManager::GetAllInputStorages(
    c10::DeviceIndex device_idx) {
  std::vector<c10::StorageImpl*> data_storages;
  auto npu_data_ctx =
      GetDeviceContext<InputContext>(device_idx, input_contexts_);
  std::lock_guard<std::mutex> lock(npu_data_ctx->ctx_lock);
  for (auto& data_storage : npu_data_ctx->input_storage_impls) {
    data_storages.push_back(data_storage.get());
  }
  return data_storages;
}

std::vector<c10::DeviceIndex> NpuGraphContextManager::GetDevicesHasLiveTensor() {
  std::lock_guard<std::mutex> lock(lock_);
  std::vector<c10::DeviceIndex> res;
  for (auto &item : output_contexts_) {
    std::lock_guard<std::mutex> lock(item.second->ctx_lock);
    if (!item.second->output_storage_impl.empty()) {
      res.push_back(item.first);
    }
  }
  return res;
}

void NpuGraphContextManager::AddNoneOutputNode(const NodePtr none_out_node) {
  auto npu_output_ctx =
    GetDeviceContext<OutputContext>(c10_npu::current_device(),
                                    output_contexts_);
  std::lock_guard<std::mutex> lock(npu_output_ctx->ctx_lock);
  npu_output_ctx->none_output_nodes.emplace_back(none_out_node);
}

std::vector<NodePtr> NpuGraphContextManager::GetNoneOutputNode(c10::DeviceIndex device_idx) {
  auto npu_output_ctx =
    GetDeviceContext<OutputContext>(device_idx,
                                    output_contexts_);
  return npu_output_ctx->none_output_nodes;
}

void NpuGraphContextManager::EraseNoneOutputNode(c10::DeviceIndex device_idx) {
  auto npu_output_ctx =
    GetDeviceContext<OutputContext>(device_idx,
                                    output_contexts_);
    npu_output_ctx->none_output_nodes.clear();
}
} // namespace native
} // namespace at_npu

