#pragma once

#include <map>
#include <mutex>

#include <c10/core/Device.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/order_preserving_flat_hash_map.h>
#include <c10/core/StorageImpl.h>

#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/framework/graph/util/NPUGraph.h"

namespace at_npu {
namespace native {
// do not affect the life cycle of StorageImpl by weak intrusive ptr
struct OutputContext {
  std::mutex ctx_lock;

  // must be ordered container for hash key generate
  ska_ordered::order_preserving_flat_hash_map<
      uint64_t,
      c10::weak_intrusive_ptr<c10::StorageImpl>>
      output_storage_impl;
  std::vector<NodePtr> none_output_nodes;
};

// affect the life cycle of StorageImpl
struct InputContext {
public:
  void AddInput(const c10::intrusive_ptr<c10::StorageImpl>& storage);

public:
  std::mutex ctx_lock;

  // must be ordered container for hash key generate
  std::vector<c10::intrusive_ptr<c10::StorageImpl>> input_storage_impls;
  ska::flat_hash_set<uint64_t> uid_of_input_in_ctx;
};

class NpuGraphContextManager {
public:
  static NpuGraphContextManager& GetInstance() {
    static NpuGraphContextManager manager;
    return manager;
  }

  NpuGraphContextManager(const NpuGraphContextManager&) = delete;
  NpuGraphContextManager(NpuGraphContextManager&&) = delete;
  NpuGraphContextManager& operator=(const NpuGraphContextManager&) = delete;
  NpuGraphContextManager& operator=(NpuGraphContextManager&&) = delete;

  ~NpuGraphContextManager() = default;

  void AddOutputStorage(const c10::intrusive_ptr<c10::StorageImpl> storage);
  void EraseOutputStorage(c10::DeviceIndex device_idx, uint64_t storage_id);

  /**
   * NB
   * Consider this scenario:
   * def test(t):
   *     y = torch.ones([2,3]).npu()
   *     t += y
   *     return t
   * t = torch.ones([2,3]).npu()
   * t = test(t)
   * print(t1)
   *
   * The life cycle of y is as long as the function test
   * if we store the weak ptr, when we run graph
   * we can not get she StorageImpl of y
   * so we need to storage the intrusive_ptr of y
   * which represent "Data" passed form host to device
   *
   */
  void AddInputStorage(const c10::intrusive_ptr<c10::StorageImpl> storage);

  void EraseInputStorage(c10::DeviceIndex device_idx);

  std::vector<c10::StorageImpl*> GetAllStorageOfLiveTensors(c10::DeviceIndex device_idx);

  std::vector<c10::StorageImpl*> GetAllInputStorages(c10::DeviceIndex device_idx);

  std::vector<c10::DeviceIndex> GetDevicesHasLiveTensor();

  void AddNoneOutputNode(const NodePtr none_out_node);

  std::vector<NodePtr> GetNoneOutputNode(c10::DeviceIndex device_idx);

  void EraseNoneOutputNode(c10::DeviceIndex device_idx);

private:
  NpuGraphContextManager() = default;

  template <typename ctx_type>
  ctx_type* GetDeviceContext(
      c10::DeviceIndex device_idx,
      std::map<c10::DeviceIndex, std::unique_ptr<ctx_type>>& ctxs) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = ctxs.find(device_idx);
    if (it == ctxs.end()) {
      it = ctxs.emplace(device_idx, new ctx_type()).first;
    }
    return it->second.get();
  }

  std::mutex lock_;
  std::map<c10::DeviceIndex, std::unique_ptr<OutputContext>> output_contexts_;
  std::map<c10::DeviceIndex, std::unique_ptr<InputContext>> input_contexts_;
};
} // namespace native
} // namespace at_npu