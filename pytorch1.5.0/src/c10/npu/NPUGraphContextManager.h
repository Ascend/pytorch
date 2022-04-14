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

#include <c10/core/Device.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/order_preserving_flat_hash_map.h>
#include <c10/npu/NPUGraph.h>
#include <map>
#include <mutex>
namespace c10 {
struct StorageImpl;

namespace npu {
namespace graph {
// do not affect the life cycle of StorageImpl by weak intrusive ptr
struct OutputContext {
  std::mutex ctx_lock;

  // must be ordered container for hash key generate
  ska_ordered::order_preserving_flat_hash_map<
      uint64_t,
      c10::weak_intrusive_ptr<StorageImpl>>
      output_storage_impl;
  std::vector<NodePtr> none_output_nodes;
};

// affect the life cycle of StorageImpl
struct InputContext {
public:
  void AddInput(const c10::intrusive_ptr<StorageImpl>& storage);

public:
  std::mutex ctx_lock;

  // must be ordered container for hash key generate
  std::vector<c10::intrusive_ptr<StorageImpl>> input_storage_impls;
  ska::flat_hash_set<uint64_t> uid_of_input_in_ctx;
};

class C10_API NpuGraphContextManager {
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

  void AddOutputStorage(const c10::intrusive_ptr<StorageImpl> storage);
  void EraseOutputStorage(DeviceIndex device_idx, uint64_t storage_id);

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
  void AddInputStorage(const c10::intrusive_ptr<StorageImpl> storage);

  // used for cpu tensor for device id of it must be 0 or -1
  // long name is used to avoid wrong calls
  void AddInputStorageForCpuTensorBySpecifiedDeviceId(
      const c10::intrusive_ptr<StorageImpl> storage,
      DeviceIndex device_index);

  void EraseInputStorage(DeviceIndex device_idx);

  std::vector<StorageImpl*> GetAllStorageOfLiveTensors(DeviceIndex device_idx);

  std::vector<StorageImpl*> GetAllInputStorages(DeviceIndex device_idx);

  std::vector<DeviceIndex> GetDevicesHasLiveTensor();

  void AddNoneOutputNode(const NodePtr none_out_node);

  std::vector<NodePtr> GetNoneOutputNode(DeviceIndex device_idx);

  void EraseNoneOutputNode(DeviceIndex device_idx);
private:
  NpuGraphContextManager() = default;

  template <typename ctx_type>
  ctx_type* GetDeviceContext(
      DeviceIndex device_idx,
      std::map<DeviceIndex, std::unique_ptr<ctx_type>>& ctxs) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = ctxs.find(device_idx);
    if (it == ctxs.end()) {
      it = ctxs.emplace(device_idx, new ctx_type()).first;
    }
    return it->second.get();
  }

  std::mutex lock_;
  std::map<DeviceIndex, std::unique_ptr<OutputContext>> output_contexts_;
  std::map<DeviceIndex, std::unique_ptr<InputContext>> input_contexts_;
};
} // namespace graph
} // namespace npu
} // namespace c10