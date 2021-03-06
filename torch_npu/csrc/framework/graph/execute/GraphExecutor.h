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
#include "torch_npu/csrc/framework/graph/cache/GraphCacher.h"
#include "torch_npu/csrc/framework/graph/util/NPUGraph.h"
#include <c10/core/StorageImpl.h>
#include <c10/macros/Export.h>

#ifdef SUCCESS
#undef SUCCESS
#endif
#ifdef FAILED
#undef FAILED
#endif
#include <third_party/acl/inc/ge/ge_api.h>

#include <vector>

namespace at_npu {
namespace native {

using namespace at_npu::native::hash_utils;

using GeOutPutOpType =
    std::vector<std::pair<ge::Operator, std::vector<size_t>>>;

struct CombinedInfo {
  std::vector<NodePtr> nodes;
  std::vector<ge::Tensor> tensors;
  std::vector<hash_t> hash_of_topo_and_attr;
  std::vector<hash_t> hash_of_shape;
};

class GraphExecutor {
public:
  GraphExecutor(const GraphExecutor&) = delete;
  GraphExecutor(GraphExecutor&&) = delete;
  GraphExecutor& operator=(const GraphExecutor&) = delete;
  GraphExecutor& operator=(GraphExecutor&&) = delete;
  ~GraphExecutor() = default;

  void ConstructAndExecuteGraph();

  static GraphExecutor& GetInstance() {
    static GraphExecutor instance;
    return instance;
  }

  void SetVerbose(bool verbose) {
    verbose_ = verbose;
  }

  void Finalize();

private:
  GraphExecutor() = default;

  void Init();

  /**
   * NB
   * Currently, in graph mode, there are two limitations
   * 1, after your first graph launching, you can not change device,
   * the init_device_id_ will be the id
   * of first device which has input tensor.
   *
   * 2, you can not construct graph in two different device.
   */
  bool CheckDeviceIdAndInit();

  void RunGraph(
      uint32_t graph_id,
      CombinedInfo& inputs,
      CombinedInfo& outputs);

  static void ConstructOps(CombinedInfo& output);

  std::vector<ge::Operator> GetInputOps();

  GeOutPutOpType GetOutputOps();

  CombinedInfo GetInputCombinedInfo();

  CombinedInfo GetOutputCombinedInfo();

  static ge::Tensor PrepareInputTensor(
      const c10::StorageImpl* const storage,
      const ge::TensorDesc& desc,
      size_t capacity);

  static ge::Tensor PrepareOutputTenosr(
      c10::StorageImpl* storage,
      const ge::TensorDesc& desc);

  void ResetGraphOutputs();

  void RefreshGraphInputs();

  void ClearDataStore();

  static uint32_t graph_id;

  c10::DeviceIndex init_device_id_ = -1;

  bool verbose_ = false;

  std::unique_ptr<ge::Session> session_ = nullptr;

  GraphCache cacher_;
};
} // namespace native
} // namespace at_npu