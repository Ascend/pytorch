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

#include <torch_npu/csrc/framework/graph/util/NPUGraph.h>
#include <torch_npu/csrc/framework/graph/util/NPUHashUtils.h>
#include <third_party/acl/inc/graph/tensor.h>

#include <unordered_map>

namespace at_npu {
namespace native {

using namespace at_npu::native::hash_utils;

class GraphCache {
public:
  c10::optional<uint32_t> GetCacheGraphId(
      const std::vector<hash_t>& inputs_topo_hash,
      const std::vector<hash_t>& inputs_shape_hash,
      const std::vector<hash_t>& outputs_topo_hash,
      const std::vector<hash_t>& outputs_shape_hash,
      uint32_t cur_graph_id);

  static hash_t GetTensorTopoHash(
      const at_npu::native::Value& graph_value,
      const ge::TensorDesc& tensor_desc);

  static hash_t GetTensorShapeHash(
      const hash_t& topo_hash,
      const ge::TensorDesc& tensor_desc);

private:
  static hash_t GetGraphTopoHash(
      const std::vector<hash_t>& inputs_topo_hash,
      const std::vector<hash_t>& outputs_topo_hash);

  static hash_t GetGraphShapeHash(
      const std::vector<hash_t>& inputs_shape_hash,
      const std::vector<hash_t>& outputs_shape_hash);

  std::unordered_map<hash_t, std::unordered_map<hash_t, uint32_t>> graph_cache_;
};
} // namespace native
} // namespace at_npu
