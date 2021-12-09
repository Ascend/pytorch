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

#include "GraphCacher.h"

namespace at {
namespace native {
namespace npu {
hash_t GraphCache::GetGraphTopoHash(
    const std::vector<hash_t>& inputs_topo_hash,
    const std::vector<hash_t>& outputs_topo_hash) {
  hash_t graph_topo_hash = multi_hash(inputs_topo_hash);
  graph_topo_hash = multi_hash(graph_topo_hash, outputs_topo_hash);
  return graph_topo_hash;
}

hash_t GraphCache::GetGraphShapeHash(
    const std::vector<hash_t>& inputs_shape_hash,
    const std::vector<hash_t>& outputs_shape_hash) {
  hash_t graph_shape_hash = multi_hash(inputs_shape_hash);
  graph_shape_hash = multi_hash(graph_shape_hash, outputs_shape_hash);
  return graph_shape_hash;
}

hash_t GraphCache::GetTensorShapeHash(
    const hash_t& topo_hash,
    const ge::TensorDesc& tensor_desc) {
  return multi_hash(
      topo_hash,
      tensor_desc.GetOriginShape().GetDimNum(),
      tensor_desc.GetOriginShape().GetDims());
}

hash_t GraphCache::GetTensorTopoHash(
    const Value& graph_value,
    const ge::TensorDesc& tensor_desc) {
  return multi_hash(
      graph_value.GetValueHash(),
      tensor_desc.GetDataType(),
      tensor_desc.GetOriginFormat(),
      tensor_desc.GetFormat());
}

c10::optional<uint32_t> GraphCache::GetCacheGraphId(
    const std::vector<hash_t>& inputs_topo_hash,
    const std::vector<hash_t>& inputs_shape_hash,
    const std::vector<hash_t>& outputs_topo_hash,
    const std::vector<hash_t>& outputs_shape_hash,
    uint32_t cur_graph_id) {
  hash_t topo_hash = GetGraphTopoHash(inputs_topo_hash, outputs_topo_hash);
  hash_t shape_hash = GetGraphShapeHash(inputs_shape_hash, outputs_shape_hash);
  auto iter = graph_cache_.find(topo_hash);
  if (iter != graph_cache_.end()) {
    auto& shape_map = iter->second;
    auto shape_iter = shape_map.find(shape_hash);
    if (shape_iter != shape_map.end()) {
      return shape_iter->second;
    } else {
      shape_map[shape_hash] = cur_graph_id;
    }
  } else {
    graph_cache_[topo_hash] = {{shape_hash, cur_graph_id}};
  }
  return c10::nullopt;
}
} // namespace npu
} // namespace native
} // namespace at