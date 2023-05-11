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
