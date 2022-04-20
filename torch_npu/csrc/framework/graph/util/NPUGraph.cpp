#include "NPUGraph.h"

namespace at_npu {
namespace native {
hash_t Value::GetValueHash() const {
  return value_hash_.value_or(
      hash_utils::hash_combine(cur_node_->GetNodeHash(), value_index_));
}
} // namespace native
} // namespace at_npu