#ifndef __RUNTIME_UTILS_H__
#define __RUNTIME_UTILS_H__

#include <set>
#include <unordered_map>

#include "common/common.h"
#include "common/visible.h"
#include "ir/graph.h"

namespace fxrt {
namespace runtime {
// Environment variable config keys
constexpr const char kEnableAclGraphEnv[] = "FXRT_ENABLE_ACLGRAPH"; // set value `on` to enable aclgraph

// Canonical input position indices
constexpr size_t kFirstInput = 0;
constexpr size_t kSecondInput = 1;
extern const std::unordered_map<ops::Op, size_t> opsOutputFromInputIndex;
extern const std::unordered_map<ops::Op, size_t> opsOutputValueFromInputIndex;
extern const std::set<ops::Op> dummyOpsSet;
extern const std::set<ops::Op> forceResizeOpsSet;

// Check whether AclGraph mode is enabled via environment variable (cached on first call).
FXRT_EXPORT bool IsAclGraphEnabled();

inline bool IsSkipRecordRefCount(ir::NodePtr tensor) {
  CHECK_IF_NULL(tensor);
  return (tensor->op == ops::Op_End || tensor->op == ops::Op_load || tensor->op == ops::Op_update_state);
}

inline bool IsNodeOutputFromInput(ir::NodePtr tensor) {
  CHECK_IF_NULL(tensor);
  return opsOutputFromInputIndex.find(tensor->op) != opsOutputFromInputIndex.end();
}

inline bool IsDummyNode(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  return dummyOpsSet.find(node->op) != dummyOpsSet.end();
}

inline bool IsSkipBuildOpRunner(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  return (
      IsNodeOutputFromInput(node) || node->op == ops::Op_End || node->op == ops::Op_make_tuple ||
      node->op == ops::Op_tuple_getitem);
}

inline bool IsDAKernelNeedForceResize(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  return forceResizeOpsSet.find(node->op) != forceResizeOpsSet.end();
}

} // namespace runtime
} // namespace fxrt
#endif // __RUNTIME_UTILS_H__
