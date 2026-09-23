#include "runtime/utils/utils.h"
#include <set>
#include <unordered_map>
#include <vector>

namespace fxrt {
namespace runtime {
const std::unordered_map<ops::Op, size_t> opsOutputFromInputIndex = {
    {ops::Op_return, kFirstInput},
    {ops::Op_depend, kFirstInput},
    {ops::Op_load, kFirstInput},
    {ops::Op_update_state, kFirstInput},
};

const std::unordered_map<ops::Op, size_t> opsOutputValueFromInputIndex = {
    {ops::Op_reshape_ext, kFirstInput},
};

const std::set<ops::Op> dummyOpsSet = {
    ops::Op_tuple_getitem,
    ops::Op_depend,
    ops::Op_make_tuple,
    ops::Op_reshape_ext,
};

const std::set<ops::Op> forceResizeOpsSet = {
    ops::Op_flash_attention_score,
    ops::Op_paged_attention,
};

// Check whether AclGraph mode is enabled (result is cached after the first call).
bool IsAclGraphEnabled() {
  static bool ret = []() {
    static const char* enable_acl_graph = std::getenv(fxrt::runtime::kEnableAclGraphEnv);
    return (enable_acl_graph != nullptr) && (std::string_view(enable_acl_graph) == "on");
  }();
  return ret;
}
} // namespace runtime
} // namespace fxrt
