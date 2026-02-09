#include "torch_npu/csrc/core/npu/interface/AclCallDecorator.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

namespace torch_npu {
namespace acl_log {

namespace {
std::once_flag g_once;
bool g_filter_disabled = false;

void InitFilter()
{
    const char* v = std::getenv("TORCH_NPU_LOGS_FILTER");
    g_filter_disabled = (v != nullptr);
}
}  // anonymous namespace


bool ShouldLogAclApi(const char* api_name)
{
    std::call_once(g_once, InitFilter);

    if (!g_filter_disabled) {
        if (std::strstr(api_name, "Query") != nullptr ||
            std::strstr(api_name, "Event") != nullptr) {
            return false;
        }
    }

    return true;
}

}  // namespace acl_log
}  // namespace torch_npu
