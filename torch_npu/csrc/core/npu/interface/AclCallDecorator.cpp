#include "torch_npu/csrc/core/npu/interface/AclCallDecorator.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include "torch_npu/csrc/logging/Logger.h"
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace torch_npu {
namespace acl_log {

namespace {
std::once_flag g_once;
bool g_filter_disabled = false;
bool g_log_enabled = false;

void InitFilter()
{
    const char* v = std::getenv("TORCH_NPU_LOGS_FILTER");
    g_filter_disabled = (v != nullptr);
    // Log gate resolved once: check PTA acl-logger level.
    auto& logger = GetAclLogger();
    bool pta_on = (logger && logger->getAllowLevel() <= npu_logging::LoggingLevel::INFO);
    bool asc_on = false;
    try {
        asc_on = c10_npu::option::OptionsManager::isACLGlobalLogOn(ACL_INFO);
    } catch (...) {
        asc_on = false;
    }
    g_log_enabled = pta_on || asc_on;
}
}  // anonymous namespace


bool ShouldLogAclApi(const char* api_name)
{
    std::call_once(g_once, InitFilter);

    if (!g_log_enabled) {
        return false;
    }
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
