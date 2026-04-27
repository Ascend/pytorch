#pragma once

#include <sstream>
#include "torch_npu/csrc/logging/LogContext.h"

namespace torch_npu {
namespace acl_log {

bool ShouldLogAclApi(const char* api_name);

inline std::shared_ptr<npu_logging::Logger>& GetAclLogger()
{
    static std::shared_ptr<npu_logging::Logger> logger = npu_logging::logging().getLogger("torch_npu.acl");
    return logger;
}

}  // namespace acl_log
}  // namespace torch_npu


#define ACL_CALL_LOG(api_name, args_expr)                          \
    do                                                            \
    {                                                             \
        if (torch_npu::acl_log::ShouldLogAclApi(api_name))         \
        {                                                         \
            try                                                   \
            {                                                     \
                std::ostringstream _acl_oss;                     \
                _acl_oss << args_expr;                          \
                TORCH_NPU_LOGI(torch_npu::acl_log::GetAclLogger(), "[ACL CALL] %s(%s)", api_name, _acl_oss.str().c_str()); \
                ASCEND_LOGI("[ACL CALL] %s(%s)", api_name, _acl_oss.str().c_str()); \
            }                                                     \
            catch (...)                                           \
            {                                                     \
            }                                                     \
        }                                                         \
    } while (0)
