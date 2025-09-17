#pragma once

#if defined(USE_NPU)

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"

typedef void* NPUdeviceptr;

typedef void* NPUfunction;

#define AOTI_RUNTIME_DEVICE_CHECK(EXPR)                                                                                \
    do {                                                                                                               \
        const aclError code = EXPR;                                                                                    \
        if (code != ACL_SUCCESS) {                                                                                     \
            throw std::runtime_error(std::string("NPU error core: ") + std::to_string(code) + std::string(" ") +       \
                                     std::string(__FILE__) + std::string(":") + std::to_string(__LINE__));             \
        }                                                                                                              \
    } while (0)

namespace torch::aot_inductor {

using DeviceStreamType = aclrtStream;

} // namespace torch::aot_inductor

#else

#define AOTI_RUNTIME_DEVICE_CHECK(EXPR)                                                                                \
    bool ok = EXPR;                                                                                                    \
    if (!ok) {                                                                                                         \
        throw std::runtime_error("CPU runtime error");                                                                 \
    }

namespace torch::aot_inductor {

using DeviceStreamType = void*;

} // namespace torch::aot_inductor

#endif // USE_NPU
