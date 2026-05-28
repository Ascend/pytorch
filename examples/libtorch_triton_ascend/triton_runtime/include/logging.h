#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace triton_runtime {

inline bool is_log_enabled() {
    static bool enabled = []() {
        const char* env = std::getenv("LOG_TORCH_TRITON_RUNTIME");
        return env != nullptr && std::strcmp(env, "1") == 0;
    }();
    return enabled;
}

} // namespace triton_runtime

#define TRT_DEBUG(fmt, ...)                                                 \
    do {                                                                    \
        if (::triton_runtime::is_log_enabled()) {                          \
            std::fprintf(stderr, "[TRT] " fmt "\n", ##__VA_ARGS__);         \
        }                                                                   \
    } while (0)