#ifndef FLOP_COUNT_H
#define FLOP_COUNT_H

#include "torch_npu/csrc/flopcount/FlopCountContext.h"

#define FLOP_COUNT(flopcount_func, ...) \
do { \
    FlopCountContext& context = FlopCountContext::GetInstance(); \
    if (context.isEnabled()) { \
        int64_t flops = flopcount_func(__VA_ARGS__); \
        context.traversedCount += flops; \
        if (!context.isPaused()) { \
            context.recordedCount += flops; \
        } \
    } \
} while (0)

#endif
