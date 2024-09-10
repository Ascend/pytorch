#ifndef FLOP_COUNT_CONTEXT_H
#define FLOP_COUNT_CONTEXT_H

#include <cstdint>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

class TORCH_NPU_API FlopCountContext {
public:
    int64_t recordedCount;
    int64_t traversedCount;

    static FlopCountContext& GetInstance();
    void enable();
    void disable();
    void pause();
    void resume();
    void reset();
    bool isEnabled();
    bool isPaused();

private:
    bool isEnabled_;
    bool isPaused_;
    FlopCountContext() : isEnabled_(false),  isPaused_(false), recordedCount(0), traversedCount(0) {}
};

#endif // FLOP_COUNT_CONTEXT_H
