#include "torch_npu/csrc/flopcount/FlopCountContext.h"

FlopCountContext &FlopCountContext::GetInstance()
{
    static FlopCountContext instance;
    return instance;
}

bool FlopCountContext::isEnabled()
{
    return isEnabled_;
}

bool FlopCountContext::isPaused()
{
    return isPaused_;
}

void FlopCountContext::enable()
{
    isEnabled_ = true;
}

void FlopCountContext::disable()
{
    isEnabled_ = false;
}

void FlopCountContext::pause()
{
    isPaused_ = true;
}

void FlopCountContext::resume()
{
    isPaused_ = false;
}

void FlopCountContext::reset()
{
    isEnabled_ = false;
    isPaused_ = false;
    recordedCount = 0;
    traversedCount = 0;
}
