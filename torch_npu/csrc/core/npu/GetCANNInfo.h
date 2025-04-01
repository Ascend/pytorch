#ifndef THNP_GETCANNINFO_INC
#define THNP_GETCANNINFO_INC
#include "torch_npu/csrc/core/npu/NPUMacros.h"


TORCH_NPU_API std::string GetCANNVersion(const std::string& module = "CANN");

/*
support version format: a.b.c, a.b.RCd, a.b.Tg, a.b.RCd.alphaf
formula: ((a+1) * 100000000) + ((b+1) * 1000000) + ((c+1) * 10000) + ((d+1) * 100 + 5000) + ((g+1) * 100) - (100 - f)
*/
bool IsGteCANNVersion(const std::string version, const std::string module = "CANN");

bool IsGteDriverVersion(const std::string driverVersion);

#endif