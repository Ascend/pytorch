#ifndef THNP_GETAFFINITY_INC
#define THNP_GETAFFINITY_INC
#include <string>

std::string GetAffinityCPUBaseInfo(int card_id);
c10_npu::CoreIdRange parseAffinityCPU(const std::string cpuString);
void GetExclusiveAffinityCPU();
c10_npu::CoreIdRange GetAssignAffinityCPU(int card_id);

#endif