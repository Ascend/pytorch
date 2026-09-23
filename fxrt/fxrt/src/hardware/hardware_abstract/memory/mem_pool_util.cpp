#include "hardware/hardware_abstract/memory/mem_pool_util.h"
#include <map>

namespace fxrt {
namespace memory {
namespace mem_pool {
const std::map<MemType, std::string> kMemTypeStr = {
    {MemType::kWeight, "Weight"},
    {MemType::kConstantValue, "ConstantValue"},
    {MemType::kKernel, "Kernel"},
    {MemType::kGraphOutput, "GraphOutput"},
    {MemType::kSomas, "Somas"},
    {MemType::kSomasOutput, "SomasOutput"},
    {MemType::kGeConst, "GeConst"},
    {MemType::kGeFixed, "GeFixed"},
    {MemType::kBatchMemory, "BatchMemory"},
    {MemType::kContinuousMemory, "ContinuousMemory"},
    {MemType::kPyNativeInput, "PyNativeInput"},
    {MemType::kPyNativeOutput, "PyNativeOutput"},
    {MemType::kWorkSpace, "WorkSpace"},
    {MemType::kOther, "Other"}};

std::string MemTypeToStr(MemType memType) {
  return kMemTypeStr.at(memType);
}
} // namespace mem_pool
} // namespace memory
} // namespace fxrt
