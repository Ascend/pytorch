#ifndef NPUVARIABLES_H
#define NPUVARIABLES_H

namespace c10_npu {
enum class SocVersion {
  UnsupportedSocVersion = -1,
  Ascend910PremiumA = 100,
  Ascend910ProA,
  Ascend910A,
  Ascend910ProB,
  Ascend910B,
  Ascend310P1 = 200,
  Ascend310P2,
  Ascend310P3,
  Ascend310P4,
  Ascend910B1 = 220,
  Ascend910B2,
  Ascend910B2C,
  Ascend910B3,
  Ascend910B4,
  Ascend310B1 = 240,
  Ascend310B2,
  Ascend310B3,
  Ascend310B4,
  Ascend910C1 = 250,
  Ascend910C2,
  Ascend910C3,
  Ascend910C4
};

void SetSocVersion(const char* const socVersion);

const SocVersion& GetSocVersion();

bool IsSupportInfNan();

bool IsBF16Supported();
}  // namespace c10_npu
#endif

