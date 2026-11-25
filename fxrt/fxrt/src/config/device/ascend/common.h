#ifndef __CONFIG_DEVICE_ASCEND_COMMON_H__
#define __CONFIG_DEVICE_ASCEND_COMMON_H__

#include <cstdint>

namespace fxrt {
namespace config {
namespace ascend {
enum class AclCubeMathType : int8_t {
  KEEP_DTYPE = 0,
  ALLOW_FP32_DOWN_PRECISION = 1,
  USE_FP16 = 2,
  USE_HF32 = 3,
};

enum class SocVersion {
  UnsupportedSocVersion = -1,
  k910PremiumA = 100,
  k910ProA,
  k910A,
  k910ProB,
  k910B,
  k310P1 = 200,
  k310P2,
  k310P3,
  k310P4,
  k310P5,
  k310P7,
  k910B1 = 220,
  k910B2,
  k910B2C,
  k910B3,
  k910B4,
  k910B4_1,
  k310B1 = 240,
  k310B2,
  k310B3,
  k310B4,
  k910_9391 = 250,
  k910_9392,
  k910_9381,
  k910_9382,
  k910_9372,
  k910_9362,
  k950PR_9579 = 260,
  k950PR_9589,
  k950PR_9599,
  k950PR_958b,
  k950PR_957b,
  k950PR_957bx,
  k950PR_957c,
  k950PR_957d,
  k950PR_950z,
  k950DT_950x,
  k950DT_950y,
  k950DT_9571,
  k950DT_9572,
  k950DT_9573,
  k950DT_9574,
  k950DT_9575,
  k950DT_9576,
  k950DT_9577,
  k950DT_9578,
  k950DT_9581,
  k950DT_9582,
  k950DT_9582x,
  k950DT_9583,
  k950DT_9584,
  k950DT_9585,
  k950DT_9586,
  k950DT_9587,
  k950DT_9588,
  k950DT_9591,
  k950DT_9592,
  k950DT_9595,
  k950DT_9596,
  k950DT_95A1,
  k950DT_95A2
};

} // namespace ascend
} // namespace config
} // namespace fxrt
#endif // __CONFIG_DEVICE_ASCEND_COMMON_H__
