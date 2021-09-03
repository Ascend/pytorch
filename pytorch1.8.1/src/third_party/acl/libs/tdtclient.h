#include <stdint.h>

#ifdef __cplusplus
using TDT_StatusT = uint32_t;
#else
typedef uint32_t TDT_StatusT;
#endif

extern "C" {

TDT_StatusT TsdOpen(const uint32_t phyDeviceId, const uint32_t rankSize);
TDT_StatusT TsdClose(const uint32_t phyDeviceId);
}
