#include <tdtclient.h>

extern "C" {
TDT_StatusT TsdOpen(const uint32_t phyDeviceId, const uint32_t rankSize){return 0;}
TDT_StatusT TsdClose(const uint32_t phyDeviceId){return 0;}
}
