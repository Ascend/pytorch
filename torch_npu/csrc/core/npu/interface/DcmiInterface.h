#pragma once

#include "third_party/dcmi/inc/dcmi_interface_api.h"

namespace c10_npu {
namespace dcmi {

int DcmiInit(void);
int DcmiGetCardNumList(int *card_num, int *card_list, int list_len);
int DcmiGetAffinityCpuInfoByDeviceId(int card_id, int device_id, char *affinity_cpu, int *length);
int DcmiGetDeviceIdInCard(int card_id, int *device_id_max, int *mcu_id, int *cpu_id);

}

}