/*
 * Copyright: Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Author: huawei
 * Date: 2021-03-17 17:46:08
 * @LastEditors: huawei
 * @LastEditTime: 2022-11-03 11:17:04
 * Description: DCMI API Reference
 */

/***************************************************************************************/

#ifdef __linux
#define DCMIDLLEXPORT
#else
#define DCMIDLLEXPORT _declspec(dllexport)
#endif

#define TOPO_INFO_MAX_LENTH   32 // topo info max length

DCMIDLLEXPORT int dcmi_init(void);

DCMIDLLEXPORT int dcmi_get_card_num_list(int *card_num, int *card_list, int list_len);  // card_num is the number of device.

DCMIDLLEXPORT int dcmi_get_affinity_cpu_info_by_device_id(int card_id, int device_id, char *affinity_cpu, int *length);  // card_id is the ID of NPU card.

DCMIDLLEXPORT int dcmi_get_device_id_in_card(int card_id, int *device_id_max, int *mcu_id, int *cpu_id);
