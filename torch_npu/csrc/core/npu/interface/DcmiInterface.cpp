#include <string>
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/interface/DcmiInterface.h"

namespace c10_npu {
namespace dcmi {

#undef TORCH_NPU_LOAD_FUNC
#define TORCH_NPU_LOAD_FUNC(funcName) \
    TORCH_NPU_REGISTER_FUNCTION(libdcmi, funcName)

#undef TORCH_NPU_GET_FUNC
#define TORCH_NPU_GET_FUNC(funcName) \
    TORCH_NPU_GET_FUNCTION(libdcmi, funcName)

TORCH_NPU_REGISTER_LIBRARY(libdcmi)
TORCH_NPU_LOAD_FUNC(dcmi_get_affinity_cpu_info_by_device_id)
TORCH_NPU_LOAD_FUNC(dcmi_init)
TORCH_NPU_LOAD_FUNC(dcmi_get_device_id_in_card)
TORCH_NPU_LOAD_FUNC(dcmi_get_card_num_list)

// dcmiv2 functions
TORCH_NPU_LOAD_FUNC(dcmiv2_init)
TORCH_NPU_LOAD_FUNC(dcmiv2_get_affinity_cpu_info_by_device_id)
TORCH_NPU_LOAD_FUNC(dcmiv2_get_affinity_cpu_info_by_dev_id)
TORCH_NPU_LOAD_FUNC(dcmiv2_get_device_list)

int DcmiInit(void)
{
    using dcmiInitFunc = int(*)(void);
    static dcmiInitFunc func_v2 = nullptr;
    func_v2 = (dcmiInitFunc)TORCH_NPU_GET_FUNC(dcmiv2_init);
    if (func_v2 != nullptr) {
        return func_v2();
    }
    static dcmiInitFunc func = nullptr;
    func = (dcmiInitFunc)TORCH_NPU_GET_FUNC(dcmi_init);
    if (func == nullptr) {
        TORCH_CHECK(false, "Failed to find function dcmi_init, "
                    " maybe your hdk version is too low, please upgrade it.", PTA_ERROR(ErrCode::NOT_FOUND))
    }
    return func();
}

int DcmiGetCardNumList(int *card_num, int *card_list, int list_len)
{
    using dcmiv2GetDeviceListFunc = int(*)(int *, int *, int);
    static dcmiv2GetDeviceListFunc func_v2 = nullptr;
    func_v2 = (dcmiv2GetDeviceListFunc)TORCH_NPU_GET_FUNC(dcmiv2_get_device_list);
    if (func_v2 != nullptr) {
        return func_v2(card_list, card_num, list_len);
    }
    using dcmiGetCardNumListFunc = int(*)(int *, int *, int);
    static dcmiGetCardNumListFunc func = nullptr;
    func = (dcmiGetCardNumListFunc)TORCH_NPU_GET_FUNC(dcmi_get_card_num_list);
    if (func == nullptr) {
        TORCH_CHECK(false, "Failed to find function dcmi_get_card_num_list, "
                    " maybe your hdk version is too low, please upgrade it.", PTA_ERROR(ErrCode::NOT_FOUND))
    }
    return func(card_num, card_list, list_len);
}

int DcmiGetAffinityCpuInfoByDeviceId(int card_id, int device_id, char *affinity_cpu, int *length)
{
    // Use dcmiv2_get_affinity_cpu_info_by_dev_id first
    using dcmiv2GetAffinityCpuInfoByDevIdFunc = int(*)(int, char *, int *);
    static dcmiv2GetAffinityCpuInfoByDevIdFunc func_v2_1 = nullptr;
    if (func_v2_1 == nullptr) {
        func_v2_1 = (dcmiv2GetAffinityCpuInfoByDevIdFunc)TORCH_NPU_GET_FUNC(dcmiv2_get_affinity_cpu_info_by_dev_id);
    }
    if (func_v2_1 != nullptr) {
        return func_v2_1(card_id, affinity_cpu, length);
    }

    using dcmiv2GetAffinityCpuInfoByDeviceIdFunc = int(*)(int, char *, int *);
    static dcmiv2GetAffinityCpuInfoByDeviceIdFunc func_v2 = nullptr;
    if (func_v2 == nullptr) {
        func_v2 = (dcmiv2GetAffinityCpuInfoByDeviceIdFunc)TORCH_NPU_GET_FUNC(dcmiv2_get_affinity_cpu_info_by_device_id);
    }
    if (func_v2 != nullptr) {
        return func_v2(card_id, affinity_cpu, length);
    }

    using dcmiGetAffinityCpuInfoByDeviceIdFunc = int(*)(int, int, char *, int *);
    static dcmiGetAffinityCpuInfoByDeviceIdFunc func = nullptr;
    if (func == nullptr) {
        func = (dcmiGetAffinityCpuInfoByDeviceIdFunc)TORCH_NPU_GET_FUNC(dcmi_get_affinity_cpu_info_by_device_id);
    }
    if (func != nullptr) {
        return func(card_id, device_id, affinity_cpu, length);
    }
    TORCH_CHECK(false, "Failed to get affinity cpu info, maybe your hdk version is too low, please upgrade it", PTA_ERROR(ErrCode::NOT_FOUND));
}

int DcmiGetDeviceIdInCard(int card_id, int *device_id_max, int *mcu_id, int *cpu_id)
{
    // Check if V2 interface exists to mock success
    using dcmiv2GetDeviceListFunc = int(*)(int *, int *, int);
    static dcmiv2GetDeviceListFunc func_v2 = nullptr;
    func_v2 = (dcmiv2GetDeviceListFunc)TORCH_NPU_GET_FUNC(dcmiv2_get_device_list);
    if (func_v2 != nullptr) {
        if (device_id_max) *device_id_max = 1;
        if (mcu_id) *mcu_id = 0;
        if (cpu_id) *cpu_id = 0;
        return 0; // Success
    }

    using dcmiGetDeviceIdInCardFunc = int(*)(int, int *, int *, int *);
    static dcmiGetDeviceIdInCardFunc func = nullptr;
    func = (dcmiGetDeviceIdInCardFunc)TORCH_NPU_GET_FUNC(dcmi_get_device_id_in_card);
    if (func == nullptr) {
        TORCH_CHECK(false, "Failed to find function dcmi_get_device_id_in_card, "
                    " maybe your hdk version is too low, please upgrade it", PTA_ERROR(ErrCode::NOT_FOUND))
    }
    return func(card_id, device_id_max, mcu_id, cpu_id);
}

}

}