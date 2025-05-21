#include <string>
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/interface/DcmiInterface.h"

namespace c10_npu {
namespace dcmi {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
    REGISTER_FUNCTION(libdcmi, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName) \
    GET_FUNCTION(libdcmi, funcName)

REGISTER_LIBRARY(libdcmi)
LOAD_FUNCTION(dcmi_get_affinity_cpu_info_by_device_id)
LOAD_FUNCTION(dcmi_init)
LOAD_FUNCTION(dcmi_get_device_id_in_card)
LOAD_FUNCTION(dcmi_get_card_num_list)

int DcmiInit(void)
{
    using dcmiInitFunc = int(*)(void);
    static dcmiInitFunc func = nullptr;
    func = (dcmiInitFunc)GET_FUNC(dcmi_init);
    if (func == nullptr) {
        TORCH_CHECK(false, "Failed to find function dcmi_init, "
                    " maybe your hdk version is too low, please upgrade it.", PTA_ERROR(ErrCode::NOT_FOUND))
    }
    return func();
}

int DcmiGetCardNumList(int *card_num, int *card_list, int list_len)
{
    using dcmiGetCardNumListFunc = int(*)(int *, int *, int);
    static dcmiGetCardNumListFunc func = nullptr;
    func = (dcmiGetCardNumListFunc)GET_FUNC(dcmi_get_card_num_list);
    if (func == nullptr) {
        TORCH_CHECK(false, "Failed to find function dcmi_get_card_num_list, "
                    " maybe your hdk version is too low, please upgrade it.", PTA_ERROR(ErrCode::NOT_FOUND))
    }
    return func(card_num, card_list, list_len);
}

int DcmiGetAffinityCpuInfoByDeviceId(int card_id, int device_id, char *affinity_cpu, int *length)
{
    using dcmiGetAffinityCpuInfoByDeviceIdFunc = int(*)(int, int, char *, int *);
    static dcmiGetAffinityCpuInfoByDeviceIdFunc func = nullptr;
    func = (dcmiGetAffinityCpuInfoByDeviceIdFunc)GET_FUNC(dcmi_get_affinity_cpu_info_by_device_id);
    if (func == nullptr) {
        TORCH_CHECK(false, "Failed to find function dcmi_get_affinity_cpu_info_by_device_id, "
                    " maybe your hdk version is too low, please upgrade it", PTA_ERROR(ErrCode::NOT_FOUND));
    }
    return func(card_id, device_id, affinity_cpu, length);
}

int DcmiGetDeviceIdInCard(int card_id, int *device_id_max, int *mcu_id, int *cpu_id)
{
    using dcmiGetDeviceIdInCardFunc = int(*)(int, int *, int *, int *);
    static dcmiGetDeviceIdInCardFunc func = nullptr;
    func = (dcmiGetDeviceIdInCardFunc)GET_FUNC(dcmi_get_device_id_in_card);
    if (func == nullptr) {
        TORCH_CHECK(false, "Failed to find function dcmi_get_device_id_in_card, "
                    " maybe your hdk version is too low, please upgrade it", PTA_ERROR(ErrCode::NOT_FOUND))
    }
    return func(card_id, device_id_max, mcu_id, cpu_id);
}

}

}