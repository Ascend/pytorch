#include <unordered_map>
#include "torch_npu/csrc/core/npu/interface/DcmiInterface.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUAffinityController.h"

constexpr int NPU_OK = 0;

static int DcmiInit()
{
    int ret = c10_npu::dcmi::DcmiInit();
    if (ret != NPU_OK) {
        TORCH_CHECK(false, "Failed to init dcmi. ", PTA_ERROR(ErrCode::INTERNAL));
    }
    return ret;
}

std::string GetAffinityCPUBaseInfo(int card_id)
{
    int ret = DcmiInit();
    int device_id = 0;
    int device_id_max = 0;
    int mcu_id = 0;
    int cpu_id = 0;
    ret = c10_npu::dcmi::DcmiGetDeviceIdInCard(card_id, &device_id_max, &mcu_id, &cpu_id);
    if (ret != NPU_OK) {
        TORCH_NPU_WARN_ONCE("dcmi_get_device_id_in_card is not supported. "
                            "The npu_affine configuration of CPU_AFFINITY_CONF will be disabled.");
        return "";
    }
    device_id = std::max(0, device_id_max - 1);
    char affinity_cpu[TOPO_INFO_MAX_LENTH] = {0};
    int length = 0;
    ret = c10_npu::dcmi::DcmiGetAffinityCpuInfoByDeviceId(card_id, device_id, affinity_cpu, &length);
    if (ret == NPU_OK) {
        return affinity_cpu;
    }
    TORCH_NPU_WARN_ONCE("dcmi_get_affinity_cpu_info_by_device_id is not supported. "
                        "The npu_affine configuration of CPU_AFFINITY_CONF will be disabled.");
    return "";
}

std::unordered_map<int, c10_npu::CoreIdRange> CardIdAffinityCPU;

c10_npu::CoreIdRange parseAffinityCPU(const std::string cpuString)
{
    size_t pos = cpuString.find("-");
    if (pos != std::string::npos) {
        std::string start = cpuString.substr(0, pos);
        std::string end = cpuString.substr(pos + 1);
        int startNum = stoi(start);
        int endNum = stoi(end);
        if (startNum < endNum) {
            return c10_npu::CoreIdRange{startNum, endNum};
        }
    }
    TORCH_CHECK(false, "affinity cpu " + cpuString + " is error ", PTA_ERROR(ErrCode::VALUE));
}

void GetExclusiveAffinityCPU()
{
    int ret = DcmiInit();
    int device_count = 0;
    int card_id_list[16];
    int list_len = 16;
    ret = c10_npu::dcmi::DcmiGetCardNumList(&device_count, card_id_list, list_len);
    std::unordered_map<std::string, int> SameAffinityCpuNum;
    std::map<int, std::string> CardIdAffinityCpuDefault;
    for (int i = 0; i < device_count; i++) {
        std::string affinity_cpu = GetAffinityCPUBaseInfo(i);
        if (affinity_cpu.empty()) {
            return;
        }
        CardIdAffinityCpuDefault[i] = affinity_cpu;
        auto it = SameAffinityCpuNum.find(affinity_cpu);
        if (it != SameAffinityCpuNum.end()) {
            SameAffinityCpuNum[affinity_cpu] = it->second + 1;
        } else {
            SameAffinityCpuNum[affinity_cpu] = 1;
        }
    }
    std::unordered_map<std::string, int> offsetMap;
    for (const auto& it : CardIdAffinityCpuDefault) {
        int card_id = it.first;
        std::string affinity_cpu = it.second;
        int same_num = 1;
        auto find_same_affinity_cpu = SameAffinityCpuNum.find(affinity_cpu);
        if (find_same_affinity_cpu != SameAffinityCpuNum.end()) {
            same_num = find_same_affinity_cpu->second;
        }
        int offset = 0;
        auto find_offset = offsetMap.find(affinity_cpu);
        if (find_offset != offsetMap.end()) {
            offset = find_offset->second;
        }
        c10_npu::CoreIdRange cpu_range = parseAffinityCPU(affinity_cpu);
        unsigned int length = (cpu_range.end - cpu_range.start + 1) / static_cast<unsigned int>(same_num);
        c10_npu::CoreIdRange exclusiveAffinityCpu = {
            cpu_range.start + static_cast<unsigned int>(offset) * length,
            (cpu_range.start + length - 1) + static_cast<unsigned int>(offset) * length};
        offsetMap[affinity_cpu] = offset + 1;
        CardIdAffinityCPU[card_id] = exclusiveAffinityCpu;
    }
}

c10_npu::CoreIdRange GetAssignAffinityCPU(int card_id)
{
    GetExclusiveAffinityCPU();
    if (CardIdAffinityCPU.empty()) {
        return {0, 0};
    }
    auto it = CardIdAffinityCPU.find(card_id);
    if (it != CardIdAffinityCPU.end()) {
        return it->second;
    }
    TORCH_CHECK(false, "card_id ", std::to_string(card_id), " is invalid.", PTA_ERROR(ErrCode::VALUE));
}
