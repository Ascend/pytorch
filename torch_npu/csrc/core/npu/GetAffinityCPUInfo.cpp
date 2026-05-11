#include <c10/core/Device.h>
#include <climits>
#include <iterator>
#include <unordered_map>

#include <torch_npu/csrc/core/npu/NPUAffinityController.h>
#include <torch_npu/csrc/core/npu/NPUException.h>
#include <torch_npu/csrc/core/npu/interface/DcmiInterface.h>

namespace {
constexpr int NPU_OK = 0;
using c10_npu::CoreId;
using c10_npu::CoreIdList;
using c10_npu::isAllDigits;

std::unordered_map<int, CoreIdList> CardIdAffinityCPU;

void DcmiInit() {
  static bool initialized = false;
  if (!initialized) {
    int ret = c10_npu::dcmi::DcmiInit();
    TORCH_CHECK(
        ret == NPU_OK,
        "Failed to init dcmi. Error code: ",
        ret,
        PTA_ERROR(ErrCode::ACL));
    initialized = true;
  }
}

std::string GetAffinityCPUBaseInfo(int card_id) {
  int device_id = 0;
  int device_id_max = 0;
  int mcu_id = 0;
  int cpu_id = 0;
  int ret = c10_npu::dcmi::DcmiGetDeviceIdInCard(
      card_id, &device_id_max, &mcu_id, &cpu_id);
  if (ret != NPU_OK) {
    TORCH_NPU_WARN_ONCE(
        "dcmi get device id in card is not supported. "
        "The npu_affine configuration of CPU_AFFINITY_CONF will be disabled.");
    return "";
  }
  device_id = std::max(0, device_id_max - 1);
  char affinity_cpu[TOPO_INFO_MAX_LENTH] = {0};
  int length = 0;
  ret = c10_npu::dcmi::DcmiGetAffinityCpuInfoByDeviceId(
      card_id, device_id, affinity_cpu, &length);
  if (ret == NPU_OK) {
    return affinity_cpu;
  }
  TORCH_NPU_WARN_ONCE(
      "dcmi get affinity cpu info by device id is not supported. "
      "The npu_affine configuration of CPU_AFFINITY_CONF will be disabled.");
  return "";
}

CoreIdList parseAffinityCores(const std::string cpuString) {
  CoreIdList cores;
  TORCH_CHECK(
      !cpuString.empty(),
      "Affinity cpu string is empty",
      PTA_ERROR(ErrCode::VALUE));
  std::stringstream ss_value(cpuString);
  std::string range;
  while (std::getline(ss_value, range, ',')) {
    size_t dashPos = range.find('-');
    if (dashPos != std::string::npos) {
      std::string startStr = range.substr(0, dashPos);
      std::string endStr = range.substr(dashPos + 1);
      if (isAllDigits(startStr) && isAllDigits(endStr)) {
        CoreId start = static_cast<CoreId>(std::stoi(startStr));
        CoreId end = static_cast<CoreId>(std::stoi(endStr));
        for (CoreId id = start; id <= end; ++id) {
          cores.insert(id);
        }
      }
    }
  }
  return cores;
}

void GetExclusiveAffinityCPU() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  initialized = true;
  DcmiInit();

  int device_count = 0;
  int card_id_list[16];
  int list_len = 16;
  TORCH_CHECK(
      c10_npu::dcmi::DcmiGetCardNumList(
          &device_count, card_id_list, list_len) == NPU_OK,
      "Dcmi get card num failed.");
  std::unordered_map<std::string, std::vector<int>> rangeAndDevs;
  for (int i = 0; i < device_count; i++) {
    std::string affinity_range = GetAffinityCPUBaseInfo(i);
    if (affinity_range.empty()) {
      TORCH_NPU_WARN_ONCE(
          "Get affinity cpu info by device id is not supported, "
          "The npu_affine configuration of CPU_AFFINITY_CONF will be disabled.");
      return;
    }
    rangeAndDevs[affinity_range].push_back(i);
    ASCEND_LOGD(
        "Device_id: %d, affinity_range: %s.", i, affinity_range.c_str());
  }
  for (auto& [affinity_range, dev_list] : rangeAndDevs) {
    CoreIdList cores = parseAffinityCores(affinity_range);
    int per_dev_cores = cores.size() / dev_list.size();
    if (per_dev_cores == 0) {
      ASCEND_LOGD("Insufficient cores for device allocation.");
      for (int i = 0; i < dev_list.size(); ++i) {
        CardIdAffinityCPU[dev_list[i]].insert(cores.begin(), cores.end());
      }
      continue;
    }
    for (int i = 0; i < dev_list.size(); ++i) {
      auto& dev_i = CardIdAffinityCPU[dev_list[i]];
      auto it = cores.begin();
      std::advance(it, i * per_dev_cores);
      auto end_it = cores.begin();
      if (i == dev_list.size() - 1) {
        end_it = cores.end();
      } else {
        std::advance(end_it, (i + 1) * per_dev_cores);
      }
      dev_i.insert(it, end_it);
    }
  }
}
} // namespace

namespace c10_npu {
CoreIdList GetAffinityCores(int card_id) {
  GetExclusiveAffinityCPU();
  if (CardIdAffinityCPU.empty()) {
    return CoreIdList{};
  }
  auto it = CardIdAffinityCPU.find(card_id);
  if (it != CardIdAffinityCPU.end()) {
    return it->second;
  }
  TORCH_CHECK(
      false,
      "Can't get affinity cores for card_id ",
      std::to_string(card_id),
      PTA_ERROR(ErrCode::VALUE));
}
} // namespace c10_npu
