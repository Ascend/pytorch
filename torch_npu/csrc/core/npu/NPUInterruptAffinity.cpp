#include "torch_npu/csrc/core/npu/NPUInterruptAffinity.h"

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <dirent.h>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "torch_npu/csrc/core/npu/GetAffinityCPUInfo.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/interface/DcmiInterface.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"

namespace c10_npu {

namespace {

// Paths and files used for IRQ affinity binding on NPU devices:
//
//   /proc/irq/<irq_num>/smp_affinity
//     - Each IRQ has a subdirectory under /proc/irq/.
//     - Writing a hex CPU mask to the smp_affinity file pins that IRQ
//       to the specified CPU core(s).
//
//   /sys/bus/pci/devices/<domain:bus:device.func>/msi_irqs/
//     - Each PCIe device is enumerated under /sys/bus/pci/devices/ by
//       its BDF address (e.g. "0000:c1:00.0").
//     - The msi_irqs/ directory lists the MSI IRQ numbers assigned to
//       that device as regular files named by the IRQ number.
//
// These two mechanisms together let us discover which IRQs belong to
// a given NPU device and then redirect them to the desired CPU cores.

constexpr const char* kIrqAffinityPath = "/proc/irq/";
constexpr const char* kSmpAffinityFile = "/smp_affinity";
constexpr const char* kPciSysfsPath = "/sys/bus/pci/devices/";
constexpr const char* kMsiIrqsDir = "/msi_irqs";
bool kNeedRestartIrqbalance = false;

bool writeSmpAffinity(int physical_id, int irq_num, const std::string& cpu_mask) {
  std::string filepath =
      std::string(kIrqAffinityPath) + std::to_string(irq_num) + kSmpAffinityFile;
  std::ofstream file(filepath);
  if (!file.is_open()) {
    ASCEND_LOGW("Physical device-%d: failed to open %s for writing.", physical_id, filepath.c_str());
    return false;
  }
  file << cpu_mask;
  file.close();
  if (file.fail()) {
    ASCEND_LOGW("Physical device-%d: failed to write to %s.", physical_id, filepath.c_str());
    return false;
  }
  ASCEND_LOGI("Physical device-%d: smp_affinity for irq %d set to %s.", physical_id, irq_num, cpu_mask.c_str());
  return true;
}

// Convert a CPU core number to hex affinity mask string.
// CPU 0  -> "00000001"
// CPU 31 -> "80000000"
// CPU 32 -> "00000001,00000000"
std::string cpuToMask(unsigned int cpu) {
  unsigned int group = cpu / 32;
  unsigned int bit = cpu % 32;
  unsigned int value = 1u << bit;
  char buf[512];
  int ret = std::snprintf(buf, sizeof(buf), "%08x", value);
  if (ret < 0 || static_cast<size_t>(ret) >= sizeof(buf)) {
    ASCEND_LOGW("cpuToMask: snprintf failed, ret=%d", ret);
    return "";
  }
  std::string mask(buf);
  for (unsigned int i = 0; i < group; i++) {
    mask += ",00000000";
  }
  return mask;
}

// Get the PCIe bus address for an NPU device via the DCMI interface.
// Returns empty string on failure.
std::string getPciAddress(int physical_id) {
  c10_npu::dcmi::DcmiInit();
  c10_npu::dcmi::DcmiPcieInfo pcie_info = {};
  static const auto soc = c10_npu::GetSocVersion();
  int card_id = 0, die_id = 0;
  if (soc >= c10_npu::SocVersion::Ascend910_9391 && soc < c10_npu::SocVersion::Ascend950) {
    // On A3 series, 2 dies per card, physical_id = card_id * 2 + die_id.
    card_id = physical_id / 2;
    die_id = physical_id % 2;
  } else if (soc >= c10_npu::SocVersion::Ascend910B1 && soc < c10_npu::SocVersion::Ascend310B1) {
    // On A2 series, 1 die per card.
    card_id = physical_id;
  } else {
    TORCH_CHECK(false, "Unsupported SOC version %d.", static_cast<int>(soc));
  }
  int acl_ret = c10_npu::dcmi::DcmiGetDevicePcieInfoV2(card_id, die_id, &pcie_info);
  if (acl_ret != 0) {
    ASCEND_LOGW("Physical device-%d: failed to get PCIe info, ret=%d", physical_id, acl_ret);
    return "";
  }
  ASCEND_LOGD("Physical device-%d: PCIe fields: vendor=0x%x subvendor=0x%x "
              "deviceid=0x%x subdeviceid=0x%x domain=%d bus=0x%x dev=0x%x func=0x%x",
              physical_id,
              pcie_info.venderid, pcie_info.subvenderid,
              pcie_info.deviceid, pcie_info.subdeviceid,
              pcie_info.domain,
              pcie_info.bdf_busid, pcie_info.bdf_deviceid, pcie_info.bdf_funcid);

  // Format as "domain:bus:device.func", e.g. "0000:c1:00.0"
  // %x produces lowercase hex, matching sysfs path conventions
  char buf[32];
  int ret = std::snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%01x",
                        pcie_info.domain,
                        pcie_info.bdf_busid,
                        pcie_info.bdf_deviceid,
                        pcie_info.bdf_funcid);
  if (ret < 0 || static_cast<size_t>(ret) >= sizeof(buf)) {
    ASCEND_LOGW("Physical device-%d: snprintf for PCIe address failed, ret=%d", physical_id, ret);
    return "";
  }
  ASCEND_LOGI("Physical device-%d: PCIe address formatted as '%s'", physical_id, buf);
  return std::string(buf);
}

// Collect all numeric IRQ entries from the device's msi_irqs sysfs directory.
std::set<int> collectMsiIrqs(int physical_id, const std::string& pci_addr) {
  std::set<int> irqs = {};
  std::string dir_path = std::string(kPciSysfsPath) + pci_addr + kMsiIrqsDir;
  DIR* dir = ::opendir(dir_path.c_str());
  if (!dir) {
    ASCEND_LOGW("Physical device-%d: failed to open %s for device %s.", physical_id, dir_path.c_str(), pci_addr.c_str());
    return irqs;
  }
  struct dirent* entry;
  while ((entry = ::readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      char* end = nullptr;
      long val = std::strtol(entry->d_name, &end, 10);
      if (end != entry->d_name && *end == '\0') {
        irqs.insert(static_cast<int>(val));
      }
    }
  }
  ::closedir(dir);
  return irqs;
}

// Scan /proc/interrupts for all sq_send_trigger_irq entries.
// Uses a function-local static cache so the scan runs only once per process.
const std::vector<int>& getSqSendTriggerIrqs(int physical_id) {
  // Use vector (not set) because this list is only traversed sequentially, never
  // looked up; vector's contiguous memory is more cache-friendly for iteration.
  static const std::vector<int> candidates = [physical_id]() -> std::vector<int> {
    std::vector<int> result;
    std::ifstream proc_interrupts("/proc/interrupts");
    if (!proc_interrupts.is_open()) {
      ASCEND_LOGW("Physical device-%d: failed to open /proc/interrupts.", physical_id);
      return result;
    }
    std::string line;
    std::regex sq_regex("sq_send_trigger_irq");
    while (std::getline(proc_interrupts, line)) {
      if (!std::regex_search(line, sq_regex)) {
        continue;
      }
      auto colon_pos = line.find(':');
      if (colon_pos == std::string::npos) {
        continue;
      }
      std::string irq_str = line.substr(0, colon_pos);
      irq_str.erase(0, irq_str.find_first_not_of(" \t"));
      char* end = nullptr;
      long val = std::strtol(irq_str.c_str(), &end, 10);
      if (end == irq_str.c_str() || *end != '\0') {
        continue;
      }
      result.push_back(static_cast<int>(val));
    }
    ASCEND_LOGD("Physical device-%d: found %zu sq_send_trigger_irq in /proc/interrupts.",
                physical_id, result.size());
    return result;
  }();
  return candidates;
}

// Find the sq_send_trigger_irq belonging to a specific device by taking the
// intersection of the device's MSI IRQ set and the global sq candidates list.
// Returns the IRQ number on success, or -1 if not found.
int findDeviceSqIrq(const std::set<int>& msi_irqs,
                    const std::vector<int>& sq_candidates) {
  for (int candidate : sq_candidates) {
    if (msi_irqs.count(candidate) > 0) {
      return candidate;
    }
  }
  return -1;
}

// Write smp_affinity for both sq_irq and cq_irq (= sq_irq + 1),
// then log the result.  Returns true on success.
bool writeDeviceIrqAffinity(int physical_id,
                            const std::string& pci_addr,
                            int sq_irq,
                            unsigned int sq_cpu,
                            unsigned int cq_cpu) {
  int cq_irq = sq_irq + 1;
  std::string sq_mask = cpuToMask(sq_cpu);
  std::string cq_mask = cpuToMask(cq_cpu);

  if (!writeSmpAffinity(physical_id, sq_irq, sq_mask)) {
    ASCEND_LOGW("Physical device-%d: Failed to write smp_affinity for sq_irq %d",
                physical_id, sq_irq);
    return false;
  }
  if (!writeSmpAffinity(physical_id, cq_irq, cq_mask)) {
    ASCEND_LOGW("Physical device-%d: Failed to write smp_affinity for cq_irq %d",
                physical_id, cq_irq);
    return false;
  }

  ASCEND_LOGI(
      "Physical device-%d (PCI %s): sq_send_trigger_irq IRQ=%d -> CPU%u, "
      "cq_update_irq IRQ=%d -> CPU%u",
      physical_id, pci_addr.c_str(), sq_irq, sq_cpu, cq_irq, cq_cpu);
  return true;
}

// Wraps ::system() to distinguish system-level failures (fork, OOM, etc.)
// from command execution failures. Returns the raw ::system() value:
//   -1  → system() itself failed (errno logged)
//    0  → command executed successfully
//   >0  → command executed but exited with non-zero status
int SystemCommand(const char* cmd) {
  if (cmd == nullptr) {
    ASCEND_LOGW("SystemCommand: cmd is nullptr.");
    return -1;
  }
  int ret = ::system(cmd);
  if (ret == -1) {
      ASCEND_LOGW("system() call failed, errno=%d, cmd=[%s]", errno, cmd);
  } else if (ret != 0) {
    ASCEND_LOGW("system execute %s command failed, ret=%d", cmd, ret);
  }
  return ret;
}
} // namespace

bool bindIrqAffinity(int device_id, const CoreIdList& irq_cores)
{
  if (irq_cores.empty()) {
    ASCEND_LOGW("Device-%d: irq_cores is empty.", device_id);
    return false;
  }
  if (irq_cores.size() < 2) {
    ASCEND_LOGW(
        "Device-%d: irq_cores size %zu is insufficient, need at least 2 cores.",
        device_id, irq_cores.size());
    return false;
  }

  // Take the first two cores from the ordered set
  auto it = irq_cores.begin();
  unsigned int sq_cpu = *it;
  ++it;
  unsigned int cq_cpu = *it;

  // Step 1: Get PCIe address for this NPU device
  int32_t logic_id = 0;
  int acl_ret = c10_npu::acl::AclrtGetLogicDevIdByUserDevId(device_id, &logic_id);
  if (acl_ret != ACL_SUCCESS) {
    ASCEND_LOGW("Failed to get logic device id for user device id %d, ret = %d", device_id, acl_ret);
    return false;
  }
  ASCEND_LOGD("Device-%d: logic id = %d", device_id, logic_id);
  // In non-container scenarios, the physical device ID is the same as the logical device ID.
  int32_t physical_id = logic_id;
  std::string pci_addr = getPciAddress(physical_id);
  if (pci_addr.empty()) {
    ASCEND_LOGW("Physical device-%d: Cannot find PCIe address, skipping IRQ binding.", physical_id);
    return false;
  }
  ASCEND_LOGD("Physical device-%d: PCIe address = %s", physical_id, pci_addr.c_str());

  // Step 2: Read the device's MSI IRQ list from sysfs
  std::set<int> msi_irqs = collectMsiIrqs(physical_id, pci_addr);
  ASCEND_LOGD("Physical device-%d: MSI IRQs count = %zu", physical_id, msi_irqs.size());
  if (msi_irqs.empty()) {
    ASCEND_LOGW(
        "Physical device-%d (PCI %s): msi_irqs folder is empty or not found, skipping IRQ binding.",
        physical_id, pci_addr.c_str());
    return false;
  }

  // Step 3: Get all sq_send_trigger_irq entries (cached after first call)
  const std::vector<int>& sq_candidates = getSqSendTriggerIrqs(physical_id);
  if (sq_candidates.empty()) {
    ASCEND_LOGW("Physical device-%d: No sq_send_trigger_irq found in /proc/interrupts, skipping IRQ binding.", physical_id);
    return false;
  }

  // Step 4: Find the sq_irq that belongs to THIS device
  int sq_irq = findDeviceSqIrq(msi_irqs, sq_candidates);
  if (sq_irq < 0) {
    ASCEND_LOGW(
        "Physical device-%d (PCI %s): sq_send_trigger_irq not found in MSI IRQs, skipping IRQ binding.",
        physical_id, pci_addr.c_str());
    return false;
  }

  // Step 5: Write smp_affinity for sq_irq and cq_irq
  return writeDeviceIrqAffinity(physical_id, pci_addr, sq_irq, sq_cpu, cq_cpu);
}

void stopIrqbalance()
{
  // Check if systemctl is available
  int ret = SystemCommand("command -v systemctl > /dev/null 2>&1");
  if (ret != 0) {
    return;
  }
  // Check if irqbalance.service is installed
  ret = SystemCommand("systemctl list-unit-files --quiet irqbalance.service > /dev/null 2>&1");
  if (ret != 0) {
    return;
  }
  // Stop if running
  ret = SystemCommand("systemctl is-active --quiet irqbalance");
  if (ret == 0) {
    ASCEND_LOGI("irqbalance service is running, attempting to stop it.");
    int stop_ret = SystemCommand("systemctl stop irqbalance");
    if (stop_ret == 0) {
      ASCEND_LOGW("To bind IRQ affinity, the irqbalance service has been stopped successfully. "
        "It will be restarted when the process exits normally. If the process exits abnormally "
        "(e.g., killed or coredump), the irqbalance service needs to be restarted manually by "
        "the user using 'sudo systemctl restart irqbalance' (requires root privileges).");
      kNeedRestartIrqbalance = true;
    }
  } else if (ret > 0) {
    ASCEND_LOGI("irqbalance service is not running, no need to stop.");
  }
  return;
}

void restartIrqbalance()
{
  if (!kNeedRestartIrqbalance) {
    return;
  }
  // Check if systemctl is available
  // Most mainstream server operating systems released after 2015 use systemd
  // by default and support the systemctl command.
  int ret = SystemCommand("command -v systemctl > /dev/null 2>&1");
  if (ret != 0) {
    return;
  }
  ret = SystemCommand("systemctl restart irqbalance");
  if (ret != 0) {
    return;
  }
  ASCEND_LOGI("irqbalance service restarted.");
  kNeedRestartIrqbalance = false;
}
} // namespace c10_npu
