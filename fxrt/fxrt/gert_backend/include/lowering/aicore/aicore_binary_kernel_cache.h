#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

#include "lowering/aicore/aicore_binary_kernel_selector.h"

namespace fx_converter {

// Process-lifetime Path C SelectBinaryKernel cache (config JSON, companion .json/.o, selected
// kernels). Thread-safe. Presence of FX_CONVERTER_DISABLE_KERNEL_CACHE disables lookup/store;
// file reads still use the bulk reader.
class AiCoreBinaryKernelCache {
 public:
  static AiCoreBinaryKernelCache& Instance();
  static bool Enabled();

  // Load+parse binary_info_config.json once per path; copy the opp_type subtree into op_cfg.
  // Returns false only on invalid arguments. *found is false when the file is missing, unparsable,
  // or does not contain opp_type (caller should skip that category, same as uncached miss).
  bool LookupOpCfg(const std::string& cfg_path, const char* opp_type, nlohmann::json* op_cfg, bool* found);

  std::string ReadFileText(const std::string& path, std::string* err);
  std::vector<char> ReadFileBytes(const std::string& path, std::string* err);

  bool LookupSelected(const BinarySelectRequest& req, SelectedBinaryKernel* out);
  void StoreSelected(const BinarySelectRequest& req, const SelectedBinaryKernel& sel);

  AiCoreBinaryKernelCache(const AiCoreBinaryKernelCache&) = delete;
  AiCoreBinaryKernelCache& operator=(const AiCoreBinaryKernelCache&) = delete;

 private:
  AiCoreBinaryKernelCache() = default;

  bool LookupConfigLocked(const std::string& cfg_path, const char* opp_type, nlohmann::json* op_cfg, bool* found);

  std::mutex mu_;
  std::unordered_map<std::string, nlohmann::json> configs_;
  std::unordered_map<std::string, std::string> texts_;
  std::unordered_map<std::string, std::vector<char>> bins_;
  std::unordered_map<std::string, SelectedBinaryKernel> selected_;
};

} // namespace fx_converter
