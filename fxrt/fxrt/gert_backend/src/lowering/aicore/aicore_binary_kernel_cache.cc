#include "lowering/aicore/aicore_binary_kernel_cache.h"

#include <cstdlib>

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fx_converter {
namespace {

void AppendIoSpecs(std::ostringstream* oss, const std::vector<BinaryIoSpec>& specs) {
  *oss << specs.size() << '\n';
  for (const auto& spec : specs) {
    *oss << static_cast<int>(spec.dtype) << ',' << static_cast<int>(spec.format) << '\n';
  }
}

std::string MakeSelectedKey(const BinarySelectRequest& req) {
  std::ostringstream oss;
  oss << (req.opp_type != nullptr ? req.opp_type : "") << '\n';
  oss << (req.prefer_high_performance ? '1' : '0') << '\n';
  oss << req.attr_key_suffix << '\n';
  // A variadic packed-x request carries one representative spec for N inputs, and each N
  // selects a different .o (ConcatD/d,p/dtype,format,N/...). Without the count in the
  // key, an N=2 selection would be replayed for an N=3 node and the kernel writes zeros.
  oss << req.packed_input_count << '\n';
  AppendIoSpecs(&oss, req.inputs);
  AppendIoSpecs(&oss, req.outputs);
  return oss.str();
}

bool ReadFileToBytes(const std::string& path, std::vector<char>* out, std::string* err) {
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if (!ifs) {
    if (err != nullptr) {
      *err = "failed to open " + path;
    }
    return false;
  }
  ifs.seekg(0, std::ios::end);
  const std::streamoff nbytes = ifs.tellg();
  if (nbytes < 0) {
    if (err != nullptr) {
      *err = "failed to stat " + path;
    }
    return false;
  }
  ifs.seekg(0, std::ios::beg);
  out->assign(static_cast<size_t>(nbytes), '\0');
  if (nbytes > 0 && !ifs.read(out->data(), nbytes)) {
    if (err != nullptr) {
      *err = "failed to read " + path;
    }
    out->clear();
    return false;
  }
  return true;
}

bool LoadParseConfig(const std::string& path, nlohmann::json* cfg) {
  std::vector<char> buf;
  if (!ReadFileToBytes(path, &buf, nullptr) || buf.empty()) {
    return false;
  }
  try {
    *cfg = nlohmann::json::parse(buf.begin(), buf.end());
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool ExtractOpCfg(const nlohmann::json& cfg, const char* opp_type, nlohmann::json* op_cfg, bool* found) {
  if (found != nullptr) {
    *found = false;
  }
  if (!cfg.contains(opp_type)) {
    return true;
  }
  *op_cfg = cfg[opp_type];
  if (found != nullptr) {
    *found = true;
  }
  return true;
}

} // namespace

AiCoreBinaryKernelCache& AiCoreBinaryKernelCache::Instance() {
  static AiCoreBinaryKernelCache* cache = new AiCoreBinaryKernelCache();
  return *cache;
}

bool AiCoreBinaryKernelCache::Enabled() {
  return true;
}

bool AiCoreBinaryKernelCache::LookupConfigLocked(
    const std::string& cfg_path,
    const char* opp_type,
    nlohmann::json* op_cfg,
    bool* found) {
  const auto it = configs_.find(cfg_path);
  if (it == configs_.end()) {
    return false;
  }
  (void)ExtractOpCfg(it->second, opp_type, op_cfg, found);
  return true;
}

bool AiCoreBinaryKernelCache::LookupOpCfg(
    const std::string& cfg_path,
    const char* opp_type,
    nlohmann::json* op_cfg,
    bool* found) {
  if (op_cfg == nullptr || opp_type == nullptr || opp_type[0] == '\0') {
    return false;
  }
  if (found != nullptr) {
    *found = false;
  }
  if (Enabled()) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (LookupConfigLocked(cfg_path, opp_type, op_cfg, found)) {
        return true;
      }
    }
    nlohmann::json parsed;
    if (!LoadParseConfig(cfg_path, &parsed)) {
      return true;
    }
    std::lock_guard<std::mutex> lock(mu_);
    const auto inserted = configs_.emplace(cfg_path, std::move(parsed));
    return ExtractOpCfg(inserted.first->second, opp_type, op_cfg, found);
  }
  nlohmann::json parsed;
  if (!LoadParseConfig(cfg_path, &parsed)) {
    return true;
  }
  return ExtractOpCfg(parsed, opp_type, op_cfg, found);
}

std::string AiCoreBinaryKernelCache::ReadFileText(const std::string& path, std::string* err) {
  if (Enabled()) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = texts_.find(path);
    if (it != texts_.end()) {
      return it->second;
    }
  }
  std::vector<char> buf;
  if (!ReadFileToBytes(path, &buf, err)) {
    return {};
  }
  std::string text(buf.begin(), buf.end());
  if (Enabled()) {
    std::lock_guard<std::mutex> lock(mu_);
    texts_.emplace(path, text);
  }
  return text;
}

std::vector<char> AiCoreBinaryKernelCache::ReadFileBytes(const std::string& path, std::string* err) {
  if (Enabled()) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = bins_.find(path);
    if (it != bins_.end()) {
      return it->second;
    }
  }
  std::vector<char> buf;
  if (!ReadFileToBytes(path, &buf, err)) {
    return {};
  }
  if (Enabled()) {
    std::lock_guard<std::mutex> lock(mu_);
    bins_.emplace(path, buf);
  }
  return buf;
}

bool AiCoreBinaryKernelCache::LookupSelected(const BinarySelectRequest& req, SelectedBinaryKernel* out) {
  if (out == nullptr || !Enabled()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = selected_.find(MakeSelectedKey(req));
  if (it == selected_.end()) {
    return false;
  }
  *out = it->second;
  return true;
}

void AiCoreBinaryKernelCache::StoreSelected(const BinarySelectRequest& req, const SelectedBinaryKernel& sel) {
  if (!Enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mu_);
  selected_.emplace(MakeSelectedKey(req), sel);
}

} // namespace fx_converter
