#include "lowering/aicore/aicore_binary_kernel_selector.h"

#include <cctype>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "acl/acl_rt.h"
#include "common/ascend_paths.h"
#include "common/logger.h"
#include "lowering/aicore/aicore_binary_kernel_cache.h"
#include "nlohmann/json.hpp"

namespace fx_converter {
namespace {

bool FileExists(const std::string &path) {
  std::ifstream ifs(path);
  return ifs.good();
}

std::string ReadFileText(const std::string &path, std::string *err) {
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if (!ifs) {
    if (err != nullptr) {
      *err = "failed to open " + path;
    }
    return {};
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

std::string TrimAscii(std::string s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
    s.pop_back();
  }
  size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) {
    ++i;
  }
  return s.substr(i);
}

std::string ToLowerAscii(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// GE PlatformUtils: Short_SoC_version from platform_config/<aclrtGetSocName>.ini.
std::string ParseIniKey(const std::string &text, const char *key) {
  std::istringstream iss(text);
  std::string line;
  const std::string key_s(key);
  while (std::getline(iss, line)) {
    const auto eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    if (TrimAscii(line.substr(0, eq)) == key_s) {
      return TrimAscii(line.substr(eq + 1));
    }
  }
  return {};
}

// Host toolkit arch for platform_config .ini (not a SoC scan).
const char *HostToolkitArchDir() {
#if defined(__x86_64__)
  return "x86_64-linux";
#else
  return "aarch64-linux";
#endif
}

std::string FindPlatformConfigIni(const std::string &soc_name) {
  // CANN: <toolkit>/<host-arch>/data/platform_config/<Soc>.ini
  // ASCEND_HOME_PATH may already be the arch dir, so also try without it.
  const std::string rel = std::string("/data/platform_config/") + soc_name + ".ini";
  const char *arch = HostToolkitArchDir();
  for (const auto &root : CollectAscendToolkitRoots()) {
    const std::string with_arch = root + "/" + arch + rel;
    if (FileExists(with_arch)) {
      return with_arch;
    }
    const std::string bare = root + rel;
    if (FileExists(bare)) {
      return bare;
    }
  }
  return {};
}

std::vector<std::string> CategoryDirs(const std::string &config_soc_dir) {
  static const char *kCats[] = {"ops_math", "ops_legacy", "ops_nn", "ops_cv", "ops_transformer", "ops_oam"};
  std::vector<std::string> out;
  for (const char *c : kCats) {
    const std::string p = config_soc_dir + "/" + c + "/binary_info_config.json";
    if (FileExists(p)) {
      out.emplace_back(c);
    }
  }
  return out;
}

// Running device only: aclrtGetSocName -> platform.ini Short_SoC_version -> config/<soc>/.
// No cross-chip fallback.
std::string CurrentDeviceSocDir(const std::string &kernel_root) {
  const char *soc_name = aclrtGetSocName();
  if (soc_name == nullptr || soc_name[0] == '\0') {
    return {};
  }
  const std::string ini_path = FindPlatformConfigIni(soc_name);
  if (ini_path.empty()) {
    return {};
  }
  const std::string short_soc = ParseIniKey(ReadFileText(ini_path, nullptr), "Short_SoC_version");
  if (short_soc.empty()) {
    return {};
  }
  const std::string soc = ToLowerAscii(short_soc);
  if (CategoryDirs(kernel_root + "/config/" + soc).empty()) {
    return {};
  }
  return soc;
}

// Minimal ELF64 little-endian reader over the in-memory .o bytes; only the symbol table
// is needed to count the kernel entry functions, so no libelf dependency is introduced.
uint16_t ReadLe16(const char *p) {
  return static_cast<uint16_t>(static_cast<uint8_t>(p[0])) |
         static_cast<uint16_t>(static_cast<uint8_t>(p[1]) << 8);
}

uint32_t ReadLe32(const char *p) {
  return static_cast<uint32_t>(static_cast<uint8_t>(p[0])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[3])) << 24);
}

uint64_t ReadLe64(const char *p) {
  return static_cast<uint64_t>(ReadLe32(p)) | (static_cast<uint64_t>(ReadLe32(p + 4)) << 32);
}

bool IsElf64LittleEndian(const char *data) {
  return data[0] == 0x7f && data[1] == 'E' && data[2] == 'L' && data[3] == 'F' && data[4] == 2 &&
         data[5] == 1;
}

// Count STT_FUNC symbols named "<bin_file_name>_<tilingKey>"; those are the registered kernel
// entry points the launch function lookup indexes by tilingKey. (kernelName in the companion
// JSON is already the full entry name including the key, so the shared prefix to match on is
// the bin file name.) Returns 0 when the symbol table cannot be read (caller then keeps the
// safe default of using the tiling-emitted key).
uint32_t CountKernelFunctions(const std::vector<char> &bin, const std::string &bin_file_name) {
  constexpr size_t kEhdrSize = 64;
  constexpr size_t kShdrSize = 64;
  constexpr uint8_t kSymtabType = 2;   // SHT_SYMTAB
  constexpr uint8_t kFuncType = 2;     // STT_FUNC
  if (bin.size() < kEhdrSize || bin_file_name.empty()) {
    return 0;
  }
  const char *data = bin.data();
  if (!IsElf64LittleEndian(data)) {
    return 0;
  }
  const uint64_t shoff = ReadLe64(data + 40);
  const uint16_t shentsize = ReadLe16(data + 58);
  const uint16_t shnum = ReadLe16(data + 60);
  if (shentsize < kShdrSize || shoff + static_cast<uint64_t>(shentsize) * shnum > bin.size()) {
    return 0;
  }
  const std::string prefix = bin_file_name + "_";
  uint32_t count = 0;
  for (uint16_t s = 0; s < shnum; ++s) {
    const char *sh = data + shoff + static_cast<uint64_t>(s) * shentsize;
    if (ReadLe32(sh + 4) != kSymtabType) {
      continue;
    }
    const uint64_t sym_off = ReadLe64(sh + 24);
    const uint64_t sym_size = ReadLe64(sh + 32);
    const uint32_t str_link = ReadLe32(sh + 40);  // Elf64_Shdr::sh_link, not sh_name at offset 0
    const uint64_t entsize = ReadLe64(sh + 56);
    if (str_link >= shnum || entsize == 0 || sym_off + sym_size > bin.size()) {
      continue;
    }
    const char *str_sh = data + shoff + static_cast<uint64_t>(str_link) * shentsize;
    const uint64_t str_off = ReadLe64(str_sh + 24);
    const uint64_t str_size = ReadLe64(str_sh + 32);
    if (str_off + str_size > bin.size()) {
      continue;
    }
    for (uint64_t off = 0; off + 24 <= sym_size; off += entsize) {
      const char *sym = data + sym_off + off;
      const uint32_t name_off = ReadLe32(sym);
      const uint8_t info = static_cast<uint8_t>(sym[4]);
      if ((info & 0xf) != kFuncType || name_off >= str_size) {
        continue;
      }
      const char *name = data + str_off + name_off;
      if (strncmp(name, prefix.c_str(), prefix.size()) == 0) {
        ++count;
      }
    }
  }
  return count;
}

// First name of a width is the compileInfo rewrite ("float32" for 4 bytes).
// Bit mode uses the GE code on that same row (DT_INT32 for 4 bytes).
struct DtypeWidth {
  const char *name;
  uint8_t bytes;
  ge::DataType bit;
};

constexpr DtypeWidth kDtypeWidths[] = {
    {"int8", 1, ge::DT_INT8},       {"uint8", 1, ge::DT_INT8},   {"bool", 1, ge::DT_INT8},
    {"float16", 2, ge::DT_INT16},   {"int16", 2, ge::DT_INT16},  {"uint16", 2, ge::DT_INT16},
    {"float32", 4, ge::DT_INT32},   {"int32", 4, ge::DT_INT32},  {"uint32", 4, ge::DT_INT32},
    {"int64", 8, ge::DT_INT64},     {"uint64", 8, ge::DT_INT64}, {"double", 8, ge::DT_INT64},
};

size_t DtypeNameBytes(const std::string &name) {
  const auto it = std::find_if(std::begin(kDtypeWidths), std::end(kDtypeWidths),
                               [&name](const DtypeWidth &row) { return name == row.name; });
  return it == std::end(kDtypeWidths) ? 0 : it->bytes;
}

const char *CompileInfoDtypeName(size_t bytes) {
  const auto it = std::find_if(std::begin(kDtypeWidths), std::end(kDtypeWidths),
                               [bytes](const DtypeWidth &row) { return row.bytes == bytes; });
  return it == std::end(kDtypeWidths) ? "" : it->name;
}

ge::DataType BitModeDtype(size_t bytes) {
  const auto it = std::find_if(std::begin(kDtypeWidths), std::end(kDtypeWidths),
                               [bytes](const DtypeWidth &row) { return row.bytes == bytes; });
  return it == std::end(kDtypeWidths) ? ge::DT_UNDEFINED : it->bit;
}

void RewriteCompileInfoDtype(nlohmann::json *ci, size_t input0_bytes) {
  if (input0_bytes == 0 || ci == nullptr || !ci->contains("vars") || !(*ci)["vars"].is_object()) {
    return;
  }
  nlohmann::json &vars = (*ci)["vars"];
  if (!vars.contains("dtype") || !vars["dtype"].is_string()) {
    return;
  }
  const size_t have = DtypeNameBytes(vars["dtype"].get<std::string>());
  const char *fixed = CompileInfoDtypeName(input0_bytes);
  if (have != 0 && fixed[0] != '\0' && have != input0_bytes) {
    vars["dtype"] = fixed;
  }
}

bool ReadPositiveOpParaSize(const nlohmann::json &j, SelectedBinaryKernel *out, std::string *err) {
  auto fail = [&](const char *why) {
    if (err != nullptr) {
      *err = std::string("BuildAiCoreBinArtifacts: opParaSize ") + why + " for key=" + out->simplified_key;
    }
    return false;
  };
  if (!j.contains("opParaSize") || !j["opParaSize"].is_number_integer()) {
    return fail("missing");
  }
  const int value = j["opParaSize"].get<int>();
  if (value <= 0) {
    return fail("must be positive");
  }
  out->op_para_size = static_cast<uint32_t>(value);
  return true;
}

// Dump compileInfo as before. A non-object still dumps and leaves pattern empty.
// vars.dtype is rewritten here when its width differs from input 0, so pack does not parse again.
void CollectVarAttrNames(const nlohmann::json &ci, std::vector<std::string> *names) {
  names->clear();
  if (!ci.is_object() || !ci.contains("_var_attrs") || !ci["_var_attrs"].is_array()) {
    return;
  }
  for (const auto &item : ci["_var_attrs"]) {
    if (item.is_object() && item.contains("name") && item["name"].is_string()) {
      names->push_back(item["name"].get<std::string>());
    }
  }
}

void CaptureCompileInfo(const nlohmann::json &j, size_t input0_bytes, SelectedBinaryKernel *out) {
  const nlohmann::json *src = nullptr;
  if (j.contains("compileInfo")) {
    src = &j["compileInfo"];
  } else if (j.contains("supportInfo") && j["supportInfo"].contains("compileInfo")) {
    src = &j["supportInfo"]["compileInfo"];
  }
  if (src == nullptr) {
    out->compile_info_json = "{}";
    return;
  }
  nlohmann::json ci = *src;
  if (ci.is_object()) {
    if (ci.contains("_pattern") && ci["_pattern"].is_string()) {
      out->compile_info_pattern = ci["_pattern"].get<std::string>();
    }
    RewriteCompileInfoDtype(&ci, input0_bytes);
    CollectVarAttrNames(ci, &out->var_attr_names);
  }
  out->compile_info_json = ci.dump();
}

bool FillFromJsonText(SelectedBinaryKernel *out, size_t input0_bytes, std::string *err) {
  try {
    const nlohmann::json j = nlohmann::json::parse(out->json_text);
    if (j.contains("binFileName") && j["binFileName"].is_string()) {
      out->bin_file_name = j["binFileName"].get<std::string>();
    }
    if (j.contains("kernelName") && j["kernelName"].is_string()) {
      out->kernel_name = j["kernelName"].get<std::string>();
    } else {
      out->kernel_name = out->bin_file_name;
    }
    if (j.contains("magic") && j["magic"].is_string()) {
      out->magic = j["magic"].get<std::string>();
    }
    if (j.contains("coreType") && j["coreType"].is_string()) {
      out->core_type = j["coreType"].get<std::string>();
    }
    if (j.contains("taskRation") && j["taskRation"].is_string()) {
      out->task_ration = j["taskRation"].get<std::string>();
    }
    if (j.contains("intercoreSync") && j["intercoreSync"].is_number_integer()) {
      out->intercore_sync = j["intercoreSync"].get<int>();
    }
    if (!ReadPositiveOpParaSize(j, out, err)) {
      return false;
    }
    // Some binaries ship empty compileInfo; tiling_parse still needs a JSON object.
    CaptureCompileInfo(j, input0_bytes, out);
    return true;
  } catch (const std::exception &e) {
    if (err != nullptr) {
      *err = std::string("parse kernel json failed: ") + e.what();
    }
    return false;
  }
}

void ExtractBinJsonPaths(const nlohmann::json &item, std::string *bin_rel, std::string *json_rel) {
  if (item.contains("binPath") && item["binPath"].is_string()) {
    *bin_rel = item["binPath"].get<std::string>();
  }
  if (item.contains("jsonPath") && item["jsonPath"].is_string()) {
    *json_rel = item["jsonPath"].get<std::string>();
  } else if (item.contains("binInfo") && item["binInfo"].contains("jsonFilePath")) {
    *json_rel = item["binInfo"]["jsonFilePath"].get<std::string>();
  }
}

const char *BinaryListFieldName(const nlohmann::json &op_cfg) {
  if (op_cfg.contains("binaryList")) {
    return "binaryList";
  }
  if (op_cfg.contains("binList")) {
    return "binList";
  }
  return nullptr;
}

// Attr-suffix candidate: built_key is a proper prefix and next char is ',' or '/'.
bool IsBuiltKeyAttrPrefix(const std::string &sk, const std::string &built_key) {
  if (sk.size() <= built_key.size() || sk.compare(0, built_key.size(), built_key) != 0) {
    return false;
  }
  const char next = sk[built_key.size()];
  return next == ',' || next == '/';
}

// Prefer preferred key; otherwise keep the shortest longer key.
bool PreferPrefixCandidate(bool best_empty, bool best_is_preferred, bool is_preferred, size_t best_len,
                           size_t cand_len) {
  if (best_empty) {
    return true;
  }
  if (best_is_preferred && !is_preferred) {
    return false;
  }
  if (!is_preferred && (best_is_preferred || cand_len >= best_len)) {
    return false;
  }
  return true;
}

bool TryUpdatePrefixBest(const nlohmann::json &item, const std::string &sk, bool is_preferred,
                         std::string *best_sk, std::string *best_bin, std::string *best_json,
                         bool *best_is_preferred) {
  if (!PreferPrefixCandidate(best_sk->empty(), *best_is_preferred, is_preferred, best_sk->size(), sk.size())) {
    return false;
  }
  std::string cand_bin;
  std::string cand_json;
  ExtractBinJsonPaths(item, &cand_bin, &cand_json);
  if (cand_bin.empty() || cand_json.empty()) {
    return false;
  }
  *best_sk = sk;
  *best_bin = std::move(cand_bin);
  *best_json = std::move(cand_json);
  *best_is_preferred = is_preferred;
  return true;
}

void ConsiderPrefixSimplifiedKeys(const nlohmann::json &item, const std::string &built_key,
                                  const std::string &preferred, std::string *best_sk, std::string *best_bin,
                                  std::string *best_json, bool *best_is_preferred) {
  if (!item.contains("simplifiedKey") || !item["simplifiedKey"].is_array()) {
    return;
  }
  for (const auto &sk_j : item["simplifiedKey"]) {
    if (!sk_j.is_string()) {
      continue;
    }
    const std::string sk = sk_j.get<std::string>();
    if (!IsBuiltKeyAttrPrefix(sk, built_key)) {
      continue;
    }
    const bool is_preferred = !preferred.empty() && sk == preferred;
    (void)TryUpdatePrefixBest(item, sk, is_preferred, best_sk, best_bin, best_json, best_is_preferred);
  }
}

// Exact simplifiedKey hit.
bool MatchInBinaryListExact(const nlohmann::json &op_cfg, const std::string &key, std::string *bin_rel,
                            std::string *json_rel, std::string *matched_key) {
  const char *list_key = BinaryListFieldName(op_cfg);
  if (list_key == nullptr || !op_cfg[list_key].is_array()) {
    return false;
  }
  for (const auto &item : op_cfg[list_key]) {
    if (!item.contains("simplifiedKey") || !item["simplifiedKey"].is_array()) {
      continue;
    }
    for (const auto &sk : item["simplifiedKey"]) {
      if (!sk.is_string() || sk.get<std::string>() != key) {
        continue;
      }
      ExtractBinJsonPaths(item, bin_rel, json_rel);
      if (!bin_rel->empty() && !json_rel->empty()) {
        if (matched_key != nullptr) {
          *matched_key = key;
        }
        return true;
      }
    }
  }
  return false;
}

// When exact misses: binaryList key has built key as prefix and remainder is an attr suffix
// (next char is ',' or '/'). Prefer preferred_attr_suffix when set; else shortest longer key.
bool MatchInBinaryListPrefix(const nlohmann::json &op_cfg, const std::string &built_key,
                             const std::string &preferred_attr_suffix, std::string *bin_rel, std::string *json_rel,
                             std::string *matched_key) {
  const char *list_key = BinaryListFieldName(op_cfg);
  if (list_key == nullptr || !op_cfg[list_key].is_array()) {
    return false;
  }
  const std::string preferred =
      preferred_attr_suffix.empty() ? std::string() : (built_key + "/" + preferred_attr_suffix);
  std::string best_sk;
  std::string best_bin;
  std::string best_json;
  bool best_is_preferred = false;
  for (const auto &item : op_cfg[list_key]) {
    ConsiderPrefixSimplifiedKeys(item, built_key, preferred, &best_sk, &best_bin, &best_json, &best_is_preferred);
  }
  if (best_sk.empty()) {
    return false;
  }
  *bin_rel = std::move(best_bin);
  *json_rel = std::move(best_json);
  if (matched_key != nullptr) {
    *matched_key = std::move(best_sk);
  }
  return true;
}

bool MatchInBinaryList(const nlohmann::json &op_cfg, const std::string &key, const std::string &preferred_attr_suffix,
                       std::string *bin_rel, std::string *json_rel, std::string *matched_key) {
  // Prefer exact IO key, then IO+attr (TopKV2 sorted/largest), then shortest attr prefix.
  if (MatchInBinaryListExact(op_cfg, key, bin_rel, json_rel, matched_key)) {
    return true;
  }
  if (!preferred_attr_suffix.empty()) {
    const std::string with_attr = key + "/" + preferred_attr_suffix;
    if (MatchInBinaryListExact(op_cfg, with_attr, bin_rel, json_rel, matched_key)) {
      return true;
    }
  }
  bin_rel->clear();
  json_rel->clear();
  return MatchInBinaryListPrefix(op_cfg, key, preferred_attr_suffix, bin_rel, json_rel, matched_key);
}

// binary_info often stores "ascend910b/neg/xxx.o" while files live under
// "ascend910b/ops_legacy/neg/xxx.o". Try both layouts.
std::string ResolveKernelRelPath(const std::string &kernel_root, const std::string &soc, const std::string &category,
                                 const std::string &rel) {
  const std::string direct = kernel_root + "/" + rel;
  if (FileExists(direct)) {
    return direct;
  }
  // Insert category after soc prefix: "<soc>/<rest>" -> "<soc>/<category>/<rest>"
  const std::string prefix = soc + "/";
  if (rel.rfind(prefix, 0) == 0) {
    const std::string with_cat = kernel_root + "/" + soc + "/" + category + "/" + rel.substr(prefix.size());
    if (FileExists(with_cat)) {
      return with_cat;
    }
  }
  return direct;  // caller will fail on open
}

}  // namespace

std::string ResolveOppTbeKernelRoot() {
  const std::string rel = "/opp/built-in/op_impl/ai_core/tbe/kernel";
  for (const auto &root : CollectAscendToolkitRoots()) {
    const std::string cand = root + rel;
    if (FileExists(cand + "/config") || FileExists(cand)) {
      return cand;
    }
  }
  return {};
}

// GE GenerateSimpleKey DTYPE_MODE_BIT: replace dtype by size class (float32→int32, ...).
static int DtypeCodeForSimplifiedKey(ge::DataType dtype, bool bit_mode) {
  if (!bit_mode) {
    return static_cast<int>(dtype);
  }
  const int nbytes = ge::GetSizeByDataType(dtype);
  if (nbytes <= 0) {
    return static_cast<int>(dtype);
  }
  const ge::DataType mapped = BitModeDtype(static_cast<size_t>(nbytes));
  if (mapped == ge::DT_UNDEFINED) {
    return static_cast<int>(dtype);
  }
  return static_cast<int>(mapped);
}

std::string BuildSimplifiedKey(const char *opp_type, int deterministic, int impl_mode,
                               const std::vector<BinaryIoSpec> &inputs, const std::vector<BinaryIoSpec> &outputs,
                               const BinaryDtypeModes *dtype_modes, int32_t packed_input_count) {
  std::ostringstream oss;
  oss << (opp_type != nullptr ? opp_type : "") << "/d=" << deterministic << ",p=" << impl_mode;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const bool bit =
        dtype_modes != nullptr && i < dtype_modes->input_bit.size() && dtype_modes->input_bit[i];
    oss << '/' << DtypeCodeForSimplifiedKey(inputs[i].dtype, bit) << ','
        << static_cast<int>(inputs[i].format);
    if (i == 0 && packed_input_count >= 0) {
      // GE dynamic packed-x: the representative x slot carries the packed input count.
      oss << ',' << packed_input_count;
    }
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    const bool bit =
        dtype_modes != nullptr && i < dtype_modes->output_bit.size() && dtype_modes->output_bit[i];
    oss << '/' << DtypeCodeForSimplifiedKey(outputs[i].dtype, bit) << ','
        << static_cast<int>(outputs[i].format);
  }
  return oss.str();
}

// Read params.inputs/outputs[].dtypeMode == "bit" (GE fusion_binary_info).
static BinaryDtypeModes ParseDtypeModesFromOpCfg(const nlohmann::json &op_cfg, size_t nin, size_t nout) {
  BinaryDtypeModes modes;
  modes.input_bit.assign(nin, false);
  modes.output_bit.assign(nout, false);
  if (!op_cfg.contains("params") || !op_cfg["params"].is_object()) {
    return modes;
  }
  const auto &params = op_cfg["params"];
  auto fill = [](const nlohmann::json &arr, std::vector<bool> *out) {
    if (!arr.is_array() || out == nullptr) {
      return;
    }
    for (size_t i = 0; i < out->size() && i < arr.size(); ++i) {
      // A dynamic input arrives as a one-group nested array (Pack/ConcatD
      // "inputs": [[{x ...}]]); the representative object carries the group dtypeMode.
      const nlohmann::json *slot = &arr[i];
      if (slot->is_array() && !slot->empty()) {
        slot = &(*slot)[0];
      }
      if (!slot->is_object() || !slot->contains("dtypeMode") || !(*slot)["dtypeMode"].is_string()) {
        continue;
      }
      (*out)[i] = ((*slot)["dtypeMode"].get<std::string>() == "bit");
    }
  };
  if (params.contains("inputs")) {
    fill(params["inputs"], &modes.input_bit);
  }
  if (params.contains("outputs")) {
    fill(params["outputs"], &modes.output_bit);
  }
  return modes;
}

enum class BinarySearchOutcome { kFound, kNoMatch, kError };

static BinarySearchOutcome TrySelectBinaryInConfigs(const std::string &kernel_root, const std::string &soc,
                                                    const std::vector<std::pair<int, int>> &dp_order,
                                                    const BinarySelectRequest &req, SelectedBinaryKernel *out,
                                                    std::vector<std::string> *tried_keys, std::string *err) {
  AiCoreBinaryKernelCache &cache = AiCoreBinaryKernelCache::Instance();
  const std::string config_soc = kernel_root + "/config/" + soc;
  for (const auto &cat : CategoryDirs(config_soc)) {
    const std::string cfg_path = config_soc + "/" + cat + "/binary_info_config.json";
    nlohmann::json op_cfg;
    bool found_op = false;
    if (!cache.LookupOpCfg(cfg_path, req.opp_type, &op_cfg, &found_op) || !found_op) {
      continue;
    }
    // GE dtypeMode=bit (e.g. Gather): encode IO by byte-width class in simplifiedKey.
    const BinaryDtypeModes dtype_modes =
        ParseDtypeModesFromOpCfg(op_cfg, req.inputs.size(), req.outputs.size());
    for (const auto &dp : dp_order) {
      const std::string key = BuildSimplifiedKey(req.opp_type, dp.first, dp.second, req.inputs, req.outputs,
                                                 &dtype_modes, req.packed_input_count);
      tried_keys->push_back(key);
      std::string bin_rel;
      std::string json_rel;
      std::string matched_key;
      if (!MatchInBinaryList(op_cfg, key, req.attr_key_suffix, &bin_rel, &json_rel, &matched_key)) {
        continue;
      }
      SelectedBinaryKernel selected;
      selected.opp_type = req.opp_type;
      selected.simplified_key = matched_key.empty() ? key : matched_key;
      selected.bin_abs_path = ResolveKernelRelPath(kernel_root, soc, cat, bin_rel);
      selected.json_abs_path = ResolveKernelRelPath(kernel_root, soc, cat, json_rel);
      selected.json_text = cache.ReadFileText(selected.json_abs_path, err);
      if (selected.json_text.empty()) {
        return BinarySearchOutcome::kError;
      }
      selected.bin_bytes = cache.ReadFileBytes(selected.bin_abs_path, err);
      if (selected.bin_bytes.empty()) {
        return BinarySearchOutcome::kError;
      }
      const int input0_nbytes = req.inputs.empty() ? 0 : ge::GetSizeByDataType(req.inputs.front().dtype);
      const size_t input0_bytes = input0_nbytes > 0 ? static_cast<size_t>(input0_nbytes) : 0;
      if (!FillFromJsonText(&selected, input0_bytes, err)) {
        return BinarySearchOutcome::kError;
      }
      *out = std::move(selected);
      RT_GLOG(INFO) << "SelectBinaryKernel hit opp=" << req.opp_type << " key=" << out->simplified_key
                    << " bin=" << out->bin_abs_path << " cache=miss";
      return BinarySearchOutcome::kFound;
    }
  }
  return BinarySearchOutcome::kNoMatch;
}

bool SelectBinaryKernel(const BinarySelectRequest &req, SelectedBinaryKernel &out, std::string *err) {
  if (req.opp_type == nullptr || req.opp_type[0] == '\0') {
    if (err != nullptr) {
      *err = "SelectBinaryKernel: invalid request";
    }
    return false;
  }
  AiCoreBinaryKernelCache &cache = AiCoreBinaryKernelCache::Instance();
  if (cache.LookupSelected(req, &out)) {
    out.function_count = CountKernelFunctions(out.bin_bytes, out.bin_file_name);
    RT_GLOG(INFO) << "SelectBinaryKernel hit opp=" << req.opp_type << " key=" << out.simplified_key
                  << " bin=" << out.bin_abs_path << " cache=hit";
    return true;
  }
  const std::string kernel_root = ResolveOppTbeKernelRoot();
  if (kernel_root.empty()) {
    if (err != nullptr) {
      *err = "SelectBinaryKernel: OPP tbe/kernel root not found (set ASCEND_HOME_PATH)";
    }
    return false;
  }
  const std::string soc = CurrentDeviceSocDir(kernel_root);
  if (soc.empty()) {
    if (err != nullptr) {
      const char *soc_name = aclrtGetSocName();
      const std::string device = (soc_name != nullptr && soc_name[0] != '\0') ? soc_name : "unknown";
      *err = "SelectBinaryKernel: no binary config for current device soc=" + device + " under " + kernel_root +
             "/config (missing platform Short_SoC_version or config dir)";
    }
    return false;
  }

  std::vector<std::pair<int, int>> dp_order;
  if (req.prefer_high_performance) {
    dp_order = {{0, 1}, {1, 1}, {0, 0}, {1, 0}};
  } else {
    dp_order = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  }

  std::vector<std::string> tried_keys;
  const BinarySearchOutcome outcome =
      TrySelectBinaryInConfigs(kernel_root, soc, dp_order, req, &out, &tried_keys, err);
  if (outcome == BinarySearchOutcome::kFound) {
    out.function_count = CountKernelFunctions(out.bin_bytes, out.bin_file_name);
    cache.StoreSelected(req, out);
    return true;
  }
  if (outcome == BinarySearchOutcome::kError) {
    // Mid-loop read/parse failure: err already holds the callee's message; do not overwrite.
    return false;
  }

  if (err != nullptr) {
    std::ostringstream oss;
    oss << "SelectBinaryKernel: no binary for opp_type=" << req.opp_type << "; tried keys:";
    for (const auto &k : tried_keys) {
      oss << " [" << k << ']';
    }
    *err = oss.str();
  }
  return false;
}

}  // namespace fx_converter
