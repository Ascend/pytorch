#include "lowering/aicore/aicore_bin_artifacts.h"

#include <cstring>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include "acl/acl_rt.h"
#include "common/logger.h"
#include "lowering/aicore/aicore_op_work.h"
#include "lowering/aicore/aicore_rt_layout.h"
#include "platform/platform_info.h"
#include "runtime/rt.h"

#ifndef RT_DEV_BINARY_MAGIC_ELF
#define RT_DEV_BINARY_MAGIC_ELF 0x43554245U
#endif
#ifndef RT_DEV_BINARY_MAGIC_ELF_AIVEC
#define RT_DEV_BINARY_MAGIC_ELF_AIVEC 0x41415246U
#endif
#ifndef RT_DEV_BINARY_MAGIC_ELF_AICUBE
#define RT_DEV_BINARY_MAGIC_ELF_AICUBE 0x41494343U
#endif

namespace fx_converter {
namespace {

constexpr size_t kWorkspaceCap = 16;

// Companion JSON "magic" → rtDevBinary_t.magic (must match CANN runtime/kernel.h).
uint32_t MagicFromString(const std::string &magic) {
  if (magic == "RT_DEV_BINARY_MAGIC_ELF_AICUBE") {
    return RT_DEV_BINARY_MAGIC_ELF_AICUBE;
  }
  if (magic == "RT_DEV_BINARY_MAGIC_ELF") {
    // Mix AIC/AIV ELF (TopKV2). Do not fall through to AIVEC.
    return RT_DEV_BINARY_MAGIC_ELF;
  }
  // Default / empty: VectorCore AIVEC (Mul / SoftmaxV2 / Tile / GatherElementsV2).
  return RT_DEV_BINARY_MAGIC_ELF_AIVEC;
}

size_t AlignUp(size_t v, size_t align) {
  if (align == 0) {
    return v;
  }
  return ((v + align - 1) / align) * align;
}

bool PackBinData(const SelectedBinaryKernel &sel, AiCoreBinArtifacts *out, std::string *err) {
  if (sel.bin_bytes.empty()) {
    if (err != nullptr) {
      *err = "empty binary bytes for key=" + sel.simplified_key;
    }
    return false;
  }
  const size_t total = sizeof(aicore_rt::BinData) + sel.bin_bytes.size();
  auto blob = std::make_shared<std::vector<uint8_t>>(total, 0);
  auto *bin = reinterpret_cast<aicore_rt::BinData *>(blob->data());
  bin->magic = MagicFromString(sel.magic);
  bin->version = 0;
  bin->data = nullptr;
  bin->length = sel.bin_bytes.size();
  // GE kernel overwrites bin->data with &bin->placeholder at runtime.
  std::memcpy(&bin->placeholder, sel.bin_bytes.data(), sel.bin_bytes.size());
  out->bindata_blob = std::move(blob);
  out->binary_magic = MagicFromString(sel.magic);
  out->magic = sel.magic.empty() ? "RT_DEV_BINARY_MAGIC_ELF_AIVEC" : sel.magic;
  return true;
}

bool PackComputeNodeDesc(size_t input_num, size_t output_num, uint32_t op_para_size, AiCoreBinArtifacts *out) {
  using Desc = aicore_rt::ComputeNodeDesc;
  auto buf = std::make_shared<std::vector<uint8_t>>(sizeof(Desc), 0);
  auto *desc = reinterpret_cast<Desc *>(buf->data());
  desc->input_num = input_num;
  desc->output_num = output_num;
  desc->workspace_cap = kWorkspaceCap;
  desc->max_tiling_data = AlignUp(static_cast<size_t>(op_para_size), sizeof(uintptr_t));
  desc->need_shape_buffer = false;
  desc->need_overflow = false;
  desc->compiled_args_size = 0;
  out->compute_node_desc = std::move(buf);
  return true;
}

// GE tbe_task_builder_adapter::InitInput: Mix enhance + FFTS_PLUS prepends
// ArgsInfo(INPUT, DIRECT_ADDR, start_index=0xFFFF). Launch UpdateEachArgsInfo
// then fills that slot via aclrtGetHardwareSyncAddr (intercoreSync Mix Sort).
constexpr int32_t kFftsSyncArgsStartIndex = 0xFFFF;

bool PackArgsInfoDesc(size_t input_num, size_t output_num, bool need_ffts_addr, AiCoreBinArtifacts *out) {
  using ArgsInfo = aicore_rt::ArgsInfo;
  // GE InitIoArgsInfo creates one IoArg per ArgsInfo entry and advances the args
  // slot once; DIRECT_ADDR entries must therefore be one-per-IO with arg_size=1.
  // OUTPUT start_index is relative (GE adds ComputeNodeDesc.input_num).
  const size_t ffts_slots = need_ffts_addr ? 1U : 0U;
  std::vector<ArgsInfo> infos(ffts_slots + input_num + output_num);
  size_t slot = 0;
  if (need_ffts_addr) {
    infos[slot] = {};
    infos[slot].Init(ArgsInfo::ArgsInfoType::INPUT, ArgsInfo::ArgsInfoFormat::DIRECT_ADDR, kFftsSyncArgsStartIndex, 1);
    ++slot;
  }
  for (size_t i = 0; i < input_num; ++i) {
    infos[slot] = {};
    infos[slot].Init(ArgsInfo::ArgsInfoType::INPUT, ArgsInfo::ArgsInfoFormat::DIRECT_ADDR, static_cast<int32_t>(i), 1);
    ++slot;
  }
  for (size_t i = 0; i < output_num; ++i) {
    infos[slot] = {};
    infos[slot].Init(ArgsInfo::ArgsInfoType::OUTPUT, ArgsInfo::ArgsInfoFormat::DIRECT_ADDR, static_cast<int32_t>(i),
                      1);
    ++slot;
  }

  const size_t args_info_size = infos.size() * sizeof(ArgsInfo);
  const size_t total_size = sizeof(aicore_rt::ArgsInfosDesc) + args_info_size;
  auto buf = std::make_shared<std::vector<uint8_t>>(total_size, 0);
  auto *desc = reinterpret_cast<aicore_rt::ArgsInfosDesc *>(buf->data());
  const size_t in_info_num = ffts_slots + input_num;
  desc->Init(/*input_args_info_num*/ in_info_num, /*output_args_info_num*/ output_num, /*valid_in*/ in_info_num,
             /*valid_out*/ output_num);
  std::memcpy(desc->MutableArgsInfoBase(), infos.data(), args_info_size);
  out->args_info_desc = std::move(buf);
  return true;
}

void SetPackErr(std::string *err, const std::string &msg) {
  if (err != nullptr) {
    *err = msg;
  }
}

void CollectAbiDtypes(const AiCoreOpWork &work, std::vector<ge::DataType> &input_dtypes,
                      std::vector<ge::DataType> &output_dtypes) {
  input_dtypes.reserve(work.ins.size());
  std::transform(work.ins.begin(), work.ins.end(), std::back_inserter(input_dtypes),
                 [&work](const TensorRef &in) { return BinaryAbiDtype(work.entry, in.dtype); });
  output_dtypes.reserve(work.dtypes.size());
  std::transform(work.dtypes.begin(), work.dtypes.end(), std::back_inserter(output_dtypes),
                 [&work](ge::DataType dt) { return BinaryAbiDtype(work.entry, dt); });
}

void CompileInfoStrings(const SelectedBinaryKernel &sel, const std::string &tag, std::string &ci, std::string &key) {
  ci = sel.compile_info_json.empty() ? "{}" : sel.compile_info_json;
  const std::string &base = sel.simplified_key.empty() ? sel.bin_file_name : sel.simplified_key;
  key = base + "#" + tag;
}

// Intercore synchronization alone does not imply the FFTS+ args ABI (e.g. Ascend950).
// Match GE InitInput/ParseFftsMode: only ffts-plus platforms prepend the sync address.
bool NeedFftsSyncArgsAddr(const SelectedBinaryKernel &sel, bool *need_ffts_addr, std::string *err) {
  *need_ffts_addr = false;
  if (sel.intercore_sync == 0 ||
      !aicore_rt::IsMixLaunchCoreType(aicore_rt::CoreTypeIndexFromJson(sel.core_type))) {
    return true;
  }
  const char *soc_name = aclrtGetSocName();
  if (soc_name == nullptr || soc_name[0] == '\0') {
    SetPackErr(err, "BuildAiCoreBinArtifacts: cannot query SoC for FFTS+ args ABI");
    return false;
  }
  fe::PlatFormInfos infos;
  fe::OptionalInfos opti;
  auto &ge_platform = fe::PlatformInfoManager::GeInstance();
  (void)ge_platform.InitializePlatformInfo();
  if (ge_platform.GetPlatformInfos(soc_name, infos, opti) != 0U) {
    auto &platform = fe::PlatformInfoManager::Instance();
    (void)platform.InitializePlatformInfo();
    if (platform.GetPlatformInfos(soc_name, infos, opti) != 0U) {
      SetPackErr(err, "BuildAiCoreBinArtifacts: cannot load platform for FFTS+ args ABI: " + std::string(soc_name));
      return false;
    }
  }
  std::string ffts_mode;
  *need_ffts_addr = infos.GetPlatformRes("SoCInfo", "ffts_mode", ffts_mode) && ffts_mode == "ffts-plus";
  return true;
}

bool ValidateBuildAiCoreBinArgs(const char *opp_type, size_t input_num, size_t output_num, std::string *err) {
  if (opp_type == nullptr) {
    SetPackErr(err, "BuildAiCoreBinArtifacts: null opp_type");
    return false;
  }
  // Range guard only. Catalog CSV arity is checked when the binary is selected.
  if (input_num == 0 || input_num > kMaxAiCoreIoNum || output_num == 0 || output_num > kMaxAiCoreIoNum) {
    SetPackErr(err, "BuildAiCoreBinArtifacts: input/output slot count out of sanity range [1,32]");
    return false;
  }
  return true;
}

bool AssignAiCoreBinKernelMeta(const SelectedBinaryKernel &sel, AiCoreBinArtifacts *out, std::string *err) {
  if (sel.op_para_size == 0) {
    if (err != nullptr) {
      *err = "BuildAiCoreBinArtifacts: opParaSize missing for key=" + sel.simplified_key;
    }
    return false;
  }
  out->op_para_size = sel.op_para_size;
  out->kernel_name = sel.kernel_name.empty() ? sel.bin_file_name : sel.kernel_name;
  out->core_type = sel.core_type;
  out->task_ration = sel.task_ration;
  out->intercore_sync = sel.intercore_sync;
  out->function_count = sel.function_count;
  return true;
}

void AssignAiCoreBinStringHandles(const SelectedBinaryKernel &sel, AiCoreOpWork &work) {
  AiCoreBinArtifacts *out = &work.art;
  const char *opp_type = work.bind.opp_type;
  std::string ci;
  std::string key;
  CompileInfoStrings(sel, work.tag, ci, key);
  // Companion JSON including the trailing NUL. FxAiCoreTiling passes it to OpImpl tiling_parse.
  out->compile_info_cstr = std::make_shared<std::string>(ci.c_str(), ci.size() + 1);

  // Unique per node so SinkNodeBinWithHandle does not share one rt handle across
  // multiple AiCore nodes that happen to select the same .o.
  auto key_cstr = std::make_shared<std::string>(key.c_str(), key.size() + 1);
  out->bin_key_cstr = key_cstr;
  out->compile_info_key_cstr = std::move(key_cstr);

  const std::string node_info = std::string(opp_type) + ":" + work.tag;
  out->node_info_cstr = std::make_shared<std::string>(node_info.c_str(), node_info.size() + 1);
}

}  // namespace

ge::DataType BinaryAbiDtype(const OpCatalogEntry &entry, ge::DataType dtype) {
  if (entry.behavior != nullptr && entry.behavior->bool_as_int8 && dtype == ge::DT_BOOL) {
    return ge::DT_INT8;
  }
  return dtype;
}

bool BuildAiCoreBinArtifacts(const SelectedBinaryKernel &sel, AiCoreOpWork &work, std::string *err) {
  std::vector<ge::DataType> input_dtypes;
  std::vector<ge::DataType> output_dtypes;
  CollectAbiDtypes(work, input_dtypes, output_dtypes);
  // GE CalcMixL2InArgsNum: ComputeNodeDesc/ArgsInfo count the padded gen_placeholder slots.
  const size_t packed_in_num = input_dtypes.size() + work.optional.pad;
  if (!ValidateBuildAiCoreBinArgs(work.bind.opp_type, packed_in_num, output_dtypes.size(), err)) {
    return false;
  }
  work.art = AiCoreBinArtifacts{};
  if (!AssignAiCoreBinKernelMeta(sel, &work.art, err)) {
    return false;
  }

  if (!PackBinData(sel, &work.art, err) ||
      !PackComputeNodeDesc(packed_in_num, output_dtypes.size(), work.art.op_para_size, &work.art)) {
    return false;
  }
  bool need_ffts_addr = false;
  if (!NeedFftsSyncArgsAddr(sel, &need_ffts_addr, err)) {
    return false;
  }
  if (need_ffts_addr) {
    RT_GLOG(INFO) << "BuildAiCoreBinArtifacts: prepend FFTS sync ArgsInfo (intercoreSync="
                              << sel.intercore_sync << " coreType=" << sel.core_type << ")";
  }
  if (!PackArgsInfoDesc(packed_in_num, output_dtypes.size(), need_ffts_addr, &work.art)) {
    return false;
  }

  AssignAiCoreBinStringHandles(sel, work);
  return true;
}

}  // namespace fx_converter
