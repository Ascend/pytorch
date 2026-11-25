#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "graph/types.h"

namespace fx_converter {

// One IO slot occupied in a simplifiedKey (dtype enum int + format enum int).
struct BinaryIoSpec {
  ge::DataType dtype = ge::DT_FLOAT;
  ge::Format format = ge::FORMAT_ND;
};

struct BinarySelectRequest {
  const char* opp_type = nullptr;
  std::vector<BinaryIoSpec> inputs;
  std::vector<BinaryIoSpec> outputs;
  // Prefer high_performance (p=1) then default (p=0); both d=0/1 are tried.
  bool prefer_high_performance = true;
  // Optional attr suffix after the dtype/format key (e.g. TopKV2 "1,,1" for sorted/largest).
  // When set, prefix match prefers built_key+"/"+attr_key_suffix over the shortest hit.
  std::string attr_key_suffix;
  // GE dynamic packed-x convention (Pack/ConcatD): the single representative input slot
  // is emitted as "dtype,format,N" with N = actual packed input count. -1 = fixed arity.
  int32_t packed_input_count = -1;
};

// Selected CANN AICore binary kernel (.o + companion .json).
struct SelectedBinaryKernel {
  std::string opp_type;
  std::string simplified_key;
  std::string bin_abs_path;
  std::string json_abs_path;
  std::string json_text;
  std::vector<char> bin_bytes;
  // Prefer supportInfo.compileInfo dump; fall back to full json text.
  std::string compile_info_json;
  std::string kernel_name;
  std::string magic;
  // Companion JSON coreType: "VectorCore" / "MIX" / "AiCore" / ...
  std::string core_type;
  // Companion JSON taskRation "c:v" (e.g. TopKV2 Mix Sort "0:1") for MixType selection.
  std::string task_ration;
  int32_t intercore_sync = 0;
  // Taken from the companion JSON while it is parsed. 0 is rejected.
  uint32_t op_para_size = 0;
  // compileInfo._pattern. Empty when the object has no pattern (OpImpl tiling).
  std::string compile_info_pattern;
  // compileInfo._var_attrs names. These scalars sit in the tiling struct; a launch
  // with any of them unset faults, so pack requires each name on the op desc.
  std::vector<std::string> var_attr_names;
  std::string bin_file_name; // without suffix
  // Number of kernel entry functions in the .o (STT_FUNC symbols named
  // "<kernel_name>_<tilingKey>"). A single-function bin launches with function-lookup key 0
  // because an OpImpl tiling may emit a schedule id no function is registered under; a
  // multi-function bin needs the tiling-emitted key to pick the right variant.
  uint32_t function_count = 0;
};

// Resolve OPP tbe/kernel root from ASCEND_* env / toolkit defaults.
std::string ResolveOppTbeKernelRoot();

// Per-IO dtypeMode from binary_info_config params (GE GenerateSimpleKey).
// true  = "bit": encode by GetSizeByDataType (float32→int32 code, float16→int16, ...).
// false = "normal": encode ge::DataType enum int directly.
struct BinaryDtypeModes {
  std::vector<bool> input_bit;
  std::vector<bool> output_bit;
};

// Build SIMPLE_MODE simplifiedKey:
//   OpType/d={0|1},p={impl}/{dtype},{format}/...
// Optional bit-mode vectors align with GE DTYPE_MODE_BIT (Gather / dtypeMode=bit).
// packed_input_count >= 0 appends ",N" to the first input slot (GE dynamic packed-x).
std::string BuildSimplifiedKey(
    const char* opp_type,
    int deterministic,
    int impl_mode,
    const std::vector<BinaryIoSpec>& inputs,
    const std::vector<BinaryIoSpec>& outputs,
    const BinaryDtypeModes* dtype_modes = nullptr,
    int32_t packed_input_count = -1);

// Match binary_info_config.json under kernel/config/<Short_SoC_version tolower>/*/ then load .o+.json.
bool SelectBinaryKernel(const BinarySelectRequest& req, SelectedBinaryKernel& out, std::string* err);

} // namespace fx_converter
