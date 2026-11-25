#pragma once

/**
 * Layout-compatible POD / enums for wiring libgert AiCore RT kernels without
 * pulling heavy GE internal headers (sink_node_bin.h / rt_kernel_launch_args_ex.h).
 * Must stay binary-compatible with GE runtime v2 definitions.
 */

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

#include "exe_graph/runtime/tiling_context.h"

namespace fx_converter {
namespace aicore_rt {

struct BinData {
  uint32_t magic = 0;
  uint32_t version = 0;
  const void* data = nullptr;
  uint64_t length = 0;
  uint64_t placeholder = 0;
};
static_assert(std::is_standard_layout<BinData>::value, "BinData must be standard layout");

struct DynDesc {
  bool is_group_first{false};
  uint32_t io_group_id{0};
  uint32_t io_index{0};
  uint32_t dyn_num{0};
  uint32_t group_size{0};
  uint32_t group_offset{0};
  uint32_t ptr_offset{0};
};

struct ArgsInfo {
  enum class ArgsInfoType { INPUT = 0, OUTPUT = 1, TYPE_END };
  enum class ArgsInfoFormat { DIRECT_ADDR = 0, FOLDED_DESC_ADDR = 1, FORMAT_END };
  void Init(ArgsInfoType cur_arg_type, ArgsInfoFormat cur_arg_fmt, int32_t start_idx, uint32_t cur_arg_size) {
    arg_type = cur_arg_type;
    arg_format = cur_arg_fmt;
    start_index = start_idx;
    arg_size = cur_arg_size;
  }
  ArgsInfoType arg_type = ArgsInfoType::INPUT;
  ArgsInfoFormat arg_format = ArgsInfoFormat::DIRECT_ADDR;
  int32_t start_index = 0;
  uint32_t arg_size = 0;
  bool folded_first{false};
  bool be_folded{false};
  DynDesc dyn_desc{};
  size_t data_size{0U};
};

struct ArgsInfosDesc {
  void Init(size_t input_args_info_num, size_t output_args_info_num, size_t input_valid_num, size_t output_valid_num) {
    input_args_info_num_ = input_args_info_num;
    output_args_info_num_ = output_args_info_num;
    input_valid_num_ = input_valid_num;
    output_valid_num_ = output_valid_num;
    args_infos_size_ = (input_args_info_num + output_args_info_num) * sizeof(ArgsInfo);
  }
  void* MutableArgsInfoBase() {
    return args_infos_;
  }

  size_t input_args_info_num_ = 0;
  size_t output_args_info_num_ = 0;
  size_t input_valid_num_ = 0;
  size_t output_valid_num_ = 0;
  size_t args_infos_size_ = 0;
  uint8_t args_infos_[8]{};
};
static_assert(std::is_standard_layout<ArgsInfosDesc>::value, "ArgsInfosDesc must be standard layout");

struct ComputeNodeDesc {
  size_t input_num = 0;
  size_t output_num = 0;
  size_t workspace_cap = 0;
  size_t max_tiling_data = 0;
  bool need_shape_buffer = false;
  bool need_overflow = false;
  size_t compiled_args_size = 0;
  uint8_t compiled_args[8]{};
};
static_assert(std::is_standard_layout<ComputeNodeDesc>::value, "ComputeNodeDesc must be standard layout");

enum class AllocLaunchArgOutputs : size_t {
  kHolder = 0,
  kRtArg = 1,
  kRtArgArgsBase = 2,
  kTilingDataBase = 3,
  kNum = 4
};

enum class TilingExOutputIndex : size_t { kRtArg = gert::TilingContext::kOutputNum, kNum };

// Must match gert::kernel::CoreTypeIndex (runtime/v2/.../platform.h).
enum class CoreTypeIndex : int32_t {
  kAiCore = 0,
  kVectorCore = 1,
  kMix = 2,
  kMixAic = 3,
  kMixAiv = 4,
  kMixAiCore = 5,
  kMixAiVector = 6,
  kOtherType = 7
};

// Companion JSON coreType. FFTS-sync detection and Mix launch both use this table.
inline CoreTypeIndex CoreTypeIndexFromJson(const std::string& core_type) {
  if (core_type == "MIX" || core_type == "Mix") {
    return CoreTypeIndex::kMix;
  }
  if (core_type == "AiCore" || core_type == "AIC") {
    return CoreTypeIndex::kAiCore;
  }
  if (core_type == "MIX_AIC" || core_type == "MIX_AICORE") {
    return CoreTypeIndex::kMixAiCore;
  }
  if (core_type == "MIX_AIV" || core_type == "MIX_VECTOR_CORE") {
    return CoreTypeIndex::kMixAiVector;
  }
  return CoreTypeIndex::kVectorCore;
}

inline bool IsMixLaunchCoreType(CoreTypeIndex idx) {
  return idx == CoreTypeIndex::kMix || idx == CoreTypeIndex::kMixAic || idx == CoreTypeIndex::kMixAiv ||
      idx == CoreTypeIndex::kMixAiCore || idx == CoreTypeIndex::kMixAiVector;
}

enum class AppendCoreTypeToPlatformInputIndex : int32_t {
  kPlatform = 0,
  kCoreType = 1,
  kOpAiCoreNum = 2,
  kOpVectorCoreNum = 3,
  kCoreNumInfos = 4,
  kNum = 5
};

enum class InputCommon : int32_t {
  kStream = 0,
  kBinHandle,
  kBlockDim,
  kWorkspaceAddr,
  kShapeBufferAddr,
  kCfg,
  kIoNum,
  kScheduleMode,
  kDfxArgs,
  kRtArg,
  kLocalMemSize,
  kNum
};

enum class WithHandle : int32_t { kTilingKey = static_cast<int32_t>(InputCommon::kNum), kNodeInfo, kIoAddrs, kNum };

// Must match gert::kernel::MixType / MixCoreArgs (ai_core_launch_kernel.h).
// Appended after IO addrs+shapes for LaunchMixKernelWithHandle (GE ProcMixVectorCoreTask).
enum class MixType : int32_t { kMixAiCore = 0, kMixVectorCore = 1 };

struct MixCoreArgs {
  MixType mix_type = MixType::kMixAiCore;
  uint32_t all_core_num = 0;
  uint32_t vec_core_num = 0;
  uint32_t ai_core_num = 0;
};
static_assert(std::is_standard_layout<MixCoreArgs>::value, "MixCoreArgs must be standard layout");

struct DfxExeArg {
  bool need_assert{false};
  bool need_print{false};
  int64_t buffer_size{0};
};

} // namespace aicore_rt
} // namespace fx_converter
