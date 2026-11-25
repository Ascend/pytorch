#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "graph/types.h"
#include "lowering/aicore/aicore_binary_kernel_selector.h"
#include "register/op_catalog.h"

namespace fx_converter {

struct AiCoreOpWork;

// Sanity ceiling for one op's input/output slot count (fixed rows and variadic packed-x
// alike; the prebuilt packed-x DSL binaries ship one .o per input count up to 32). This is
// only a corruption guard. Adding an op with more outputs (e.g. 4) needs no change here.
constexpr size_t kMaxAiCoreIoNum = 32;

// Convert-time artifacts for GE-isomorphic AiCore RT wiring (n-ary ElemWise/Broadcast).
// Kept alive via BuiltExecuteGraph::binary_bindata_blobs (and related vectors).
struct AiCoreBinArtifacts {
  // Contiguous BinData header + .o bytes for SinkNodeBinWithHandle Const.
  std::shared_ptr<std::vector<uint8_t>> bindata_blob;
  uint32_t binary_magic = 0;
  // COMPILE_INFO_JSON text including trailing '\0' for TilingParse Const.
  std::shared_ptr<std::string> compile_info_cstr;
  // Contiguous ComputeNodeDesc (+ compiled_args) for AllocLaunchArg input 0.
  std::shared_ptr<std::vector<uint8_t>> compute_node_desc;
  // Contiguous ArgsInfosDesc (+ ArgsInfo[]) for AllocLaunchArg input 1.
  std::shared_ptr<std::vector<uint8_t>> args_info_desc;
  // Hash / kernel_bin_id string including '\0' (may be empty string "").
  std::shared_ptr<std::string> bin_key_cstr;
  // Debug node_info string including '\0'.
  std::shared_ptr<std::string> node_info_cstr;
  // COMPILE_INFO_KEY c-string including '\0'.
  std::shared_ptr<std::string> compile_info_key_cstr;
  // Companion JSON opParaSize. 0 is unset; BuildAiCoreBinArtifacts fails when the
  // field is missing or non-positive instead of guessing a tiling buffer.
  uint32_t op_para_size = 0;
  std::string kernel_name;
  std::string magic;
  // From SelectedBinaryKernel::function_count. A single-function bin launches with the
  // function-lookup key 0 (the OpImpl tiling may emit a schedule id no function is
  // registered under); a multi-function bin uses the tiling-emitted key. 0 means "unknown,
  // use the tiling-emitted key" (the safe default).
  uint32_t function_count = 0;
  // From companion JSON coreType (VectorCore / MIX / ...). Drives AppendCoreType + Mix launch.
  std::string core_type;
  // Companion JSON taskRation / intercoreSync (Mix Sort TopKV2: "0:1" / 1).
  std::string task_ration;
  int32_t intercore_sync = 0;
};

bool BuildAiCoreBinArtifacts(const SelectedBinaryKernel& sel, AiCoreOpWork& work, std::string* err);

// PadV3 BOOL tensors stay BOOL on the graph; CANN bins only ship the INT8 ABI.
ge::DataType BinaryAbiDtype(const OpCatalogEntry& entry, ge::DataType dtype);

} // namespace fx_converter
