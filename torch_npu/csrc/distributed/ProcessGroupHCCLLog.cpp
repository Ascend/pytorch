#include "ProcessGroupHCCLLog.hpp"

#include <cstdlib>
#include <exception>
#include <utility>

#include "torch_npu/csrc/distributed/ProcessGroupHCCL.hpp"
#include "torch_npu/csrc/logging/LogContext.h"

namespace c10d_npu::hccl_log {

namespace {

template <typename Enum, size_t N>
const char* findHcclLogName(Enum value, const std::pair<Enum, const char*> (&names)[N], const char* fallback) {
  for (const auto& [key, name] : names) {
    if (key == value) {
      return name;
    }
  }
  return fallback;
}

} // namespace

const std::shared_ptr<npu_logging::Logger> loggerHccl = npu_logging::logging().getLogger("torch_npu.hccl");

const char* reduceOpNameForLog(c10d::ReduceOp::RedOpType op) {
  static constexpr std::pair<c10d::ReduceOp::RedOpType, const char*> names[] = {
      {c10d::ReduceOp::SUM, "SUM"},
      {c10d::ReduceOp::AVG, "AVG"},
      {c10d::ReduceOp::PRODUCT, "PRODUCT"},
      {c10d::ReduceOp::MIN, "MIN"},
      {c10d::ReduceOp::MAX, "MAX"},
      {c10d::ReduceOp::BAND, "BAND"},
      {c10d::ReduceOp::BOR, "BOR"},
      {c10d::ReduceOp::BXOR, "BXOR"},
      {c10d::ReduceOp::PREMUL_SUM, "PREMUL_SUM"},
  };
  return findHcclLogName(op, names, "UNKNOWN");
}

const char* reduceOpNameForLog(HcclReduceOp op) {
  static constexpr std::pair<HcclReduceOp, const char*> names[] = {
      {HCCL_REDUCE_SUM, "SUM"},
      {HCCL_REDUCE_PROD, "PRODUCT"},
      {HCCL_REDUCE_MIN, "MIN"},
      {HCCL_REDUCE_MAX, "MAX"},
  };
  return findHcclLogName(op, names, "UNKNOWN");
}

const char* hcclDataTypeNameForLog(HcclDataType type) {
  static constexpr std::pair<HcclDataType, const char*> names[] = {
      {HCCL_DATA_TYPE_INT8, "int8"},
      {HCCL_DATA_TYPE_INT16, "int16"},
      {HCCL_DATA_TYPE_INT32, "int32"},
      {HCCL_DATA_TYPE_FP16, "fp16"},
      {HCCL_DATA_TYPE_FP32, "fp32"},
      {HCCL_DATA_TYPE_INT64, "int64"},
      {HCCL_DATA_TYPE_UINT64, "uint64"},
      {HCCL_DATA_TYPE_UINT8, "uint8"},
      {HCCL_DATA_TYPE_UINT16, "uint16"},
      {HCCL_DATA_TYPE_UINT32, "uint32"},
      {HCCL_DATA_TYPE_FP64, "fp64"},
      {HCCL_DATA_TYPE_BFP16, "bfp16"},
      {HCCL_DATA_TYPE_FP8E4M3, "fp8e4m3"},
      {HCCL_DATA_TYPE_FP8E5M2, "fp8e5m2"},
      {HCCL_DATA_TYPE_FP8E8M0, "fp8e8m0"},
  };
  return findHcclLogName(type, names, "unknown");
}

long long hcclGlobalRankForLog(const std::vector<uint32_t>& ranks, int64_t pgRank) {
  if (pgRank < 0 || (!ranks.empty() && pgRank >= static_cast<int64_t>(ranks.size()))) {
    return -1;
  }
  return ranks.empty() ? pgRank : ranks[pgRank];
}

std::string hcclRankPrefixForLog(const std::string& groupName, ProcessGroupHCCL& pg) {
  const auto pgRank = pg.getRank();
  const auto globalRank = hcclGlobalRankForLog(pg.getOptions()->global_ranks_in_group, pgRank);
  const auto* rankEnv = std::getenv("RANK");
  char* end = nullptr;
  // TorchNPU Logger already prints the global rank from RANK when present.
  if (rankEnv != nullptr && globalRank >= 0 && std::strtol(rankEnv, &end, 10) == globalRank && end != rankEnv &&
      *end == '\0') {
    return c10::str("[", groupName, " pg_rank=", pgRank, "] ");
  }
  return c10::str("[", groupName, " global_rank=", globalRank, " pg_rank=", pgRank, "] ");
}

std::string hcclCommNameForLog(HcclComm comm) {
  if (comm == nullptr) {
    return "<unavailable>";
  }

  char name[COMM_NAME_MAX_LENGTH] = {};
  try {
    if (at_npu::hccl::HcclGetCommNameFace(comm, name) == HCCL_SUCCESS && name[0] != '\0') {
      name[sizeof(name) - 1] = '\0';
      return name;
    }
  } catch (const std::exception&) {
    // Some CANN versions do not provide the diagnostic name query.
  }
  return "<unavailable>";
}

} // namespace c10d_npu::hccl_log
