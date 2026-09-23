#include <optional>
#include <vector>

#include "ops/ascend/aclnn/aclnn_moe_distribute_combine_v2.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/opapi_lib_loader.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
using OptionalTensor = std::optional<ir::TensorPtr>;

inline std::optional<ir::TensorPtr> GetOptionalTensor(const ir::Value* value) {
  return value->IsTensor() ? std::optional(value->ToTensor()) : std::nullopt;
}

inline bool HasV3OnlyParam(const std::vector<const ir::Value*>& input) {
  return input[9]->IsTensor() || input[28]->ToInt() != 0 || input[29]->ToInt() != 0 || input[30]->ToInt() != 0;
}

inline bool HasV4OnlyParam(const std::vector<const ir::Value*>& input) {
  return input[14]->IsTensor();
}
} // namespace

constexpr size_t kExpandXIdx = 0;
constexpr size_t kExpertIdsIdx = 1;
constexpr size_t kAssistInfoForCombineIdx = 2;
constexpr size_t kEpSendCountsIdx = 3;
constexpr size_t kExpertScalesIdx = 4;
constexpr size_t kTpSendCountsIdx = 5;
constexpr size_t kXActiveMaskIdx = 6;
constexpr size_t kExpandScalesIdx = 7;
constexpr size_t kSharedExpertXIdx = 8;
constexpr size_t kElasticInfoIdx = 9;
constexpr size_t kOriXIdx = 10;
constexpr size_t kConstExpertAlpha1Idx = 11;
constexpr size_t kConstExpertAlpha2Idx = 12;
constexpr size_t kConstExpertVIdx = 13;
constexpr size_t kPerformanceInfoIdx = 14;
constexpr size_t kGroupEpIdx = 15;
constexpr size_t kEpWorldSizeIdx = 16;
constexpr size_t kEpRankIdIdx = 17;
constexpr size_t kMoeExpertNumIdx = 18;
constexpr size_t kGroupTpIdx = 19;
constexpr size_t kTpWorldSizeIdx = 20;
constexpr size_t kTpRankIdIdx = 21;
constexpr size_t kExpertShardTypeIdx = 22;
constexpr size_t kSharedExpertNumIdx = 23;
constexpr size_t kSharedExpertRankNumIdx = 24;
constexpr size_t kGlobalBsIdx = 25;
constexpr size_t kCommQuantModeIdx = 26;
constexpr size_t kCommAlgIdx = 27;
constexpr size_t kZeroExpertNumIdx = 28;
constexpr size_t kCopyExpertNumIdx = 29;
constexpr size_t kConstExpertNumIdx = 30;

AclnnMoeDistributeCombineV2::AclnnMoeDistributeCombineV2() {
  if (GET_ACLNN_OP_FUNC(std::string("aclnnMoeDistributeCombineV4")) != nullptr) {
    executor_v4_ = std::make_unique<AclnnExecutor>("aclnnMoeDistributeCombineV4");
  }
  if (GET_ACLNN_OP_FUNC(std::string("aclnnMoeDistributeCombineV3")) != nullptr) {
    executor_v3_ = std::make_unique<AclnnExecutor>("aclnnMoeDistributeCombineV3");
  }
  executor_v2_ = std::make_unique<AclnnExecutor>("aclnnMoeDistributeCombineV2");
}

OpsErrorCode AclnnMoeDistributeCombineV2::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  use_v4_ = executor_v4_ != nullptr;
  use_v3_ = !use_v4_ && executor_v3_ != nullptr;
  if (!use_v4_ && HasV4OnlyParam(input)) {
    RT_GLOG(ERROR) << "aclnnMoeDistributeCombineV4 is unavailable, but V4-only parameters are provided.";
    return INVALID_PARAM;
  }
  if (!use_v4_ && !use_v3_ && HasV3OnlyParam(input)) {
    RT_GLOG(ERROR) << "aclnnMoeDistributeCombineV3 is unavailable, but V3-only parameters are provided.";
    return INVALID_PARAM;
  }
  active_executor_ = use_v4_ ? executor_v4_.get() : (use_v3_ ? executor_v3_.get() : executor_v2_.get());

  auto global_bs_real = input[kGlobalBsIdx]->ToInt();
  if (global_bs_real == 0) {
    global_bs_real = input[kExpertIdsIdx]->ToTensor()->Shape()[0] * input[kEpWorldSizeIdx]->ToInt();
  }

  if (use_v4_) {
    active_executor_->GetWorkspaceSize(
        static_cast<uint64_t*>(workspaceSize),
        input[kExpandXIdx]->ToTensor(),
        input[kExpertIdsIdx]->ToTensor(),
        input[kAssistInfoForCombineIdx]->ToTensor(),
        input[kEpSendCountsIdx]->ToTensor(),
        input[kExpertScalesIdx]->ToTensor(),
        GetOptionalTensor(input[kTpSendCountsIdx]),
        GetOptionalTensor(input[kXActiveMaskIdx]),
        OptionalTensor{},
        OptionalTensor{},
        OptionalTensor{},
        GetOptionalTensor(input[kExpandScalesIdx]),
        GetOptionalTensor(input[kSharedExpertXIdx]),
        GetOptionalTensor(input[kElasticInfoIdx]),
        GetOptionalTensor(input[kOriXIdx]),
        GetOptionalTensor(input[kConstExpertAlpha1Idx]),
        GetOptionalTensor(input[kConstExpertAlpha2Idx]),
        GetOptionalTensor(input[kConstExpertVIdx]),
        GetOptionalTensor(input[kPerformanceInfoIdx]),
        input[kGroupEpIdx]->ToString(),
        input[kEpWorldSizeIdx]->ToInt(),
        input[kEpRankIdIdx]->ToInt(),
        input[kMoeExpertNumIdx]->ToInt(),
        input[kGroupTpIdx]->ToString(),
        input[kTpWorldSizeIdx]->ToInt(),
        input[kTpRankIdIdx]->ToInt(),
        input[kExpertShardTypeIdx]->ToInt(),
        input[kSharedExpertNumIdx]->ToInt(),
        input[kSharedExpertRankNumIdx]->ToInt(),
        global_bs_real,
        0,
        input[kCommQuantModeIdx]->ToInt(),
        0,
        input[kCommAlgIdx]->ToString(),
        input[kZeroExpertNumIdx]->ToInt(),
        input[kCopyExpertNumIdx]->ToInt(),
        input[kConstExpertNumIdx]->ToInt(),
        output->ToTensor());
    return SUCCESS;
  }

  if (use_v3_) {
    active_executor_->GetWorkspaceSize(
        static_cast<uint64_t*>(workspaceSize),
        input[kExpandXIdx]->ToTensor(),
        input[kExpertIdsIdx]->ToTensor(),
        input[kAssistInfoForCombineIdx]->ToTensor(),
        input[kEpSendCountsIdx]->ToTensor(),
        input[kExpertScalesIdx]->ToTensor(),
        GetOptionalTensor(input[kTpSendCountsIdx]),
        GetOptionalTensor(input[kXActiveMaskIdx]),
        OptionalTensor{},
        OptionalTensor{},
        OptionalTensor{},
        GetOptionalTensor(input[kExpandScalesIdx]),
        GetOptionalTensor(input[kSharedExpertXIdx]),
        GetOptionalTensor(input[kElasticInfoIdx]),
        GetOptionalTensor(input[kOriXIdx]),
        GetOptionalTensor(input[kConstExpertAlpha1Idx]),
        GetOptionalTensor(input[kConstExpertAlpha2Idx]),
        GetOptionalTensor(input[kConstExpertVIdx]),
        input[kGroupEpIdx]->ToString(),
        input[kEpWorldSizeIdx]->ToInt(),
        input[kEpRankIdIdx]->ToInt(),
        input[kMoeExpertNumIdx]->ToInt(),
        input[kGroupTpIdx]->ToString(),
        input[kTpWorldSizeIdx]->ToInt(),
        input[kTpRankIdIdx]->ToInt(),
        input[kExpertShardTypeIdx]->ToInt(),
        input[kSharedExpertNumIdx]->ToInt(),
        input[kSharedExpertRankNumIdx]->ToInt(),
        global_bs_real,
        0,
        input[kCommQuantModeIdx]->ToInt(),
        0,
        input[kCommAlgIdx]->ToString(),
        input[kZeroExpertNumIdx]->ToInt(),
        input[kCopyExpertNumIdx]->ToInt(),
        input[kConstExpertNumIdx]->ToInt(),
        output->ToTensor());
    return SUCCESS;
  }

  active_executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[kExpandXIdx]->ToTensor(),
      input[kExpertIdsIdx]->ToTensor(),
      input[kAssistInfoForCombineIdx]->ToTensor(),
      input[kEpSendCountsIdx]->ToTensor(),
      input[kExpertScalesIdx]->ToTensor(),
      GetOptionalTensor(input[kTpSendCountsIdx]),
      GetOptionalTensor(input[kXActiveMaskIdx]),
      OptionalTensor{},
      OptionalTensor{},
      OptionalTensor{},
      GetOptionalTensor(input[kExpandScalesIdx]),
      GetOptionalTensor(input[kSharedExpertXIdx]),
      input[kGroupEpIdx]->ToString(),
      input[kEpWorldSizeIdx]->ToInt(),
      input[kEpRankIdIdx]->ToInt(),
      input[kMoeExpertNumIdx]->ToInt(),
      input[kGroupTpIdx]->ToString(),
      input[kTpWorldSizeIdx]->ToInt(),
      input[kTpRankIdIdx]->ToInt(),
      input[kExpertShardTypeIdx]->ToInt(),
      input[kSharedExpertNumIdx]->ToInt(),
      input[kSharedExpertRankNumIdx]->ToInt(),
      global_bs_real,
      0,
      input[kCommQuantModeIdx]->ToInt(),
      0,
      input[kCommAlgIdx]->ToString(),
      output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnMoeDistributeCombineV2::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto global_bs_real = input[kGlobalBsIdx]->ToInt();
  if (global_bs_real == 0) {
    global_bs_real = input[kExpertIdsIdx]->ToTensor()->Shape()[0] * input[kEpWorldSizeIdx]->ToInt();
  }

  if (use_v4_) {
    active_executor_->Launch(
        workspace,
        workspaceSize,
        stream,
        input[kExpandXIdx]->ToTensor(),
        input[kExpertIdsIdx]->ToTensor(),
        input[kAssistInfoForCombineIdx]->ToTensor(),
        input[kEpSendCountsIdx]->ToTensor(),
        input[kExpertScalesIdx]->ToTensor(),
        GetOptionalTensor(input[kTpSendCountsIdx]),
        GetOptionalTensor(input[kXActiveMaskIdx]),
        OptionalTensor{},
        OptionalTensor{},
        OptionalTensor{},
        GetOptionalTensor(input[kExpandScalesIdx]),
        GetOptionalTensor(input[kSharedExpertXIdx]),
        GetOptionalTensor(input[kElasticInfoIdx]),
        GetOptionalTensor(input[kOriXIdx]),
        GetOptionalTensor(input[kConstExpertAlpha1Idx]),
        GetOptionalTensor(input[kConstExpertAlpha2Idx]),
        GetOptionalTensor(input[kConstExpertVIdx]),
        GetOptionalTensor(input[kPerformanceInfoIdx]),
        input[kGroupEpIdx]->ToString(),
        input[kEpWorldSizeIdx]->ToInt(),
        input[kEpRankIdIdx]->ToInt(),
        input[kMoeExpertNumIdx]->ToInt(),
        input[kGroupTpIdx]->ToString(),
        input[kTpWorldSizeIdx]->ToInt(),
        input[kTpRankIdIdx]->ToInt(),
        input[kExpertShardTypeIdx]->ToInt(),
        input[kSharedExpertNumIdx]->ToInt(),
        input[kSharedExpertRankNumIdx]->ToInt(),
        global_bs_real,
        0,
        input[kCommQuantModeIdx]->ToInt(),
        0,
        input[kCommAlgIdx]->ToString(),
        input[kZeroExpertNumIdx]->ToInt(),
        input[kCopyExpertNumIdx]->ToInt(),
        input[kConstExpertNumIdx]->ToInt(),
        output->ToTensor());
    return SUCCESS;
  }

  if (use_v3_) {
    active_executor_->Launch(
        workspace,
        workspaceSize,
        stream,
        input[kExpandXIdx]->ToTensor(),
        input[kExpertIdsIdx]->ToTensor(),
        input[kAssistInfoForCombineIdx]->ToTensor(),
        input[kEpSendCountsIdx]->ToTensor(),
        input[kExpertScalesIdx]->ToTensor(),
        GetOptionalTensor(input[kTpSendCountsIdx]),
        GetOptionalTensor(input[kXActiveMaskIdx]),
        OptionalTensor{},
        OptionalTensor{},
        OptionalTensor{},
        GetOptionalTensor(input[kExpandScalesIdx]),
        GetOptionalTensor(input[kSharedExpertXIdx]),
        GetOptionalTensor(input[kElasticInfoIdx]),
        GetOptionalTensor(input[kOriXIdx]),
        GetOptionalTensor(input[kConstExpertAlpha1Idx]),
        GetOptionalTensor(input[kConstExpertAlpha2Idx]),
        GetOptionalTensor(input[kConstExpertVIdx]),
        input[kGroupEpIdx]->ToString(),
        input[kEpWorldSizeIdx]->ToInt(),
        input[kEpRankIdIdx]->ToInt(),
        input[kMoeExpertNumIdx]->ToInt(),
        input[kGroupTpIdx]->ToString(),
        input[kTpWorldSizeIdx]->ToInt(),
        input[kTpRankIdIdx]->ToInt(),
        input[kExpertShardTypeIdx]->ToInt(),
        input[kSharedExpertNumIdx]->ToInt(),
        input[kSharedExpertRankNumIdx]->ToInt(),
        global_bs_real,
        0,
        input[kCommQuantModeIdx]->ToInt(),
        0,
        input[kCommAlgIdx]->ToString(),
        input[kZeroExpertNumIdx]->ToInt(),
        input[kCopyExpertNumIdx]->ToInt(),
        input[kConstExpertNumIdx]->ToInt(),
        output->ToTensor());
    return SUCCESS;
  }

  active_executor_->Launch(
      workspace,
      workspaceSize,
      stream,
      input[kExpandXIdx]->ToTensor(),
      input[kExpertIdsIdx]->ToTensor(),
      input[kAssistInfoForCombineIdx]->ToTensor(),
      input[kEpSendCountsIdx]->ToTensor(),
      input[kExpertScalesIdx]->ToTensor(),
      GetOptionalTensor(input[kTpSendCountsIdx]),
      GetOptionalTensor(input[kXActiveMaskIdx]),
      OptionalTensor{},
      OptionalTensor{},
      OptionalTensor{},
      GetOptionalTensor(input[kExpandScalesIdx]),
      GetOptionalTensor(input[kSharedExpertXIdx]),
      input[kGroupEpIdx]->ToString(),
      input[kEpWorldSizeIdx]->ToInt(),
      input[kEpRankIdIdx]->ToInt(),
      input[kMoeExpertNumIdx]->ToInt(),
      input[kGroupTpIdx]->ToString(),
      input[kTpWorldSizeIdx]->ToInt(),
      input[kTpRankIdIdx]->ToInt(),
      input[kExpertShardTypeIdx]->ToInt(),
      input[kSharedExpertNumIdx]->ToInt(),
      input[kSharedExpertRankNumIdx]->ToInt(),
      global_bs_real,
      0,
      input[kCommQuantModeIdx]->ToInt(),
      0,
      input[kCommAlgIdx]->ToString(),
      output->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(moe_distribute_combine_v2, AclnnMoeDistributeCombineV2, Ascend);
} // namespace ops
} // namespace fxrt
