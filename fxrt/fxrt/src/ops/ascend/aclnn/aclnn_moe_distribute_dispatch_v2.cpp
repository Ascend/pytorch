#include <optional>
#include <vector>

#include "ops/ascend/aclnn/aclnn_moe_distribute_dispatch_v2.h"
#include "ops/ascend/aclnn/utils/opapi_lib_loader.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
namespace {
using OptionalTensor = std::optional<ir::TensorPtr>;

inline OptionalTensor GetOptionalTensor(const ir::Value* value) {
  return value->IsTensor() ? std::optional(value->ToTensor()) : std::nullopt;
}

inline bool HasV3OnlyParam(const std::vector<const ir::Value*>& input) {
  return input[5]->IsTensor() || input[21]->ToInt() != 0 || input[22]->ToInt() != 0 || input[23]->ToInt() != 0;
}

inline bool HasV4OnlyParam(const std::vector<const ir::Value*>& input) {
  return input[6]->IsTensor();
}
} // namespace

constexpr size_t kXIdx = 0;
constexpr size_t kExpertIdsIdx = 1;
constexpr size_t kScalesIdx = 2;
constexpr size_t kXActiveMaskIdx = 3;
constexpr size_t kExpertScalesIdx = 4;
constexpr size_t kElasticInfoIdx = 5;
constexpr size_t kPerformanceInfoIdx = 6;
constexpr size_t kGroupEpIdx = 7;
constexpr size_t kEpWorldSizeIdx = 8;
constexpr size_t kEpRankIdIdx = 9;
constexpr size_t kMoeExpertNumIdx = 10;
constexpr size_t kGroupTpIdx = 11;
constexpr size_t kTpWorldSizeIdx = 12;
constexpr size_t kTpRankIdIdx = 13;
constexpr size_t kExpertShardTypeIdx = 14;
constexpr size_t kSharedExpertNumIdx = 15;
constexpr size_t kSharedExpertRankNumIdx = 16;
constexpr size_t kQuantModeIdx = 17;
constexpr size_t kGlobalBsIdx = 18;
constexpr size_t kExpertTokenNumsTypeIdx = 19;
constexpr size_t kCommAlgIdx = 20;
constexpr size_t kZeroExpertNumIdx = 21;
constexpr size_t kCopyExpertNumIdx = 22;
constexpr size_t kConstExpertNumIdx = 23;

constexpr size_t kExpandXOutIdx = 0;
constexpr size_t kDynamicScalesOutIdx = 1;
constexpr size_t kAssistInfoForCombineOutIdx = 2;
constexpr size_t kExpertTokenNumsOutIdx = 3;
constexpr size_t kEpRecvCountsOutIdx = 4;
constexpr size_t kTpRecvCountsOutIdx = 5;
constexpr size_t kExpandScalesOutIdx = 6;

AclnnMoeDistributeDispatchV2::AclnnMoeDistributeDispatchV2() {
  if (GET_ACLNN_OP_FUNC(std::string("aclnnMoeDistributeDispatchV4")) != nullptr) {
    executor_v4_ = std::make_unique<AclnnExecutor>("aclnnMoeDistributeDispatchV4");
  }
  if (GET_ACLNN_OP_FUNC(std::string("aclnnMoeDistributeDispatchV3")) != nullptr) {
    executor_v3_ = std::make_unique<AclnnExecutor>("aclnnMoeDistributeDispatchV3");
  }
  executor_v2_ = std::make_unique<AclnnExecutor>("aclnnMoeDistributeDispatchV2");
}

OpsErrorCode AclnnMoeDistributeDispatchV2::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  auto& output_tuple = output->ToTuple();

  use_v4_ = executor_v4_ != nullptr;
  use_v3_ = !use_v4_ && executor_v3_ != nullptr;
  if (!use_v4_ && HasV4OnlyParam(input)) {
    RT_GLOG(ERROR) << "aclnnMoeDistributeDispatchV4 is unavailable, but V4-only parameters are provided.";
    return INVALID_PARAM;
  }
  if (!use_v4_ && !use_v3_ && HasV3OnlyParam(input)) {
    RT_GLOG(ERROR) << "aclnnMoeDistributeDispatchV3 is unavailable, but V3-only parameters are provided.";
    return INVALID_PARAM;
  }
  active_executor_ = use_v4_ ? executor_v4_.get() : (use_v3_ ? executor_v3_.get() : executor_v2_.get());

  auto global_bs_real = input[kGlobalBsIdx]->ToInt();
  if (global_bs_real == 0) {
    global_bs_real = input[kXIdx]->ToTensor()->Shape()[0] * input[kEpWorldSizeIdx]->ToInt();
  }

  if (use_v4_) {
    active_executor_->GetWorkspaceSize(
        static_cast<uint64_t*>(workspaceSize),
        input[kXIdx]->ToTensor(),
        input[kExpertIdsIdx]->ToTensor(),
        GetOptionalTensor(input[kScalesIdx]),
        GetOptionalTensor(input[kXActiveMaskIdx]),
        GetOptionalTensor(input[kExpertScalesIdx]),
        GetOptionalTensor(input[kElasticInfoIdx]),
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
        input[kQuantModeIdx]->ToInt(),
        global_bs_real,
        input[kExpertTokenNumsTypeIdx]->ToInt(),
        input[kCommAlgIdx]->ToString(),
        input[kZeroExpertNumIdx]->ToInt(),
        input[kCopyExpertNumIdx]->ToInt(),
        input[kConstExpertNumIdx]->ToInt(),
        (*output_tuple)[kExpandXOutIdx]->ToTensor(),
        (*output_tuple)[kDynamicScalesOutIdx]->ToTensor(),
        (*output_tuple)[kAssistInfoForCombineOutIdx]->ToTensor(),
        (*output_tuple)[kExpertTokenNumsOutIdx]->ToTensor(),
        (*output_tuple)[kEpRecvCountsOutIdx]->ToTensor(),
        (*output_tuple)[kTpRecvCountsOutIdx]->ToTensor(),
        (*output_tuple)[kExpandScalesOutIdx]->ToTensor());
    return SUCCESS;
  }

  if (use_v3_) {
    active_executor_->GetWorkspaceSize(
        static_cast<uint64_t*>(workspaceSize),
        input[kXIdx]->ToTensor(),
        input[kExpertIdsIdx]->ToTensor(),
        GetOptionalTensor(input[kScalesIdx]),
        GetOptionalTensor(input[kXActiveMaskIdx]),
        GetOptionalTensor(input[kExpertScalesIdx]),
        GetOptionalTensor(input[kElasticInfoIdx]),
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
        input[kQuantModeIdx]->ToInt(),
        global_bs_real,
        input[kExpertTokenNumsTypeIdx]->ToInt(),
        input[kCommAlgIdx]->ToString(),
        input[kZeroExpertNumIdx]->ToInt(),
        input[kCopyExpertNumIdx]->ToInt(),
        input[kConstExpertNumIdx]->ToInt(),
        (*output_tuple)[kExpandXOutIdx]->ToTensor(),
        (*output_tuple)[kDynamicScalesOutIdx]->ToTensor(),
        (*output_tuple)[kAssistInfoForCombineOutIdx]->ToTensor(),
        (*output_tuple)[kExpertTokenNumsOutIdx]->ToTensor(),
        (*output_tuple)[kEpRecvCountsOutIdx]->ToTensor(),
        (*output_tuple)[kTpRecvCountsOutIdx]->ToTensor(),
        (*output_tuple)[kExpandScalesOutIdx]->ToTensor());
    return SUCCESS;
  }

  active_executor_->GetWorkspaceSize(
      static_cast<uint64_t*>(workspaceSize),
      input[kXIdx]->ToTensor(),
      input[kExpertIdsIdx]->ToTensor(),
      GetOptionalTensor(input[kScalesIdx]),
      GetOptionalTensor(input[kXActiveMaskIdx]),
      GetOptionalTensor(input[kExpertScalesIdx]),
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
      input[kQuantModeIdx]->ToInt(),
      global_bs_real,
      input[kExpertTokenNumsTypeIdx]->ToInt(),
      input[kCommAlgIdx]->ToString(),
      (*output_tuple)[kExpandXOutIdx]->ToTensor(),
      (*output_tuple)[kDynamicScalesOutIdx]->ToTensor(),
      (*output_tuple)[kAssistInfoForCombineOutIdx]->ToTensor(),
      (*output_tuple)[kExpertTokenNumsOutIdx]->ToTensor(),
      (*output_tuple)[kEpRecvCountsOutIdx]->ToTensor(),
      (*output_tuple)[kTpRecvCountsOutIdx]->ToTensor(),
      (*output_tuple)[kExpandScalesOutIdx]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnMoeDistributeDispatchV2::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto& output_tuple = output->ToTuple();

  auto global_bs_real = input[kGlobalBsIdx]->ToInt();
  if (global_bs_real == 0) {
    global_bs_real = input[kXIdx]->ToTensor()->Shape()[0] * input[kEpWorldSizeIdx]->ToInt();
  }

  if (use_v4_) {
    active_executor_->Launch(
        workspace,
        workspaceSize,
        stream,
        input[kXIdx]->ToTensor(),
        input[kExpertIdsIdx]->ToTensor(),
        GetOptionalTensor(input[kScalesIdx]),
        GetOptionalTensor(input[kXActiveMaskIdx]),
        GetOptionalTensor(input[kExpertScalesIdx]),
        GetOptionalTensor(input[kElasticInfoIdx]),
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
        input[kQuantModeIdx]->ToInt(),
        global_bs_real,
        input[kExpertTokenNumsTypeIdx]->ToInt(),
        input[kCommAlgIdx]->ToString(),
        input[kZeroExpertNumIdx]->ToInt(),
        input[kCopyExpertNumIdx]->ToInt(),
        input[kConstExpertNumIdx]->ToInt(),
        (*output_tuple)[kExpandXOutIdx]->ToTensor(),
        (*output_tuple)[kDynamicScalesOutIdx]->ToTensor(),
        (*output_tuple)[kAssistInfoForCombineOutIdx]->ToTensor(),
        (*output_tuple)[kExpertTokenNumsOutIdx]->ToTensor(),
        (*output_tuple)[kEpRecvCountsOutIdx]->ToTensor(),
        (*output_tuple)[kTpRecvCountsOutIdx]->ToTensor(),
        (*output_tuple)[kExpandScalesOutIdx]->ToTensor());
    return SUCCESS;
  }

  if (use_v3_) {
    active_executor_->Launch(
        workspace,
        workspaceSize,
        stream,
        input[kXIdx]->ToTensor(),
        input[kExpertIdsIdx]->ToTensor(),
        GetOptionalTensor(input[kScalesIdx]),
        GetOptionalTensor(input[kXActiveMaskIdx]),
        GetOptionalTensor(input[kExpertScalesIdx]),
        GetOptionalTensor(input[kElasticInfoIdx]),
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
        input[kQuantModeIdx]->ToInt(),
        global_bs_real,
        input[kExpertTokenNumsTypeIdx]->ToInt(),
        input[kCommAlgIdx]->ToString(),
        input[kZeroExpertNumIdx]->ToInt(),
        input[kCopyExpertNumIdx]->ToInt(),
        input[kConstExpertNumIdx]->ToInt(),
        (*output_tuple)[kExpandXOutIdx]->ToTensor(),
        (*output_tuple)[kDynamicScalesOutIdx]->ToTensor(),
        (*output_tuple)[kAssistInfoForCombineOutIdx]->ToTensor(),
        (*output_tuple)[kExpertTokenNumsOutIdx]->ToTensor(),
        (*output_tuple)[kEpRecvCountsOutIdx]->ToTensor(),
        (*output_tuple)[kTpRecvCountsOutIdx]->ToTensor(),
        (*output_tuple)[kExpandScalesOutIdx]->ToTensor());
    return SUCCESS;
  }

  active_executor_->Launch(
      workspace,
      workspaceSize,
      stream,
      input[kXIdx]->ToTensor(),
      input[kExpertIdsIdx]->ToTensor(),
      GetOptionalTensor(input[kScalesIdx]),
      GetOptionalTensor(input[kXActiveMaskIdx]),
      GetOptionalTensor(input[kExpertScalesIdx]),
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
      input[kQuantModeIdx]->ToInt(),
      global_bs_real,
      input[kExpertTokenNumsTypeIdx]->ToInt(),
      input[kCommAlgIdx]->ToString(),
      (*output_tuple)[kExpandXOutIdx]->ToTensor(),
      (*output_tuple)[kDynamicScalesOutIdx]->ToTensor(),
      (*output_tuple)[kAssistInfoForCombineOutIdx]->ToTensor(),
      (*output_tuple)[kExpertTokenNumsOutIdx]->ToTensor(),
      (*output_tuple)[kEpRecvCountsOutIdx]->ToTensor(),
      (*output_tuple)[kTpRecvCountsOutIdx]->ToTensor(),
      (*output_tuple)[kExpandScalesOutIdx]->ToTensor());
  return SUCCESS;
}

FXRT_REG_OP(moe_distribute_dispatch_v2, AclnnMoeDistributeDispatchV2, Ascend);
} // namespace ops
} // namespace fxrt
