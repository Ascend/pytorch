#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <cstdint>
#include <memory>  // 智能指针，管理策略对象
#include <stdexcept>
#include <ATen/Tensor.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch::aot_inductor {
#define MAX_GEARS_NUM 64
#define COMMON_MIN_SIZE 1

#define MIN_BS_GEAR 1
#define MAX_BS_GEAR 8192

#define MIN_SEQ_GEAR 1
#define MAX_SEQ_GEAR 1310720

#define INDEX_SHIFT_AMOUNT 4
#define SIZE_SHIFT_AMOUNT 20
#define DIM_SHIFT_AMOUNT 56
#define OP_MASK 0xF
#define INDEX_MASK 0xFFFF
#define SIZE_MASK 0xFFFFFFFFF
#define DIM_MASK 0xFF

#define POLICY_TIMES_FACTOR 2

#define PADDING_VALUES_PER_DIM 2

enum class ShapePolicy {
    TIMES,
    CUSTOM
};

enum class ShapeType {
    BATCHSIZE = 0,
    SEQLEN
};

class ShapeOpStrategyBase {
public:
    ShapeOpStrategyBase() = default;
    virtual ~ShapeOpStrategyBase() = default;

    int64_t FindClosestGear(int64_t cur_size,
        const std::vector<int64_t> &gears, int64_t min_gear, int64_t max_gear);

    uint64_t EncodeTransformStep(int op, int index, int64_t original_size, int dimension);

    void DecodeTransformStep(uint64_t operation, int& op, int& index, int64_t& original_size, int& dimension);

    std::vector<int> m_indices;
    double m_value;
    int64_t m_min_gear;
    int64_t m_max_gear;
    std::vector<int64_t> m_gears;
};

class BSShapeOpStrategy : public ShapeOpStrategyBase {
public:
    BSShapeOpStrategy() = default;
    virtual ~BSShapeOpStrategy() = default;

    void InitializeCore(std::vector<int64_t>& gears, int dimension, std::vector<int>& indices, double value = 0.0);

    virtual void Transform(std::vector<at::Tensor> &inputs, std::vector<std::vector<at::Tensor>> &outputs) = 0;

    virtual void Recover(std::vector<std::vector<at::Tensor>> &inputs, std::vector<at::Tensor> &outputs) = 0;

    int m_dimension;
};

class SeqShapeOpStrategy : public ShapeOpStrategyBase {
public:
    SeqShapeOpStrategy() = default;
    virtual ~SeqShapeOpStrategy() = default;

    void InitializeCore(std::vector<int64_t>& gears, std::vector<int>& dimensions,
        std::vector<int>& indices, double value = 0.0);

    virtual void Transform(std::vector<at::Tensor> &inputs, std::vector<at::Tensor> &outputs) = 0;

    std::vector<int> m_dimensions;
};

class DefaultBSShapeOp : public BSShapeOpStrategy {
public:
    void Transform(std::vector<at::Tensor> &inputs, std::vector<std::vector<at::Tensor>> &outputs) override;

    void Recover(std::vector<std::vector<at::Tensor>> &inputs, std::vector<at::Tensor> &outputs) override;

private:
    void GenerateExpectedRes(std::vector<at::Tensor> &inputs, std::vector<std::vector<at::Tensor>> &mid_results,
        std::vector<std::vector<at::Tensor>> &outputs);

    void TransformValidate(std::vector<at::Tensor> &inputs);

    void RecoverValidate(std::vector<std::vector<at::Tensor>> &inputs);

    void CleanRecords()
    {
        m_records.clear();
    }

    std::vector<uint64_t> m_records;
};

class DefaultSeqShapeOp : public SeqShapeOpStrategy {
public:
    void Transform(std::vector<at::Tensor> &inputs, std::vector<at::Tensor> &outputs) override;

private:
    void TransformValidate(std::vector<at::Tensor> &inputs);
};

class NPUShapeHandling {
public:
    NPUShapeHandling()
        : m_bs_strategy(std::make_unique<DefaultBSShapeOp>()),
        m_seq_strategy(std::make_unique<DefaultSeqShapeOp>()),
        initialized(false),
        handle_batchsize(false),
        handle_sequence(false)
    {}

    ~NPUShapeHandling() = default;

    // Strategy Register
    void RegisterBatchSizeStrategy(std::shared_ptr<BSShapeOpStrategy> custom_strategy);
    void RegisterSequenceStrategy(std::shared_ptr<SeqShapeOpStrategy> custom_strategy);

    void Initialize(ShapeType type, std::vector<int64_t> &gears, std::vector<int>& dimensions,
        std::vector<int>& indices, double value = 0.0);

    void Initialize(ShapeType type, int64_t min_size, int64_t max_size, ShapePolicy policy,
        std::vector<int>& dimensions, std::vector<int>& indices, double value = 0.0);

    void Transform(std::vector<at::Tensor> &inputs, std::vector<std::vector<at::Tensor>> &outputs);

    void Recover(std::vector<std::vector<at::Tensor>> &inputs, std::vector<at::Tensor> &outputs);

private:
    void GenerateGears(int64_t min_size, int64_t max_size, ShapePolicy policy, std::vector<int64_t>& gears);

    bool initialized;
    bool handle_batchsize;
    bool handle_sequence;
    ShapePolicy m_policy;
    std::shared_ptr<BSShapeOpStrategy> m_bs_strategy;
    std::shared_ptr<SeqShapeOpStrategy> m_seq_strategy;
};
}  // namespace torch::aot_inductor