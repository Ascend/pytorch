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
#define MIN_SHAPE_SIZE 1
#define MAX_SHAPE_SIZE 8192
#define DEFAULT_MIN_SIZE 1
#define DEFAULT_MAX_SIZE 1024
#define MAX_SHAPE_GEARS 64
#define DEFAULT_SHAPE_TIMES 2
#define INDEX_SHIFT_AMOUNT 4
#define SIZE_SHIFT_AMOUNT 20
#define DIM_SHIFT_AMOUNT 36
#define PADDING_VALUES_PER_DIM 2

enum class ShapePolicy { TIMES, CONSTANT };

enum class OpType { PAD = 0, SPLIT };

// ------------------------------ 1. 策略基类：定义统一接口 ------------------------------
// 所有Transform/Recover的实现都需继承此类，确保接口一致
class ShapeOpStrategy {
public:
    ShapeOpStrategy()
    {}
    // 析构函数设为虚函数，确保子类析构被调用
    virtual ~ShapeOpStrategy() = default;

    // 纯虚接口：与原NPUShapeHandling的Transform参数完全一致
    virtual void Transform(
        // 传入NPUShapeHandling的核心配置，供策略使用
        const std::vector<int> &sizes, int min_size, int max_size,
        // 原Transform的输入输出参数
        std::vector<at::Tensor> &inputs, std::vector<int> &indexs, std::vector<std::vector<at::Tensor>> &outputs,
        int dim = 0, double value = 0.0) = 0;

    // 纯虚接口：与原NPUShapeHandling的Recover参数完全一致
    virtual void Recover(
        // 传入NPUShapeHandling的核心配置
        const std::vector<int> &sizes, int min_size, int max_size,
        // 原Recover的输入输出参数
        std::vector<std::vector<at::Tensor>> &inputs, std::vector<at::Tensor> &outputs) = 0;

    // 辅助函数：封装FindClosestSize（所有策略共用，避免重复实现）
    int FindClosestSize(int target_size, const std::vector<int> &sizes, int min_size, int max_size);

    // 辅助函数：封装PackOpInfo（所有策略共用）
    uint64_t PackOpInfo(int op, int index, int ori_size, int dim);
};

class DefaultShapeOp : public ShapeOpStrategy {
public:
    void Transform(const std::vector<int> &sizes, int min_size, int max_size, std::vector<at::Tensor> &inputs,
        std::vector<int> &indexs, std::vector<std::vector<at::Tensor>> &outputs, int dim, double value) override;

    void Recover(const std::vector<int> &sizes, int min_size, int max_size,
        std::vector<std::vector<at::Tensor>> &inputs, std::vector<at::Tensor> &outputs) override;

private:
    void GenerateExpectedRes(std::vector<at::Tensor> &inputs, std::vector<int> &indexs,
        std::vector<std::vector<at::Tensor>> &mid_results, std::vector<std::vector<at::Tensor>> &outputs);

    void TransformValidate(std::vector<at::Tensor> &inputs, std::vector<int> &indexs, int dim);

    void RecoverValidate(std::vector<std::vector<at::Tensor>> &inputs);

    void CleanRecords()
    {
        m_records.clear();
    }

    std::vector<uint64_t> m_records;  // 操作记录，供策略读写
};

class NPUShapeHandling {
public:
    // 原构造函数：初始化配置，默认使用DefaultShapeOp策略
    NPUShapeHandling(std::vector<int> &sizes);

    NPUShapeHandling(int min_size, int max_size, ShapePolicy policy);

    ~NPUShapeHandling() = default;

    // ------------------------------ 关键：策略注册接口 ------------------------------
    // 外部通过此接口注入自定义策略，替换默认实现
    void RegisterShapeOpStrategy(std::shared_ptr<ShapeOpStrategy> custom_strategy)
    {
        if (!custom_strategy) {
            throw std::invalid_argument("Custom strategy cannot be null");
        }
        m_strategy = std::move(custom_strategy);  // 转移所有权，替换策略
    }

    // ------------------------------ 委托执行：无具体实现，全部交给策略 ------------------------------
    void Transform(std::vector<at::Tensor> &inputs, std::vector<int> &indexs,
        std::vector<std::vector<at::Tensor>> &outputs, int dim = 0, double value = 0.0)
    {
        // 委托给当前策略的Transform
        m_strategy->Transform(m_sizes, m_min_size, m_max_size, inputs, indexs, outputs, dim, value);
    }

    void Recover(std::vector<std::vector<at::Tensor>> &inputs, std::vector<at::Tensor> &outputs)
    {
        // 委托给当前策略的Recover
        m_strategy->Recover(m_sizes, m_min_size, m_max_size, inputs, outputs);
    }

    // 原public辅助函数：保留，供外部或策略调用
    int FindClosestSize(int target_size)
    {
        return m_strategy->FindClosestSize(target_size, m_sizes, m_min_size, m_max_size);
    }

private:
    // 原成员变量：保留配置信息
    ShapePolicy m_policy;
    std::vector<int> m_sizes;
    int m_min_size;
    int m_max_size;

    // 新增：策略对象指针（智能指针自动管理内存）
    std::shared_ptr<ShapeOpStrategy> m_strategy;
};
}  // namespace torch::aot_inductor