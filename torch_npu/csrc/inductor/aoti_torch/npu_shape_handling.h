#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <cstdint>
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

enum class ShapePolicy {
    TIMES,
    CONSTANT
};

enum class OpType {
    PAD = 0,
    SPLIT
};

// <OpType>, <Tensor_Index>, <Tensor_DIM>, <Point>
class NPUShapeHandling {
public:
    NPUShapeHandling(std::vector<int> &sizes);
    NPUShapeHandling(int min_size, int max_size, ShapePolicy policy);
    ~NPUShapeHandling() = default;
    // x,y,z
    // <x1, y1, z1>, <x2, y2, z2>
    void Transform(std::vector<at::Tensor> &inputs, std::vector<int> &indexs,
        std::vector<std::vector<at::Tensor>> &outputs, int dim = 0, double value = 0.0);
    void Recover(std::vector<std::vector<at::Tensor>> &inputs, std::vector<at::Tensor> &outputs);
    int FindClosestSize(int target_size);
private:
    void GenerateExpectedRes(std::vector<at::Tensor> &inputs, std::vector<int> &indexs,
        std::vector<std::vector<at::Tensor>> &mid_results, std::vector<std::vector<at::Tensor>> &outputs);
    void TransformValidate(std::vector<at::Tensor> &inputs, std::vector<int> &indexs, int dim);
    void RecoverValidate(std::vector<std::vector<at::Tensor>> &inputs);
    uint64_t PackOpInfo(int op, int index, int ori_size, int dim);
    void CleanRecords()
    {
        m_records.clear();
    }
    ShapePolicy m_policy;
    std::vector<int> m_sizes;
    int m_min_size;
    int m_max_size;
    std::vector<uint64_t> m_records;
};
}