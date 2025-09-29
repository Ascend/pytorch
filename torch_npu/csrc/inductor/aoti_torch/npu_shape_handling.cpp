#include "torch_npu/csrc/inductor/aoti_torch/npu_shape_handling.h"

#include <algorithm>

namespace F = torch::nn::functional;

namespace torch::aot_inductor {
NPUShapeHandling::NPUShapeHandling(std::vector<int> &sizes)
    : m_sizes(sizes)
{
    TORCH_CHECK(sizes.size() > 0, "The size list cannot be empty.");
    TORCH_CHECK(sizes.size() <= MAX_SHAPE_GEARS, "The maximum number of supported gear is 64.");
    m_policy = ShapePolicy::CONSTANT;
    // sort sizes
    std::sort(m_sizes.begin(), m_sizes.end());
    m_min_size = m_sizes.front();
    m_max_size = m_sizes.back();
    TORCH_CHECK(m_min_size >= MIN_SHAPE_SIZE, "The minimum gear is 1.");
    TORCH_CHECK(m_max_size <= MAX_SHAPE_SIZE, "The maximum gear is 8192.");
}

NPUShapeHandling::NPUShapeHandling(int min_size, int max_size, ShapePolicy policy)
    : m_min_size(min_size), m_max_size(max_size), m_policy(policy)
{
    TORCH_CHECK(m_min_size <= m_max_size, "The min_size cannot be greater than the max_size.");
    TORCH_CHECK(m_min_size >= MIN_SHAPE_SIZE, "The minimum gear is 1.");
    TORCH_CHECK(m_max_size <= MAX_SHAPE_SIZE, "The maximum gear is 8192.");

    int cur_size = min_size;
    while (cur_size <= max_size) {
        m_sizes.push_back(cur_size);
        cur_size *= DEFAULT_SHAPE_TIMES;
    }

    m_max_size = m_sizes.back();
}

int NPUShapeHandling::FindClosestSize(int target_size)
{
    if (m_min_size >= target_size) {
        return m_min_size;
    }

    if (m_max_size < target_size) {
        return -1;
    }

    int left = 0;
    int right = m_sizes.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (m_sizes[mid] < target_size) {
            left = mid + 1;
        } else if (m_sizes[mid] > target_size) {
            right = mid - 1;
        } else {
            return target_size;
        }
    }

    return m_sizes[left];
}

uint64_t NPUShapeHandling::PackOpInfo(int op, int index, int ori_size, int dim)
{
    /*
     * Using a 64-bit integer to represent a specific transformation operation
     * op: padding or split
     * index: the position of the tensor being operated on in the current step within the list
     * ori_size: the size of the specified dimension before the operation
     * dim: the dimension being operated on
     * The composition is: dim(highest 28 bits) | ori_size(next 16 bits) | index(next 16 bits) | op(lowest 4 bits)
     */
    uint64_t operation = 0;
    operation |= static_cast<uint64_t>(op) & 0x0F;
    operation |= (static_cast<uint64_t>(index) & 0xFFFF) << INDEX_SHIFT_AMOUNT;
    operation |= (static_cast<uint64_t>(ori_size) & 0xFFFF) << SIZE_SHIFT_AMOUNT;
    operation |= (static_cast<uint64_t>(dim) & 0xFFFFFFF) << DIM_SHIFT_AMOUNT;
    return operation;
}

void NPUShapeHandling::GenerateExpectedRes(std::vector<at::Tensor> &inputs, std::vector<int> &indexs,
    std::vector<std::vector<at::Tensor>> &mid_results, std::vector<std::vector<at::Tensor>> &outputs)
{
    size_t num_splits = mid_results.front().size();
    outputs.resize(num_splits);
    for (size_t i = 0; i < num_splits; i++) {
        outputs[i].reserve(inputs.size());
        size_t pos = 0;
        for (size_t j = 0; j < inputs.size(); j++) {
            if (indexs[pos] == j) {
                outputs[i].push_back(mid_results[pos++][i]);
            } else {
                outputs[i].push_back(inputs[j]);
            }
        }
    }
}

void NPUShapeHandling::TransformValidate(std::vector<at::Tensor> &inputs, std::vector<int> &indexs, int dim)
{
    TORCH_CHECK(indexs.size() > 0, "At least on tensor needs to be transformed.");

    std::sort(indexs.begin(), indexs.end());
    TORCH_CHECK(indexs.front() >= 0, "Index must be a non-negative number.");
    TORCH_CHECK(indexs.back() < inputs.size(), "The index must be within the range of the tensor vector.");
    TORCH_CHECK(dim >= 0, "Dim must be a non-negative number,");

    int dim_size = -1;
    for (size_t i = 0; i < indexs.size(); i++) {
        TORCH_CHECK(dim < inputs[indexs[i]].dim(), "The dim cannot exceed the actual dimension size of the tensor.");
        if (dim_size == -1) {
            dim_size = inputs[indexs[i]].size(dim);
        } else {
            TORCH_CHECK(dim_size == inputs[indexs[i]].size(dim),
                "The specified dimension sizes of all tensors to be transformed must be consistent.");
        }
    }
}

void NPUShapeHandling::Transform(std::vector<at::Tensor> &inputs, std::vector<int> &indexs,
    std::vector<std::vector<at::Tensor>> &outputs, int dim, double value)
{
    TransformValidate(inputs, indexs, dim);
    std::vector<std::vector<at::Tensor>> mid_results;
    bool first_operation = true;
    int ori_dim_size = inputs[indexs.front()].size(dim);
    if (ori_dim_size > m_max_size) {
        // split first
        CleanRecords();
        first_operation = false;
        for (int i = 0; i < indexs.size(); i++) {
            mid_results.push_back(torch::split(inputs[i], m_max_size, dim));
        }
        uint64_t ops = PackOpInfo(1, 0, ori_dim_size, dim);
        m_records.push_back(ops);
    }

    if (first_operation) {
        size_t length = indexs.size();
        mid_results.resize(length);
        // no split operation was performed
        for (size_t i = 0; i < length; i++) {
            mid_results[i].push_back(inputs[indexs[i]]);
        }
    }

    int last_tensor_index = mid_results[0].size() - 1;
    int last_tensor_size = mid_results[0][last_tensor_index].size(dim);
    int padding_size = FindClosestSize(last_tensor_size);
    if (padding_size != last_tensor_size) {
        // pad here
        if (first_operation) {
            CleanRecords();
        }
        
        for (int i = 0; i < mid_results.size(); i++) {
            int total_dims = mid_results[i][last_tensor_index].dim();
            std::vector<int64_t> padding(total_dims * PADDING_VALUES_PER_DIM, 0);
            padding[padding.size() - 1 - dim * PADDING_VALUES_PER_DIM] = padding_size - last_tensor_size;
            mid_results[i][last_tensor_index] = F::pad(mid_results[i][last_tensor_index],
                F::PadFuncOptions(padding).mode(torch::kConstant).value(value));
        }

        uint64_t ops = PackOpInfo(0, last_tensor_index, last_tensor_size, dim);
        m_records.push_back(ops);
    }

    GenerateExpectedRes(inputs, indexs, mid_results, outputs);
}

static std::vector<std::vector<at::Tensor>> Transpose2dVector(const std::vector<std::vector<at::Tensor>> &src)
{
    size_t rows = src.size();
    size_t cols = src[0].size();
    std::vector<std::vector<at::Tensor>> dst(cols, std::vector<at::Tensor>(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            dst[j][i] = src[i][j];
        }
    }
    return dst;
}

void NPUShapeHandling::RecoverValidate(std::vector<std::vector<at::Tensor>> &inputs)
{
    if (m_records.empty()) {
        TORCH_CHECK(inputs.size() == 1, "No transformation record found");
    }
}

/*
* input: {{res1_1, res2_1...resn_1}, {res1_2, res2_2...resn_2}...{res1_m, res2_m...resn_m}}
* output: {res1, res2...resn}
*/
void NPUShapeHandling::Recover(std::vector<std::vector<at::Tensor>> &inputs, std::vector<at::Tensor> &outputs)
{
    if (inputs.empty() || inputs.front().empty()) {
        return;
    }
    RecoverValidate(inputs);
    size_t rows = inputs.size();
    size_t cols = inputs[0].size();
    if (m_records.empty()) {
        for (size_t i = 0; i < cols; i++) {
            outputs.push_back(inputs[0][i]);
        }
        return;
    }

    auto tmp = Transpose2dVector(inputs);

    while (!m_records.empty()) {
        uint64_t operation = m_records.back();
        int op = static_cast<int>(operation & 0x0F);
        int index = static_cast<int>((operation >> 4) & 0xFFFF);
        int ori_size = static_cast<int>((operation >> 20) & 0xFFFF);
        int dim = static_cast<int>((operation >> 36) & 0xFFFFFFF);
        if (op == 0) {
            // handle pad
            for (size_t i = 0; i < tmp.size(); i++) {
                tmp[i][index] = tmp[i][index].slice(dim, 0, ori_size);
            }
            m_records.pop_back();
            continue;
        }

        // handle split, the last operation
        for (size_t i = 0; i < tmp.size(); i++) {
            outputs.push_back(at::cat(tmp[i], dim));
        }
        m_records.pop_back();
        return;
    }

    // no transformation operation or no split operation
    for (size_t i = 0; i < cols; i++) {
        outputs.push_back(tmp[i][0]);
    }
    return;
}
}