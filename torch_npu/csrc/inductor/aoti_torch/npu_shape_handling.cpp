#include "torch_npu/csrc/inductor/aoti_torch/npu_shape_handling.h"

#include <algorithm>
#include <cmath>

namespace F = torch::nn::functional;

namespace torch::aot_inductor {
int64_t ShapeOpStrategyBase::FindClosestGear(int64_t cur_size,
    const std::vector<int64_t> &gears, int64_t min_gear, int64_t max_gear)
{
    TORCH_CHECK(max_gear >= cur_size, "Input size (", cur_size, ") exceeds the max gear (", max_gear, ").");
    if (min_gear >= cur_size) {
        return min_gear;
    }

    int left = 0;
    int right = gears.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (gears[mid] < cur_size) {
            left = mid + 1;
        } else if (gears[mid] > cur_size) {
            right = mid - 1;
        } else {
            return cur_size;
        }
    }

    return gears[left];
}

uint64_t ShapeOpStrategyBase::EncodeTransformStep(int op, int index, int64_t original_size, int dimension)
{
    /*
     ◦ Using a 64-bit integer to represent a specific transformation operation
     ◦ op: padding or split
     ◦ index: the position of the tensor being operated on in the current step within the list
     ◦ original_size: the size of the specified dimension before the operation
     ◦ dimension: the dimension being operated on
     ◦ The composition is: dimension(highest 8 bits) | original_size(next 36 bits) | index(next 16 bits) | op(lowest 4 bits)
     */
    TORCH_CHECK(op >= 0 && op <= OP_MASK,
        "op should be in range [0, ", OP_MASK, "]");
    TORCH_CHECK(index >= 0 && index <= INDEX_MASK,
        "tensor index should be in range [0, ", INDEX_MASK, "]");
    TORCH_CHECK(original_size >= 0 && original_size <= SIZE_MASK,
        "tensor size should be in range [0, ", SIZE_MASK, "]");
    TORCH_CHECK(dimension >= 0 && dimension <= DIM_MASK,
        "dimension should be in range [0, ", DIM_MASK, "]");
    uint64_t operation = 0;
    operation |= static_cast<uint64_t>(op) & OP_MASK;
    operation |= (static_cast<uint64_t>(index) & INDEX_MASK) << INDEX_SHIFT_AMOUNT;
    operation |= (static_cast<uint64_t>(original_size) & SIZE_MASK) << SIZE_SHIFT_AMOUNT;
    operation |= (static_cast<uint64_t>(dimension) & DIM_MASK) << DIM_SHIFT_AMOUNT;
    return operation;
}

void ShapeOpStrategyBase::DecodeTransformStep(uint64_t operation, int& op, int& index,
    int64_t& original_size, int& dimension)
{
    op = static_cast<int>(operation & OP_MASK);
    index = static_cast<int>((operation >> INDEX_SHIFT_AMOUNT) & INDEX_MASK);
    original_size = static_cast<int64_t>((operation >> SIZE_SHIFT_AMOUNT) & SIZE_MASK);
    dimension = static_cast<int>((operation >> DIM_SHIFT_AMOUNT) & DIM_MASK);
}

void BSShapeOpStrategy::InitializeCore(std::vector<int64_t>& gears, int dimension,
    std::vector<int>& indices, double value)
{
    TORCH_CHECK(!gears.empty(), "At least one batch size gear must be provided.");
    TORCH_CHECK(gears.size() <= MAX_GEARS_NUM,
        "Number of batch size gears (", gears.size(), ") exceeds maximum supported (", MAX_GEARS_NUM, ").");

    std::sort(gears.begin(), gears.end());
    TORCH_CHECK(gears.front() >= MIN_BS_GEAR,
        "Minimum batch size (", gears.front(), ") must be >= ", MIN_BS_GEAR, ".");
    TORCH_CHECK(gears.back() <= MAX_BS_GEAR,
        "Maximum batch size (", gears.back(), ") must be <= ", MAX_BS_GEAR, ".");
    TORCH_CHECK(std::adjacent_find(gears.begin(), gears.end()) == gears.end(),
        "Batch size gears must be unique.")
    
    TORCH_CHECK(dimension >= 0, "Dimension must be non-negative (got ", dimension, ").");
    
    TORCH_CHECK(indices.size() > 0, "At least one tensor index must be provided for transformation.");
    
    std::sort(indices.begin(), indices.end());
    TORCH_CHECK(indices.front() >= 0,
        "Tensor index must be non-negative (found negative index: ", indices.front(), ").");
    TORCH_CHECK(std::adjacent_find(indices.begin(), indices.end()) == indices.end(),
        "Tensor indices must be unique.")

    m_dimension = dimension;
    m_indices = indices;
    m_value = value;
    m_gears = gears;
    m_min_gear = gears.front();
    m_max_gear = gears.back();
}

void SeqShapeOpStrategy::InitializeCore(std::vector<int64_t>& gears, std::vector<int>& dimensions,
    std::vector<int>& indices, double value)
{
    TORCH_CHECK(!gears.empty(), "At least one sequence gear must be provided.");
    TORCH_CHECK(gears.size() <= MAX_GEARS_NUM,
        "Number of sequence length gears (", gears.size(), ") exceeds maximum supported (", MAX_GEARS_NUM, ").");
    
    std::sort(gears.begin(), gears.end());
    TORCH_CHECK(gears.front() >= MIN_SEQ_GEAR,
        "Minimum sequence length (", gears.front(), ") must be >= ", MIN_SEQ_GEAR, ".");
    TORCH_CHECK(gears.back() <= MAX_SEQ_GEAR,
        "Maximum sequence length (", gears.back(), ") must be <= ", MAX_SEQ_GEAR, ".");
    TORCH_CHECK(std::adjacent_find(gears.begin(), gears.end()) == gears.end(),
        "Sequence length gears must be unique.")

    TORCH_CHECK(indices.size() > 0 && dimensions.size() > 0,
        "At least one tensor index must be provided for transformation.");
    TORCH_CHECK(indices.size() == dimensions.size(),
        "The length of indices must be consistent with the length of dimensions.");
    for (auto dimension : dimensions) {
        TORCH_CHECK(dimension >= 0, "Dimension must be non-negative (got ", dimension, ").");
    }

    std::vector<size_t> tmp_indices(indices.size());
    std::iota(tmp_indices.begin(), tmp_indices.end(), 0);
    std::sort(tmp_indices.begin(), tmp_indices.end(),
        [&indices](size_t i, size_t j) {
            return indices[i] < indices[j];
        });
    TORCH_CHECK(indices[tmp_indices.front()] >= 0,
        "Tensor index must be non-negative (found negative index: ", indices[tmp_indices.front()], ").");
    TORCH_CHECK(std::adjacent_find(indices.begin(), indices.end()) == indices.end(),
        "Tensor indices must be unique.")

    for (auto i : tmp_indices) {
        m_indices.push_back(indices[i]);
        m_dimensions.push_back(dimensions[i]);
    }
    
    m_value = value;
    m_gears = gears;
    m_min_gear = gears.front();
    m_max_gear = gears.back();
}

void DefaultBSShapeOp::GenerateExpectedRes(std::vector<at::Tensor> &inputs,
    std::vector<std::vector<at::Tensor>> &mid_results, std::vector<std::vector<at::Tensor>> &outputs)
{
    size_t num_splits = mid_results.front().size();
    outputs.resize(num_splits);
    for (size_t i = 0; i < num_splits; i++) {
        outputs[i].reserve(inputs.size());
        size_t pos = 0;
        for (size_t j = 0; j < inputs.size(); j++) {
            if (m_indices[pos] == j) {
                outputs[i].push_back(mid_results[pos++][i]);
            } else {
                outputs[i].push_back(inputs[j]);
            }
        }
    }
}

void DefaultBSShapeOp::TransformValidate(std::vector<at::Tensor> &inputs)
{
    TORCH_CHECK(m_indices.back() < inputs.size(),
        "Index out of the range of the tensor vector, check inputs or update the indices config.");

    int dim_size = -1;
    for (auto index : m_indices) {
        TORCH_CHECK(m_dimension < inputs[index].dim(),
            "The dimension exceeds the actual tensor's dimension range, check inputs or update the dimension config.");
        if (dim_size == -1) {
            dim_size = inputs[index].size(m_dimension);
        } else {
            TORCH_CHECK(dim_size == inputs[index].size(m_dimension),
                "The size of the BS dimension for all tensors to be transformed must be consistent.");
        }
    }
}

void DefaultBSShapeOp::Transform(std::vector<at::Tensor> &inputs, std::vector<std::vector<at::Tensor>> &outputs)
{
    TransformValidate(inputs);
    std::vector<std::vector<at::Tensor>> mid_results;
    bool first_operation = true;
    int64_t ori_dim_size = inputs[m_indices.front()].size(m_dimension);
    if (ori_dim_size > m_max_gear) {
        // split first
        CleanRecords();
        first_operation = false;
        for (int i = 0; i < m_indices.size(); i++) {
            mid_results.push_back(torch::split(inputs[m_indices[i]], m_max_gear, m_dimension));
        }
        uint64_t ops = EncodeTransformStep(1, 0, ori_dim_size, m_dimension);
        m_records.push_back(ops);
    }

    if (first_operation) {
        size_t length = m_indices.size();
        mid_results.resize(length);
        // no split operation was performed
        for (size_t i = 0; i < length; i++) {
            mid_results[i].push_back(inputs[m_indices[i]]);
        }
    }

    int last_tensor_index = mid_results[0].size() - 1;
    int64_t last_tensor_size = mid_results[0][last_tensor_index].size(m_dimension);
    int64_t target_gear = FindClosestGear(last_tensor_size, m_gears, m_min_gear, m_max_gear);
    if (target_gear != last_tensor_size) {
        // pad here
        if (first_operation) {
            CleanRecords();
        }

        for (int i = 0; i < mid_results.size(); i++) {
            int total_dims = mid_results[i][last_tensor_index].dim();
            std::vector<int64_t> padding(total_dims * PADDING_VALUES_PER_DIM, 0);
            padding[padding.size() - 1 - m_dimension * PADDING_VALUES_PER_DIM] = target_gear - last_tensor_size;
            mid_results[i][last_tensor_index] = F::pad(
                mid_results[i][last_tensor_index], F::PadFuncOptions(padding).mode(torch::kConstant).value(m_value));
        }

        uint64_t ops = EncodeTransformStep(0, last_tensor_index, last_tensor_size, m_dimension);
        m_records.push_back(ops);
    }

    GenerateExpectedRes(inputs, mid_results, outputs);
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

void DefaultBSShapeOp::RecoverValidate(std::vector<std::vector<at::Tensor>> &inputs)
{
    if (m_records.empty()) {
        TORCH_CHECK(inputs.size() == 1, "No transformation record found");
    }
}

/*
• input: {{res1_1, res2_1...resn_1}, {res1_2, res2_2...resn_2}...{res1_m, res2_m...resn_m}}

• output: {res1, res2...resn}

*/
void DefaultBSShapeOp::Recover(std::vector<std::vector<at::Tensor>> &inputs, std::vector<at::Tensor> &outputs)
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
        int op = 0;
        int index = 0;
        int64_t original_size = 0;
        int dimension = 0;
        DecodeTransformStep(operation, op, index, original_size, dimension);

        if (op == 0) {
            // handle pad
            for (size_t i = 0; i < tmp.size(); i++) {
                tmp[i][index] = tmp[i][index].slice(dimension, 0, original_size);
            }
            m_records.pop_back();
            continue;
        }

        // handle split, the last operation
        for (size_t i = 0; i < tmp.size(); i++) {
            outputs.push_back(at::cat(tmp[i], dimension));
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

void DefaultSeqShapeOp::TransformValidate(std::vector<at::Tensor> &inputs)
{
    TORCH_CHECK(m_indices.back() < inputs.size(),
        "Index out of the range of the tensor vector, check inputs or update the indices config.");

    int dim_size = -1;
    for (size_t i = 0; i < m_indices.size(); i++) {
        TORCH_CHECK(m_dimensions[i] < inputs[m_indices[i]].dim(),
            "The dimension exceeds the actual tensor's dimension range, check inputs or update the dimension config.");
        TORCH_CHECK(m_max_gear >= inputs[m_indices[i]].size(m_dimensions[i]),
            "The length of the input tensor sequence exceeds the maximum supported gear, check the input.");
    }
}

void DefaultSeqShapeOp::Transform(std::vector<at::Tensor> &inputs, std::vector<at::Tensor> &outputs)
{
    TransformValidate(inputs);
    size_t pos = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        if (m_indices[pos] != i) {
            outputs.push_back(inputs[i]);
            continue;
        }

        int64_t sequence_length = inputs[i].size(m_dimensions[pos]);
        int64_t target_gear = FindClosestGear(sequence_length, m_gears, m_min_gear, m_max_gear);
        if (target_gear == sequence_length) {
            outputs.push_back(inputs[i]);
            pos ++;
            continue;
        }

        // pad here
        int total_dims = inputs[i].dim();
        std::vector<int64_t> padding(total_dims * PADDING_VALUES_PER_DIM, 0);
        padding[padding.size() - 1 - m_dimensions[pos] * PADDING_VALUES_PER_DIM] = target_gear - sequence_length;
        outputs.push_back(F::pad(inputs[i], F::PadFuncOptions(padding).mode(torch::kConstant).value(m_value)));

        pos ++;
    }
}

void NPUShapeHandling::RegisterBatchSizeStrategy(std::shared_ptr<BSShapeOpStrategy> custom_strategy)
{
    TORCH_CHECK(custom_strategy != nullptr, "Custom strategy cannot be null.");
    if (initialized && handle_batchsize && m_bs_strategy) {
        *custom_strategy = *m_bs_strategy;
    }
    m_bs_strategy = std::move(custom_strategy);
}

void NPUShapeHandling::RegisterSequenceStrategy(std::shared_ptr<SeqShapeOpStrategy> custom_strategy)
{
    TORCH_CHECK(custom_strategy != nullptr, "Custom strategy cannot be null.");
    if (initialized && handle_sequence && m_seq_strategy) {
        *custom_strategy = *m_seq_strategy;
    }
    m_seq_strategy = std::move(custom_strategy);
}

void NPUShapeHandling::Initialize(ShapeType type, std::vector<int64_t> &gears, std::vector<int>& dimensions,
    std::vector<int>& indices, double value)
{
    m_policy = ShapePolicy::CUSTOM;
    if (type == ShapeType::BATCHSIZE) {
        m_bs_strategy->InitializeCore(gears, dimensions.front(), indices, value);
        handle_batchsize = true;
    } else {
        m_seq_strategy->InitializeCore(gears, dimensions, indices, value);
        handle_sequence = true;
    }
    initialized = true;
}

void NPUShapeHandling::Initialize(ShapeType type, int64_t min_size, int64_t max_size, ShapePolicy policy,
    std::vector<int>& dimensions, std::vector<int>& indices, double value)
{
    m_policy = policy;
    
    std::vector<int64_t> gears;
    GenerateGears(min_size, max_size, policy, gears);

    if (type == ShapeType::BATCHSIZE) {
        m_bs_strategy->InitializeCore(gears, dimensions.front(), indices, value);
        handle_batchsize = true;
    } else {
        m_seq_strategy->InitializeCore(gears, dimensions, indices, value);
        handle_sequence = true;
    }
    initialized = true;
}

void NPUShapeHandling::Transform(std::vector<at::Tensor> &inputs, std::vector<std::vector<at::Tensor>> &outputs)
{
    TORCH_CHECK(initialized, "Shape handling is not initialized.");
    std::vector<at::Tensor> seq_output;
    if (handle_sequence && handle_batchsize) {
        m_seq_strategy->Transform(inputs, seq_output);
        m_bs_strategy->Transform(seq_output, outputs);
    } else if (handle_sequence) {
        m_seq_strategy->Transform(inputs, seq_output);
        outputs.push_back(seq_output);
    } else if (handle_batchsize) {
        m_bs_strategy->Transform(inputs, outputs);
    }
}

void NPUShapeHandling::Recover(std::vector<std::vector<at::Tensor>> &inputs, std::vector<at::Tensor> &outputs)
{
    TORCH_CHECK(initialized, "Shape handling is not initialized.");
    if (handle_batchsize) {
        m_bs_strategy->Recover(inputs, outputs);
    } else {
        for (size_t i = 0; i < inputs[0].size(); i++) {
            outputs.push_back(inputs[0][i]);
        }
    }
}

void NPUShapeHandling::GenerateGears(int64_t min_size, int64_t max_size,
    ShapePolicy policy, std::vector<int64_t>& gears)
{
    TORCH_CHECK(min_size <= max_size, "The min_size cannot be greater than the max_size.");
    TORCH_CHECK(min_size >= COMMON_MIN_SIZE, "The min_size should be a positive number.");
    TORCH_CHECK(policy == ShapePolicy::TIMES,
        "Currently, only the TIMES policy supports passing min_size and max_size paramters.");
    gears.push_back(min_size);

    if (min_size == max_size) {
        return;
    }

    int exp = static_cast<int>(std::ceil(std::log2(static_cast<double>(min_size))));
    int64_t gear = static_cast<int64_t>(1) << exp;

    if (gear == min_size) {
        gear <<= 1;
        ++exp;
    }

    while (gear > 0 && gear <= max_size) {
        gears.push_back(gear);
        if (gear > max_size / POLICY_TIMES_FACTOR) {
            break;
        }
        gear <<= 1;
    }

    if (gears.back() != max_size) {
        gears.push_back(max_size);
    }
    return;
}

}  // namespace torch::aot_inductor