#include "torch_npu/csrc/flopcount/FlopCounter.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

int64_t FlopCounter::mm_flop(const at::Tensor &tensor1, const at::Tensor &tensor2)
{
    // Count flops for matmul.
    // Inputs contains the shapes of two matrices.
    auto dim_tensor1 = tensor1.dim();
    auto dim_tensor2 = tensor2.dim();
    TORCH_CHECK(dim_tensor1 > 0 && dim_tensor2 > 0, "matmul got error dimentions: ", "(", dim_tensor1, ", ",
                dim_tensor2, ")", OPS_ERROR(ErrCode::PARAM));
    // A(x1, m, k1) and B(x2, k2, n)
    // Get x1 and x2's infer sizes
    auto x1_size = dim_tensor1 > 2 ? dim_tensor1 - 2 : 0;
    auto x2_size = dim_tensor2 > 2 ? dim_tensor2 - 2 : 0;
    at::IntArrayRef x1_sizes(tensor1.sizes().data(), x1_size);
    at::IntArrayRef x2_sizes(tensor2.sizes().data(), x2_size);
    std::vector<int64_t> output_size = at::infer_size(x1_sizes, x2_sizes);
    // Get m
    if (dim_tensor1 >= 2) {
        output_size.push_back(tensor1.size(-2));
    }
    // Get n
    if (dim_tensor2 >= 2) {
        output_size.push_back(tensor2.size(-1));
    }
    // Get k1 and k2
    int64_t k = tensor1.size(-1);
    // Compute
    int64_t flop = 2 * k;
    for (const auto& elem : output_size) {
        flop *= elem;
    }
    return flop;
}

int64_t FlopCounter::all_gather_mm_flop(const at::Tensor &self, const at::Tensor &mat2, int64_t world_size, int64_t gather_index)
{
    int64_t mm_flops = mm_flop(self, mat2);
    return gather_index == 0 ? mm_flops * world_size : mm_flops;
}

int64_t FlopCounter::addmm_flop(const at::Tensor &mat1, const at::Tensor &mat2)
{
    return mm_flop(mat1, mat2);
}

int64_t FlopCounter::bmm_flop(const at::Tensor &self, const at::Tensor &mat2)
{
    // Count flops for the bmm operation.
    // Inputs should be a list of length 2.
    // Inputs contains the shapes of two tensor.
    int64_t b = self.size(0);
    int64_t m = self.size(1);
    int64_t k = self.size(2);
    int64_t b2 = mat2.size(0);
    int64_t k2 = mat2.size(1);
    int64_t n = mat2.size(2);
    TORCH_CHECK(b == b2 && k == k2, "The tensor dimension is incorrect", PTA_ERROR(ErrCode::VALUE));
    return b * m * n * 2 * k;
}

int64_t FlopCounter::baddbmm_flop(const at::Tensor &batch1, const at::Tensor &batch2)
{
    return bmm_flop(batch1, batch2);
}

int64_t conv_flop_count(std::vector<int64_t> x_shape, std::vector<int64_t> w_shape, std::vector<int64_t> out_shape, bool transposed)
{
    // Count flops for convolution. Note only multiplication is
    // counted. Computation for bias are ignored.
    // Flops for a transposed convolution are calculated as
    // flops = (x_shape[2:] * prod(w_shape) * batch_size).
    // Args:
    //     x_shape (std::vector<int64_t>): The input shape before convolution.
    //     w_shape (std::vector<int64_t>): The filter shape.
    //     out_shape (std::vector<int64_t>): The output shape after convolution.
    //     transposed (bool): is the convolution transposed
    // Returns:
    //     int: the number of flops
    int64_t batch_size = x_shape[0];
    std::vector<int64_t> conv_shape = transposed ? out_shape : std::vector<int64_t>(out_shape.begin() + 2, out_shape.end());
    int64_t c_out = w_shape[0];
    int64_t c_in = w_shape[1];
    int64_t filter_size = std::accumulate(w_shape.begin() + 2, w_shape.end(), 1, std::multiplies<int>());

    int64_t flop = std::accumulate(conv_shape.begin(), conv_shape.end(), 1, std::multiplies<int>()) * filter_size * batch_size * c_out * c_in * 2;
    return flop;
}

int64_t FlopCounter::conv_flop(const at::Tensor &input, const at::Tensor &weight, bool transposed, at::Tensor output)
{
    // Count flops for convolution.
    std::vector<int64_t> out_shape(output.sizes().begin(), output.sizes().end());
    std::vector<int64_t> x_shape(input.sizes().begin(), input.sizes().end());
    std::vector<int64_t> w_shape(weight.sizes().begin(), weight.sizes().end());

    return conv_flop_count(x_shape, w_shape, out_shape, transposed);
}

std::vector<int64_t> t(std::vector<int64_t> shape)
{
    return {shape[1], shape[0], shape[2], shape[3]};
}

int64_t FlopCounter::conv_backward_flop(const at::Tensor &grad_output, const at::Tensor &input,
    const at::Tensor &weight, bool transposed, ::std::array<bool, 3> output_mask,
    const at::Tensor &gradInput, const at::Tensor &gradeWeight)
{
    std::vector<int64_t> grad_output_shape(grad_output.sizes().begin(), grad_output.sizes().end());
    std::vector<int64_t> w_shape(weight.sizes().begin(), weight.sizes().end());
    std::vector<int64_t> input_shape(input.sizes().begin(), input.sizes().end());

    int64_t flop_count = 0;

    if (output_mask[0]) {
        std::vector<int64_t> grad_input_shape(gradInput.sizes().begin(), gradInput.sizes().end());
        flop_count += conv_flop_count(grad_output_shape, w_shape, grad_input_shape, !transposed);
    }

    if (output_mask[1]) {
        std::vector<int64_t> grad_weight_shape(gradeWeight.sizes().begin(), gradeWeight.sizes().end());
        if (transposed) {
            flop_count += conv_flop_count(t(grad_output_shape), t(input_shape), t(grad_weight_shape), false);
        } else {
            flop_count += conv_flop_count(t(input_shape), t(grad_output_shape), t(grad_weight_shape), false);
        }
    }

    return flop_count;
}

std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>> _unpack_flash_attention_nested_shapes(std::vector<int64_t> query,
    std::vector<int64_t> key, std::vector<int64_t> value, int64_t head_num, std::vector<int64_t> grad_out,
    c10::ArrayRef<int64_t> cum_seq_q, c10::ArrayRef<int64_t> cum_seq_k, std::string input_layer_str)
{
    // Given inputs to a flash_attention_(forward|backward) kernel, this will handle behavior for
    // GQA and MQA and TND

    // for GQA and MQA, the dim 2 or 3 of kv should equal to q
    // for general, shape should view to [B, N, S, D]
    TORCH_CHECK(head_num != 0, "Divisor head_num may be 0, please check it.")
    std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>> result;
    int64_t q_1 = query[1];
    int64_t q_2 = query[2];
    int64_t k_1 = key[1];
    int64_t k_2 = key[2];
    int64_t v_1 = value[1];
    int64_t v_2 = value[2];

    // for GQA and MQA
    if (input_layer_str == "SBH" || input_layer_str == "BSH" || input_layer_str == "BSND") {
        if (q_2 != k_2 && q_2 != v_2) {
            k_2 = q_2;
            v_2 = q_2;
        }
    } else {
        if (q_1 != k_1 && q_1 != v_1) {
            k_1 = q_1;
            v_1 = q_1;
        }
    }

    if (input_layer_str == "BSH") {
        std::vector<int64_t> new_query_shape = {query[0], head_num, q_1, q_2 / head_num};
        std::vector<int64_t> new_key_shape = {key[0], head_num, k_1, k_2 / head_num};
        std::vector<int64_t> new_value_shape = {value[0], head_num, v_1, v_2 / head_num};
        std::vector<int64_t> new_grad_out_shape;
        if (!grad_out.empty()) {
            new_grad_out_shape = new_query_shape;
        }
        result.emplace_back(new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape);
    } else if (input_layer_str == "SBH") {
        std::vector<int64_t> new_query_shape = {q_1, head_num, query[0], q_2 / head_num};
        std::vector<int64_t> new_key_shape = {k_1, head_num, key[0], k_2 / head_num};
        std::vector<int64_t> new_value_shape = {v_1, head_num, value[0], v_2 / head_num};
        std::vector<int64_t> new_grad_out_shape;
        if (!grad_out.empty()) {
            new_grad_out_shape = new_query_shape;
        }
        result.emplace_back(new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape);
    } else if (input_layer_str == "BNSD") {
        std::vector<int64_t> new_grad_out_shape;
        if (!grad_out.empty()) {
            new_grad_out_shape = query;
        }
        result.emplace_back(query, key, value, new_grad_out_shape);
    } else if (input_layer_str == "BSND") {
        std::vector<int64_t> new_query_shape = {query[0], q_2, q_1, query[3]};
        std::vector<int64_t> new_key_shape = {key[0], k_2, k_1, key[3]};
        std::vector<int64_t> new_value_shape = {value[0], v_2, v_1, value[3]};
        std::vector<int64_t> new_grad_out_shape;
        if (!grad_out.empty()) {
            new_grad_out_shape = new_query_shape;
        }
        result.emplace_back(new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape);
    } else if (input_layer_str == "TND") {
        TORCH_CHECK(!cum_seq_q.empty(), "The actual_seq_qlen should not be empty when TND");
        TORCH_CHECK(!cum_seq_k.empty(), "The actual_seq_kvlen should not be empty when TND");
        TORCH_CHECK(cum_seq_q.size() == cum_seq_k.size(), "The size of actual_seq_qlen should be equal to actual_seq_kvlen when TND");

        size_t sizeValue = cum_seq_q.size();
        TORCH_CHECK(sizeValue <= static_cast<size_t>(std::numeric_limits<int64_t>::max()), "cum_seq_q.size() is too large to be represented as an int64_t", OPS_ERROR(ErrCode::PARAM));
        int64_t b = static_cast<int64_t>(sizeValue);
        TORCH_CHECK(b != 0, "Divisor b may be 0, please check it.")
        std::vector<int64_t> new_query_shape = {b, q_1, query[0] / b, q_2};
        std::vector<int64_t> new_key_shape = {b, k_1, key[0] / b, k_2};
        std::vector<int64_t> new_value_shape = {b, v_1, value[0] / b, v_2};
        std::vector<int64_t> new_grad_out_shape;
        if (!grad_out.empty()) {
            new_grad_out_shape = new_query_shape;
        }
        result.emplace_back(new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape);
    }

    return result;
}

inline int64_t safe_multiply(const std::vector<int64_t>& dims)
{
    int64_t product = 1;
    for (auto dim : dims) {
        if (dim == 0) {
            return 0;
        }
        TORCH_CHECK(product <= INT64_MAX / dim, "Integer overflow in multiply.", OPS_ERROR(ErrCode::PARAM))
        product *= dim;
    }
    return product;
}

inline int64_t safe_sum(const std::initializer_list<int64_t>& values)
{
    int64_t sum = 0;
    for (auto val : values) {
        TORCH_CHECK(sum <= INT64_MAX - val, "Integer overflow in sum.", OPS_ERROR(ErrCode::PARAM));
        sum += val;
    }
    return sum;
}

int64_t sdpa_flop_count(const std::vector<int64_t> query_shape, const std::vector<int64_t> key_shape, const std::vector<int64_t> value_shape)
{
    int64_t b;
    int64_t h;
    int64_t s_q;
    int64_t d_q;
    int64_t _b2;
    int64_t _h2;
    int64_t s_k;
    int64_t _d2;
    int64_t _b3;
    int64_t _h3;
    int64_t _s3;
    int64_t d_v;

    b = query_shape[0];
    h = query_shape[1];
    s_q = query_shape[2];
    d_q = query_shape[3];

    _b2 = key_shape[0];
    _h2 = key_shape[1];
    s_k = key_shape[2];
    _d2 = key_shape[3];

    _b3 = value_shape[0];
    _h3 = value_shape[1];
    _s3 = value_shape[2];
    d_v = value_shape[3];

    TORCH_CHECK(b == _b2 && b == _b3, "the dim of 0 is not equal between q and kv");
    TORCH_CHECK(h == _h2 && h == _h3, "the dim of 1 is not equal between q and kv");
    TORCH_CHECK(s_k == _s3, "the dim of 2 is not equal between k and v");
    TORCH_CHECK(d_q == _d2, "the dim of 3 is not equal between q and k");

    int64_t total_flops = safe_sum({
        safe_multiply({b, h, s_q, d_q, s_k, 2}), // q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
        safe_multiply({b, h, s_q, s_k, d_v, 2})  // scores: [b, h, s_q, s_k] @ v: [b, h, s_k, d_v] -> out: [b, h, s_q, d_v]
    });

    return total_flops;
}

int64_t sdpa_backward_flop_count(const std::vector<int64_t> query_shape, const std::vector<int64_t> key_shape, const std::vector<int64_t> value_shape, const std::vector<int64_t> grad_out_shape)
{
    int64_t b;
    int64_t h;
    int64_t s_q;
    int64_t d_q;
    int64_t _b2;
    int64_t _h2;
    int64_t s_k;
    int64_t _d2;
    int64_t _b3;
    int64_t _h3;
    int64_t _s3;
    int64_t d_v;
    int64_t _b4;
    int64_t _h4;
    int64_t _s4;
    int64_t d_4;

    b = query_shape[0];
    h = query_shape[1];
    s_q = query_shape[2];
    d_q = query_shape[3];

    _b2 = key_shape[0];
    _h2 = key_shape[1];
    s_k = key_shape[2];
    _d2 = key_shape[3];

    _b3 = value_shape[0];
    _h3 = value_shape[1];
    _s3 = value_shape[2];
    d_v = value_shape[3];

    _b4 = grad_out_shape[0];
    _h4 = grad_out_shape[1];
    _s4 = grad_out_shape[2];
    d_4 = grad_out_shape[3];

    TORCH_CHECK(b == _b2 && b == _b3 && b == _b4, "the dim of 0 is not equal between qkv and grad");
    TORCH_CHECK(h == _h2 && h == _h3 && h == _h4, "the dim of 1 is not equal between qkv and grad");
    TORCH_CHECK(s_k == _s3, "the dim of 2 is not equal between k and v");
    TORCH_CHECK(s_q == _s4, "the dim of 2 is not equal between q and grad");
    TORCH_CHECK(d_q == _d2, "the dim of 3 is not equal between q and k");
    TORCH_CHECK(d_v == d_4, "the dim of 3 is not equal between v and grad");

    int64_t total_flops = safe_sum({
        safe_multiply({b, h, s_q, d_v, s_k, 2}), // gradOut: [b, h, s_q, d_v] @ v: [b, h, d_v, s_k] -> gradScores: [b, h, s_q, s_k]
        safe_multiply({b, h, s_k, s_q, d_v, 2}), // scores: [b, h, s_k, s_q] @ gradOut: [b, h, s_q, d_v] -> gradV: [b, h, s_k, d_v]
        safe_multiply({b, h, s_q, s_k, d_q, 2}), // gradScores: [b, h, s_q, s_k] @ k: [b, h, s_k, d_q] -> gradQ: [b, h, s_q, d_q]
        safe_multiply({b, h, d_q, s_q, s_k, 2})  // q: [b, h, d_q, s_q] @ gradScores: [b, h, s_q, s_k] -> gradK: [b, h, d_q, s_k]
    });

    return total_flops;
}

int64_t FlopCounter::flash_attention_forward_flop(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, int64_t head_num,
    const std::string &input_layout, const c10::OptionalIntArrayRef &actual_seq_qlen,
    const c10::OptionalIntArrayRef &actual_seq_kvlen)
{
    std::vector<int64_t> grad_out_shape;
    std::vector<int64_t> query_shape(query.sizes().begin(), query.sizes().end());
    std::vector<int64_t> key_shape(key.sizes().begin(), key.sizes().end());
    std::vector<int64_t> value_shape(value.sizes().begin(), value.sizes().end());
    auto ac_seq_qlen_tmp = actual_seq_qlen.value_or(c10::ArrayRef<int64_t>{});
    auto ac_seq_kvlen_tmp = actual_seq_kvlen.value_or(c10::ArrayRef<int64_t>{});

    auto sizes = _unpack_flash_attention_nested_shapes(query_shape, key_shape, value_shape, head_num, grad_out_shape, ac_seq_qlen_tmp, ac_seq_kvlen_tmp, input_layout);

    int64_t total_flops = 0;
    for (const auto& [query_shape_new, key_shape_new, value_shape_new, _] : sizes) {
        total_flops += sdpa_flop_count(query_shape_new, key_shape_new, value_shape_new);
    }
    return total_flops;
}

int64_t FlopCounter::flash_attention_backward_flop(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &dy, int64_t head_num,
    const std::string &input_layout, const c10::OptionalIntArrayRef &actual_seq_qlen,
    const c10::OptionalIntArrayRef &actual_seq_kvlen)
{
    std::vector<int64_t> dy_shape(query.sizes().begin(), query.sizes().end());
    std::vector<int64_t> query_shape(query.sizes().begin(), query.sizes().end());
    std::vector<int64_t> key_shape(key.sizes().begin(), key.sizes().end());
    std::vector<int64_t> value_shape(value.sizes().begin(), value.sizes().end());
    auto ac_seq_qlen_tmp = actual_seq_qlen.value_or(c10::ArrayRef<int64_t>{});
    auto ac_seq_kvlen_tmp = actual_seq_kvlen.value_or(c10::ArrayRef<int64_t>{});

    auto sizes = _unpack_flash_attention_nested_shapes(query_shape, key_shape, value_shape, head_num, dy_shape, ac_seq_qlen_tmp, ac_seq_kvlen_tmp, input_layout);

    int64_t total_flops = 0;
    for (const auto& [query_shape_new, key_shape_new, value_shape_new, grad_out_shape] : sizes) {
        total_flops += sdpa_backward_flop_count(query_shape_new, key_shape_new, value_shape_new, grad_out_shape);
    }
    return total_flops;
}
