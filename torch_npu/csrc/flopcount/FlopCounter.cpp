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
            flop_count += conv_flop_count(t(grad_output_shape), t(input_shape), t(grad_weight_shape), transposed=false);
        } else {
            flop_count += conv_flop_count(t(input_shape), t(grad_output_shape), t(grad_weight_shape), transposed=false);
        }
    }

    return flop_count;
}
