#ifndef FLOP_COUNTER_H
#define FLOP_COUNTER_H

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

class FlopCounter {
public:
    FlopCounter() = default;
    ~FlopCounter() = default;

    static int64_t mm_flop(const at::Tensor &self, const at::Tensor &mat2);
    static int64_t all_gather_mm_flop(const at::Tensor &self, const at::Tensor &mat2, int64_t world_size, int64_t gather_index);
    static int64_t addmm_flop(const at::Tensor &mat1, const at::Tensor &mat2);
    static int64_t bmm_flop(const at::Tensor &self, const at::Tensor &mat2);
    static int64_t baddbmm_flop(const at::Tensor &batch1, const at::Tensor &batch2);
    static int64_t conv_flop(const at::Tensor &input, const at::Tensor &weight, bool transposed, at::Tensor output);
    static int64_t conv_backward_flop(const at::Tensor &grad_output, const at::Tensor &input,
        const at::Tensor &weight, bool transposed, ::std::array<bool, 3> output_mask,
        const at::Tensor &gradInput, const at::Tensor &gradeWeight);
    static int64_t flash_attention_forward_flop(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        int64_t head_num, const std::string &input_layout, const c10::OptionalIntArrayRef &actual_seq_qlen,
        const c10::OptionalIntArrayRef &actual_seq_kvlen);
    static int64_t flash_attention_backward_flop(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        const at::Tensor &dy, int64_t head_num, const std::string &input_layout,
        const c10::OptionalIntArrayRef &actual_seq_qlen,
        const c10::OptionalIntArrayRef &actual_seq_kvlen);
};

#endif
