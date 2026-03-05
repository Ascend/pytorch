#include <c10/util/intrusive_ptr.h>
#include <torch/library.h>
#include "torch_npu/csrc/distributed/ProcessGroupHCCL.hpp"

namespace npu_custom_dist {
namespace ops {
namespace {

TORCH_LIBRARY(npu_custom_dist, m) {

    m.class_<c10d_npu::ProcessGroupHCCL>("ProcessGroupHCCL").def(torch::init<int64_t, int64_t>());
    m.def("wrap_reduce_scatter_base_uneven_inner(Tensor output_tensor, Tensor input_tensor, int[] input_split_sizes, __torch__.torch.classes.npu_custom_dist.ProcessGroupHCCL process_group_hccl, __torch__.torch.classes.c10d.ReduceOp reduce_op, int timeout) -> __torch__.torch.classes.c10d.Work");
    m.def("wrap_allgather_base_uneven_inner(Tensor output_tensor, Tensor input_tensor, int[] output_split_sizes, __torch__.torch.classes.npu_custom_dist.ProcessGroupHCCL process_group_hccl, int timeout) -> __torch__.torch.classes.c10d.Work");
    m.def("wrap_batch_isend_irecv_inner(Tensor[] tensors, str[] op_type, int[] remote_rank_list, __torch__.torch.classes.npu_custom_dist.ProcessGroupHCCL process_group_hccl) -> __torch__.torch.classes.c10d.Work");
}

} // anonymous namespace

namespace {
c10::intrusive_ptr<c10d::Work> wrap_reduce_scatter_base_uneven_inner(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    std::vector<int64_t> input_split_sizes,
    c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL> process_group_hccl,
    c10::intrusive_ptr<c10d::ReduceOp> reduce_op,
    int64_t timeout)
{
    auto work = process_group_hccl->_reduce_scatter_base_uneven_inner(
        output_tensor,
        input_tensor,
        input_split_sizes,
        c10d::ReduceScatterOptions{
            *reduce_op.get(),
            std::chrono::milliseconds(timeout)
        });
    return work;
}

c10::intrusive_ptr<c10d::Work> wrap_allgather_base_uneven_inner(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    std::vector<int64_t> output_split_sizes,
    c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL> process_group_hccl,
    int64_t timeout)
{
    auto work = process_group_hccl->_allgather_base_uneven_inner(
        output_tensor,
        input_tensor,
        output_split_sizes,
        c10d::AllgatherOptions{std::chrono::milliseconds(timeout)});
    return work;
}

c10::intrusive_ptr<c10d::Work> wrap_batch_isend_irecv_inner(
    at::TensorList tensors,
    std::vector<std::string> op_type,
    std::vector<int64_t> remote_rank_list,
    c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL> process_group_hccl)
{
    auto tensor_vec = tensors.vec();
    auto work = process_group_hccl->batch_isend_irecv_inner(
        op_type,
        tensor_vec,
        remote_rank_list);
    return work;
}

TORCH_LIBRARY_IMPL(npu_custom_dist, PrivateUse1, m) {
    m.impl("wrap_reduce_scatter_base_uneven_inner", TORCH_FN(wrap_reduce_scatter_base_uneven_inner));
    m.impl("wrap_allgather_base_uneven_inner", TORCH_FN(wrap_allgather_base_uneven_inner));
    m.impl("wrap_batch_isend_irecv_inner", TORCH_FN(wrap_batch_isend_irecv_inner));
}
} // anonymous namespace
} // namespace ops
} // namespace npu_custom_dist
