#include <ATen/Functions.h>
#include <c10/util/intrusive_ptr.h>
#include <optional>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/distributed/ProcessGroupHCCL.hpp"

namespace npu_custom_dist {
namespace ops {
namespace {

TORCH_LIBRARY(npu_custom_dist, m) {
  m.def(
      "functional_all_reduce(Tensor input, str reduce_op, str group_name, float? premul_factor=None) -> Tensor",
      {at::Tag::pt2_compliant_tag});
    m.class_<c10d_npu::ProcessGroupHCCL>("ProcessGroupHCCL").def(torch::init<int64_t, int64_t>());
    m.def("wrap_reduce_scatter_base_uneven_inner(Tensor output_tensor, Tensor input_tensor, int[] input_split_sizes, __torch__.torch.classes.npu_custom_dist.ProcessGroupHCCL process_group_hccl, __torch__.torch.classes.c10d.ReduceOp reduce_op, int timeout) -> __torch__.torch.classes.c10d.Work");
    m.def("wrap_allgather_base_uneven_inner(Tensor output_tensor, Tensor input_tensor, int[] output_split_sizes, __torch__.torch.classes.npu_custom_dist.ProcessGroupHCCL process_group_hccl, int timeout) -> __torch__.torch.classes.c10d.Work");
    m.def("wrap_alltoallvc_inner(Tensor output_tensor, Tensor input_tensor, int[] send_count_matrix, __torch__.torch.classes.npu_custom_dist.ProcessGroupHCCL process_group_hccl, int timeout) -> __torch__.torch.classes.c10d.Work");
    m.def("wrap_batch_isend_irecv_inner(Tensor[] tensors, str[] op_type, int[] remote_rank_list, __torch__.torch.classes.npu_custom_dist.ProcessGroupHCCL process_group_hccl) -> __torch__.torch.classes.c10d.Work");
}

} // anonymous namespace

namespace {
c10d::ReduceOp functional_reduce_op(const std::string& name, std::optional<double> premul_factor) {
  if (name == "premul_sum") {
    TORCH_CHECK(premul_factor.has_value(), "functional_all_reduce: premul_sum requires premul_factor");
    return c10d::makeNCCLPreMulSum(*premul_factor);
  }
  TORCH_CHECK(!premul_factor.has_value(), "functional_all_reduce: premul_factor is only valid for premul_sum");
  if (name == "sum") {
    return c10d::ReduceOp::SUM;
  }
  if (name == "avg") {
    return c10d::ReduceOp::AVG;
  }
  if (name == "min") {
    return c10d::ReduceOp::MIN;
  }
  if (name == "max") {
    return c10d::ReduceOp::MAX;
  }
  TORCH_CHECK(false, "functional_all_reduce: reduction '", name, "' is outside the CCU_MS support matrix");
}

at::Tensor functional_all_reduce(
    const at::Tensor& input,
    const std::string& reduce_op,
    const std::string& group_name,
    std::optional<double> premul_factor) {
  c10d::AllreduceOptions opts;
  opts.reduceOp = functional_reduce_op(reduce_op, premul_factor);
  c10_npu::NPUGuard guard(input.device());
  auto group = c10d::resolve_process_group(group_name);
  auto backend = group->getBackend(c10::DeviceType::PrivateUse1);
  // Only an unwrapped HCCL backend exposes this independent entry.
  auto* hccl = dynamic_cast<c10d_npu::ProcessGroupHCCL*>(backend.get());
  TORCH_CHECK(
      hccl != nullptr,
      "functional_all_reduce requires an unwrapped ProcessGroupHCCL NPU backend; "
      "ProcessGroupWrapper (TORCH_DISTRIBUTED_DEBUG=DETAIL) is not supported");
  auto preparedInput = c10d_npu::ProcessGroupHCCL::prepareFunctionalAllReduceInput(input);
  TORCH_CHECK(
      !input.is_complex() || opts.reduceOp == c10d::ReduceOp::SUM || opts.reduceOp == c10d::ReduceOp::AVG,
      "functional_all_reduce: complex input only supports SUM/AVG",
      DIST_ERROR(ErrCode::NOT_SUPPORT));
  auto output = at::empty(preparedInput.sizes(), preparedInput.options());
  auto work = hccl->allreduce_out(preparedInput, output, opts);
  auto result = input.is_complex() ? at::view_as_complex(output) : output;
  c10d::register_work(result, work);
  return result;
}

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

c10::intrusive_ptr<c10d::Work> wrap_alltoallvc_inner(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    std::vector<int64_t> send_count_matrix,
    c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL> process_group_hccl,
    int64_t timeout)
{
    auto work = process_group_hccl->_alltoallvc_inner(
        output_tensor,
        input_tensor,
        send_count_matrix,
        c10d::AllToAllOptions{std::chrono::milliseconds(timeout)});
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
  m.impl("functional_all_reduce", TORCH_FN(functional_all_reduce));
    m.impl("wrap_reduce_scatter_base_uneven_inner", TORCH_FN(wrap_reduce_scatter_base_uneven_inner));
    m.impl("wrap_allgather_base_uneven_inner", TORCH_FN(wrap_allgather_base_uneven_inner));
    m.impl("wrap_alltoallvc_inner", TORCH_FN(wrap_alltoallvc_inner));
    m.impl("wrap_batch_isend_irecv_inner", TORCH_FN(wrap_batch_isend_irecv_inner));
}
} // anonymous namespace
} // namespace ops
} // namespace npu_custom_dist
