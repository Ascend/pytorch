#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu{
namespace native{
    
using namespace at_npu::native;

at::Tensor NPUNativeFunctions::npu_hcom_allgather(const at::Tensor& self,
    int64_t rank_size,
    c10::string_view group,
    double alpha,
    double beta,
    c10::optional<int64_t> hccl_comm) {
    AT_ERROR("The output of allgather needs to be specified according to external parameters,"
             "so npu_hcom_allgather is not implemented.");
}

at::Tensor& NPUNativeFunctions::npu_hcom_allgather_out(const at::Tensor& self,
    int64_t rank_size,
    c10::string_view group,
    double alpha,
    double beta,
    c10::optional<int64_t> hccl_comm,
    at::Tensor& out) {
  c10::SmallVector<int64_t, N> out_tensor_target_shape{out.sizes()};

  auto tmp_out = out.view(-1).clone();
  OpCommand cmd;
  cmd.Name("HcomAllGather")
      .Input(self)
      .Attr("rank_size", rank_size)
      .Attr<std::string>("group", std::string(group).data())
      .Attr("alpha", static_cast<float>(alpha))
      .Attr("beta", static_cast<float>(beta));
  if (hccl_comm.has_value()) {
      cmd.Attr("comm", hccl_comm.value());
  }
  cmd.Output(tmp_out)
      .Run();

  // Torch.allgather and cann.HcomAllGather have different gather methods,
  // which will lead to inconsistent operator output shape.
  // Therefore, it is necessary to refresh the shape again after HcomAllGather calculation.
  out = tmp_out.view(out_tensor_target_shape).clone();
  return out;
}
} // namespace native
} // namespace at_npu