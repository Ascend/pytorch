#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

auto check_and_trans_dim(const at::Tensor& self, at::IntArrayRef dim) {
  std::vector<int64_t> result_dim;
  auto self_dim = self.dim();
  for (int64_t i = 0; i < dim.size(); i++) {
      int64_t tmp_dim = c10::maybe_wrap_dim(dim[i], self_dim);
      result_dim.emplace_back(tmp_dim);
  }
  std::sort(result_dim.begin(), result_dim.end());
  return result_dim;
}

auto get_result_names(const at::Tensor& self, at::IntArrayRef dim, bool keepdim){
  auto names = self.names();
  std::vector<at::Dimname> result_names;
  for (int64_t i = 0; i < names.size(); i++) {
    result_names.emplace_back(names[i]);
  }
  if (!keepdim) {
    for (int64_t i = dim.size() - 1; i >= 0; i--) {
      int64_t need_remove_dim = dim[i];
      result_names.erase(result_names.begin() + need_remove_dim);
    }
  }
  return result_names;
}

at::Tensor& var_after_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mean_broadcast,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  bool if_std = false;
  OpCommand cmd;
  cmd.Name("ReduceStdV2Update")
      .Input(self)
      .Input(mean_broadcast)
      .Output(result)
      .Attr("dim", dim)
      .Attr("if_std", if_std)
      .Attr("unbiased", unbiased)
      .Attr("keepdim", keepdim)
      .Run();
  return result;
}

std::tuple<at::Tensor&, at::Tensor&> var_mean_compute(
    at::Tensor& variance,
    at::Tensor& mean,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  auto mean_output_size_keepdim = var_npu_output_size(self, dim, true);
  auto mean_output_size_not_keepdim = var_npu_output_size(self, dim, false);
  mean = at::mean(self, dim, false);
  mean.resize_(mean_output_size_keepdim);
  at::Tensor mean_broadcast = NPUNativeFunctions::npu_broadcast(mean, self.sizes());
  if (!keepdim) {
    mean.resize_(mean_output_size_not_keepdim);
  }
  var_after_out_nocheck(variance, self, mean_broadcast, dim, unbiased, keepdim);
  return std::tuple<at::Tensor&, at::Tensor&>(variance, mean);
}

std::tuple<at::Tensor&, at::Tensor&> var_mean_out_nocheck(
    at::Tensor& variance,
    at::Tensor& mean,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  c10::SmallVector<int64_t, N> dim_now =
      dim.empty() ? CalcuOpUtil::GetDimlistForTensor(self) : c10::SmallVector<int64_t, N>(dim);
  auto mean_output_size_keepdim = var_npu_output_size(self, dim_now, true);
  auto mean_output_size_not_keepdim = var_npu_output_size(self, dim_now, false);
  auto ori_type = self.scalar_type();
  TORCH_CHECK((ori_type == c10::ScalarType::Half || ori_type == c10::ScalarType::Float),
      "Var Mean only support float16 or float32 type.");
  TORCH_CHECK((variance.scalar_type() == mean.scalar_type() && variance.scalar_type() == ori_type),
      "mean's type and variance' type must be equal to input's type.");
  var_mean_compute(variance, mean, self, dim_now, unbiased, keepdim);

  return std::tuple<at::Tensor&, at::Tensor&>(variance, mean);
}

at::Tensor& NPUNativeFunctions::var_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim,
    at::Tensor& result) {
  auto unbiased = !(correction.has_value() && correction.value().toInt() == 0);
  // check and trans dim
  auto dim_now = check_and_trans_dim(self, dim.value_or(at::IntArrayRef{}));
  auto output_size = var_npu_output_size(self, dim_now, keepdim);
  at::Tensor mean = OpPreparation::apply_tensor(self, output_size);

  OpPreparation::CheckOut(
      {self},
      result,
      self,
      output_size);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    var_mean_out_nocheck(contiguous_result, mean, self, dim_now, unbiased, keepdim);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    var_mean_out_nocheck(result, mean, self, dim_now, unbiased, keepdim);
  }
  return result;
}

at::Tensor& NPUNativeFunctions::var_out(
    const at::Tensor& self,
    at::DimnameList dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim,
    at::Tensor& result) {
  return at::var_out(result, self, dimnames_to_positions(self, dim), correction, keepdim);
}

at::Tensor& NPUNativeFunctions::var_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  return at::var_out(result, self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

at::Tensor& NPUNativeFunctions::var_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  return at::var_out(result, self, dimnames_to_positions(self, dim), c10::make_optional<c10::Scalar>(unbiased ? 1 : 0),
      keepdim);
}

at::Tensor NPUNativeFunctions::var(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  auto unbiased = !(correction.has_value() && correction.value().toInt() == 0);
  auto dim_now = check_and_trans_dim(self, dim.value_or(at::IntArrayRef{}));
  auto output_size = var_npu_output_size(self, dim_now, keepdim);

  at::Tensor variance = OpPreparation::apply_tensor(self, output_size);
  at::Tensor mean = OpPreparation::apply_tensor(self, output_size);
  var_mean_out_nocheck(variance, mean, self, dim_now, unbiased, keepdim);

  return variance;
}

at::Tensor NPUNativeFunctions::var(
    const at::Tensor& self,
    at::DimnameList dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  return at::var(self, dimnames_to_positions(self, dim), correction, keepdim);
}

at::Tensor NPUNativeFunctions::var(const at::Tensor& self, bool unbiased) {
  c10::SmallVector<int64_t, N> dim = CalcuOpUtil::GetDimlistForTensor(self);
  return at::var(self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), false);
}

at::Tensor NPUNativeFunctions::var(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  return at::var(self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

at::Tensor NPUNativeFunctions::var(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return at::var(self, dimnames_to_positions(self, dim),
      c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::var_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  auto unbiased = !(correction.has_value() && correction.value().toInt() == 0);
  auto dim_now = check_and_trans_dim(self, dim.value_or(at::IntArrayRef{}));
  auto output_size = var_npu_output_size(self, dim_now, keepdim);

  at::Tensor variance = OpPreparation::apply_tensor(self, output_size);
  at::Tensor mean = OpPreparation::apply_tensor(self, output_size);
  var_mean_out_nocheck(variance, mean, self, dim_now, unbiased, keepdim);

  return std::tuple<at::Tensor, at::Tensor>(variance, mean);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::var_mean(
    const at::Tensor& self,
    at::DimnameList dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  return at::var_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::var_mean(const at::Tensor& self, bool unbiased) {
  c10::SmallVector<int64_t, SIZE> dim = CalcuOpUtil::GetDimlistForTensor(self);
  return at::var_mean(self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), false);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::var_mean(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return at::var_mean(self, dimnames_to_positions(self, dim),
      c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::var_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  return at::var_mean(self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

} // namespace native
} // namespace at_npu
