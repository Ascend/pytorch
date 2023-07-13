// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/core/npu/SecondaryStreamGuard.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"

namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using torch::autograd::Function;

enum class DropOutStatus{
  DROPOUT_NORMAL = 0,
  DROPOUT_NONE,
  DROPOUT_ALL
};

DropOutStatus get_dropout_status(double keep_prob) {
  if (keep_prob == 0) {
    return DropOutStatus::DROPOUT_ALL;
  }
  if (keep_prob == 1.) {
    return DropOutStatus::DROPOUT_NONE;
  }
  return DropOutStatus::DROPOUT_NORMAL;
}

at::Tensor format_trans(const at::Tensor &at_tensor) {
  return at_tensor.defined() ? NPUNativeFunctions::npu_format_cast(at_tensor, ACL_FORMAT_ND) : at_tensor;
}

at::Tensor dropout_gen_mask_impl(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
    const int64_t offset, const int64_t numels) {
  int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
  c10::TensorOptions options = self.options();
  at::Tensor mask = OpPreparation::ApplyTensorWithoutFormat(at::IntArrayRef{length + 32}, options.dtype(at::kByte));
  at::SmallVector<int64_t, N> offsetList = {0, offset};
  const int64_t seed1 = 0;
  OpCommand cmd;
  cmd.Name("StatelessDropOutGenMask")
      .Input(at::IntArrayRef{numels})
      .Input(keep_prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Input(seed, at::ScalarType::Int)
      .Input(at::Scalar(seed1), at::ScalarType::Int)
      .Input(offsetList, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Output(mask)
      .Run();
  return mask;
}

at::Tensor dropout_gen_mask_dispatch(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
    const int64_t offset, const int64_t numels, const bool gen_mask_parallel, const bool sync) {
  at::Tensor mask;

  if (gen_mask_parallel) {
    auto original_stream = c10_npu::getCurrentNPUStream();
    {
      // During the life cycle of this raii instance, the calcu stream is set as the
      // secondary stream, and tasks are distributed to the secondary stream. At the
      // same time, according to the one-stream-one-pool principle, memory is also
      // alloced from the pool of the secondary stream.
      c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
      mask = dropout_gen_mask_impl(self, keep_prob, seed, offset, numels);
      if (sync) {
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(original_stream));
      }
    }
    // When tasks on multiple streams read and write the same block of memory,
    // recordStream needs to be called to ensure the correctness of memory reuse.
    c10_npu::NPUCachingAllocator::recordStream(mask.storage().data_ptr(), original_stream);
  } else {
    mask = dropout_gen_mask_impl(self, keep_prob, seed, offset, numels);
  }
  return mask;
}

at::Tensor dropout_gen_mask(const at::Tensor &self, double keep_prob, int64_t head_num, bool gen_mask_parallel,
    bool sync, int64_t &seed, int64_t &offset, int64_t &numels) {
  at::Tensor drop_mask;
  numels = self.size(0) * head_num * self.size(1) * self.size(1); // [B,N,S,S]
  int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
  length += 32;
  if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
    const auto gen = at_npu::detail::getDefaultNPUGenerator();
    auto pair = at::check_generator<NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
    seed = pair.first;
    offset = pair.second;
    drop_mask = dropout_gen_mask_dispatch(self, at::Scalar(keep_prob), at::Scalar(seed), offset, numels, gen_mask_parallel, sync);
  } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
    drop_mask = at::zeros(at::IntArrayRef{length}, self.options().dtype(at::kByte));
  }
  return drop_mask;
}

std::vector<at::Tensor> npu_flash_attention_backward(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &dy,
    int64_t head_num,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &drop_mask,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_in,
    const c10::optional<at::Tensor> &attention_in,
    double scale_value,
    double keep_prob,
    int64_t pre_tockens,
    int64_t next_tockens)
{
  bool is_flash = true;
  const at::Tensor &pse_const = pse.value_or(at::Tensor());
  const at::Tensor &drop_mask_const = drop_mask.value_or(at::Tensor());
  const at::Tensor &padding_mask_const = padding_mask.value_or(at::Tensor());
  const at::Tensor &atten_mask_const = atten_mask.value_or(at::Tensor());
  const at::Tensor &softmax_max_const = softmax_max.value_or(at::Tensor());
  const at::Tensor &softmax_sum_const = softmax_sum.value_or(at::Tensor());
  const at::Tensor &softmax_const = softmax_in.value_or(at::Tensor());
  const at::Tensor &attention_const = attention_in.value_or(at::Tensor());

  at::Tensor format_query = NPUNativeFunctions::npu_format_cast(query, ACL_FORMAT_ND);
  at::Tensor format_key = NPUNativeFunctions::npu_format_cast(key, ACL_FORMAT_ND);
  at::Tensor format_value = NPUNativeFunctions::npu_format_cast(value, ACL_FORMAT_ND);
  at::Tensor format_dy = NPUNativeFunctions::npu_format_cast(dy, ACL_FORMAT_ND);

  at::Tensor format_pse = format_trans(pse_const);
  at::Tensor format_drop_mask = format_trans(drop_mask_const);
  at::Tensor format_padding_mask = format_trans(padding_mask_const);
  at::Tensor format_atten_mask = format_trans(atten_mask_const);
  at::Tensor format_softmax_max = format_trans(softmax_max_const);
  at::Tensor format_softmax_sum = format_trans(softmax_sum_const);
  at::Tensor format_softmax = format_trans(softmax_const);
  at::Tensor format_attention = format_trans(attention_const);
  at::Tensor dtype_atten_mask =
      (format_atten_mask.defined() && format_atten_mask.scalar_type() != query.scalar_type()) ?
      NPUNativeFunctions::npu_dtype_cast(format_atten_mask, query.scalar_type()) : format_atten_mask;

  at::Tensor dq = OpPreparation::ApplyTensorWithoutFormat(format_query);
  at::Tensor dk = OpPreparation::ApplyTensorWithoutFormat(format_key);
  at::Tensor dv = OpPreparation::ApplyTensorWithoutFormat(format_value);

  EXEC_NPU_NO_FORMAT_CHECK_CMD(
      aclnnFlashAttentionScoreGrad, format_query, format_key, format_value, format_dy,
      format_pse, format_drop_mask, format_padding_mask, dtype_atten_mask,
      format_softmax_max, format_softmax_sum, format_softmax, format_attention, scale_value, keep_prob,
      pre_tockens, next_tockens, is_flash, head_num, dq, dk, dv);

  return {dq, dk, dv,
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}

class NPUFlashAttentionFunction : public torch::autograd::Function<NPUFlashAttentionFunction> {
public:
  static std::vector<at::Tensor> forward(
      AutogradContext *ctx, const at::Tensor &query, const at::Tensor &key,
      const at::Tensor &value, int64_t head_num, const c10::optional<at::Tensor> &pse_opt,
      const c10::optional<at::Tensor> &padding_mask_opt, const c10::optional<at::Tensor> &atten_mask_opt,
      double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, bool gen_mask_parallel, bool sync)
  {
    const at::Tensor &pse = pse_opt.value_or(at::Tensor());
    const at::Tensor &padding_mask = padding_mask_opt.value_or(at::Tensor());
    const at::Tensor &atten_mask = atten_mask_opt.value_or(at::Tensor());

    TORCH_CHECK(query.dim() == 3, "The shapes of the input query should be 3-dimensional, but got ", query.dim(), "-dimensional");
    TORCH_CHECK(key.dim() == 3, "The shapes of the input key should be 3-dimensional, but got ", key.dim(), "-dimensional");
    TORCH_CHECK(value.dim() == 3, "The shapes of the input value should be 3-dimensional, but got ", value.dim(), "-dimensional");
    TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1, "The keep_prob value must be in range of [0, 1], but got ", keep_prob);
    at::Tensor attention_score = OpPreparation::ApplyTensorWithoutFormat(query);

    at::Tensor format_query = NPUNativeFunctions::npu_format_cast(query, ACL_FORMAT_ND);
    at::Tensor format_key = NPUNativeFunctions::npu_format_cast(key, ACL_FORMAT_ND);
    at::Tensor format_value = NPUNativeFunctions::npu_format_cast(value, ACL_FORMAT_ND);

    at::Tensor format_pse = format_trans(pse);
    at::Tensor format_padding_mask = format_trans(padding_mask);
    at::Tensor format_atten_mask = format_trans(atten_mask);
    at::Tensor dtype_atten_mask =
      (format_atten_mask.defined() && format_atten_mask.scalar_type() != query.scalar_type()) ?
      NPUNativeFunctions::npu_dtype_cast(format_atten_mask, query.scalar_type()) : format_atten_mask;

    int64_t seed;
    int64_t offset;
    int64_t numels;
    at::Tensor format_drop_mask = dropout_gen_mask(format_query, keep_prob, head_num, gen_mask_parallel, sync,
        seed, offset, numels);

    size_t dim = (query.scalar_type() == at::ScalarType::Float) ? 8 : 16;
    at::Tensor softmax_max = OpPreparation::ApplyTensorWithoutFormat(query,
        {query.size(0), head_num, query.size(1), dim}); // [B, N, S, dim]
    at::Tensor softmax_sum = OpPreparation::ApplyTensorWithoutFormat(query,
        {query.size(0), head_num, query.size(1), dim}); // [B, N, S, dim]

    bool is_flash = true;
    at::Tensor softmax_out;

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFlashAttentionScore, format_query, format_key, format_value,
        format_pse, format_drop_mask, format_padding_mask, dtype_atten_mask,
        scale, keep_prob, pre_tockens, next_tockens, head_num, is_flash,
        softmax_max, softmax_sum, softmax_out, attention_score);

    at::AutoNonVariableTypeMode g;

    ctx->save_for_backward({format_query, format_key, format_value, softmax_max, softmax_sum, softmax_out,
                            format_pse, format_padding_mask, format_atten_mask, attention_score});

    ctx->saved_data["scale"] = scale;
    ctx->saved_data["keep_prob"] = keep_prob;
    ctx->saved_data["pre_tockens"] = pre_tockens;
    ctx->saved_data["next_tockens"] = next_tockens;
    ctx->saved_data["is_flash"] = is_flash;
    ctx->saved_data["head_num"] = head_num;
    ctx->saved_data["gen_mask_parallel"] = gen_mask_parallel;
    ctx->saved_data["sync"] = sync;
    ctx->saved_data["seed"] = seed;
    ctx->saved_data["offset"] = offset;
    ctx->saved_data["numels"] = numels;

    return {attention_score};
  }

  static std::vector<at::Tensor> backward(AutogradContext *ctx, std::vector<at::Tensor> grad_outputs)
  {
    auto scale = ctx->saved_data["scale"].toDouble();
    auto keep_prob = ctx->saved_data["keep_prob"].toDouble();
    auto pre_tockens = ctx->saved_data["pre_tockens"].toInt();
    auto next_tockens = ctx->saved_data["next_tockens"].toInt();
    auto is_flash = ctx->saved_data["is_flash"].toBool();
    auto head_num = ctx->saved_data["head_num"].toInt();
    auto gen_mask_parallel = ctx->saved_data["gen_mask_parallel"].toBool();
    auto sync = ctx->saved_data["sync"].toBool();
    auto seed = ctx->saved_data["seed"].toInt();
    auto offset = ctx->saved_data["offset"].toInt();
    auto numels = ctx->saved_data["numels"].toInt();
    auto saved = ctx->get_saved_variables();

    auto query = saved[0];
    auto key = saved[1];
    auto value = saved[2];
    auto softmax_max = saved[3];
    auto softmax_sum = saved[4];
    auto softmax_out = saved[5];
    auto pse = saved[6];
    auto padding_mask = saved[7];
    auto atten_mask = saved[8];
    auto attention_score = saved[9];

    int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
    length += 32;
    at::Tensor drop_mask;
    if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
      drop_mask = dropout_gen_mask_dispatch(query, at::Scalar(keep_prob), at::Scalar(seed), offset, numels, gen_mask_parallel, sync);
    } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
      drop_mask = at::zeros(at::IntArrayRef{length}, query.options().dtype(at::kByte));
    }

    return npu_flash_attention_backward(query,
        key, value, grad_outputs[0], head_num, pse, drop_mask, padding_mask, atten_mask,
        softmax_max, softmax_sum, softmax_out, attention_score, scale,
        keep_prob, pre_tockens, next_tockens);
  }
};

std::vector<at::Tensor> NPUNativeFunctions::npu_flash_attention_grad(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &dy,
    int64_t head_num,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_in,
    const c10::optional<at::Tensor> &attention_in,
    double scale_value,
    double keep_prob,
    int64_t pre_tockens,
    int64_t next_tockens,
    bool gen_mask_parallel,
    bool sync)
{
  TORCH_CHECK(query.dim() == 3, "The shapes of the input query should be 3-dimensional, but got ", query.dim(), "-dimensional");
  TORCH_CHECK(key.dim() == 3, "The shapes of the input key should be 3-dimensional, but got ", key.dim(), "-dimensional");
  TORCH_CHECK(value.dim() == 3, "The shapes of the input value should be 3-dimensional, but got ", value.dim(), "-dimensional");
  TORCH_CHECK(dy.dim() == 3, "The shapes of the input dy should be 3-dimensional, but got ", dy.dim(), "-dimensional");
  TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1, "The keep_prob value must be in range of [0, 1], but got ", keep_prob);
  int64_t seed;
  int64_t offset;
  int64_t numels;
  at::Tensor drop_mask = dropout_gen_mask(query, keep_prob, head_num, gen_mask_parallel, sync,
      seed, offset, numels);

  return npu_flash_attention_backward(query,
      key, value, dy, head_num, pse, drop_mask, padding_mask, atten_mask,
      softmax_max, softmax_sum, softmax_in, attention_in, scale_value,
      keep_prob, pre_tockens, next_tockens);
}

std::vector<at::Tensor> NPUNativeFunctions::npu_flash_attention(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, int64_t head_num,
    const c10::optional<at::Tensor> &pse, const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, bool gen_mask_parallel, bool sync)
{
  return NPUFlashAttentionFunction::apply(query, key, value, head_num, pse, padding_mask,
      atten_mask, scale, keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync);
}
} // namespace native
} // namespace at_npu
