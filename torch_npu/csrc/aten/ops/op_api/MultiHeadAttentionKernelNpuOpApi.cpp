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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using torch::autograd::Function;

enum class DropOutStatus { DROPOUT_NORMAL = 0, DROPOUT_NONE, DROPOUT_ALL };
static const double DROPOUT_NONE_NUM = 1.0;
static const int64_t LENGTH_STRIDE = 32;
static const int64_t DIM_SUPPORT = 3;

static DropOutStatus get_dropout_status(double keep_prob) {
  if (keep_prob == 0) {
    return DropOutStatus::DROPOUT_ALL;
  }
  if (keep_prob == DROPOUT_NONE_NUM) {
    return DropOutStatus::DROPOUT_NONE;
  }
  return DropOutStatus::DROPOUT_NORMAL;
}

static at::Tensor format_trans(const at::Tensor& at_tensor) {
  if (at_tensor.defined()) {
    TORCH_CHECK(at_npu::key::isDeviceTensor(at_tensor), "only npu tensor is supported",
                OPS_ERROR(ErrCode::NOT_SUPPORT));
    return NPUNativeFunctions::npu_format_cast(at_tensor, ACL_FORMAT_ND);
  }
  return at_tensor;
}

static at::Tensor dropout_gen_mask_impl(const at::Tensor& self, const at::Scalar& keep_prob, const at::Scalar& seed,
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

static at::Tensor dropout_gen_mask_dispatch(const at::Tensor& self, const at::Scalar& keep_prob, const at::Scalar& seed,
                                            const int64_t offset, const int64_t numels, const bool gen_mask_parallel,
                                            const bool sync) {
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
  } else {
    mask = dropout_gen_mask_impl(self, keep_prob, seed, offset, numels);
  }
  return mask;
}

static at::Tensor dropout_gen_mask(const at::Tensor& self, double keep_prob, int64_t head_num, std::string input_layout,
                                   bool gen_mask_parallel, bool sync, int64_t& seed, int64_t& offset, int64_t& numels) {
  at::Tensor drop_mask;
  if (input_layout == "BSH") {
    numels = self.size(0) * head_num * self.size(1) * self.size(1);  // [B,N,S,S]
  } else if (input_layout == "SBH") {
    numels = self.size(1) * head_num * self.size(0) * self.size(0);  // [B,N,S,S]
  }
  int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
  length += LENGTH_STRIDE;
  if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
    const auto gen = at_npu::detail::getDefaultNPUGenerator();
    auto pair = at::check_generator<NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
    seed = static_cast<int64_t>(pair.first);
    offset = static_cast<int64_t>(pair.second);
    drop_mask = dropout_gen_mask_dispatch(self, at::Scalar(keep_prob), at::Scalar(seed), offset, numels,
                                          gen_mask_parallel, sync);
  } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
    drop_mask = at::zeros(at::IntArrayRef{length}, self.options().dtype(at::kByte));
  }
  return drop_mask;
}

std::vector<at::Tensor> npu_multi_head_attention_score_backward(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& dy, int64_t head_num,
    const std::string input_layout, const c10::optional<at::Tensor>& pse, const c10::optional<at::Tensor>& drop_mask,
    const c10::optional<at::Tensor>& padding_mask, const c10::optional<at::Tensor>& atten_mask,
    const c10::optional<at::Tensor>& softmax_in, const c10::optional<at::Tensor>& attention_in, double scale_value,
    double keep_prob, int64_t pre_tockens, int64_t next_tockens) {
  double scale = 1;

  const at::Tensor& pse_const = pse.value_or(at::Tensor());
  const at::Tensor& drop_mask_const = drop_mask.value_or(at::Tensor());
  const at::Tensor& padding_mask_const = padding_mask.value_or(at::Tensor());
  const at::Tensor& atten_mask_const = atten_mask.value_or(at::Tensor());
  const at::Tensor& softmax_const = softmax_in.value_or(at::Tensor());
  const at::Tensor& attention_const = attention_in.value_or(at::Tensor());

  at::Tensor format_query = format_trans(query);
  at::Tensor format_query_scaled = at::mul(format_query, at::Scalar(scale));
  at::Tensor format_key = format_trans(key);
  at::Tensor format_value = format_trans(value);
  at::Tensor format_dy = format_trans(dy);

  at::Tensor format_pse = format_trans(pse_const);
  at::Tensor format_drop_mask = format_trans(drop_mask_const);
  at::Tensor format_padding_mask = format_trans(padding_mask_const);
  at::Tensor format_atten_mask = format_trans(atten_mask_const);
  at::Tensor format_softmax = format_trans(softmax_const);
  at::Tensor format_attention = format_trans(attention_const);
  at::Tensor dtype_atten_mask = (format_atten_mask.defined() && format_atten_mask.scalar_type() != query.scalar_type())
                                    ? custom_ops::npu_dtype_cast(format_atten_mask, query.scalar_type())
                                    : format_atten_mask;
  at::Tensor dq = OpPreparation::ApplyTensorWithoutFormat(format_query_scaled);
  at::Tensor dk = OpPreparation::ApplyTensorWithoutFormat(format_key);
  at::Tensor dv = OpPreparation::ApplyTensorWithoutFormat(format_value);
  char* input_layout_ptr = const_cast<char*>(input_layout.c_str());
  at::Tensor dpse;
  if (format_pse.defined()) {
    dpse = OpPreparation::ApplyTensorWithoutFormat(format_pse);
  } else {
    dpse = at::empty({0}, query.options());
  }

  EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnMultiHeadAttentionScoreGrad, format_query, format_key, format_value, format_dy,
                               format_pse, format_drop_mask, format_padding_mask, dtype_atten_mask, format_softmax,
                               format_attention, scale_value, keep_prob, pre_tockens, next_tockens, head_num,
                               input_layout_ptr, dq, dk, dv, dpse);

  if (!format_pse.defined()) {
    at::Tensor dpse_required;
    dpse = dpse_required;
  }

  at::Tensor dq_scalared = at::mul(dq, at::Scalar(scale));
  return {dq_scalared,  dk,           dv,           at::Tensor(), at::Tensor(), dpse,         at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}

class NPUMultiHeadAttentionFunction : public torch::autograd::Function<NPUMultiHeadAttentionFunction> {
 public:
  static std::vector<at::Tensor> forward(
      AutogradContext *ctx, const at::Tensor &query, const at::Tensor &key,
      const at::Tensor &value, int64_t head_num, c10::string_view input_layout, const c10::optional<at::Tensor> &pse_opt,
      const c10::optional<at::Tensor> &padding_mask_opt, const c10::optional<at::Tensor> &atten_mask_opt,
      double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, bool gen_mask_parallel, bool sync)
      {
    const at::Tensor &pse = pse_opt.value_or(at::Tensor());
    const at::Tensor &padding_mask = padding_mask_opt.value_or(at::Tensor());
    const at::Tensor &atten_mask = atten_mask_opt.value_or(at::Tensor());

    TORCH_CHECK(query.dim() == 3, "The shapes of the input query should be 3-dimensional, but got ", query.dim(), "-dimensional",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key.dim() == 3, "The shapes of the input key should be 3-dimensional, but got ", key.dim(), "-dimensional",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(value.dim() == 3, "The shapes of the input value should be 3-dimensional, but got ", value.dim(), "-dimensional",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1, "The keep_prob value must be in range of [0, 1], but got ", keep_prob,
                OPS_ERROR(ErrCode::VALUE));
    std::string input_layout_str = std::string(input_layout);
    for (auto &c : input_layout_str) {
        c = toupper(c);
    }
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH",
        "The input_layout should be BSH/SBH(case-insensitive), but got ", input_layout,
        OPS_ERROR(ErrCode::TYPE));

    int64_t B = 0;
    int64_t S0 = 0; // S for query
    int64_t S1 = 0; // S for key & value
    int64_t H = 0;
    if (input_layout_str == "BSH") {
        B = query.size(0);
        S0 = query.size(1);
        S1 = key.size(1);
        H = query.size(2);
    } else if (input_layout_str == "SBH") {
        B = query.size(1);
        S0 = query.size(0);
        S1 = key.size(0);
        H = query.size(2);
    }

    double scale_value = scale;

    at::Tensor format_query = format_trans(query);
    at::Tensor attention_score = OpPreparation::ApplyTensorWithoutFormat(format_query);
    at::Tensor format_key = format_trans(key);
    at::Tensor format_value = format_trans(value);

    at::Tensor format_pse = format_trans(pse);
    at::Tensor format_padding_mask = format_trans(padding_mask);
    at::Tensor format_atten_mask = format_trans(atten_mask);
    at::Tensor dtype_atten_mask =
      (format_atten_mask.defined() && format_atten_mask.scalar_type() != query.scalar_type()) ?
      custom_ops::npu_dtype_cast(format_atten_mask, query.scalar_type()) : format_atten_mask;

    int64_t seed;
    int64_t offset;
    int64_t numels;
    at::Tensor format_drop_mask = dropout_gen_mask(format_query, keep_prob, head_num, input_layout_str,
        gen_mask_parallel, sync, seed, offset, numels);

    at::Tensor softmax_out;
    // [B, N, S0, S1]
    softmax_out = OpPreparation::ApplyTensorWithoutFormat(format_query, {B, head_num, S0, S1});

    char* input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFlashAttentionScore, format_query, format_key, format_value,
        format_pse, format_drop_mask, format_padding_mask, dtype_atten_mask,
        scale, keep_prob, pre_tockens, next_tockens, head_num, input_layout_ptr,
        softmax_out, attention_score);

    if (!sync) {
        c10_npu::NPUEvent npu_event;
        npu_event.record(c10_npu::getCurrentNPUStream());
        npu_event.block(c10_npu::getCurrentSecondaryStream());
    }

    at::AutoNonVariableTypeMode g;

    ctx->save_for_backward({query, format_key, format_value, softmax_out,
                            format_pse, format_padding_mask, format_atten_mask, attention_score});

    ctx->saved_data["scale"] = scale;
    ctx->saved_data["keep_prob"] = keep_prob;
    ctx->saved_data["pre_tockens"] = pre_tockens;
    ctx->saved_data["next_tockens"] = next_tockens;
    ctx->saved_data["head_num"] = head_num;
    ctx->saved_data["input_layout"] = input_layout_str;
    ctx->saved_data["gen_mask_parallel"] = gen_mask_parallel;
    ctx->saved_data["sync"] = sync;
    ctx->saved_data["seed"] = seed;
    ctx->saved_data["offset"] = offset;
    ctx->saved_data["numels"] = numels;

    return {attention_score};
  }

    static std::vector<at::Tensor> backward(AutogradContext* ctx, std::vector<at::Tensor> grad_outputs) {
    auto scale = ctx->saved_data["scale"].toDouble();
    auto keep_prob = ctx->saved_data["keep_prob"].toDouble();
    auto pre_tockens = ctx->saved_data["pre_tockens"].toInt();
    auto next_tockens = ctx->saved_data["next_tockens"].toInt();
    auto head_num = ctx->saved_data["head_num"].toInt();
    auto input_layout = ctx->saved_data["input_layout"].toStringRef();
    auto gen_mask_parallel = ctx->saved_data["gen_mask_parallel"].toBool();
    auto sync = ctx->saved_data["sync"].toBool();
    auto seed = ctx->saved_data["seed"].toInt();
    auto offset = ctx->saved_data["offset"].toInt();
    auto numels = ctx->saved_data["numels"].toInt();
    auto saved = ctx->get_saved_variables();

    auto query = saved[0];
    auto key = saved[1];
    auto value = saved[2];
    auto softmax_out = saved[3];
    auto pse = saved[4];
    auto padding_mask = saved[5];
    auto atten_mask = saved[6];
    auto attention_score = saved[7];

    int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
    length += LENGTH_STRIDE;
    at::Tensor drop_mask;
    if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
      drop_mask = dropout_gen_mask_dispatch(query, at::Scalar(keep_prob), at::Scalar(seed), offset, numels,
                                            gen_mask_parallel, sync);
    } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
      drop_mask = at::zeros(at::IntArrayRef{length}, query.options().dtype(at::kByte));
    }

    auto results = npu_multi_head_attention_score_backward(
        query, key, value, grad_outputs[0], head_num, input_layout, pse, drop_mask, padding_mask, atten_mask,
        softmax_out, attention_score, scale, keep_prob, pre_tockens, next_tockens);

    if (!sync) {
      c10_npu::NPUEvent npu_event;
      npu_event.record(c10_npu::getCurrentNPUStream());
      npu_event.block(c10_npu::getCurrentSecondaryStream());
    }
    return results;
  }
};

std::vector<at::Tensor> NPUNativeFunctions::npu_multi_head_attention_grad(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& dy, int64_t head_num,
    c10::string_view input_layout, const c10::optional<at::Tensor>& pse, const c10::optional<at::Tensor>& padding_mask,
    const c10::optional<at::Tensor>& atten_mask, const c10::optional<at::Tensor>& softmax_in,
    const c10::optional<at::Tensor>& attention_in, double scale_value, double keep_prob, int64_t pre_tockens,
    int64_t next_tockens, bool gen_mask_parallel, bool sync) {
  TORCH_CHECK(query.dim() == DIM_SUPPORT, "The shapes of the input query should be ", DIM_SUPPORT,
              "-dimensional, but got ", query.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(key.dim() == DIM_SUPPORT, "The shapes of the input key should be ", DIM_SUPPORT, "-dimensional, but got ",
              key.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(value.dim() == DIM_SUPPORT, "The shapes of the input value should be ", DIM_SUPPORT,
              "-dimensional, but got ", value.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(dy.dim() == DIM_SUPPORT, "The shapes of the input dy should be ", DIM_SUPPORT, "-dimensional, but got ",
              dy.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1, "The keep_prob value must be in range of [0, 1], but got ", keep_prob,
              OPS_ERROR(ErrCode::VALUE));
  std::string input_layout_str = std::string(input_layout);
  for (auto& c : input_layout_str) {
    c = toupper(c);
    }
  TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH",
              "The input_layout should be BSH/SBH(case-insensitive), but got ", input_layout, OPS_ERROR(ErrCode::TYPE));
  int64_t seed;
  int64_t offset;
  int64_t numels;
  at::Tensor drop_mask =
      dropout_gen_mask(query, keep_prob, head_num, input_layout_str, gen_mask_parallel, sync, seed, offset, numels);

  auto result = npu_multi_head_attention_score_backward(query, key, value, dy, head_num, input_layout_str, pse,
                                                        drop_mask, padding_mask, atten_mask, softmax_in, attention_in,
                                                        scale_value, keep_prob, pre_tockens, next_tockens);

  if (!sync) {
    c10_npu::NPUEvent npu_event;
    npu_event.record(c10_npu::getCurrentNPUStream());
    npu_event.block(c10_npu::getCurrentSecondaryStream());
  }
  return result;
}

std::vector<at::Tensor> NPUNativeFunctions::npu_multi_head_attention_score(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, int64_t head_num, c10::string_view input_layout,
    const c10::optional<at::Tensor> &pse, const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, bool gen_mask_parallel, bool sync)
{
    return NPUMultiHeadAttentionFunction::apply(query, key, value, head_num, input_layout, pse, padding_mask,
        atten_mask, scale, keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync);
}
}  // namespace native
}  // namespace at_npu
