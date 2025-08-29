#include "ScheduleContext.h"
#include <sstream>
#include <limits>
#include <vector>
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace torch_npu {
namespace afd {
namespace {
constexpr uint32_t kSuccess = 0;
constexpr uint32_t kFailure = 1;

constexpr int32_t kRunFlagStop = 0;
constexpr int32_t kRunFlagRunning = 1;

constexpr int32_t kScheduleModeFfn = 0;
constexpr int32_t kScheduleModeAttention = 1;
constexpr uint64_t kBufAlignSize = 512;

inline uint64_t AlignUp(uint64_t num, uint64_t align)
{
    return ((num + align - 1) / align) * align;
}

template<typename T>
class IntegerChecker {
public:
    template<typename T1>
    static bool Compat(const T1 v)
    {
        static_assert(((sizeof(T) <= sizeof(uint64_t)) && (sizeof(T1) <= sizeof(uint64_t))),
                      "IntegerChecker can only check integers less than 64 bits");
        if (v >= static_cast<T1>(0)) {
            return static_cast<uint64_t>(v) <= static_cast<uint64_t>(std::numeric_limits<T>::max());
        }
        return static_cast<int64_t>(v) >= static_cast<int64_t>(std::numeric_limits<T>::min());
    }
};

template<typename TLhs, typename TRhs, typename TRet>
bool MulOverflow(TLhs lhs, TRhs rhs, TRet &ret)
{
#if __GNUC__ >= 5
    return __builtin_mul_overflow(lhs, rhs, &ret);
#else
    if ((!IntegerChecker<TRet>::Compat(lhs)) || (!IntegerChecker<TRet>::Compat(rhs))) {
        return true;
    }
    if ((lhs == 0) || (rhs == 0)) {
        ret = 0;
        return false;
    }
    TRet reminder = std::numeric_limits<TRet>::max() / static_cast<TRet>(rhs);
    const TRet lhs_ret_type = static_cast<TRet>(lhs);
    if (lhs_ret_type < 0) {
        if (reminder > 0) {
            reminder *= static_cast<TRet>(-1);
        }
        if (lhs_ret_type < reminder) {
            return true;
        }
    } else {
        if (reminder < 0) {
            reminder *= static_cast<TRet>(-1);
        }
        if (lhs_ret_type > reminder) {
            return true;
        }
    }
    ret = static_cast<TRet>(lhs) * static_cast<TRet>(rhs);
    return false;
#endif
}

template<typename TLhs, typename TRhs, typename TRet>
bool AddOverflow(TLhs lhs, TRhs rhs, TRet &ret)
{
#if __GNUC__ >= 5
    return __builtin_add_overflow(lhs, rhs, &ret);
#else
    if ((!IntegerChecker<TRet>::Compat(lhs)) || (!IntegerChecker<TRet>::Compat(rhs))) {
        return true;
    }
    if (rhs >= 0) {
        if (static_cast<TRet>(lhs) > std::numeric_limits<TRet>::max() - static_cast<TRet>(rhs)) {
            return true;
        }
    } else {
        if (static_cast<TRet>(lhs) < std::numeric_limits<TRet>::min() - static_cast<TRet>(rhs)) {
            return true;
        }
    }
    ret = static_cast<TRet>(lhs) + static_cast<TRet>(rhs);
    return false;
#endif
}
}  // namespace
ScheduleContextHolder::ScheduleContextHolder(int32_t schedule_mode, uint32_t session_num, uint32_t micro_batch_num,
                                             uint32_t micro_batch_size, uint32_t selected_expert_num,
                                             uint32_t expert_num, uint32_t attn_to_ffn_token_size,
                                             uint32_t ffn_to_attn_token_size, uint64_t ffn_window,
                                             uint64_t ffn_window_size, uint64_t attention_window,
                                             uint64_t attention_window_size)
{
    context_.common.schedule_mode = schedule_mode;
    context_.common.session_num = session_num;
    context_.common.micro_batch_num = micro_batch_num;
    context_.common.micro_batch_size = micro_batch_size;
    context_.common.selected_expert_num = selected_expert_num;
    context_.common.expert_num = expert_num;
    context_.common.attn_to_ffn_token_size = attn_to_ffn_token_size;
    context_.common.ffn_to_attn_token_size = ffn_to_attn_token_size;
    ffn_window_ = ffn_window;
    ffn_window_size_ = ffn_window_size;
    attention_window_ = attention_window;
    attention_window_size_ = attention_window_size;
}

uint64_t ScheduleContextHolder::CalcFfnTokenInfoSize() const
{
    uint64_t token_info_size = sizeof(int32_t) * static_cast<uint64_t>(context_.common.selected_expert_num);
    if (MulOverflow(token_info_size, static_cast<uint64_t>(context_.common.micro_batch_size), token_info_size)) {
        ASCEND_LOGE("check mul with micro_batch_size over flow failed.");
        return 0UL;
    }
    uint64_t flag_and_layer_id_size = sizeof(int32_t) * 2;
    if (AddOverflow(token_info_size, flag_and_layer_id_size, token_info_size)) {
        ASCEND_LOGE("check add flag and layer id over flow failed.");
        return 0UL;
    }

    if (MulOverflow(token_info_size, static_cast<uint64_t>(context_.common.micro_batch_num), token_info_size)) {
        ASCEND_LOGE("check mul with micro_batch_num over flow failed.");
        return 0UL;
    }
    if (MulOverflow(token_info_size, static_cast<uint64_t>(context_.common.session_num), token_info_size)) {
        ASCEND_LOGE("check mul with session_num over flow failed.");
        return 0UL;
    }
    return token_info_size;
}

uint32_t ScheduleContextHolder::InitFfnTokenInfoBuf() const
{
    std::unique_ptr<uint8_t[]> tmp_buf(new(std::nothrow) uint8_t[context_.ffn.token_info_buf_size]);
    if (tmp_buf == nullptr) {
        ASCEND_LOGE("alloc token info host tmp buf failed, buf_size=%lu", context_.ffn.token_info_buf_size);
        return kFailure;
    }
    auto tmp_buf_int = reinterpret_cast<int32_t *>(tmp_buf.get());
    for (uint32_t session_id = 0; session_id < context_.common.session_num; ++session_id) {
        for (uint32_t micro_batch_id = 0; micro_batch_id < context_.common.micro_batch_num; ++micro_batch_id) {
            // flag
            *tmp_buf_int++ = 0;
            // layer_id
            *tmp_buf_int++ = 0;
            for (uint32_t idx = 0;
                 idx < context_.common.micro_batch_size * context_.common.selected_expert_num; ++idx) {
                // expert_id
                *tmp_buf_int++ = INT32_MAX;
            }
        }
    }
    auto token_info_buf = reinterpret_cast<void *>(static_cast<uintptr_t>(context_.ffn.token_info_buf));
    auto ret = aclrtMemcpy(token_info_buf, context_.ffn.token_info_buf_size, tmp_buf.get(),
                           context_.ffn.token_info_buf_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        ASCEND_LOGE("ACL memory copy token info buf failed, size_=%lu, token_info_buf ptr=%lu.",
                    context_.ffn.token_info_buf_size, token_info_buf);
        return kFailure;
    }
    return kSuccess;
}

uint32_t ScheduleContextHolder::InitFfn()
{
    uint64_t token_info_size = CalcFfnTokenInfoSize();
    if (token_info_size == 0U) {
        return kFailure;
    }

    uint64_t token_info_aligned_size = AlignUp(token_info_size, kBufAlignSize);
    if (token_info_aligned_size < token_info_size) {
        ASCEND_LOGE("token_info_size=" << token_info_size << " overflow after align with " << kBufAlignSize << ".");
        return kFailure;
    }
    if (ffn_window_size_ <= token_info_aligned_size) {
        ASCEND_LOGE("ffn_window_size=%lu must be > token_info_aligned_size=%lu.",
                    ffn_window_size_, token_info_aligned_size);
        return kFailure;
    }

    context_.ffn.token_info_buf = ffn_window_;
    context_.ffn.token_info_buf_size = token_info_size;
    auto ret = InitFfnTokenInfoBuf();
    if (ret != kSuccess) {
        return ret;
    }

    if (AddOverflow(ffn_window_, token_info_aligned_size, context_.ffn.token_data_buf)) {
        ASCEND_LOGE("check ffn_window add token_info_size over flow failed.");
        return kFailure;
    }

    // can't calc token_data_buf_size as the data type is unknown.
    context_.ffn.token_data_buf_size = ffn_window_size_ - token_info_aligned_size;

    // calc output size
    context_.ffn.layer_ids_buf_size = sizeof(int32_t) * context_.common.session_num;
    context_.ffn.session_ids_buf_size = sizeof(int32_t) * context_.common.session_num;
    context_.ffn.micro_batch_ids_buf_size = sizeof(int32_t) * context_.common.session_num;
    context_.ffn.expert_ids_buf_size = sizeof(int32_t) * context_.common.session_num *
                                       context_.common.micro_batch_size * context_.common.selected_expert_num;

    ASCEND_LOGI("Init ffn success, token_info_buf=%lu, token_info_buf_size=%lu, "
                "token_data_buf=%lu, token_data_buf_size=%lu.",
                context_.ffn.token_info_buf, context_.ffn.token_info_buf_size,
                context_.ffn.token_data_buf, context_.ffn.token_data_buf_size);
    return kSuccess;
}

uint64_t ScheduleContextHolder::CalcAttentionTokenInfoSize() const
{
    uint64_t token_info_size = sizeof(int32_t) * static_cast<uint64_t>(context_.common.selected_expert_num);
    if (MulOverflow(token_info_size, static_cast<uint64_t>(context_.common.micro_batch_size), token_info_size)) {
        ASCEND_LOGE("check mul with micro_batch_size over flow failed.");
        return 0UL;
    }

    if (MulOverflow(token_info_size, static_cast<uint64_t>(context_.common.micro_batch_num), token_info_size)) {
        ASCEND_LOGE("check mul with micro_batch_num over flow failed.");
        return 0UL;
    }
    return token_info_size;
}

uint32_t ScheduleContextHolder::InitAttention()
{
    uint64_t token_info_size = CalcAttentionTokenInfoSize();
    if (token_info_size == 0U) {
        return kFailure;
    }

    uint64_t token_info_aligned_size = AlignUp(token_info_size, kBufAlignSize);
    if (token_info_aligned_size < token_info_size) {
        ASCEND_LOGE("token_info_size=%lu overflow after align with %lu.", token_info_size, kBufAlignSize);
        return kFailure;
    }
    if (attention_window_size_ <= token_info_aligned_size) {
        ASCEND_LOGE("attention_window_size=%lu must be > token_info_aligned_size= %lu.",
                    attention_window_size_, token_info_aligned_size);
        return kFailure;
    }

    context_.attention.token_info_buf = attention_window_;
    context_.attention.token_info_buf_size = token_info_size;
    auto ret = aclrtMemset(reinterpret_cast<void *>(static_cast<uintptr_t>(context_.attention.token_info_buf)),
                           token_info_size, '\0', token_info_size);
    if (ret != ACL_ERROR_NONE) {
        ASCEND_LOGE("ACL memset attention context to 0 failed, addr=%lu, size=%zu.",
                    context_.attention.token_info_buf, token_info_size);
        return kFailure;
    }
    if (AddOverflow(attention_window_, token_info_aligned_size, context_.attention.token_data_buf)) {
        ASCEND_LOGE("check attention_window add token_info_size over flow failed.");
        return kFailure;
    }
    // can't calc token_data_buf_size as the data type is unknown.
    context_.attention.token_data_buf_size = attention_window_size_ - token_info_aligned_size;
    // init to micro_batch_num - 1, scheduler will scan from (micro_batch_id + 1) % micro_batch_num.
    context_.attention.micro_batch_id = context_.common.micro_batch_num - 1U;
    ASCEND_LOGI("Init attention success, token_info_buf=%lu, token_info_buf_size=%lu, token_data_buf=%lu, "
                "token_data_buf_size=%lu, micro_batch_id=%u.",
                context_.attention.token_info_buf, context_.attention.token_info_buf_size,
                context_.attention.token_data_buf, context_.attention.token_data_buf_size,
                context_.attention.micro_batch_id);
    return kSuccess;
}

uint32_t ScheduleContextHolder::Init()
{
    if (init_flag_) {
        ASCEND_LOGI("Already been initialized, does not need to be initialized again.");
        return kSuccess;
    }
    ASCEND_LOGI("Init begin, schedule_mode=%d, session_num=%u, micro_batch_num=%u, micro_batch_size=%u, "
                "selected_expert_num=%u, ffn_window=%lu, ffn_window_size=%lu, "
                "attention_window=%lu, attention_window_size_=%lu.",
                context_.common.schedule_mode, context_.common.session_num, context_.common.micro_batch_num,
                context_.common.micro_batch_size, context_.common.selected_expert_num,
                ffn_window_, ffn_window_size_, attention_window_, attention_window_size_);
    if (!CheckParams()) {
        return kFailure;
    }
    uint32_t ret = kSuccess;
    if (context_.common.schedule_mode == kScheduleModeFfn) {
        ret = InitFfn();
    } else if (context_.common.schedule_mode == kScheduleModeAttention) {
        ret = InitAttention();
    }
    if (ret != kSuccess) {
        return ret;
    }
    context_.control.run_flag = kRunFlagRunning;
    ret = AllocAndAssignDevMem();
    if (ret != kSuccess) {
        return ret;
    }
    init_flag_ = true;
    ASCEND_LOGI("init success.");
    return kSuccess;
}

uint32_t ScheduleContextHolder::AllocAndAssignDevMem()
{
    auto dev_tensor_options = at::TensorOptions(c10::DeviceType::PrivateUse1).dtype(torch::kInt8);
    if (context_.common.schedule_mode == kScheduleModeFfn) {
        uint64_t layer_id_buf_size_align_up = AlignUp(context_.ffn.layer_ids_buf_size, kBufAlignSize);
        uint64_t expert_buf_size_align_up = AlignUp(context_.ffn.expert_ids_buf_size, kBufAlignSize);
        // session_ids_buf and micro_batch_ids_buf size is same as layer_id_buf, so multiply 3.
        workspace_size_ = layer_id_buf_size_align_up * 3UL + expert_buf_size_align_up;
        workspace_tensor_ = at::empty(std::vector<int64_t>({workspace_size_}), dev_tensor_options);
        auto workspace_addr = reinterpret_cast<uintptr_t>(workspace_tensor_.data_ptr());
        if (workspace_addr == 0UL) {
            ASCEND_LOGE("alloc workspace failed, workspace_size_=%lu.", workspace_size_);
            return kFailure;
        }
        context_.ffn.layer_ids_buf = workspace_addr;
        context_.ffn.session_ids_buf = context_.ffn.layer_ids_buf + layer_id_buf_size_align_up;
        context_.ffn.micro_batch_ids_buf = context_.ffn.session_ids_buf + layer_id_buf_size_align_up;
        context_.ffn.expert_ids_buf = context_.ffn.micro_batch_ids_buf + layer_id_buf_size_align_up;

        ASCEND_LOGI("alloc and assign ffn dev mem success, layer_ids_buf=%lu, layer_ids_buf_size=%lu, "
                    "session_ids_buf=%lu, session_ids_buf_size=%lu, "
                    "micro_batch_ids_buf=%lu, micro_batch_ids_buf_size=%lu, "
                    "expert_ids_buf=%lu, expert_ids_buf_size=%lu.",
                    context_.ffn.layer_ids_buf, context_.ffn.layer_ids_buf_size,
                    context_.ffn.session_ids_buf, context_.ffn.session_ids_buf_size,
                    context_.ffn.micro_batch_ids_buf, context_.ffn.micro_batch_ids_buf_size,
                    context_.ffn.expert_ids_buf, context_.ffn.expert_ids_buf_size);
    }
    std::vector<int64_t> context_shape = {sizeof(ScheduleContext)};
    // 将 ScheduleContext 的内存包装成 Tensor（零拷贝）
    at::Tensor host_tensor = at::from_blob(&context_,
                                           context_shape,
                                           at::TensorOptions().dtype(torch::kInt8));
    context_tensor_ = at::empty(context_shape, dev_tensor_options);
    context_tensor_.copy_(host_tensor);
    return kSuccess;
}

bool ScheduleContextHolder::CheckFfnParams() const
{
    if (ffn_window_ == 0UL) {
        ASCEND_LOGE("check ffn param failed, ffn_window can't be 0.");
        return false;
    }
    if (ffn_window_size_ == 0UL) {
        ASCEND_LOGE("check ffn param failed, ffn_window_size can't be 0.");
        return false;
    }
    return true;
}

bool ScheduleContextHolder::CheckAttentionParams() const
{
    if (attention_window_ == 0UL) {
        ASCEND_LOGE("check attention param failed, ffn_window can't be 0.");
        return false;
    }
    if (attention_window_size_ == 0UL) {
        ASCEND_LOGE("check attention param failed, ffn_window_size can't be 0.");
        return false;
    }
    return true;
}

bool ScheduleContextHolder::CheckParams() const
{
    if ((context_.common.session_num == 0U) || (context_.common.micro_batch_num == 0U) ||
        (context_.common.micro_batch_size == 0U) || (context_.common.selected_expert_num == 0U) ||
        (context_.common.expert_num == 0U)) {
        ASCEND_LOGE("session_num[%u], micro_batch_num[%u], micro_batch_size[%u], selected_expert_num[%u], "
                    "expert_num[%u] can't be 0.", context_.common.session_num, context_.common.micro_batch_num,
                    context_.common.micro_batch_size, context_.common.selected_expert_num, context_.common.expert_num);
        return false;
    }

    if ((context_.common.attn_to_ffn_token_size % kBufAlignSize) != 0U) {
        ASCEND_LOGE("attn_to_ffn_token_size[%lu] must be align with %lu.", context_.common.attn_to_ffn_token_size,
                    kBufAlignSize);
        return false;
    }
    if ((context_.common.ffn_to_attn_token_size % kBufAlignSize) != 0U) {
        ASCEND_LOGE("ffn_to_attn_token_size[%lu] must be align with %lu.", context_.common.ffn_to_attn_token_size,
                    kBufAlignSize);
        return false;
    }

    if (context_.common.schedule_mode == kScheduleModeFfn) {
        return CheckFfnParams();
    } else if (context_.common.schedule_mode == kScheduleModeAttention) {
        return CheckAttentionParams();
    } else {
        ASCEND_LOGE("check schedule_mode=%d failed, only support [%d, %d] now.", context_.common.schedule_mode,
                    kScheduleModeFfn, kScheduleModeAttention);
        return false;
    }
}

uint32_t ScheduleContextHolder::StopSchedule()
{
    context_.control.run_flag = kRunFlagStop;
    at::Tensor run_flag_host_tensor = at::from_blob(
        &context_.control.run_flag,
        {static_cast<int64_t>(sizeof(context_.control.run_flag))},
        at::TensorOptions().dtype(torch::kInt8)
    );
    size_t offset = offsetof(ScheduleContext::ControlArea, run_flag) + offsetof(ScheduleContext, control);
    auto run_flag_view = context_tensor_.slice(0, offset, offset + sizeof(context_.control.run_flag));
    run_flag_view.copy_(run_flag_host_tensor);
    return kSuccess;
}

std::pair<uint32_t, at::Tensor> ScheduleContextHolder::GetContextTensor() const
{
    if (!init_flag_) {
        return std::make_pair(kFailure, at::Tensor());
    }

    return std::make_pair(kSuccess, context_tensor_);
}

uint32_t ScheduleContextHolder::GetScheduleContextFromDev(ScheduleContext &context) const
{
    // 将 ScheduleContext 的内存包装成 Tensor（零拷贝）
    at::Tensor host_tensor = at::from_blob(&context,
                                           {static_cast<int64_t>(sizeof(context))},
                                           at::TensorOptions().dtype(torch::kInt8)
    );
    host_tensor.copy_(context_tensor_);
    return kSuccess;
}

std::string ScheduleContextHolder::GetScheduleContextInfo() const
{
    if (!init_flag_) {
        return "Error: schedule context is not inited!";
    }
    ScheduleContext tmp_context{};
    auto ret = GetScheduleContextFromDev(tmp_context);
    if (ret != kSuccess) {
        ASCEND_LOGE("Get schedule context from device failed.");
        return "Error: copy schedule context to host failed, error=" + std::to_string(ret);
    }
    return ToString(tmp_context);
}

std::string ScheduleContextHolder::ToString(const ScheduleContext &context)
{
    std::stringstream ss;
    ss << "schedule context: schedule_mode=" << context.common.schedule_mode
       << ", session_num=" << context.common.session_num << ", micro_batch_num=" << context.common.micro_batch_num
       << ", micro_batch_size=" << context.common.micro_batch_size
       << ", selected_expert_num=" << context.common.selected_expert_num << ", expert_num=" << context.common.expert_num
       << ", attn_to_ffn_token_size=" << context.common.attn_to_ffn_token_size
       << ", ffn_to_attn_token_size=" << context.common.ffn_to_attn_token_size
       << ", run_flag=" << context.control.run_flag;
    if (context.common.schedule_mode == kScheduleModeFfn) {
        ss << ", ffn info: token_info_buf=" << context.ffn.token_info_buf
           << ", token_info_buf_size=" << context.ffn.token_info_buf_size
           << ", token_data_buf=" << context.ffn.token_data_buf
           << ", token_data_buf_size=" << context.ffn.token_data_buf_size << ", polling_index="
           << context.ffn.polling_index
           << ", layer_ids_buf=" << context.ffn.layer_ids_buf << ", layer_ids_buf_size="
           << context.ffn.layer_ids_buf_size
           << ", session_ids_buf=" << context.ffn.session_ids_buf
           << ", session_ids_buf_size=" << context.ffn.session_ids_buf_size
           << ", micro_batch_ids_buf=" << context.ffn.micro_batch_ids_buf
           << ", micro_batch_ids_buf_size=" << context.ffn.micro_batch_ids_buf_size
           << ", expert_ids_buf=" << context.ffn.expert_ids_buf
           << ", expert_ids_buf_size=" << context.ffn.expert_ids_buf_size << ", out_num=" << context.ffn.out_num << ";";
    } else if (context.common.schedule_mode == kScheduleModeAttention) {
        ss << ", attention info: token_info_buf=" << context.attention.token_info_buf
           << ", token_info_buf_size=" << context.attention.token_info_buf_size
           << ", token_data_buf=" << context.attention.token_data_buf
           << ", token_data_buf_size=" << context.attention.token_data_buf_size
           << ", micro_batch_id=" << context.attention.micro_batch_id << ";";
    }
    return ss.str();
}
}  // namespace afd
}  // namespace torch_npu